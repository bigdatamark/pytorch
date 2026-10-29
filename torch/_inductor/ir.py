# [MODULE DOC] Core TorchInductor intermediate representation (IR). Defines the IR node hierarchy produced by lowerings:
# TensorBox / StorageBox / View wrappers, Buffer and Operation nodes (Pointwise, Reduction, Scatter, ExternKernel
# and its subclasses, templates, fallback and collective kernels, conditionals, loops), Layout classes (FixedLayout,
# FlexibleLayout, NonOwningLayout, MutationLayout), index/stride/reindexing helpers and code generation hooks
# that the scheduler and wrapper codegen consume. See the "Note: Inductor IR" text below for the box/buffer model.
# Import-time side effects: detects whether triton is importable (sets triton_version and has_triton), creates the
# module logger and defines type aliases and stride-order constants.
from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import itertools
import logging
import operator
import textwrap
import traceback
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from enum import Enum
from functools import partial
from numbers import Number
from typing import (
    Any,
    cast,
    ClassVar,
    Literal,
    overload,
    Protocol,
    SupportsFloat,
    SupportsInt,
    TYPE_CHECKING,
    TypeAlias,
    TypeVar,
    Union,
)
from typing_extensions import (
    assert_never,
    Never,
    override,
    ParamSpec,
    Self,
    TypedDict,
    TypeIs,
)
from unittest.mock import patch

import sympy
from sympy import Expr, Integer, Symbol

import torch._export.serde.schema as export_schema
import torch._library.custom_ops as custom_ops
import torch._library.utils as library_utils
import torch._logging
import torch.fx
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import identity
from torch._export.serde.serialize import GraphModuleSerializer
from torch._higher_order_ops.auto_functionalize import can_auto_functionalize
from torch._inductor import metrics
from torch._inductor.utils import get_free_symbols
from torch._library.fake_class_registry import FakeScriptObject, maybe_to_fake_obj
from torch._library.opaque_object import get_opaque_obj_repr, is_custom_class_obj
from torch._native.ops.sum import inner_tree_plan
from torch._prims_common import (
    compute_required_storage_length,
    is_boolean_dtype,
    is_contiguous_for_memory_format_or_false,
    is_float_dtype,
    make_channels_last_strides_for,
    StrideType,
)
from torch.fx.experimental.symbolic_shapes import (
    _remove_effect_token_unbacked_bindings,
    compute_unbacked_bindings,
    free_symbols,
    free_unbacked_symbols,
    GuardOnDataDependentSymNode,
    has_free_unbacked_symbols,
    IterateExprs,
    rebind_unbacked,
    resolve_unbacked_bindings,
    ShapeEnv,
    SymTypes,
)
from torch.fx.node import Node
from torch.utils._ordered_set import OrderedSet
from torch.utils._python_dispatch import _disable_current_modes
from torch.utils._sympy.functions import (
    CleanDiv,
    FloorDiv,
    Max,
    Min,
    Mod,
    ModularIndexing,
)
from torch.utils._sympy.symbol import SymT

from . import config, dependencies
from .codegen.common import (
    BackendFeature,
    CodegenSymbol,
    get_scheduling_for_device,
    index_prevent_reordering,
    Kernel,
)
from .dependencies import (
    Dep,
    extract_free_symbols,
    extract_input_node_reduction_ranges,
    extract_read_writes,
    SymbolUsageCollectorOpsHandler,
    var_builder,
)
from .loop_body import LoopBody
from .ops_handler import OpCounterCSE, OpCountResult, ReductionType, StoreMode
from .runtime.benchmarking import benchmarker
from .runtime.hints import DeviceProperties, ReductionHint
from .utils import (
    argsort,
    argsort_sym,
    cache_on_self,
    cache_on_self_and_args,
    ceildiv,
    convert_shape_to_inductor,
    convert_shape_to_symint,
    developer_warning,
    do_bench_using_profiling,
    dtype_from_size,
    get_dtype_size,
    get_kernel_metadata,
    GPU_ALIGN_BYTES,
    ir_dataclass,
    is_dynamic,
    is_gpu,
    sympy_dot,
    sympy_index_symbol,
    sympy_index_symbol_with_prefix,
    sympy_product,
    sympy_subs,
    tensor_is_aligned,
    VarRanges,
)
from .virtualized import ops, OpsValue, V


if TYPE_CHECKING:
    from torch.fx.experimental.symbolic_shapes import SympyBoolean
    from torch.fx.node import Argument
    from torch.types import IntLikeType

    from .codegen.cutlass.template import CUTLASSTemplate
    from .codegen.wrapper import PythonWrapperCodegen
    from .graph import GraphLowering
    from .kernel.gemm_epilogue import GemmReductionPlan
    from .utils import IndentedBuffer

else:
    CUTLASSTemplate: TypeAlias = object


try:
    import triton

    triton_version = triton.__version__
    has_triton = True
except ImportError:
    triton_version = None
    has_triton = False


_P = ParamSpec("_P")
_T = TypeVar("_T")
_U = TypeVar("_U")
_V = TypeVar("_V")

_IntLike: TypeAlias = int | Expr
_NumLike: TypeAlias = int | float | Expr

_OpOverloads: TypeAlias = torch._ops.OpOverload | torch._ops.HigherOrderOperator


class ArgProperty(TypedDict, total=False):
    # [CLASS DOC] HELPER: Typed mapping (all keys optional) describing one kernel argument from a schema: name, JIT type and default value; used by extern kernels for cpp wrapper codegen.
    name: str
    type: torch.JitType
    default_value: object


log = logging.getLogger(__name__)
indent = functools.partial(textwrap.indent, prefix="  ")
aten = torch.ops.aten

""" [Note: Inductor IR]

Inductor's IR is produced by executing 'lowering' code (see lowering.py).  Each
lowering is registered to a particular aten operator, and expects inputs that
correspond to the aten schema.  However, in place of torch Tensor inputs, lowerings
expect Inductor TensorBox inputs.

TensorBox IR represents torch tensors.  Tensors are sometimes single objects owning
storage, and sometimes views of another Tensor's storage.  Mutating tensor operations
(such as add_()) affect the underlying storage and any associated views.  Other operations
(such as .t_()) update metadata about the current view but don't modify the underlying storage.

To model this in Inductor, the IR distinguishes between TensorBox, View, StorageBox and Buffer.

TensorBox is the top level IR construct that any lowering should produce and maps to a torch.Tensor
output from an operation.  But just as torch.Tensors take different forms, TensorBox IR can
reference View IR or directly reference StorageBox IRs.

Some Inductor lowerings produce new sets of 'Box'es, while others (such as .t() or other view ops)
may take an existing TensorBox and point it to a new underlying View IR.

Tensors that directly own storage are represented as a chain of:
TensorBox -> StorageBox -> Buffer
where Buffer is a simple (1D) allocation, and StorageBox introduces the concept of a Layout.

If you mutate the data of such a tensor, we swing the StorageBox pointer to point to a new buffer
(leaving the old buffer unmodified and functionalizing the operation).

Tensors backed by views add one more indirection to the IR.
TensorBox -> View -> StorageBox -> Buffer
In these cases, the underlying StorageBox/Buffer will be shared with the pre-view TensorBox.

Computation is represented by Operation nodes, with each operation producing 1
or more output Buffers. In the case of mutations, these will be new Buffers that have the
mutated buffer listed in its get_mutation_names().

It is also possible to have an InputBuffer for which there is no corresponding Operation,
e.g. it may be a graph input or compile time constant.

"""


_NodeOrNodes: TypeAlias = Union[
    int,
    "TensorBox",
    dict[str, "TensorBox"],
    "Symbol",
    "IRNode",
    Sequence[
        Union[int, dict[str, "TensorBox"], "TensorBox", "Symbol", "IRNode"] | None
    ],
]


def _is_static(x: object) -> TypeIs[int | Integer]:
    # [FUNCTION DOC] HELPER: Return True if x is a Python int or sympy Integer (a statically known value).
    return isinstance(x, (int, Integer))


@dataclasses.dataclass(frozen=True)
class GraphPartitionSignature:
    # [CLASS DOC] Frozen record describing one graph partition's interface for codegen: the symbol inputs needed,
    # mapping of input names to IR nodes/expressions/torchbind objects, output nodes, per-input deallocation flags,
    # whether to skip cudagraph, and the names of constants read or written by the partition.
    # symbol inputs that are necessary for codegen
    symbol_inputs: OrderedSet[sympy.Symbol]

    # mapping from partition input name to IRNode or Expr. Need the name str since
    # we cannot get name from Expr.
    input_nodes: dict[str, IRNode | Expr | TorchBindObject]
    output_nodes: list[IRNode]

    # mapping from partition input name to a boolean for whether deallocating it
    # in the partition function
    input_deallocation: dict[str, bool]
    skip_cudagraph: bool

    # name of constants read/written by the graph partition
    constant_names: list[str]


def validate_ir(node_or_nodes: _NodeOrNodes | None) -> None:
    # [FUNCTION DOC] Check that the given node, or (nested) list/tuple/dict of nodes, contains only IR types allowed as
    # top-level lowering results (TensorBox, ExpandView, scalars, sympy exprs, effectful/opaque kernels, etc.). None is allowed.
    # Raises AssertionError naming the offending type and pointing at the IR note. No pytree is used on purpose.
    def _check_tensorbox(nodes: _NodeOrNodes | None) -> None:
        # [FUNCTION DOC] HELPER: Recursive worker for validate_ir; walks containers and raises AssertionError for unsupported leaf types.
        # Could expand this to check deeper properties
        # (e.g. TensorBox points to View or StorageBox)
        if nodes is None:
            pass
        elif isinstance(nodes, (list, tuple)):
            for node in nodes:
                _check_tensorbox(node)
        elif isinstance(nodes, dict):
            for node in nodes.values():
                _check_tensorbox(node)
        else:
            if not isinstance(
                nodes,
                (
                    ExpandView,
                    DynamicScalar,
                    AssertScalar,
                    TensorBox,
                    sympy.logic.boolalg.Boolean,
                    Expr,
                    int,
                    EffectfulKernel,
                    ShapeAsConstantBuffer,
                    OpaqueMultiOutput,
                ),
            ):
                raise AssertionError(
                    f"Found {type(nodes)}, which is not a supported top level IR node. See [Note: Inductor IR]"
                )

    # Be picky about the accepted data structure (don't use pytree here)
    _check_tensorbox(node_or_nodes)


def ops_wrapper(name: str) -> Callable[..., OpsValue]:
    # [FUNCTION DOC] Return a function that forwards its arguments to the ops handler method with the given name.
    # Raises AssertionError if name is not a string. The lookup on the virtualized ops object happens at call time.
    if not isinstance(name, str):
        raise AssertionError(type(name))

    def fn(*args: object, **kwargs: object) -> OpsValue:
        # [FUNCTION DOC] HELPER: Look up the named method on the current ops handler and call it with the given arguments.
        return getattr(ops, name)(*args, **kwargs)

    return fn


def inverse_reorder(order: Sequence[int]) -> Callable[[Sequence[_T]], Sequence[_T]]:
    # [FUNCTION DOC] Given a permutation order, return a reindexing function that applies the inverse permutation to an
    # index sequence (output position i takes the element at the position where order has value i).
    # Returned function raises AssertionError if the index length differs from the order length.
    inv_order = dict(zip(order, range(len(order))))

    def reindex(index: Sequence[_T]) -> Sequence[_T]:
        # [FUNCTION DOC] HELPER: Apply the inverse permutation to an index sequence and return the reordered list.
        if len(index) != len(inv_order):
            raise AssertionError("Expected len(index) == len(inv_order)")
        return [index[inv_order[i]] for i in range(len(index))]

    return reindex


def same_reorder(order: Sequence[int]) -> Callable[[Sequence[_T]], Sequence[_T]]:
    # [FUNCTION DOC] Return a reindexing function that permutes an index sequence by order (result position i takes the
    # element at position order[i]). Raises AssertionError on a length mismatch when called.
    def reindex(index: Sequence[_T]) -> Sequence[_T]:
        # [FUNCTION DOC] HELPER: Permute the index sequence according to order and return the new list.
        if len(index) != len(order):
            raise AssertionError("Expected len(index) == len(order)")
        return [index[order[i]] for i in range(len(index))]

    return reindex


def fuse_reindexing(
    reindex1: Callable[[Sequence[_U]], Sequence[_V]],
    reindex2: Callable[[Sequence[_T]], Sequence[_U]],
) -> Callable[[Sequence[_T]], Sequence[_V]]:
    # [FUNCTION DOC] Compose two reindexing functions so the second is applied to the index first and the first is applied to its result.
    def reindex(index: Sequence[_T]) -> Sequence[_V]:
        # [FUNCTION DOC] HELPER: Apply reindex2 then reindex1 to the index.
        return reindex1(reindex2(index))

    return reindex


NHWC_STRIDE_ORDER = [3, 0, 2, 1]
NHWDC_STRIDE_ORDER = [4, 0, 3, 2, 1]


def get_fill_order(
    seq: Sequence[int | torch.SymInt | Expr], shape_env: ShapeEnv | None = None
) -> Sequence[int]:
    # [FUNCTION DOC] Convert a sequence of strides into a fill order (argsort of the strides). Uses a plain argsort when
    # all strides are static ints or no shape_env is given; otherwise uses the symbolic argsort which can handle
    # unbacked symbols with the help of the shape_env. Returns a sequence of dimension indices.
    """
    Convert strides to fill order (argsort)
    """
    if shape_env is None or all(isinstance(s, (int, sympy.Integer)) for s in seq):
        sorted_idx: Sequence[int] = argsort(seq)
    else:
        # argsort_sym handles unbacked symints (with the help of the shape_env)
        sorted_idx = argsort_sym(shape_env, seq)
    return sorted_idx


def stride_order2fill_order(order: Sequence[int | Integer]) -> Sequence[int]:
    # [FUNCTION DOC] Convert a stride order into its inverse permutation, the fill order (e.g. for channels last the stride
    # order 3,0,2,1 maps to the fill order 1,3,2,0). Pure function.
    """
    Convert stride order to fill order
    For channel last format,

    stride order = [3, 0, 2, 1] and fill order = [1, 3, 2, 0]
    """
    lookup = {pos: idx for idx, pos in enumerate(order)}
    fill_order = [lookup[i] for i in range(len(order))]
    return fill_order


def get_stride_order(
    seq: Sequence[int | torch.SymInt | Expr], shape_env: ShapeEnv | None = None
) -> Sequence[int]:
    # [FUNCTION DOC] Convert strides to a stride order: for each dimension, its rank in the fill order of the strides.
    # Delegates to get_fill_order and returns a list of ints.
    """
    Convert strides to stride order
    """
    sorted_idx: Sequence[int] = get_fill_order(seq, shape_env)
    out = [0 for _ in range(len(seq))]
    for i, elem in enumerate(sorted_idx):
        out[elem] = i
    return out


@overload
def ir_node_to_tensor(x: None, replace_symbols_with_hints: bool = False) -> None: ...


@overload
def ir_node_to_tensor(
    x: IRNode, replace_symbols_with_hints: bool = False
) -> torch.Tensor: ...


def ir_node_to_tensor(
    x: IRNode | None, replace_symbols_with_hints: bool = False
) -> torch.Tensor | None:
    # [FUNCTION DOC] Materialize an IR node as a real (zero-filled) torch tensor with the node's size, stride, dtype and device,
    # for use as a meta/example input. None maps to None. Strides come from the layout when the node has storage and layout,
    # otherwise contiguous strides are used. If replace_symbols_with_hints is True, symbolic sizes/strides are replaced
    # by concrete optimization hints so no guards are installed; otherwise they stay symbolic. Allocation happens with
    # guards suppressed on the shape env. Returns the new tensor.
    # When replace_symbols_with_hints=False (default), sizes/strides remain as
    # symbolic expressions, so downstream operations on the resulting tensor (e.g.,
    # shape comparisons inside a kernel's meta function) may install guards. When
    # True, symbolic expressions are replaced with concrete integer hints via
    # size_hint, preventing any downstream guards.
    if x is None:
        return None

    shape_fn: Callable[[int | Expr], int | Expr]
    if replace_symbols_with_hints:
        shape_fn = V.graph.sizevars.optimization_hint
    else:
        shape_fn = identity
    size = [shape_fn(s) for s in x.get_size()]
    stride: StrideType
    if is_storage_and_layout(x):
        stride = [shape_fn(s) for s in x.get_layout().stride]
    else:
        stride = FlexibleLayout.contiguous_strides(size)
    dtype = x.get_dtype()
    device = x.get_device()
    size = convert_shape_to_symint(size)
    # pyrefly: ignore [bad-assignment]
    stride = convert_shape_to_symint(stride)
    with V.graph.sizevars.shape_env.suppress_guards():
        t = torch.empty_strided(
            size=size, stride=stride, dtype=dtype, device=device
        ).zero_()
    return t


def may_convert_to_optional(
    value: Sequence[_T] | None,
) -> Sequence[_T | None] | None:
    # [FUNCTION DOC] Map an empty list to a list holding a single None (so cpp wrapper codegen emits an optional
    # placeholder rather than empty braces); any other value is returned unchanged.
    if isinstance(value, list) and not value:
        # [None] makes sure the cpp wrapper codegen will generate something like
        # {std::nullopt} instead of {}
        return [None]
    return value


def get_device_type(
    x: IRNode | OutputSpec | torch.device | None | str,
) -> str | None:
    # [FUNCTION DOC] Return the device type string for a string, torch.device, IRNode or OutputSpec; None and str pass through.
    # Anything else triggers an assertion failure.
    if isinstance(x, str) or x is None:
        return x
    elif isinstance(x, torch.device):
        return x.type
    elif isinstance(x, (IRNode, OutputSpec)):
        return get_device_type(x.get_device())
    # pyrefly: ignore [bad-argument-type]
    assert_never(f"get_device_type({x}: {type(x).__name__})")


def is_triton(x: IRNode | torch.device | None | str) -> bool:
    # [FUNCTION DOC] Return True if the device of x is scheduled by a Triton-based backend. For cpu, cuda and xpu this
    # reads the corresponding backend setting in config; for other devices it checks whether the registered scheduling
    # class is a TritonScheduling subclass. Returns False for unknown devices or None. Raises AssertionError if the
    # registered scheduling is not a class.
    device = get_device_type(x)
    # Special case cpu and cuda as using the method below
    # to determine if the scheduler is a triton scheduler subclass
    # requires instantiating a scheduler for them
    if device in ["cpu", "cuda", "xpu"]:
        if getattr(config, f"{device}_backend") == "triton":
            return True
        return False
    if (
        device is None
        or (device_scheduling := get_scheduling_for_device(device)) is None
    ):
        return False
    from .codegen.triton import TritonScheduling

    if not isinstance(device_scheduling, type):
        raise AssertionError(type(device_scheduling))
    return issubclass(device_scheduling, TritonScheduling)


def is_cpu(x: IRNode | torch.device | None | str) -> bool:
    # [FUNCTION DOC] Return True if x refers to a cpu device.
    return get_device_type(x) == "cpu"


def is_aligned_realized_tensor(x: Buffer | TensorBox, alignment: int) -> bool:
    # [FUNCTION DOC] Return True if the realized buffer/tensor x is aligned: all strides but the last are multiples of
    # alignment and the last stride is 1 (or the last dim has size at most 1). Returns False for non-IRNodes, unknown
    # strides, or strides/sizes with unbacked symbols. The result is obtained via a guard-or-false on the size
    # variables, which may add a guard so the graph recompiles when the answer changes.
    if (
        not isinstance(x, IRNode)
        or x.maybe_get_stride() is None
        or free_unbacked_symbols(x.get_stride())
        or free_unbacked_symbols(x.get_size())
    ):
        return False

    aligned_strides = sympy.And(
        *(sympy.Eq(Mod(s, alignment), 0) for s in x.get_stride()[:-1])
    )
    aligned_last_dim = sympy.Or(
        sympy.Eq(x.get_stride()[-1], 1), sympy.Le(x.get_size()[-1], 1)
    )
    is_aligned = sympy.And(aligned_strides, aligned_last_dim)

    # Make sure to guard to recompile when necessary.
    return V.graph.sizevars.guard_or_false(is_aligned)


def significant_strides_equal(
    strides1: Sequence[_IntLike],
    strides2: Sequence[_IntLike],
    shape: Sequence[_IntLike],
) -> bool:
    # [FUNCTION DOC] Return True if two stride lists are equal ignoring dimensions of size 0 or 1 (empty tensors compare
    # equal always). Raises AssertionError if the lengths of shape and the strides differ.
    # Uses guard-or-false comparisons, so it may add guards.
    """
    Returns true if the strides are equal, ignoring dimensions of size 0 or 1.
    If any dimension is size 0, all strides are insignificant since the tensor
    is empty.
    """
    if not (len(shape) == len(strides1) and len(strides1) == len(strides2)):
        raise AssertionError(
            "Expected len(shape) == len(strides1) and len(strides1) == len(strides2)"
        )
    if any(V.graph.sizevars.statically_known_equals(dim, 0) for dim in shape):
        return True
    for dim, s1, s2 in zip(shape, strides1, strides2):
        if V.graph.sizevars.statically_known_leq(dim, 1):
            continue

        if not V.graph.sizevars.guard_or_false(sympy.Eq(s1, s2)):
            return False
    return True


def try_match_insignificant_strides(
    tensor: IRNode,
    strides: Sequence[int | torch.SymInt],
) -> IRNode:
    # [FUNCTION DOC] If tensor is realized with a layout and its strides differ from strides only at insignificant dims (size 0/1),
    # return a new TensorBox over a ReinterpretView with those dims' strides replaced by the requested ones. Otherwise
    # (unrealized, already equal, or real stride differences such as NHWC vs NCHW) return the input unchanged.
    """
    Tries to match the strides of the tensor to those in the meta_strides. Strides of insignificant
    dimensions - size 0 or 1 - will be updated.

    If there are real stride differences (NHWC vs NCHW), or the tensor is not realized, then the input will be returned
    """
    if not is_storage_and_layout(tensor):
        return tensor

    if all(
        V.graph.sizevars.statically_known_equals(s1, s2)
        for s1, s2 in zip(strides, tensor.get_stride())
    ):
        return tensor

    if not significant_strides_equal(strides, tensor.get_stride(), tensor.get_size()):
        return tensor

    storage, old_layout = as_storage_and_layout(tensor)
    new_stride = [*old_layout.stride]
    is_empty = tensor.is_zero_elements()
    for i, s in enumerate(tensor.get_size()):
        if is_empty or V.graph.sizevars.statically_known_leq(s, 1):
            new_stride[i] = strides[i]

    new_layout = FixedLayout(
        old_layout.device,
        old_layout.dtype,
        old_layout.size,
        new_stride,
        old_layout.offset,
        old_layout.is_pinned,
    )
    return TensorBox(ReinterpretView(data=storage, layout=new_layout))


def gm_original_output_strides(gm: torch.fx.GraphModule) -> None:
    # [FUNCTION DOC] Mark every output of the fx graph module as user visible and record the original output strides
    # via compile_fx.record_original_output_strides. Mutates node metadata on the output node of gm; returns None.
    output_node = gm.graph.find_nodes(op="output")[0]
    output_node.meta["user_visible_output_idxs"] = [
        idx for idx, _ in enumerate(output_node.args)
    ]
    from torch._inductor.compile_fx import record_original_output_strides

    record_original_output_strides(gm)


def get_symbolic_inputs(inputs: Sequence[IRNode]) -> list[Expr]:
    # [FUNCTION DOC] Collect the free symbols (backed and unbacked) appearing in the sizes and strides of the given
    # input nodes and return them as a list with duplicates removed.
    sym_vars: OrderedSet[Expr] = OrderedSet()
    for inp in inputs:
        sym_vars |= get_free_symbols(inp.get_size(), unbacked_only=False)
        sym_vars |= get_free_symbols(inp.get_stride(), unbacked_only=False)

    return list(sym_vars)


def try_get_name(x):
    # [FUNCTION DOC] HELPER: Peel TensorBox, view and StorageBox wrappers off x and return the name of the resulting
    # Buffer, or None if the innermost object is not a Buffer.
    if isinstance(x, TensorBox):
        x = x.data
    if isinstance(x, BaseView):
        x = x.unwrap_view()
    if isinstance(x, StorageBox):
        x = x.data
    return x.get_name() if isinstance(x, Buffer) else None


class IRNode:
    # [CLASS DOC] Abstract base class of all Inductor IR nodes. Holds class-level context shared by node construction
    # (the current set of origin fx nodes and the current stream index) and defines the interface (size, stride, dtype,
    # device, name, realize, codegen hooks, dependency/read-write queries) that concrete subclasses override;
    # most base methods raise NotImplementedError.
    """Base class for all intermediate representation (IR) nodes in TorchInductor.

    Note:
        This is an abstract base class. Most methods raise NotImplementedError
        and must be overridden by concrete subclasses.
    """

    _current_origins: ClassVar[OrderedSet[Any]] = OrderedSet()
    _current_stream_idx: ClassVar[int | None] = None
    _current_mempool: ClassVar[tuple[int, int] | None] = None

    # NB: These are kinda weird,
    origins: OrderedSet[Any] = dataclasses.field(init=False)
    # traces back to where the IRNode is created in Inductor
    traceback: list[str] | None = dataclasses.field(init=False)
    origin_node: torch.fx.Node | None = dataclasses.field(init=False)
    # Annotations dict for storing metadata (e.g., KernelTemplateChoice)
    annotations: dict[str, object] = dataclasses.field(init=False)
    # User-annotated stream index from FX node metadata (set during lowering)
    stream_idx: int | None = dataclasses.field(init=False)
    # User-annotated CUDA MemPool from FX node metadata (set during lowering)
    mempool: tuple[int, int] | None = dataclasses.field(init=False)

    @staticmethod
    @contextlib.contextmanager
    def current_origins(origins: OrderedSet[Node]) -> Generator[None, None, None]:
        # [FUNCTION DOC] Context manager that temporarily extends the class-level set of current origin fx nodes with the given origins; nodes created inside record them. Restores the previous set on exit, even on error.
        old = IRNode._current_origins
        IRNode._current_origins = old | origins
        try:
            yield
        finally:
            IRNode._current_origins = old

    @staticmethod
    @contextlib.contextmanager
    def current_stream_idx(
        stream_idx: int | None,
    ) -> Generator[None, None, None]:
        # [FUNCTION DOC] Context manager that sets the class-level current stream index for nodes created inside it and restores the previous value on exit.
        old = IRNode._current_stream_idx
        IRNode._current_stream_idx = stream_idx
        try:
            yield
        finally:
            IRNode._current_stream_idx = old

    @staticmethod
    @contextlib.contextmanager
    def current_mempool(
        mempool: tuple[int, int] | None,
    ) -> Generator[None, None, None]:
        # [FUNCTION DOC] Context manager that sets the class-level current CUDA mempool (pair of ints) for nodes created inside it and restores the previous value on exit.
        old = IRNode._current_mempool
        IRNode._current_mempool = mempool
        try:
            yield
        finally:
            IRNode._current_mempool = old

    @staticmethod
    def is_realized_node(node: IRNode) -> bool:
        # [FUNCTION DOC] Return True if node is already materialized (ComputedBuffer, InputsKernel, InputBuffer, ReinterpretView or TemplateBuffer).
        return isinstance(
            node,
            (
                ComputedBuffer,
                InputsKernel,
                InputBuffer,
                ReinterpretView,
                TemplateBuffer,
            ),
        )

    def wrap_for_lowering(self) -> IRNode:
        # [FUNCTION DOC] Wrap this node in a TensorBox so it can be returned from lowerings.
        return TensorBox.create(self)

    def _post_init_setattr(self, attr: str, value: Any) -> None:
        # [FUNCTION DOC] HELPER: Set an attribute on a (possibly frozen) dataclass instance bypassing its setattr; for use in post-init invariants and provenance info.
        # Intended for use in __post_init__ for enforcing an invariant on a dataclass
        # If you must, can also be used for setting provenance info
        # We would like to try and minimize these usages though
        object.__setattr__(self, attr, value)

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Post-init hook: records current origins, optional creation traceback (if config.debug_ir_traceback), an empty origin_node, an empty annotations mapping, and the current stream index and mempool onto the node.
        origins = OrderedSet(self._current_origins)
        self._post_init_setattr("origins", origins)
        self._post_init_setattr(
            "traceback", traceback.format_stack() if config.debug_ir_traceback else None
        )
        self._post_init_setattr("origin_node", None)
        # Annotations dict for storing metadata (e.g., KernelTemplateChoice)
        self._post_init_setattr("annotations", {})
        self._post_init_setattr("stream_idx", self._current_stream_idx)
        self._post_init_setattr("mempool", self._current_mempool)

    def get_read_names(self) -> OrderedSet[str]:
        # [FUNCTION DOC] Return the set of buffer names read by this node (names of its reads).
        return OrderedSet(dep.name for dep in self.get_reads())

    def get_traceback(self) -> list[str] | None:
        # [FUNCTION DOC] Return the recorded creation traceback, or None.
        return self.traceback

    def get_origin_node(self) -> torch.fx.Node | None:
        # [FUNCTION DOC] Return the fx node this IR node was lowered from, if any.
        return self.origin_node

    def get_defining_op(self) -> Operation | None:
        # [FUNCTION DOC] Return the Operation that defines this node; the base class has none and returns None.
        return None

    def get_subgraphs(self) -> list[Subgraph]:
        # [FUNCTION DOC] Return the subgraphs contained in this node; empty list in the base class.
        """Return subgraphs contained in this node"""
        return []

    def get_stack_traces(self) -> OrderedSet[str]:
        # [FUNCTION DOC] Collect user-model stack traces for this node as an ordered set. Uses the origin fx nodes (for ExternKernel with an origin node, only that node); falls back to pre-grad node mapping in the inductor debug tables when a node has no stack trace.
        # Return stack traces to user model code
        # A single IRNode could correspond to multiple lines of code
        stack_traces: OrderedSet[str] = OrderedSet()
        origins = self.origins
        if isinstance(self, ExternKernel):
            origin_node = self.get_origin_node()
            if self.origin_node:
                origins = OrderedSet([origin_node])
        for node in origins:
            if hasattr(node, "stack_trace") and node.stack_trace:
                # nodes in the backward graph don't have mapping to pre_grad_graph
                stack_traces.add(node.stack_trace)
            else:
                pre_grad_nodes = (
                    torch._inductor.debug._inductor_post_to_pre_grad_nodes.get(
                        "postToPre",
                        {},
                        # pyrefly: ignore [missing-attribute]
                    ).get(node.name, [])
                )
                if not isinstance(pre_grad_nodes, list):
                    continue
                for node_name in pre_grad_nodes:
                    stack_trace = (
                        torch._inductor.debug._inductor_pre_grad_node_stack_trace.get(
                            node_name, None
                        )
                    )
                    if stack_trace:
                        stack_traces.add(stack_trace)
        return stack_traces

    def common_repr(self, shorten: bool = True) -> Sequence[str]:
        # [FUNCTION DOC] HELPER: Build repr lines: the origins (truncated to about 64 chars if shorten) followed by any stack traces wrapped in braces.
        origins = f"origins={getattr(self, 'origins', '')}"
        if shorten and len(origins) > 64:
            # this can get *very* long
            origins = f"{origins[:61]}..."
        if not self.get_stack_traces():
            return [origins]

        stack_trace_str = []
        for stack_trace in self.get_stack_traces():
            stack_trace_str.append("stack_traces = {")
            stack_trace_str += stack_trace.split("\n")
            stack_trace_str.append("}")
        return [origins] + stack_trace_str

    def str_helper(
        self, lines: Sequence[object], shorten: bool = True, multiline: bool = True
    ) -> str:
        # [FUNCTION DOC] HELPER: Format a repr string of the form ClassName(lines...) from the given lines plus common repr lines; multiline puts each item on its own indented line.
        lines = list(lines) + list(self.common_repr(shorten))
        lines = list(map(str, lines))
        if multiline:
            # pyrefly: ignore [no-matching-overload]
            new_lines = indent(",\n".join(lines))
            return f"{type(self).__name__}(\n{new_lines}\n)"
        else:
            return f"{type(self).__name__}({lines})"

    def get_dtype(self) -> torch.dtype:
        # [FUNCTION DOC] Return the element dtype stored on the node.
        return self.dtype

    def maybe_get_dtype(self) -> torch.dtype | None:
        # [FUNCTION DOC] Return dtype, or None if get_dtype raises NotImplementedError.
        try:
            return self.get_dtype()
        except NotImplementedError:
            return None

    def get_layout(self) -> Layout:
        # [FUNCTION DOC] Return the node's Layout; base class raises NotImplementedError.
        raise NotImplementedError(f"get_layout() is not implemented by {type(self)}!")

    def maybe_get_layout(self) -> Layout | None:
        # [FUNCTION DOC] Return the layout or None when not implemented.
        try:
            return self.get_layout()
        except NotImplementedError:
            return None

    def get_output_spec(self) -> OutputSpec:
        # [FUNCTION DOC] Return the output spec, which by default is the layout.
        return self.get_layout()

    def maybe_get_output_spec(self) -> OutputSpec | None:
        # [FUNCTION DOC] Return the output spec or None when not implemented.
        try:
            return self.get_output_spec()
        except NotImplementedError:
            return None

    def has_tensor_output(self) -> bool:
        # [FUNCTION DOC] Return True for a single tensor output (the output spec is a Layout); False for multi-output nodes.
        """True for single tensor output (excludes MultiOutput)"""
        return isinstance(self.maybe_get_output_spec(), Layout)

    def get_size(self) -> Sequence[_IntLike]:
        # [FUNCTION DOC] Return the size (sequence of ints/exprs); base class raises NotImplementedError.
        raise NotImplementedError(f"get_size() is not implemented by {type(self)}!")

    def maybe_get_size(self) -> Sequence[_IntLike] | None:
        # [FUNCTION DOC] Return the size or None when not implemented.
        try:
            return self.get_size()
        except NotImplementedError:
            return None

    @property
    def shape(self) -> _IntLike | sympy.Rel | Sequence[_IntLike]:
        # [FUNCTION DOC] Property alias for get_size.
        return self.get_size()

    def get_numel(self) -> Expr:
        # [FUNCTION DOC] Return the symbolic product of the sizes.
        return sympy_product(self.get_size())

    def is_zero_elements(self) -> bool:
        # [FUNCTION DOC] Return True if the node is statically known to have zero elements.
        return V.graph.sizevars.statically_known_true(sympy.Eq(self.get_numel(), 0))

    def realize(self) -> str | None:
        # [FUNCTION DOC] Materialize the node into memory, ending fusion into it; the base class raises NotImplementedError. Subclasses return an optional buffer name.
        """
        If the IRNode refers to data which has not been materialized (e.g.,
        it is a Pointwise/Reduction that could potentially have more
        compute fused into it), realize the IRNode into physical memory,
        ending the possibility of fusing into it, but allowing, e.g., multiple
        users to access the data without having to recompute.

        Check StorageBox.realize for a particularly notable implementation.

        TODO(ezyang): I think, in principle, every IRNode should have an
        implementation of this, and most of the time no-op is OK, but you
        really do have to audit each IRNode for this, so for now, raise
        an error if it's not implemented.  Note that some code in graph.py
        will catch this thrown error and suppress it with a warning.
        """
        raise NotImplementedError(f"realize NYI on {type(self)}")

    def codegen_reference(self, writer: IndentedBuffer | None = None) -> str:
        # [FUNCTION DOC] Return the string used by wrapper codegen to refer to this node; base raises NotImplementedError.
        raise NotImplementedError(f"codegen_reference NYI on {type(self)}")

    def get_device(self) -> torch.device | None:
        # [FUNCTION DOC] Return the device of the node; None in the base class.
        return None

    def get_device_or_error(self) -> torch.device:
        # [FUNCTION DOC] Return the device, raising AssertionError if it is None.
        device = self.get_device()
        if device is None:
            raise AssertionError("Expected device is not None")
        return device

    def has_exceeded_max_reads(self) -> bool:
        # [FUNCTION DOC] Return whether the node exceeds the maximum read count; False in the base class.
        return False

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        # [FUNCTION DOC] Return a function mapping an index to an ops value that loads this node; base raises NotImplementedError.
        raise NotImplementedError(type(self).__name__)

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        # [FUNCTION DOC] Return a function mapping an index to a flat offset expression; base raises NotImplementedError.
        raise NotImplementedError(type(self).__name__)

    def get_stride(self) -> Sequence[_IntLike]:
        # [FUNCTION DOC] Return the strides; base raises NotImplementedError.
        raise NotImplementedError(type(self).__name__)

    def maybe_get_stride(self) -> Sequence[_IntLike] | None:
        # [FUNCTION DOC] Return strides or None when not implemented.
        try:
            return self.get_stride()
        except NotImplementedError:
            return None

    def get_name(self) -> str:
        # [FUNCTION DOC] Return the buffer name; base raises NotImplementedError.
        raise NotImplementedError(type(self).__name__)

    def maybe_get_name(self) -> str | None:
        # [FUNCTION DOC] Return the name or None when not implemented.
        try:
            return self.get_name()
        except NotImplementedError:
            return None

    def is_input_buffer(self) -> bool:
        # [FUNCTION DOC] Return True if this node's name is a graph input; False if it has no name.
        try:
            return self.get_name() in V.graph.graph_inputs
        except NotImplementedError:
            return False

    def has_large_inner_fn(self, threshold: int | None = None) -> bool:
        # [FUNCTION DOC] Return whether the node has a large inner function; False in the base class.
        return False

    def mark_reuse(self, users: int, *, graph_reuse: bool = True) -> None:
        # [FUNCTION DOC] Hint that this node's value will be reused users times; graph_reuse says whether it is graph fanout. No-op in the base class.
        """
        Hint that this node's value will be reused.

        `users` estimates the reuse count. When `graph_reuse` is true, this
        represents graph fanout and can trigger fanout-based realization
        heuristics. When false, the reuse comes from loop-level indexing, such
        as a broadcast expand, and should not be treated as graph fanout.
        """

    def realize_hint(self) -> None:
        # [FUNCTION DOC] Hint that the node may be worth realizing; no-op in the base class.
        pass

    def unwrap_view(self) -> IRNode:
        # [FUNCTION DOC] Return the underlying non-view node; base raises NotImplementedError.
        raise NotImplementedError(type(self).__name__)

    def freeze_layout(self) -> None:
        # [FUNCTION DOC] Fix a flexible layout in place; base raises NotImplementedError.
        raise NotImplementedError(type(self).__name__)

    def freeze_layout_with_stride_order(
        self, order: Sequence[int], allow_padding: bool = False
    ) -> None:
        # [FUNCTION DOC] Freeze the layout to the given stride order, optionally allowing padding; base raises NotImplementedError.
        raise NotImplementedError(type(self).__name__)

    def freeze_layout_with_fill_order(self, order: Sequence[int]) -> None:
        # [FUNCTION DOC] Freeze the layout to the given fill order; base raises NotImplementedError.
        raise NotImplementedError(type(self).__name__)

    def freeze_layout_with_same_order(self, stride: Sequence[_IntLike]) -> None:
        # [FUNCTION DOC] Freeze the layout to follow the order of the given stride; base raises NotImplementedError.
        raise NotImplementedError(type(self).__name__)

    def freeze_layout_with_exact_strides(
        self, exact_strides: Sequence[_IntLike], allow_padding: bool = False
    ) -> None:
        # [FUNCTION DOC] Freeze the layout to exact strides, optionally allowing padding; base raises NotImplementedError.
        raise NotImplementedError(type(self).__name__)

    def get_read_writes(self) -> dependencies.ReadWrites:
        # [FUNCTION DOC] Return the ReadWrites dependency summary; base raises NotImplementedError.
        raise NotImplementedError(type(self).__name__)

    def get_reads(self) -> OrderedSet[Dep]:
        # [FUNCTION DOC] Return the set of read dependencies from the read/writes summary.
        return self.get_read_writes().reads

    def num_reads(self) -> int:
        # [FUNCTION DOC] Return the number of read dependencies.
        return len(self.get_reads())

    def get_storage_numel(self) -> _IntLike:
        # [FUNCTION DOC] Return the storage element count; base raises NotImplementedError.
        raise NotImplementedError(type(self).__name__)

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return free symbols used by the node (only unbacked ones if requested); base raises NotImplementedError.
        raise NotImplementedError(type(self).__name__)

    def get_reduction_type(self) -> str | None:
        # [FUNCTION DOC] Return the reduction type name; base raises NotImplementedError.
        raise NotImplementedError(type(self).__name__)

    def get_reduction_size(self) -> Sequence[Expr]:
        # [FUNCTION DOC] Return the reduction sizes; base raises NotImplementedError.
        raise NotImplementedError(type(self).__name__)

    def is_extern(self) -> bool:
        # [FUNCTION DOC] Return True if the node is an extern kernel call; False in the base class.
        return False

    def is_no_op(self) -> bool:
        # [FUNCTION DOC] Return True if the node generates no code; False in the base class.
        return False

    def constant_to_device(self, device: torch.device) -> IRNode:
        # [FUNCTION DOC] Return a copy of a constant moved to the device; base raises NotImplementedError.
        raise NotImplementedError(type(self).__name__)

    def get_mutation_names(self) -> Sequence[str]:
        # [FUNCTION DOC] Return names of buffers mutated; base raises NotImplementedError.
        raise NotImplementedError(type(self).__name__)

    def get_operation_name(self) -> str:
        # [FUNCTION DOC] Return the operation name; base raises NotImplementedError.
        raise NotImplementedError(type(self).__name__)

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        # [FUNCTION DOC] Return names of inputs that alias the output; base raises NotImplementedError.
        raise NotImplementedError(type(self).__name__)

    if TYPE_CHECKING:

        @property
        def dtype(self) -> torch.dtype: ...


@ir_dataclass(frozen=False)
class Operation:
    # [CLASS DOC] Base class of IR nodes that perform computation (as opposed to data). Tracks an operation_name and per-operation config patches, and exposes read/write dependency, origin, stream and mempool queries that subclasses supply or override.
    def __post_init__(self) -> None:
        # [FUNCTION DOC] Post-init hook: initialises operation_name to None and config patches to an empty mapping.
        self.operation_name: str | None = None
        self._config_patches: dict[str, Any] = {}

    def get_device(self) -> torch.device | None:
        # [FUNCTION DOC] Return the device; subclasses must override (raises NotImplementedError).
        raise NotImplementedError

    def get_origin_node(self) -> torch.fx.Node | None:
        # [FUNCTION DOC] Return the origin fx node; raises AssertionError if the attribute is missing.
        if not hasattr(self, "origin_node"):
            raise AssertionError('Expected hasattr(self, "origin_node")')
        return self.origin_node

    def get_origins(self) -> OrderedSet[Any]:
        # [FUNCTION DOC] Return the set of origin nodes; raises AssertionError if the attribute is missing.
        if not hasattr(self, "origins"):
            raise AssertionError('Expected hasattr(self, "origins")')
        return self.origins

    def get_stream_idx(self) -> int | None:
        # [FUNCTION DOC] Return the stream index; raises AssertionError if the attribute is missing.
        if not hasattr(self, "stream_idx"):
            raise AssertionError('Expected hasattr(self, "stream_idx")')
        return self.stream_idx

    def get_mempool(self) -> tuple[int, int] | None:
        # [FUNCTION DOC] Return the mempool; raises AssertionError if the attribute is missing.
        if not hasattr(self, "mempool"):
            raise AssertionError('Expected hasattr(self, "mempool")')
        return self.mempool

    def get_operation_name(self) -> str:
        # [FUNCTION DOC] Return the operation name; raises AssertionError if it has not been assigned.
        if self.operation_name is None:
            raise AssertionError("Expected self.operation_name is not None")
        return self.operation_name

    def get_buffer_name(self) -> str | None:
        # [FUNCTION DOC] Return the name of the output buffer if there is a single one; None here.
        return None

    def get_config_patches(self) -> dict[str, Any]:
        # [FUNCTION DOC] Return the config patches mapping for this operation.
        """Get config patches for this operation (e.g., coordinate_descent_tuning)."""
        return self._config_patches

    def set_config_patches(self, patches: dict[str, Any]) -> None:
        # [FUNCTION DOC] Replace the config patches mapping for this operation.
        """Set config patches for this operation."""
        self._config_patches = patches

    def is_extern(self) -> bool:
        # [FUNCTION DOC] Return True for extern kernels; False here.
        return False

    def is_no_op(self) -> bool:
        # [FUNCTION DOC] Return True if the operation produces no code; False here.
        return False

    def get_read_writes(self) -> dependencies.ReadWrites:
        # [FUNCTION DOC] Return read/write dependencies; subclasses must override (NotImplementedError).
        raise NotImplementedError

    def is_user_of(self, name: str) -> bool:
        # [FUNCTION DOC] Return True if this operation reads the buffer with the given name.
        return name in self.get_read_names()

    def get_read_names(self) -> OrderedSet[str]:
        # [FUNCTION DOC] Return the set of buffer names read.
        return OrderedSet(dep.name for dep in self.get_reads())

    def get_reads(self) -> OrderedSet[Dep]:
        # [FUNCTION DOC] Return the set of read dependencies.
        return self.get_read_writes().reads

    def get_outputs(self) -> list[Buffer]:
        # [FUNCTION DOC] Return the list of output buffers; subclasses must override (NotImplementedError).
        raise NotImplementedError

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return the unbacked symbols defined by this operation; empty here.
        return OrderedSet()

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return the free symbols that must be in scope to codegen this operation (only unbacked ones if requested). Not transitive. Empty here.
        """
        When unbacked_only=True:
        Returns the unbacked symbols which are required to be in scope in
        order to successfully perform codegen for this buffer.  For example,
        a buffer that corresponds to an extern kernel call that takes i0 as
        an argument would return {i0} here.  This is used to generate necessary
        dependencies that ensure we actually bind i0 in codegen before you
        try to use it.

        Note that this is NOT transitive; in particular, if this buffer takes
        in as input another buffer with dynamic shape (e.g., (i0,)), we will
        not report it here, because you will already have a dependency
        on that buffer, which will eventually have a dependency on i0 if
        necessary.

        When unbacked_only=False:
        Similar to `unbacked_only=True` but including all free symbols
        instead of only free unbacked symbols.
        """
        return OrderedSet()

    def get_workspace_size(self) -> int:
        # [FUNCTION DOC] Return the extra global workspace memory in bytes the operation needs; 0 here.
        """
        Gets extra global memory size needed by this buffer.
        Some algorithms (e.g. group gemm) may require extra global memory in the generated code.
        """
        return 0


@ir_dataclass
class Loops(IRNode):
    # [CLASS DOC] Base class of loop-body IR nodes (Pointwise, Reduction, etc.). Fields: device, dtype, inner_fn (callable producing an ops value from index variables) and ranges (iteration sizes). Provides size, free-symbol, op-count and read-dependency queries, and a create helper that wraps results in a TensorBox.
    """Base class for pointwise and reduction loop-body IR nodes."""

    device: torch.device
    dtype: torch.dtype
    inner_fn: Callable[..., Any]
    ranges: Sequence[_IntLike]

    @cache_on_self_and_args("Loops")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return free symbols from the ranges and the inner function (cached per args).
        return OrderedSet().union(
            *(get_free_symbols(e, unbacked_only) for e in self.ranges),
            self.inner_fn_free_symbols(unbacked_only),
        )

    def _to_str(self, names: Sequence[str]) -> str:
        # [FUNCTION DOC] HELPER: Build a repr string from device, dtype, formatted inner function, the named attributes and the origin node.
        return self.str_helper(
            [
                f"'{self.device.type}'",
                str(self.dtype),
                self.inner_fn_str(),
            ]
            + [f"{name}={getattr(self, name)}" for name in names]
            + [f"origin_node={self.origin_node!r}"]
        )

    def __str__(self) -> str:
        # [FUNCTION DOC] Return the repr string showing the ranges.
        return self._to_str(("ranges",))

    __repr__ = __str__

    def get_device(self) -> torch.device | None:
        # [FUNCTION DOC] Return the device.
        return self.device

    def get_origin_node(self) -> torch.fx.Node | None:
        # [FUNCTION DOC] Return the origin fx node.
        return self.origin_node

    def get_size(self) -> Sequence[_IntLike]:
        # [FUNCTION DOC] Return the iteration ranges as the size.
        return self.ranges

    def get_pointwise_size(self) -> Sequence[_IntLike]:
        # [FUNCTION DOC] Return the pointwise iteration ranges (same as the size).
        return self.ranges

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> TensorBox:
        # [FUNCTION DOC] Class method: construct the node from the arguments, propagating the optional origin_node and traceback keyword arguments onto it, and return it wrapped in a TensorBox.
        origin_node = kwargs.pop("origin_node", None)
        tb = kwargs.pop("traceback", None)
        r = cls(*args, **kwargs)
        # Need to explicitly set origin_node here to propagate it down.
        # todo(chilli): I think it would be better for IRNode to directly set
        # origin_node
        r._post_init_setattr("origin_node", origin_node)
        r._post_init_setattr("traceback", tb or r.traceback)
        return TensorBox.create(r)

    @staticmethod
    def _index(ranges: Sequence[_IntLike], prefix: SymT = SymT.INDEX) -> Sequence[Expr]:
        # [FUNCTION DOC] HELPER: Build index symbols for the ranges using the prefix; ranges of size one get the constant zero.
        return [
            sympy.S.Zero if s == 1 else sympy_index_symbol_with_prefix(prefix, n)
            for n, s in enumerate(ranges)
        ]

    @cache_on_self
    def inner_fn_opcount(self) -> OpCountResult:
        # [FUNCTION DOC] Run the inner function under an op-counting handler (with flexible layout indexing allowed) and return the op count result; cached.
        opcounter = OpCounterCSE(V.MockHandler())
        with (
            V.set_ops_handler(opcounter),
            patch.object(FlexibleLayout, "allow_indexing", True),
        ):
            self.inner_fn(*self.inner_fn_args())
            return opcounter.getvalue()

    def inner_fn_args(self) -> Sequence[Sequence[_IntLike]]:
        # [FUNCTION DOC] Return the argument list for the inner function: one list of index symbols over the ranges.
        return (self._index(self.ranges),)

    @cache_on_self
    def inner_fn_str(self) -> str:
        # [FUNCTION DOC] Return the inner function rendered to a string via the kernel formatter handler; cached.
        return V.KernelFormatterHandler.ir_to_string(
            self.inner_fn, *self.inner_fn_args()
        )

    def get_realize_opcount_threshold(self, threshold: int | None = None) -> int:
        # [FUNCTION DOC] Return the op-count threshold above which the node is considered large: the larger of the given threshold (default 0) and the config value (cpu-specific or default if unset). Raises AssertionError if the config value is not an int.
        if threshold is None:
            threshold = 0
        realize_opcount_threshold = config.realize_opcount_threshold
        if realize_opcount_threshold is None:
            if is_cpu(self):
                realize_opcount_threshold = config.realize_cpu_opcount_threshold
            else:
                realize_opcount_threshold = config._realize_opcount_threshold_default
        else:
            if not isinstance(realize_opcount_threshold, int):
                raise AssertionError(
                    f"expected int realize_opcount_threshold, got {type(realize_opcount_threshold)}"
                )
        return max(threshold, realize_opcount_threshold)

    def has_large_inner_fn(self, threshold: int | None = None) -> bool:
        # [FUNCTION DOC] Return True if the inner function's op count exceeds the realize threshold.
        return self.inner_fn_opcount().num_ops > self.get_realize_opcount_threshold(
            threshold
        )

    def inner_fn_free_symbols(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        # [FUNCTION DOC] Return the free symbols used by the inner function, evaluated on index symbols for the ranges.
        index = self._index(self.ranges)
        return extract_free_symbols(self.inner_fn, index, unbacked_only=unbacked_only)

    # returns a list of op names which references the symbol
    def collect_inner_fn_symbol_usage(self, symbol: Symbol) -> OrderedSet[str]:
        # [FUNCTION DOC] Run the inner function with a handler that records which ops reference the given symbol and return those op names.
        index = self._index(self.ranges)
        handler = SymbolUsageCollectorOpsHandler(symbol)
        with (
            V.set_ops_handler(handler),
        ):
            self.inner_fn(index)
        return handler.usages

    def get_reads(self) -> OrderedSet[Dep]:
        # [FUNCTION DOC] Return the set of read dependencies by tracing the loader (and reduction sizes if this is a reduction) with flexible-layout indexing allowed.
        with patch.object(FlexibleLayout, "allow_indexing", True):
            if self.get_reduction_type():
                return extract_read_writes(
                    self.make_loader(),
                    self.get_size(),
                    self.get_reduction_size(),
                ).reads
            else:
                return extract_read_writes(
                    self.make_loader(),
                    self.get_size(),
                ).reads

    def get_read_names(self) -> OrderedSet[str]:
        # [FUNCTION DOC] Return the names of buffers read, taken from the cached op-count result of the inner function.
        return OrderedSet(self.inner_fn_opcount().read_buffers)

    def num_reads(self) -> int:
        # [FUNCTION DOC] Return the number of distinct buffers read according to the op-count result.
        return len(self.inner_fn_opcount().read_buffers)

    def get_reduction_size(self) -> Sequence[Expr]:
        # [FUNCTION DOC] Return the reduction sizes; base Loops raises NotImplementedError.
        raise NotImplementedError(
            f"get_reduction_size() is not implemented by {type(self)}!"
        )

    def get_reduction_type(self) -> str | None:
        # [FUNCTION DOC] Return the reduction type; base Loops raises NotImplementedError.
        raise NotImplementedError(
            f"get_reduction_type() is not implemented by {type(self)}!"
        )

    def constant_to_device(self, device: torch.device) -> IRNode:
        # [FUNCTION DOC] Move the node to another device; base Loops raises NotImplementedError.
        raise NotImplementedError(
            f"constant_to_device() is not implemented by {type(self)}!"
        )


def nop_loader_fn(idx: Expr | Sequence[Expr], *, dtype: torch.dtype) -> OpsValue:
    # [FUNCTION DOC] HELPER: Loader used for zero-element loops; returns a constant of the dtype (NaN for floating types, zero otherwise) regardless of the index.
    if dtype.is_floating_point:
        return ops.constant(float("nan"), dtype)
    else:
        return ops.constant(0, dtype)


@ir_dataclass
class Pointwise(Loops):
    # [CLASS DOC] Loops node for elementwise computation: each output element is computed by inner_fn from its index. Has no reduction (empty reduction size, type None) and a loader that is the inner function itself (or a no-op constant for empty outputs).
    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        # [FUNCTION DOC] Return the loader callable: a constant-producing no-op for zero-element nodes, otherwise the inner function.
        # Make zero-element loops into a no-op
        if self.is_zero_elements():
            return partial(nop_loader_fn, dtype=self.dtype)

        return self.inner_fn

    def __str__(self) -> str:
        # [FUNCTION DOC] Return the repr string showing the ranges.
        return self._to_str(("ranges",))

    __repr__ = __str__

    def get_reduction_size(self) -> Sequence[_IntLike]:
        # [FUNCTION DOC] Return an empty list (pointwise has no reduction dims).
        return []

    def get_reduction_type(self) -> str | None:
        # [FUNCTION DOC] Return None (pointwise has no reduction type).
        return None

    def store_output(
        self,
        output_name: str | None,
        indexer: Callable[[Sequence[Expr]], Never],
        vars: Sequence[Expr],
    ) -> None:
        # [FUNCTION DOC] Emit a store op writing the loaded value at vars to the output (named unnamed if None) at the position given by indexer.
        loader = self.make_loader()
        return ops.store(output_name or "unnamed", indexer(vars), loader(vars))

    def constant_to_device(self, device: torch.device) -> IRNode:
        # [FUNCTION DOC] Return an equivalent Pointwise on the given device whose loader is wrapped so constant buffers report that device. Valid only if all reads are constants.
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Pointwise(
            device=device,
            dtype=self.dtype,
            inner_fn=loader,
            ranges=self.ranges,
        )


@ir_dataclass
class Scatter(Pointwise):
    # [CLASS DOC] Pointwise node that writes to a computed output index (output_indexer) rather than the iteration index, with an optional store mode (scatter_mode, e.g. accumulate).
    output_indexer: Callable[[Sequence[Expr]], Expr]
    scatter_mode: StoreMode = None

    def constant_to_device(self, device: torch.device) -> IRNode:
        # [FUNCTION DOC] Return an equivalent Scatter on the given device (constants re-targeted), preserving output_indexer and scatter_mode. Valid only if all reads are constants.
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Scatter(
            device=device,
            dtype=self.dtype,
            inner_fn=loader,
            ranges=self.ranges,
            output_indexer=self.output_indexer,
            scatter_mode=self.scatter_mode,
        )

    def store_output(
        self,
        output_name: str | None,
        indexer: Callable[[Sequence[Expr]], Never],
        vars: Sequence[Expr],
    ) -> Any:
        # [FUNCTION DOC] Emit a store op writing the loaded value at the indexer-mapped output index, using scatter_mode as the store mode; returns the op result.
        loader = self.make_loader()
        if output_name is None:
            output_name = "unnamed"
        return ops.store(
            output_name,
            indexer(self.output_indexer(vars)),
            loader(vars),
            mode=self.scatter_mode,
        )


REDUCTION_COMBINE_FN: dict[str, Callable[..., OpsValue]] = {
    "any": ops_wrapper("logical_or"),
    "max": ops_wrapper("maximum"),
    "fmax": ops_wrapper("fmaximum"),
    "min": ops_wrapper("minimum"),
    "prod": ops_wrapper("mul"),
    "sum": ops_wrapper("add"),
    "dot": ops_wrapper("add"),
    "xor_sum": ops_wrapper("bitwise_xor"),
}


def get_reduction_combine_fn(
    reduction_type: str, dtype: torch.dtype, arg_break_ties_left: bool = True
) -> Callable[..., object]:
    # [FUNCTION DOC] Return the binary combine function for a reduction type. Simple types (any, max, fmax, min, prod, sum, dot, xor_sum) map to ops wrappers; for float max/min with config.strict_signed_zero the result is fixed up so equal inputs pick the second. Arg-reductions combine (value, index) pairs, with NaN-aware comparison for floats and ties broken toward the lower index when arg_break_ties_left else the higher. welford_combine merges (mean, m2, weight) triples, guarding against infinite equal means. Raises NotImplementedError for unknown types.
    if reduction_type in REDUCTION_COMBINE_FN:
        combine_fn = REDUCTION_COMBINE_FN[reduction_type]

        if (
            config.strict_signed_zero
            and reduction_type in ("max", "min")
            and is_float_dtype(dtype)
        ):

            def strict_signed_zero_combine_fn(a: object, b: object) -> OpsValue:
                # [FUNCTION DOC] HELPER: Combine two values then select the second when they compare equal (preserves signed-zero semantics).
                value = combine_fn(a, b)
                return ops.where(ops.eq(a, b), b, value)

            return strict_signed_zero_combine_fn

        return combine_fn

    elif reduction_type in (
        "argmax",
        "argmin",
        "argmax_value",
        "argmin_value",
        "argmax_with_value",
        "argmin_with_value",
    ):

        def argmax_combine_fn(
            a: tuple[object, object], b: tuple[object, object]
        ) -> tuple[OpsValue, OpsValue]:
            # [FUNCTION DOC] HELPER: Combine two (value, index) pairs, keeping the winner by lt/gt on value (NaN handled for floats) with ties resolved by index according to arg_break_ties_left; returns the new (value, index).
            a_value, a_index = a
            b_value, b_index = b

            if reduction_type in ("argmin", "argmin_value", "argmin_with_value"):
                mask = ops.lt(a_value, b_value)
            else:
                mask = ops.gt(a_value, b_value)

            equal = ops.eq(a_value, b_value)
            if is_float_dtype(dtype):
                a_isnan = ops.ne(a_value, a_value)
                b_isnan = ops.ne(b_value, b_value)
                mask = ops.logical_or(mask, ops.gt(a_isnan, b_isnan))
                equal = ops.logical_or(equal, ops.logical_and(a_isnan, b_isnan))

            tie = (
                ops.lt(a_index, b_index)
                if arg_break_ties_left
                else ops.gt(a_index, b_index)
            )
            mask = ops.logical_or(mask, ops.logical_and(equal, tie))
            return (
                ops.where(mask, a_value, b_value),
                ops.where(mask, a_index, b_index),
            )

        return argmax_combine_fn

    elif reduction_type == "welford_combine":

        def welford_combine_fn(
            a: tuple[OpsValue, OpsValue, OpsValue],
            b: tuple[OpsValue, OpsValue, OpsValue],
        ) -> tuple[OpsValue, OpsValue, OpsValue]:
            # [FUNCTION DOC] HELPER: Merge two Welford partial results (mean, m2, weight) into one using the parallel update formula; delta is zero when both means are the same infinity.
            a_mean, a_m2, a_weight = a
            b_mean, b_m2, b_weight = b

            # Guard against inf - inf = NaN when both means are infinite and
            # equal. This occurs during FP16/BF16 LayerNorm when inputs
            # overflow to inf.
            delta = ops.where(
                ops.logical_and(ops.isinf(a_mean), ops.eq(a_mean, b_mean)),
                ops.constant(0.0, torch.float32),
                b_mean - a_mean,
            )
            new_weight = a_weight + b_weight
            w2_over_w = b_weight / new_weight
            return (
                a_mean + delta * w2_over_w,
                a_m2 + b_m2 + delta * delta * a_weight * w2_over_w,
                new_weight,
            )

        return welford_combine_fn

    else:
        raise NotImplementedError(f"unknown reduction_type={reduction_type}")


@ir_dataclass
class Reduction(Loops):
    # [CLASS DOC] Loops node for a reduction. Extra fields: reduction_ranges, reduction_type, src_dtype (dtype is the destination dtype), reduction_hint, and strict_sum_multirow / strict_sum_rblock for exact eager-order sum tiling. inner_fn takes both the output index and the reduction index. Provides num_splits (split heuristics) and create (which may split a reduction into multiple stages).
    r"""IR node representing a reduction over one or more iteration dimensions."""

    reduction_ranges: Sequence[_IntLike]
    reduction_type: ReductionType
    # self.dtype represents the dst dtype
    src_dtype: torch.dtype
    reduction_hint: ReductionHint
    # Exact eager tile; rblock 1 is the split final stage.
    strict_sum_multirow: bool = False
    strict_sum_rblock: int | None = None

    def __str__(self) -> str:
        # [FUNCTION DOC] Return the repr string showing ranges, reduction ranges and type.
        return self._to_str(("ranges", "reduction_ranges", "reduction_type"))

    __repr__ = __str__

    @cache_on_self_and_args("Reduction")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        # [FUNCTION DOC] Return free symbols of the parent plus those in the reduction ranges (cached).
        return super().get_free_symbol_uses(unbacked_only) | OrderedSet().union(
            *(get_free_symbols(e, unbacked_only) for e in self.reduction_ranges)
        )

    def get_reduction_size(self) -> Sequence[_IntLike]:
        # [FUNCTION DOC] Return the reduction ranges.
        return self.reduction_ranges

    def get_reduction_type(self) -> str | None:
        # [FUNCTION DOC] Return the reduction type.
        return self.reduction_type

    def store_reduction(
        self,
        output_name: str | None,
        indexer: Callable[[Sequence[Expr]], Never],
        vars: Sequence[Expr],
        reduction_vars: Sequence[Symbol],
    ) -> None:
        # [FUNCTION DOC] Emit a reduction op over the inner function evaluated at vars and reduction_vars, then store the reduction result at the indexer position (output named unnamed if None).
        value = ops.reduction(
            self.dtype,
            self.src_dtype,
            self.reduction_type,
            self.inner_fn(vars, reduction_vars),
        )
        ops.store_reduction(output_name or "unnamed", indexer(vars), value)

    def index_length(self) -> int:
        # [FUNCTION DOC] Return the total number of index dimensions: output ranges plus reduction ranges.
        return len(self.ranges) + len(self.reduction_ranges)

    def inner_fn_args(self) -> Sequence[Sequence[Expr]]:
        # [FUNCTION DOC] Return the arguments for the inner function: output index symbols and reduction index symbols (r0 prefix).
        index = self._index(self.ranges)
        rindex = self._index(self.reduction_ranges, SymT.R0_INDEX)
        return (index, rindex)

    def inner_fn_free_symbols(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        # [FUNCTION DOC] Return free symbols used by the inner function evaluated on both output and reduction index symbols.
        index = self._index(self.ranges)
        rindex = self._index(self.reduction_ranges, SymT.R0_INDEX)
        return extract_free_symbols(
            self.inner_fn, index, rindex, unbacked_only=unbacked_only
        )

    def constant_to_device(self, device: torch.device) -> IRNode:
        # [FUNCTION DOC] Return an equivalent Reduction on the given device with constants re-targeted, copying strict-sum settings. The reduction hint is reset to DEFAULT. Valid only if all reads are constants.
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Reduction(
            device=device,
            dtype=self.dtype,
            inner_fn=loader,
            ranges=self.ranges,
            reduction_ranges=self.reduction_ranges,
            reduction_type=self.reduction_type,
            src_dtype=self.src_dtype,
            reduction_hint=ReductionHint.DEFAULT,
            strict_sum_multirow=self.strict_sum_multirow,
            strict_sum_rblock=self.strict_sum_rblock,
        )

    @staticmethod
    def num_splits(
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[_P, OpsValue],
        ranges: Sequence[_IntLike],
        reduction_ranges: Sequence[_IntLike],
        reduction_type: ReductionType | Literal["scan"],
        reduction_numel: Expr,
        input_node: IRNode | None = None,
    ) -> tuple[ReductionHint, _IntLike]:
        # [FUNCTION DOC] Static method choosing a (ReductionHint, split count) for a reduction. Returns (DEFAULT, 1) when unbacked symbols lack hints, for dot reductions, or when small/large enough not to split. Otherwise uses the backend's split factor (only when splitting is enabled: scan, or no single-element reduction, not arg-reduction, and config.split_reductions). For a single output element it returns INNER with the split, or -1 split when the input node's own ranges already match. Else it inspects read indices of a trial Reduction (possibly deciding input layouts, a side effect) and counts strides to classify the reduction as INNER or OUTER and returns the matching split.
        """
        Choose the reduction hint and split count from shape and input stride information.
        """
        # Use optimization_hint when all unbacked symbols have explicit hints,
        # otherwise fall back conservatively.
        exprs = [reduction_numel, sympy_product(ranges)]
        if not V.graph.sizevars.all_unbacked_explicitly_hinted(exprs):
            return ReductionHint.DEFAULT, 1
        reduction_numel_hint = V.graph.sizevars.optimization_hint(reduction_numel)
        numel = sympy_product(ranges)
        numel_hint = V.graph.sizevars.optimization_hint(numel)

        # The Triton backend adds REDUCE_TO_SINGLE_ELEMENT unconditionally if the
        # cooperative_reductions feature flag is enabled, but we should still use a
        # split scan if we don't actually do a cooperative reduction.
        should_reduce_to_single_element = V.graph.has_feature(
            device, BackendFeature.REDUCE_TO_SINGLE_ELEMENT
        ) and (
            not is_triton(device)
            or V.choices.should_use_cooperative_reduction(
                device, numel, reduction_numel
            )
        )
        arg_reduction_types = (
            "argmax",
            "argmin",
            "argmax_value",
            "argmin_value",
            "argmax_with_value",
            "argmin_with_value",
        )

        should_split = reduction_type == "scan" or (
            not should_reduce_to_single_element
            and reduction_type not in arg_reduction_types
            and config.split_reductions
        )

        if reduction_type == "dot":
            # Don't split when doing native matmul
            return ReductionHint.DEFAULT, 1

        props = DeviceProperties.create(device)
        num_sm = props.multi_processor_count
        min_elements_per_thread = 32
        if should_split:
            inner_reduction_splits: Callable[[int, int], int] = functools.partial(
                V.choices.reduction_split_factor, device, inner_reduction=True
            )
            outer_reduction_splits: Callable[[int, int], int] = functools.partial(
                V.choices.reduction_split_factor, device, inner_reduction=False
            )
        else:

            def inner_reduction_splits(
                reduction_numel_hint: int,
                numel_hint: int,
            ) -> int:
                # [FUNCTION DOC] HELPER: Split-factor stand-in used when splitting is disabled; always returns one.
                return 1

            outer_reduction_splits = inner_reduction_splits

        # easy cases
        if numel_hint == 1:
            split = inner_reduction_splits(reduction_numel_hint, numel_hint)
            if split == 1:
                # No need to split.
                return ReductionHint.INNER, split
            if input_node is not None and isinstance(input_node, TensorBox):
                with patch.object(FlexibleLayout, "allow_indexing", True):
                    (
                        new_ranges,
                        new_reduction_ranges,
                    ) = extract_input_node_reduction_ranges(input_node)
                if new_ranges is not None and new_reduction_ranges is not None:
                    extracted_numel_hint = (
                        V.graph.sizevars.replace_backed_symbols_with_hints(
                            sympy_product(new_ranges + new_reduction_ranges)
                        )
                    )
                    if reduction_numel_hint == extracted_numel_hint:
                        log.debug(
                            "Use previous IRNode's range and reduction_ranges instead of split. "
                            "current ranges: %s, current reduction ranges: %s, current split: %d, "
                            "new ranges: %s, new reduction ranges: %s",
                            ranges,
                            reduction_ranges,
                            split,
                            new_ranges,
                            new_reduction_ranges,
                        )
                        # If the input_node or its dependent nodes are also Reduction nodes,
                        # use reduction_sizes of this node or its dependent nodes directly.
                        return ReductionHint.INNER, -1
            return ReductionHint.INNER, split
        if (
            reduction_numel_hint <= min_elements_per_thread
            or numel_hint >= num_sm * 2 * 32
        ):
            return ReductionHint.DEFAULT, 1

        r = Reduction(
            device=device,
            dtype=dst_dtype,
            inner_fn=inner_fn,
            ranges=ranges,
            reduction_ranges=reduction_ranges,
            reduction_type=reduction_type if reduction_type != "scan" else "sum",
            src_dtype=src_dtype,
            reduction_hint=ReductionHint.DEFAULT,
        )

        def get_read_indices(r: Reduction) -> tuple[Sequence[Expr], bool]:
            # [FUNCTION DOC] HELPER: Build a temporary ComputedBuffer for the reduction and collect read index expressions that cover all output vars (full-size reads), else reads that vary over the reduction (broadcast reads). Also decides layouts of the buffers read and reports whether any stride changed. Returns (indices, changed). Raises AssertionError if the device or range vars are missing.
            device = r.get_device()
            if device is None:
                raise AssertionError("Expected device is not None")
            cb = ComputedBuffer(
                name=None,
                layout=FlexibleLayout(
                    device=device,
                    dtype=r.get_dtype(),
                    size=r.get_size(),
                ),
                data=r,
            )
            read_writes = cb.get_read_writes()
            # try finding the full size producer
            # TODO this will fail for something like ((1, N) * (N, 1)).sum()
            # this would also possibly be wrong for producers with the different contiguity but we hope those cases are rare
            if read_writes.range_vars is None:
                raise AssertionError("Expected read_writes.range_vars is not None")
            range_vars = [
                var
                for var in read_writes.range_vars
                if isinstance(var, Expr) and not isinstance(var, sympy.Number)
            ]
            (_, reduction_vars), _ = dependencies.index_vars_squeeze(
                r.get_size(), r.get_reduction_size()
            )
            reduction_vars = [
                var
                for var in reduction_vars
                if isinstance(var, Expr) and not isinstance(var, sympy.Number)
            ]
            indices = []
            broadcasted_reduction_indices = []
            changed = False
            for md in sorted(read_writes.reads, key=lambda x: x.name):
                free_symbols = md.index.free_symbols
                is_full_size_read = all(var in free_symbols for var in range_vars)
                # Prefer full-size reads, but fall back to reads that still
                # vary across the reduction.  Missing reduction vars are
                # zero-stride broadcasts, not proof of an inner reduction.
                is_broadcasted_reduction_read = not is_full_size_read and any(
                    var in free_symbols for var in reduction_vars
                )
                if is_full_size_read:
                    indices.append(md.index)
                elif is_broadcasted_reduction_read:
                    broadcasted_reduction_indices.append(md.index)
                if is_full_size_read or is_broadcasted_reduction_read:
                    if md.name in V.graph.name_to_buffer:
                        buf = V.graph.name_to_buffer[md.name]
                        original_stride = getattr(buf.layout, "stride", None)
                        buf.decide_layout()
                        if getattr(buf.layout, "stride", None) != original_stride:
                            changed = True
            return indices or broadcasted_reduction_indices, changed

        indices, changed = get_read_indices(r)
        if changed:
            indices, _ = get_read_indices(r)

        if len(indices) == 0:
            # TODO determine splits when all inputs are broadcast
            return ReductionHint.DEFAULT, 1

        (_, reduction_vars), ranges1 = dependencies.index_vars_squeeze(
            r.get_size(), r.get_reduction_size()
        )
        num_outer = 0
        num_inner = 0
        for i in indices:
            j = V.graph.sizevars.simplify_with_ranges(i, ranges1)
            strides = V.graph.sizevars.stride_hints(
                j, reduction_vars, list(ranges1.keys())
            )
            # A 0 stride does not make a reduction contiguous.
            # This can happen when the reduction ranges contains a 1.
            outer = all(s == 0 or s > 1 for s in strides)
            if outer:
                num_outer += 1
            else:
                num_inner += 1
        if num_inner > num_outer:
            return ReductionHint.INNER, inner_reduction_splits(
                reduction_numel_hint, numel_hint
            )
        else:
            return ReductionHint.OUTER, outer_reduction_splits(
                reduction_numel_hint, numel_hint
            )

    @staticmethod
    def _unroll_reduction_fn(
        inner_fn: Callable[[Sequence[_IntLike], Sequence[_IntLike]], OpsValue],
        reduction_ranges: Sequence[_IntLike],
        reduction_type: str,
        src_dtype: torch.dtype,
    ) -> Callable[[Sequence[_IntLike]], OpsValue]:
        # [FUNCTION DOC] HELPER: Convert a reduction inner function into a pointwise function by statically unrolling the reduction range (guarded to ints) and folding values with the combine function. For arg-reductions the values are (value, flat index) pairs and the result selects the value, the index, or both depending on type.
        """Convert inner_fn from a reduction to an pointwise"""
        reduction_ranges = V.graph.sizevars.guard_int_seq(reduction_ranges)

        combine_fn = get_reduction_combine_fn(reduction_type, src_dtype)

        def fn(index: Sequence[_IntLike]) -> Any:
            # [FUNCTION DOC] HELPER: Fold the per-reduction-index values at an output index with the combine function.
            return functools.reduce(
                combine_fn,
                (
                    value_fn(index, rindex)
                    for rindex in itertools.product(
                        *[range(x) for x in reduction_ranges]
                    )
                ),
            )

        value_fn: Callable[[Sequence[_IntLike], Sequence[_IntLike]], Any]
        if reduction_type in (
            "argmin",
            "argmax",
            "argmin_value",
            "argmax_value",
            "argmin_with_value",
            "argmax_with_value",
        ):
            flatten_index = _fixed_indexer(
                reduction_ranges,
                FlexibleLayout.contiguous_strides(reduction_ranges),
            )

            def value_fn(
                index: Sequence[_IntLike], rindex: Sequence[_IntLike]
            ) -> tuple[OpsValue, OpsValue]:
                # [FUNCTION DOC] HELPER: Compute (value, flat reduction index) for an arg-reduction, using the inner function's own pair if it returns a tuple.
                rindex = [sympy.expand(i) for i in rindex]
                value = inner_fn(index, rindex)
                if isinstance(value, tuple):
                    return value
                return (
                    value,
                    ops.index_expr(flatten_index(rindex), torch.int64),
                )

            if reduction_type in ("argmin_with_value", "argmax_with_value"):
                return fn
            if reduction_type in ("argmin_value", "argmax_value"):
                return lambda index: fn(index)[0]
            return lambda index: fn(index)[1]
        else:
            value_fn = inner_fn
            return fn

    @classmethod
    # pyrefly: ignore [bad-override]
    def create(
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        ranges: Sequence[_IntLike],
        reduction_ranges: Sequence[_IntLike],
        reduction_type: ReductionType,
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
        input_node: IRNode | None = None,
        *,
        strict_sum: bool = False,
    ) -> TensorBox:
        # [FUNCTION DOC] Class method building a reduction and returning a TensorBox. Handles special cases: zero reduction elements produce a pointwise constant (only sum, xor_sum, prod, any supported, otherwise AssertionError); with strict_sum it may pick an exact eager-order tiling (strict_sum_multirow/rblock, optional pre-split) when sizes are hinted and indexing fits in int32, padding the reduction range with a masked default; a reduction of one element becomes a pointwise op; small static reductions are unrolled into pointwise; otherwise num_splits (or the strict split) picks hint and split count (bumped to config.min_num_split for dynamic shapes). A split of -1 reuses the input node ranges (multilayer existing ranges), a split above 1 builds a multilayer reduction (and under mix-order reductions records original fn/ranges and split size on the split first-stage buffer). Else a single Reduction is created.
        """
        Create a reduction node. May split the reduction to multiple layers to expose
        more parallelism.
        """
        reduction_numel = V.graph.sizevars.simplify(sympy_product(reduction_ranges))

        if reduction_numel == 0:
            # N.B. This is a hack to generate the literal of the given type
            # Ideally, we should be fixing `def constant` in triton.py
            # but it breaks due to hardcoded dtypes in other places
            def py_cnst(val: object) -> bool | float | int:
                # [FUNCTION DOC] HELPER: Convert a Python number to a bool, float or int matching the destination dtype; raises AssertionError on a mismatching value type.
                if dst_dtype == torch.bool:
                    return bool(val)
                elif dst_dtype.is_floating_point:
                    if not isinstance(val, SupportsFloat):
                        raise AssertionError(type(val))
                    return float(val)
                else:
                    if not isinstance(val, SupportsInt):
                        raise AssertionError(type(val))
                    return int(val)

            rtypes_to_inits = {
                "sum": py_cnst(0),
                "xor_sum": py_cnst(0),
                "prod": py_cnst(1),
                "any": py_cnst(0),
                # "all" is desugared to `!any(!val)`
            }

            if reduction_type not in rtypes_to_inits:
                raise AssertionError(
                    f"{reduction_type} not supported for zero-dimension tensors!"
                )

            def const_fn(index: int) -> OpsValue:
                # [FUNCTION DOC] HELPER: Inner function returning the identity constant of the reduction for any index.
                return ops.constant(rtypes_to_inits[reduction_type], dst_dtype)

            return Pointwise.create(
                device=device,
                dtype=src_dtype,
                inner_fn=const_fn,
                ranges=list(ranges),
            )

        strict_split = None
        strict_sum_multirow = False
        strict_sum_rblock = None
        if strict_sum and reduction_hint in (
            ReductionHint.DEFAULT,
            ReductionHint.INNER,
        ):
            num_outputs = sympy_product(ranges)
            if V.graph.sizevars.all_unbacked_explicitly_hinted(
                [reduction_numel, num_outputs]
            ):
                guarded_reduction_numel = V.graph.sizevars.guard_int(reduction_numel)
                vector_size = inner_tree_plan.vec_size(src_dtype.itemsize)
                num_batches_ub = ceildiv(guarded_reduction_numel, 32 * vector_size)
                int32_max = 2**31 - 1
                indexing_safe = (
                    guarded_reduction_numel <= int32_max
                    and V.graph.sizevars.evaluate_expr(
                        sympy.And(
                            sympy.Gt(num_outputs, 0),
                            sympy.Le(num_outputs, int32_max),
                            sympy.Le(num_outputs * num_batches_ub, int32_max),
                        ),
                        fallback_value=False,
                    )
                )
                if indexing_safe:
                    strict_split = 1
                    strict_sum_multirow = (
                        guarded_reduction_numel
                        <= vector_size * inner_tree_plan._K_MULTIROW_MAX_LOADS
                    )
                    if strict_sum_multirow:
                        num_loads = ceildiv(guarded_reduction_numel, vector_size)
                        num_loads = 1 << (num_loads - 1).bit_length()
                        strict_sum_rblock = num_loads * vector_size
                    else:
                        params = inner_tree_plan.compute_inner_tree_params(
                            guarded_reduction_numel,
                            1,
                            vector_size,
                        )
                        strict_sum_rblock = params.batch_total_elements
                        if (
                            params.num_batches > inner_tree_plan._K_TWO_KERNEL_THRESHOLD
                            and guarded_reduction_numel % strict_sum_rblock == 0
                        ):
                            strict_split = params.num_batches
        strict_sum = strict_split is not None

        if strict_sum_multirow:
            if strict_sum_rblock is None:
                raise AssertionError("strict multirow sum requires a reduction block")
            actual_reduction_numel = reduction_numel
            original_inner_fn = inner_fn
            default = cls.default_value(reduction_type, dst_dtype)

            def padded_inner_fn(
                index: Sequence[Symbol], reduction_index: Sequence[Symbol]
            ) -> OpsValue:
                # [FUNCTION DOC] HELPER: Wrap the original inner function so reduction indices beyond the real reduction size read the default value (masked), for padded strict multirow sums.
                (rindex,) = reduction_index

                index_dtype = dtype_from_size(actual_reduction_numel)
                mask = ops.lt(
                    ops.index_expr(rindex, index_dtype),
                    ops.index_expr(actual_reduction_numel, index_dtype),
                )
                body = functools.partial(original_inner_fn, index, reduction_index)
                return ops.masked(mask, body, default)

            inner_fn = cast(Callable[..., Any], padded_inner_fn)
            reduction_ranges = [sympy.Integer(strict_sum_rblock)]
            reduction_numel = sympy.Integer(strict_sum_rblock)

        if reduction_numel == 1 and not strict_sum:
            # this reduction is actually a pointwise op
            if reduction_type in ("argmin", "argmax"):

                def fn(index: int) -> OpsValue:
                    # [FUNCTION DOC] HELPER: Pointwise replacement for arg reductions over one element; always the constant zero index.
                    return ops.constant(0, dst_dtype)

            else:

                def fn(index: int) -> OpsValue:
                    # [FUNCTION DOC] HELPER: Pointwise replacement for a single-element reduction; calls the inner function with zero reduction indices.
                    reduction_index = [sympy.S.Zero for _ in reduction_ranges]
                    return inner_fn(index, reduction_index)

            return Pointwise.create(
                device=device, dtype=dst_dtype, inner_fn=fn, ranges=ranges
            )

        if (
            isinstance(reduction_numel, Integer)
            and int(reduction_numel) < config.unroll_reductions_threshold
            and (sympy_product(ranges) != 1 or is_gpu(device.type))
            and reduction_type != "dot"
            and not strict_sum
        ):
            # When native matmul, don't unroll the dot reduction.

            # NB: This works around https://github.com/pytorch/pytorch/issues/140457
            # since turning reductions into pointwise ops can exacerbate this problem
            return Pointwise.create(
                device=device,
                dtype=dst_dtype,
                inner_fn=cls._unroll_reduction_fn(
                    inner_fn, reduction_ranges, reduction_type, src_dtype
                ),
                ranges=ranges,
            )

        # triton doesn't support reduce to single element well, so break it up
        if strict_split is None:
            hint, split = cls.num_splits(
                device,
                dst_dtype,
                src_dtype,
                inner_fn,
                ranges,
                reduction_ranges,
                reduction_type,
                reduction_numel,
                input_node,
            )
        else:
            hint, split = ReductionHint.INNER, strict_split

        def _maybe_increase_split(split: int) -> int:
            # [FUNCTION DOC] HELPER: Raise a split count above one to at least config.min_num_split for dynamic reduction sizes; static sizes are untouched.
            # don't apply min_num_split constraint for static shape case.
            if _is_static(reduction_numel):
                return split
            if split > 1:
                return max(split, config.min_num_split)
            else:
                return split

        if strict_split is None:
            split = _maybe_increase_split(split)

        # intermediate reduction in split can contain complex indexing,
        # and num_splits will fail to correctly set the hint
        # reuse the passed hint if available
        if reduction_hint == ReductionHint.DEFAULT:
            reduction_hint = hint
        if split == -1:
            if input_node is None:
                raise AssertionError("Expected input_node is not None")
            with patch.object(FlexibleLayout, "allow_indexing", True):
                new_ranges, new_reduction_ranges = extract_input_node_reduction_ranges(
                    input_node
                )
            if new_ranges is None:
                raise AssertionError("Expected new_ranges is not None")
            if new_reduction_ranges is None:
                raise AssertionError("Expected new_reduction_ranges is not None")
            return cls.create_multilayer_existing_ranges(
                device,
                dst_dtype,
                src_dtype,
                inner_fn,
                ranges,
                reduction_ranges,
                new_ranges,
                new_reduction_ranges,
                reduction_type,
                reduction_hint,
            )
        elif split > 1:
            # triton doesn't support reduce to single element well, so break it up
            out = cls.create_multilayer(
                device,
                dst_dtype,
                src_dtype,
                inner_fn,
                ranges,
                reduction_ranges,
                reduction_type,
                split,
                reduction_hint,
                input_node,
                strict_sum=strict_sum,
            )

            # Find the reduction that get split
            split_reduction = None
            if config.triton.mix_order_reduction and isinstance(out, TensorBox):

                def _find_split_reduction(
                    cur_node: TensorBox,
                ) -> ComputedBuffer | None:
                    # [FUNCTION DOC] HELPER: Find the ComputedBuffer holding the first-stage split reduction read by the given node; returns None if the node does not read exactly one computed buffer. Raises AssertionError if that buffer is not a reduction.
                    read_names = cur_node.get_read_names()
                    if len(read_names) != 1:
                        return None

                    bufname = next(iter(read_names))
                    if bufname not in V.graph.name_to_buffer:
                        return None
                    buf = V.graph.name_to_buffer[bufname]
                    if not isinstance(buf, ComputedBuffer):
                        return None

                    if buf.data.get_reduction_type() is None:
                        raise AssertionError(
                            "Expected buf.data.get_reduction_type() is not None"
                        )

                    return buf

                split_reduction = _find_split_reduction(out)

            if split_reduction:
                # If a reduction is split to more than 2 layers,
                # say there are 3 layers,
                # we always have the correct setting for layer1 (top layer).
                # The setting on layer2 may be incorrect but it's fine
                # since they are never get used.
                # TODO: should we skip setting these fields for layer2
                if not isinstance(split_reduction.data, Reduction):
                    raise AssertionError(f"{type(split_reduction.data)}")
                split_reduction._split_size = split_reduction.data.reduction_ranges[0]
                split_reduction._original_inner_fn = inner_fn
                split_reduction._original_ranges = ranges
                split_reduction._original_reduction_ranges = reduction_ranges
            return out

        out = TensorBox.create(
            Reduction(
                device=device,
                dtype=dst_dtype,
                inner_fn=inner_fn,
                ranges=ranges,
                reduction_ranges=reduction_ranges,
                reduction_type=reduction_type,
                src_dtype=src_dtype,
                reduction_hint=reduction_hint,
                strict_sum_multirow=strict_sum_multirow,
                strict_sum_rblock=strict_sum_rblock,
            )
        )
        return out

    @staticmethod
    def default_accumulator(
        reduction_type: str, dtype: torch.dtype
    ) -> _NumLike | Sequence[_NumLike]:
        # [FUNCTION DOC] Static method returning the initial accumulator for a reduction type and dtype: lowest value for max-like (minus infinity, False, or integer min), highest for min-like, zero/one for sum-like and prod, tuples for welford and online softmax. Raises KeyError for unknown types.
        if reduction_type in (
            "max",
            "fmax",
            "argmax",
            "argmax_value",
            "argmax_with_value",
        ):
            if is_float_dtype(dtype):
                return float("-inf")
            elif is_boolean_dtype(dtype):
                return False
            else:
                return torch.iinfo(dtype).min
        if reduction_type in ("min", "argmin", "argmin_value", "argmin_with_value"):
            if is_float_dtype(dtype):
                return float("inf")
            elif is_boolean_dtype(dtype):
                return True
            else:
                return torch.iinfo(dtype).max

        zero = False if is_boolean_dtype(dtype) else 0
        one = True if is_boolean_dtype(dtype) else 1
        return {
            "sum": zero,
            "prod": one,
            "dot": zero,
            "xor_sum": zero,
            "any": zero,
            "welford_reduce": (zero, zero, zero),
            "welford_combine": (zero, zero, zero),
            "online_softmax_reduce": (float("-inf"), zero),
        }[reduction_type]

    @staticmethod
    def default_value(
        reduction_type: str, dtype: torch.dtype
    ) -> _NumLike | Sequence[_NumLike]:
        # [FUNCTION DOC] Static method returning the default fill value for a reduction: zero for welford_reduce, otherwise the default accumulator.
        if reduction_type == "welford_reduce":
            return 0
        return Reduction.default_accumulator(reduction_type, dtype)

    @staticmethod
    def _multilayer_second_step_hint(
        split: _IntLike, numel_hint: int, reduction_hint: ReductionHint
    ) -> ReductionHint:
        # [FUNCTION DOC] Static method choosing the hint for the second stage of a multilayer reduction: OUTER becomes OUTER_TINY when split and element counts are small; otherwise the hint is unchanged (also unchanged for split -1).
        if split == -1:
            return reduction_hint
        if split <= 512 and numel_hint <= 512 and reduction_hint == ReductionHint.OUTER:
            return ReductionHint.OUTER_TINY
        if (
            split <= 1024
            and numel_hint <= 256
            and reduction_hint == ReductionHint.OUTER
        ):
            return ReductionHint.OUTER_TINY

        return reduction_hint

    @classmethod
    def check_for_split_dense_dim_reindexing(
        cls, reduction_numel: _IntLike, input_node: IRNode | None
    ) -> int | None:
        # [FUNCTION DOC] Class method: when reducing over the whole input and its layout has a unit stride in a non-last dimension, return that dimension index so the reduction can be reindexed over the dense dim; otherwise None. Side effect: realizes the input node and finalizes its layout; returns None when the layout is unavailable.
        """
        If we are reducing over the full tensor, and it is non-dense in the last dimension,
        reindex so we reduce over the dense dimension. initially just handle complete
        reduction case
        """
        if input_node is None:
            return None

        if not V.graph.sizevars.statically_known_equals(
            input_node.get_numel(), reduction_numel
        ):
            return None

        input_node.realize()
        try:
            # finalize layout
            as_storage_and_layout(input_node)
        except NotImplementedError:
            return None

        strides = input_node.get_stride()

        for i, s in enumerate(strides[:-1]):
            if V.graph.sizevars.statically_known_equals(s, 1):
                return i

        return None

    @classmethod
    def _multilayer_wrap_loader(
        cls,
        loader: Callable[..., OpsValue],
        reduction_ranges: Sequence[_IntLike],
        reduction_numel: _IntLike,
        split: _IntLike,
        block_size: _IntLike,
        default: _NumLike | Sequence[_NumLike],
        input_node: IRNode | None = None,
    ) -> Callable[..., object]:
        # [FUNCTION DOC] HELPER: Build the loader for the first stage of a split reduction. It reshapes the flat split index into the original reduction ranges (using the dense dim if any), computes the global index from block and in-block position, and masks out-of-range elements with the default when the reduction size is not statically divisible by split.
        dense_index = cls.check_for_split_dense_dim_reindexing(
            reduction_numel, input_node
        )
        reindex = View.dynamic_reshape_indexer(
            reduction_ranges, [reduction_numel], dense_index
        )
        need_mask = not V.graph.sizevars.statically_known_true(
            sympy.Eq(Mod(reduction_numel, split), 0)
        )

        def wrapper_fn(
            index: Sequence[Symbol], reduction_index: Sequence[Symbol]
        ) -> OpsValue:
            # [FUNCTION DOC] HELPER: Load one element for (output index, split block) and in-block reduction index, masked to the default when out of range.
            (reduction_index,) = reduction_index
            *new_index, reduction_block = index
            indices = block_size * reduction_block + reduction_index

            def body() -> OpsValue:
                # [FUNCTION DOC] HELPER: Load the original value at the reindexed flat position.
                return loader(new_index, reindex([indices]))

            if need_mask:
                index_dtype = dtype_from_size(reduction_numel)
                mask = ops.lt(
                    ops.index_expr(indices, index_dtype),
                    ops.index_expr(reduction_numel, index_dtype),
                )
                return ops.masked(mask, body, default)
            else:
                return body()

        return wrapper_fn

    @classmethod
    def _multilayer_wrap_loader_existing_ranges(
        cls,
        loader: Callable[[Sequence[Expr], Sequence[Expr]], OpsValue],
        original_ranges: Sequence[Expr],
        original_reduction_ranges: Sequence[Expr],
        new_ranges: Sequence[Integer],
        new_reduction_ranges: Sequence[Integer],
    ) -> Callable[[Sequence[Expr], Sequence[Expr]], OpsValue]:
        # [FUNCTION DOC] HELPER: Wrap a loader so a reduction over previously computed ranges is expressed over new ranges. Requires every original range to be one (AssertionError otherwise); reshapes the combined new index back to the original reduction index.
        if not all(r == 1 for r in original_ranges):
            raise AssertionError(
                f"Only enabled for numel_hint == 1, found {original_ranges=}"
            )
        reindex = View.dynamic_reshape_indexer(
            original_reduction_ranges, tuple(new_ranges) + tuple(new_reduction_ranges)
        )

        def wrapper_fn(
            merged_index: Sequence[Expr],
            new_reduction_index: Sequence[Expr],
        ) -> OpsValue:
            # [FUNCTION DOC] HELPER: Split the merged index into original and new indices and call the loader at the reshaped original reduction index.
            original_idx = merged_index[: len(original_ranges)]
            new_index = merged_index[len(original_ranges) :]
            return loader(
                original_idx,
                reindex(tuple(new_index) + tuple(new_reduction_index)),
            )

        return wrapper_fn

    @classmethod
    def create_multilayer_helper(
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        wrapper_fn: Callable[..., Any],
        original_ranges: Sequence[Expr],
        original_reduction_ranges: Sequence[Expr],
        new_ranges: list[Expr],
        new_reduction_ranges: list[Integer],
        reduction_type: ReductionType,
        split: _IntLike,
        reduction_hint: ReductionHint,
        strict_sum: bool = False,
    ) -> TensorBox:
        # [FUNCTION DOC] Class method building a two-stage reduction: the first stage is a reduction over new_reduction_ranges with new_ranges (kept in fp32 for half precision types, then realized), the second stage reduces its output over the extra dimensions to the original ranges with an adjusted hint. Raises AssertionError if original_ranges is not a prefix of new_ranges. Returns a TensorBox.
        """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
        # triton will automatically compute reductions in fp32 if reducing over fp16/bf16
        # within the kernel. keep the intermediate in fp32 so as to keep the whole reduction
        # in fp32 and not reduce precision by breaking up the kernel into multiple layers
        intermediate_dtype = (
            dst_dtype
            if dst_dtype not in (torch.float16, torch.bfloat16)
            else torch.float
        )
        intermediate = Reduction.create(
            device,
            intermediate_dtype,
            src_dtype,
            wrapper_fn,
            new_ranges,
            new_reduction_ranges,
            reduction_type,
            reduction_hint,
            strict_sum=strict_sum,
        )
        intermediate.realize()
        intermediate_loader = intermediate.make_loader()

        def intermediate_fn(
            index: Sequence[_IntLike], reduction_index: Sequence[_IntLike]
        ) -> OpsValue:
            # [FUNCTION DOC] HELPER: Load the first-stage result at the concatenated output and reduction index.
            return intermediate_loader([*index, *reduction_index])

        numel_hint = V.graph.sizevars.optimization_hint(sympy_product(original_ranges))
        reduction_hint = cls._multilayer_second_step_hint(
            split, numel_hint, reduction_hint
        )

        if original_ranges != new_ranges[: len(original_ranges)]:
            raise AssertionError(
                "Expected original_ranges == new_ranges[: len(original_ranges)]"
            )
        return TensorBox.create(
            Reduction(
                device=device,
                dtype=dst_dtype,
                inner_fn=intermediate_fn,
                ranges=original_ranges,
                reduction_ranges=new_ranges[len(original_ranges) :],
                reduction_type=reduction_type,
                src_dtype=src_dtype,
                reduction_hint=reduction_hint,
                strict_sum_rblock=1 if strict_sum else None,
            )
        )

    @classmethod
    def create_multilayer(
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        ranges: Sequence[_IntLike],
        reduction_ranges: Sequence[_IntLike],
        reduction_type: ReductionType,
        split: _IntLike,
        reduction_hint: ReductionHint,
        input_node: IRNode | None = None,
        *,
        strict_sum: bool = False,
    ) -> TensorBox:
        # [FUNCTION DOC] Class method splitting one reduction into split partial reductions: computes the block size, wraps the inner function with a split-aware loader and delegates to create_multilayer_helper with the extra split dimension. Returns a TensorBox.
        """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
        # TODO(jansel): realize the reduction so we can do dynamic indexing
        reduction_numel = sympy_product(reduction_ranges)
        block_size = FloorDiv(reduction_numel + (split - 1), split)
        default = cls.default_value(reduction_type, dst_dtype)
        wrapper_fn = cls._multilayer_wrap_loader(
            inner_fn,
            reduction_ranges,
            reduction_numel,
            split,
            block_size,
            default,
            input_node,
        )

        return cls.create_multilayer_helper(
            device,
            dst_dtype,
            src_dtype,
            wrapper_fn,
            ranges,
            reduction_ranges,
            [*ranges, split],
            [block_size],
            reduction_type,
            split,
            reduction_hint,
            strict_sum,
        )

    @classmethod
    def create_multilayer_existing_ranges(
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        original_ranges: Sequence[_IntLike],
        original_reduction_ranges: Sequence[_IntLike],
        new_ranges: list[Integer],
        new_reduction_ranges: list[Integer],
        reduction_type: ReductionType,
        reduction_hint: ReductionHint,
    ) -> TensorBox:
        # [FUNCTION DOC] Class method building a multilayer reduction that reuses the ranges of the producer reduction (new_ranges, new_reduction_ranges) instead of an arbitrary split; delegates to create_multilayer_helper with split -1. Returns a TensorBox.
        """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
        wrapper_fn = cls._multilayer_wrap_loader_existing_ranges(
            inner_fn,
            original_ranges,
            original_reduction_ranges,
            new_ranges,
            new_reduction_ranges,
        )
        return cls.create_multilayer_helper(
            device,
            dst_dtype,
            src_dtype,
            wrapper_fn,
            original_ranges,
            original_reduction_ranges,
            [*original_ranges, *new_ranges],
            new_reduction_ranges,
            reduction_type,
            -1,
            reduction_hint,
        )


def _fixed_indexer(
    size: Sequence[_IntLike],
    stride: Sequence[_IntLike] | None = None,
    offset: Expr = Integer(0),
) -> Callable[[Sequence[_IntLike]], Expr]:
    # [FUNCTION DOC] HELPER: Return a function mapping an index to the flat offset expression offset plus the sum of index times stride over dimensions whose size is not one. Stride must be given.
    """A closure containing math to read a given element"""

    def indexer(index: Sequence[_IntLike]) -> Expr:
        # [FUNCTION DOC] HELPER: Compute the flat offset for an index; raises AssertionError if stride is missing or lengths differ from size.
        if not (stride is not None and len(index) == len(stride)):
            raise AssertionError(
                "Expected stride is not None and len(index) == len(stride)"
            )
        if len(index) != len(size):
            raise AssertionError("Expected len(index) == len(size)")
        result = offset
        for idx, st, sz in zip(index, stride, size):
            if sz != 1:
                result = result + idx * st
        return result

    return indexer


INNER_FN_TY: TypeAlias = Callable[[Sequence[Expr], Sequence[Expr]], OpsValue]


class MultiOutputReduction(Reduction):
    # [CLASS DOC] Reduction whose inner function may produce several values (a tuple) and which exposes only one of them, selected by output_index, as its result. Used by arg-with-value, welford and online softmax reductions. Several instances over the same inputs are created, one per output.
    output_index: int

    def __init__(
        self,
        device: torch.device,
        dst_dtype: torch.dtype,
        inner_fns: INNER_FN_TY | Sequence[INNER_FN_TY],
        ranges: Sequence[Integer],
        reduction_ranges: Sequence[Integer],
        reduction_type: ReductionType,
        src_dtype: torch.dtype,
        reduction_hint: ReductionHint,
        output_index: int,
    ):
        # [FUNCTION DOC] Build the reduction from one inner function or a sequence of them; with several, the loader returns a tuple of their results. Calls the Reduction initializer (with default strict-sum settings) and records output_index.
        if callable(inner_fns):
            inner_fns = (inner_fns,)

        loader: Callable[[Sequence[Expr], Sequence[Expr]], Any]
        if len(inner_fns) == 1:
            loader = inner_fns[0]
        else:

            def loader(
                idx: Sequence[Expr], reduction_idx: Sequence[Expr]
            ) -> tuple[OpsValue, ...]:
                # [FUNCTION DOC] HELPER: Evaluate every inner function at the same indices and return the results as a tuple.
                return tuple(fn(idx, reduction_idx) for fn in inner_fns)

        super().__init__(
            device=device,
            dtype=dst_dtype,
            inner_fn=loader,
            ranges=ranges,
            reduction_ranges=reduction_ranges,
            reduction_type=reduction_type,
            src_dtype=src_dtype,
            reduction_hint=reduction_hint,
        )
        self.output_index = output_index

    def store_reduction(
        self,
        output_name: str | None,
        indexer: Callable[[Sequence[Expr]], Never],
        vars: Sequence[Expr],
        reduction_vars: Sequence[Symbol],
    ) -> Any:
        # [FUNCTION DOC] Emit the reduction op, require a tuple/list of values (AssertionError otherwise), pick the output_index-th value and emit a store of it (named unnamed if None).
        values = ops.reduction(
            self.dtype,
            self.src_dtype,
            self.reduction_type,
            self.inner_fn(vars, reduction_vars),
        )
        if not isinstance(values, (tuple, list)):
            raise AssertionError(type(values))
        value = values[self.output_index]
        return ops.store_reduction(output_name or "unnamed", indexer(vars), value)


class ArgReduction(MultiOutputReduction):
    # [CLASS DOC] MultiOutputReduction for argmax_with_value / argmin_with_value: produces two outputs, the selected value (dst dtype) and its int64 index. The create method returns both as TensorBoxes.
    """Multi-output arg reduction that returns both the selected value and index."""

    @classmethod
    def create(  # type: ignore[override]
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        ranges: Sequence[Expr],
        reduction_ranges: Sequence[Expr],
        reduction_type: ReductionType,
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
        input_node: IRNode | None = None,
    ) -> Sequence[TensorBox]:
        # [FUNCTION DOC] Class method creating the value and index outputs. Only the two with_value reduction types are accepted and zero reduction size raises AssertionError. A one-element reduction becomes two pointwise nodes (the value and a zero index); small static reductions are unrolled into two pointwise nodes; otherwise a hint is chosen via num_splits (a split other than one raises AssertionError) and two realized reduction outputs are returned.
        if reduction_type not in ("argmax_with_value", "argmin_with_value"):
            raise AssertionError(f"unexpected reduction_type {reduction_type!r}")
        reduction_numel = V.graph.sizevars.simplify(sympy_product(reduction_ranges))
        if reduction_numel == 0:
            raise AssertionError(
                f"{reduction_type} not supported for zero-dimension tensors!"
            )

        if reduction_numel == 1:

            def value_fn(index: Sequence[Expr]) -> OpsValue:
                # [FUNCTION DOC] HELPER: Return the value part of the inner function evaluated at zero reduction index.
                reduction_index = [sympy.S.Zero for _ in reduction_ranges]
                value = inner_fn(index, reduction_index)
                if isinstance(value, tuple):
                    return value[0]
                return value

            def index_fn(index: Sequence[Expr]) -> OpsValue:
                # [FUNCTION DOC] HELPER: Return the constant int64 zero index.
                return ops.constant(0, torch.int64)

            return (
                Pointwise.create(
                    device=device,
                    dtype=dst_dtype,
                    inner_fn=value_fn,
                    ranges=list(ranges),
                ),
                Pointwise.create(
                    device=device,
                    dtype=torch.int64,
                    inner_fn=index_fn,
                    ranges=list(ranges),
                ),
            )

        if (
            isinstance(reduction_numel, Integer)
            and int(reduction_numel) < config.unroll_reductions_threshold
            and (sympy_product(ranges) != 1 or is_gpu(device.type))
        ):
            unrolled_fn = Reduction._unroll_reduction_fn(
                inner_fn, reduction_ranges, reduction_type, src_dtype
            )

            def project(index: Sequence[Expr], output_index: int) -> OpsValue:
                # [FUNCTION DOC] HELPER: Evaluate the unrolled function and return the requested element of the (value, index) result; AssertionError if not a tuple.
                result = unrolled_fn(index)
                if not isinstance(result, tuple):
                    raise AssertionError(f"expected tuple result, got {type(result)}")
                return result[output_index]

            return tuple(
                Pointwise.create(
                    device=device,
                    dtype=dtype,
                    inner_fn=partial(project, output_index=output_index),
                    ranges=list(ranges),
                )
                for output_index, dtype in enumerate((dst_dtype, torch.int64))
            )

        hint, split = Reduction.num_splits(
            device,
            dst_dtype,
            src_dtype,
            inner_fn,
            ranges,
            reduction_ranges,
            reduction_type,
            reduction_numel,
            input_node,
        )
        if split != 1:
            raise AssertionError("arg reductions do not support split reductions")
        if reduction_hint == ReductionHint.DEFAULT:
            reduction_hint = hint

        results = tuple(
            TensorBox.create(
                cls(
                    device,
                    dtype,
                    inner_fn,
                    ranges,
                    reduction_ranges,
                    reduction_type,
                    src_dtype,
                    reduction_hint,
                    output_index,
                )
            )
            for output_index, dtype in enumerate((dst_dtype, torch.int64))
        )
        for result in results:
            result.realize()
        return results


class OnlineSoftmaxReduction(MultiOutputReduction):
    # [CLASS DOC] MultiOutputReduction computing the running maximum and sum-of-exponentials for softmax in one pass (reduction type online_softmax_reduce). Supports splitting the reduction into two stages via create_multilayer.
    """Multi-output reduction that computes softmax max/sum in one pass."""

    @classmethod
    def _create_no_split(
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        ranges: Sequence[Expr],
        reduction_ranges: Sequence[Expr],
        num_output: int,
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
        input_node: IRNode | None = None,
    ) -> Sequence[TensorBox]:
        # [FUNCTION DOC] Class method creating num_output realized MultiOutputReduction nodes (one per output) without any splitting; returns them as a tuple of TensorBoxes.
        results = tuple(
            TensorBox.create(
                MultiOutputReduction(
                    device,
                    dst_dtype,
                    inner_fn,
                    ranges,
                    reduction_ranges,
                    "online_softmax_reduce",
                    src_dtype,
                    reduction_hint,
                    output_idx,
                )
            )
            for output_idx in range(num_output)
        )
        for t in results:
            t.realize()
        return results

    @classmethod
    def create(  # type: ignore[override]
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        ranges: Sequence[Expr],
        reduction_ranges: Sequence[Expr],
        num_output: int,
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
        input_node: IRNode | None = None,
    ) -> Sequence[TensorBox]:
        # [FUNCTION DOC] Class method choosing a hint and split via Reduction.num_splits; uses a two-stage reduction if the split is above one, otherwise the unsplit version. Returns the output TensorBoxes.
        reduction_numel = V.graph.sizevars.simplify(sympy_product(reduction_ranges))
        hint, split = Reduction.num_splits(
            device,
            dst_dtype,
            src_dtype,
            inner_fn,
            ranges,
            reduction_ranges,
            reduction_type="online_softmax_reduce",
            reduction_numel=reduction_numel,
            input_node=input_node,
        )
        if reduction_hint == ReductionHint.DEFAULT:
            reduction_hint = hint
        if split > 1:
            return cls.create_multilayer(
                device,
                dst_dtype,
                src_dtype,
                inner_fn,
                ranges,
                reduction_ranges,
                num_output,
                split,
                reduction_hint,
                input_node,
            )
        return cls._create_no_split(
            device,
            dst_dtype,
            src_dtype,
            inner_fn,
            ranges,
            reduction_ranges,
            num_output,
            reduction_hint,
            input_node,
        )

    @classmethod
    def create_multilayer(
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        ranges: Sequence[Expr],
        reduction_ranges: Sequence[Expr],
        num_output: int,
        split: _IntLike,
        reduction_hint: ReductionHint,
        input_node: IRNode | None = None,
    ) -> Sequence[TensorBox]:
        # [FUNCTION DOC] Class method implementing the two-stage online softmax reduction: stage one reduces each block of the reduction range (masked elements contribute negative infinity and weight zero), keeping half precision in fp32 and realizing the partial results; stage two reduces the partials over the split dimension. Returns the outputs of the second stage.
        reduction_numel = sympy_product(reduction_ranges)
        block_size = FloorDiv(reduction_numel + (split - 1), split)
        dense_index = cls.check_for_split_dense_dim_reindexing(
            reduction_numel, input_node
        )
        reindex = View.dynamic_reshape_indexer(
            reduction_ranges, [reduction_numel], dense_index
        )
        need_mask = not V.graph.sizevars.statically_known_true(
            sympy.Eq(Mod(reduction_numel, split), 0)
        )

        def wrapper_fn(
            index: Sequence[Symbol], reduction_index: Sequence[Symbol]
        ) -> tuple[OpsValue, OpsValue]:
            # [FUNCTION DOC] HELPER: First-stage inner function; loads the element for the split block and in-block index and returns (value, one), with masked positions giving (negative infinity, zero) when masking is needed.
            (reduction_index,) = reduction_index
            *new_index, reduction_block = index
            indices = block_size * reduction_block + reduction_index

            def body() -> OpsValue:
                # [FUNCTION DOC] HELPER: Load the original value at the reindexed flat position.
                return inner_fn(new_index, reindex([indices]))

            one = ops.constant(1, src_dtype)
            if need_mask:
                index_dtype = dtype_from_size(reduction_numel)
                mask = ops.lt(
                    ops.index_expr(indices, index_dtype),
                    ops.index_expr(reduction_numel, index_dtype),
                )
                return (
                    ops.masked(mask, body, float("-inf")),
                    ops.where(mask, one, ops.constant(0, src_dtype)),
                )
            else:
                return body(), one

        intermediate_dtype = (
            dst_dtype
            if dst_dtype not in (torch.float16, torch.bfloat16)
            else torch.float
        )
        intermediates = cls._create_no_split(
            device,
            intermediate_dtype,
            src_dtype,
            wrapper_fn,
            [*ranges, split],
            [block_size],
            num_output,
            reduction_hint,
        )
        for i in intermediates:
            i.realize()

        intermediate_loaders = tuple(i.make_loader() for i in intermediates)

        def intermediate_fn(
            index: Sequence[_IntLike], reduction_index: Sequence[_IntLike]
        ) -> tuple[OpsValue, ...]:
            # [FUNCTION DOC] HELPER: Load each partial result at the concatenated index and return them as a tuple.
            return tuple(
                loader([*index, *reduction_index]) for loader in intermediate_loaders
            )

        numel_hint = V.graph.sizevars.optimization_hint(sympy_product(ranges))
        reduction_hint = cls._multilayer_second_step_hint(
            split, numel_hint, reduction_hint
        )

        return cls._create_no_split(
            device,
            dst_dtype,
            intermediate_dtype,
            intermediate_fn,
            ranges,
            [split],
            num_output,
            reduction_hint,
        )


class WelfordReduction(MultiOutputReduction):
    # [CLASS DOC] MultiOutputReduction producing the three Welford outputs (mean, m2, weight). Reduction types are welford_reduce (from raw values) and welford_combine (from partial triples). Handles empty, one-element and split reductions.
    @classmethod
    def create(  # type: ignore[override]
        cls,
        device: torch.device,
        dtype: torch.dtype,
        inner_fns: Sequence[Callable[..., Any]],
        ranges: list[Integer],
        reduction_ranges: list[Integer],
        reduction_type: ReductionType,
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
    ) -> Sequence[TensorBox]:
        # [FUNCTION DOC] Class method creating the three Welford outputs. Raises AssertionError for other reduction types or when config.mtia.disable_welford_reduction is set. Empty reductions give constant zeros; one-element reductions copy the input (weight one for welford_reduce); otherwise num_splits decides between a two-stage version and three realized reduction outputs. Returns the outputs.
        if reduction_type not in ("welford_reduce", "welford_combine"):
            raise AssertionError(
                'Expected reduction_type in ("welford_reduce", "welford_combine")'
            )
        if config.mtia.disable_welford_reduction:
            raise AssertionError(
                "welford reduction usage is explicitly disabled, please check you config"
            )

        reduction_numel = V.graph.sizevars.simplify(sympy_product(reduction_ranges))

        def const(val: int) -> TensorBox:
            # [FUNCTION DOC] HELPER: Make a pointwise node filled with the constant value in the given dtype over the output ranges.
            def inner_fn(idx: Sequence[Expr]) -> OpsValue:
                # [FUNCTION DOC] HELPER: Return the constant for any index.
                return ops.constant(
                    val,
                    dtype,
                )

            return Pointwise.create(
                device=device,
                dtype=dtype,
                inner_fn=inner_fn,
                ranges=list(ranges),
            )

        if reduction_numel == 0:
            mean = const(0)
            m2 = const(0)
            weight = const(0)
            return mean, m2, weight

        if reduction_numel == 1:

            def copy(
                loader: Callable[[Sequence[Expr], Sequence[Expr]], OpsValue],
            ) -> TensorBox:
                # [FUNCTION DOC] HELPER: Make a pointwise node that evaluates the loader at zero reduction index.
                def inner_fn(idx: Sequence[Expr]) -> OpsValue:
                    # [FUNCTION DOC] HELPER: Call the loader at the index and a zero reduction index.
                    reduction_index = [sympy.S.Zero for _ in reduction_ranges]
                    return loader(idx, reduction_index)

                return Pointwise.create(
                    device=device,
                    dtype=dtype,
                    inner_fn=inner_fn,
                    ranges=list(ranges),
                )

            if reduction_type == "welford_reduce":
                return copy(inner_fns[0]), const(0), const(1)
            else:
                return tuple(copy(fn) for fn in inner_fns)

        # TODO: Unrolled reduction
        # if (
        #     isinstance(reduction_numel, Integer)
        #     and int(reduction_numel)
        #     < config.unroll_reductions_threshold
        #     and sympy_product(ranges) != 1
        # ):
        #     return Pointwise.create(
        #         device,
        #         dst_dtype,
        #         cls._unroll_reduction_fn(
        #             inner_fn, reduction_ranges, reduction_type, src_dtype,
        #         ),
        #         ranges,
        #     )

        # triton doesn't support reduce to single element well, so break it up
        hint, split = Reduction.num_splits(
            device,
            dtype,
            dtype,
            inner_fns[0],
            ranges,
            reduction_ranges,
            reduction_type=reduction_type,
            reduction_numel=reduction_numel,
        )
        # intermediate reduction in split can contain complex indexing,
        # and num_splits will fail to correctly set the hint
        # reuse the passed hint if available
        if reduction_hint == ReductionHint.DEFAULT:
            reduction_hint = hint
        if split > 1:
            # triton doesn't support reduce to single element well, so break it up
            return cls.create_multilayer(
                device,
                dtype,
                inner_fns,
                ranges,
                reduction_ranges,
                reduction_type,
                split,
                reduction_hint,
            )

        results = [
            TensorBox.create(
                WelfordReduction(
                    device,
                    dtype,
                    inner_fns,
                    ranges,
                    reduction_ranges,
                    reduction_type,
                    dtype,
                    reduction_hint,
                    output_idx,
                )
            )
            for output_idx in range(3)
        ]
        for t in results:
            t.realize()
        return results

    @staticmethod
    def default_value(
        reduction_type: str, dtype: torch.dtype
    ) -> _NumLike | Sequence[_NumLike]:
        # [FUNCTION DOC] Static method: default fill for Welford is the triple of zeros regardless of arguments.
        return (0, 0, 0)

    @classmethod
    def create_multilayer(  # type: ignore[override]
        cls,
        device: torch.device,
        dtype: torch.dtype,
        inner_fns: Sequence[Callable[..., Any]],
        ranges: list[Integer],
        reduction_ranges: list[Integer],
        reduction_type: ReductionType,
        split: _IntLike,
        reduction_hint: ReductionHint,
    ) -> Sequence[TensorBox]:
        # [FUNCTION DOC] Class method splitting a Welford reduction into two stages. If masking is needed for welford_reduce, it is rewritten as welford_combine with explicit mean, weight constants of zero and one so masked items carry no weight. Stage one reduces blocks with wrapped loaders (results realized); stage two combines the partial triples over the split dimension with welford_combine. Returns the three outputs.
        """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
        reduction_numel = sympy_product(reduction_ranges)
        need_mask = not V.graph.sizevars.statically_known_true(
            sympy.Eq(Mod(reduction_numel, split), 0)
        )

        if need_mask and reduction_type != "welford_combine":
            # If we need mask, then "welford_reduce" doesn't work because
            # masked inputs shouldn't count towards the welford weight

            def constant(
                idx: Sequence[Expr], reduction_idx: Sequence[Expr], value: int
            ) -> OpsValue:
                # [FUNCTION DOC] HELPER: Inner function returning the constant value regardless of index.
                return ops.constant(value, dtype)

            return cls.create_multilayer(
                device=device,
                dtype=dtype,
                inner_fns=(
                    inner_fns[0],
                    partial(constant, value=0),
                    partial(constant, value=1),
                ),
                ranges=ranges,
                reduction_ranges=reduction_ranges,
                reduction_type="welford_combine",
                split=split,
                reduction_hint=reduction_hint,
            )

        block_size = FloorDiv(reduction_numel + (split - 1), split)
        intermediates = WelfordReduction.create(
            device,
            dtype,
            tuple(
                cls._multilayer_wrap_loader(
                    loader,
                    reduction_ranges,
                    reduction_numel,
                    split,
                    block_size,
                    default=0,
                )
                for loader in inner_fns
            ),
            [*ranges, split],
            [block_size],
            reduction_type,
            reduction_hint,
        )
        for i in intermediates:
            i.realize()

        def intermediate_loader_fn(
            index: Sequence[Expr],
            reduction_index: Sequence[Expr],
            loader: Callable[[Sequence[Expr]], OpsValue],
        ) -> OpsValue:
            # [FUNCTION DOC] HELPER: Call a partial result loader with the concatenated output and reduction index.
            return loader([*index, *reduction_index])

        # numel_hint is only used to choose between ReductionHint.OUTER vs
        # OUTER_TINY, which is a performance heuristic, not a correctness decision.
        numel_hint = V.graph.sizevars.optimization_hint(sympy_product(ranges))
        reduction_hint = cls._multilayer_second_step_hint(
            split, numel_hint, reduction_hint
        )
        return WelfordReduction.create(
            device,
            dtype,
            tuple(
                partial(intermediate_loader_fn, loader=i.make_loader())
                for i in intermediates
            ),
            ranges,
            [split],
            # welford_reduce turns one input into three outputs, which are combined with welford_combine
            "welford_combine",
            reduction_hint,
        )


@ir_dataclass
class Scan(Loops):
    # [CLASS DOC] Loops node for a scan (prefix computation) along scan_ranges. Fields: scan_ranges, size (full size, ranges plus scan ranges), combine_fn, reindex (maps output and scan vars to the full index), reduction_hint, output_index and per-output dtypes and inner functions. Mimics a reduction so it can use reduction codegen paths (reduction type is custom).
    scan_ranges: list[Integer]
    size: list[Integer]
    combine_fn: Callable[[tuple[Any, ...], tuple[Any, ...]], tuple[Any, ...]]
    reindex: Callable[[Sequence[_IntLike], Sequence[_IntLike]], Sequence[_IntLike]]
    reduction_hint: ReductionHint
    output_index: int
    # output_index indexes the following tuples
    dtypes: tuple[torch.dtype, ...]
    inner_fns: tuple[Callable[..., Any], ...]

    # HACK we mimic reduction

    @cache_on_self_and_args("Scan")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        # [FUNCTION DOC] Return free symbols of the parent plus those in scan_ranges and size (cached).
        # TODO: Can combine_fn/reindex close over unbacked symbols? If so, we
        # need to explicitly represent the closure so we can pull out unbacked
        # symbols here
        return (
            super().get_free_symbol_uses(unbacked_only)
            | OrderedSet().union(
                *(get_free_symbols(e, unbacked_only) for e in self.scan_ranges)
            )
            | OrderedSet().union(
                *(get_free_symbols(e, unbacked_only) for e in self.size)
            )
        )

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Validate that ranges plus scan ranges match the size in length (AssertionError otherwise), then run the parent post-init.
        if len(self.ranges) + len(self.scan_ranges) != len(self.size):
            raise AssertionError(
                "Expected len(self.ranges) + len(self.scan_ranges) == len(self.size)"
            )
        super().__post_init__()

    def store_reduction(
        self,
        output_name: str | None,
        indexer: Callable[[Sequence[_IntLike]], Never],
        vars: Sequence[Expr],
        scan_vars: Sequence[Symbol],
    ) -> Any:
        # [FUNCTION DOC] Compute the full index with reindex, evaluate all inner functions, emit a scan op with the combine function, and store the output_index-th result at the indexed position (named unnamed if None).
        idx = self.reindex(vars, scan_vars)
        values = tuple(inner_fn(idx) for inner_fn in self.inner_fns)
        result = ops.scan(self.dtypes, self.combine_fn, values)
        return ops.store(
            output_name or "unnamed", indexer(idx), result[self.output_index]
        )

    def get_reduction_type(self) -> str | None:
        # [FUNCTION DOC] Return the string custom as the reduction type.
        # return self.scan_op
        return "custom"

    def get_reduction_size(self) -> Sequence[Expr]:
        # [FUNCTION DOC] Return the scan ranges as the reduction size.
        return self.scan_ranges

    def get_size(self) -> Sequence[Expr]:
        # [FUNCTION DOC] Return the full size.
        return self.size

    def get_pointwise_size(self) -> Sequence[Expr]:
        # [FUNCTION DOC] Return the non-scan (pointwise) ranges.
        return self.ranges

    def index_length(self) -> int:
        # [FUNCTION DOC] Return the total index length: pointwise ranges plus scan ranges.
        return len(self.ranges) + len(self.scan_ranges)

    def inner_fn_args(self) -> Sequence[Sequence[_IntLike]]:
        # [FUNCTION DOC] Return the inner function arguments: the full index built by reindexing output index symbols and scan index symbols (r0 prefix).
        index = self._index(self.ranges)
        rindex = self._index(self.scan_ranges, SymT.R0_INDEX)
        idx = self.reindex(index, rindex)
        return (idx,)

    def inner_fn_free_symbols(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        # [FUNCTION DOC] Return free symbols of the inner function evaluated on the reindexed full index.
        index = self._index(self.ranges)
        rindex = self._index(self.scan_ranges, SymT.R0_INDEX)
        idx = self.reindex(index, rindex)
        return extract_free_symbols(self.inner_fn, idx, unbacked_only=unbacked_only)

    @classmethod
    def create(  # type: ignore[override]
        cls,
        device: torch.device,
        dtypes: tuple[torch.dtype, ...],
        inner_fns: tuple[Callable[[Sequence[Expr]], Any], ...],
        size: list[Integer],
        axis: int,
        combine_fn: Callable[[tuple[Any, ...], tuple[Any, ...]], tuple[Any, ...]],
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
        *,
        # Whether we have the option to fallback to aten
        can_fallback_to_aten: bool = True,
        **kwargs: Any,
    ) -> Sequence[TensorBox | None]:
        # [FUNCTION DOC] Class method creating realized scan outputs along axis, one per dtype. Returns a list of None entries if the device lacks scan (or tuple reduction for multi-output) support, if splitting is needed but unsupported and can_fallback_to_aten is set (split needs triton support, a single output and non-deterministic mode; otherwise a SplitScan is used or the split is dropped). A scan over at most one element becomes pointwise copies. Raises AssertionError if dtypes and inner_fns lengths differ. Extra keyword arguments are forwarded to the node constructor.
        pointwise_ranges = [*size[:axis], *size[axis + 1 :]]
        scan_ranges = [size[axis]]

        if not V.graph.has_feature(device, BackendFeature.SCAN):
            return [None] * len(dtypes)

        if len(dtypes) > 1 and not V.graph.has_feature(
            device, BackendFeature.TUPLE_REDUCTION
        ):
            return [None] * len(dtypes)

        sizevars = V.graph.sizevars
        scan_numel = sizevars.simplify(sympy_product(scan_ranges))

        if len(dtypes) != len(inner_fns):
            raise AssertionError("Expected len(dtypes) == len(inner_fns)")

        # Scan with a single element is just a copy
        if sizevars.statically_known_true(sympy.Le(scan_numel, 1)):
            return [
                Pointwise.create(
                    device=device,
                    dtype=dtypes[output_index],
                    inner_fn=inner_fns[output_index],
                    ranges=size,
                )
                for output_index in range(len(dtypes))
            ]

        reduction_hint, num_splits = cls.num_splits(
            device=device,
            dtype=dtypes[0],
            inner_fn=inner_fns[0],
            axis=axis,
            pointwise_ranges=pointwise_ranges,
            scan_ranges=scan_ranges,
            combine_fn=combine_fn,
            scan_numel=scan_numel,
        )
        scan_type = Scan
        if num_splits > 1:
            triton_supports_split = (
                # pyrefly: ignore [unsupported-operation]
                torch.version.hip is None or (has_triton and triton_version >= "3.3.0")
            )
            supports_split = (
                triton_supports_split
                and len(dtypes) == 1
                and not torch.are_deterministic_algorithms_enabled()
            )
            if not supports_split:
                if can_fallback_to_aten:
                    # Fallback to ATen
                    return [None] * len(dtypes)
                else:
                    num_splits = 1
            else:
                scan_type = SplitScan

        def reindex(index: Sequence[Expr], scan_index: Sequence[Expr]) -> list[Expr]:
            # [FUNCTION DOC] HELPER: Build the full index by inserting the scan index at position axis in the pointwise index; AssertionError on length mismatches.
            if len(scan_index) != len(scan_ranges):
                raise AssertionError("Expected len(scan_index) == len(scan_ranges)")
            if len(index) != len(pointwise_ranges):
                raise AssertionError("Expected len(index) == len(pointwise_ranges)")
            return [*index[:axis], *scan_index, *index[axis:]]

        results = [
            TensorBox.create(
                scan_type(
                    device=device,
                    dtype=dtypes[output_index],
                    dtypes=dtypes,
                    inner_fn=inner_fns[output_index],
                    inner_fns=inner_fns,
                    size=size,
                    ranges=pointwise_ranges,
                    scan_ranges=scan_ranges,
                    combine_fn=combine_fn,
                    reindex=reindex,
                    reduction_hint=reduction_hint,
                    output_index=output_index,
                    **kwargs,
                )
            )
            for output_index in range(len(dtypes))
        ]

        for result in results:
            result.realize()

        return results

    @classmethod
    def num_splits(
        cls,
        device: torch.device,
        dtype: torch.dtype,
        inner_fn: Callable[[Sequence[Expr]], OpsValue],
        axis: int,
        pointwise_ranges: list[Integer],
        scan_ranges: list[Integer],
        combine_fn: Callable[[tuple[Any, ...], tuple[Any, ...]], tuple[Any, ...]],
        scan_numel: Expr,
    ) -> tuple[ReductionHint, _IntLike]:
        # [FUNCTION DOC] Class method choosing a (hint, split count) for a scan by delegating to the reduction heuristic with a wrapped inner function (no scan-specific heuristic yet).
        # TODO: custom splitting heuristic for scan
        def wrapper_fn(idx: Sequence[Expr], reduction_idx: Sequence[Expr]) -> OpsValue:
            # [FUNCTION DOC] HELPER: Adapt the scan inner function to the reduction calling convention by inserting the reduction index at the scan axis.
            return inner_fn([*idx[:axis], *reduction_idx, *idx[axis:]])

        return Reduction.num_splits(
            device=device,
            dst_dtype=dtype,
            src_dtype=dtype,
            inner_fn=wrapper_fn,
            ranges=pointwise_ranges,
            reduction_ranges=scan_ranges,
            reduction_type="scan",
            reduction_numel=scan_numel,
        )


# This signifies a scan op that should go through TritonSplitScanKernel codegen on CUDA.
@ir_dataclass
class SplitScan(Scan):
    # [CLASS DOC] Marker subclass of Scan indicating the scan should be emitted through the split-scan Triton kernel on CUDA; adds no fields.
    pass


@ir_dataclass
class Sort(Loops):
    # [CLASS DOC] Loops node for sorting along sort_ranges of key/value tuples (stable and descending flags), mimicking a reduction for codegen. One node exists per output (output_index), sharing dtypes and inner_fns. size is the full shape; reindex builds the full index from pointwise and sort index vars.
    # Sorts a tuple of key, value pairs
    sort_ranges: list[Integer]
    size: list[Integer]
    reindex: Callable[[Sequence[Expr], Sequence[Expr]], Sequence[Expr]]
    reduction_hint: ReductionHint
    output_index: int
    # output_index indexes the following tuples
    dtypes: tuple[torch.dtype, ...]
    inner_fns: tuple[Callable[..., Any], ...]

    stable: bool
    descending: bool

    # HACK we mimic reduction

    @cache_on_self_and_args("Sort")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        # [FUNCTION DOC] Return free symbols of the parent plus those in sort_ranges and size (cached).
        return (
            super().get_free_symbol_uses(unbacked_only)
            | OrderedSet().union(
                *(get_free_symbols(e, unbacked_only) for e in self.sort_ranges)
            )
            | OrderedSet().union(
                *(get_free_symbols(e, unbacked_only) for e in self.size)
            )
        )

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Validate that ranges plus sort ranges match the size in length (AssertionError otherwise), then run the parent post-init.
        if len(self.ranges) + len(self.sort_ranges) != len(self.size):
            raise AssertionError(
                "Expected len(self.ranges) + len(self.sort_ranges) == len(self.size)"
            )
        super().__post_init__()

    def store_reduction(
        self,
        output_name: str | None,
        indexer: Callable[[Sequence[Expr]], Expr],
        vars: Sequence[Expr],
        reduction_vars: Sequence[Expr],
    ) -> Any:
        # [FUNCTION DOC] Compute the full index, evaluate all inner functions, emit a sort op (with stable and descending flags) and store the output_index-th result at the indexed position (named unnamed if None).
        idx = self.reindex(vars, reduction_vars)
        values = tuple(inner_fn(idx) for inner_fn in self.inner_fns)
        result = ops.sort(self.dtypes, values, self.stable, self.descending)
        return ops.store(
            output_name or "unnamed", indexer(idx), result[self.output_index]
        )

    def get_reduction_type(self) -> str | None:
        # [FUNCTION DOC] Return the string sort as the reduction type.
        return "sort"

    def get_reduction_size(self) -> Sequence[Integer]:
        # [FUNCTION DOC] Return the sort ranges as the reduction size.
        return self.sort_ranges

    def get_size(self) -> Sequence[Integer]:
        # [FUNCTION DOC] Return the full size.
        return self.size

    def get_pointwise_size(self) -> Sequence[_IntLike]:
        # [FUNCTION DOC] Return the non-sorted (pointwise) ranges.
        return self.ranges

    def index_length(self) -> int:
        # [FUNCTION DOC] Return the total index length: pointwise plus sort ranges.
        return len(self.ranges) + len(self.sort_ranges)

    def inner_fn_args(self) -> Sequence[Sequence[Expr]]:
        # [FUNCTION DOC] Return the inner function arguments: the reindexed full index from output and sort index symbols.
        index = self._index(self.ranges)
        rindex = self._index(self.sort_ranges, SymT.R0_INDEX)
        idx = self.reindex(index, rindex)
        return (idx,)

    def inner_fn_free_symbols(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        # [FUNCTION DOC] Return free symbols of the inner function evaluated on the reindexed full index.
        index = self._index(self.ranges)
        rindex = self._index(self.sort_ranges, SymT.R0_INDEX)
        idx = self.reindex(index, rindex)
        return extract_free_symbols(self.inner_fn, idx, unbacked_only=unbacked_only)

    @classmethod
    def create(  # type: ignore[override]
        cls,
        device: torch.device,
        dtypes: tuple[torch.dtype, ...],
        inner_fns: tuple[Callable[[list[Expr]], Any], ...],
        size: list[Integer],
        axis: int,
        stable: bool,
        descending: bool,
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
        **kwargs: Any,
    ) -> Sequence[TensorBox | None]:
        # [FUNCTION DOC] Class method creating realized sort outputs along axis, one per dtype. Returns a list of None entries if the device lacks sort support or the sort cannot use a persistent Triton kernel (persistent reductions must be enabled, and unless decompose_sort_ops is set the sort size must be statically at most 512). Sorting at most one element yields pointwise copies. Raises AssertionError if dtypes and inner_fns lengths differ.
        pointwise_ranges = [*size[:axis], *size[axis + 1 :]]
        sort_ranges = [size[axis]]

        if not V.graph.has_feature(device, BackendFeature.SORT):
            return [None] * len(dtypes)

        sizevars = V.graph.sizevars
        sort_numel = sizevars.simplify(sympy_product(sort_ranges))

        # Heuristic, smallest rblock where triton usually outperforms aten.sort.
        # It also isn't bandwidth bound so fusion is unlikely to help.
        # When decompose_sort_ops is enabled, skip the size limit to always
        # attempt Triton sort (index dtype is widened to int32 in lowering).
        if config.triton.decompose_sort_ops:
            is_persistent_kernel = config.triton.persistent_reductions
        else:
            max_rblock = 512
            is_persistent_kernel = (
                config.triton.persistent_reductions
                and sizevars.statically_known_true(sympy.Le(sort_numel, max_rblock))
            )
        if not is_persistent_kernel:
            # We only support persistent triton kernels
            return [None] * len(dtypes)

        if len(dtypes) != len(inner_fns):
            raise AssertionError("Expected len(dtypes) == len(inner_fns)")

        # Sort with a single element is just a copy
        if sizevars.statically_known_true(sympy.Le(sort_numel, 1)):
            return [
                Pointwise.create(
                    device=device,
                    dtype=dtypes[output_index],
                    inner_fn=inner_fns[output_index],
                    ranges=size,
                )
                for output_index in range(len(dtypes))
            ]

        def reindex(index: Sequence[Expr], sort_index: Sequence[Expr]) -> list[Expr]:
            # [FUNCTION DOC] HELPER: Build the full index by inserting the sort index at position axis in the pointwise index; AssertionError on length mismatches.
            if len(sort_index) != len(sort_ranges):
                raise AssertionError("Expected len(sort_index) == len(sort_ranges)")
            if len(index) != len(pointwise_ranges):
                raise AssertionError("Expected len(index) == len(pointwise_ranges)")
            return [*index[:axis], *sort_index, *index[axis:]]

        results = [
            TensorBox.create(
                Sort(
                    device=device,
                    dtype=dtypes[output_index],
                    dtypes=dtypes,
                    inner_fn=inner_fns[output_index],
                    inner_fns=inner_fns,
                    size=size,
                    ranges=pointwise_ranges,
                    sort_ranges=sort_ranges,
                    reindex=reindex,
                    reduction_hint=reduction_hint,
                    output_index=output_index,
                    stable=stable,
                    descending=descending,
                    **kwargs,
                )
            )
            for output_index in range(len(dtypes))
        ]

        for result in results:
            result.realize()

        return results


def is_storage_and_layout(x: IRNode) -> bool:
    # [FUNCTION DOC] Return True if x can be viewed as a storage with a layout (without freezing the layout), False if that raises NotImplementedError.
    try:
        as_storage_and_layout(x, freeze=False)
        return True
    except NotImplementedError:
        return False


def is_contiguous_storage_and_layout(x: IRNode) -> bool:
    # [FUNCTION DOC] Return True if x has storage and layout and that layout is contiguous after the padding that would be applied to a flexible layout; False if there is no layout. Does not freeze the layout.
    try:
        _buffer, layout = as_storage_and_layout(x, freeze=False)
        # pad the stride here so we will NOT claim a tensor as contiguous
        # if a padding is gonna happen.
        if layout.should_pad_strides():
            if not isinstance(layout, FlexibleLayout):
                raise AssertionError(type(layout))
            layout = FixedLayout(
                layout.device,
                layout.dtype,
                layout.size,
                layout._pad_strides(layout.stride, layout.size, layout.dtype),
                layout.offset,
                layout.is_pinned,
            )
        return layout.is_contiguous()
    except NotImplementedError:
        return False


def is_dense_contiguous_storage_and_layout(x: IRNode) -> bool:
    # [FUNCTION DOC] Return True if x has a contiguous layout whose storage size statically equals offset plus numel (no extra storage); False otherwise.
    try:
        _buffer, layout = as_storage_and_layout(x, freeze=False)
        if not layout.is_contiguous():
            return False
        return V.graph.sizevars.statically_known_equals(
            layout.storage_size(), layout.offset + sympy_product(layout.size)
        )
    except NotImplementedError:
        return False


def as_storage_and_layout(
    x: IRNode,
    freeze: bool = True,
    want_contiguous: bool = False,
    stride_order: Sequence[int | Integer] | None = None,
    allow_padding: bool = False,
    exact_strides: Sequence[int | Integer] | None = None,
) -> tuple[StorageBox, Layout]:
    # [FUNCTION DOC] Simplify x into a (StorageBox, Layout) pair, unwrapping TensorBox, StorageBox, Buffer and ReinterpretView. For a Buffer and when freeze is set, the layout is frozen as contiguous, in a given stride order or exact strides (with optional padding), or decided by default. For a ReinterpretView the base is resolved without the stride arguments and the view layout is returned. Raises NotImplementedError for anything else.
    """
    Try to simplify x into a StorageBox and a Layout.

    allow_padding only affect how we apply stride_order. When allow_padding
    is True, we have the freedom to add padding when applying the stride_order.
    """
    if isinstance(x, TensorBox):
        return as_storage_and_layout(
            x.data,
            freeze=freeze,
            want_contiguous=want_contiguous,
            stride_order=stride_order,
            allow_padding=allow_padding,
            exact_strides=exact_strides,
        )
    if isinstance(x, StorageBox):
        _, layout = as_storage_and_layout(
            x.data,
            freeze=freeze,
            want_contiguous=want_contiguous,
            stride_order=stride_order,
            allow_padding=allow_padding,
            exact_strides=exact_strides,
        )
        return x, x.data.get_layout()
    if isinstance(x, Buffer):
        if freeze:
            if want_contiguous:
                x.freeze_layout()
                if not x.get_layout().is_contiguous():
                    raise AssertionError("Expected x.get_layout().is_contiguous()")
            elif stride_order is not None:
                x.freeze_layout_with_stride_order(
                    stride_order, allow_padding=allow_padding
                )
            elif exact_strides is not None:
                x.freeze_layout_with_exact_strides(
                    exact_strides, allow_padding=allow_padding
                )
            else:
                x.decide_layout()
        return StorageBox(x), x.get_layout()
    if isinstance(x, ReinterpretView):
        # making the base of x contiguous or stride_ordered will not necessarily make
        # the ReinterpretView either, so don't pass along those arguments
        buffer, _ = as_storage_and_layout(
            x.data,
            freeze=freeze,
        )
        return buffer, x.layout
    raise NotImplementedError


def is_stride_order_storage_and_layout(
    x: IRNode, stride_order: Sequence[int | Integer]
) -> bool:
    # [FUNCTION DOC] Return True if x has storage and layout and the layout follows the given stride order; False if no layout is available.
    try:
        _buffer, layout = as_storage_and_layout(x, freeze=False)
        return layout.is_stride_ordered(stride_order)
    except NotImplementedError:
        return False


def is_unaligned(node: IRNode) -> bool:
    # [FUNCTION DOC] Return True if the node may be unaligned: for a ReinterpretView, if its offset in bytes is not statically a multiple of the GPU alignment or its base is unaligned; for a Buffer, if its name is recorded as unaligned in the graph. Wrappers are looked through; anything else is assumed aligned.
    if isinstance(node, (TensorBox, StorageBox)):
        return is_unaligned(node.data)

    if isinstance(node, ReinterpretView):
        layout = node.layout
        has_unaligned_layout = not V.graph.sizevars.statically_known_multiple_of(
            layout.offset * get_dtype_size(layout.dtype), GPU_ALIGN_BYTES
        )
        return is_unaligned(node.data) or has_unaligned_layout

    if isinstance(node, Buffer):
        return node.get_name() in V.graph.unaligned_buffers

    # assume to be aligned otherwise
    return False


@ir_dataclass
class BaseView(IRNode):
    # [CLASS DOC] Base class of view nodes wrapping another IR node (data). Most queries delegate to data (dtype, layout, device, name, realize, reads); subclasses supply make_reindexer, which maps view indices to base indices. Loader and indexer are composed from the base's and the reindexer.
    data: IRNode

    @cache_on_self_and_args("BaseView")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        # [FUNCTION DOC] Return the free symbols of the wrapped data (cached).
        return self.data.get_free_symbol_uses(unbacked_only)

    def make_reindexer(self) -> Callable[[Sequence[Expr]], Sequence[Expr]]:
        # [FUNCTION DOC] Return the function mapping view indices to base indices; subclasses must override (raises NotImplementedError).
        raise NotImplementedError(f"make_reindexer NYI on {self}")

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        # [FUNCTION DOC] Return an indexer composed of the base's indexer applied after the reindexer.
        inner = self.data.make_indexer()
        reindex = self.make_reindexer()

        def indexer(idx: Sequence[Expr]) -> Expr:
            # [FUNCTION DOC] HELPER: Apply the reindexer then the base indexer.
            return inner(reindex(idx))

        return indexer

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        # [FUNCTION DOC] Return a loader composed of the base's loader applied after the reindexer.
        inner = self.data.make_loader()
        reindex = self.make_reindexer()

        def loader(idx: Sequence[Expr]) -> OpsValue:
            # [FUNCTION DOC] HELPER: Apply the reindexer then the base loader.
            return inner(reindex(idx))

        return loader

    @property
    def dtype(self) -> torch.dtype:
        # [FUNCTION DOC] Property: dtype of the wrapped data.
        return self.data.get_dtype()

    def get_layout(self) -> Layout:
        # [FUNCTION DOC] Return the layout of the wrapped data.
        return self.data.get_layout()

    def get_device(self) -> torch.device | None:
        # [FUNCTION DOC] Return the device of the wrapped data.
        return self.data.get_device()

    def get_origin_node(self) -> torch.fx.Node | None:
        # [FUNCTION DOC] Return None; views carry no origin node of their own.
        return None

    def get_name(self) -> str:
        # [FUNCTION DOC] Return the name of the wrapped data.
        return self.data.get_name()

    def get_pointwise_size(self) -> Sequence[Expr]:
        # [FUNCTION DOC] Return the view size as the pointwise size.
        return self.get_size()

    def mark_reuse(self, users: int, *, graph_reuse: bool = True) -> None:
        # [FUNCTION DOC] Forward the reuse hint to the wrapped data.
        return self.data.mark_reuse(users, graph_reuse=graph_reuse)

    def has_exceeded_max_reads(self) -> bool:
        # [FUNCTION DOC] Return whether the wrapped data exceeded its maximum reads.
        return self.data.has_exceeded_max_reads()

    def realize(self) -> str | None:
        # [FUNCTION DOC] Realize the wrapped data and return its result.
        return self.data.realize()

    def realize_hint(self) -> None:
        # [FUNCTION DOC] Forward the realize hint to the wrapped data.
        self.data.realize_hint()

    def get_storage_numel(self) -> _IntLike:
        # [FUNCTION DOC] Return the storage element count of the wrapped data.
        return self.data.get_storage_numel()

    def is_extern(self) -> bool:
        # [FUNCTION DOC] Return whether the wrapped data is an extern kernel.
        return self.data.is_extern()

    def is_module_buffer(self) -> bool:
        # [FUNCTION DOC] Delegate to the wrapped data's is_module_buffer. KNOWN ISSUE: raises AssertionError unless the wrapped data is itself a BaseView, so it cannot terminate on a non-view base here.
        if not isinstance(self.data, BaseView):
            raise AssertionError(type(self.data))
        return self.data.is_module_buffer()

    def get_read_names(self) -> OrderedSet[str]:
        # [FUNCTION DOC] Return the read names of the wrapped data.
        return self.data.get_read_names()

    def get_reads(self) -> OrderedSet[Dep]:
        # [FUNCTION DOC] Return read dependencies obtained by tracing this view's loader over its size with flexible-layout indexing allowed.
        with patch.object(FlexibleLayout, "allow_indexing", True):
            return extract_read_writes(
                self.make_loader(),
                self.get_size(),
            ).reads

    def unwrap_view(self) -> IRNode:
        # [FUNCTION DOC] Strip nested views and return the innermost non-view node.
        x: IRNode = self
        while isinstance(x, BaseView):
            x = x.data
        return x

    def constant_to_device(self, device: torch.device) -> IRNode:
        # [FUNCTION DOC] Return a Pointwise on the given device that applies this view's loader with constants re-targeted. Valid only if all reads are constants.
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Pointwise(
            device=device,
            dtype=self.get_dtype(),
            inner_fn=loader,
            ranges=self.get_size(),
        )


@ir_dataclass
class ExpandView(BaseView):
    # [CLASS DOC] View that broadcasts its data to a larger size (field size). Use create, which returns a ReinterpretView with zero strides for broadcast dims when the input has storage and layout, otherwise an ExpandView.
    size: Sequence[Expr]

    @staticmethod
    def _normalize_size(x: IRNode, new_size: Sequence[_IntLike]) -> Sequence[_IntLike]:
        # [FUNCTION DOC] Static method resolving a requested expand size: expands expressions, replaces -1 with the existing size (the input is left-padded with missing leading dims), and verifies broadcast compatibility for static shapes (via optimization hints), raising AssertionError on mismatch or invalid -1. Returns the new size list.
        """Replace `-1` with correct sizes"""
        sizevars = V.graph.sizevars
        new_size = [sympy.expand(s) for s in new_size]
        old_size = x.get_size()
        old_size = [None] * (len(new_size) - len(old_size)) + list(old_size)
        if len(new_size) != len(old_size):
            raise AssertionError("Expected len(new_size) == len(old_size)")
        for i in range(len(new_size)):
            if new_size[i] == -1:
                if old_size[i] is None:
                    raise AssertionError("Expected old_size[i] is not None")
                new_size[i] = old_size[i]
            elif old_size[i] is None or V.graph.sizevars.is_size_one_or_false(
                old_size[i]
            ):
                pass
            elif not has_free_unbacked_symbols(
                old_size
            ) and not has_free_unbacked_symbols(new_size):
                # Sanity check: Expect broadcast compatibility
                #
                # NB: new_size[i] == old_size[i] is expected to already be
                # guarded because the meta formula was expected to have taught
                # us this equality.
                v1 = new_size[i]
                v2 = old_size[i]
                if v1 is None:
                    raise AssertionError("Expected v1 is not None")
                if v2 is None:
                    raise AssertionError("Expected v2 is not None")
                diff = v1 - v2
                if (
                    sizevars.optimization_hint(
                        diff,
                        fallback=0,
                    )
                    != 0
                ):
                    raise AssertionError(
                        f"Broadcast failed in ExpandView({x.get_size()}, {new_size}) on dimension {i}"
                    )
        return new_size

    @classmethod
    def create(cls, x: IRNode, new_size: Sequence[_IntLike]) -> BaseView:
        # [FUNCTION DOC] Class method creating an expanded view. If x has storage and layout, returns a ReinterpretView with a fixed layout where new or size-one dims get stride zero; otherwise returns an ExpandView node. AssertionError if the new size has fewer dims than the input.
        new_size = cls._normalize_size(x, new_size)

        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            skip = len(new_size) - len(old_layout.size)
            if skip < 0:
                raise AssertionError("Expected skip >= 0")
            new_stride = [sympy.S.Zero] * skip
            for stride, size in zip(old_layout.stride, old_layout.size):
                new_stride.append(
                    stride
                    if not V.graph.sizevars.is_size_one_or_false(size)
                    else sympy.S.Zero
                )
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                list(new_size),
                new_stride,
                old_layout.offset,
                old_layout.is_pinned,
            )
            return ReinterpretView(data=storage, layout=new_layout)

        return ExpandView(data=x, size=new_size)

    def get_size(self) -> Sequence[Expr]:
        # [FUNCTION DOC] Return the expanded size.
        return self.size

    def make_reindexer(
        self,
    ) -> Callable[[Sequence[Expr]], Sequence[Expr]]:
        # [FUNCTION DOC] Return a reindex function mapping an expanded index to the base index: drops the leading broadcast-added dims and sets the index to zero wherever the base dimension has size one. AssertionError if lengths mismatch when called.
        target = self.get_size()
        actual = self.data.get_size()
        skip = len(target) - len(actual)

        def reindex(
            index: Sequence[Expr],
        ) -> Sequence[Expr]:
            # [FUNCTION DOC] HELPER: Drop leading indices and zero out broadcast (size one) dims of the base; AssertionError on length mismatch.
            index = list(index[skip:])
            if len(index) != len(actual):
                raise AssertionError("Expected len(index) == len(actual)")
            for i in range(len(actual)):
                if actual[i] == 1:
                    # zero out broadcast dimension
                    index[i] = sympy.S.Zero
            return index

        return reindex


@ir_dataclass
class PermuteView(BaseView):
    # [CLASS DOC] View applying a dimension permutation (dims) to its data. Use create, which yields a ReinterpretView with permuted size and stride when the input has storage and layout.
    dims: list[Expr]

    @classmethod
    def create(cls, x: IRNode, dims: Sequence[int]) -> BaseView:
        # [FUNCTION DOC] Class method creating a permuted view. Negative dims are normalized and dims must be a permutation (AssertionError otherwise). Returns a ReinterpretView with permuted sizes and strides if x has storage and layout, else a PermuteView node.
        dims = cls._map_neg_dims(dims)
        if OrderedSet(dims) != OrderedSet(range(len(dims))):
            raise AssertionError(
                "Expected OrderedSet(dims) == OrderedSet(range(len(dims)))"
            )

        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                [old_layout.size[i] for i in dims],
                [old_layout.stride[i] for i in dims],
                old_layout.offset,
                old_layout.is_pinned,
            )
            return ReinterpretView(data=storage, layout=new_layout)

        return PermuteView(data=x, dims=dims)

    @classmethod
    def _map_neg_dims(cls, dims: Sequence[int]) -> list[int]:
        # [FUNCTION DOC] HELPER: Map negative dim values to non-negative by adding the number of dims.
        return [dim if dim >= 0 else len(dims) + dim for dim in dims]

    def get_size(self) -> Sequence[Expr]:
        # [FUNCTION DOC] Return the data size permuted by dims; AssertionError if dims is not a permutation.
        if OrderedSet(self._map_neg_dims(self.dims)) != OrderedSet(
            range(len(self.dims))
        ):
            raise AssertionError(
                "Expected OrderedSet(self._map_neg_dims(self.dims)) == OrderedSet( range(len(self.dims)) )"
            )
        size = self.data.get_size()
        return [size[i] for i in self.dims]

    def get_stride(self) -> Sequence[Expr]:
        # [FUNCTION DOC] Return the data strides permuted by dims; AssertionError if dims is not a permutation.
        if not (
            OrderedSet(self._map_neg_dims(self.dims))
            == OrderedSet(range(len(self.dims)))
        ):
            raise AssertionError("dims must be a permutation of range(len(dims))")
        stride = self.data.get_stride()
        return [stride[i] for i in self.dims]

    def make_reindexer(
        self,
    ) -> Callable[[Sequence[Expr]], Sequence[Expr]]:
        # [FUNCTION DOC] Return a reindex function applying the inverse permutation to view indices to get base indices.
        inv = {j: i for i, j in enumerate(self.dims)}
        inv = [inv[i] for i in range(len(self.dims))]
        if OrderedSet(inv) != OrderedSet(range(len(self.dims))):
            raise AssertionError(
                "Expected OrderedSet(inv) == OrderedSet(range(len(self.dims)))"
            )

        def reindex(
            index: Sequence[Expr],
        ) -> Sequence[Expr]:
            # [FUNCTION DOC] HELPER: Reorder the index by the inverse permutation.
            return [index[i] for i in inv]

        return reindex


@ir_dataclass
class SqueezeView(BaseView):
    # [CLASS DOC] View that removes size-one dimensions from its data. Constructed only through create (direct construction raises AssertionError). Provides a squeezer helper that returns the new size and a reindexer.
    @classmethod
    def create(cls, x: IRNode, *, dim: int | None = None) -> IRNode:
        # [FUNCTION DOC] Class method squeezing one dimension (dim, which must have size one and be in range, else AssertionError) or, if dim is None, all dimensions that are size one. Returns a ReinterpretView with the dims dropped when x has storage and layout; otherwise a generic View with the reduced size.
        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            new_size = []
            new_stride = []
            if dim is not None:
                if not isinstance(dim, int):
                    raise AssertionError(type(dim))
                if not (0 <= dim and dim < len(old_layout.size)):
                    raise AssertionError(
                        "Expected 0 <= dim and dim < len(old_layout.size)"
                    )

            for i, (size, stride) in enumerate(zip(old_layout.size, old_layout.stride)):
                if dim is None:
                    # Only append if dim is not squeezed out
                    if not V.graph.sizevars.is_size_one_or_false(size):
                        new_size.append(size)
                        new_stride.append(stride)
                else:
                    if i != dim:
                        new_size.append(size)
                        new_stride.append(stride)
                    else:
                        if size != 1:
                            raise AssertionError("expected squeezed size to be 1")

            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                new_stride,
                old_layout.offset,
                old_layout.is_pinned,
            )
            return ReinterpretView(data=storage, layout=new_layout)

        if dim is None:
            return View.create(
                x,
                [
                    s
                    for s in x.get_size()
                    if not V.graph.sizevars.is_size_one_or_false(s)
                ],
            )
        else:
            if x.get_size()[dim] != 1:
                raise AssertionError("Expected x.get_size()[dim] == 1")
            return View.create(x, [s for i, s in enumerate(x.get_size()) if i != dim])

    @staticmethod
    def squeezer(
        size: Sequence[Expr],
    ) -> tuple[list[int], Callable[[Sequence[Expr]], tuple[Expr, ...]]]:
        # [FUNCTION DOC] Static method returning the squeezed size (dims not equal to one) and a reindex function that inserts zero for the removed dims; the reindex raises AssertionError on an index of wrong length.
        new_size = [s for s in size if s != 1]
        not_one = [i for i, s in enumerate(size) if s != 1]
        length = len(size)

        def reindex(index: Sequence[Expr]) -> tuple[Expr, ...]:
            # [FUNCTION DOC] HELPER: Expand a squeezed index back to full rank by filling removed dims with zero.
            if len(index) != len(not_one):
                raise AssertionError(f"{index} {not_one}")
            new_index: list[Expr] = [sympy.S.Zero] * length
            for idx, s in zip(not_one, index):
                new_index[idx] = s
            return tuple(new_index)

        return new_size, reindex

    def __init__(self, data: Any) -> None:
        # [FUNCTION DOC] Always raises AssertionError; use SqueezeView.create instead.
        raise AssertionError("use SqueezeView.create()")


@ir_dataclass
class GenericView(BaseView):
    # [CLASS DOC] View with an arbitrary size and a user-supplied reindex function mapping new indices to base indices. Base of View.
    size: Sequence[Expr]
    reindex: Callable[[Sequence[Expr]], Sequence[Expr]]

    def make_reindexer(
        self,
    ) -> Callable[[Sequence[Expr]], Sequence[Expr]]:
        # [FUNCTION DOC] Return the stored reindex function.
        return self.reindex

    def reindex_str(self) -> str:
        # [FUNCTION DOC] HELPER: Return a printable lambda-like string showing the reindex applied to index symbols.
        index_old = [
            sympy_index_symbol_with_prefix(SymT.INDEX, n) for n in range(len(self.size))
        ]
        index_new = list(self.reindex(index_old))
        return f"lambda {', '.join(map(str, index_old))}: {index_new}"

    def __str__(self) -> str:
        # [FUNCTION DOC] Return a string repr with the data, size and reindex text.
        return self.str_helper(
            [self.data, f"size={self.size}", f"reindex={self.reindex_str()}"]
        )

    __repr__ = __str__

    @classmethod
    def create(
        cls,
        x: IRNode,
        new_size: Sequence[Expr],
        reindex: Callable[[Sequence[Expr]], Sequence[Expr]],
    ) -> BaseView:
        # [FUNCTION DOC] Class method building a view node over x with the given size and reindex function.
        return cls(data=x, size=list(new_size), reindex=reindex)

    def get_size(self) -> Sequence[Expr]:
        # [FUNCTION DOC] Return the stored size.
        return self.size


@ir_dataclass
class View(GenericView):
    # [CLASS DOC] GenericView implementing tensor reshape. create returns the input unchanged for no-op reshapes, a ReinterpretView with computed strides when a stride-preserving view is possible (or the input is dense contiguous), and otherwise a View whose reindex function maps new indices to old indices through index math (dynamic_reshape_indexer).
    """
    This class handles tensor reshaping by computing appropriate index transformations
    to map the new shape back to the original storage layout.
    """

    @staticmethod
    def handle_negative_index(idx: Expr, size: Expr) -> Expr:
        # [FUNCTION DOC] Static method: expand idx and size, and if the index is negative (decided through the shape environment, which may add a guard) add the size to it. Returns the index.
        idx = sympy.expand(idx)
        size = sympy.expand(size)
        evaluate_expr = V.graph.sizevars.shape_env.evaluate_expr
        if evaluate_expr(sympy.Lt(idx, 0)):
            idx = idx + size
        return idx

    @classmethod
    @override
    def create(cls, x: IRNode, new_size: Sequence[Expr]) -> IRNode:  # type: ignore[override]
        if not isinstance(new_size, Sequence):
        # [FUNCTION DOC] Class method performing a reshape. Resolves a -1 size; returns x when old and new sizes are statically equal; for zero-element targets returns a view with a dummy zero reindex; if x is dense and contiguous returns a ReinterpretView with contiguous strides (forcing exact contiguous strides first); if x has no storage layout falls back to an index-math view; otherwise tries to compute valid strides (size oblivious when unbacked symbols are present) and returns a ReinterpretView on success. If strides cannot be computed it uses the dynamic reshape indexer, or on data-dependent guard failure forces x contiguous. Raises AssertionError if new_size is not a sequence.
            raise AssertionError(type(new_size))
        old_size, new_size = cls.resolve_negative_size(x.get_size(), new_size)

        # Skip pointless views
        if V.graph.sizevars.statically_known_list_equals(old_size, new_size):
            return x

        unbacked_symbols_in_sizes = (
            len(free_unbacked_symbols(old_size)) > 0
            or len(free_unbacked_symbols(new_size)) > 0
        )
        is_contiguous = is_dense_contiguous_storage_and_layout(x)

        def create_reinterpret_view(
            inp: IRNode, new_size: Sequence[Expr], new_stride: Sequence[Expr]
        ) -> ReinterpretView:
            # [FUNCTION DOC] HELPER: Force inp to contiguous exact strides, then return a ReinterpretView of its storage with the new size and stride.
            inp = ExternKernel.require_exact_strides(
                inp, FlexibleLayout.contiguous_strides(inp.get_size())
            )
            storage, old_layout = as_storage_and_layout(inp)
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                new_stride,
                old_layout.offset,
                old_layout.is_pinned,
            )
            return ReinterpretView(data=storage, layout=new_layout)

        def handle_unbacked_or_dynamic_reshape(
            x: IRNode,
        ) -> IRNode:
            # [FUNCTION DOC] HELPER: Reshape when no direct view is possible: try the dynamic reshape indexer and wrap in a View; if unbacked symbols prevent it, make x contiguous and return a contiguous ReinterpretView. Reads old_size and new_size from the enclosing scope.
            """
            Handle the case where view is not possible with current strides.
            Try dynamic_reshape_indexer first; if it fails with unbacked
            symbols (guard_or_false can't resolve comparisons), fall back
            to making the tensor contiguous.
            """
            nonlocal old_size, new_size
            try:
                reindex = cls.dynamic_reshape_indexer(old_size, new_size)
                return cls(data=x, size=list(new_size), reindex=reindex)
            except GuardOnDataDependentSymNode:
                # dynamic_reshape_indexer cannot handle unbacked SymInts
                # because guard_or_false can't resolve size comparisons.
                # https://github.com/pytorch/pytorch/issues/145561
                x = ExternKernel.require_contiguous(x)
                return create_reinterpret_view(
                    x, new_size, FlexibleLayout.contiguous_strides(new_size)
                )

        if 0 in new_size:

            def fake_reindex(index: Any) -> tuple[int, ...]:
                # [FUNCTION DOC] HELPER: Reindex for empty tensors; always the zero index of the old rank.
                return tuple([0] * len(old_size))

            return cls(data=x, size=list(new_size), reindex=fake_reindex)

        # TODO: a new class for FixedTransferLayout that output layout is constrained by input layout
        elif is_contiguous:
            # Input is contiguous, output can use contiguous strides
            return create_reinterpret_view(
                x, new_size, FlexibleLayout.contiguous_strides(new_size)
            )

        # Input is non-contiguous. Check if we can get storage/layout.
        if not is_storage_and_layout(x):
            # Can't get storage/layout (e.g., for Pointwise nodes)
            return handle_unbacked_or_dynamic_reshape(x)

        # Try to compute valid output strides.
        storage, old_layout = as_storage_and_layout(x, freeze=False)

        old_stride = old_layout.stride

        # Convert sympy exprs to SymInt for _compute_stride, then convert back
        old_size_symint = V.graph.sizevars.to_symints_or_ints(old_size)
        old_stride_symint = V.graph.sizevars.to_symints_or_ints(old_stride)
        new_size_symint = V.graph.sizevars.to_symints_or_ints(new_size)

        from torch._subclasses.fake_impls import _compute_stride

        # Use size_oblivious=True for unbacked symbols to avoid DDE errors
        new_stride_symint = _compute_stride(
            old_size_symint,
            old_stride_symint,
            new_size_symint,
            size_oblivious=unbacked_symbols_in_sizes,
        )

        if new_stride_symint is not None:
            # Convert SymInt back to sympy expressions
            new_stride = [
                s.node.expr if hasattr(s, "node") else sympy.Integer(s)
                for s in new_stride_symint
            ]
            # View is possible with computed strides
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                new_stride,
                old_layout.offset,
                old_layout.is_pinned,
            )
            return ReinterpretView(data=storage, layout=new_layout)

        # View not possible with current strides
        return handle_unbacked_or_dynamic_reshape(x)

    @staticmethod
    def resolve_negative_size(
        old_size: Sequence[Expr], new_size: Sequence[Expr]
    ) -> tuple[list[Expr], list[Expr]]:
        # [FUNCTION DOC] Static method simplifying old and new sizes and replacing a -1 entry in the new size by the quotient of the element counts; checks the element counts are equal (via the size variable check). Returns (old_size, new_size).
        new_size = [V.graph.sizevars.simplify(x) for x in new_size]
        old_size = [V.graph.sizevars.simplify(x) for x in old_size]

        new_size = list(new_size)
        for i in range(len(new_size)):
            if new_size[i] == -1:
                new_size[i] = sympy.S.One
                new_size[i] = CleanDiv(sympy_product(old_size), sympy_product(new_size))
                break

        V.graph.sizevars.check_equals(sympy_product(old_size), sympy_product(new_size))
        return old_size, new_size

    @classmethod
    def dynamic_reshape_indexer(
        cls,
        old_size: Sequence[_IntLike],
        new_size: Sequence[_IntLike],
        dense_dim: int | None = None,
    ) -> Callable[[Sequence[_T]], Sequence[_V]]:
        # [FUNCTION DOC] Class method returning a reindex function for reshaping old_size into new_size (optionally treating dense_dim as innermost when flattening). Tries the direct algorithm; if it fails (assertion, data-dependent guard or index error) reshapes via a flattened intermediate by fusing two reindexers.
        try:
            reindex = cls._dynamic_reshape_indexer(old_size, new_size, dense_dim)
        except (AssertionError, GuardOnDataDependentSymNode, IndexError):
            # optimistic algorithm failed, lets do a fallback
            flat = [sympy_product(old_size)]
            reindex1 = cls._dynamic_reshape_indexer(old_size, flat)
            reindex2 = cls._dynamic_reshape_indexer(flat, new_size)
            reindex = fuse_reindexing(reindex1, reindex2)
        return reindex

    @staticmethod
    def _dynamic_reshape_indexer(
        old_size: Sequence[Expr],
        new_size: Sequence[Expr],
        dense_dim: int | None = None,
    ) -> Callable[[Sequence[Expr]], Sequence[Expr]]:
        # [FUNCTION DOC] Static method computing a reindex via index arithmetic only: walks old and new sizes from the innermost dimension, matching equal sizes, merging several new dims into one old dim, or splitting one new dim into several old dims with modular indexing, using guard-free comparisons with divisibility fallback. Verifies leftover sizes are one. Returns a function mapping new index symbols to old index expressions. Raises AssertionError on inconsistent states.
        """
        Perform a reshape entirely by modifying indexing math
        """
        guard_or_false = V.graph.sizevars.guard_or_false

        def compare_sizes(a: Expr, b: Expr) -> int:
            # [FUNCTION DOC] HELPER: Compare two symbolic sizes returning -1, 0 or 1; when undecidable (unbacked symbols) falls back to divisibility, and raises GuardOnDataDependentSymNode otherwise.
            """
            Compare two symbolic sizes, returning -1 if a < b, 0 if a == b, 1 if a > b.

            For unbacked symbols, guard_or_false returns False, so we fall back
            to divisibility checks.
            """
            if guard_or_false(sympy.Eq(a, b)):
                return 0
            if guard_or_false(sympy.Lt(a, b)):
                return -1
            if guard_or_false(sympy.Gt(a, b)):
                return 1

            # Divisibility fallback for unbacked symbols:
            # e.g. comparing u0 vs u0*u1, statically_known_multiple_of(u0*u1, u0)
            # returns True so we take the merge path (return -1).
            # The merge reindex for old=[u0, u1] -> new=[u0*u1] is:
            #   for k in range(u0 * u1):
            #     i = k // u1
            #     j = k % u1
            #     z[k] = x[i, j] + 1
            # Two cases where this could seem wrong but is still safe:
            #   u1=1: the merge reindex is correct since the extra dim is
            #         size 1 (a no-op). e.g. k // 1 = k, k % 1 = 0.
            #   u0=0: loop is range(0), no kernel runs, result doesn't matter.

            if V.graph.sizevars.statically_known_multiple_of(b, a):
                return -1
            if V.graph.sizevars.statically_known_multiple_of(a, b):
                return 1

            raise GuardOnDataDependentSymNode(sympy.Eq(a, b))

        # TODO: These symbols may not escape, if they don't assert so and
        # treat them as temporary
        vars = [
            sympy_index_symbol_with_prefix(SymT.VIEW, i) for i in range(len(new_size))
        ]

        stack_new = list(zip(vars, new_size))
        stack_old = list(old_size)

        # process the dense dim first
        reordering_dense_dim = (
            dense_dim is not None
            and dense_dim != len(stack_old) - 1
            and len(new_size) == 1
        )
        if reordering_dense_dim:
            if dense_dim is None:  # mypy
                raise AssertionError("Expected dense_dim is not None")
            old_dim = stack_old.pop(dense_dim)
            stack_old.append(old_dim)

        view_expr = []
        while stack_new and stack_old:
            size_old = stack_old.pop()
            var, size_new = stack_new.pop()
            if size_old == 1:
                view_expr.append(sympy.S.Zero)
                stack_new.append((var, size_new))  # re-add
            elif size_new == 1:
                stack_old.append(size_old)  # re-add
            elif compare_sizes(size_new, size_old) == 0:
                view_expr.append(var)
            elif compare_sizes(size_new, size_old) < 0:
                while compare_sizes(size_new, size_old) < 0:
                    var2, size_new2 = stack_new.pop()
                    var = var2 * size_new + var
                    size_new = size_new * size_new2
                view_expr.append(var)
                V.graph.sizevars.check_equals(size_new, size_old)
            elif compare_sizes(size_new, size_old) > 0:
                divisor = sympy.S.One
                modulus = size_old
                view_expr.append(ModularIndexing(var, divisor, modulus))
                divisor = divisor * modulus
                while compare_sizes(size_new, size_old) > 0:
                    modulus = stack_old.pop()
                    view_expr.append(ModularIndexing(var, divisor, modulus))
                    divisor = divisor * modulus
                    size_old = size_old * modulus
                V.graph.sizevars.check_equals(size_new, size_old)
            else:
                raise AssertionError

        while stack_old:
            size_old = stack_old.pop()
            V.graph.sizevars.check_equals(size_old, 1)
            view_expr.append(sympy.S.Zero)

        while stack_new:
            var, size_new = stack_new.pop()
            V.graph.sizevars.check_equals(size_new, 1)

        if dense_dim is not None and len(new_size) == 1:
            view_expr.reverse()
            # Move the last expression (dense dim) to its original position
            dense_expr = view_expr.pop()
            view_expr.insert(dense_dim, dense_expr)
        else:
            view_expr.reverse()

        if len(view_expr) != len(old_size):
            raise AssertionError("Expected len(view_expr) == len(old_size)")

        def reindex(
            index: Sequence[Expr],
        ) -> Sequence[Expr]:
            # [FUNCTION DOC] HELPER: Substitute the index into the precomputed old-index expressions; AssertionError if length differs from the number of new dims.
            if len(index) != len(vars):
                raise AssertionError((len(index), len(vars)))
            replacements = dict(zip(vars, index))
            return tuple(sympy_subs(x, replacements) for x in view_expr)

        return reindex


@ir_dataclass
class ReinterpretView(BaseView):
    # [CLASS DOC] View that reinterprets the storage of its data with an explicit Layout (size, stride, offset, dtype). Post-init collapses nested views to the underlying data. Loads go through the layout's indexer (bitcasting if dtype differs from the data).
    """Pretend our storage has a different layout"""

    layout: Layout

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Run parent post-init and, if data is a view, replace it with its unwrapped underlying node.
        super().__post_init__()
        if isinstance(self.data, BaseView):
            object.__setattr__(self, "data", self.data.unwrap_view())

    def __str__(self) -> str:
        # [FUNCTION DOC] Return a repr string with data and layout.
        return self.str_helper(
            [
                self.data,
                self.layout,
            ]
        )

    __repr__ = __str__

    def get_name(self) -> str:
        # [FUNCTION DOC] Return the name of the underlying data.
        return self.data.get_name()

    def get_device(self) -> torch.device | None:
        # [FUNCTION DOC] Return the device from the layout.
        return self.layout.device

    def get_origin_node(self) -> torch.fx.Node | None:
        # [FUNCTION DOC] Return None; no origin node.
        return None

    @property
    def dtype(self) -> torch.dtype:
        # [FUNCTION DOC] Property: dtype from the layout.
        return self.layout.dtype

    def get_size(self) -> Sequence[Expr]:
        # [FUNCTION DOC] Return the layout size as a list.
        return list(self.layout.size)

    def get_stride(self) -> Sequence[Expr]:
        # [FUNCTION DOC] Return the layout stride as a list.
        return list(self.layout.stride)

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        # [FUNCTION DOC] Return a loader that loads from the named buffer at the layout's offset index, bitcasting to the layout dtype when it differs from the data dtype.
        def loader(index: Sequence[Expr]) -> OpsValue:
            # [FUNCTION DOC] HELPER: Load at the layout-indexed position, with bitcast when dtypes differ.
            indexer = self.layout.make_indexer()
            tmp_loader = ops.load(self.get_name(), indexer(index))
            if self.layout.dtype != self.data.dtype:
                return ops.to_dtype_bitcast(tmp_loader, self.dtype, self.data.dtype)
            else:
                return tmp_loader

        return loader

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        # [FUNCTION DOC] Return the layout's indexer.
        return self.layout.make_indexer()

    def get_layout(self) -> Layout:
        # [FUNCTION DOC] Return the layout.
        return self.layout

    def freeze_layout(self) -> None:
        # [FUNCTION DOC] No-op; the layout is already fixed.
        pass

    @cache_on_self_and_args("ReinterpretView")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return free symbols in the layout size, stride and offset (cached).
        return (
            get_free_symbols(self.layout.size, unbacked_only)
            | get_free_symbols(self.layout.stride, unbacked_only)
            | get_free_symbols(self.layout.offset, unbacked_only)
        )

    def codegen_reference(self, writer: IndentedBuffer | None = None) -> str:
        # [FUNCTION DOC] Return wrapper code text for a reinterpretation of the data buffer with the layout's size, stride, offset and dtype; may emit helper lines through the given writer or the wrapper's writeline. Reinterpret adds to the existing offset and disables view tracking.
        # reinterpret_tensor is similar to as_strided except:
        # - offset is added to the existing offset (rather than replacing it)
        # - view tracking is disabled similar to unsafe_view
        return V.graph.wrapper_code.codegen_reinterpret_view(
            self.data,
            self.layout.size,
            self.layout.stride,
            self.layout.offset,
            writer.writeline if writer is not None else V.graph.wrapper_code.writeline,
            dtype=self.layout.dtype,
        )

    def num_reads(self) -> int:
        # [FUNCTION DOC] Return 1 (reads one buffer).
        return 1


@ir_dataclass
class DtypeView(BaseView):
    # [CLASS DOC] View that reinterprets data bits as target_dtype. Use create, which gives a ReinterpretView with a changed dtype when storage and layout exist, otherwise a DtypeView with a bitcasting loader.
    """Pretend our storage has a different type"""

    target_dtype: torch.dtype

    @classmethod
    def create(cls, x: IRNode, new_dtype: torch.dtype) -> BaseView:
        # [FUNCTION DOC] Class method creating a bitcast view: ReinterpretView with the same size/stride but new dtype if x has storage and layout, else a DtypeView.
        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            new_layout = FixedLayout(
                old_layout.device,
                new_dtype,
                old_layout.size,
                old_layout.stride,
                old_layout.offset,
                old_layout.is_pinned,
            )
            return ReinterpretView(data=storage, layout=new_layout)
        return DtypeView(data=x, target_dtype=new_dtype)

    def __str__(self) -> str:
        # [FUNCTION DOC] Return a repr string with data and target dtype.
        return self.str_helper([self.data, self.target_dtype])

    __repr__ = __str__

    @property
    def dtype(self) -> torch.dtype:
        # [FUNCTION DOC] Property: the target dtype.
        return self.target_dtype

    def get_size(self) -> Sequence[Expr]:
        # [FUNCTION DOC] Return the data size.
        return self.data.get_size()

    def make_reindexer(self) -> Callable[[Sequence[Expr]], Sequence[Expr]]:
        # [FUNCTION DOC] Return the identity reindex function.
        def reindex(index: Sequence[Expr]) -> Sequence[Expr]:
            # [FUNCTION DOC] HELPER: Return the index unchanged.
            return index

        return reindex

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        # [FUNCTION DOC] Return a loader that loads from the data and bitcasts from the data dtype to the target dtype.
        inner = self.data.make_loader()

        def loader(idx: Sequence[Expr]) -> OpsValue:
            # [FUNCTION DOC] HELPER: Load and bitcast to the target dtype.
            return ops.to_dtype_bitcast(inner(idx), self.target_dtype, self.data.dtype)

        return loader


class SliceView(View):
    # [CLASS DOC] View (generic with reindex) representing a strided slice along one dimension: tensor[start:end:step] on dim. create returns a ReinterpretView with adjusted stride and offset when the input has storage and layout, otherwise a view with an index-math reindex.
    """View that represents a slice along a single dimension.

    Corresponds to tensor[..., start:end:step, ...].
    """

    @classmethod
    def create_with_size(
        cls,
        x: IRNode,
        dim: int,
        start: Expr,
        size: Expr,
        step: Expr,
    ) -> IRNode:
        # [FUNCTION DOC] Class method building a slice view with an explicit result size on dim; its reindex multiplies the index on dim by step and adds start (AssertionError on wrong index rank).
        new_size = list(x.get_size())
        new_size[dim] = size

        def reindex(
            index: Sequence[Expr],
        ) -> Sequence[Expr]:
            # [FUNCTION DOC] HELPER: Map a slice index to the base index on the sliced dim.
            if len(index) != len(new_size):
                raise AssertionError(f"wrong ndim {index} {new_size}")
            index = list(index)
            index[dim] = index[dim] * step + start
            return index

        return cls(data=x, size=new_size, reindex=reindex)

    @classmethod
    def normalize_start_end(
        cls, x: IRNode, dim: int, start: int, end: int
    ) -> tuple[int, int]:
        # [FUNCTION DOC] Class method clamping start and end into the range zero to the size of dim, with start not exceeding end. Negative values are wrapped by adding the size. Uses symbolic Min/Max when unbacked or symbolic min/max are involved, otherwise evaluates through the size variables (which may add guards). None values map to defaults. Returns (start, end).
        """
        Normalize start and end such that both are in the range
        [0, x.get_size()[dim]] and start <= end.
        """
        sizevars = V.graph.sizevars
        dim_size = x.get_size()[dim]

        if any(free_unbacked_symbols(x) for x in (start, end, dim_size)):
            min_func = Min
            max_func = Max
        elif any(
            # Only needed when backed_size_oblivious is on.
            x.has(sympy.Min, sympy.Max, Min, Max)
            for x in (start, end, dim_size)
            if isinstance(x, Expr)
        ):
            min_func = Min
            max_func = Max
        else:
            min_func = sizevars.evaluate_min
            max_func = sizevars.evaluate_max

        def clamp(x: Expr, lower: int, upper: int) -> Expr:
            # [FUNCTION DOC] HELPER: Clamp x into [lower, upper], emitting min/max only when bounds are not statically known to hold.
            clamped_lower = (
                x if sizevars.statically_known_geq(x, lower) else max_func(x, lower)
            )
            clamped_full = (
                clamped_lower
                if sizevars.statically_known_leq(clamped_lower, upper)
                else min_func(clamped_lower, upper)
            )
            return clamped_full

        def clamp_wrap(
            val: int | None, lower: int, upper: int, default: Expr | int
        ) -> Expr | int:
            # [FUNCTION DOC] HELPER: Normalize a possibly-negative or None bound and clamp it.
            if val is None:
                # TODO(rec): can this really happen?
                return default
            val = cls.handle_negative_index(val, dim_size)
            return clamp(val, lower, upper)

        start = clamp_wrap(start, 0, dim_size, 0)
        end = clamp_wrap(end, start, dim_size, dim_size)
        return start, end

    @classmethod
    def create(  # type: ignore[override]
        cls,
        x: IRNode,
        dim: int,
        start: int,
        end: int,
        step: int = 1,
        clamp: bool = True,
    ) -> IRNode:
        # [FUNCTION DOC] Class method creating a slice view of x on dim. Returns x itself for a full-range step-one slice. Normalizes bounds unless clamp is False (used by split_with_sizes), computes the new size by ceil division by step, and returns a ReinterpretView with scaled stride and shifted offset when x has storage and layout, else a generic slice view. KNOWN ISSUE: the step check passes whenever step is a sympy expression, so positivity is only enforced for plain numbers.
        step = sympy.expand(step)
        if not (isinstance(step, Expr) or step > 0):
            raise AssertionError(step)
        try:
            if start == 0 and end >= 2**63 - 1 and step == 1:
                return x
        except TypeError:
            pass

        new_size = list(x.get_size())

        # NB: Ordinarily we default to clamping.
        # We only don't clamp for split_with_sizes. For split_with_sizes, sizes should be already valid
        # failing in this situation is ok, since invalid sizes could trigger silent errors.
        if clamp:
            start, end = cls.normalize_start_end(x, dim, start, end)

        new_size[dim] = FloorDiv(end - start + (step - 1), step)

        if is_storage_and_layout(x):
            # Fast path
            storage, old_layout = as_storage_and_layout(x)
            new_stride = list(old_layout.stride)
            new_stride[dim] = new_stride[dim] * step
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                new_stride,
                old_layout.offset + old_layout.stride[dim] * start,
                old_layout.is_pinned,
            )
            return ReinterpretView(data=storage, layout=new_layout)

        # redirect to a generic view
        return cls.create_with_size(x, dim, start, new_size[dim], step)


@ir_dataclass
class BaseConstant(IRNode):
    # [CLASS DOC] Base class for scalar constant nodes: zero-dimensional (empty size), have dtype and device, no origin node and no reads.
    dtype: torch.dtype
    device: torch.device

    def get_size(self) -> Sequence[Expr]:
        # [FUNCTION DOC] Return the empty size (scalar).
        return ()

    def get_device(self) -> torch.device | None:
        # [FUNCTION DOC] Return the device.
        return self.device

    def get_origin_node(self) -> torch.fx.Node | None:
        # [FUNCTION DOC] Return None; constants have no origin node.
        return None

    def get_reads(self) -> OrderedSet[Dep]:
        # [FUNCTION DOC] Return an empty set of reads.
        return OrderedSet()


@ir_dataclass
class Constant(BaseConstant):
    # [CLASS DOC] Scalar constant with a Python value, dtype and device. Its loader emits a constant op; realize is a no-op.
    value: Any
    dtype: torch.dtype
    device: torch.device

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        # [FUNCTION DOC] Return a loader producing the constant value in the dtype for any index.
        def loader(index: Sequence[Expr]) -> OpsValue:
            # [FUNCTION DOC] HELPER: Emit the constant op.
            return ops.constant(self.value, self.dtype)

        return loader

    def realize(self) -> str | None:
        # [FUNCTION DOC] No-op; constants need no materialization.
        pass

    def constant_to_device(self, device: torch.device) -> IRNode:
        # [FUNCTION DOC] Return a copy of the constant on the given device.
        return Constant(value=self.value, dtype=self.dtype, device=device)


@ir_dataclass
class IndexingConstant(BaseConstant):
    # [CLASS DOC] Scalar constant defined by an index expression (symbolic integer) with dtype and device; the loader emits an index expression op.
    index: Any
    dtype: torch.dtype
    device: torch.device

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        # [FUNCTION DOC] Return a loader that emits the index expression in the dtype for any index.
        def loader(index: Sequence[Expr]) -> OpsValue:
            # [FUNCTION DOC] HELPER: Emit the index expression op.
            return ops.index_expr(self.index, self.dtype)

        return loader

    def constant_to_device(self, device: torch.device) -> IRNode:
        # [FUNCTION DOC] Return a copy on the given device.
        return IndexingConstant(index=self.index, dtype=self.dtype, device=device)


def is_contiguous_strides_for_shape(
    stride: Sequence[_IntLike], shape: Sequence[_IntLike]
) -> bool:
    # [FUNCTION DOC] Return True if stride is statically contiguous for the shape, iterating from the innermost dim and skipping size-one dims. Each stride must statically equal either the product of inner sizes or the product of inner sizes with each clamped to at least one (to tolerate zero-sized dims).
    expected_stride = 1
    expected_stride_max = 1
    for x, y in reversed(tuple(zip(shape, stride))):
        if x == 1:
            continue

        if not V.graph.sizevars.statically_known_equals(
            y, expected_stride
        ) and not V.graph.sizevars.statically_known_equals(y, expected_stride_max):
            return False

        expected_stride_max *= Max(1, x)
        expected_stride *= x

    return True


def get_align_for_dtype(dtype: torch.dtype) -> int:
    # [FUNCTION DOC] Return the padding alignment in elements: config.padding_alignment_bytes divided by the dtype item size.
    return config.padding_alignment_bytes // dtype.itemsize


class OutputSpec:
    # [CLASS DOC] Abstract base for Layout, MultiOutputLayout and NoneLayout: describes the memory layout of an Operation's output. Subclasses implement device, storage size and free-symbol queries.
    """Abstract base for Layout, MultiOutputLayout, NoneLayout.
    Represents the memory layout of the output of an Operation."""

    def get_device(self) -> torch.device | None:
        # [FUNCTION DOC] Return the device; abstract (NotImplementedError).
        raise NotImplementedError(type(self).__name__)

    def storage_size(self) -> int:
        # [FUNCTION DOC] Return the storage size; abstract (NotImplementedError).
        raise NotImplementedError(type(self).__name__)

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return free symbols (unbacked only if requested); abstract (NotImplementedError).
        raise NotImplementedError(type(self).__name__)


@ir_dataclass
class Layout(OutputSpec):
    # [CLASS DOC] Base class for tensor layouts: device, dtype, size, stride, offset and pinned flag, with strides defaulting to contiguous. Provides contiguity, transposed and stride-order checks, stride padding (comprehensive padding for flexible layouts), conversion to a FixedLayout and an indexer; equality compares all fields. Subclasses: FixedLayout (immutable strides) and FlexibleLayout (strides may still be chosen).
    """
    Layout base class

    Carries tensor meta-information including offset and
    whether it is pinned.
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        size: Sequence[_IntLike],
        stride: Sequence[Expr] | None = None,
        offset: Expr = Integer(0),
        is_pinned: bool = False,
    ) -> None:
        # [FUNCTION DOC] Initialize the layout. Strides default to contiguous strides of size. Raises AssertionError if size and stride lengths differ, any size is not int-like, or a pinned layout is not on cpu.
        if stride is None:
            stride = FlexibleLayout.contiguous_strides(size)
        # pyrefly: ignore [read-only]
        self.device = device
        self.dtype = dtype
        if len(size) != len(stride):
            raise AssertionError(f"size={size}, stride={stride}")
        if not all(isinstance(s, _IntLike) for s in size):
            raise AssertionError("Expected all(isinstance(s, _IntLike) for s in size)")
        self._size = size
        self._stride = stride
        self._offset = offset
        self.is_pinned = is_pinned
        # is_pinned implies cpu
        if not ((not self.is_pinned) or (self.device.type == "cpu")):
            raise AssertionError("Only CPU tensors can be pinned")

    @property
    def size(self) -> Sequence[_IntLike]:
        # [FUNCTION DOC] Property: the size.
        return self._size

    @size.setter
    def size(self, value: Sequence[_IntLike]) -> None:
        # [FUNCTION DOC] Set the size.
        self._size = value

    @property
    def stride(self) -> Sequence[Expr]:
        # [FUNCTION DOC] Property: the stride.
        return self._stride

    @stride.setter
    def stride(self, value: Sequence[Expr]) -> None:
        # [FUNCTION DOC] Set the stride.
        self._stride = value

    @property
    def offset(self) -> Expr:
        # [FUNCTION DOC] Property: the storage offset.
        return self._offset

    @offset.setter
    def offset(self, value: Expr) -> None:
        # [FUNCTION DOC] Set the storage offset.
        self._offset = value

    def __str__(self) -> str:
        # [FUNCTION DOC] Return a string with the class name, device, dtype, size, stride and, if set, offset and pinned flag.
        offset = ""
        if self.offset != 0:
            offset = f", offset={self.offset}"

        device_index_str = "" if self.device.index is None else f":{self.device.index}"
        is_pinned_str = ""
        if self.is_pinned:
            is_pinned_str = f", is_pinned={self.is_pinned}"
        return (
            f"{type(self).__name__}('{self.device.type}{device_index_str}', {self.dtype}, "
            f"size={self.size}, stride={self.stride}{offset}{is_pinned_str})"
        )

    __repr__ = __str__

    def get_device(self) -> torch.device:
        # [FUNCTION DOC] Return the device.
        return self.device

    def get_example(self) -> torch.Tensor:
        # [FUNCTION DOC] Return an empty strided fake tensor with the layout's size, stride, dtype, device and pinned flag, created in the fake mode.
        with V.fake_mode:
            return torch.empty_strided(
                convert_shape_to_symint(self.size),
                convert_shape_to_symint(self.stride),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.is_pinned,
            )

    def is_contiguous(self) -> bool:
        # [FUNCTION DOC] Return whether the strides are contiguous for the size.
        return is_contiguous_strides_for_shape(self.stride, self.size)

    @staticmethod
    def is_channels_last_contiguous(
        shape: Sequence[_IntLike], strides: Sequence[_IntLike]
    ) -> bool:
        # [FUNCTION DOC] Static method: return True if a 4D or 5D shape (with channel dim not one) has strides equal to channels-last strides on all non-one dims.
        ndim = len(shape)
        if ndim not in [4, 5] or shape[1] == 1:
            return False
        for left, right, size in zip(
            # pyrefly: ignore [bad-specialization]
            strides,
            # pyrefly: ignore [bad-specialization]
            make_channels_last_strides_for(shape),
            shape,
        ):
            if size != 1 and left != right:
                return False
        return True

    def is_transposed(self) -> bool:
        # [FUNCTION DOC] Return True if the layout is the transpose of a contiguous layout (strides match reversed contiguous strides of the reversed size, ignoring size-one dims).
        for left, right, size in zip(
            self.stride,
            reversed(FlexibleLayout.contiguous_strides(list(reversed(self.size)))),
            self.size,
        ):
            if size != 1 and left != right:
                return False
        return True

    @staticmethod
    def _stride_expr_ge_or_false(left: _IntLike, right: _IntLike) -> bool:
        # [FUNCTION DOC] Static method: symbolic test whether stride left is at least stride right, with zero handled specially and unbacked divisibility proven structurally. Returns False when it cannot prove it.
        """
        Returns true if left is symbolically greater than or equal to right,
        assuming both expressions are tensor strides.
        """
        sizevars = V.graph.sizevars
        if sizevars.guard_or_false(sympy.Eq(right, 0)):
            return True
        if sizevars.guard_or_false(sympy.Eq(left, 0)):
            return False
        # Prove unbacked divisibility structurally, and keep SymPy from
        # introducing reciprocal powers into any backed-symbol guard.
        return (
            sizevars.guard_or_false(sympy.Ge(left, right))
            or sizevars.statically_known_multiple_of(left, right)
            or sizevars.guard_or_false(sympy.Eq(Mod(left, right), 0))
        )

    def is_stride_ordered(self, order: Sequence[int]) -> bool:
        # [FUNCTION DOC] Return True if the strides follow the given stride order (ascending order values correspond to ascending strides), ignoring size-one dims and treating equal strides as compatible. Raises AssertionError if the order length differs from the stride count.
        if len(self.stride) != len(order):
            raise AssertionError("Expected len(self.stride) == len(order)")

        # ignore dimensions of size 1, they don't affect layout
        non_1_indices = [
            i
            for i, dim in enumerate(self.size)
            if not V.graph.sizevars.statically_known_equals(dim, 1)
        ]

        stride = [self.stride[i] for i in non_1_indices]
        order: Sequence[int] = [order[i] for i in non_1_indices]

        def sorted_indices(arr: Sequence[int]) -> Sequence[int]:
            # [FUNCTION DOC] HELPER: Return the rank of each element within the sorted list.
            sorted_arr = sorted(arr)
            return [sorted_arr.index(element) for element in arr]

        # since we may have removed dimensions, need to re-sort & re-index order
        order = sorted_indices(order)

        # reorder the stride given order
        stride_ordered = [-1] * len(order)
        for i in range(len(order)):
            stride_ordered[order[i]] = stride[i]
        # check if it is in ascending order
        for i in range(len(order) - 1):
            left = stride_ordered[i]
            right = stride_ordered[i + 1]
            if V.graph.sizevars.guard_or_false(sympy.Eq(left, right)):
                continue
            if self._stride_expr_ge_or_false(left, right):
                return False
            if not self._stride_expr_ge_or_false(right, left):
                return False
        return True

    def is_channels_last_stride_ordered(self) -> bool:
        # [FUNCTION DOC] Return True if the strides follow the channels-last stride order for the rank.
        # create channels_last order(NCHW, NCDHW, the C is the first order).
        order = [0] + list(reversed(range(1, len(self.stride) - 1)))
        order = [len(order)] + order
        return self.is_stride_ordered(order)

    @staticmethod
    def _pad_strides(
        in_strides: Sequence[int], size: Sequence[Expr], dtype: torch.dtype
    ) -> Sequence[int]:
        # [FUNCTION DOC] Static method computing padded strides that keep the stride order but round strides above config.padding_stride_threshold up to multiples of the dtype alignment. Returns the input unchanged for empty strides, channels-last contiguous layouts (unless config.pad_channels_last), nodes marked dislike_padding, dynamic shapes (unless config.pad_dynamic_shapes), strides with unbacked symbols, or when nothing needed padding. Increments metrics.num_comprehensive_padding when it pads.
        """
        The padding does not change stride order but makes sure all strides larger
        than the threshold are multiple of align.
        """
        align = get_align_for_dtype(dtype)
        if len(in_strides) == 0:
            return in_strides

        if not config.pad_channels_last and Layout.is_channels_last_contiguous(
            size, in_strides
        ):
            return in_strides

        current_fx_node = V.get_current_node()
        if hasattr(current_fx_node, "meta") and current_fx_node.meta.get(
            "dislike_padding", False
        ):
            return in_strides

        # Skip padding the strides for dynamic shapes based on config.pad_dynamic_shape
        # Checking both shape and strides, as there are cases where only one is dynamic
        is_dynamic = not all(
            isinstance(s, (int, sympy.Integer))
            for s in itertools.chain(in_strides, size)
        )
        if not config.pad_dynamic_shapes and is_dynamic:
            return in_strides

        shape_env = V.graph._shape_env if hasattr(V.graph, "_shape_env") else None

        def contains_unbacked_symints(expr: Expr | int) -> bool:
            # [FUNCTION DOC] HELPER: Return True if the expression contains unbacked symints according to the shape env (False if there is none).
            if shape_env is None:
                return False
            if not isinstance(expr, Expr):
                return False
            return any(shape_env.is_unbacked_symint(s) for s in expr.free_symbols)

        # Skip padding the strides when it contains unbacked symints for now.
        if shape_env and any(contains_unbacked_symints(s) for s in in_strides):
            return in_strides

        stride_order = get_stride_order(in_strides, shape_env)
        fill_order = stride_order2fill_order(stride_order)

        new_strides = [0 for _ in range(len(in_strides))]
        # since we pad when the layout is flexible, we can decide the
        # smallest stride to be 1.
        new_strides[fill_order[0]] = 1

        padded = False
        for rank, idx in enumerate(fill_order[1:], start=1):
            prev_idx = fill_order[rank - 1]
            stride = new_strides[prev_idx] * size[prev_idx]
            # Static stride and meets padding conditions OR
            # Dynamic stride and config.pad_dynamic_shape=True
            require_padding = (
                isinstance(stride, (int, sympy.Integer))
                and stride > config.padding_stride_threshold
                and stride % align != 0
            ) or (isinstance(stride, Expr) and config.pad_dynamic_shapes)
            new_strides[idx] = stride
            if require_padding:
                new_strides[idx] = ceildiv(stride, align) * align
                padded = True

        if not padded:
            # Consider a tensor with shape [256, 1, 5, 5]
            # Avoid strides like [25, 5, 5, 1] being padded to equivalent strides
            # [25, 25, 5, 1].
            return in_strides

        # pyrefly: ignore [bad-assignment]
        metrics.num_comprehensive_padding += 1
        return new_strides

    def pad_strides(self) -> None:
        # [FUNCTION DOC] Pad this FlexibleLayout's strides in place (AssertionError if not flexible or stride missing).
        if not isinstance(self, FlexibleLayout):
            raise AssertionError(type(self))
        if self.stride is None:
            raise AssertionError("Expected self.stride is not None")
        self.stride = self._pad_strides(self.stride, self.size, self.dtype)

    def should_pad_strides(self) -> bool:
        # [FUNCTION DOC] Return True if comprehensive padding is enabled and this is a FlexibleLayout.
        return config.comprehensive_padding and isinstance(self, FlexibleLayout)

    def as_fixed(self) -> FixedLayout:
        # [FUNCTION DOC] Return self if already fixed; otherwise pad strides if appropriate (mutating this layout) and return a FixedLayout with the same fields.
        if isinstance(self, FixedLayout):
            return self

        if self.should_pad_strides():
            self.pad_strides()
        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            self.stride,
            self.offset,
            self.is_pinned,
        )

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        # [FUNCTION DOC] Return an indexer for the layout via its fixed form. Raises AssertionError unless indexing is allowed on flexible layouts.
        if not FlexibleLayout.allow_indexing:
            raise AssertionError(f"convert {type(self).__name__} to FixedLayout first")
        return self.as_fixed().make_indexer()

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Layout equality: same kind of layout (a Layout instance) with equal device, dtype, size, stride, offset and pinned flag.
        return (
            isinstance(other, Layout)
            and self.device == other.device
            and self.dtype == other.dtype
            and self.size == other.size
            and self.stride == other.stride
            and self.offset == other.offset
            and self.is_pinned == other.is_pinned
        )

    def storage_size(self) -> Expr:
        # [FUNCTION DOC] Return the number of elements of storage the layout needs, from size, stride and offset.
        return compute_required_storage_length(self.size, self.stride, self.offset)  # type: ignore[arg-type]

    @cache_on_self_and_args("Layout")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return free symbols in size, stride and offset (cached).
        return (
            get_free_symbols(self.size, unbacked_only)
            | get_free_symbols(self.stride, unbacked_only)
            | get_free_symbols(self.offset, unbacked_only)
        )


class FixedLayout(Layout):
    # [CLASS DOC] Layout whose strides are final and cannot change; supplies a closure indexer over size, stride and offset.
    """A Tensor layout we cannot change"""

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        # [FUNCTION DOC] Return an indexer computing offset plus the sum of index times stride over non-size-one dims.
        """A closure containing math to read a given element"""
        return _fixed_indexer(self.size, self.stride, self.offset)


class FlexibleLayout(Layout):
    # [CLASS DOC] Layout whose strides may still be chosen later (frozen into a FixedLayout via as_fixed). Class flag allow_indexing gates indexer creation. Offers stride constructors (contiguous, fill order, stride order, memory format, same order as another stride). Layout changes must not add or remove free symbols.
    """
    A Tensor layout that we are allowed to change

    Assumption: layout change should NOT add or remove free symbols
    """

    allow_indexing = False

    def get_fixed_layout_without_freezing(self) -> FixedLayout:
        # [FUNCTION DOC] Return the FixedLayout this layout would become when frozen, computed on a deep copy so this layout is unchanged.
        """
        Compute what the strides would be if this layout were frozen,
        without actually modifying the layout. This is used for speculative
        stride computation during Triton template code generation.
        """
        # Create a temporary copy and use as_fixed to keep freezing path in sync
        return copy.deepcopy(self).as_fixed()

    # WARNING!  This doesn't handle zero size tensors correctly
    @staticmethod
    def contiguous_strides(sizes: Sequence[int]) -> list[Expr]:
        # [FUNCTION DOC] Static method: contiguous strides for sizes (empty for no dims). KNOWN ISSUE: zero-sized tensors are not handled correctly (per code warning).
        if len(sizes) == 0:
            return []
        reversed_strides = [sympy.S.One]
        for size in reversed(sizes[1:]):
            reversed_strides.append(size * reversed_strides[-1])
        return list(reversed(reversed_strides))

    @staticmethod
    def fill_ordered(sizes: Sequence[int], order: Sequence[int]) -> list[Expr]:
        # [FUNCTION DOC] Static method: strides given a fill order (the first listed dim gets stride one, each next gets the product of previous sizes). AssertionError if order is not a permutation.
        """
        Create a stride based on the order the dimensions should be filled in.

        In this format, channels last would be:
            [1, 3, 2, 0]
        """
        if OrderedSet(range(len(sizes))) != OrderedSet(order):
            raise AssertionError((sizes, order))
        next_stride = sympy.S.One
        strides = [None] * len(order)

        for i in order:
            strides[i] = next_stride
            next_stride = next_stride * sizes[i]
        return strides

    @staticmethod
    def stride_ordered(sizes: Sequence[int], order: Sequence[int]) -> Sequence[Expr]:
        # [FUNCTION DOC] Static method: strides given a stride order (converted to a fill order). AssertionError if order is not a permutation.
        """
        Create a stride based on the sorted order of a permuted range.

        In this format, channels last would be:
            [3, 0, 2, 1]
        """
        if OrderedSet(range(len(sizes))) != OrderedSet(order):
            raise AssertionError(
                "Expected OrderedSet(range(len(sizes))) == OrderedSet(order)"
            )
        fill_order = stride_order2fill_order(order)
        return FlexibleLayout.fill_ordered(sizes, fill_order)

    @staticmethod
    def stride_ordered_for_memory_format(
        sizes: Sequence[int], memory_format: torch.memory_format
    ) -> Sequence[Expr]:
        # [FUNCTION DOC] Static method: strides for a memory format (channels_last, channels_last_3d, contiguous). Raises NotImplementedError (after a debug log) for other formats such as preserve_format.
        """
        Create a stride based on a memory format.

        Memory format is translasted into a stride order,
        so channels_last is the same as:
            FlexibleLayout.stride_ordered(sizes, [3, 0, 2, 1])

        This interface does not support memory_format `torch.preserve_format`
        which should be used to deduce a format from another source
        """
        if memory_format == torch.channels_last:
            return FlexibleLayout.stride_ordered(sizes, NHWC_STRIDE_ORDER)
        elif memory_format == torch.channels_last_3d:
            return FlexibleLayout.stride_ordered(sizes, NHWDC_STRIDE_ORDER)
        elif memory_format == torch.contiguous_format:
            return FlexibleLayout.contiguous_strides(sizes)
        else:
            log.debug(
                "stride_ordered_for_memory_format, unsuppored memory_format: %s",
                memory_format,
            )
            raise NotImplementedError

    @staticmethod
    def same_ordered(
        sizes: Sequence[int], stride: Sequence[_IntLike]
    ) -> Sequence[Expr]:
        # [FUNCTION DOC] Static method: strides with the same dimension ordering as the given stride, using guarding hints for symbolic strides. AssertionError on length mismatch.
        """
        Create a stride that has the same stride order as given stride

        For example, if given stride is [1000, 1, 100, 10],
        the fill order should be [1, 3, 2, 0]
        """
        if len(sizes) != len(stride):
            raise AssertionError("Expected len(sizes) == len(stride)")
        stride = V.graph.sizevars.guarding_hints_or_throw(stride)
        fill_order = sorted(range(len(stride)), key=stride.__getitem__)
        return FlexibleLayout.fill_ordered(sizes, fill_order)

    @property
    def size(self) -> Sequence[Expr]:
        # [FUNCTION DOC] Property: the size.
        return self._size

    @size.setter
    def size(self, value: Sequence[Expr]) -> None:
        # [FUNCTION DOC] Set the size after asserting its free symbols are unchanged from creation (AssertionError otherwise).
        self.assert_free_symbol_uses_unchanged("size", value)
        self._size = value

    @property
    def stride(self) -> Sequence[Expr]:
        # [FUNCTION DOC] Property: the stride.
        return self._stride

    @stride.setter
    def stride(self, value: Sequence[Expr]) -> None:
        # [FUNCTION DOC] Set the stride after asserting its free symbols are unchanged from creation.
        self.assert_free_symbol_uses_unchanged("stride", value)
        self._stride = value

    @property
    def offset(self) -> Expr:
        # [FUNCTION DOC] Property: the offset.
        return self._offset

    @offset.setter
    def offset(self, value: Expr) -> None:
        # [FUNCTION DOC] Set the offset after asserting its free symbols are unchanged from creation.
        self.assert_free_symbol_uses_unchanged("offset", value)
        self._offset = value

    def as_stride_order(
        self, order: Sequence[int], allow_padding: bool = False
    ) -> FixedLayout:
        # [FUNCTION DOC] Return a FixedLayout with strides from the given stride order, padded when comprehensive padding applies and allow_padding is set.
        new_stride = self.stride_ordered(self.size, order)
        if self.should_pad_strides() and allow_padding:
            new_stride = self._pad_strides(new_stride, self.size, self.dtype)

        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            new_stride,
            self.offset,
            self.is_pinned,
        )

    def as_exact_strides(
        self, exact_strides: Sequence[_IntLike], allow_padding: bool = False
    ) -> FixedLayout:
        # [FUNCTION DOC] Return a FixedLayout with exactly the given strides, padded when comprehensive padding applies and allow_padding is set.
        new_stride = exact_strides
        if self.should_pad_strides() and allow_padding:
            new_stride = self._pad_strides(new_stride, self.size, self.dtype)

        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            new_stride,
            self.offset,
            self.is_pinned,
        )

    def as_fill_order(self, order: Sequence[int]) -> FixedLayout:
        # [FUNCTION DOC] Return a FixedLayout with strides from the given fill order, padded when comprehensive padding applies (padding is not optional here).
        new_stride: Sequence[int] = self.fill_ordered(self.size, order)
        if self.should_pad_strides():
            new_stride = self._pad_strides(new_stride, self.size, self.dtype)
        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            new_stride,
            self.offset,
            self.is_pinned,
        )

    def as_same_order(self, stride: Sequence[_IntLike]) -> FixedLayout:
        # [FUNCTION DOC] Return a FixedLayout whose strides have the same dimension order as the given stride, padded when comprehensive padding applies.
        new_stride = self.same_ordered(self.size, stride)
        if self.should_pad_strides():
            new_stride = self._pad_strides(new_stride, self.size, self.dtype)
        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            new_stride,
            self.offset,
            self.is_pinned,
        )

    def get_initial_free_symbol_uses(self) -> dict[tuple[str, bool], sympy.Symbol]:
        # [FUNCTION DOC] Snapshot the free symbols (all and unbacked-only) of size, stride and offset, keyed by name and flag.
        initial_free_symbols = {}
        for name in ["size", "stride", "offset"]:
            for unbacked_only in [True, False]:
                key = (name, unbacked_only)
                initial_free_symbols[key] = OrderedSet(
                    get_free_symbols(getattr(self, name), unbacked_only)
                )

        return initial_free_symbols

    def assert_free_symbol_uses_unchanged(self, name: str, value: IterateExprs) -> None:
        # [FUNCTION DOC] Check that the new value for size, stride or offset has the same free symbols (all and unbacked-only) as at creation; raises AssertionError otherwise.
        for unbacked_only in [True, False]:
            old_free_symbols = self.initial_free_symbols[(name, unbacked_only)]
            new_free_symbols = OrderedSet(get_free_symbols(value, unbacked_only))
            if new_free_symbols != old_free_symbols:
                raise AssertionError(
                    f"Expected free symbols unchanged, but got {new_free_symbols} vs {old_free_symbols}"
                )

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        size: Sequence[Expr],
        stride_order: Sequence[int | Integer] | None = None,
        is_pinned: bool = False,
    ) -> None:
        # [FUNCTION DOC] Initialize with contiguous strides, or strides from the fill-ordered stride_order if given; record the initial free symbols for later change checks.
        if stride_order:
            strides = FlexibleLayout.fill_ordered(size, stride_order)
        else:
            strides = FlexibleLayout.contiguous_strides(size)
        super().__init__(device, dtype, size, strides, is_pinned=is_pinned)

        # record the initial free symbols to check that we do not add new free symbols
        # later when modifying sizes, strides, and offsets.
        self.initial_free_symbols = self.get_initial_free_symbol_uses()


class NonOwningLayout(Layout):
    # [CLASS DOC] Layout of a tensor that is a view into the storage of another tensor (field view); copies device, dtype, size and stride from the view's layout.
    """Is a view into the storage of another tensor"""

    def __init__(self, view: BaseView | TensorBox) -> None:
        # [FUNCTION DOC] Copy device, dtype, size and stride from the view's layout and remember the view.
        layout = view.get_layout()
        super().__init__(
            layout.device,
            layout.dtype,
            layout.size,
            layout.stride,
        )
        self.view = view

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        # [FUNCTION DOC] Return the indexer of the fixed form of this layout.
        return self.as_fixed().make_indexer()

    def maybe_guard_aligned(self) -> bool:
        # [FUNCTION DOC] Return True if the view offset is zero or statically a multiple of the alignment.
        offset = self.view.get_layout().offset
        if offset == 0:
            return True
        from .utils import ALIGNMENT

        return V.graph.sizevars.statically_known_multiple_of(offset, ALIGNMENT)

    @cache_on_self_and_args("NonOwningLayout")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return the free symbols of the layout of the buffer underneath the view (cached). AssertionError if the view is not a ReinterpretView over a StorageBox of a Buffer.
        if not isinstance(self.view, ReinterpretView):
            raise AssertionError("Expected isinstance(self.view, ReinterpretView)")
        box = self.view.data
        if not isinstance(box, StorageBox):
            raise AssertionError(type(box))
        input_buffer = box.data
        if not isinstance(input_buffer, Buffer):
            raise AssertionError(type(box))
        return input_buffer.layout.get_free_symbol_uses(unbacked_only)


class CommBufferType(Enum):
    # [CLASS DOC] Enumeration of communication buffer kinds; currently only symmetric memory.
    SYMM_MEM = "symm_mem"


class CommBufferLayout(FixedLayout):
    # [CLASS DOC] Fixed layout marking a buffer as a communication buffer (with type and process group name). Striding equals FixedLayout. Such buffers neither serve as source nor target of in-place reuse.
    """
    A layout that signifies the buffer is a comm buffer.
    In terms of striding, the layout is identical to `FixedLayout`.

    Buffers with this layout do not participate in in-place reuse - it can be
    neither the source nor the target for in-place reuse.

    For detailed motivation and usage of this layout, see
    NOTE [lowering-time collective optimization].
    """

    comm_buffer_type: CommBufferType
    group_name: str

    def __init__(
        self,
        layout: FlexibleLayout | FixedLayout,
        comm_buffer_type: CommBufferType,
        group_name: str,
    ):
        # [FUNCTION DOC] Create from an existing layout (frozen to fixed if flexible), copying its fields, and record the comm buffer type and group name.
        fixed = layout.as_fixed() if isinstance(layout, FlexibleLayout) else layout
        super().__init__(
            device=fixed.device,
            dtype=fixed.dtype,
            size=fixed.size,
            stride=fixed.stride,
            offset=fixed.offset,
            is_pinned=fixed.is_pinned,
        )
        self.comm_buffer_type = comm_buffer_type
        self.group_name = group_name


@ir_dataclass
class NoneLayout(OutputSpec):
    # [CLASS DOC] OutputSpec for operations with no tensor output: holds an optional device and placeholder size and stride, storage size zero. Does not derive from Layout. Nodes with this layout need scheduler dependencies set up manually.
    # This is janky, I figured out what fields to populate by just running
    # the model I was interested in and adding properties/methods as needed.
    # This doesn't inherit from Layout because Layout assumes you have stuff
    # like sizes, but I don't really have anything here.
    #
    # If you have an ir.Node with NoneLayout, you probably need to setup
    # dependencies manually in scheduler

    device: torch.device | None
    size: list[int] = dataclasses.field(default_factory=lambda: [0])
    stride: list[int] = dataclasses.field(default_factory=lambda: [0])

    def storage_size(self) -> int:
        # [FUNCTION DOC] Return zero.
        return 0

    def as_fixed(self) -> OutputSpec:
        # [FUNCTION DOC] Return self (already fixed).
        return self

    def get_device(self) -> torch.device | None:
        # [FUNCTION DOC] Return the device.
        return self.device


class MutationLayoutSHOULDREMOVE(Layout):
    # [CLASS DOC] HELPER: Layout of a buffer that mutates a target node: takes device, dtype and size from the target, and delegates stride, storage size and indexing to the real layout of the underlying buffer. Marks the underlying buffer as mutated on creation. Marked in the name as one to be removed.
    def __init__(self, target: IRNode) -> None:
        # [FUNCTION DOC] Copy device, dtype and size from the target, remember it, and mark the underlying buffer as mutated in the graph.
        super().__init__(
            target.get_device_or_error(),
            target.get_dtype(),
            target.get_size(),
            None,
        )
        self.target = target
        name = self.get_buffer().get_name()
        V.graph.mark_buffer_mutated(name)

    @property
    def stride(self) -> Sequence[Expr]:  # type: ignore[override]
        return self.real_layout().stride

    @stride.setter  # type: ignore[override]
    def stride(self, value: Never) -> None:
        # [FUNCTION DOC] Property: stride of the real underlying layout.
        # [FUNCTION DOC] Ignore attempts to set the stride.
        pass  # ignore setting of stride

    def storage_size(self) -> Expr:
        # [FUNCTION DOC] Return the storage size of the real layout.
        return self.real_layout().storage_size()

    def get_buffer(self) -> Buffer:
        # [FUNCTION DOC] Return the underlying Buffer by unwrapping mutation layouts, views and boxes; AssertionError if the result is not a Buffer.
        def unwrap_views(target: Any) -> Any:
            # [FUNCTION DOC] HELPER: Recursively unwrap mutation layouts, views and mutable boxes.
            if isinstance(target, MutationLayoutSHOULDREMOVE):
                return unwrap_views(target.target)
            if isinstance(target, BaseView):
                return unwrap_views(target.unwrap_view())
            if isinstance(target, MutableBox):
                return unwrap_views(target.data)
            return target

        result = unwrap_views(self.target)
        if not isinstance(result, Buffer):
            raise AssertionError(type(result))
        return result

    def real_layout(self) -> Layout:
        # [FUNCTION DOC] Return the layout of the underlying buffer; AssertionError if it is not a Layout.
        layout = self.get_buffer().layout
        if not isinstance(layout, Layout):
            raise AssertionError("Expected isinstance(layout, Layout)")
        return layout

    @classmethod
    def realize_into(
        cls, src: IRNode, dst: IRNode, unsafe_alias: bool = False
    ) -> IRNode:
        # [FUNCTION DOC] Class method making dst reflect src by mutating dst. Realizes dst and marks it mutated, realizes a hint for src; unless unsafe_alias is set, copies src into a new pointwise node first (sizes verified equal) so src is not aliased; then realizes src and replaces its flexible layout with a mutation layout over dst. Returns the src buffer. Raises AssertionError on unexpected node shapes or non-flexible layouts. Side effects: mutates graph state and src layout.
        dst.realize()
        # NOTE: We must realize users of `dst` before we realize `src`, since
        # realization order determines scheduling order. Otherwise, src's
        # mutation would be scheduled before the existing users of dst!
        V.graph.mark_buffer_mutated(dst.get_name())

        if isinstance(src, TensorBox):
            src = src.data

        # We copy the contents of src into dst. In most cases this should
        # be fused into a single kernel by the scheduler.
        # NOTE: We cannot change src's layout to mutate dst directly as this
        # would alias src to dst, which is not correct as further mutations to
        # dst would effect users of src. However if there are no more users of
        # dst, we can alias src to dst.
        src.realize_hint()

        if not unsafe_alias:
            node = Pointwise.create(
                device=src.get_device(),
                dtype=src.get_dtype(),
                inner_fn=src.make_loader(),
                ranges=[
                    V.graph.sizevars.check_equals_and_simplify(a, b)
                    for a, b in zip(src.get_size(), dst.get_size())
                ],
            )
            if not isinstance(node, (BaseView, MutableBox)):
                raise AssertionError(
                    "Expected isinstance(node, (BaseView, MutableBox))"
                )
            src = node.data

        src.realize()
        if not hasattr(src, "data"):
            raise AssertionError(src)
        if not isinstance(src.data.layout, FlexibleLayout):
            raise AssertionError(type(src.data.layout))
        src.data.layout = MutationLayoutSHOULDREMOVE(dst)
        return src.data

    def as_fixed(self) -> Self:  # type: ignore[override]
        return self

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        # [FUNCTION DOC] Return self.
        # [FUNCTION DOC] Return the target's indexer.
        return self.target.make_indexer()


@ir_dataclass(frozen=False)
class Buffer(IRNode, CodegenSymbol):
    # [CLASS DOC] Base class of IR nodes that own storage: a (possibly None) name and an OutputSpec layout. Provides size, stride, dtype, offset and device queries via the layout, layout freezing variants (stride order, fill order, same order, exact strides), a loader that loads by name, alias and mutation info, and default no-op realize. ordering_only is a class flag that is false here. Multi-output buffers define their outputs separately.
    # Name is sometimes None; e.g., ForceInPlace, where there isn't
    # a meaningful name
    name: str | None
    layout: OutputSpec
    ordering_only: ClassVar[bool] = False

    # Multi-output buffers will define 'outputs: List[Buffer]'. Confusingly,
    # MultiOutput does NOT define this!

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Run parent post-init and clear the origin node.
        super().__post_init__()
        self._post_init_setattr("origin_node", None)

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        # [FUNCTION DOC] Return the layout's indexer.
        return self.get_layout().make_indexer()

    def get_name(self) -> str:
        # [FUNCTION DOC] Return the buffer name; AssertionError if unset.
        if not self.name:
            raise AssertionError(self)
        return self.name

    def get_buffer_name(self) -> str | None:
        # [FUNCTION DOC] Return the name (may be None).
        return self.name

    def get_example(self) -> torch.Tensor | torch.SymInt:
        # [FUNCTION DOC] Return an example tensor from the layout; NotImplementedError if the output spec is not a Layout.
        if isinstance(self.layout, Layout):
            return self.layout.get_example()
        raise NotImplementedError(type(self.layout).__name__)

    def get_device(self) -> torch.device | None:
        # [FUNCTION DOC] Return the device from the output spec.
        return self.get_output_spec().get_device()

    def get_defining_op(self) -> Operation | None:
        # [FUNCTION DOC] Return None; a plain buffer has no defining operation.
        return None

    @property
    def dtype(self) -> torch.dtype:
        # [FUNCTION DOC] Property: dtype from the layout.
        return self.get_layout().dtype

    def get_size(self) -> Sequence[Expr]:
        # [FUNCTION DOC] Return the layout size as a list.
        return [*self.get_layout().size]

    def get_stride(self) -> list[Expr]:
        # [FUNCTION DOC] Return the layout stride as a list.
        return [*self.get_layout().stride]

    def get_offset(self) -> Expr:
        # [FUNCTION DOC] Return the layout offset.
        return self.get_layout().offset

    def get_layout(self) -> Layout:
        # [FUNCTION DOC] Return the layout; NotImplementedError if the output spec is not a Layout.
        if isinstance(self.layout, Layout):
            return self.layout
        raise NotImplementedError(type(self.layout).__name__)

    def get_output_spec(self) -> OutputSpec:
        # [FUNCTION DOC] Return the output spec, i.e. the layout field.
        return self.layout

    def get_storage_numel(self) -> int:
        # [FUNCTION DOC] Return the storage element count, taken to be the element count.
        return self.get_numel()

    def get_is_pinned(self) -> bool:
        # [FUNCTION DOC] Return the pinned flag from the layout.
        return self.get_layout().is_pinned

    def freeze_layout(self) -> None:
        # [FUNCTION DOC] Freeze a flexible layout to fixed; layouts that are not Layout or are NonOwningLayout are left alone.
        if isinstance(self.layout, Layout) and not isinstance(
            self.layout, NonOwningLayout
        ):
            self.layout = self.layout.as_fixed()

    def freeze_layout_with_stride_order(
        self, order: Sequence[int], allow_padding: bool = False
    ) -> None:
        # [FUNCTION DOC] Freeze the layout to the given stride order (optionally padded); AssertionError unless the layout is flexible.
        if not isinstance(self.layout, FlexibleLayout):
            raise AssertionError(type(self.layout))
        self.layout = self.layout.as_stride_order(order, allow_padding=allow_padding)

    def freeze_layout_with_fill_order(self, order: Sequence[int]) -> None:
        # [FUNCTION DOC] Freeze the layout to the given fill order; AssertionError unless the layout is flexible.
        if not isinstance(self.layout, FlexibleLayout):
            raise AssertionError(type(self.layout))
        self.layout = self.layout.as_fill_order(order)

    def freeze_layout_with_same_order(self, stride: Sequence[int]) -> None:
        # [FUNCTION DOC] Freeze the layout to follow the order of the given stride; AssertionError unless flexible.
        if not isinstance(self.layout, FlexibleLayout):
            raise AssertionError(type(self.layout))
        self.layout = self.layout.as_same_order(stride)

    def freeze_layout_with_exact_strides(
        self, exact_strides: Sequence[int], allow_padding: bool = False
    ) -> None:
        # [FUNCTION DOC] Freeze the layout to the given exact strides (optionally padded); AssertionError unless flexible.
        if not isinstance(self.layout, FlexibleLayout):
            raise AssertionError(type(self.layout))
        self.layout = self.layout.as_exact_strides(
            exact_strides, allow_padding=allow_padding
        )

    def is_zero_elements(self) -> bool:
        # [FUNCTION DOC] Return True if the buffer is statically known to be empty.
        return V.graph.sizevars.statically_known_true(sympy.Eq(self.get_numel(), 0))

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        # [FUNCTION DOC] Return a loader that loads from this buffer by name at the layout-indexed position; for empty buffers a constant no-op loader (NaN or zero) is returned.
        # Loading from a zero-element buffer is a no-op
        if self.is_zero_elements():
            return partial(nop_loader_fn, dtype=self.get_dtype())

        def loader(index: Sequence[Expr]) -> OpsValue:
            # [FUNCTION DOC] HELPER: Emit a load op for the buffer name and indexed position.
            indexer = self.make_indexer()
            return ops.load(self.name or "unnamed", indexer(index))

        return loader

    def codegen_reference(self, writer: IndentedBuffer | None = None) -> str:
        # [FUNCTION DOC] Return the buffer name used to refer to it in generated code.
        return self.get_name()

    def decide_layout(self) -> None:
        # [FUNCTION DOC] No-op hook for deciding a flexible layout; subclasses override.
        pass

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        # [FUNCTION DOC] Return the name of the viewed buffer if the layout is non-owning, otherwise an empty tuple.
        if isinstance(self.layout, NonOwningLayout):
            return [self.layout.view.get_name()]
        return ()

    def get_mutation_names(self) -> Sequence[str]:
        # [FUNCTION DOC] Return the name of the mutation target if the layout is a mutation layout, otherwise an empty tuple.
        if isinstance(self.layout, MutationLayoutSHOULDREMOVE):
            return [self.layout.target.get_name()]
        return ()

    def get_read_names(self) -> OrderedSet[str]:
        # [FUNCTION DOC] Return a set holding this buffer's own name.
        return OrderedSet([self.get_name()])

    @cache_on_self_and_args("Buffer")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return an empty set (buffers use no free symbols themselves); cached.
        return OrderedSet()

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return an empty set of unbacked symbols defined.
        return OrderedSet()

    def realize(self) -> str | None:
        # [FUNCTION DOC] No-op; a buffer is already materialized.
        pass

    def should_allocate(self) -> bool:
        # [FUNCTION DOC] Return False: buffers do not allocate memory by default.
        # Returns False by default.
        return False


@ir_dataclass(frozen=False)
class OperationBuffer(Buffer, Operation):
    # [CLASS DOC] Buffer that is also an Operation: an operation producing exactly one output, itself. Combines both post-init hooks and shares the Operation operation name lookup.
    # An operation that produces a single output buffer
    def get_outputs(self) -> list[Buffer]:
        # [FUNCTION DOC] Return a list holding just this buffer.
        return [self]

    def get_defining_op(self) -> Operation:
        # [FUNCTION DOC] Return this node as its own defining operation.
        return self

    # Skip implementation in Buffer
    get_operation_name = Operation.get_operation_name

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Run both the Buffer and Operation post-init hooks.
        Buffer.__post_init__(self)
        Operation.__post_init__(self)


class InputBuffer(Buffer):
    # [CLASS DOC] Buffer for a graph input or compile-time value with no producing Operation; counts as one read.
    def num_reads(self) -> int:
        # [FUNCTION DOC] Return 1.
        return 1


class DonatedBuffer(InputBuffer):
    # [CLASS DOC] InputBuffer for a donated buffer: a saved tensor in backward that does not alias forward inputs, user outputs or backward outputs, and so may be reused in place to save memory.
    """
    Represents a donated buffer which is a saved tensor that is not alias to any
    fwd inputs, fwd user outputs, and bwd outputs. We generally cannot inplace
    reuse the input tensor memory during backward since it might be used in another
    function. However, donated buffer can be inplace reused during backward
    to save memory.
    """


class ConstantBuffer(InputBuffer):
    # [CLASS DOC] InputBuffer for a constant tensor held by the graph; loading uses the graph's constant name, optionally redirected to override_device (class-level setting used while moving constants between devices).
    override_device: torch.device | None = None

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        # [FUNCTION DOC] Return a loader emitting a load of the constant (name resolved for override_device) at the layout-indexed position.
        def loader(index: Sequence[Expr]) -> OpsValue:
            # [FUNCTION DOC] HELPER: Emit the load op.
            indexer = self.get_layout().make_indexer()
            return ops.load(
                V.graph.constant_name(self.get_name(), self.override_device),
                indexer(index),
            )

        return loader

    def constant_to_device(self, device: torch.device) -> IRNode:
        # [FUNCTION DOC] Return a ConstantBuffer for the same constant on the given device, with the same layout.
        return ConstantBuffer(
            name=V.graph.constant_name(self.get_name(), device), layout=self.layout
        )


@ir_dataclass
class NoneAsConstantBuffer(IRNode):
    # [CLASS DOC] IR node standing for a None value passed as a constant; no reads, no free symbols, codegen reference is the wrapper's none string, output spec is a NoneLayout, no tensor output.
    def get_reads(self) -> OrderedSet[Dep]:
        # [FUNCTION DOC] Return an empty set of reads.
        return OrderedSet()

    @cache_on_self_and_args("NoneAsConstantBuffer")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return an empty set (cached).
        return OrderedSet()

    def codegen_reference(self, writer: IndentedBuffer | None = None) -> str:
        # [FUNCTION DOC] Return the wrapper code's literal for None.
        return V.graph.wrapper_code.none_str

    def get_output_spec(self) -> OutputSpec:
        # [FUNCTION DOC] Return a NoneLayout with no device.
        return NoneLayout(device=None)

    def has_tensor_output(self) -> bool:
        # [FUNCTION DOC] Return False.
        return False


@ir_dataclass
class ShapeAsConstantBuffer(IRNode):
    # [CLASS DOC] IR node wrapping a symbolic shape expression (expr) so it can be an argument to extern kernels; codegen emits the size variable.
    expr: Expr

    @cache_on_self_and_args("ShapeAsConstantBuffer")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return the free symbols of the expression (cached).
        return get_free_symbols(self.expr, unbacked_only)

    def codegen_reference(self, writer: IndentedBuffer | None = None) -> str:
        # [FUNCTION DOC] Return the wrapper code for the size expression.
        return V.graph.wrapper_code.codegen_sizevar(self.expr)

    def has_tensor_output(self) -> bool:
        # [FUNCTION DOC] Return False.
        return False


@dataclasses.dataclass(frozen=True)
class ExtraIndexingConstraints:
    # [CLASS DOC] Frozen record of extra indexing constraints (var ranges and extra expressions) produced by the cpp fusion path to force compatible index and reduction ranges across scheduler nodes; passed through simplify_and_reorder and recompute_size_and_body.
    """Extra indexing constraints appended during simplify_and_reorder.

    Produced by the cpp fusion path to force compatible index/reduce ranges
    across scheduler nodes, then threaded through simplify_and_reorder and
    recompute_size_and_body. Replaces a bare tuple[VarRanges, list[Expr]].
    """

    ranges: VarRanges
    exprs: list[sympy.Expr]


@ir_dataclass(frozen=False)
class ComputedBuffer(OperationBuffer):
    # [CLASS DOC] OperationBuffer computed by a Loops node (data), as opposed to an input. Fields for split reductions (_split_size, _original_inner_fn, _original_ranges, _original_reduction_ranges) allow temporarily viewing the first stage as the unsplit reduction. Class flag _force_realize forces realization when set. Provides layout decisions, loop body and dependency extraction, fusion-related size/body simplification and codegen hooks.
    """
    Represents a buffer that is computed during kernel execution rather than being an input.
    """

    data: Loops
    _force_realize: ClassVar[bool] = False

    # fields for split reduction
    _split_size: int | None = None
    _original_inner_fn: Callable[..., Any] | None = None
    _original_ranges: Sequence[_IntLike] | None = None
    _original_reduction_ranges: Sequence[_IntLike] | None = None

    @contextlib.contextmanager
    def with_original_inner_fn(self) -> Iterator[None]:
        # [FUNCTION DOC] Context manager that temporarily replaces the data with the original unsplit Reduction (using the saved original function and ranges) and a dummy fixed layout, clears the cached default sizes/body, and restores data and layout on exit. Raises AssertionError if the split fields are missing or data is not a Reduction.
        if self._split_size is None:
            raise AssertionError("Expected self._split_size is not None")
        if self._original_inner_fn is None:
            raise AssertionError("Expected self._original_inner_fn is not None")
        if self._original_ranges is None:
            raise AssertionError("Expected self._original_ranges is not None")
        if self._original_reduction_ranges is None:
            raise AssertionError("Expected self._original_reduction_ranges is not None")

        if not isinstance(self.data, Reduction):
            raise AssertionError(f"{type(self.data)}")
        old_data = self.data
        old_layout = self.layout
        try:
            new_data = Reduction(
                device=old_data.device,
                dtype=old_data.dtype,
                inner_fn=self._original_inner_fn,
                ranges=self._original_ranges,
                reduction_ranges=self._original_reduction_ranges,
                reduction_type=old_data.reduction_type,
                src_dtype=old_data.src_dtype,
                reduction_hint=old_data.reduction_hint,
                strict_sum_multirow=old_data.strict_sum_multirow,
                strict_sum_rblock=old_data.strict_sum_rblock,
            )
            self.data = new_data
            # this layout does not matter since we skip tl.store
            # later
            self.layout = FixedLayout(
                old_data.device,
                old_data.dtype,
                self._original_ranges,
            )
            self.get_default_sizes_body.clear_cache(self)
            yield
        finally:
            self.data = old_data
            self.layout = old_layout

    @staticmethod
    @contextlib.contextmanager
    def force_realize() -> Iterator[None]:
        # [FUNCTION DOC] Static context manager that sets the class-level force-realize flag for its body and restores the old value on exit.
        old_value = ComputedBuffer._force_realize
        try:
            ComputedBuffer._force_realize = True
            yield
        finally:
            ComputedBuffer._force_realize = old_value

    def get_computed_buffer_name(self) -> str | None:
        # [FUNCTION DOC] Return the buffer's own name, else the name of the data node if it has one, else None.
        """
        Returns self.name if it exists, otherwise returns the name of the data node if that exists.
        If neither exist, returns None.
        """
        if self.name is not None:
            return self.name
        if hasattr(self.data, "name"):
            return self.data.name
        return None

    def num_reads(self) -> int:
        # [FUNCTION DOC] Return the number of reads of the computation (delegates to data).
        return self.data.num_reads()

    def get_reads(self) -> OrderedSet[Dep]:
        # [FUNCTION DOC] Return the read dependencies (delegates to data).
        return self.data.get_reads()

    def get_read_names(self) -> OrderedSet[str]:
        # [FUNCTION DOC] Return the read names (delegates to data).
        return self.data.get_read_names()

    def get_read_writes(self) -> dependencies.ReadWrites:
        # [FUNCTION DOC] Return the ReadWrites of the store: for Reduction, Scan, Sort or Pointwise data, trace the store function over the pointwise (and reduction) sizes with flexible indexing allowed; for other data, an empty ReadWrites.
        if not isinstance(self.data, (Reduction, Scan, Sort, Pointwise)):
            return dependencies.ReadWrites(
                reads=OrderedSet(),
                writes=OrderedSet(),
                index_exprs=OrderedSet(),
            )

        with patch.object(FlexibleLayout, "allow_indexing", True):
            if self.data.get_reduction_type():
                return extract_read_writes(
                    self.get_store_function(),
                    self.data.get_pointwise_size(),
                    self.data.get_reduction_size(),
                )
            else:
                return extract_read_writes(
                    self.get_store_function(),
                    self.data.get_size(),
                )

    @cache_on_self_and_args("ComputedBuffer")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return free symbols from the layout and the data, plus those from the traced read/writes when the data has a store function (cached). Needed for unbacked-symbol dependency tracking.
        # Ordinarily, we'd like to just peek at the arguments list,
        # but ComputedBuffers have no argument list.
        #
        # Morally, this logic needs to be synchronized with the
        # KernelArgs.size calls, which are responsible for making symbols make
        # there way as kernel arguments (and it is precisely passing in one of
        # those symbols that establishes a dependency).  However, we haven't
        # started codegen yet so we can't directly reuse that logic.
        #
        # One thing you might wonder is if this is enough for a ComputedBuffer
        # denoting a reduction over i0.  Empirically, it is enough, but for an
        # unusual reason: we only need accurate dependencies for item() call,
        # but it's impossible to end up with a reduction over i0 from an
        # item() call without a regular non-reduction buffer first.
        result = self.layout.get_free_symbol_uses(
            unbacked_only
        ) | self.data.get_free_symbol_uses(unbacked_only)

        if self.has_store_function():
            result |= self.get_read_writes().get_free_symbol_uses(unbacked_only)
        return result

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        # [FUNCTION DOC] Return a loader. Non-reduction buffers that are not mutated, have no reads and are not force-realized are inlined (the data loader is returned); otherwise a load from the buffer name.
        if (
            not self.get_reduction_type()
            and self.name not in V.graph.mutated_buffers
            and self.num_reads() == 0
            and not self._force_realize
        ):
            # inline this op rather than generating ops.load()
            return self.data.make_loader()
        return super().make_loader()

    def has_store_function(self) -> bool:
        # [FUNCTION DOC] Return True if the data is a Reduction, Scan, Sort or Pointwise node.
        return isinstance(self.data, (Reduction, Scan, Sort, Pointwise))

    def get_store_function(self) -> Callable[..., None]:
        # [FUNCTION DOC] Return a callable that stores the computation into this buffer: data store_reduction for Reduction, Scan, Sort, or store_output for Pointwise, bound to the buffer name and fixed-layout indexer. AssertionError for other data types.
        indexer = self.get_layout().as_fixed().make_indexer()
        if isinstance(self.data, (Reduction, Scan, Sort)):
            return partial(self.data.store_reduction, self.name, indexer)
        else:
            if not isinstance(self.data, Pointwise):
                raise AssertionError(type(self.data))
            return partial(self.data.store_output, self.name, indexer)

    def get_fill_order(self) -> list[int] | None:
        # [FUNCTION DOC] If the layout is still flexible, derive a fill order for the dimensions from the stride patterns of the memory reads (ignoring star deps and reading reduction vars as zero) using the scheduler's loop-order picker. Returns a list of dim indices or None. Raises AssertionError if reads include other dependency kinds.
        """
        If our layout is still flexible, try to determine the stride order based on stride orders of reads.

        TODO(jansel): A better algorithm here would look at downstream consumers of this
                      value and try to do global graph-level layout optimization.
                      This is also something just begging to be autotuned.
        """
        if isinstance(self.layout, FlexibleLayout):
            (index_vars, reduction_vars), _ = dependencies.index_vars_squeeze(
                self.data.get_pointwise_size(), self.data.get_reduction_size()
            )
            reads = self.get_read_writes().reads
            # only consider reads to buffer of same size
            # ignore StarDeps because they don't contribute stride information
            if not all(
                isinstance(r, (dependencies.StarDep, dependencies.MemoryDep))
                for r in reads
            ):
                raise AssertionError(
                    "Expected all( isinstance(r, (dependencies.StarDep, dependencies.MemoryDep)) for r in reads )"
                )
            reads = [
                sympy_subs(r.index, {v: sympy.S.Zero for v in reduction_vars if v != 0})
                for r in reads
                if isinstance(r, dependencies.MemoryDep)
            ]

            if reads:
                if isinstance(self.data, (Scan, Sort)):
                    indices = self.data.reindex(index_vars, reduction_vars)
                else:
                    indices = index_vars
                stride_lengths = [
                    V.graph.sizevars.stride_hints(expr, indices) for expr in reads
                ]
                from .scheduler import pick_loop_order

                return pick_loop_order(stride_lengths, self.get_size())

        return None

    def decide_layout(self) -> None:
        # [FUNCTION DOC] If the layout is flexible, freeze it using the derived fill order, or contiguous if none is found.
        if isinstance(self.layout, FlexibleLayout):
            order = self.get_fill_order()
            if order:
                self.freeze_layout_with_fill_order(order)
            else:
                self.freeze_layout()

    @cache_on_self
    def get_default_sizes_body(
        self,
    ) -> tuple[
        tuple[list[Expr], list[Expr]],
        LoopBody,
        tuple[list[Expr], list[Expr]],
    ]:
        # [FUNCTION DOC] Build the default loop body: squeezed index variables with a name prefix q, the loop body tracing the store function (with constants targeting this device), and the index/reduction variables and sizes. Returns ((index sizes, reduce sizes), LoopBody, (index vars, reduce vars)); cached on the node.
        args, var_ranges = dependencies.index_vars_squeeze(
            self.get_pointwise_size(), self.get_reduction_size(), prefix="q"
        )
        with patch.object(ConstantBuffer, "override_device", self.get_device()):
            body = LoopBody(
                self.get_store_function(),
                (args if self.get_reduction_type() else args[:1]),
                var_ranges,
                *args,
            )
        index_vars = []
        reduce_vars: list[Any] = []
        index_size = []
        reduce_size = []
        for v, s in var_ranges.items():
            if v in args[0]:
                if reduce_vars:
                    raise AssertionError("Expected not reduce_vars")
                index_vars.append(v)
                index_size.append(s)
            else:
                if v not in args[1]:
                    raise AssertionError("Expected v in args[1]")
                reduce_vars.append(v)
                reduce_size.append(s)
        return (index_size, reduce_size), body, (index_vars, reduce_vars)

    def simplify_and_reorder(
        self,
        extra_indexing_constraints: ExtraIndexingConstraints | None = None,
        recompute_sizes_body_func: Callable[..., Any] | None = None,
    ) -> tuple[tuple[list[Expr], list[Expr]], LoopBody | None]:
        # [FUNCTION DOC] Backend-agnostic loop transformation: starting from the default body (optionally recomputed via recompute_sizes_body_func), removes size-one dims, fuses contiguous dims and reorders loops by stride, separately for iteration and reduction dims. Extra indexing constraints (ranges must equal the body var ranges, else AssertionError) add index formulas that prevent over-simplification. Merging of loops is skipped on GPU when config.loop_ordering_after_fusion is set. Returns ((iteration ranges, reduction ranges), new LoopBody).
        """
        This is a main place where we do loop transformations in a
        backend-agnostic way.

        Here we:
            1) Remove any 1 dimensions
            2) Fuse contiguous dimensions together
            3) Reorder dimensions based on stride orders

        Optional argument extra_indexing_constraints can be used to append additional
        indexing expressions to existing ones derived from buffer's body. This can be useful
        to fuse scheduler nodes with compatible ranges, e.g. (s0*s1*...,) and (s0, s1, s2, ...)
        on CPU by preventing indexing simplifications and obtaining index/reduce ranges for
        the scheduler node compatible with other nodes.
        Optional argument recompute_sizes_body_func can be used to recompute sizes and body
        on the default body. This can be useful to append additional loop transformations.
        """
        (
            (index_size, reduce_size),
            body,
            (index_vars, reduce_vars),
        ) = self.get_default_sizes_body()

        if recompute_sizes_body_func:
            (
                (index_size, reduce_size),
                body,
                (index_vars, reduce_vars),
            ) = recompute_sizes_body_func(
                (index_size, reduce_size), body, (index_vars, reduce_vars)
            )

        index_formulas = [*body.indexing_exprs.values()]
        if extra_indexing_constraints is not None:
            expected_var_ranges = body.var_ranges
            if expected_var_ranges != extra_indexing_constraints.ranges:
                raise AssertionError(
                    (
                        expected_var_ranges,
                        extra_indexing_constraints.ranges,
                    )
                )
            # remove already existing expressions
            extra_indexing_expr = [
                e for e in extra_indexing_constraints.exprs if e not in index_formulas
            ]
            index_formulas += extra_indexing_expr

        memory_addrs = [*body.get_write_exprs()]
        if not V.graph.has_feature(self, BackendFeature.PREFER_STORE_LOOP_ORDER):
            memory_addrs.extend(body.get_read_exprs())

        def simplify_and_reorder(
            x_vars: Sequence[sympy.Symbol],
            support_vars: Sequence[sympy.Symbol],
            sizes: Sequence[int],
            simplify_loops: bool,
        ) -> tuple[
            list[int],
            Callable[[Sequence[int]], Sequence[int]],
            Callable[[Sequence[int]], Sequence[int]],
        ]:
            # [FUNCTION DOC] HELPER: Reorder one group of loop variables by memory stride and optionally merge loops. For native matmul (dot reduction, three dims) loop order is reverted to the default if reordering moved the batch axis from outermost. Returns (new sizes, reindex to apply, reorder-only reindex).
            newsizes, reindex0, reindex1 = self._apply_loop_reordering(
                x_vars, support_vars, sizes, memory_addrs
            )

            # When using native matmul, the codegen assumes the following loop order,
            # regardless of the stride of A and B:
            #
            #   for z -> y -> x -> r:  C[z, y, x] += A[z, y, r] * B[z, r, x]
            # or
            #   for z -> x -> y -> r:  C[z, y, x] += A[z, y, r] * B[z, r, x]
            #
            # The critical point is the position of the "z" (batch) axis in bmm.
            # It is fine to swap the y and x axes (e.g., (z, y, x, r) or (z, x, y, r)),
            # but reordering the z axis (e.g., (y, x, z, r)) breaks codegen.
            #
            # Therefore, if loop reordering changes the "z" location in bmm,
            # it should be reverted to the default.
            # This may not always produce the optimal loop order when strides
            # do not align with the default assumption.
            #
            # TODO: Consider extending tl.dot codegen to support arbitrary loop orders.
            if self.get_reduction_type() == "dot" and len(sizes) == 3:
                order = list(range(len(sizes)))  # default order

                # if z axis is not the outermost, use the default reorder.
                if reindex0(order)[0] != 0:
                    newsizes = [sizes[i] for i in order]
                    reindex0 = same_reorder(order)
                    reindex1 = inverse_reorder(order)

            # for NHWC: reindex0([0,1,2,3]) = [0,2,3,1], reindex1([0,1,2,3]) = [0,3,2,1]
            x_vars = reindex0(x_vars)

            if simplify_loops:
                newsizes, reindex2, _prune = V.graph.sizevars._simplify_loops(
                    x_vars,
                    newsizes,
                    index_prevent_reordering(index_formulas, x_vars, newsizes),
                )
                reindex = fuse_reindexing(reindex1, reindex2)
            else:
                reindex = reindex1
            return newsizes, reindex, reindex1

        support_vars = index_vars + reduce_vars
        should_merge_loops = (
            not is_gpu(get_device_type(self)) or not config.loop_ordering_after_fusion
        )
        iter_ranges, iter_reindex, _ = simplify_and_reorder(
            index_vars,
            support_vars,
            index_size,
            should_merge_loops,
        )

        # Like iteration dimensions, we may also want to delay merging reduction dimensions.
        # E.g., if we reduce a tensor [M, N, K] for its M and N dimensions followed by a pointwise
        # kernel, merging M and N dimension too early makes it hard to decide what loop order
        # we should pick for the piontwise kernel so that it is fusible with the reduction.
        reduce_ranges, reduce_reindex, _ = simplify_and_reorder(
            reduce_vars, support_vars, reduce_size, should_merge_loops
        )

        # retrace the loop body with simplification and reordering applied
        (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(
            iter_ranges,
            reduce_ranges,
            prefix="p",
        )
        body = LoopBody(
            body,
            [iter_reindex(iter_vars), reduce_reindex(reduce_vars)],
            var_ranges,
            iter_vars,
            reduce_vars,
        )
        return (iter_ranges, reduce_ranges), body

    @staticmethod
    def _apply_loop_reordering(
        index_vars: Sequence[sympy.Symbol],
        support_vars: Sequence[sympy.Symbol],
        sizes: Sequence[int],
        memory_addrs: list[Expr],
        priority_idx: list[int] | None = None,
    ) -> tuple[
        list[int],
        Callable[[Sequence[int]], Sequence[int]],
        Callable[[Sequence[int]], Sequence[int]],
    ]:
        # [FUNCTION DOC] Static method choosing a loop order from stride hints of the memory addresses using the scheduler's picker, honoring priority indices. On any failure (logged as a warning when config.debug) keeps the original order. Returns (reordered sizes, forward reorder function, inverse reorder function).
        """
        Shuffle the order of loops around to hopefully improve performance.
        """
        from .scheduler import pick_loop_order

        if priority_idx is None:
            priority_idx = []

        try:
            strides = [
                V.graph.sizevars.stride_hints(expr, index_vars, support_vars)
                for expr in memory_addrs
            ]
            if not (
                len(strides) == len(memory_addrs) and len(strides[0]) == len(index_vars)
            ):
                raise AssertionError(
                    "Expected len(strides) == len(memory_addrs) and len(strides[0]) == len( index_vars )"
                )
            order = list(reversed(pick_loop_order(strides, sizes, priority_idx)))
        except Exception:
            if config.debug:
                log.warning(
                    "Did not simplify complex index:\n%s\n%s",
                    dict(zip(index_vars, sizes)),
                    memory_addrs,
                )
            order = list(range(len(sizes)))
        sizes = [sizes[i] for i in order]
        return sizes, same_reorder(order), inverse_reorder(order)

    def get_pointwise_size(self) -> Sequence[Expr]:
        # [FUNCTION DOC] Return the pointwise size of the data.
        return self.data.get_pointwise_size()

    def get_reduction_size(self) -> Sequence[Expr]:
        # [FUNCTION DOC] Return the reduction size of the data.
        return self.data.get_reduction_size()

    def get_reduction_type(self) -> str | None:
        # [FUNCTION DOC] Return the reduction type of the data.
        return self.data.get_reduction_type()

    def is_no_op(self) -> bool:
        # [FUNCTION DOC] Return True if the data has statically zero elements.
        return self.data.is_zero_elements()

    def should_allocate(self) -> bool:
        # [FUNCTION DOC] Return True: a computed buffer needs allocation.
        return True

    def constant_to_device(self, device: torch.device) -> IRNode:
        # [FUNCTION DOC] Move the computation to another device via the data node; valid only if all reads are constants.
        """Move this to a given device. Requires that all reads are to constants."""
        return self.data.constant_to_device(device)


@dataclasses.dataclass(frozen=True)
class FinalizeCodegenResult:
    # [CLASS DOC] Frozen record returned by TemplateBuffer finalize-codegen hooks for external backends: generated source text, imports, call preamble lines and call argument strings.
    """Structured result from TemplateBuffer._finalize_codegen for external backends."""

    source: str
    imports: list[str]
    call_preamble: list[str]
    call_args: list[str]


class _HasAliasingOrMutation(Protocol):
    # [CLASS DOC] HELPER: Protocol describing the one method of scheduler nodes that prologue fusion needs (has_aliasing_or_mutation, declared as a one-line stub), used to avoid a circular import with the scheduler.
    """Minimal view of scheduler.BaseSchedulerNode used by prologue fusion.

    ir.py cannot import scheduler (circular), so this documents the single
    method consumed by has_aliasing_or_mutation_for_prologue_fusion instead
    of typing the argument as Any.
    """

    def has_aliasing_or_mutation(self) -> bool: ...


class TemplateBuffer(OperationBuffer):
    # [CLASS DOC] Base class for template operators that support epilogue and prologue fusion (subclasses include Triton and Helion template buffers). Holds the unwrapped inputs, a kernel render factory, names of fusable epilogue outputs and prologue inputs, multi-output children, named inputs, mutated inputs (with MutationOutput nodes), and per-template fusion overrides. Registers itself as a buffer and operation with the graph on creation.
    """
    Base class for template operators that support epilogue and prologue fusion.
    Subclasses: TritonTemplateBuffer (built-in Triton templates),
    HelionTemplateBuffer (Helion kernels), etc.
    """

    def __init__(
        self,
        layout: OutputSpec,
        inputs: Sequence[IRNode],
        make_kernel_render: Callable[..., Any] | None,
        mutated_inputs: Iterable[IRNode] | None = None,
        allowed_prologue_inps: OrderedSet[str] | None = None,
        named_inputs: dict[str, IRNode] | None = None,
    ) -> None:
        # [FUNCTION DOC] Create the template buffer with a layout and inputs (unwrapped to storage), register it as a buffer and operation in the graph, and initialize annotations, epilogue-fusable outputs, multi-output children, named inputs, mutated inputs and their MutationOutput nodes (requires the first input to be an IRNode, AssertionError otherwise), allowed prologue inputs and fusion overrides (None meaning follow global config).
        super().__init__(name=None, layout=layout)
        self.inputs = InputsKernel.unwrap_storage(inputs)
        self.make_kernel_render = make_kernel_render
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)
        # Annotations dict for storing metadata (e.g., KernelTemplateChoice)
        self.annotations: dict[str, object] = {}

        # Output buffer names eligible for epilogue fusion.
        # Maps buffer name → kernel parameter name (e.g. "buf3" → "result").
        self.epilogue_fusable_outputs: dict[str, str] = {}
        # For multi-output kernels: maps child buffer name → MultiOutput
        # node.  Used by call_kernel to emit tuple-unpacking lines.
        self._multi_output_children: dict[str, MultiOutput] = {}
        # Maps kernel parameter name → IRNode for each tensor input.
        # Used by ExternalTritonTemplateKernel to set up prologue fusion and
        # by HelionTemplateBuffer to resolve call arguments.
        self._named_inputs: dict[str, IRNode] = (
            dict(named_inputs) if named_inputs else {}
        )

        # Inputs that the kernel mutates in-place
        self.mutated_inputs = mutated_inputs
        self.mutation_outputs: list[MutationOutput] = []
        if mutated_inputs is not None:
            first_input = self.inputs[0]
            if not isinstance(first_input, IRNode):
                raise AssertionError(type(first_input))
            device = first_input.get_device()
            self.mutation_outputs = [
                MutationOutput(NoneLayout(device=device), buf, self)
                for buf in mutated_inputs
            ]
        # Input buffer names eligible for prologue fusion.
        self.allowed_prologue_inps: OrderedSet[str] = (
            allowed_prologue_inps or OrderedSet()
        )
        # Per-template fusion overrides.  None means fall back to global
        # config.epilogue_fusion / config.prologue_fusion.
        self.allow_epilogue_fusion: bool | None = None
        self.allow_prologue_fusion: bool | None = None

    @property
    def dtype(self) -> torch.dtype:
        # [FUNCTION DOC] Property: dtype from the layout; raises NotImplementedError for multi-output layouts.
        if isinstance(self.layout, MultiOutputLayout):
            raise NotImplementedError(
                "Multi-output templates do not have a single dtype"
            )
        return self.get_layout().dtype

    def get_read_writes(self) -> dependencies.ReadWrites:
        # [FUNCTION DOC] Return normalized read/write dependencies via extract_read_writes.
        return self.extract_read_writes(normalize=True)

    def _read_deps_from_inputs(self, normalize: bool) -> OrderedSet[dependencies.Dep]:
        # [FUNCTION DOC] HELPER: Build read dependencies from every input (each must be a ReinterpretView or Buffer with a Layout, else AssertionError) by tracing a dummy load over its size.
        """Build read dependencies from all inputs."""
        reads: OrderedSet[dependencies.Dep] = OrderedSet()
        for inp_raw in self.inputs:
            if not isinstance(inp_raw, (ReinterpretView, Buffer)):
                raise AssertionError(type(inp_raw))
            inp: ReinterpretView | Buffer = inp_raw
            if not isinstance(inp.layout, Layout):
                raise AssertionError(type(inp.layout))
            inp_indexer = inp.layout.make_indexer()

            def dummy(index: Sequence[Any], rindex: Sequence[Any]) -> Any:
                # [FUNCTION DOC] HELPER: Dummy body that loads the input at the indexed position; requires no reduction index.
                if len(rindex) != 0:
                    raise AssertionError("Expected len(rindex) == 0")
                return ops.load(inp.get_name(), inp_indexer(index))

            reads |= dependencies.extract_read_writes(
                dummy, inp.get_size(), (), normalize=normalize
            ).reads
        return reads

    def extract_read_writes(self, normalize: bool = False) -> dependencies.ReadWrites:
        # [FUNCTION DOC] Extract read/write dependencies. For multi-output layouts, synthesizes a single trivial write dependency and reads from the inputs. Otherwise traces a dummy store with the layout indexer over the size and adds the input reads. normalize controls dependency normalization.
        """Extract read/write dependencies for this TemplateBuffer.

        When the layout is MultiOutputLayout (multi-output templates), the
        buffer itself has no data layout, so we cannot build an indexer.
        Instead, synthesize a trivial write dep and derive read deps from
        the named tensor inputs (``_named_inputs``).  For single-output
        templates with a concrete layout, fall through to the standard path.
        """
        if isinstance(self.layout, MultiOutputLayout):
            writes: OrderedSet[dependencies.Dep] = OrderedSet(
                [
                    dependencies.MemoryDep(
                        self.get_name(), sympy.Integer(0), var_names=(), size=()
                    ),
                ]
            )
            return dependencies.ReadWrites(
                reads=self._read_deps_from_inputs(normalize),
                writes=writes,
                index_exprs=OrderedSet(),
                range_vars=None,
                var_ranges=None,
            )

        name = self.get_name()
        indexer = self.get_layout().make_indexer()

        def dummy(index: Sequence[Any], rindex: Sequence[Any]) -> Any:
            # [FUNCTION DOC] HELPER: Dummy body that stores a placeholder at the indexed position; requires no reduction index.
            if len(rindex) != 0:
                raise AssertionError("Expected len(rindex) == 0")
            return ops.store(name, indexer(index), "fake")

        deps = dependencies.extract_read_writes(
            dummy, self.get_size(), (), normalize=normalize
        )
        deps.reads |= self._read_deps_from_inputs(normalize)
        return deps

    def get_reduction_size(self) -> Sequence[Expr]:
        # [FUNCTION DOC] Return the reduction size. KNOWN ISSUE: returns the sympy constant one rather than a sequence, despite the annotation.
        return sympy.S.One

    def get_reduction_type(self) -> str | None:
        # [FUNCTION DOC] Return None; templates are not reductions.
        return None

    def should_allocate(self) -> bool:
        # [FUNCTION DOC] Return True: templates allocate their output.
        return True

    def simplify_and_reorder(
        self,
        extra_indexing_constraints: ExtraIndexingConstraints | None = None,
        recompute_sizes_body_func: Callable[..., Any] | None = None,
    ) -> tuple[tuple[Sequence[Expr], list[Expr]], LoopBody | None]:
        # [FUNCTION DOC] Return the size and an empty reduction list without loop transformation, and no loop body; arguments are ignored.
        return (
            (
                self.get_size(),
                [],
            ),
            None,
        )

    def is_multi_outputs_template(self) -> bool:
        # [FUNCTION DOC] Return True if the layout is a MultiOutputLayout.
        """Whether this template produces multiple outputs via MultiOutputLayout."""
        return isinstance(self.layout, MultiOutputLayout)

    def get_allowed_prologue_inps(self) -> OrderedSet[str]:
        # [FUNCTION DOC] Return the names of inputs allowed for prologue fusion.
        return self.allowed_prologue_inps

    def has_aliasing_or_mutation_for_prologue_fusion(
        self, scheduler_node: _HasAliasingOrMutation
    ) -> bool:
        # [FUNCTION DOC] Return whether aliasing or mutation blocks prologue fusion; by default defers to the scheduler node. Subclasses may override.
        """Return whether this template's aliasing/mutation blocks prologue fusion.

        The default preserves the scheduler's conservative behavior. External
        template subclasses may override this when they can prove a prologue
        producer only feeds independent, non-mutated template inputs.
        """
        return scheduler_node.has_aliasing_or_mutation()

    def _finalize_codegen(
        self, hook_outputs: dict[str, str]
    ) -> FinalizeCodegenResult | None:
        # [FUNCTION DOC] Hook called after epilogue/prologue codegen with rendered placeholder-to-code outputs; returns a FinalizeCodegenResult for custom source or None to keep the default path. Default returns None.
        """Called after epilogue/prologue subgraph codegen with rendered hook outputs.

        ``hook_outputs`` maps placeholder keys (e.g. ``<STORE_OUTPUT_0>``,
        ``<LOAD_INPUT_x>``) to the Triton code generated by Inductor for
        each fused subgraph.

        Return a ``FinalizeCodegenResult`` to provide custom source code and
        call metadata, or ``None`` to use the default codegen path.
        """
        return None

    @classmethod
    def realize_template_input(cls, tb: TensorBox) -> IRNode:
        # [FUNCTION DOC] Class method realizing a TensorBox as a template input: MultiOutput data is returned as is; otherwise the input is realized, unwrapped from its storage box and any flexible layout is frozen. Returns the IR node.
        """Realize a TensorBox, preserving MultiOutput layout (unlike ExternKernel.realize_input)."""
        if isinstance(tb, TensorBox) and isinstance(tb.data, MultiOutput):
            return tb.data
        result = ExternKernel.realize_input(tb)
        if isinstance(result, StorageBox):
            result = result.data
        if isinstance(result.layout, FlexibleLayout):  # type: ignore[union-attr]
            result.freeze_layout()
        return result

    @classmethod
    def build_multi_outputs(
        cls,
        template_buf: TemplateBuffer,
        structured: object,
        *,
        direct_alias_at_leaf: dict[int, IRNode] | None = None,
        on_tensor_leaf: Callable[[str, MultiOutput, list[tuple[type, int]], int], None]
        | None = None,
        on_non_tensor_leaf: Callable[[int], None] | None = None,
    ) -> tuple[TensorBox, ...]:
        # [FUNCTION DOC] Class method walking a nested list/tuple output structure and creating a MultiOutput node for each distinct tensor leaf (registered as a child of the template buffer, deduplicated by tensor identity, or aliased to a given direct alias node). Optional callbacks are invoked for tensor and non-tensor leaves. Returns a tuple of TensorBoxes.
        """Walk a structured output tree, creating MultiOutput nodes for tensor leaves."""
        seen_outputs: dict[int, TensorBox] = {}
        leaf_counter = itertools.count()

        def walk(output: object, indices: list[tuple[type, int]]) -> list[TensorBox]:
            # [FUNCTION DOC] HELPER: Recursively traverse the structure tracking container type and index paths, creating or reusing TensorBoxes for tensor leaves and skipping other leaves.
            if isinstance(output, (list, tuple)):
                results: list[TensorBox] = []
                for i, item in enumerate(output):
                    results.extend(walk(item, [*indices, (type(output), i)]))
                return results
            leaf_idx = next(leaf_counter)
            if isinstance(output, torch.Tensor):
                if direct_alias_at_leaf and leaf_idx in direct_alias_at_leaf:
                    return [TensorBox.create(direct_alias_at_leaf[leaf_idx])]
                tid = id(output)
                if tid in seen_outputs:
                    return [seen_outputs[tid]]
                mo = MultiOutput(
                    FallbackKernel.tensor_to_layout(output), template_buf, indices
                )
                template_buf._multi_output_children[mo.get_name()] = mo
                if on_tensor_leaf is not None:
                    on_tensor_leaf(mo.get_name(), mo, indices, leaf_idx)
                tb = TensorBox(mo)
                seen_outputs[tid] = tb
                return [tb]
            # Non-tensor leaf (int, SymInt, None, etc.)
            if on_non_tensor_leaf is not None:
                on_non_tensor_leaf(leaf_idx)
            return []

        return tuple(walk(structured, []))


class TritonTemplateBuffer(TemplateBuffer):
    # [CLASS DOC] TemplateBuffer for built-in Triton templates. Its own name is the sole epilogue-fusable output. Supports multiple outputs by treating extra output buffers as mutated inputs. Keeps optional subgraph inputs/outputs whose free symbols are included in its own.
    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[IRNode],
        make_kernel_render: Callable[_P, _T] | None,
        mutated_inputs: Iterable[IRNode] | None = None,
        allowed_prologue_inps: OrderedSet[str] | None = None,
    ) -> None:
        # [FUNCTION DOC] Create via the TemplateBuffer initializer, then set the epilogue-fusable output to map its own name to itself and initialize subgraph inputs and outputs to None. Raises AssertionError if the name was not assigned.
        """
        NOTE:[TritonTemplates with multiple outputs]
        We want the ability for TritonTemplates to output multiple tensors. Triton
        kernels have no notion of outputs and this is done by creating tensors that
        are then mutated by the kernel. Currently our STORE_OUTPUT codegen doesn't
        support creating multinode outputs for triton templates.
        We work around this by creating an extra input buffer during the lowering
        and we mark them as mutated inputs.
        """
        super().__init__(
            layout,
            inputs,
            make_kernel_render,
            mutated_inputs=mutated_inputs,
            allowed_prologue_inps=allowed_prologue_inps,
        )
        if self.name is None:
            raise AssertionError("Expected self.name is not None")
        self.epilogue_fusable_outputs = {self.name: self.name}

        self.subgraph_inps: list[IRNode | Expr | None] | None = None
        self.subgraph_outs: list[IRNode | None] | None = None

    @cache_on_self_and_args("TritonTemplateBuffer")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return the free symbols of the parent plus those of the subgraph inputs (expressions or IR nodes) and outputs; entries must be None otherwise (AssertionError). Cached.
        res = super().get_free_symbol_uses(unbacked_only)
        subgraph_outs = self.subgraph_outs if self.subgraph_outs else []
        subgraph_inps = self.subgraph_inps if self.subgraph_inps else []

        for inp in subgraph_inps:
            if isinstance(inp, Expr):
                res.update(get_free_symbols(inp, unbacked_only))
            elif isinstance(inp, IRNode):
                res.update(inp.get_free_symbol_uses(unbacked_only))
            else:
                if inp is not None:
                    raise AssertionError("Expected inp is None")

        for out in subgraph_outs:
            if isinstance(out, IRNode):
                res.update(out.get_free_symbol_uses(unbacked_only))
            else:
                if out is not None:
                    raise AssertionError("Expected out is None")

        return res

    def get_outputs(self) -> list[Buffer]:
        # [FUNCTION DOC] Return this buffer followed by its mutation output nodes.
        return [self, *self.mutation_outputs]

    def __str__(self) -> str:
        # [FUNCTION DOC] Return a short string showing the layout.
        out = f"TritonTemplateBuffer(layout={self.layout})"
        return out


PrimitiveInfoType = int | float | bool | str | list[int | str | float | bool]


class ChoiceCaller:
    # [CLASS DOC] Abstract representation of one autotuning choice (template or extern kernel). Holds name, input nodes, layout, description, failed flag, a cudagraph-benchmark flag, annotations and slots for subgraph-based choices (graph module, decomposition and kwargs, config patches). During autotuning benchmark is called, and if the choice wins output_node builds the IR. Subclasses: TritonTemplateCaller, CUTLASSTemplateCaller and others.
    """
    Represents a possible choice used in autotune_process.py.
    During autotuning, self.benchmark() is first called to get benchmark result,
    and if this choice is selected, self.output_node() is called to get the output_node.

    Children classes: TritonTemplateCaller, CUTLASSTemplateCaller.
    """

    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        description: str,
    ) -> None:
        # [FUNCTION DOC] Store the name, input nodes, layout and description and initialize defaults (not failed, no cudagraph benchmarking, empty annotations and subgraph-related fields).
        super().__init__()
        self.name = name
        self.layout = layout
        self.input_nodes = input_nodes
        # An additional description used to describe the choice (useful for
        # knowing what autotuning is choosing)
        self.description = description
        self.failed: bool = False
        # When True, benchmark using CUDA graph capture/replay
        self._benchmark_with_cudagraphs: bool = False
        # A place to store annotations that can be read post benchmarking
        # Use this to shuttle information between ChoieCaller generation
        # and the end of benchmarking
        self.annotations: dict[str, object] = {}
        # Subclass-overridden attributes for subgraph-based choices
        self.gm: torch.fx.GraphModule | None = None
        self.decomposition: Callable[..., Any] | None = None
        self.decomposition_kwargs: dict[str, object] = {}
        self.config_patches: dict[str, Any] = {}

    def benchmark(self, *args: Any, out: torch.Tensor) -> float:
        # [FUNCTION DOC] Benchmark the choice's callable on the given arguments and return time: with CUDA graph capture if flagged, with the profiling-based timer if config.profile_bandwidth_with_do_bench_using_profiling is set, otherwise with the standard benchmarker writing into out.
        algo = self.to_callable()
        if self._benchmark_with_cudagraphs:
            return benchmarker.benchmark_gpu_with_cuda_graph(lambda: algo(*args))
        if config.profile_bandwidth_with_do_bench_using_profiling:
            return do_bench_using_profiling(lambda: algo(*args))  # type: ignore[arg-type]
        return benchmarker.benchmark(algo, args, {"out": out}, device=None)

    def call_name(self) -> str:
        # [FUNCTION DOC] Return the call name; subclasses must override (NotImplementedError).
        raise NotImplementedError

    def to_callable(self) -> Callable[..., Any]:
        # [FUNCTION DOC] Return a callable running the choice; subclasses must override (NotImplementedError).
        raise NotImplementedError

    def kernel_hash_key(self) -> str:
        # [FUNCTION DOC] Return the hash key of the underlying kernel; defaults to hash_key (no runtime params).
        """
        Hash key for the underlying kernel. By default, we assume there are no
        runtime params, so kernel hash key defaults to choice caller's hash key.
        """
        return self.hash_key()

    def hash_key(self) -> str:
        # [FUNCTION DOC] Return the choice's hash key; subclasses must override (NotImplementedError).
        raise NotImplementedError

    def output_node(self) -> TensorBox:
        # [FUNCTION DOC] Return the TensorBox for the chosen implementation; subclasses must override (NotImplementedError).
        raise NotImplementedError

    def info_dict(self) -> dict[str, PrimitiveInfoType | list[PrimitiveInfoType]]:
        # [FUNCTION DOC] Return an info mapping logged to the autotune log; empty by default.
        """Information returned here is logged to the autotune log file when that is enabled."""
        return {}

    def autoheuristic_id(self) -> str:
        # [FUNCTION DOC] Return the autoheuristic identifier; default is the string unsupported_choice.
        return "unsupported_choice"

    def mark_failed(self) -> None:
        # [FUNCTION DOC] Mark the choice as failed so it can be removed later.
        """
        Mark the choice as failed so that it can be
        removed later. Useful for when we decouple
        compilation and tuning.
        """
        self.failed = True


class TritonTemplateCallerBase(ChoiceCaller):
    # [CLASS DOC] ChoiceCaller base for Triton template choices; adds get_make_kernel_render (abstract).
    def get_make_kernel_render(self) -> Any:
        # [FUNCTION DOC] Return the kernel render factory; subclasses must override (NotImplementedError).
        raise NotImplementedError


_NVUniversalGemmCallerClass: type | None = None


def _is_output_plannable_nvgemm_choice(choice: ChoiceCaller) -> bool:
    # [FUNCTION DOC] HELPER: Return True if the choice is an NVUniversalGemmCaller; the class is imported lazily and cached in a module global, and False is returned if the import fails.
    global _NVUniversalGemmCallerClass
    if _NVUniversalGemmCallerClass is None:
        try:
            from torch._inductor.codegen.nv_universal_gemm import NVUniversalGemmCaller

            _NVUniversalGemmCallerClass = NVUniversalGemmCaller
        except ImportError:
            return False
    return isinstance(choice, _NVUniversalGemmCallerClass)


class MultiTemplateBuffer(TritonTemplateBuffer):
    # [CLASS DOC] Buffer with multiple backing implementation choices (Triton templates, extern kernels, NV gemm). During scheduling, with a possible epilogue, each choice is benchmarked with the epilogue to pick one; otherwise the fastest base choice is used. Caches timings per hint override, tracks which renderer is bound (render kind and caller) and whether all choices allow output planning.
    """
    Represents a Buffer with multiple backing implementation choices.

    Choices can be TritonTemplates or ExternKernels. During scheduling if there is a potential
    epilogue we will benchmark each of the choices with the epilogue to determine an implementation.
    Otherwise, the fastest base choice will be chosen.
    """

    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[IRNode],
        choice_timings_fn: Callable[[int | None], dict[ChoiceCaller, float]],
        unfiltered_choices: list[ChoiceCaller],
        allowed_prologue_inps: OrderedSet[str],
    ) -> None:
        # [FUNCTION DOC] Initialize via TritonTemplateBuffer without a kernel renderer; store the timing function, unfiltered choices, original inputs, an output-plannable flag (true if every choice is a Triton caller, NV gemm caller or extern kernel with out variant) and empty renderer/caller state.
        super().__init__(
            layout=layout,
            inputs=inputs,
            make_kernel_render=None,
            allowed_prologue_inps=allowed_prologue_inps,
        )
        self._choice_timings_fn = choice_timings_fn
        self._choice_timings: dict[int | None, dict[ChoiceCaller, float]] = {}
        self._choices: list[ChoiceCaller] = unfiltered_choices
        self.original_inputs = inputs
        self._output_plannable = all(
            isinstance(choice, TritonTemplateCallerBase)
            or _is_output_plannable_nvgemm_choice(choice)
            or (
                isinstance(choice, torch._inductor.select_algorithm.ExternKernelCaller)
                and choice.has_out_variant
            )
            for choice in unfiltered_choices
        )
        self._make_kernel_renders: dict[int | None, Any] = {}
        self._render_kind: str | None = None
        # Tracks the bound caller so the fusion-benchmark loop's per-iteration
        # swap is not silently overridden by re-selection from choice_timings().
        self._render_caller: ChoiceCaller | None = None

    @property
    def output_plannable(self) -> bool:
        # [FUNCTION DOC] Property: True when all choices are Triton templates, NVUniversalGemm callers or extern kernels with an out variant.
        """True when all choices are TritonTemplates, NVUniversalGemmCallers, or Extern Kernels with out variants."""
        return self._output_plannable

    @property
    def choices(self) -> list[ChoiceCaller]:
        # [FUNCTION DOC] Property: the list of choices.
        return self._choices

    def choice_timings(
        self, hint_override: int | None = None
    ) -> dict[ChoiceCaller, float]:
        # [FUNCTION DOC] Return the mapping from choice to benchmark time for the hint override, computing and caching it on first use.
        if hint_override not in self._choice_timings:
            self._choice_timings[hint_override] = self._choice_timings_fn(hint_override)
        return self._choice_timings[hint_override]

    @contextlib.contextmanager
    def swap_as_triton_caller(self, caller: TritonTemplateCallerBase) -> Iterator[None]:
        # [FUNCTION DOC] Context manager that binds a Triton template caller's kernel render to this buffer (kind triton) and restores the previous render, kind and caller on exit. Raises AssertionError if the caller is not a TritonTemplateCaller or its layout differs from this layout.
        if not isinstance(
            caller, torch._inductor.select_algorithm.TritonTemplateCaller
        ):
            raise AssertionError(type(caller))
        if self.layout != caller.layout:
            raise AssertionError("Expected self.layout == caller.layout")

        render = self.make_kernel_render
        prev_kind = self._render_kind
        prev_caller = self._render_caller
        self.make_kernel_render = caller.get_make_kernel_render()
        self._render_kind = "triton"
        self._render_caller = caller
        try:
            yield
        finally:
            self.make_kernel_render = render
            self._render_kind = prev_kind
            self._render_caller = prev_caller

    def finalize_as_triton_caller(self, caller: TritonTemplateCallerBase) -> None:
        # [FUNCTION DOC] Permanently bind a Triton template caller's kernel render to this buffer; AssertionError if the caller is the wrong type or its size or stride differs from this buffer.
        if not isinstance(
            caller, torch._inductor.select_algorithm.TritonTemplateCaller
        ):
            raise AssertionError(type(caller))
        if self.get_size() != caller.layout.size:
            raise AssertionError("Expected self.get_size() == caller.layout.size")
        if self.get_stride() != caller.layout.stride:
            raise AssertionError("Expected self.get_stride() == caller.layout.stride")
        self.make_kernel_render = caller.get_make_kernel_render()
        self._render_kind = "triton"
        self._render_caller = caller

    @contextlib.contextmanager
    def swap_as_nvgemm_caller(self, caller: ChoiceCaller) -> Iterator[None]:
        # [FUNCTION DOC] Context manager that binds an NVUniversalGemmCaller's kernel render (kind nvgemm) and restores the previous state on exit; AssertionError for the wrong type or layout mismatch.
        from torch._inductor.codegen.nv_universal_gemm import NVUniversalGemmCaller

        if not isinstance(caller, NVUniversalGemmCaller):
            raise AssertionError(f"expected NVUniversalGemmCaller, got {type(caller)}")
        if self.layout != caller.layout:
            raise AssertionError("Expected self.layout == caller.layout")

        render = self.make_kernel_render
        prev_kind = self._render_kind
        prev_caller = self._render_caller
        self.make_kernel_render = caller.get_make_kernel_render()
        self._render_kind = "nvgemm"
        self._render_caller = caller
        try:
            yield
        finally:
            self.make_kernel_render = render
            self._render_kind = prev_kind
            self._render_caller = prev_caller

    def finalize_as_nvgemm_caller(self, caller: ChoiceCaller) -> None:
        # [FUNCTION DOC] Permanently bind an NVUniversalGemmCaller's kernel render to this buffer; AssertionError for the wrong type or size/stride mismatch.
        from torch._inductor.codegen.nv_universal_gemm import NVUniversalGemmCaller

        if not isinstance(caller, NVUniversalGemmCaller):
            raise AssertionError(f"expected NVUniversalGemmCaller, got {type(caller)}")
        if self.get_size() != caller.layout.size:
            raise AssertionError("Expected self.get_size() == caller.layout.size")
        if self.get_stride() != caller.layout.stride:
            raise AssertionError("Expected self.get_stride() == caller.layout.stride")
        self.make_kernel_render = caller.get_make_kernel_render()
        self._render_kind = "nvgemm"
        self._render_caller = caller

    def get_min_choice(
        self, hint_override: int | None = None
    ) -> tuple[ChoiceCaller, float]:
        # [FUNCTION DOC] Return the fastest choice and its time for the hint override.
        timings = self.choice_timings(hint_override=hint_override)
        min_choice = min(timings, key=timings.get)  # type: ignore[arg-type]
        return (min_choice, timings[min_choice])

    def finalize_as_triton_callers(
        self, callers: dict[int | None, TritonTemplateCallerBase]
    ) -> None:
        # [FUNCTION DOC] Bind kernel renders for several hint overrides; the entry for no override becomes the default render and caller (kind triton).
        """Finalize with multiple callers for different hint overrides"""
        for hint_override, caller in callers.items():
            self._make_kernel_renders[hint_override] = caller.get_make_kernel_render()

        # Set the default to be the one without hint override
        self.make_kernel_render = self._make_kernel_renders[None]
        self._render_kind = "triton"
        self._render_caller = callers[None]


class CUTLASSTemplateBuffer(TemplateBuffer):
    # [CLASS DOC] TemplateBuffer for CUTLASS template kernels; records workspace size in bytes, the template and whether epilogue fusion is supported.
    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[IRNode],
        make_kernel_render: Callable[_P, _T],
        workspace_size: int,
        template: CUTLASSTemplate,
        supports_epilogue_fusion: bool,
    ) -> None:
        # [FUNCTION DOC] Initialize the template buffer and store the workspace size, template and epilogue-fusion support flag.
        super().__init__(layout, inputs, make_kernel_render)
        # Global memory (in bytes) needed for this template.
        self.workspace_size = workspace_size
        self.template = template
        self.supports_epilogue_fusion = supports_epilogue_fusion

    def get_workspace_size(self) -> int:
        # [FUNCTION DOC] Return the workspace size in bytes, zero if unset.
        return self.workspace_size if self.workspace_size is not None else 0

    def emulate_store_fn(self) -> None:
        # [FUNCTION DOC] Emit placeholder store ops for each output so dependency analysis sees the writes.
        for output in self.get_outputs():
            ops.store(output.get_name(), None, None)


class CppTemplateBuffer(TemplateBuffer):
    # [CLASS DOC] TemplateBuffer for C++ template kernels; stores the template, the chosen choice and an optional list of output buffers (used when the layout is multi-output).
    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[IRNode],
        make_kernel_render: Callable[_P, _T],
        template: CUTLASSTemplate,
        choice: Any,
    ) -> None:
        # [FUNCTION DOC] Initialize the template buffer and store the template and choice; outputs start as None.
        super().__init__(layout, inputs, make_kernel_render)
        self.template = template
        self.choice = choice
        self.outputs: list[Buffer] | None = None

    def get_layout(self) -> Layout:
        # [FUNCTION DOC] Return the layout; for multi-output layouts, the layout of the first output buffer (AssertionError if outputs are missing or not buffers/layouts), otherwise the standard layout.
        if isinstance(self.layout, MultiOutputLayout):
            if not isinstance(self.outputs, Iterable):
                raise AssertionError(type(self.outputs))

            first_output = self.outputs[0]
            if not isinstance(first_output, Buffer):
                raise AssertionError(type(first_output))
            layout = first_output.layout
            if not isinstance(layout, Layout):
                raise AssertionError(type(layout))
            return layout
        else:
            return super().get_layout()


class CuteDSLTemplateBuffer(TemplateBuffer):
    # [CLASS DOC] TemplateBuffer for CuteDSL (CUTLASS Python DSL) template kernels. Its outputs are itself followed by MutationOutput nodes for any mutated inputs.
    """
    Buffer for CuteDSL (CUTLASS Python DSL) template kernels.
    Similar to other template buffers but specialized for CuteDSL operations.
    """

    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[IRNode],
        make_kernel_render: Callable[_P, _T],
        template: Any,
        mutated_inputs: Iterable[IRNode] | None = None,
    ) -> None:
        # [FUNCTION DOC] Initialize the template buffer, record the template and mutated inputs and build the output list (self plus mutation outputs on the first input's device; AssertionError if the first input is not an IR node).
        super().__init__(layout, inputs, make_kernel_render)
        self.template = template
        self.mutated_inputs = mutated_inputs
        self.outputs: list[Buffer] = [self]

        if mutated_inputs is not None:
            if not isinstance(self.inputs[0], IRNode):
                raise AssertionError(type(self.inputs[0]))
            device = self.inputs[0].get_device()
            self.outputs += [
                MutationOutput(NoneLayout(device=device), buf, self)
                for buf in mutated_inputs
            ]

    def get_outputs(self) -> list[Buffer]:
        # [FUNCTION DOC] Return the list of output buffers.
        return self.outputs


class NVUniversalGemmBuffer(TemplateBuffer):
    # [CLASS DOC] TemplateBuffer for NVIDIA Universal GEMM kernels: generates simple Python code calling the cutlass operators library rather than using Jinja templates. Stores the kernel, accumulator type, variant, scale and swizzle types for operands, workspace size, epilogue-fusion support, operand swap flag and an optional bias node (last input, consumed by a fixed bias-add epilogue). Its kernel render is its own method.
    """
    Buffer for NVIDIA Universal GEMM kernels.

    Unlike CuteDSL templates which use Jinja templates, this generates
    simpler Python code that directly calls the cutlass.operators library.
    """

    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[IRNode],
        kernel: Any,
        accumulator_type: Any,
        variant: Any,  # GemmVariant, use Any to avoid circular import
        workspace_size: int = 0,
        scale_type_a: Any | None = None,
        scale_type_b: Any | None = None,
        swizzle_type_a: Any | None = None,
        swizzle_type_b: Any | None = None,
        supports_epilogue_fusion: bool = False,
        swap_ab: bool = False,
        bias_node: Buffer | None = None,
    ) -> None:
        # [FUNCTION DOC] Initialize the template buffer without a render function, store all gemm parameters and kernel metadata (operator name and minimum compute capability), and set the kernel render to the buffer's own method.
        # We pass None initially, then override with our method below
        super().__init__(layout, inputs, make_kernel_render=None)
        self.kernel = kernel
        self.accumulator_type = accumulator_type
        self.outputs: list[Buffer] = [self]
        self.workspace_size = workspace_size
        self.variant = variant
        self.scale_type_a = scale_type_a
        self.scale_type_b = scale_type_b
        self.swizzle_type_a = swizzle_type_a
        self.swizzle_type_b = swizzle_type_b
        self.supports_epilogue_fusion = supports_epilogue_fusion
        self.swap_ab = swap_ab
        # When set, the last entry of `inputs` is an addmm bias consumed as a
        # fixed bias-add epilogue; the GEMM operands are the remaining inputs.
        self.bias_node = bias_node
        # Store kernel metadata for code generation since kernels aren't serializeable yet
        self.kernel_metadata = {
            "kernel_name": kernel.metadata.operator_name,
            "min_cc": kernel.designed_for_min_cc,
        }
        # Override the instance attribute set by parent with our method
        # This is necessary because TemplateBuffer stores make_kernel_render as instance attr
        self.make_kernel_render = self._make_kernel_render

    def get_workspace_size(self) -> int:
        # [FUNCTION DOC] Return the workspace size in bytes.
        """Return the workspace size in bytes."""
        return self.workspace_size

    def get_outputs(self) -> list[Buffer]:
        # [FUNCTION DOC] Return the list of output buffers.
        return self.outputs

    def _make_kernel_render(
        self,
        out_node: Any,
        hint_override: int | None = None,
        epilogue_fn_code: str | None = None,
        epilogue_reads: list[str] | None = None,
        epilogue_writes: list[str] | None = None,
        epilogue_var_renames: dict[str, Any] | None = None,
        local_reduce: GemmReductionPlan | None = None,
    ) -> tuple[Any, Any]:
        # [FUNCTION DOC] Build the kernel renderer: unwraps tensor/storage boxes of the inputs, splits off the bias node if present, and creates an NVUniversalGemmKernel with metadata, accumulator type, variant, scale/swizzle types and optional epilogue (code, reads, writes, renames, local reduction plan). Returns (kernel object, render function). Raises AssertionError if epilogue code is given without variable renames.
        """
        Create a kernel renderer for code generation.

        Returns (kernel, render) tuple where:
        - kernel: NVUniversalGemmKernel object with call_kernel() method
        - render: function that returns source code string
        """
        if epilogue_fn_code is not None:
            if epilogue_var_renames is None:
                raise AssertionError("epilogue_fn_code requires epilogue_var_renames")

        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
            NVUniversalGemmKernel,
        )
        from torch._inductor.utils import Placeholder

        input_nodes: list[Any] = []
        for inp in self.inputs:
            if isinstance(inp, TensorBox):
                inp = inp.data
            if isinstance(inp, StorageBox):
                inp = inp.data
            input_nodes.append(inp)

        # For a baked addmm bias, the bias is the last input and is consumed by
        # the epilogue, not as a GEMM operand.
        bias_node = None
        if self.bias_node is not None:
            bias_node = input_nodes[-1]
            input_nodes = input_nodes[:-1]

        kernel_name = str(Placeholder.KERNEL_NAME)

        render_kernel = NVUniversalGemmKernel(
            kernel_name=kernel_name,
            input_nodes=input_nodes,
            output_node=out_node,
            kernel_metadata=self.kernel_metadata,
            accumulator_type=self.accumulator_type,
            workspace_size=self.workspace_size,
            variant=self.variant,
            scale_type_a=self.scale_type_a,
            scale_type_b=self.scale_type_b,
            swizzle_type_a=self.swizzle_type_a,
            swizzle_type_b=self.swizzle_type_b,
            epilogue_fn_code=epilogue_fn_code,
            epilogue_reads=epilogue_reads,
            epilogue_writes=epilogue_writes,
            epilogue_var_renames=epilogue_var_renames,
            local_reduce=local_reduce,
            swap_ab=self.swap_ab,
            bias_node=bias_node,
        )

        def render():
            # [FUNCTION DOC] HELPER: Render the kernel source.
            return render_kernel.render()

        return render_kernel, render


def is_node_sequence(
    nodes: Sequence[IRNode | Sequence[IRNode]],
) -> TypeIs[Sequence[IRNode]]:
    # [FUNCTION DOC] Return True if every element is an IRNode (type narrowing to a flat sequence of nodes).
    return all(isinstance(n, IRNode) for n in nodes)


@ir_dataclass(frozen=False)
class InputsKernel(OperationBuffer):
    # [CLASS DOC] OperationBuffer that reads an explicit list of inputs (IR nodes or lists of nodes) with star dependencies, i.e. whole-buffer reads and writes. Counts as an extern operation with one read. Provides unwrapping of inputs to underlying buffers or reinterpret views.
    inputs: Sequence[IRNode | Sequence[IRNode]]

    def input_name(self, i: int) -> str:
        # [FUNCTION DOC] Return the name of the i-th input; AssertionError if it is a list of nodes.
        input = self.inputs[i]
        if not isinstance(input, IRNode):
            raise AssertionError("Expected isinstance(input, IRNode)")
        return input.get_name()

    def get_read_writes(self) -> dependencies.ReadWrites:
        # [FUNCTION DOC] Return ReadWrites with star dependencies on every input buffer (symbolic shape constants skipped) and on every output; no index expressions.
        reads = OrderedSet[dependencies.Dep]()
        StarDep = dependencies.StarDep
        for input in self.inputs:
            if isinstance(input, Sequence):
                reads.update(StarDep(x.get_name()) for x in input)
            elif isinstance(input, ShapeAsConstantBuffer):
                # Skip creating dependency for symbolics as they're visible globally
                continue
            else:
                reads.add(StarDep(input.get_name()))

        writes = OrderedSet[dependencies.Dep](
            StarDep(buf.get_name()) for buf in self.get_outputs()
        )

        return dependencies.ReadWrites(
            reads=reads,
            writes=writes,
            index_exprs=OrderedSet(),
        )

    def get_reads(self) -> OrderedSet[Dep]:
        # [FUNCTION DOC] Return the read dependencies from the read/writes.
        return self.get_read_writes().reads

    @classmethod
    def unwrap_storage_for_input(cls, x: IRNode) -> IRNode:
        # [FUNCTION DOC] Class method unwrapping an input to a Buffer or ReinterpretView (or a torch bind object): strips TensorBox and StorageBox, realizes non-reinterpret views, recursing if the result is boxed again. Raises AssertionError for other types.
        if isinstance(x, TensorBox):
            x = x.data
        if isinstance(x, StorageBox):
            x = x.data
        if isinstance(x, BaseView) and not isinstance(x, ReinterpretView):
            x = ExternKernel.realize_input(x)
        if isinstance(x, TensorBox):
            # when converting to ReinterpretView fails in the
            # realize_input call above, the result will be wrapped
            # into TensorBox / StorageBox pair as a result of the
            # cls.copy_input call; so we should unwrap recursively
            return cls.unwrap_storage_for_input(x)
        if isinstance(x, TorchBindObject):
            return x
        if not isinstance(x, (Buffer, ReinterpretView)):
            raise AssertionError(type(x))
        return x

    @staticmethod
    def unwrap_storage(
        inputs: Sequence[IRNode | Sequence[IRNode]],
    ) -> list[IRNode | Sequence[IRNode]]:
        # [FUNCTION DOC] Static method unwrapping each input, or each element of nested lists of inputs, with unwrap_storage_for_input; returns a new list.
        inputs_new: list[IRNode | Sequence[IRNode]] = []
        for x in inputs:
            if isinstance(x, Sequence):
                x = [InputsKernel.unwrap_storage_for_input(i) for i in x]
            else:
                x = InputsKernel.unwrap_storage_for_input(x)
            inputs_new.append(x)
        return inputs_new

    def is_extern(self) -> bool:
        # [FUNCTION DOC] Return True.
        return True

    def num_reads(self) -> int:
        # [FUNCTION DOC] Return 1.
        return 1

    @cache_on_self_and_args("InputsKernel")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return the union of free symbols over all inputs, including nested lists (cached).
        r = OrderedSet[sympy.Symbol]()
        for inp in self.inputs:
            if isinstance(inp, IRNode):
                r |= inp.get_free_symbol_uses(unbacked_only)
            else:
                for inner_inp in inp:
                    r |= inner_inp.get_free_symbol_uses(unbacked_only)
        return r


class NopKernel(InputsKernel):
    # [CLASS DOC] InputsKernel that generates no code and reads nothing (a no-op).
    def is_no_op(self) -> bool:
        # [FUNCTION DOC] Return True.
        return True

    def get_reads(self) -> OrderedSet[Dep]:
        # [FUNCTION DOC] Return an empty set of reads.
        return OrderedSet()


class ConcatKernel(NopKernel):
    # [CLASS DOC] NopKernel for concatenation: no real kernel exists; the inputs are made to write into slices of one new storage by changing their storage/layout.
    """
    There isn't actually a real kernel for concat, we just change the
    storage for the upstream data.
    """

    @classmethod
    def create(cls, inputs: Sequence[IRNode], dim: int) -> StorageBox:
        # [FUNCTION DOC] Class method concatenating inputs along dim and returning a StorageBox over a new ConcatKernel. Validates dim and that dtype, device and all other dims agree (AssertionError otherwise). Output strides are contiguous (padded when comprehensive padding is on), switching to channels-last if any fixed input layout is channels-last or, for inputs without storage layouts, if fx meta values are channels-last. The output layout is fixed (pinned only if all inputs are). Each input is realized into a slice view of the output (via realize_into with clamping off); on GPU, plain input-buffer operations are grouped for foreach when supported. Registers the buffer and operation with the graph.
        """
        Create the concat kernel from inputs
        """
        device = inputs[0].get_device()
        dtype = inputs[0].get_dtype()
        new_size = list(inputs[0].get_size())
        offsets_start = [0]
        offsets_end = [new_size[dim]]
        if not (0 <= dim < len(new_size)):
            raise AssertionError("Expected 0 <= dim < len(new_size)")
        for i in range(1, len(inputs)):
            input_size = inputs[i].get_size()
            offsets_start.append(new_size[dim])
            if len(input_size) != len(new_size):
                raise AssertionError("Expected len(input_size) == len(new_size)")
            if inputs[i].get_dtype() != dtype:
                raise AssertionError("Expected inputs[i].get_dtype() == dtype")
            if inputs[i].get_device() != device:
                raise AssertionError("Expected inputs[i].get_device() == device")
            for j in range(len(new_size)):
                if j == dim:
                    new_size[j] = new_size[j] + input_size[j]
                else:
                    new_size[j] = V.graph.sizevars.check_equals_and_simplify(
                        new_size[j], input_size[j]
                    )
            offsets_end.append(new_size[dim])

        output_stride: Sequence[int] = FlexibleLayout.contiguous_strides(new_size)
        if config.comprehensive_padding:
            # Ensure the output stride matches the alignment requirements
            output_stride = Layout._pad_strides(
                output_stride, new_size, inputs[0].dtype
            )

        # If any of the inputs is in CL format, use CL format for the output
        for i in range(len(inputs)):
            x = inputs[i]
            if is_storage_and_layout(x):
                layout = x.get_layout()
                if isinstance(
                    layout, FixedLayout
                ) and Layout.is_channels_last_contiguous(layout.size, layout.stride):
                    # use CL stride for the output
                    output_stride = make_channels_last_strides_for(new_size)
                    break
        any_input_is_storage_and_layout = any(is_storage_and_layout(x) for x in inputs)
        fx_node_args = V.graph.current_node.args[0]
        # If any of the inputs has meta tensor and the meta tensor is in CL format, use CL format for the output
        # Skip this check when fx_node_args is not a list (e.g., called from _pad_as_cat).
        if (
            any_input_is_storage_and_layout is False
            and isinstance(fx_node_args, list)
            and any(
                # pyrefly: ignore [missing-attribute]
                "val" in arg.meta
                and (
                    is_contiguous_for_memory_format_or_false(
                        # pyrefly: ignore [missing-attribute]
                        arg.meta["val"],
                        memory_format=torch.channels_last,
                    )
                    or is_contiguous_for_memory_format_or_false(
                        # pyrefly: ignore [missing-attribute]
                        arg.meta["val"],
                        memory_format=torch.channels_last_3d,
                    )
                )
                for arg in fx_node_args
            )
        ):
            output_stride = make_channels_last_strides_for(new_size)

        is_pinned = all(
            is_storage_and_layout(x) and x.get_layout().is_pinned for x in inputs
        )

        if device is None:
            raise AssertionError("Expected device is not None")
        concat_kernel = ConcatKernel(
            name=None,
            layout=FixedLayout(
                device=device,
                dtype=dtype,
                size=new_size,
                stride=output_stride,
                is_pinned=is_pinned,
            ),
            inputs=[],
        )
        kernel = StorageBox(concat_kernel)
        op_names = []
        for i, inp in enumerate(inputs):
            if not isinstance(inp, (BaseView, MutableBox)):
                raise AssertionError(type(inp))
            input_buffer = cls.realize_into(
                inp,
                SliceView.create(
                    kernel, dim, offsets_start[i], offsets_end[i], clamp=False
                ),
            )
            if not isinstance(input_buffer, Buffer):
                raise AssertionError(type(input_buffer))
            if not isinstance(concat_kernel.inputs, list):
                raise AssertionError(type(concat_kernel.inputs))
            concat_kernel.inputs.append(input_buffer)

            if isinstance(inp.data, BaseView):
                input_unwrapped = inp.data.unwrap_view()
            else:
                input_unwrapped = inp.data

            if (
                isinstance(input_unwrapped, StorageBox)
                and input_unwrapped.is_input_buffer()
                and (dev := inp.get_device()) is not None
                and is_gpu(dev.type)
                and not is_dynamic(input_buffer)
            ):
                op_names.append(input_buffer.get_operation_name())

        if len(op_names) > 1 and V.graph.has_feature(device, BackendFeature.FOREACH):
            V.graph.register_operation_list(op_names)

        concat_kernel.name = V.graph.register_buffer(concat_kernel)
        concat_kernel.inputs = cls.unwrap_storage(concat_kernel.inputs)
        V.graph.register_operation(concat_kernel)

        return kernel

    @classmethod
    def can_realize_into_without_copy(
        cls, src: IRNode, dst: IRNode | None = None
    ) -> bool:
        # [FUNCTION DOC] Class method deciding if src can write directly into dst without a copy. For a MultiTemplateBuffer this needs a fixed, output-plannable layout and (if dst is given) statically equal strides; otherwise src must have a flexible layout that is not an ExternKernelAlloc. Raises AssertionError if src is not a view or storage box.
        if isinstance(src, TensorBox):
            # unwrap a TensorBox
            return cls.can_realize_into_without_copy(src.data, dst)

        if not isinstance(src, (BaseView, StorageBox)):
            raise AssertionError(type(src))
        if isinstance(src.data, MultiTemplateBuffer):
            if (
                not isinstance(src.data.layout, FixedLayout)
                or not src.data.output_plannable
            ):
                return False

            # we call can_realize_into_without_copy in cat lowering before we've decided
            # on output format, optimistically assume layout matches
            if dst is None:
                return True

            # otherwise, check equality of layouts
            if len(src.get_stride()) != len(dst.get_stride()):
                return False

            return all(
                V.graph.sizevars.statically_known_equals(s1, s2)
                for s1, s2 in zip(src.get_stride(), dst.get_stride())
            )

        return (
            hasattr(src.data, "layout")
            and isinstance(src.data.layout, FlexibleLayout)
            and not isinstance(src.data, ExternKernelAlloc)
        )

    @cache_on_self_and_args("ConcatKernel")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return the free symbols as computed by NopKernel (cached).
        return NopKernel.get_free_symbol_uses(self, unbacked_only)

    @classmethod
    def realize_into(cls, src: IRNode, dst: IRNode) -> IRNode:
        # [FUNCTION DOC] Class method making src write into the dst slice. dst is converted to a ReinterpretView if possible (AssertionError otherwise). TensorBoxes are unwrapped; for a StorageBox that can realize without copy, its layout becomes a non-owning layout over dst and its buffer is returned. Otherwise a pointwise copy of src is made (sizes checked equal) and realized into dst. Mutates the layout of src.
        # Attempt to turn this into a ReinterpretView rather than assert.
        # This has concessions around layout, as as_storage_and_layout
        # can cause us to go from flexible to fixed layout.
        if not isinstance(dst, ReinterpretView):
            if is_storage_and_layout(dst):
                storage, layout = as_storage_and_layout(dst)
                dst = ReinterpretView(data=storage, layout=layout)
        if not isinstance(dst, ReinterpretView):
            raise AssertionError(type(dst))
        if isinstance(src, TensorBox):
            # unwrap a TensorBox
            return cls.realize_into(src.data, dst)

        if isinstance(src, StorageBox):
            src.realize()
            # ExternKernelAlloc has specific requirements for output layout, should create a copy
            if not hasattr(src.data, "layout"):
                raise AssertionError('Expected hasattr(src.data, "layout")')
            if cls.can_realize_into_without_copy(src, dst):
                # pyrefly: ignore [missing-attribute]
                src.data.layout = NonOwningLayout(dst)
                return src.data
        # introduce a copy
        pw = Pointwise.create(
            device=src.get_device(),
            dtype=src.get_dtype(),
            inner_fn=src.make_loader(),
            ranges=[
                V.graph.sizevars.check_equals_and_simplify(a, b)
                for a, b in zip(src.get_size(), dst.get_size())
            ],
        )
        return cls.realize_into(pw, dst)

    def should_allocate(self) -> bool:
        # [FUNCTION DOC] Return True: the concat output needs an allocation.
        return True


@contextlib.contextmanager
def _track_fresh_unbacked_symbols(shape_env: ShapeEnv) -> Iterator[None]:
    # [FUNCTION DOC] HELPER: Context manager that enables tracking of fresh unbacked symbols on the shape env for its body and restores the previous setting.
    prev = shape_env._ignore_fresh_unbacked_symbols_set(False)
    try:
        yield
    finally:
        shape_env._ignore_fresh_unbacked_symbols_set(prev)


def _fallback_kernel_symbol_tracking_context(
    shape_env: ShapeEnv | None,
) -> AbstractContextManager[None]:
    # [FUNCTION DOC] HELPER: Return a tracking context manager when the shape env currently ignores fresh unbacked symbols (per thread), otherwise a null context.
    if shape_env is not None and shape_env._ignore_fresh_unbacked_symbols_tls():
        return _track_fresh_unbacked_symbols(shape_env)
    return nullcontext()


@dataclasses.dataclass(frozen=True)
class ProcessKernelResult:
    # [CLASS DOC] Frozen record returned by ExternKernel.process_kernel: example_output (result of running the kernel on example inputs), tensor_args (realized IR inputs), non_tensor_args, unflatten_args (callable rebuilding the original args and kwargs from replacement lists) and unbacked_bindings (symbol to key path map or None).
    """Structured result of ExternKernel.process_kernel.

    Replaces the positional 5-tuple that was unpacked at every call site.
    unflatten_args(new_tensor_args, new_non_tensor_args) reconstructs the
    original (args, kwargs) tree from replacement lists.
    """

    example_output: Any
    tensor_args: list[IRNode]
    non_tensor_args: list[object]
    unflatten_args: Callable[[Any, Any], Any]
    unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] | None


@ir_dataclass(frozen=False)
class ExternKernel(InputsKernel):
    # [CLASS DOC] OperationBuffer for kernels not lowered to loop-level IR (custom ops, fallback aten ops). Fields: constant_args, kwargs, output_view, python and cpp kernel names, ordered kwargs for the cpp kernel, op_overload, argument and keyword properties from the schema, unbacked bindings, mutation outputs and the fx node. Provides kernel-name resolution, input realization and copying, layout constraints, and codegen hooks (codegen is abstract).
    """
    A class that represents Kernels which are not directly lowered to Inductor
    Loop Level IR, such as custom operators, or aten operators which we fallback to.
    """

    constant_args: Sequence[Any] = ()
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    output_view: ReinterpretView | None = None
    python_kernel_name: str | None = None
    cpp_kernel_name: str | None = None
    # FIXME: in some cases we sill need to explicitly pass in ordered_kwargs_for_cpp_kernel
    # We shouldn't need to do this since the information can be retrieved from op_overload._schema.
    ordered_kwargs_for_cpp_kernel: Iterable[str] = dataclasses.field(
        default_factory=list
    )
    op_overload: _OpOverloads | None = None
    arg_properties: list[ArgProperty] | None = None
    allarg_properties: dict[str, ArgProperty] = dataclasses.field(default_factory=dict)
    kwarg_properties: dict[str, ArgProperty] | None = None
    unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] = dataclasses.field(
        default_factory=dict
    )
    mutation_outputs: list[MutationOutput] = dataclasses.field(default_factory=list)

    def __init__(
        self,
        name: str | None,
        layout: OutputSpec,
        inputs: Sequence[IRNode | Sequence[IRNode]],
        constant_args: Sequence[Any] = (),
        kwargs: dict[str, Any] | None = None,
        output_view: ReinterpretView | None = None,
        python_kernel_name: str | None = None,
        cpp_kernel_name: str | None = None,
        ordered_kwargs_for_cpp_kernel: Iterable[str] = (),
        op_overload: _OpOverloads | None = None,
    ) -> None:
        # [FUNCTION DOC] Initialize the extern kernel: store constant args, kwargs, output view, op overload, kernel names (derived from the overload when not given), ordered cpp kwargs, schema-based argument properties, empty unbacked bindings and mutation outputs, the current fx node and an annotations mapping.
        super().__init__(
            name=name,
            layout=layout,
            inputs=inputs,
        )
        self.constant_args = constant_args
        self.kwargs = kwargs if kwargs else {}
        self.output_view = output_view
        self.op_overload = op_overload
        self.set_cpp_kernel_name(cpp_kernel_name)
        self.set_python_kernel_name(python_kernel_name)
        self.ordered_kwargs_for_cpp_kernel = ordered_kwargs_for_cpp_kernel
        self.collect_arg_kwarg_properties()
        self.unbacked_bindings = {}
        self.mutation_outputs = []
        self.fx_node = V.graph.current_node
        # Annotations dict for storing metadata (e.g., KernelTemplateChoice)
        self.annotations: dict[str, object] = {}

    def get_outputs(self) -> list[Buffer]:
        # [FUNCTION DOC] Return this buffer followed by its mutation outputs.
        return [self, *self.mutation_outputs]

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return an empty set of defined unbacked symbols.
        return OrderedSet()

    def get_read_writes(self) -> dependencies.ReadWrites:
        # [FUNCTION DOC] Return the input-based read/writes plus star dependencies on any IR nodes found in constant args and kwargs.
        read_writes = super().get_read_writes()

        def add_ir_read(value: Any) -> None:
            # [FUNCTION DOC] HELPER: Add a star read dependency for an IR node value that has a name.
            if isinstance(value, IRNode):
                name = value.maybe_get_name()
                if name is not None:
                    read_writes.reads.add(dependencies.StarDep(name))

        pytree.tree_map_(
            add_ir_read,
            (self.constant_args, self.kwargs),
            is_leaf=lambda value: isinstance(value, IRNode),
        )
        return read_writes

    def collect_arg_kwarg_properties(self) -> None:
        # [FUNCTION DOC] Fill argument property tables from the op overload schema when it is an OpOverload (names, types, defaults for positional, all and keyword-only arguments; default ordered cpp kwargs from keyword-only names), otherwise empty property dicts.
        # if self.op_overload is torch._ops.OpOverload, we can use its schema to collect additional
        # information for args and kwargs, e.g. type and default value, to help with the cpp wrapper codegen
        self.arg_properties = (
            [
                {
                    "name": x.name,
                    "type": x.real_type,
                    "default_value": x.default_value,
                }
                for x in self.op_overload._schema.arguments
                if not x.kwarg_only
            ]
            if isinstance(self.op_overload, torch._ops.OpOverload)
            else [{} for i in range(len(self.inputs))]
        )
        self.allarg_properties = (
            {
                x.name: {"type": x.real_type, "default_value": x.default_value}
                for x in self.op_overload._schema.arguments
            }
            if isinstance(self.op_overload, torch._ops.OpOverload)
            else {}
        )
        # FIXME: self.kwargs does not always match kwargs defined in schema, so sometimes
        # ordered_kwargs_for_cpp_kernel is explicitly passed in.
        if isinstance(self.op_overload, torch._ops.OpOverload):
            if not self.ordered_kwargs_for_cpp_kernel:
                self.ordered_kwargs_for_cpp_kernel = [
                    x.name for x in self.op_overload._schema.arguments if x.kwarg_only
                ]
            self.schema_kwargs = [
                x for x in self.op_overload._schema.arguments if x.kwarg_only
            ]
        else:
            self.schema_kwargs = []

    def decide_layout(self) -> None:
        # [FUNCTION DOC] If the layout is flexible, apply the kernel's layout constraint then freeze it.
        if isinstance(self.layout, FlexibleLayout):
            self.apply_constraint()
            self.freeze_layout()

    def codegen_comment(
        self, wrapper: PythonWrapperCodegen, kernel_name: str | None = None
    ) -> None:
        # [FUNCTION DOC] Write origin comment text and provenance debug handle for the kernel into the wrapper.
        origin_str, _detailed_origin_str = get_kernel_metadata(self, wrapper)
        if origin_str:
            wrapper.make_comment(origin_str)

        if not kernel_name:
            kernel_name = self.try_get_kernel_name()
        if kernel_name:
            from .debug import set_kernel_post_grad_provenance_tracing

            debug_handle = set_kernel_post_grad_provenance_tracing(
                self, kernel_name, is_extern=True
            )
            wrapper.write_provenance_debug_handle(kernel_name, debug_handle)

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Generate wrapper code for the kernel; subclasses must override (NotImplementedError).
        raise NotImplementedError

    def set_cpp_kernel_name(self, cpp_kernel_name: str | None = None) -> None:
        # [FUNCTION DOC] Set the cpp kernel name. In cpp wrapper mode with an OpOverload and no name given, derive it: for aten ops the at::_ops call name (default overload uses the base name, otherwise name with underscores), with a version suffix if the op has a newer versioned shim; for other namespaces the schema name.
        self.cpp_kernel_name = cpp_kernel_name
        if not V.graph.cpp_wrapper or not isinstance(
            self.op_overload, torch._ops.OpOverload
        ):
            return

        kernel = self.op_overload
        if self.cpp_kernel_name is None:
            # Try to construct cpp_kernel_name from op_overload
            if kernel.namespace == "aten":
                # Calling with the default kernel name can lead to ambiguous behavior like the following example.
                # repeat_interleave(const at::Tensor & repeats, std::optional<int64_t> output_size=std::nullopt)
                # repeat_interleave(const at::Tensor & self, int64_t repeats,
                #       std::optional<int64_t> dim=std::nullopt, std::optional<int64_t> output_size=std::nullopt)
                opname = (
                    kernel.__name__.split(".")[0]
                    if kernel._overloadname == "default"
                    else kernel.__name__.replace(".", "_")
                )
                # If the op has a versioned c_shim entry, call the latest _v{N}
                # variant so new AOTI artifacts match the current op schema. The
                # unversioned shim is retained solely for BC with existing
                # already-compiled artifacts.
                from torchgen.aoti.fallback_ops import inductor_fallback_ops

                version_info = inductor_fallback_ops.get(f"aten.{kernel.__name__}", {})
                latest_version = max(
                    (int(v[1:]) for v in version_info if v.startswith("v")),
                    default=1,
                )
                if latest_version > 1:
                    opname = f"{opname}_v{latest_version}"
                self.cpp_kernel_name = f"at::_ops::{opname}::call"
            else:
                self.cpp_kernel_name = kernel._schema.name

    def set_python_kernel_name(self, python_kernel_name: str | None) -> None:
        # [FUNCTION DOC] Set the python kernel name; when not given derive it from the op overload (higher-order ops under torch.ops.higher_order, other ops from module and name).
        self.python_kernel_name = python_kernel_name
        if python_kernel_name is not None:
            return

        kernel = self.op_overload
        if kernel is None:
            pass
        elif isinstance(kernel, torch._ops.HigherOrderOperator):
            self.python_kernel_name = f"torch.ops.higher_order.{kernel.__name__}"
        else:
            self.python_kernel_name = (
                f"{kernel.__module__.replace('._ops.', '.ops.')}.{kernel.__name__}"
            )

    def try_get_kernel_name(self) -> str | None:
        # [FUNCTION DOC] Return the kernel name used in generated code: the python name under fx wrapper or python wrapper, or in cpp wrapper the C shim function name for the cpp kernel name (None if unknown). Raises AssertionError if the cpp wrapper is not a CppWrapperCpu.
        from .codegen.cpp_wrapper_cpu import CppWrapperCpu

        device = d.type if (d := self.get_device()) else V.graph.device_type
        if V.graph.fx_wrapper:
            return self.python_kernel_name
        elif V.graph.cpp_wrapper:
            if not isinstance(V.graph.wrapper_code, CppWrapperCpu):
                raise AssertionError(type(V.graph.wrapper_code))
            if self.cpp_kernel_name is None:
                return None
            return V.graph.wrapper_code.get_c_shim_func_name(
                self.cpp_kernel_name, device
            )
        else:
            return self.python_kernel_name

    def get_kernel_name(self) -> str:
        # [FUNCTION DOC] Return the kernel name, raising AssertionError if it is not available.
        name = self.try_get_kernel_name()
        if name is None:
            raise AssertionError("Expected name is not None")
        return name

    @staticmethod
    def copy_input(x: IRNode) -> TensorBox:
        # [FUNCTION DOC] Static method making a realized pointwise copy of x (preserving origin node and traceback) and returning it as a TensorBox.
        pw = Pointwise.create(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=x.make_loader(),
            ranges=x.get_size(),
            origin_node=x.get_origin_node(),
            traceback=x.get_traceback(),
        )
        pw.realize()
        return pw

    @classmethod
    def process_kernel(
        cls, kernel: _OpOverloads, *args: Any, **kwargs: Any
    ) -> ProcessKernelResult:
        # [FUNCTION DOC] Class method preparing an extern kernel call. Flattens args and kwargs, separating IR node inputs (tensor_args) from other values (symbolic expressions become symints; random generator states and opaque object states become real objects for the example run). Realizes tensor inputs and freezes their layouts, builds example inputs (graph constants, torchbind objects possibly replayed on fresh copies, opaque example values, generator states, or empty tensors from IR nodes), then reruns the kernel under the python dispatcher to get the example output with accurate strides. When a shape env exists, rebinds unbacked symbols and computes unbacked bindings (dropping the effect token for with_effects nodes). Disables cudagraphs with a reason if a sparse tensor is output and graph partition is off. Returns a ProcessKernelResult.
        """Partition kernel args into tensor and non-tensor, realize tensor inputs,
        re-run fake tensor propagation with the realized strides, and return a
        ProcessKernelResult (see that class for field semantics).
        """
        binded_args = {"args": args, "kwargs": kwargs}

        args_flat, args_spec = pytree.tree_flatten(binded_args)

        args_flat_is_tensor: list[bool] = []
        # tensor_args can be either tensor or torchbind objects
        tensor_args: list[IRNode] = []
        non_tensor_args: list[object] = []
        real_non_tensor_args: list[
            FakeScriptObject
            | torch._C.Generator
            | torch._C.ScriptObject
            | torch.Tensor
            | IntLikeType
        ] = []
        for arg in args_flat:
            match arg:
                case Expr():
                    node = V.graph.sizevars.shape_env.create_symintnode(arg, hint=None)
                    args_flat_is_tensor.append(False)
                    non_tensor_args.append(node)
                    real_non_tensor_args.append(node)

                case GeneratorState():
                    args_flat_is_tensor.append(False)
                    non_tensor_args.append(arg)
                    device_index = arg.device.index
                    if not (
                        arg.device.type in ["cuda", "xpu"] and device_index is not None
                    ):
                        raise AssertionError(
                            'Expected arg.device.type in ["cuda", "xpu"] and device_index is not None'
                        )
                    real_non_tensor_args.append(
                        torch._C._accelerator_getDefaultGenerator(
                            device_index
                        ).clone_state()
                    )

                case OpaqueObjectState():
                    args_flat_is_tensor.append(False)
                    non_tensor_args.append(arg)
                    real_non_tensor_args.append(arg.value)

                case IRNode():
                    args_flat_is_tensor.append(True)
                    tensor_args.append(arg)

                case _:
                    args_flat_is_tensor.append(False)
                    non_tensor_args.append(arg)
                    real_non_tensor_args.append(arg)

        def unflatten_args(
            new_tensor_args: Sequence[_T], new_non_tensor_args: Sequence[_T]
        ) -> tuple[list[_T], dict[str, _T]]:
            # [FUNCTION DOC] HELPER: Rebuild the original args and kwargs from replacement tensor and non-tensor lists, consuming each in the original flattened order.
            result = []
            it_tensors = iter(new_tensor_args)
            it_non_tensors = iter(new_non_tensor_args)
            for is_tensor in args_flat_is_tensor:
                if is_tensor:
                    result.append(next(it_tensors))
                else:
                    result.append(next(it_non_tensors))
            r = pytree.tree_unflatten(result, args_spec)
            return r.get("args", []), r.get("kwargs", {})

        tensor_args = [cls.realize_input(x) for x in tensor_args]

        # freeze layout otherwise our output stride calculation might
        # become incorrect
        for x in tensor_args:
            if is_storage_and_layout(x):
                as_storage_and_layout(x, freeze=True)

        # Rerun fake tensor propagation, because Inductor may have changed the
        # strides of inputs and we need to determine accurately what the
        # output stride will be.
        example_args: list[
            torch.Tensor | torch._C.ScriptObject | FakeScriptObject | torch.Generator
        ] = []

        # We need to retain the constant values of fake tensors that we originally
        # propagated the graph with, because for some operators running without a
        # constant would trigger an error / DataDependentException
        # TorchBind fake objects are mutable and may have been advanced by AOT
        # tracing. Replay effectful ops on a fresh per-graph copy of the guarded
        # object state instead of the already-mutated lowering value.
        replay_torchbind = (
            isinstance(kernel, torch._higher_order_ops.torchbind.CallTorchBind)
            or V.current_node.target is torch._higher_order_ops.effects.with_effects
        )
        for x in tensor_args:
            # if x is a view of a constant, we need to realize the view
            # (we can't pass the constant into the kernel directly)
            if not isinstance(x, BaseView) and x.get_name() in V.graph.constants:
                example_args.append(V.graph.constants[x.get_name()])
            elif (
                not isinstance(x, BaseView)
                and x.get_name() in V.graph.torchbind_constants
            ):
                example_args.append(V.graph.torchbind_constants[x.get_name()])
            elif isinstance(x, TorchBindObject):
                if replay_torchbind:
                    replay_objects = V.graph.torchbind_replay_objects
                    if x.get_name() not in replay_objects:
                        replay_objects[x.get_name()] = maybe_to_fake_obj(
                            V.fake_mode, x.get_real_obj()
                        )
                    example_args.append(replay_objects[x.get_name()])
                else:
                    example_args.append(x.get_value())
            elif isinstance(x, OpaqueMultiOutput):
                example_args.append(x.opaque_example_value)
            elif isinstance(x, torch._inductor.ir.GeneratorState):
                device_index = x.device.index
                if not (x.device.type in ["cuda", "xpu"] and device_index is not None):
                    raise AssertionError(
                        'Expected x.device.type in ["cuda", "xpu"] and device_index is not None'
                    )
                example_args.append(
                    torch._C._accelerator_getDefaultGenerator(
                        device_index
                    ).clone_state()
                )
            else:
                example_args.append(ir_node_to_tensor(x))

        new_args, new_kwargs = unflatten_args(example_args, real_non_tensor_args)
        shape_env = V.fake_mode.shape_env
        with (
            enable_python_dispatcher(),
            _fallback_kernel_symbol_tracking_context(shape_env),
        ):
            example_output = kernel(*new_args, **new_kwargs)

        unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] | None = None
        if shape_env:
            node_meta_val = V.current_node.meta.get("val")
            ctx: AbstractContextManager[None] = nullcontext()
            if V.current_node.target is torch._higher_order_ops.effects.with_effects:
                # remove the first effect token in meta["val"] and meta["unbacked_bindings"]
                # pyrefly: ignore[unsupported-operation]
                node_meta_val = node_meta_val[1]
                ctx = _remove_effect_token_unbacked_bindings(V.current_node)

            with ctx:
                rebind_unbacked(shape_env, V.current_node, example_output)
            unbacked_bindings = compute_unbacked_bindings(
                shape_env, example_output, node_meta_val
            )

        example_out_li = (
            [example_output]
            if not isinstance(example_output, (list, tuple))
            else example_output
        )
        # When graph_partition is enabled, skip - partitioning handles sparse outputs
        for t in example_out_li:
            if (
                isinstance(t, torch.Tensor)
                and t.is_sparse
                and not config.graph_partition
            ):
                msg = "sparsity not handled. Please file issue for sparse inference weights."
                if stack_trace := V.graph.current_node.meta.get("stack_trace", None):
                    msg = f"{msg} Found from : \n {stack_trace}"
                V.graph.disable_cudagraphs_reason = msg

        return ProcessKernelResult(
            example_output=example_output,
            tensor_args=tensor_args,
            non_tensor_args=non_tensor_args,
            unflatten_args=unflatten_args,
            unbacked_bindings=unbacked_bindings,
        )

    @classmethod
    def convert_to_reinterpret_view(cls, x: IRNode) -> ReinterpretView:
        # [FUNCTION DOC] Class method converting a non-reinterpret view into a ReinterpretView for passing to extern kernels, avoiding copies. The underlying layout is frozen (channels-last if the producing fx node's value is channels-last and the layout is flexible). Strides and offset are recovered from the view's indexer; raises NotImplementedError if the index is not an affine function of the loop vars, and AssertionError if the input is not a view or has no buffer.
        """
        In order to pass this to an extern kernel we need a
        ReinterpretView not a View.  This allows us to avoid some
        unneeded copies.
        """
        if not isinstance(x, BaseView):
            raise AssertionError(type(x))
        if isinstance(x, ReinterpretView):
            return x

        # NOTE: Don't use extract_read_writes here as it fails when
        # make_loader() inlines the computation
        x_unwrap_view = x.unwrap_view()
        buf = V.graph.get_buffer(x_unwrap_view.get_name())
        if buf is None:
            raise AssertionError("Expected buf is not None")
        x_unwrap_view_fx_node = buf.get_origin_node()
        # Prefer channels last format according to how the format is set from eager.
        if (
            x_unwrap_view_fx_node is not None
            and "val" in x_unwrap_view_fx_node.meta
            and isinstance(x_unwrap_view, (ReinterpretView, Buffer, MutableBox))
            and isinstance(x_unwrap_view.layout, FlexibleLayout)
            and (
                is_contiguous_for_memory_format_or_false(
                    x_unwrap_view_fx_node.meta["val"],
                    memory_format=torch.channels_last,
                )
                or is_contiguous_for_memory_format_or_false(
                    x_unwrap_view_fx_node.meta["val"],
                    memory_format=torch.channels_last_3d,
                )
            )
        ):
            x_unwrap_view.freeze_layout_with_same_order(
                make_channels_last_strides_for(x_unwrap_view.get_size())
            )
        else:
            x_unwrap_view.freeze_layout()

        index_args, var_ranges = dependencies.index_vars_squeeze(
            x.get_size(), prefix="r"
        )
        range_vars = index_args[0]
        index = x.make_indexer()(range_vars)

        index = V.graph.sizevars.simplify_with_ranges(index, var_ranges)
        strides = V.graph.sizevars.stride_vars(index, range_vars)
        offset = V.graph.sizevars.offset_var(index, range_vars)
        expected = sympy_dot(range_vars, strides) + offset

        if index != expected:
            log.debug(
                "convert_to_reinterpret_view failed: stride=%s offset=%s index=%s",
                strides,
                offset,
                index,
            )
            raise NotImplementedError

        return ReinterpretView(
            data=x.data,
            layout=FixedLayout(
                device=x.get_device_or_error(),
                dtype=x.get_dtype(),
                size=x.get_size(),
                stride=strides,
                offset=offset,
                is_pinned=False,
            ),
        )

    @classmethod
    def realize_input(cls, x: IRNode) -> IRNode:
        # [FUNCTION DOC] Class method turning any input into something an extern kernel can consume: None to NoneAsConstantBuffer, symbolic or int to ShapeAsConstantBuffer, a Constant to a graph tensor constant, boxes and reinterpret views recursively unwrapped, views realized (and converted to reinterpret views when possible), StorageBox realized, non-tensor objects passed through, anything else copied into a realized pointwise buffer.
        if x is None:
            return NoneAsConstantBuffer()
        if isinstance(x, (Expr, sympy.logic.boolalg.Boolean, int)):
            return ShapeAsConstantBuffer(expr=x)
        if isinstance(x, Constant):
            # We need to unset fake mode, or else the torch.tensor() call will
            # turn into a FakeTensor
            with _disable_current_modes():
                return V.graph.add_tensor_constant(
                    torch.tensor(x.value, dtype=x.get_dtype(), device=x.get_device())
                )
        if isinstance(x, ConstantBuffer):
            return x
        if isinstance(x, TensorBox):
            return cls.realize_input(x.data)
        if isinstance(x, ReinterpretView):
            return ReinterpretView(
                data=cls.realize_input(x.data), layout=x.get_layout()
            )
        if isinstance(x, BaseView):
            x.realize()
            if is_storage_and_layout(x.unwrap_view()):
                try:
                    return cls.convert_to_reinterpret_view(x)
                except NotImplementedError:
                    pass
        if isinstance(x, StorageBox):
            # TODO(jansel): impose layout preference on realized buffer
            x.realize()
            return x
        if isinstance(x, (NonTensorObj, ShapeAsConstantBuffer, OpaqueMultiOutput)):
            return x
        return cls.copy_input(x)

    @classmethod
    def require_stride1(cls, x: IRNode) -> IRNode:
        # [FUNCTION DOC] Class method returning x if it has some stride equal to one (or none) so the kernel can use it, else a realized copy.
        if is_storage_and_layout(x):
            if len(x.get_stride()) == 0:
                return x
            for stride in x.get_stride():
                if stride == 1:
                    return x
        return cls.copy_input(x)

    @classmethod
    def require_strides(
        cls,
        x: IRNode,
        order: Sequence[int] | None = None,
        exact_strides: Sequence[_IntLike] | None = None,
        allow_padding: bool = False,
    ) -> IRNode:
        # [FUNCTION DOC] Class method ensuring x has the requested stride order or exact strides (one must be given, else AssertionError). Empty or single-element inputs are returned as is when only an order is requested. Flexible layouts are frozen to the requirement; fixed, non-owning and mutation layouts or input buffers already satisfying it (up to insignificant strides or already-padded strides) are returned; view-of-storage inputs are converted to reinterpret views and retried; otherwise a copy is made with the required layout, preserving expanded dims (zero stride dims) by slicing and re-expanding, and complex dtypes copied via an in-place copy fallback. Mutates layouts and may create new IR nodes.
        """Ensure x has the requested stride order or exact strides, inserting a copy if needed."""
        if not (order is not None or exact_strides is not None):
            raise AssertionError(
                "Expected order is not None or exact_strides is not None"
            )
        # Layout generally doesn't matter, but some consuming external ops might have requirements
        if x.get_numel() in (0, 1) and not exact_strides:
            return x

        # require x to have the layout
        if is_storage_and_layout(x):
            if isinstance(x.get_layout(), FlexibleLayout):
                if order:
                    # If the FlexibleLayout already has the size and stride in the required order,
                    # freeze it to a FixedLayout by using its current size and stride.
                    # The behavior of using its current size and stride or the given order can be different
                    # if the size and stride has ambiguilty, for example for a 4D input where the iC = 1:
                    # size=[s0, 1, 28, 28], stride=[784, 784, 28, 1]. If the required order is [3, 0, 2, 1] (channels last),
                    # the current size and stride already satisfies this order.
                    # However by freezing it to the required order, the layout will be changed to:
                    # size=[s0, 1, 28, 28], stride=[784, 1, 28, 1]), which is not actually necessary.
                    use_current_stride_order = is_stride_order_storage_and_layout(
                        x, order
                    )
                    if use_current_stride_order:
                        x.freeze_layout()
                    else:
                        # fix flexiblelayout to be FixedLayout with stride_order
                        as_storage_and_layout(
                            x,
                            freeze=True,
                            want_contiguous=False,
                            stride_order=order,
                            allow_padding=allow_padding,
                        )
                    return x
                else:
                    # If the exact_strides is given, freeze the FlexibleLayout to a FixedLayout with the exact_strides.
                    as_storage_and_layout(
                        x,
                        freeze=True,
                        want_contiguous=False,
                        stride_order=None,
                        allow_padding=allow_padding,
                        exact_strides=exact_strides,
                    )
                    return x

            # When padding is allowed, check if the buffer's existing strides
            # match padded versions of the requested strides (e.g. concat graph
            # outputs that were already padded by ConcatKernel).
            padded_exact_strides = None
            if allow_padding and exact_strides:
                padded_exact_strides = list(
                    Layout._pad_strides(exact_strides, x.get_size(), x.get_dtype())
                )

            if isinstance(x.get_layout(), (FixedLayout, NonOwningLayout)) and (
                (order and x.get_layout().is_stride_ordered(order))
                or (
                    exact_strides
                    and significant_strides_equal(
                        exact_strides, x.get_layout().stride, x.get_size()
                    )
                )
            ):
                return (
                    try_match_insignificant_strides(x, exact_strides)
                    if exact_strides is not None
                    else x
                )
            # Accept already-padded buffers when padding is allowed
            elif (
                padded_exact_strides is not None
                and isinstance(x.get_layout(), (FixedLayout, NonOwningLayout))
                and significant_strides_equal(
                    padded_exact_strides, x.get_layout().stride, x.get_size()
                )
            ):
                return try_match_insignificant_strides(x, padded_exact_strides)
            elif isinstance(
                (mutation_layout := x.get_layout()), MutationLayoutSHOULDREMOVE
            ):
                if isinstance(
                    (real_layout := mutation_layout.real_layout()), FlexibleLayout
                ):
                    raise AssertionError(
                        "the MutationLayoutSHOULDREMOVE's real layout shouldn't be FlexibleLayout"
                    )
                elif isinstance(real_layout, FixedLayout) and (
                    (order and real_layout.is_stride_ordered(order))
                    or (
                        exact_strides
                        and significant_strides_equal(
                            exact_strides, real_layout.stride, x.get_size()
                        )
                    )
                ):
                    return x

        # TODO - Storage to InputBuffer
        if isinstance(x, InputBuffer) and (
            (order and x.get_layout().is_stride_ordered(order))
            or (
                exact_strides
                and significant_strides_equal(
                    exact_strides, x.get_layout().stride, x.get_size()
                )
            )
        ):
            return x
        if (
            isinstance(x, TensorBox)
            and isinstance(x.data, BaseView)
            and not isinstance(x.data, ReinterpretView)
            and is_storage_and_layout(unwrap_view := x.unwrap_view())
            and hasattr(unwrap_view, "data")
            and not isinstance(unwrap_view.data, ExternKernelAlloc)
        ):
            try:
                x.data = cls.convert_to_reinterpret_view(x.data)
                if order:
                    return cls.require_stride_order(
                        x, order, allow_padding=allow_padding
                    )
                elif exact_strides:
                    return cls.require_exact_strides(
                        x, exact_strides, allow_padding=allow_padding
                    )
            except NotImplementedError:
                pass

        # Preserve ExpandView representation that would be lost during copy_input
        # Without representation of the expand in inductor IR, in codegen we end up
        # launching a grid for the full size tensor and doing redundant computation
        # across expanded dims.
        # TODO: could also be good to have a codegen fix to recognize overlapping elements

        expanded_dims: list[int] | None = None
        orig_size = x.get_size()
        if exact_strides is not None:
            sizevars = V.graph.sizevars
            expanded_dims = [
                i
                for i in range(len(x.get_size()))
                if sizevars.statically_known_equals(exact_strides[i], 0)
                and sizevars.statically_known_geq(x.get_size()[i], 2)
            ]

            for dim in expanded_dims:
                x = torch._inductor.lowering.slice_(x, dim, 0, 1)

        if x.get_dtype().is_complex:
            # Triton signature codegen has no complex pointer dtype, so avoid
            # materializing complex layout constraints through a pointwise copy.
            if exact_strides is not None:
                layout = FlexibleLayout(
                    x.get_device_or_error(), x.get_dtype(), x.get_size()
                ).as_exact_strides(exact_strides, allow_padding=allow_padding)
            else:
                if order is None:
                    raise AssertionError(
                        "Expected order is not None when exact_strides is None"
                    )
                layout = FlexibleLayout(
                    x.get_device_or_error(), x.get_dtype(), x.get_size()
                ).as_stride_order(order, allow_padding=allow_padding)

            src = x
            x = torch._inductor.lowering.empty_strided(
                x.get_size(),
                layout.stride,
                dtype=x.get_dtype(),
                device=x.get_device_or_error(),
            )
            InplaceCopyFallback.create(x, src)
        else:
            # Although this is a clone, inductor is good about fusing clones into previous
            # operations if they weren't realized and their layouts were flexible.
            x = cls.copy_input(x)

            as_storage_and_layout(
                x,
                freeze=True,
                want_contiguous=False,
                stride_order=order,
                allow_padding=allow_padding,
                exact_strides=exact_strides,
            )
        if (
            order
            and not free_unbacked_symbols(x.get_size())
            and not free_unbacked_symbols(x.get_stride())
        ):
            if not is_stride_order_storage_and_layout(x, order):
                raise AssertionError(
                    "Expected is_stride_order_storage_and_layout(x, order)"
                )
        elif expanded_dims:
            if not (orig_size is not None and exact_strides is not None):
                raise AssertionError(
                    "Expected orig_size is not None and exact_strides is not None"
                )
            x = torch._inductor.lowering.expand(x, orig_size)
            # the expand will sometimes may change insignificant strides, so match them back
            return try_match_insignificant_strides(x, exact_strides)

        return x

    @classmethod
    def require_exact_strides(
        cls, x: IRNode, exact_strides: Sequence[_IntLike], allow_padding: bool = False
    ) -> IRNode:
        # [FUNCTION DOC] Class method requiring exact strides (symint strides unwrapped to expressions), optionally allowing padding.
        return cls.require_strides(
            x,
            exact_strides=[
                s.node.expr if isinstance(s, torch.SymInt) else s for s in exact_strides
            ],
            allow_padding=allow_padding,
        )

    @classmethod
    def require_stride_order(
        cls, x: IRNode, order: Sequence[int], allow_padding: bool = False
    ) -> IRNode:
        # [FUNCTION DOC] Class method requiring a stride order, optionally allowing padding.
        return cls.require_strides(x, order=order, allow_padding=allow_padding)

    @classmethod
    def require_channels_last(cls, x: IRNode) -> IRNode:
        # [FUNCTION DOC] Class method requiring channels-last (4D NHWC) stride order.
        return cls.require_stride_order(x, NHWC_STRIDE_ORDER)

    @classmethod
    def require_channels_last_3d(cls, x: IRNode) -> IRNode:
        # [FUNCTION DOC] Class method requiring channels-last 3D (5D NDHWC) stride order.
        return cls.require_stride_order(x, NHWDC_STRIDE_ORDER)

    @classmethod
    def require_contiguous(cls, x: IRNode) -> IRNode:
        # [FUNCTION DOC] Class method requiring contiguous strides; mkldnn constant tensors are returned unchanged.
        def is_mkldnn_tensor(x: IRNode) -> bool:
            # [FUNCTION DOC] HELPER: Return True if x names a graph constant that is an mkldnn tensor; False if it has no name.
            try:
                name = x.get_name()
            except (AttributeError, NotImplementedError):
                return False

            return name in V.graph.constants and V.graph.constants[name].is_mkldnn

        # TODO move this to the more proper places
        if is_mkldnn_tensor(x):
            return x
        else:
            return cls.require_exact_strides(
                x, FlexibleLayout.contiguous_strides(x.get_size())
            )

    @classmethod
    def require_contiguous_strides(cls, x: IRNode) -> IRNode:
        # [FUNCTION DOC] Class method requiring exactly contiguous strides for the size.
        # TODO: combine this with require_contiguous after
        # https://github.com/pytorch/pytorch/pull/148235 lands.
        return cls.require_exact_strides(
            x, FlexibleLayout.contiguous_strides(x.get_size())
        )

    def apply_constraint(self) -> None:
        # [FUNCTION DOC] Hook applying layout constraints before freezing; no-op by default.
        pass

    def fill_non_provided_args(
        self, args: Sequence[Any], kwargs: dict[str, Any]
    ) -> Sequence[Any]:
        # [FUNCTION DOC] Return the positional args extended with values for unprovided positional schema arguments, taken from kwargs or the schema default. Raises AssertionError if args is not a sequence or arg_properties is empty. Used by cpp wrapper codegen, which serializes defaults.
        # Previously, we want to maintain forward-compatibility by skipping
        # default args in the serialized artifacts in fbcode. However,
        # some of our shim interfaces require default values being OrderedSet.
        # Discussed with Sherlock offline and we decided to allow serializing
        # default args into the C++ wrapper code for now. We will refine this
        # part if we see real FC requirement. More details related to FC
        # can be found at:
        # https://docs.google.com/document/d/1FzWm-sHYwmRi3x_g036kOxd99KaYquUsA-L5JwOn8ys/edit?usp=sharing
        if not isinstance(args, Sequence):
            raise AssertionError(type(args))
        if not isinstance(args, list):
            args = list(args)
        if not self.arg_properties:
            raise AssertionError("ExternKernel.arg_properties should not be empty")

        n_args = len(args)
        n_pos_args = len(self.arg_properties)
        # For cpp wrapper, if some positional args are not provided, we need to check
        # if they're in the kwargs or use their default value
        if n_args < n_pos_args:
            log.debug(
                "%s has %d unprovided positional arguments. "
                "Will check if they are in the keyword arguments or will use default values.",
                self.op_overload,
                n_pos_args - n_args,
            )
            for i in range(n_args, n_pos_args):
                arg_name = self.arg_properties[i]["name"]
                args.append(
                    kwargs[arg_name]
                    if arg_name in kwargs
                    else self.arg_properties[i]["default_value"]
                )
        return args

    def codegen_const_args(self, names: list[str] | None = None) -> list[str]:
        # [FUNCTION DOC] Return code strings for constant args. For cpp wrapper, each is formatted with its schema type, looked up by the given names (length must match, AssertionError otherwise) or by position after the inputs; for the python wrapper plain formatting.
        if V.graph.cpp_wrapper:
            result = []
            # Aten ops follow the convention that tensor args are before non-tensor args,
            # in which case the following 'len(self.inputs) + i' logic works. But this
            # may not be true for other ops, and if that is the case, caller needs to
            # pass in a list of const arg names for arg_properties lookup.
            name_to_arg_properties: dict[str, ArgProperty] | None = None
            if names and self.arg_properties:
                if len(self.constant_args) != len(names):
                    raise AssertionError(
                        "names passed to codegen_const_args does not match self.constant_args"
                    )
                name_to_arg_properties = {
                    name: arg
                    for arg in self.arg_properties
                    if (name := arg.get("name")) is not None
                }

            for i, x in enumerate(self.constant_args):
                if name_to_arg_properties is not None:
                    if names is None:
                        raise AssertionError("Expected names is not None")
                    prop = name_to_arg_properties.get(names[i])
                    type_ = prop.get("type") if prop else None
                else:
                    idx = len(self.inputs) + i
                    type_ = (
                        self.arg_properties[idx].get("type")
                        if self.arg_properties and idx < len(self.arg_properties)
                        else None
                    )
                result.append(V.graph.wrapper_code.val_to_arg_str(x, type_))
            return result
        else:
            return [V.graph.wrapper_code.val_to_arg_str(a) for a in self.constant_args]

    def codegen_args(self) -> list[str]:
        # [FUNCTION DOC] Return code strings for the call args: inputs (and for cpp wrapper with an overload, constants and defaults filled via fill_non_provided_args) formatted with schema types; otherwise inputs then constant args. Raises AssertionError on missing cpp argument properties.
        if V.graph.cpp_wrapper and self.op_overload is not None:
            # cpp wrapper needs special logic to fill in missing args with default values
            inputs = self.fill_non_provided_args(
                [*self.inputs, *self.constant_args], self.kwargs
            )
            # fill_non_provided_args has handled constant args, so no need to codegen for that later
            need_codegen_constant_args = False
        else:
            inputs = self.inputs
            need_codegen_constant_args = True

        args = []
        for i, x in enumerate(inputs):
            if V.graph.cpp_wrapper:
                if not (self.arg_properties and i < len(self.arg_properties)):
                    raise AssertionError(
                        "Invalid access to ExternKernel.arg_properties"
                    )
                type_ = self.arg_properties[i].get("type")
                args.append(V.graph.wrapper_code.val_to_arg_str(x, type_))
            else:
                args.append(V.graph.wrapper_code.val_to_arg_str(x))
        if need_codegen_constant_args:
            args.extend(self.codegen_const_args())
        return args

    def get_kwargs_value(self, arg_name: str, **kwargs: Any) -> Any:
        # [FUNCTION DOC] Look up an argument value by name: provided keyword overrides first, then the node kwargs, then the schema default. Raises AssertionError if the name is unknown.
        """Given an argument name, queries for values in (in order):
        1. any provided kwargs for this function.
        2. the class self.kwargs member.
        3. any available default arguments in self.allarg_properties."""
        if arg_name in kwargs:
            return kwargs.get(arg_name)
        if arg_name in self.kwargs:
            return self.kwargs.get(arg_name)
        if (arg := self.allarg_properties.get(arg_name)) is not None:
            return arg.get("default_value")
        raise AssertionError(f"{arg_name} not in self.allarg_properties")

    def codegen_kwargs(self, skip_out: bool = False) -> list[str]:
        # [FUNCTION DOC] Return code strings for keyword args. cpp wrapper: values in the ordered cpp kwargs list (optionally skipping out), formatted with schema types, empty when all args were already filled and there are no schema kwargs; python wrapper: name equals value pairs.
        if V.graph.cpp_wrapper:
            if self.op_overload is not None and len(self.schema_kwargs) == 0:
                # All the args should have been generated by fill_non_provided_args in codegen_args
                return []

            kwargs = []
            for arg_name in self.ordered_kwargs_for_cpp_kernel:
                if skip_out and arg_name == "out":
                    # ExternKernelOut has its own logic for inserting the out parameter
                    continue

                v = self.get_kwargs_value(arg_name)
                if isinstance(v, Expr):
                    kwargs.append(v)
                else:
                    if self.allarg_properties is None:
                        raise AssertionError(
                            "Expected self.allarg_properties is not None"
                        )
                    type_ = self.allarg_properties.get(arg_name, {}).get("type")
                    kwargs.append(V.graph.wrapper_code.val_to_arg_str(v, type_))
        else:
            kwargs = [
                f"{k}={V.graph.wrapper_code.val_to_arg_str(v)}"
                for k, v in self.kwargs.items()
            ]
        return kwargs

    def get_op_name(self) -> str:
        # [FUNCTION DOC] Return the fully qualified op name from the fx node target, or unknown_op if there is no fx node.
        if self.fx_node is not None:
            target = self.fx_node.target
            op_namespace = getattr(target, "__module__", "unknown_namespace")
            op_namespace = op_namespace.replace("._ops.", ".ops.")
            op_namespace = op_namespace.rsplit(".", 1)[0]
            op_name = f"{op_namespace}.{target}"
        else:
            op_name = "unknown_op"
        return op_name

    def is_inplace_view(self) -> bool:
        # [FUNCTION DOC] Return True if the op overload carries the inplace-view tag.
        return (
            isinstance(self.op_overload, torch._ops.OpOverload)
            and torch.Tag.inplace_view in self.op_overload.tags
        )

    def should_assert_dtype(self, op_name: str) -> bool:
        # [FUNCTION DOC] Return whether output dtype should be asserted: not for inplace views and not for quantize_per_tensor, whose fake dtype is non-quantized.
        # FakeTensor does not support quantized meta tensors today, so
        # quantize_per_tensor's fake dtype intentionally remains non-quantized.
        return (
            not self.is_inplace_view()
            and not op_name.startswith("torch.ops.aten.quantize_per_tensor.")
            and (
                self.op_overload
                not in (
                    torch.ops.aten.quantize_per_tensor.default,
                    torch.ops.aten.quantize_per_tensor.tensor_qparams,
                )
            )
        )

    def codegen_size_asserts(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Emit a runtime size/stride (and dtype) assertion for the output into the wrapper when config.size_asserts is on; skips inplace views for the python wrapper, and for cpp wrapper asserts on the mutated input for inplace views.
        if not config.size_asserts:
            return
        if self.is_inplace_view() and not V.graph.cpp_wrapper:
            return
        op_name = self.get_op_name()
        name = self.get_name()
        if V.graph.cpp_wrapper:
            # inplace_view ops (e.g. set_.source_Tensor) don't declare an
            # output variable; assert on the mutated input instead.
            if self.is_inplace_view():
                if not isinstance(self.inputs[0], IRNode):
                    raise AssertionError("Expected isinstance(self.inputs[0], IRNode)")
                name = self.inputs[0].get_name()
        size = V.graph.wrapper_code.codegen_shape_tuple(self.get_size())
        stride = V.graph.wrapper_code.codegen_shape_tuple(self.get_stride())
        dtype = self.get_dtype() if self.should_assert_dtype(op_name) else None
        wrapper.write_assert_size_stride(name, size, stride, op_name, dtype)

    def codegen_alignment_asserts(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Emit a runtime alignment assertion (or a comment if the buffer is assumed unaligned) when config.alignment_asserts is on and the python wrapper is used.
        if config.alignment_asserts and not V.graph.cpp_wrapper:
            name = self.get_name()
            aligned = name not in V.graph.unaligned_buffers
            op_name = self.get_op_name()
            if aligned:
                wrapper.writeline(
                    f"assert_alignment({name}, {GPU_ALIGN_BYTES}, {op_name!r})"
                )
            else:
                wrapper.writeline(
                    f"# buffer {name} (op: {op_name}) is assumed to be not aligned"
                )

    def codegen_memory_tracking(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] When config.test_configs.track_memory_lifecycle is on (python wrapper only), emit code tracking the output tensor allocation.
        """
        Track outputs of fallback operators if config.test_configs.track_memory_lifecycle
        """
        if not config.test_configs.track_memory_lifecycle or V.graph.cpp_wrapper:
            return

        wrapper.write_memory_track_allocation_once()
        name = self.get_name()
        wrapper.writeline(f"track_tensor({name}, '{name}')")

    def get_group_stride(self) -> tuple[list[Sequence[Expr]], list[Expr]]:
        # [FUNCTION DOC] Return the output sizes and strides for template codegen: ([size, []], stride), as there is no reduction.
        """
        get output sizes and strides, for template_codegen
        """
        _size = self.get_size()
        _stride = self.get_stride()
        # iter_ranges = _size of output tensor, reduce_range = [] because no reduction
        return [_size, []], _stride

    def canonicalize(self) -> tuple[Expr, Sequence[Expr]]:
        # [FUNCTION DOC] Compute a canonical output index expression and sizes: names index vars per dim ordered by stride hints (largest first, hints only affect ordering), builds the indexer expression, simplifies loops (merging dims) and substitutes fresh variables. Returns (index expression, tuple of simplified sizes). Used to compare outputs, e.g. for conv.
        """
        Manually get canonicalization of the output index
        """
        # manually generate index formula for conv
        sizevars = V.graph.sizevars
        sizes = self.get_size()
        strides = self.get_stride()
        # Stride hints are only used as sort keys to determine dimension
        # ordering for canonicalization, not for correctness.
        strides = [sizevars.optimization_hint(x) for x in strides]
        # TODO: I can't tell if the symbols here are temporary
        index_vars = [sympy_index_symbol(f"d{i}") for i in range(len(sizes))]
        # reorder index vars according to stride
        index_order = sorted(range(len(strides)), key=strides.__getitem__, reverse=True)
        lookup = {pos: idx for idx, pos in enumerate(index_order)}
        order = [lookup[i] for i in range(len(lookup))]
        index_vars = [index_vars[i] for i in order]
        indexer = self.make_indexer()
        index = indexer(index_vars)

        new_sizes, reindex, _prune = V.graph.sizevars._simplify_loops(
            index_vars, sizes, [index]
        )

        # assign new variables each dimension to deal with numbering mismatches
        # d0, d1, d2 could become d0, d2 -- which won't match d0, d1
        _, add_var = var_builder("c")
        replacement = dict(zip(index_vars, reindex([add_var(x) for x in new_sizes])))

        index = sympy_subs(sympy.expand(index), replacement)
        return index, tuple(new_sizes)

    @cache_on_self_and_args("ExternKernel")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return free symbols of the inputs plus those found in constant args and kwargs values (only unbacked ones if requested); cached.
        # NB: It's not necessary to check regular inputs as we automatically
        # have dependencies on them
        maybe_get_symbols = (
            maybe_free_unbacked_symbols if unbacked_only else maybe_free_symbols
        )
        r = InputsKernel.get_free_symbol_uses(self, unbacked_only)
        for arg in self.constant_args:
            r |= maybe_get_symbols(arg)
        for arg in self.kwargs.values():
            r |= maybe_get_symbols(arg)
        return r

    def __str__(self) -> str:
        # [FUNCTION DOC] Return a string repr listing the python kernel name, every dataclass field and the origin node.
        kernel_name = getattr(self, "python_kernel_name", None)
        lines = [
            f"python_kernel_name={kernel_name!r}",
        ]
        lines += [
            f"{field.name}={getattr(self, field.name)}"
            for field in dataclasses.fields(self)
        ]
        lines.append(f"origin_node={self.origin_node!r}")
        return self.str_helper(lines)

    __repr__ = __str__


@ir_dataclass(frozen=False)
class ExternKernelOut(ExternKernel):
    # [CLASS DOC] ExternKernel whose output buffer is allocated by Inductor and passed to the kernel as the out argument; allocates (should_allocate is True) and registers itself as buffer and operation. Codegen delegates to the wrapper's extern-kernel-out generator.
    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Generate wrapper code for the out-variant kernel call.
        wrapper.generate_extern_kernel_out(self)

    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[IRNode],
        constant_args: Sequence[Any] = (),
        kwargs: dict[str, Any] | None = None,
        output_view: ReinterpretView | None = None,
        python_kernel_name: str | None = None,
        cpp_kernel_name: str | None = None,
        ordered_kwargs_for_cpp_kernel: Sequence[Any] = (),
        op_overload: _OpOverloads | None = None,
    ) -> None:
        # [FUNCTION DOC] Initialize with unwrapped inputs and an unnamed output, then register as buffer and operation with the graph. KNOWN ISSUE: the output_view parameter is accepted but ignored; None is always passed to the parent.
        unwrapped_inputs = self.unwrap_storage(inputs)
        if not isinstance(unwrapped_inputs, Sequence):
            raise AssertionError(type(unwrapped_inputs))
        super().__init__(
            None,
            layout,
            unwrapped_inputs,
            constant_args,
            kwargs or {},
            None,
            python_kernel_name,
            cpp_kernel_name,
            ordered_kwargs_for_cpp_kernel,
            op_overload,
        )
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def should_allocate(self) -> bool:
        # [FUNCTION DOC] Return True.
        return True


class RandomSeeds(ExternKernelOut):
    # [CLASS DOC] ExternKernelOut generating random seeds: an int64 buffer of count values filled by the randint low out op over the full int64 range, with a fixed layout on the given device.
    def __init__(self, count: int, device: torch.device) -> None:
        # [FUNCTION DOC] Build the seed buffer: fixed layout of the count, constant args for the int64 limits and shape, and the kernel names for python and cpp wrappers (randint low out).
        limits = torch.iinfo(torch.int64)
        super().__init__(
            layout=FixedLayout(
                device=device,
                dtype=torch.int64,
                size=[count],
            ),
            inputs=[],
            constant_args=[limits.min, limits.max, [count]],
            python_kernel_name="aten.randint.low_out",
            # FIXME: Ideally we should only use at::_ops::randint_low_out::call here,
            # but the signature is different from is at::randint_out. Again,
            # we can simplify the code when only keeping an ABI-compatible version.
            cpp_kernel_name="at::_ops::randint_low_out::call",
            op_overload=aten.randint.low_out,
        )


class ExternKernelAlloc(ExternKernel):
    # [CLASS DOC] ExternKernel whose kernel allocates its own output (no preallocation, should_allocate is False). Keeps an outputs list for ABI-compatible codegen. Layout constraints are not supported (apply_constraint raises).
    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Generate wrapper code for the allocating extern kernel call.
        wrapper.generate_extern_kernel_alloc(self)

    def __init__(
        self,
        layout: OutputSpec,
        inputs: Sequence[IRNode],
        constant_args: Sequence[Any] = (),
        kwargs: dict[str, Any] | None = None,
        python_kernel_name: str | None = None,
        cpp_kernel_name: str | None = None,
        ordered_kwargs_for_cpp_kernel: Sequence[Any] = (),
        op_overload: _OpOverloads | None = None,
    ) -> None:
        # [FUNCTION DOC] Initialize with unwrapped inputs (AssertionError if any is not an IR node) and an unnamed output; create an empty outputs list and register as buffer and operation with the graph.
        unwrapped_inputs = self.unwrap_storage(inputs)
        if not all(isinstance(i, IRNode) for i in unwrapped_inputs):
            raise AssertionError(
                "Expected all(isinstance(i, IRNode) for i in unwrapped_inputs)"
            )
        super().__init__(
            None,
            layout,
            cast(Sequence[IRNode], unwrapped_inputs),
            constant_args,
            kwargs or {},
            None,
            python_kernel_name,
            cpp_kernel_name,
            ordered_kwargs_for_cpp_kernel,
            op_overload,
        )
        # We need output buffers for generating kernel arguments in the
        # abi-compatible mode, where we retrieve outputs by pass each individual
        # output through the abi-compatible interface.
        self.outputs: Sequence[Any] = []
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def should_allocate(self) -> bool:
        # [FUNCTION DOC] Return False; the kernel allocates the output.
        return False

    def apply_constraint(self) -> None:
        # [FUNCTION DOC] Not supported; always raises NotImplementedError.
        raise NotImplementedError


class MutationOutput(Buffer):
    # [CLASS DOC] Output buffer representing the mutation of an existing buffer by an operation. On creation it marks the mutated buffer as mutated and registers itself; it does not allocate and is defined by the mutating operation.
    """
    An output buffer that represents the mutation of a pre-existing buffer
    """

    def __init__(
        self, layout: OutputSpec, mutated_node: IRNode, mutating_node: Operation
    ) -> None:
        # [FUNCTION DOC] Mark the mutated node's buffer as mutated, record the mutated name and mutating node, and register the buffer.
        super().__init__(name=None, layout=layout)
        mutated_node_name = mutated_node.get_name()
        V.graph.mark_buffer_mutated(mutated_node_name)
        self.mutation_names = [mutated_node_name]
        self.mutating_node: Operation = mutating_node
        self.name = V.graph.register_buffer(self)

    def get_defining_op(self) -> Operation:
        # [FUNCTION DOC] Return the mutating operation.
        return self.mutating_node

    def get_mutation_names(self) -> Sequence[str]:
        # [FUNCTION DOC] Return the names of mutated buffers.
        return self.mutation_names

    def should_allocate(self) -> bool:
        # [FUNCTION DOC] Return False.
        return False

    def get_mutation_buffers(self) -> Sequence[IRNode]:
        # [FUNCTION DOC] Return the buffers named in the mutation names that are known to the graph.
        mutation_names = self.get_mutation_names()
        return [
            buf
            for buf in (V.graph.try_get_buffer(name) for name in mutation_names)
            if buf is not None
        ]


class OrderingBarrier(NopKernel):
    # [CLASS DOC] No-op node that creates a rename chain from a source buffer to a new buffer name purely to order scheduling (readers of the source are redirected through the barrier). Uses fake weak dependencies via the ordering_only flag, so no lifetime extension, no mutation marking, no kernel and no allocation.
    """A no-op that creates a rename chain for scheduling order.

    Takes a source buffer and produces a new buffer name. The scheduler
    sees this as a node with a rename (source -> self), so future readers
    of the source are redirected through this barrier.

    Uses WeakDep(is_fake=True) instead of StarDep via the ``ordering_only``
    flag, avoiding lifetime extension and mark_buffer_mutated side effects.
    No kernel is generated and no allocation occurs.
    """

    ordering_only = True

    def __init__(self, source_node: IRNode) -> None:
        # [FUNCTION DOC] Realize the source node, create the unnamed no-op node with an empty layout reading it, record the source name as mutation name, and register buffer and operation.
        source_node.realize()
        super().__init__(
            name=None,
            layout=NoneLayout(device=source_node.get_device()),
            inputs=[source_node],
        )
        self.mutation_names = [source_node.get_name()]
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def get_mutation_names(self) -> Sequence[str]:
        # [FUNCTION DOC] Return the name of the source buffer.
        return self.mutation_names

    def should_allocate(self) -> bool:
        # [FUNCTION DOC] Return False.
        return False


class TMADescriptor(ExternKernel):
    # [CLASS DOC] ExternKernel IR node for a host-side TMA descriptor in the Triton API (for user-defined Triton kernels and templates). Instances are cached by tensor identity and metadata. Subclasses: TMADescriptorExperimental (old API) and TMADescriptorStable (new API). Holds a non-owning layout over the tensor so the tensor outlives it.
    """
    An IR node representing a generic host-side TMA descriptor in the Triton API
    Mostly useful for user-defined Triton kernels relying on host-side TMA;
    but can, in principle, be used for Inductor's Triton templates, too.

    See TMADescriptorExperimental and TMADescriptorStable for the two implementations
    (the old API and the new API)
    """

    # as TMA descriptors are immutable,
    # we can dedup them by the input args
    _CACHE: dict[Any, TMADescriptor] = {}

    @classmethod
    def _create_impl(
        cls, tensor: IRNode, tma_meta: tuple[str, tuple[Any, ...]]
    ) -> TMADescriptor:
        # [FUNCTION DOC] HELPER: Class method picking the descriptor subclass from the metadata (kind experimental or stable, plus its arguments); AssertionError for malformed or unknown metadata.
        if len(tma_meta) != 2:
            raise AssertionError("Expected len(tma_meta) == 2")
        if tma_meta[0] == "experimental":
            return TMADescriptorExperimental(tensor, *tma_meta[1])
        else:
            if tma_meta[0] != "stable":
                raise AssertionError('Expected tma_meta[0] == "stable"')
            return TMADescriptorStable(tensor, *tma_meta[1])

    @classmethod
    def create(
        cls, tensor: IRNode, tma_meta: tuple[str, tuple[Any, ...]]
    ) -> TMADescriptor:
        # [FUNCTION DOC] Class method returning the cached descriptor for (tensor identity, metadata), creating it if absent. KNOWN ISSUE: the cache key uses the object id of the tensor, so a stale entry could be hit if an id is reused.
        key = (id(tensor), tma_meta)
        if key not in cls._CACHE:
            cls._CACHE[key] = cls._create_impl(tensor, tma_meta)
        return cls._CACHE[key]

    def __init__(
        self, tensor: IRNode, inputs: Sequence[Any], constant_args: Sequence[Any]
    ) -> None:
        # [FUNCTION DOC] Initialize as an extern kernel with a non-owning layout over the tensor, the given inputs and constants; remember the tensor and register buffer and operation.
        super().__init__(
            None,
            # link back to the underlying tensor in terms of ownership
            # to avoid getting the underlying tensor deleted *before*
            # the TMADescriptor node can be deleted.
            NonOwningLayout(
                ReinterpretView(
                    data=tensor,
                    layout=tensor.get_layout(),
                )
            ),
            cast(Sequence[Buffer], inputs),
            tuple(constant_args),
            None,
        )

        self.tensor = tensor
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Generate wrapper code creating the TMA descriptor.
        wrapper.generate_tma_descriptor(self)

    def get_tensor(self) -> IRNode:
        # [FUNCTION DOC] Return the underlying tensor node.
        return self.tensor


class TMADescriptorExperimental(TMADescriptor):
    # [CLASS DOC] TMADescriptor for the old host-side API (create 1d/2d descriptor calls): stores dims, block dims, element size (default dtype item size) and rank.
    """
    the new host-side TMA Descriptor API:
    (the ones obtained via create_{1d,2d}_tma_descriptor calls).

    See also TMADescriptorStable for the new API.
    """

    def __init__(
        self,
        tensor: IRNode,
        dims: list[int | torch.SymInt],
        block_dims: list[int | torch.SymInt],
        element_size: int | None = None,
    ) -> None:
        # [FUNCTION DOC] Validate that dims has length one or two and matches block_dims (AssertionError otherwise), record dims, block dims, element size and rank, then initialize the parent with the tensor as input and the dims, block dims and element size as constants.
        if len(dims) not in (1, 2):
            raise AssertionError("Expected len(dims) in (1, 2)")
        if len(dims) != len(block_dims):
            raise AssertionError("Expected len(dims) == len(block_dims)")

        if element_size is None:
            element_size = tensor.get_dtype().itemsize

        self.dims = dims
        self.block_dims = block_dims
        self.element_size = element_size
        self.rank = len(self.dims)

        inputs = [tensor]
        constant_args = [
            *self.dims,
            *self.block_dims,
            self.element_size,
        ]

        super().__init__(
            tensor=tensor,
            inputs=inputs,
            constant_args=constant_args,
        )


class TMADescriptorStable(TMADescriptor):
    # [CLASS DOC] TMADescriptor for the new host-side API (tensor descriptor from a tensor): stores the block shape, passed as constant args.
    """
    the new host-side TMA descriptor API
    (the ones obtained via TensorDescriptor.from_tensor).

    See also TMADescriptorExperimental for the old API.
    """

    def __init__(self, tensor: IRNode, block_shape: list[int | torch.SymInt]):
        # [FUNCTION DOC] Record the block shape and initialize the parent with the tensor input and block shape as constants.
        self.block_shape = block_shape

        super().__init__(
            tensor=tensor,
            inputs=[tensor],
            constant_args=block_shape,
        )


class SubgraphBuffer(ExternKernel):
    # [CLASS DOC] ExternKernel that executes a compiled fx subgraph: lowers the graph module with the input nodes into a subgraph (with symbolic inputs added), under a config forcing no Triton autotune of gemms (ATen backends only) plus user config patches. Operations of the subgraph are tagged with the config patches. Codegen emits the subgraph call with flattened outputs.
    def __init__(
        self,
        layout: Layout,
        input_nodes: list[Buffer],
        gm: torch.fx.GraphModule,
        example_inputs: list[Any],
        subgraph_name: str,
        config_patches: dict[str, Any] | None = None,
    ):
        # [FUNCTION DOC] Initialize the extern kernel, register buffer and operation, create the subgraph and add symbolic inputs of the input nodes as subgraph graph inputs, then run the subgraph on the example inputs under merged config patches (base: no max-autotune, ATen gemm backend). If config patches are given, tag each subgraph operation with a copy of them. AssertionError if the inputs are not IR nodes.
        super().__init__(None, layout, input_nodes)
        self.gm = gm
        self.example_inputs = example_inputs
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

        self.subgraph = V.graph.make_subgraph(self.gm, example_inputs, subgraph_name)

        if not is_node_sequence(self.inputs):
            raise AssertionError("Expected is_node_sequence(self.inputs)")
        sym_inputs = get_symbolic_inputs(self.inputs)

        for sym_inp in sym_inputs:
            self.subgraph.graph_inputs[sym_inp.name] = sym_inp
            self.subgraph.graph_input_names.append(sym_inp.name)

        self.sym_inputs = [sym_var.name for sym_var in sym_inputs]

        import torch._inductor.config as inductor_config

        with V.set_graph_handler(self.subgraph):
            # Base config: don't autotune Triton, but allow other optimizations
            base_patches = {
                "max_autotune": False,
                "max_autotune_gemm": False,
                "max_autotune_gemm_backends": "ATEN",
            }
            # Merge with user config_patches (e.g., coordinate_descent_tuning)
            merged_patches: dict[str, Any] = {**base_patches, **(config_patches or {})}
            with inductor_config.patch(merged_patches):
                self.subgraph.run(*self.example_inputs)

            # Tag all operations in subgraph with config_patches
            # These will be applied during kernel codegen via SIMD scheduling
            if config_patches:
                for op in self.subgraph.operations:
                    op.set_config_patches(config_patches.copy())

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Generate wrapper code calling the subgraph with symbolic inputs followed by the referenced outer inputs, binding its output to this buffer name. AssertionError if inputs are not IR nodes.
        class CodegenGraph:
            # [CLASS DOC] HELPER: Minimal holder pairing a graph with its name for the wrapper subgraph codegen call.
            def __init__(self, graph: GraphLowering):
                # [FUNCTION DOC] HELPER: Store the graph and its name.
                self.graph = graph
                self.name = graph.name

        if not is_node_sequence(self.inputs):
            raise AssertionError("Expected is_node_sequence(self.inputs)")
        outer_inputs = [t.codegen_reference() for t in self.inputs]

        wrapper.codegen_subgraph_with_flattened_outputs(
            CodegenGraph(self.subgraph),
            [*self.sym_inputs, *outer_inputs],
            [self.name],
        )


class UserDefinedTritonKernel(ExternKernel):
    # [CLASS DOC] ExternKernel for a user-defined Triton kernel (jit function), looked up by kernel index in the kernel side table. Tracks kernel stores, argument accesses, mutable args, grid and launch kwargs, supports fusing a pointwise epilogue into the kernel (rewriting its store) and generates the kernel definition and launch call.
    """
    A user-defined triton kernel (e.g. via @triton.jit).
    """

    def get_kernel_and_metadata(self) -> tuple[Kernel, Any, list[str], list[str]]:
        # [FUNCTION DOC] Look up the kernel by index. For an autotuned kernel, returns the underlying function, its configs and names of arguments to restore or reset to zero (handling both older index-based and newer name-based triton attributes); otherwise empty config and lists. Returns (kernel, configs, restore value arg names, reset to zero arg names).
        from triton.runtime.autotuner import Autotuner

        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

        kernel = kernel_side_table.get_kernel(self.kernel_idx)
        configs = []
        restore_value_args: list[str] = []
        reset_to_zero_args: list[str] = []
        if isinstance(kernel, Autotuner):
            # https://github.com/triton-lang/triton/pull/5083
            # changes kernel.restore_idx to kernel.restore_value
            if hasattr(kernel, "restore_idx"):
                restore_value_args.extend(
                    kernel.fn.arg_names[i] for i in kernel.restore_idx
                )
            else:
                if not hasattr(kernel, "restore_value"):
                    raise AssertionError('Expected hasattr(kernel, "restore_value")')
                restore_value_args.extend(kernel.restore_value)

            if hasattr(kernel, "reset_idx"):
                for i in kernel.reset_idx:
                    reset_to_zero_args.append(kernel.fn.arg_names[i])
            else:
                if not hasattr(kernel, "reset_to_zero"):
                    raise AssertionError('Expected hasattr(kernel, "reset_to_zero")')
                reset_to_zero_args.extend(kernel.reset_to_zero)

            configs = kernel.configs
            kernel = kernel.fn

        return kernel, configs, restore_value_args, reset_to_zero_args

    def can_fuse_epilogue(self) -> bool:
        # [FUNCTION DOC] Return True if an epilogue could be fused into this kernel: the config flag must be on, the argument analysis must allow it, the kernel must have exactly one store, exactly one mutable argument (AssertionError otherwise) that is a TensorBox over a StorageBox over a ComputedBuffer of Pointwise data with all ranges zero (an empty tensor), so epilogue of undefined values is harmless.
        """
        For kernels like

        @triton.jit
        def add_kernel(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(in_ptr0 + offs, mask=mask)
            y = tl.load(in_ptr1 + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        @torch.compile
        def fn(a, b):
            out = torch.empty_like(a)
            grid = (triton.cdiv(a.numel(), 1024),)
            add_kernel[grid](a, b, out, a.numel(), BLOCK_SIZE=1024)
            return out.relu()

        We can potentially fuse the relu epilogue into the add_kernel.
        We do this by pruning the `out` tensor allocation and directly writing the relu-output.
        """

        if not config.epilogue_fusion_user_defined_triton_kernel:
            return False

        if not self.arg_accesses.can_fuse_epilogue:
            return False

        # We achieve fusion by parsing the original src into a python AST,
        # then identify the expr containing the original value written via tl.store().
        # We generate an expr for the value after the epilogue and replace that into the tl.store.
        # So far we only support the simple case where there is a single tl.store in the kernel.
        if len(self.kernel_stores.stores) != 1:
            return False

        # Only fuse if the mutated arg is originally an "empty" tensor.
        # This is because we don't know exactly which element of that tensor is being written to.
        # If the kernel only writes to a subset of the tensor, then we only apply the epilogue to that subset.
        # In these edge cases, our fusion is only correct if the original tensor is empty,
        # where the semantics is that content values are UB, and we can rely on the fact that `epilogue(UB) == UB`.
        if len(self.mutable_args) != 1:
            raise AssertionError("Expected len(self.mutable_args) == 1")
        if not isinstance(self.mutable_args[0], TensorBox):
            return False
        if not isinstance(self.mutable_args[0].data, StorageBox):
            return False
        if not isinstance(self.mutable_args[0].data.data, ComputedBuffer):
            return False
        if not isinstance(self.mutable_args[0].data.data.data, Pointwise):
            return False
        if not all(r == 0 for r in self.mutable_args[0].data.data.data.ranges):
            return False

        return True

    @override
    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Generate wrapper code for the kernel without epilogue fusion.
        return self._codegen(wrapper, epilogue_fusion=None)

    def codegen_with_epilogue_fusion(
        self, wrapper: PythonWrapperCodegen, epilogue_fusion: tuple[ComputedBuffer, str]
    ) -> None:
        # [FUNCTION DOC] Generate wrapper code with a fused epilogue, given the fused epilogue buffer and the modified kernel source.
        """
        epilogue_fusion: (fused epilogue node, modified kerel src code)
        """
        return self._codegen(wrapper, epilogue_fusion)

    def _codegen(
        self,
        wrapper: PythonWrapperCodegen,
        epilogue_fusion: tuple[ComputedBuffer, str] | None,
    ) -> None:
        # [FUNCTION DOC] Shared codegen for the user kernel: fetches kernel and metadata; with an epilogue the mutable argument is redirected to the epilogue buffer (exactly one write required, AssertionError otherwise); defines the user-defined Triton kernel in the wrapper, collects named args (skipping constexprs when Triton uses attribute dicts, dummy values for unsupported constexpr or None args, removing None args otherwise; NotImplementedError for unsupported arg types) and emits the kernel launch call with raw args, grid and metadata.
        """Overrides the parent member.
        See https://github.com/pytorch/pytorch/issues/151692"""

        from torch._inductor.utils import triton_version_uses_attrs_dict

        (
            kernel,
            configs,
            restore_value_args,
            reset_to_zero_args,
        ) = self.get_kernel_and_metadata()

        # For an epilogue containing a cast, the output arg in kwargs must point to the
        # epilogue's output so that the compiled signature uses the correct dtype.
        kernel_kwargs = self.kwargs
        epilogue_out_override: dict[str, Any] = {}
        if epilogue_fusion:
            if len(self.arg_accesses.read_writes.writes) != 1:
                raise AssertionError(
                    f"expected one write, got {len(self.arg_accesses.read_writes.writes)}"
                )
            mutable_arg_name = next(iter(self.arg_accesses.read_writes.writes)).name
            epilogue_computed_buffer, _ = epilogue_fusion
            kernel_kwargs = {**self.kwargs, mutable_arg_name: epilogue_computed_buffer}
            epilogue_out_override = {mutable_arg_name: epilogue_computed_buffer}

        (
            new_name,
            triton_meta,
            inductor_meta,
            extra_launch_args,
        ) = wrapper.define_user_defined_triton_kernel(
            kernel,
            configs,
            kernel_kwargs,
            restore_value_args,
            reset_to_zero_args,
            self.grid,
            epilogue_fusion,
            self.launch_kwargs,
        )
        named_args = {
            k: self.get_kwargs_value(k) for k in self.ordered_kwargs_for_cpp_kernel
        }
        named_args.update(epilogue_out_override)

        arg_names = [p.name for p in kernel.params]  # type: ignore[attr-defined]
        constexprs = [p.num for p in kernel.params if p.is_constexpr]  # type: ignore[attr-defined]
        constexpr_names = OrderedSet(arg_names[i] for i in constexprs)

        args: list[Any] = []
        arg_types: list[Any] = []
        raw_keys_filtered: list[Any] = []
        raw_args_filtered: list[Any] = []
        for name, arg in itertools.chain(
            named_args.items(), zip(itertools.repeat(""), extra_launch_args)
        ):
            if name in constexpr_names and triton_version_uses_attrs_dict():
                # see #160000 - we don't pass in constexpr args to speed up runtime.
                continue
            raw_keys_filtered.append(name)
            raw_args_filtered.append(arg)
            if isinstance(arg, IRNode):
                args.append(arg.codegen_reference())
                arg_types.append(arg.get_dtype())
            elif isinstance(arg, (int, float, bool, Expr)):
                args.append(arg)
                arg_types.append(type(arg))
            elif name in constexpr_names:
                # insert a dummy value for constexpr args of unsupported type
                # constexprs will end up getting baked into the kernel at compile time
                args.append(-1)
                arg_types.append(int)
            elif arg is None:
                """
                Filter out None args.

                see https://github.com/pytorch/pytorch/issues/115344

                Two cases for a None arg:
                1. The arg is already tl.constexpr, so leave it in
                2. The arg is not tl.constexpr so we have to remove it
                """
                if triton_version_uses_attrs_dict():
                    args.append(-1)
                    arg_types.append(int)
                else:
                    raw_keys_filtered.pop()
                    raw_args_filtered.pop()
            else:
                raise NotImplementedError(f"Unsupported arg type: {type(arg)}: {arg}")

        self.codegen_comment(wrapper, new_name)
        wrapper.generate_kernel_call(
            new_name,
            args,
            arg_types=arg_types,
            raw_args=raw_args_filtered,
            raw_keys=raw_keys_filtered,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            triton=True,
            device=self.get_device(),
            original_fxnode_name=self.fx_node.name,
        )

    @cache_on_self_and_args("UserDefinedTritonKernel")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return free symbols of the parent (inputs, constant args, kwargs) plus those used in the launch grid (cached).
        # add unbacked symbols used in the grid to the ones used
        # in the kwargs (the latter is generated by ExternKernel)
        return super().get_free_symbol_uses(unbacked_only) | get_free_symbols(
            self.grid, unbacked_only
        )

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return an empty set of defined unbacked symbols.
        return OrderedSet()

    def __init__(
        self,
        *,
        kernel_idx: int,
        grid: Any,
        tma_descriptor_metadata: dict[str, Any],
        kernel_args: dict[str, Any],
        launch_kwargs: tuple[str, ...],
    ) -> None:
        # [FUNCTION DOC] Build the node from the kernel index, grid, TMA descriptor metadata, kernel arguments and launch kwargs. Tensor arguments are realized and unwrapped (wrapped as TMA descriptors when listed in the metadata) and become inputs and kwargs; other arguments become constant args. Requires at least one input (AssertionError otherwise); the device is the first input's. Records ordered kwarg names from the kernel signature, parses the kernel source, identifies stores and accessed tensors, derives the mutable tensor args (tensor writes only) and creates a MutationOutput for each. Registers the operation with the graph.
        inputs: list[IRNode] = []
        kwargs: dict[str, IRNode] = {}
        constant_args: list[IRNode] = []

        for k, v in kernel_args.items():
            if isinstance(v, TensorBox):
                t = InputsKernel.unwrap_storage_for_input(self.realize_input(v))
                if k in tma_descriptor_metadata:
                    t = TMADescriptor.create(t, tma_descriptor_metadata[k])
                inputs.append(t)
                kwargs[k] = t
            else:
                constant_args.append(v)
                kwargs[k] = v

        if len(inputs) == 0:
            raise AssertionError("Expected len(inputs) != 0")
        self.device = inputs[0].get_device()

        if not isinstance(inputs, Sequence):
            raise AssertionError(type(inputs))
        super().__init__(
            None,
            NoneLayout(device=self.device),
            inputs,
            tuple(constant_args),
            kwargs,
        )
        self.kernel_idx = kernel_idx
        self.grid = grid
        self.launch_kwargs = launch_kwargs

        kernel, configs, _, _ = self.get_kernel_and_metadata()

        # If we are autotuning, not all arguments will be passed
        if not hasattr(kernel, "arg_names"):
            raise AssertionError('Expected hasattr(kernel, "arg_names")')
        self.ordered_kwargs_for_cpp_kernel = [
            arg for arg in kernel.arg_names if arg in kernel_args
        ]

        from torch._higher_order_ops.triton_kernel_wrap import (
            identify_accessed_tensors,
            identify_triton_stores,
        )

        autotuned_kwargs = configs[0].kwargs if len(configs) > 0 else {}

        import ast

        # pyrefly: ignore [missing-attribute]
        self.kernel_src = kernel.src
        self.kernel_ast = ast.parse(self.kernel_src)
        self.kernel_stores = identify_triton_stores(self.kernel_src)
        self.kernel_args = kernel_args
        # names in `arg_accesses.read_writes` are names of formal arguments in the kernel's prototype
        self.arg_accesses = identify_accessed_tensors(
            kernel,
            {**kernel_args, **autotuned_kwargs},
            tma_descriptor_metadata,
        )

        # Filter to only tensor args: with Triton 3.7+, ordered_arg_names
        # includes scalars, so writes may reference non-tensor args like SymInts.
        self.mutable_args = [
            kernel_args[key.name]
            for key in self.arg_accesses.read_writes.writes
            if isinstance(kernel_args.get(key.name), TensorBox)
        ]

        self.mutation_outputs = [
            MutationOutput(NoneLayout(device=self.device), buf, self)
            for buf in self.mutable_args
        ]
        V.graph.register_operation(self)

    def get_outputs(self) -> list[Buffer]:
        # [FUNCTION DOC] Return the mutation outputs as the list of outputs (the kernel has no allocated output).
        return list(self.mutation_outputs)

    def get_device(self) -> torch.device | None:
        # [FUNCTION DOC] Return the device of the first input.
        return self.device


class InplaceBernoulliFallback(ExternKernel):
    # [CLASS DOC] ExternKernel for in-place bernoulli sampling on a tensor, a custom class so mutation is tracked. Allocates nothing, mutates the first input, and in cpp wrapper passes a null generator argument.
    """
    This needs to be a custom class to handle mutation properly
    """

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Generate a wrapper line calling the kernel on the input with the constant args (and a null generator in cpp wrapper mode). AssertionError if inputs are not IR nodes.
        if not all(isinstance(t, IRNode) for t in self.inputs):
            raise AssertionError(
                "Expected all(isinstance(t, IRNode) for t in self.inputs)"
            )
        (x,) = (cast(IRNode, t).codegen_reference() for t in self.inputs)

        if V.graph.cpp_wrapper:
            # Inductor doesn't really support aten Generator, so the Generator kwarg is always NULL here,
            # which needs to be explicitly generated for cpp wrapper
            wrapper.writeline(
                f"{self.get_kernel_name()}({x}, {', '.join(map(repr, self.constant_args))}, NULL){wrapper.ending}"
            )
        else:
            wrapper.writeline(
                f"{self.get_kernel_name()}({x}, {', '.join(map(repr, self.constant_args))}){wrapper.ending}"
            )

    def should_allocate(self) -> bool:
        # [FUNCTION DOC] Return False.
        return False

    def get_mutation_names(self) -> Sequence[str]:
        # [FUNCTION DOC] Return the name of the mutated first input as a list.
        return [self.input_name(0)]

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return an empty set.
        return OrderedSet()

    def __init__(
        self, op_overload: _OpOverloads, x: IRNode, *constant_args: Any
    ) -> None:
        # [FUNCTION DOC] Initialize with an empty-layout output on the input device, the unwrapped input and constant args; mark the input buffer as mutated and register buffer and operation.
        super().__init__(
            None,
            NoneLayout(device=x.get_device()),
            self.unwrap_storage([x]),
            constant_args,
            op_overload=op_overload,
        )
        V.graph.mark_buffer_mutated(x.get_name())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)


# Used to deal with torch.complex types
class InplaceCopyFallback(ExternKernel):
    # [CLASS DOC] ExternKernel for an in-place copy into the first input (used for complex dtypes), custom so mutation is tracked. Calls the device copy codegen (dst, src, non_blocking).
    """
    This needs to be a custom class to handle mutation properly
    """

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Generate the device copy from the source to the destination with the non-blocking flag.
        (dst, src, non_blocking) = self.codegen_args()
        wrapper.codegen_device_copy(src, dst, non_blocking)

    def should_allocate(self) -> bool:
        # [FUNCTION DOC] Return False.
        return False

    def get_mutation_names(self) -> Sequence[str]:
        # [FUNCTION DOC] Return the name of the mutated first input as a list.
        return [self.input_name(0)]

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return an empty set.
        return OrderedSet()

    def __init__(
        self,
        layout: OutputSpec,
        inputs: Sequence[IRNode],
        constant_args: Sequence[Any],
    ) -> None:
        # [FUNCTION DOC] Initialize the copy kernel (python name aten copy_, cpp name aoti torch copy_), mark the destination (first input) as mutated and register buffer and operation.
        super().__init__(
            None,
            layout,
            inputs,
            constant_args,
            python_kernel_name="aten.copy_",
            cpp_kernel_name="aoti_torch_copy_",
        )
        V.graph.mark_buffer_mutated(inputs[0].get_name())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    @classmethod
    def create(
        cls, dst: IRNode, src: IRNode, non_blocking: bool = False
    ) -> InplaceCopyFallback:
        # [FUNCTION DOC] Class method creating the in-place copy from src into dst: realizes both inputs, takes the non_blocking flag as a constant arg and uses an empty layout on dst's device. Returns the new node.
        inputs = [cls.realize_input(t) for t in [dst, src]]
        constant_args = (non_blocking,)
        result = InplaceCopyFallback(
            NoneLayout(device=dst.get_device()),
            inputs,
            constant_args,
        )
        return result


class MutatingFirstArgExternKernel(ExternKernel):
    # [CLASS DOC] ExternKernel that mutates its first input and returns nothing; has side effects, allocates nothing, and codegen emits a plain call with inputs followed by repr'd constants.
    """
    This needs to be a custom class to handle mutation properly
    """

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Generate a call line using the kernel name with input references and repr of constant args; AssertionError if inputs are not IR nodes.
        if not is_node_sequence(self.inputs):
            raise AssertionError("Expected is_node_sequence(self.inputs)")
        argrefs = [
            *(t.codegen_reference() for t in self.inputs),
            *map(repr, self.constant_args),
        ]
        wrapper.writeline(
            f"{self.get_kernel_name()}({', '.join(argrefs)}){wrapper.ending}"
        )

    def should_allocate(self) -> bool:
        # [FUNCTION DOC] Return False.
        return False

    def get_mutation_names(self) -> Sequence[str]:
        # [FUNCTION DOC] Return the name of the mutated first input as a list.
        return [self.input_name(0)]

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return an empty set.
        return OrderedSet()

    def has_side_effects(self) -> bool:
        # [FUNCTION DOC] Return True (the kernel has side effects).
        return True


class ResizeStorageBytes(MutatingFirstArgExternKernel):
    # [CLASS DOC] MutatingFirstArgExternKernel that resizes a tensor's storage to new_size bytes using the inductor ops resize function. The variable's buffer is never reused. Dynamic sizes are not supported.
    def __init__(self, variable: IRNode, new_size: int) -> None:
        # [FUNCTION DOC] Initialize with the variable and a static integer size (AssertionError otherwise or if the variable is not a view or box), mark the buffer mutated, register it, set python and cpp kernel names and forbid buffer reuse.
        if not isinstance(new_size, int):
            raise AssertionError("TODO: dynamic shapes")
        super().__init__(
            None,
            NoneLayout(device=variable.get_device()),
            self.unwrap_storage([variable]),
            constant_args=(new_size,),
        )
        V.graph.mark_buffer_mutated(variable.get_name())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)
        self.python_kernel_name = "inductor_ops.resize_storage_bytes_"
        self.cpp_kernel_name = "torch::inductor::resize_storage_bytes_"
        if not isinstance(variable, (BaseView, StorageBox, TensorBox)):
            raise AssertionError(type(variable))
        V.graph.never_reuse_buffers.add(variable.data.get_name())


class SetSourceTensorKernel(ExternKernelAlloc):
    # [CLASS DOC] ExternKernelAlloc for the aten set_ with source tensor: re-points a tensor at another's storage. Outputs are mutation outputs for both tensors; all involved buffers are never reused; output aliases both inputs.
    def __init__(self, self_tensor: IRNode, storage_tensor: IRNode) -> None:
        # [FUNCTION DOC] Realize and freeze the storage tensor, initialize with its layout and both inputs, forbid reuse of the buffers, and create MutationOutput nodes for both tensors. AssertionError if self_tensor is not a view or box.
        storage_tensor = self.realize_input(storage_tensor)
        storage_tensor.freeze_layout()
        super().__init__(
            storage_tensor.get_layout(),
            [self_tensor, storage_tensor],
            python_kernel_name="torch.ops.aten.set_.source_Tensor",
            op_overload=torch.ops.aten.set_.source_Tensor,
        )
        if not isinstance(self_tensor, (BaseView, StorageBox, TensorBox)):
            raise AssertionError(type(self_tensor))
        V.graph.never_reuse_buffers.add(self_tensor.data.get_name())
        V.graph.never_reuse_buffers.add(storage_tensor.get_name())
        V.graph.never_reuse_buffers.add(self.get_name())
        device = storage_tensor.get_device()
        self.mutation_outputs = [
            MutationOutput(NoneLayout(device=device), self_tensor, self),
            MutationOutput(NoneLayout(device=device), storage_tensor, self),
        ]

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        # [FUNCTION DOC] Return the names of both inputs.
        return [self.input_name(0), self.input_name(1)]


class ShallowCopyDataKernel(ExternKernelAlloc):
    # [CLASS DOC] ExternKernelAlloc for shallow copy of data from one tensor to another; same structure as SetSourceTensorKernel (mutation outputs for both, no buffer reuse). KNOWN ISSUE: the op overload is set_.source_Tensor while the python kernel name is the shallow copy data op.
    def __init__(self, self_tensor: IRNode, storage_tensor: IRNode) -> None:
        # [FUNCTION DOC] Realize and freeze the storage tensor, initialize with its layout and both inputs, forbid reuse of the buffers, and create MutationOutput nodes for both tensors. AssertionError if self_tensor is not a view or box.
        storage_tensor = self.realize_input(storage_tensor)
        storage_tensor.freeze_layout()
        super().__init__(
            storage_tensor.get_layout(),
            [self_tensor, storage_tensor],
            python_kernel_name="torch.ops.aten.shallow_copy_data_",
            op_overload=torch.ops.aten.set_.source_Tensor,
        )
        if not isinstance(self_tensor, (BaseView, StorageBox, TensorBox)):
            raise AssertionError(type(self_tensor))
        V.graph.never_reuse_buffers.add(self_tensor.data.get_name())
        V.graph.never_reuse_buffers.add(storage_tensor.get_name())
        V.graph.never_reuse_buffers.add(self.get_name())
        device = storage_tensor.get_device()
        self.mutation_outputs = [
            MutationOutput(NoneLayout(device=device), self_tensor, self),
            MutationOutput(NoneLayout(device=device), storage_tensor, self),
        ]

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        # [FUNCTION DOC] Return the names of both inputs.
        return [self.input_name(0), self.input_name(1)]


class ScatterFallback(ExternKernel):
    # [CLASS DOC] ExternKernel for scatter_ and scatter_reduce_ that mutates the first input; supports a scalar or tensor source (src_is_tensor). Kernel keyword args are reduce and include_self.
    """
    This needs to be a custom class to handle mutation properly.
    This class handles both aten.scatter_ and aten.scatter_reduce_.
    It also handle the case `src` being a scalar properly.
    """

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Generate wrapper code via the scatter fallback generator.
        wrapper.generate_scatter_fallback(self)

    def should_allocate(self) -> bool:
        # [FUNCTION DOC] Return False.
        return False

    def get_mutation_names(self) -> list[str]:
        # [FUNCTION DOC] Return the mutated first input name as a list; AssertionError if it is not an IR node.
        inp = self.inputs[0]
        if not isinstance(inp, IRNode):
            raise AssertionError("Expected isinstance(inp, IRNode)")
        return [inp.get_name()]

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return an empty set.
        return OrderedSet()

    def __init__(
        self,
        op_overload: _OpOverloads,
        x: IRNode,
        dim: int,
        index: IRNode,
        src: IRNode,
        *,
        reduce: str | None = None,
        include_self: bool = True,
    ) -> None:
        # [FUNCTION DOC] Initialize: realize the tensor inputs (x, index and, if tensor, src); a scalar src becomes a constant arg alongside the dim; register the kernel with reduce and include_self kwargs, mark x mutated, register buffer and operation.
        self.src_is_tensor = isinstance(src, TensorBox)

        constant_args: tuple[Any, ...]
        if self.src_is_tensor:
            tensors = [self.realize_input(t) for t in [x, index, src]]
            constant_args = (dim,)
        else:
            tensors = [self.realize_input(t) for t in [x, index]]
            constant_args = (dim, src)

        super().__init__(
            None,
            NoneLayout(device=x.get_device()),
            self.unwrap_storage(tensors),
            constant_args,
            {"reduce": reduce, "include_self": include_self},
            python_kernel_name=str(op_overload),
            ordered_kwargs_for_cpp_kernel=["reduce", "include_self"],
            op_overload=op_overload,
        )
        V.graph.mark_buffer_mutated(x.get_name())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)


class IndexPutFallback(ExternKernel):
    # [CLASS DOC] ExternKernel for index_put_ that mutates its first input; stores the list of indices (which may contain None) and registers realized inputs x, values and the non-None indices; accumulate is a constant arg.
    """
    This needs to be a custom class to handle mutation and indices properly
    """

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Generate wrapper code via the index-put fallback generator.
        wrapper.generate_index_put_fallback(self)

    def should_allocate(self) -> bool:
        # [FUNCTION DOC] Return False.
        return False

    def get_mutation_names(self) -> Sequence[str]:
        # [FUNCTION DOC] Return the mutated first input name as a list.
        return [self.input_name(0)]

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return an empty set.
        return OrderedSet()

    def __init__(
        self,
        op_overload: torch._ops.OpOverload,
        x: IRNode,
        indices: list[Any],
        values: Sequence[Any],
        accumulate: Any,
    ) -> None:
        # [FUNCTION DOC] Initialize with the realized inputs, record indices, mark the first input mutated and register buffer and operation.
        self.indices = indices
        valid_indices = [i for i in indices if i is not None]
        # pyrefly: ignore [bad-argument-type]
        tensors = [self.realize_input(x) for x in [x, values, *valid_indices]]
        cpp_kernel_name = "aoti_torch_index_put_out"
        super().__init__(
            None,
            NoneLayout(device=x.get_device()),
            self.unwrap_storage(tensors),
            (accumulate,),
            python_kernel_name="aten.index_put_",
            cpp_kernel_name=cpp_kernel_name,
            op_overload=op_overload,
        )
        V.graph.mark_buffer_mutated(self.input_name(0))
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)


class DeviceCopy(ExternKernelOut):
    # [CLASS DOC] ExternKernelOut copying a tensor between devices (host/device transfers). create may instead turn constant inputs into constants on the target device without a copy.
    @classmethod
    def create(cls, x: IRNode, device: torch.device, non_blocking: bool) -> IRNode:
        # [FUNCTION DOC] Class method creating a device copy. If x is not extern, never mutated, reads only graph constants and runtime constant folding is off, returns x moved to the device as a constant (registering device info for cpp wrapper). Otherwise registers both devices, makes x contiguous, preserves its stride and sets pinned flags (a pinned destination for gpu to cpu non-blocking copies; marks the source layout pinned for cpu to gpu non-blocking copies), and returns a DeviceCopy with a fixed layout on the target device. AssertionError if x has no device.
        x_device = x.get_device()
        if x_device is None:
            raise AssertionError("Expected x_device is not None")
        if (
            not x.is_extern()
            # Can not apply this optimization if x has been mutated
            and try_get_name(x) not in V.graph.mutated_buffers
            and all(r in V.graph.constants for r in x.get_read_names())
            and not config.aot_inductor.use_runtime_constant_folding
        ):
            if V.graph.cpp_wrapper:
                # Even if x is promoted to be a device constant, we still need to
                # register device info to construct the correct CppWrapper class later
                V.graph.add_device_info(device)
                V.graph.add_device_info(x_device)
            return x.constant_to_device(device)

        V.graph.add_device_info(device)
        V.graph.add_device_info(x_device)
        developer_warning("DeviceCopy in input program")
        constant_args = (non_blocking,)
        # Device Copy should keep the same layout as input
        x = ExternKernel.require_contiguous(x)
        stride = None
        if x.get_size():
            # x.get_stride() may be unimplemented if x's size is empty
            stride = x.get_stride()
        is_destination_pinned = (
            is_gpu(x_device.type) and device.type == "cpu" and non_blocking
        )
        is_source_pinned = (
            x_device.type == "cpu" and is_gpu(device.type) and non_blocking
        )
        if is_source_pinned and is_storage_and_layout(x):
            x.get_layout().is_pinned = True
        return DeviceCopy(
            FixedLayout(
                device,
                x.get_dtype(),
                x.get_size(),
                stride,
                is_pinned=is_destination_pinned,
            ),
            [cls.realize_input(x)],
            constant_args,
        )

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Generate the device copy code from the input to this buffer (or the output view) with the non-blocking flag, and synchronize the device-to-host copy if the output is pinned. AssertionError if the argument count is not two.
        args = self.codegen_args()
        if len(args) != 2:
            raise AssertionError("Expected len(args) == 2")
        if self.output_view:
            wrapper.codegen_device_copy(
                args[0], self.output_view.codegen_reference(), args[1]
            )
        else:
            wrapper.codegen_device_copy(args[0], self.codegen_reference(), args[1])
        if isinstance(self.layout, Layout) and self.layout.is_pinned:
            wrapper.sync_d2h_copy(self.get_name())


class DynamicSelectStorageOffset(ExternKernel):
    # [CLASS DOC] Node defining an unbacked storage offset symbol for a select with an unbacked index: offset equals base offset plus stride times (index, or index plus size when negative); optional clamping. Lets the select decompose into as_strided with the offset computed at runtime. Has no reads and no allocation.
    """
    The result of computing a dynamic selection index is determined as follows: when the index in the
    select operation is unbacked, the actual index calculation is ambiguous for negative indices
    (index + size) versus non-negative indices (just index). To resolve this, we allocate an unbacked
    SymInt to represent the storage offset and decompose the select operation into a call to as_strided,
    computing the storage offset at runtime with this node.
    """

    def get_reads(self) -> OrderedSet[Dep]:
        # [FUNCTION DOC] Return an empty set of reads.
        return OrderedSet()

    def should_allocate(self) -> bool:
        # [FUNCTION DOC] Return False.
        return False

    def __init__(
        self,
        unbacked_offset_symbol: sympy.Symbol,
        index: sympy.Symbol,
        base_offset: sympy.Symbol | int,
        base_dim_stride: sympy.Symbol | int,
        size: sympy.Symbol | int,
        clamp: bool,
    ) -> None:
        # [FUNCTION DOC] Initialize with an empty layout on cpu and no inputs; store the offset symbol, index, base offset, base dim stride, size and clamp flag.
        super().__init__(None, NoneLayout(device=torch.device("cpu")), [])
        # This node codegen the following:
        # unbacked_offset_symbol = base_offset + base_dim_stride * (index if index >=0 else index + size)
        self.unbacked_offset_symbol = unbacked_offset_symbol
        self.index = index
        self.base_offset = base_offset
        self.base_dim_stride = base_dim_stride
        self.size = size
        self.clamp = clamp

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return the set containing the defined offset symbol.
        return OrderedSet([self.unbacked_offset_symbol])

    @cache_on_self_and_args("DynamicSelectStorageOffset")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return the free symbols of the index, size, base offset and base dim stride (cached).
        return (
            get_free_symbols(self.index, unbacked_only)
            .union(get_free_symbols(self.size, unbacked_only))
            .union(get_free_symbols(self.base_offset, unbacked_only))
            .union(get_free_symbols(self.base_dim_stride, unbacked_only))
        )

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Generate wrapper code computing the dynamic select index, with clamping per the flag.
        wrapper.codegen_dynamic_select_index(self, clamp=self.clamp)


class DynamicSliceSize(ExternKernel):
    # [CLASS DOC] Node defining an unbacked output size symbol for a slice call whose start and end semantics (negative out of bounds, negative, standard, positive out of bounds) cannot be decided statically; codegen computes the right case at runtime. Has no reads and no allocation.
    """
    Computes the output size of a slice call, handling the correct semantics in codegen.
    We do this for flexible handling for unbacked indices (to not data-dependent error).

    Slicing has 4 semantics for indices, i.e. x[start:] could be:
    1) start < -x.size(0)            -> x[0:]                    # negative out-of-bounds
    2) start in [-x.size(0), 0)      -> x[x.size(0) + start:]    # negative slicing
    3) start in [0, x.size(0))       -> x[start:]                # standard slicing
    4) start >= x.size(0)            -> empty slice              # positive out-of-bounds

    If the appropriate semantics are known beforehand, the output size is computed based on
    the start & end indices. If not (with unbacked indices), a new unbacked symbol is created
    to represent the output size, and codegen handles computing the correct case.
    """

    def get_reads(self) -> OrderedSet[Dep]:
        # [FUNCTION DOC] Return an empty set of reads.
        return OrderedSet()

    def should_allocate(self) -> bool:
        # [FUNCTION DOC] Return False.
        return False

    def __init__(
        self,
        unbacked_size_symbol: sympy.Symbol,
        start: sympy.Symbol | int,
        end: sympy.Symbol | int,
        step: sympy.Symbol | int,
        size: sympy.Symbol | int,
    ):
        # [FUNCTION DOC] Initialize with an empty layout on cpu and no inputs; store the size symbol, start, end, step and input size.
        super().__init__(None, NoneLayout(device=torch.device("cpu")), [])
        # This node codegen
        self.unbacked_size_symbol = unbacked_size_symbol
        self.start = start
        self.end = end
        self.step = step
        self.size = size

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return the set containing the defined size symbol.
        return OrderedSet([self.unbacked_size_symbol])

    @cache_on_self_and_args("DynamicSliceSize")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return the free symbols of start, end, size and step (cached).
        return (
            get_free_symbols(self.start, unbacked_only)
            .union(get_free_symbols(self.end, unbacked_only))
            .union(get_free_symbols(self.size, unbacked_only))
            .union(get_free_symbols(self.step, unbacked_only))
        )

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Generate wrapper code computing the dynamic slice size.
        wrapper.codegen_dynamic_slice_size(self)


class DynamicScalar(ExternKernel):
    # [CLASS DOC] Node for the result of a local scalar dense call: defines an unbacked symbol (sym) from data read from a tensor (keypath describes the location of the value). No reads recorded and no allocation.
    """
    The result of a call to aten._local_scalar_dense.
    """

    def get_reads(self) -> OrderedSet[Dep]:
        # [FUNCTION DOC] Return an empty set of reads.
        return OrderedSet()

    def should_allocate(self) -> bool:
        # [FUNCTION DOC] Return False.
        return False

    def __init__(
        self, sym: sympy.Symbol, keypath: pytree.KeyPath, data: IRNode
    ) -> None:
        # [FUNCTION DOC] Realize the data, initialize with an empty cpu layout and the unwrapped data input, and store the symbol and key path.
        data.realize()
        super().__init__(
            None, NoneLayout(device=torch.device("cpu")), self.unwrap_storage([data])
        )
        self.sym = sym
        self.keypath = keypath

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return the set containing the defined symbol.
        return OrderedSet([self.sym])

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Generate wrapper code extracting the dynamic scalar.
        wrapper.codegen_dynamic_scalar(self)


class AssertScalar(ExternKernel):
    # [CLASS DOC] Node for the result of an assert on a scalar condition (sympy boolean) with a message; has side effects so it is not removed, no reads and no allocation.
    """
    The result of a call to aten._assert_scalar
    """

    def get_reads(self) -> OrderedSet[Dep]:
        # [FUNCTION DOC] Return an empty set of reads.
        return OrderedSet()

    def should_allocate(self) -> bool:
        # [FUNCTION DOC] Return False.
        return False

    def __init__(self, scalar: SympyBoolean, msg: str) -> None:
        # [FUNCTION DOC] Initialize with an empty cpu layout and no inputs and store the condition and message.
        super().__init__(
            # Buffer(name, layotu)
            None,
            NoneLayout(device=torch.device("cpu")),
            # InputsKernel(inputs)
            [],
        )
        self.scalar = scalar
        self.msg = msg

    def has_side_effects(self) -> bool:
        # [FUNCTION DOC] Return True.
        return True

    @cache_on_self_and_args("AssertScalar")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return the free symbols of the condition (cached).
        return get_free_symbols(self.scalar, unbacked_only)

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Generate a runtime assertion for the scalar condition when config.scalar_asserts is on. The condition is deliberately not simplified (simplification uses runtime asserts and would make it trivially true). Python wrapper: emits a check raising a RuntimeError with the message and defines this buffer name as None; cpp wrapper: emits a check throwing a runtime error with the message and the offending symbol value; fx wrapper: does nothing yet.
        if not config.scalar_asserts:
            return
        # NB: It is EXTREMELY important not to simplify the scalar under assertion here,
        # because simplify is done with respect to runtime asserts.  So if you have
        # "u0 == 0" in the runtime asserts, if you subsequently try to
        # simplify(u0 == 0), you will get True (because we've already runtime assert'ed
        # that it's true).  But we're code generating the actual runtime assert here!!
        symbol = next(iter(self.get_free_symbol_uses(unbacked_only=False)))
        if V.graph.fx_wrapper:
            # TODO fix
            pass
        elif V.graph.cpp_wrapper:
            symbol_str = f"std::to_string({symbol})"
            sizevar = V.graph.wrapper_code.codegen_cpp_sizevar(
                self.scalar, simplify=False
            )
            # TODO: when we start compiling in C++20, annotate with [[unlikely]].
            wrapper.writeline(
                f'if (!({sizevar})) {{ throw std::runtime_error("Expected {self.msg} but received " + {symbol_str}); }}'
            )
        else:
            sizevar = V.graph.wrapper_code.codegen_python_sizevar(
                self.scalar, simplify=False
            )
            wrapper.writeline(f"if not ({sizevar}):")
            wrapper.writeline(f"    raise RuntimeError({repr(self.msg)})")
            # No one should ever use this buffer, but for uniformity
            # define the variable and assign it None
            wrapper.writeline(f"{self.get_name()} = None")


@ir_dataclass(frozen=False)
class ExternKernelNode:
    # [CLASS DOC] Mutable record pairing an extern kernel node name with its serialized export schema node (used for proxy executor serialization).
    name: str
    node: export_schema.Node


class FallbackKernel(ExternKernelAlloc):
    # [CLASS DOC] ExternKernelAlloc for operators Inductor does not lower: functional ops, view ops, in-place aten ops and auto-functionalizable mutating ops (and higher-order ops assumed functional). Tracks alias_names (arguments aliased by outputs), mutation_names (arguments mutated and returned), unbacked bindings and whether to use runtime dispatch in cpp wrapper. Codegen emits a direct fallback call, or a runtime-lookup call via the proxy executor when no C shim exists.
    """
    A class that represents a fallback kernel for handling operators that are not
    directly support by inductor. It currently supports functional ops, view ops,
    inplace aten ops, and mutating ops that are auto-functionalizable.
    """

    def __init__(
        self,
        layout: OutputSpec,
        kernel: _OpOverloads,
        tensor_args: Sequence[IRNode],
        nontensor_args: Sequence[Any],
        unflatten_args: Callable[..., Any],
        kwargs: dict[str, Any] | None = None,
        *,
        unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] | None = None,
    ) -> None:
        # [FUNCTION DOC] Initialize as an allocating extern kernel with the op overload, then record the unflatten function, kwargs and unbacked bindings and warn about the fallback. AssertionError if the kernel is not an OpOverload or HigherOrderOperator. For regular ops, derives alias and mutation info: ops that mutate and return the first arg record it; otherwise mutating ops must be auto-functionalizable or have a functionalize implementation (NotImplementedError otherwise), and every aliasing schema argument is recorded (written ones also get MutationOutput nodes). Collective functional ops and higher-order ops skip this analysis.
        super().__init__(
            layout,
            tuple(tensor_args),
            tuple(nontensor_args),
            op_overload=kernel,
        )

        self.use_runtime_dispatch = False
        self.unbacked_bindings = unbacked_bindings or {}

        if not isinstance(
            kernel, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)
        ):
            raise AssertionError(
                f"Fails to create FallbackKernel for {kernel}: {type(kernel)} not supported"
            )
        self.op_overload = kernel
        self.unflatten_args = unflatten_args
        self.kwargs = {} if kwargs is None else kwargs
        if self.python_kernel_name is None:
            raise AssertionError("Expected self.python_kernel_name is not None")
        V.graph.warn_fallback(self.python_kernel_name)

        # args that are aliased
        self.alias_names: list[str] = []
        # args that are mutated AND returned from the op
        self.mutation_names: list[str] = []

        if isinstance(self.op_overload, torch._ops.HigherOrderOperator):
            # We assume here that HOPs with FallbackKernel are functional.
            # This may not always be true! HOPs must individually opt-in to
            # FallbackKernel, so please check this if you opt-in.
            return

        if "_c10d_functional" in self.op_overload.name():
            # _c10d_functional kernels are lowered into _CollectiveKernel which
            # derives from FallbackKernel for the cpp codegen. The kernels
            # don't pass the can_auto_functionalize check, but their mutation
            # is handled properly by _CollectiveKernel.
            return

        schema = self.op_overload._schema

        # NOTE: [FallbackKernel supported operators]
        # We only support three types of operators:
        # - functional ops
        # - view ops
        # - inplace aten ops
        # - mutating ops that are auto-functionalizable. That is,
        # the operator may mutate any number of inputs, but its outputs
        # may not alias any of the inputs.
        #
        # The unsupported cases usually do not show up here (because
        # AOTAutograd functionalized them away); the only way for an in-place
        # op to show up here is if a lowering or pass introduced it.
        if torch._library.utils.mutates_and_returns_first_arg(self.op_overload):
            self.mutation_names.append(tensor_args[0].get_name())
            # Record aliasing relationship so memory planning doesn't wrongly
            # reuse its storage.
            self.alias_names.append(tensor_args[0].get_name())
            return

        def has_functionalize_impl(op: torch._ops.OpOverload) -> bool:
            # [FUNCTION DOC] HELPER: Return True if the op has a Functionalize dispatch kernel in C++ or Python.
            return torch._C._dispatch_has_kernel_for_dispatch_key(
                op.name(), torch._C.DispatchKey.Functionalize
            ) or (
                hasattr(op, "py_kernels")
                and torch._C.DispatchKey.Functionalize in op.py_kernels
            )

        if (
            schema.is_mutable
            and not can_auto_functionalize(self.op_overload)
            and not has_functionalize_impl(self.op_overload)
        ):
            raise NotImplementedError(
                f"NYI: Can't generate FallbackKernel for {self.op_overload}"
            )

        args, kwargs = self.unflatten_args(self.inputs, self.constant_args)

        def handle_aliasing_and_mutation(info: torch._C.Argument, arg: Any) -> None:
            # [FUNCTION DOC] HELPER: Check one schema argument against its value (lists for list types, no lists for tensor types; AssertionError otherwise) and, if the argument carries alias info, record aliased graph buffers (adding mutation outputs for written ones).
            # Assertions to make sure we didn't mismatch args
            if isinstance(info.type, torch.ListType):
                if not isinstance(arg, (list, tuple)):
                    raise AssertionError(type(arg))
            if library_utils.is_tensor_like_type(info.type):
                # PyTorch also accepts None and scalar types for args marked as "Tensor".
                # We're not going to check all of them here.
                if isinstance(arg, (tuple, list)):
                    raise AssertionError("Expected not isinstance(arg, (tuple, list))")

            if arg is None:
                return
            if info.alias_info is None:
                return

            def add_alias(t: IRNode) -> None:
                # [FUNCTION DOC] HELPER: Record the node's buffer name as aliased and, if the alias is written, add a MutationOutput for it.
                self.alias_names.append(t.get_name())
                if info.alias_info is None:
                    raise AssertionError("Expected info.alias_info is not None")
                if info.alias_info.is_write:
                    self.mutation_outputs.append(
                        MutationOutput(NoneLayout(device=t.get_device()), t, self)
                    )

            def add_alias_if_graph_buffer(t: Any) -> None:
                # [FUNCTION DOC] HELPER: Record aliasing for a tensor-like value, ignoring None and scalars boxed as tensors; AssertionError for non-IR values.
                if t is None:
                    return
                if isinstance(t, (Number, Expr, *SymTypes)):
                    # The dispatcher accepts Python scalars for some Tensor
                    # schema arguments by boxing them into temporary scalar tensors.
                    # Those temporaries cannot alias or mutate graph buffers.
                    return
                if not isinstance(t, IRNode):
                    raise AssertionError(type(t))
                add_alias(t)

            if library_utils.is_tensorlist_like_type(info.type):
                if arg is not None:
                    for optional_tensor_arg in arg:
                        add_alias_if_graph_buffer(optional_tensor_arg)
            else:
                if not library_utils.is_tensor_like_type(info.type):
                    raise AssertionError(
                        "Expected library_utils.is_tensor_like_type(info.type)"
                    )

                add_alias_if_graph_buffer(arg)

        for info, arg in torch._library.utils.zip_schema(schema, args, kwargs):
            handle_aliasing_and_mutation(info, arg)

    def get_read_writes(self) -> dependencies.ReadWrites:
        # [FUNCTION DOC] Return the extern read/writes, adding a read dependency on generator state arguments for the graph-safe rng state run op.
        read_writes = super().get_read_writes()

        if self.op_overload is torch._prims.rng_prims.graphsafe_run_with_rng_state:
            for arg in self.constant_args:
                if isinstance(arg, GeneratorState):
                    read_writes = read_writes.with_read(
                        dependencies.StarDep(arg.get_name())
                    )

        return read_writes

    def codegen_unbacked_symbol_defs(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Generate wrapper code defining unbacked symbols produced by the outputs.
        return wrapper.codegen_unbacked_symbol_defs_for_outputs(
            self.get_name(),
            self.codegen_outputs(),
            getattr(self, "unbacked_bindings", None),
        )

    def codegen_outputs(self) -> Sequence[Any]:
        # [FUNCTION DOC] Return the output nodes to codegen: the explicit outputs if set, else this node when it has a tensor layout, else the mutation outputs.
        if self.outputs:
            return self.outputs
        if isinstance(self.layout, Layout):
            return [self]
        return self.mutation_outputs

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return the unbacked symbols defined by this kernel, resolved from the stored bindings via the shape env (empty if none; AssertionError if resolution fails).
        if unbacked_bindings := getattr(self, "unbacked_bindings", None):
            resolved = resolve_unbacked_bindings(
                V.graph.sizevars.shape_env, unbacked_bindings
            )
            if resolved is None:
                raise AssertionError("Expected resolved is not None")
            return OrderedSet(resolved.keys())
        else:
            return OrderedSet()

    def codegen_args(self) -> list[str]:
        # [FUNCTION DOC] Return code strings for the call args: tensor inputs are replaced by their codegen references, args are rebuilt via unflatten_args, filled with defaults and typed for cpp wrapper with an OpOverload, and formatted by the wrapper. Keyword args from unflattening are merged into self.kwargs (a side effect) for codegen_kwargs to emit. AssertionError if inputs are not IR nodes.
        @dataclasses.dataclass
        class Shim:
            # [CLASS DOC] HELPER: Small wrapper whose repr is the stored string, so code references print unquoted.
            ref: Any

            def __repr__(self) -> str:
                # [FUNCTION DOC] HELPER: Return the stored reference string.
                return self.ref

        if not is_node_sequence(self.inputs):
            raise AssertionError("Expected is_node_sequence(self.inputs)")
        tensor_args = [Shim(x.codegen_reference()) for x in self.inputs]
        args, kwargs = self.unflatten_args(tensor_args, self.constant_args)
        if V.graph.cpp_wrapper and isinstance(self.op_overload, torch._ops.OpOverload):
            args = self.fill_non_provided_args(args, kwargs)
            args = [
                V.graph.wrapper_code.val_to_arg_str(x, param.real_type)
                for param, x in zip(self.op_overload._schema.arguments, args)
            ]
        else:
            args = [V.graph.wrapper_code.val_to_arg_str(x) for x in args]

        # let self.codegen_kwargs handle kwargs
        self.kwargs.update(kwargs)
        return args

    @staticmethod
    def find_device(
        tensor_args: Sequence[IRNode] | None, example_output: Sequence[Any]
    ) -> Any:
        # [FUNCTION DOC] Static method finding the device for outputs: from the tensor args (excluding torchbind objects) the first one with a device; otherwise from the example output (tensor device, cpu for script or custom class objects), recursing into lists and tuples and preferring a GPU device among several; None if undeterminable. KNOWN ISSUE: with tensor args but none having a device, indexing the empty device list would fail.
        non_torch_bind_tensor_args = (
            [t for t in tensor_args if not isinstance(t, TorchBindObject)]
            if tensor_args
            else None
        )
        if non_torch_bind_tensor_args:
            if not tensor_args:
                raise AssertionError("Expected tensor_args")
            devices = [arg.get_device() for arg in tensor_args if arg.get_device()]
            return devices[0]
        if isinstance(example_output, torch.Tensor):
            return example_output.device
        if isinstance(
            example_output, (torch._C.ScriptObject, FakeScriptObject)
        ) or is_custom_class_obj(example_output):
            return torch.device("cpu")
        if isinstance(example_output, (list, tuple)):
            device_set = OrderedSet(
                # pyrefly: ignore [bad-argument-type]
                FallbackKernel.find_device(None, x)
                for x in example_output
            )
            # Remove None
            devices = [device for device in device_set if device]
            if len(devices) == 1:
                return devices[0]
            if not devices:
                return None
            for device in devices:
                if not isinstance(device, torch.device):
                    raise AssertionError("Expected isinstance(device, torch.device)")
                if is_gpu(device.type):
                    return device
            return devices[0]
        return None

    def has_side_effects(self) -> bool:
        # [FUNCTION DOC] Return whether the op is impure according to the library utilities, evaluated on the op overload only.
        from torch._library.utils import is_impure

        # Note: We don't pass args/kwargs here because they're IRNodes, not actual values
        # The check is done on the op_overload itself
        return is_impure(self.op_overload)  # pyrefly: ignore[bad-argument-type]

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        # [FUNCTION DOC] Return the aliased input names; empty for auto-functionalizable mutating ops (their outputs do not alias inputs). AssertionError for unsupported op types.
        if not isinstance(
            self.op_overload, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)
        ):
            raise AssertionError(
                f"Fails to create FallbackKernel for {self.op_overload}: "
                f"{type(self.op_overload)} not supported"
            )

        # See [Note: FallbackKernel supported operators]: for a mutating
        # op that is auto-functionalizable, its outputs does NOT
        # alias any of the inputs.
        if (
            not isinstance(self.op_overload, torch._ops.HigherOrderOperator)
            and "_c10d_functional" not in self.op_overload.name()
            and self.op_overload._schema.is_mutable
            and can_auto_functionalize(self.op_overload)
        ):
            return []
        else:
            return self.alias_names

    def get_mutation_names(self) -> Sequence[str]:
        # [FUNCTION DOC] Return the mutated-and-returned names; AssertionError if there is more than one.
        if len(self.mutation_names) > 1:
            raise AssertionError("Expected len(self.mutation_names) <= 1")
        return self.mutation_names

    def export_extern_kernel_node(self):  # type: ignore[no-untyped-def]
        """
        ProxyExecutor Design Note
        We export the ExternFallbackNodes (for custom ops) into a serialized file
        and run it with a host side proxy executor to address the ABI problem
        This is currently only implemented for fbcode. Eventually, we will also make this work for OSS.
        Detailed design doc can be found at
        https://docs.google.com/document/d/1wC4DOZFaYym2t1Esz0X5yxlLI3RDnSiyRbUus3bkJ64/edit?usp=sharing
        """
        log.debug(
            "Extern kernel node added for node %s with target %s.",
            self.get_name(),
            self.op_overload,
        )

        if not isinstance(self, FallbackKernel):
        # [FUNCTION DOC] Build the argument list for runtime-dispatched calls: unflatten args, fill defaults and collect ordered kwarg values. In AOT mode also serialize the call and its outputs (tensor, list of tensors, optional tensor, int, none) into an export schema node, tagged when the op is a custom op, and append it to the extern kernel node list of the graph; outside AOT mode no serialization happens. Raises AssertionError for inconsistent outputs and RuntimeError for unsupported return types. Returns the list of args followed by kwargs values.
            raise AssertionError(type(self))
        args, kwargs = self.unflatten_args(self.inputs, self.constant_args)
        args = self.fill_non_provided_args(args, kwargs)
        ordered_kwargs = [
            self.get_kwargs_value(key, **kwargs)
            for key in self.ordered_kwargs_for_cpp_kernel
        ]
        target = self.op_overload

        if not V.graph.aot_mode:
            # No need to serialize in the cpp wrapper JIT mode
            return [*args, *ordered_kwargs]

        serializer = GraphModuleSerializer(None, [])  # type: ignore[arg-type]
        named_arguments = serializer.serialize_inputs(target, args, kwargs)

        # serialize_outputs
        def handle_single_output(
            return_type: torch.TensorType | torch.ListType | torch.JitType,
            output: IRNode | Sequence[IRNode],
        ) -> export_schema.Argument:
            # [FUNCTION DOC] HELPER: Convert one return value (tensor, none, tensor list, optional tensor, int) into an export schema argument according to its schema type.
            if isinstance(return_type, (torch.TensorType, torch.NoneType)):
                # For single Tensor or None
                out = output
                if isinstance(output, (list, tuple)):
                    if len(output) != 1:
                        raise AssertionError("Expected len(output) == 1")
                    out = output[0]
                if isinstance(return_type, torch.TensorType):
                    if not isinstance(out, IRNode):
                        raise AssertionError("Expected isinstance(out, IRNode)")
                    return export_schema.Argument.create(
                        as_tensor=export_schema.TensorArgument(name=out.get_name())
                    )
                else:  # NoneType
                    if out is not None:
                        raise AssertionError("Expected out is None")
                    return export_schema.Argument.create(as_none=True)
            elif isinstance(return_type, torch.ListType) and isinstance(
                return_type.getElementType(), torch.TensorType
            ):
                if not isinstance(output, Sequence):
                    raise AssertionError(type(output))
                # For single TensorList
                return export_schema.Argument.create(
                    as_tensors=[
                        export_schema.TensorArgument(name=out.get_name())
                        for out in output
                    ]
                )
            elif isinstance(return_type, torch.OptionalType) and isinstance(
                return_type.getElementType(), torch.TensorType
            ):
                # For OptionalTensor
                if output is None:
                    return export_schema.Argument.create(
                        as_optional_tensor=export_schema.OptionalTensorArgument.create(
                            as_none=True
                        )
                    )
                else:
                    if not isinstance(output, IRNode):
                        raise AssertionError("Expected isinstance(output, IRNode)")
                    return export_schema.Argument.create(
                        as_optional_tensor=export_schema.OptionalTensorArgument.create(
                            as_tensor=export_schema.TensorArgument(
                                name=output.get_name()
                            )
                        )
                    )
            elif isinstance(return_type, torch.IntType):
                return export_schema.Argument.create(as_int=output)
            else:
                raise RuntimeError(f"Unsupported return type {type(return_type)}")

        if isinstance(target, torch._higher_order_ops.torchbind.CallTorchBind):
            returns = target.schema(args[0], args[1]).returns
        else:
            returns = target._schema.returns  # type: ignore[union-attr]
        if len(returns) == 1:
            # NOTE: [special handling of all_reduce_coalesced_'s return value]
            # all_reduce_coalesced_ return a list of tensors via self.mutation_outputs
            outputs = self.codegen_outputs()
            return_type = returns[0].real_type
            output_arguments = [handle_single_output(return_type, outputs)]
        else:
            # For tuple returns, e.g "-> (Tensor, Tensor)" or "-> (Tensor, Tensor[])"
            # Not generating output args for self.mutation_outputs
            output_arguments = [
                handle_single_output(
                    return_schema.real_type,  # type: ignore[attr-defined]
                    output,
                )
                for return_schema, output in zip(returns, self.outputs)
            ]

        if self.op_overload is None:
            raise AssertionError("Expected self.op_overload is not None")
        metadata = {}
        if (
            isinstance(self.op_overload, torch._ops.OpOverload)
            and custom_ops._maybe_get_opdef(self.op_overload) is not None
        ):
            metadata["torch_library_custom_op"] = "1"

        node = ExternKernelNode(
            name=self.get_name(),
            node=export_schema.Node(
                target=self.op_overload.name(),
                inputs=named_arguments,
                outputs=output_arguments,
                metadata=metadata,
            ),
        )

        V.extern_kernel_nodes.append(node)

        return [*args, *ordered_kwargs]

    @override
    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Generate wrapper code for the fallback call. For aten ops under cpp wrapper, ops without a C shim use runtime dispatch (warning logged); custom ops use runtime dispatch unless registered with a C shim; complex scalar arguments also force runtime dispatch. Emits a comment, then either a runtime-lookup call (via the exported args) or a direct fallback call, then unbacked symbol definitions and, for tensor outputs, size, alignment and memory tracking asserts. AssertionError if the kernel is missing or not an OpOverload where required.
        """Overrides the parent member.
        See https://github.com/pytorch/pytorch/issues/151692"""
        kernel = self.op_overload
        if kernel is None:
            raise AssertionError("Expected kernel is not None")
        if kernel.namespace == "aten":
            # Aten Fallback Ops
            if not isinstance(kernel, torch._ops.OpOverload):
                raise AssertionError(type(kernel))
            if V.graph.cpp_wrapper:
                from torchgen.aoti.fallback_ops import inductor_fallback_ops

                if str(kernel) not in inductor_fallback_ops:
                    # C shim v2 is torchgen-ed, which should cover all aten ops.
                    # If you do hit a missed op, please update fallback_ops.py.
                    log.warning(
                        "%s is missing a c-shim implementation, using proxy executor as fallback",
                        kernel,
                    )
                    self.use_runtime_dispatch = True
        elif kernel.namespace == "_quantized":
            # Internal Quantized Fallback Ops
            if not isinstance(kernel, torch._ops.OpOverload):
                raise AssertionError(type(kernel))
        elif V.graph.cpp_wrapper:
            # For non-aten OpOverload, i.e. custom ops
            # If the op is in custom_ops_to_c_shims, generate direct function call
            self.use_runtime_dispatch = (
                kernel not in config.aot_inductor.custom_ops_to_c_shims
            )

        # Handle the special case where a complex number is input to a C-shim kernel for
        # a scalar input.  The torchgen'ed shim API will use type "double", which is
        # incompatible with complex numbers, forcing a fallback to runtime dispatch.
        if (
            V.graph.cpp_wrapper
            and isinstance(kernel, torch._ops.OpOverload)
            and not self.use_runtime_dispatch
        ):

            def is_number(t: torch.JitType) -> bool:
                # [FUNCTION DOC] HELPER: Return True if the schema type is a number type or an optional number type.
                if isinstance(t, torch.OptionalType):
                    return is_number(t.getElementType())
                return isinstance(t, torch.NumberType)

            # Using unflatten_args is a bit of a hack, but all the complex arguments we
            # care about are in self.constant_args, and calling unflatten_args puts them
            # in the correct order without triggering codegen.
            args, kwargs = self.unflatten_args(self.inputs, self.constant_args)
            # Append kwarg values to args.  ordered_kwargs_for_cpp_kernel is guaranteed
            # to be set, since this is an OpOverload kernel.
            args_iter = itertools.chain(
                args,
                (
                    self.get_kwargs_value(k, **kwargs)
                    for k in self.ordered_kwargs_for_cpp_kernel
                ),
            )
            self.use_runtime_dispatch = any(
                isinstance(v, complex) and is_number(a.real_type)
                for v, a in zip(args_iter, kernel._schema.arguments)
            )

        self.codegen_comment(wrapper)
        if self.use_runtime_dispatch:
            exported_args = self.export_extern_kernel_node()
            if self.python_kernel_name is None:
                raise AssertionError("Expected self.python_kernel_name is not None")
            if self.op_overload is None:
                raise AssertionError("Expected self.op_overload is not None")

            wrapper.generate_fallback_kernel_with_runtime_lookup(
                self.get_name(),
                self.python_kernel_name,
                lambda: [*self.codegen_args(), *self.codegen_kwargs()],
                self.op_overload,
                exported_args,
                # NOTE: [special handling of all_reduce_coalesced_'s return value]
                self.codegen_outputs(),
            )
        else:
            wrapper.generate_fallback_kernel(self)

        self.codegen_unbacked_symbol_defs(wrapper)
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)
            self.codegen_alignment_asserts(wrapper)
            self.codegen_memory_tracking(wrapper)

    @staticmethod
    def tensor_to_layout(output: torch.Tensor) -> FixedLayout:
        # [FUNCTION DOC] Static method building a FixedLayout from a real tensor's device, dtype, size, stride (converted to Inductor form) and pinned flag; pinned query errors are treated as not pinned.
        is_pinned = False
        try:
            is_pinned = output.is_pinned()
        except RuntimeError:
            # dispatch not implemented
            pass
        return FixedLayout(
            output.device,
            output.dtype,
            convert_shape_to_inductor(output.size()),
            convert_shape_to_inductor(output.stride()),
            is_pinned=is_pinned,
        )

    @classmethod
    def _maybe_realize_symm_mem_args(
        cls, kernel: torch._ops.OpOverload, *args: Any, **kwargs: Any
    ) -> None:
        # [FUNCTION DOC] Class method: for ops registered with symmetric-memory arguments, realize those tensor arguments as comm buffers (warning if impossible or arguments cannot be normalized); needs a group name argument. Does nothing for unregistered ops.
        """Realize symm_mem args as comm buffers based on registry metadata."""
        from torch._library.simple_registry import singleton

        qualname = kernel.__qualname__
        entry = singleton.get(qualname)
        if entry is None or not entry.symm_mem_args.is_registered():
            return

        from .comm_lowering import can_realize_as_comm_buffer, realize_as_comm_buffer

        symm_mem_args = entry.symm_mem_args.get()
        if symm_mem_args is None:
            return

        from torch.fx.operator_schemas import normalize_function

        normalized = normalize_function(
            kernel, args, kwargs, normalize_to_only_use_kwargs=True
        )
        if normalized is None:
            log.warning(
                "Failed to normalize arguments for symm_mem realization: %s",
                qualname,
            )
            return

        _, all_args = normalized

        group_name = all_args.get("group_name")
        if group_name is None:
            return

        for arg_name in symm_mem_args:
            arg_value = all_args.get(arg_name)
            tensors, _ = pytree.tree_flatten(arg_value)
            for t in tensors:
                if not isinstance(t, TensorBox):
                    continue
                if can_realize_as_comm_buffer(t, CommBufferType.SYMM_MEM):
                    realize_as_comm_buffer(t, CommBufferType.SYMM_MEM, group_name)
                else:
                    log.warning(
                        "Failed to realize %s as a symmetric memory buffer for %s",
                        arg_name,
                        qualname,
                    )

    @classmethod
    def create(cls, kernel: _OpOverloads, *args: Any, **kwargs: Any) -> FallbackKernel:
        # [FUNCTION DOC] Class method creating the IR for a fallback call, returning a TensorBox-compatible node, a structure of outputs, or an ExternKernelOut. Runs process_kernel under the fake mode (except for a known incorrect kernel), realizes symmetric-memory args for registered ops, and for custom (non-builtin) single-tensor-output functional ops tries to lower to a registered out variant with exactly one out argument (ExternKernelOut). Determines the device (cpu default for torchbind and print) and input alignment, optionally tries multi-output out lowering for custom ops outside cpp wrapper, then builds a FallbackKernel with a layout for None output (empty), single tensor (fixed layout, marking the buffer unaligned if needed) or multiple outputs (MultiOutputLayout, with MultiOutput nodes per tensor leaf, ints and symints passed through, opaque objects as OpaqueMultiOutput, other values must be None else AssertionError). Stores the flattened output list on the kernel and returns the output structure. AssertionError when the device cannot be found.
        """Create an instance of FallbackKernel from an _OpOverloads"""
        fake_incorrect_kernels = (aten._fused_moving_avg_obs_fq_helper_functional,)
        if kernel not in fake_incorrect_kernels:
            context = cast(AbstractContextManager[None], V.graph.fake_mode)
        else:
            context = nullcontext()

        with context:
            result = cls.process_kernel(kernel, *args, **kwargs)
        example_output = result.example_output
        tensor_args = result.tensor_args
        non_tensor_args = result.non_tensor_args
        unflatten_args = result.unflatten_args
        unbacked_bindings = result.unbacked_bindings

        # For ops with registered symm_mem args, realize those args as
        # symmetric memory buffers before creating the FallbackKernel.
        if isinstance(kernel, torch._ops.OpOverload):
            cls._maybe_realize_symm_mem_args(kernel, *args, **kwargs)

        # Try to lower single output functional custom ops to their out-variant.
        if (
            isinstance(kernel, torch._ops.OpOverload)
            and not torch._library.utils.is_builtin(kernel)
            and isinstance(example_output, torch.Tensor)
        ):
            from torch._library._out_variant import (
                _is_functional,
                get_out_arg_names,
                lookup_manual_out_variant,
                to_out_variant,
            )

            out_op = None
            if _is_functional(kernel._schema):
                out_op = to_out_variant(kernel)
            if out_op is None:
                out_op = lookup_manual_out_variant(kernel)

            if out_op is not None and len(get_out_arg_names(out_op)) == 1:
                layout = cls.tensor_to_layout(example_output)
                return ExternKernelOut(  # type: ignore[return-value]
                    layout=layout,
                    inputs=list(tensor_args),
                    constant_args=list(non_tensor_args),
                    kwargs=kwargs,
                    python_kernel_name=_make_out_variant_kernel_name(out_op),
                    op_overload=out_op,
                )

        # We need this extra check for input alignment since the example
        # inputs we created are always aligned.
        has_unaligned_input = any(is_unaligned(arg) for arg in tensor_args)

        device = cls.find_device(tensor_args, example_output)

        # Default to CPU for torchbind methods or HOPs that don't produce tensors
        if not device and (
            isinstance(kernel, torch._higher_order_ops.torchbind.CallTorchBind)
            or kernel is torch.ops.higher_order.print
        ):
            device = torch.device("cpu")

        def create_direct_output(output: torch.Tensor) -> FallbackKernel:
            # [FUNCTION DOC] HELPER: Build a FallbackKernel for a single tensor output with a fixed layout from the example tensor; mark its buffer unaligned if configured or if inputs or the example output are unaligned. AssertionError if the device is unknown.
            if not device:
                raise AssertionError("Not sure where to find device info")
            packed = cls(
                cls.tensor_to_layout(output),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                kwargs=kwargs,
                unbacked_bindings=unbacked_bindings,
            )
            if (
                config.assume_unaligned_fallback_output
                or has_unaligned_input
                or not tensor_is_aligned(output)
            ):
                V.graph.unaligned_buffers.add(packed.get_name())
            return packed

        # Try multi-output .out() lowering for custom ops with the out tag.
        if (
            isinstance(kernel, torch._ops.OpOverload)
            and not torch._library.utils.is_builtin(kernel)
            and not V.graph.cpp_wrapper
            and device
        ):
            out_result = ExternKernelMultiOut.try_create(
                kernel,
                example_output,
                device,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                kwargs,
                unbacked_bindings=unbacked_bindings,
                has_unaligned_input=has_unaligned_input,
            )
            if out_result is not None:
                return out_result  # type: ignore[return-value]

        if example_output is None:
            packed = cls(
                NoneLayout(device=device),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                kwargs=kwargs,
                unbacked_bindings=unbacked_bindings,
            )

        elif isinstance(example_output, torch.Tensor):
            return create_direct_output(example_output)

        else:
            if not device:
                raise AssertionError("Not sure where to find device info")
            packed = cls(
                MultiOutputLayout(device=device),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                kwargs=kwargs,
                unbacked_bindings=unbacked_bindings,
            )

        def generate_output(output: Any, indices: list[tuple[Any, int]]) -> Any:
            # [FUNCTION DOC] HELPER: Recursively convert the example output structure into IR: lists, tuples and dicts keep their structure with the access path recorded, tensors become MultiOutput nodes (unaligned tracking as above), ints pass through, symints become expressions, script or custom class objects become OpaqueMultiOutput, None stays None; other types raise AssertionError.
            if isinstance(output, (list, tuple)):
                return type(output)(
                    generate_output(output[i], indices + [(type(output), i)])
                    for i in range(len(output))
                )
            elif isinstance(output, dict):
                return {
                    key: generate_output(val, indices + [(type(output), key)])
                    for key, val in output.items()
                }
            elif isinstance(output, torch.Tensor):
                buf = MultiOutput(
                    cls.tensor_to_layout(output),
                    packed,
                    indices,
                )
                if (
                    config.assume_unaligned_fallback_output
                    or has_unaligned_input
                    or not tensor_is_aligned(output)
                ):
                    V.graph.unaligned_buffers.add(buf.name)  # type: ignore[arg-type]
                return buf
            elif isinstance(output, int):
                return output
            elif isinstance(output, torch.SymInt):
                return output.node.expr
            elif isinstance(
                output, (torch._C.ScriptObject, FakeScriptObject)
            ) or is_custom_class_obj(output):
                return OpaqueMultiOutput(
                    NoneLayout(device=device),
                    packed,
                    indices,
                    output,
                )
            else:
                if output is not None:
                    raise AssertionError(
                        f"FallbackKernel output type {type(output)} is not supported"
                    )
                return None

        outputs = generate_output(example_output, [])
        if isinstance(outputs, (list, tuple)):
            packed.outputs = outputs
        elif isinstance(outputs, dict):
            packed.outputs = tuple(outputs)
        else:
            packed.outputs = [outputs]

        return outputs


@ir_dataclass(frozen=False)
class ComplexView(FallbackKernel):
    # [CLASS DOC] FallbackKernel viewing a complex tensor as real pairs or the reverse; allocates nothing and reports its first input as aliased so the output is never reused.
    """View a complex number as two dtyped numbers or vice versa"""

    def should_allocate(self) -> bool:
        # [FUNCTION DOC] Return False (a view, no allocation).
        return False

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        # [FUNCTION DOC] Return the name of the first input, signalling the output must not be reused.
        # Signal to codegen that our output buffer isn't safe to reuse
        return [self.input_name(0)]

    def __init__(
        self,
        layout: OutputSpec,
        kernel: _OpOverloads,
        tensor_args: Sequence[IRNode],
        nontensor_args: Sequence[Any],
        unflatten_args: Callable[..., Any],
        *,
        kwargs: dict[str, Any] | None = None,
        unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] | None = None,
    ) -> None:
        # [FUNCTION DOC] Initialize as a FallbackKernel with the same arguments.
        super().__init__(
            layout,
            kernel,
            tensor_args,
            nontensor_args,
            unflatten_args,
            kwargs=kwargs,
            unbacked_bindings=unbacked_bindings,
        )


class MemoryCheckKernel(FallbackKernel):
    # [CLASS DOC] FallbackKernel for memory checking that emits a direct function call to the memory step checker instead of a normal fallback call; its constant args are the alive list, dead list and a final-step flag.
    """
    Custom kernel for memory checking that generates direct function calls

    TODO - the custom op was erroring with str inputs. should be able to custom op directly.
    """

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Emit the one-time allocation tracking setup and a call to the memory step check with the alive and dead lists (and the final-step flag with an explanatory comment when it is the final step).
        """Override codegen to write direct function call"""
        # Extract our arguments from nontensor_args
        wrapper.write_memory_track_allocation_once()
        alive_list, dead_list, is_final_step = self.constant_args

        alive_repr = repr(alive_list)
        dead_repr = repr(dead_list)
        if is_final_step:
            wrapper.writeline(
                "# note: don't currently distinguish between buffers returned and dealloc'd in last step"
            )
            call = f"check_memory_step(allocated={alive_repr}, freed={dead_repr}, is_final_step={is_final_step})"
        else:
            call = f"check_memory_step(allocated={alive_repr}, freed={dead_repr})"
        wrapper.writeline(call)


@ir_dataclass
class MultiOutputLayout(OutputSpec):
    # [CLASS DOC] OutputSpec for nodes producing several outputs (e.g. a packed tuple); holds only the device.
    device: torch.device

    def get_device(self) -> torch.device | None:
        # [FUNCTION DOC] Return the device.
        return self.device


class MultiOutput(ExternKernel):
    # [CLASS DOC] ExternKernel selecting one element of a multi-output parent node by an access path (indices of container type and position). Registers itself as buffer and operation. Reads the parent with a star dependency and writes via its own fixed layout. May skip size/stride/alignment asserts.
    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Generate the code extracting this output from the parent and, unless skipped, size and alignment assertions.
        wrapper.codegen_multi_output(self)
        if not self.skip_size_stride_alignment_checks:
            self.codegen_size_asserts(wrapper)
            self.codegen_alignment_asserts(wrapper)

    def __init__(
        self,
        layout: OutputSpec,
        input: IRNode,
        indices: list[tuple[Any, ...]],
        skip_size_stride_alignment_checks: bool = False,
    ) -> None:
        # [FUNCTION DOC] Initialize as an unnamed extern kernel reading the parent input; register buffer and operation; store the access path and the skip flag.
        super().__init__(None, layout, [input], ())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)
        self.indices = indices
        self.skip_size_stride_alignment_checks = skip_size_stride_alignment_checks

    @cache_on_self_and_args("MultiOutput")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return the free symbols of the parent input (cached); AssertionError if it is not an IR node.
        input_node = self.inputs[0]
        if not isinstance(input_node, IRNode):
            raise AssertionError(input_node)
        return input_node.get_free_symbol_uses(unbacked_only)

    def should_allocate(self) -> bool:
        # [FUNCTION DOC] Return True only if the parent is a C++ template buffer (grouped gemm), which needs this output allocated.
        return len(self.inputs) == 1 and (
            isinstance(self.inputs[0], CppTemplateBuffer)  # Grouped GEMM
        )

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        # [FUNCTION DOC] Return names of parent inputs that are fallback kernels with aliased inputs.
        return [
            inp.get_name()
            for inp in self.inputs
            if isinstance(inp, FallbackKernel)
            and len(inp.get_inputs_that_alias_output()) > 0
        ]

    def get_read_writes(self) -> dependencies.ReadWrites:
        # [FUNCTION DOC] Return ReadWrites with star-dependency reads of the parent inputs and a memory-dependency write built from this node's layout, normalized by the same policy as scheduler nodes (not normalized on GPU when loop ordering after fusion is on) so fusion checks can compare indices.
        # Reads: StarDep on parent (we don't know which elements of the
        # packed output we index into — conservative is correct).
        reads: OrderedSet[dependencies.Dep] = OrderedSet()
        for inp in self.inputs:
            if isinstance(inp, IRNode):
                reads.add(dependencies.StarDep(inp.get_name()))

        # Writes: build proper MemoryDep from our FixedLayout so the
        # scheduler can match our write with downstream epilogue reads.
        # Normalize using the same policy as SchedulerNode so that the
        # index expressions are directly comparable during fusion checks.
        name = self.get_name()
        indexer = self.get_layout().make_indexer()

        def dummy(index: Sequence[Any], rindex: Sequence[Any]) -> Any:
            # [FUNCTION DOC] HELPER: Dummy body storing a placeholder at the indexed position; requires no reduction index.
            if len(rindex) != 0:
                raise AssertionError("Expected len(rindex) == 0")
            return ops.store(name, indexer(index), "fake")

        device = self.get_device()
        should_normalize = (
            not config.loop_ordering_after_fusion
            or device is None
            or not is_gpu(device.type)
        )
        write_rw = dependencies.extract_read_writes(
            dummy, self.get_size(), (), normalize=should_normalize
        )
        return dependencies.ReadWrites(
            reads=reads,
            writes=write_rw.writes,
            index_exprs=OrderedSet(),
        )


class OpaqueMultiOutput(MultiOutput):
    # [CLASS DOC] MultiOutput for opaque objects (script objects, custom class instances): skips size/stride checks, has no dtype, and keeps the example value of the object.
    """MultiOutput for opaque objects."""

    def __init__(
        self,
        layout: OutputSpec,
        input: IRNode,
        indices: list[tuple[Any, ...]],
        opaque_value: Any,
    ) -> None:
        # [FUNCTION DOC] Initialize as a MultiOutput with size/stride checks skipped and store the opaque example value.
        super().__init__(layout, input, indices, skip_size_stride_alignment_checks=True)
        self.opaque_example_value = opaque_value

    @property  # type: ignore[override]
    def dtype(self) -> Never:
        # [FUNCTION DOC] Property that always raises AttributeError; opaque outputs have no dtype.
        raise AttributeError("OpaqueMultiOutput has no dtype")

    def wrap_for_lowering(self) -> OpaqueMultiOutput:
        # [FUNCTION DOC] Return self (no tensor box wrapping).
        return self

    def get_read_writes(self) -> dependencies.ReadWrites:
        # [FUNCTION DOC] Return ReadWrites with star-dependency reads of the parent inputs and a star-dependency write of this node.
        reads: OrderedSet[dependencies.Dep] = OrderedSet()
        for inp in self.inputs:
            if isinstance(inp, IRNode):
                reads.add(dependencies.StarDep(inp.get_name()))
        writes: OrderedSet[dependencies.Dep] = OrderedSet(
            [dependencies.StarDep(self.get_name())]
        )
        return dependencies.ReadWrites(
            reads=reads,
            writes=writes,
            index_exprs=OrderedSet(),
        )


class AllocatingMultiOutput(MultiOutput):
    # [CLASS DOC] MultiOutput whose buffer Inductor allocates, used for outputs of out-variant calls that write directly into pre-allocated buffers, so no tuple-indexing code is generated.
    """MultiOutput with Inductor-controlled allocation for .out() variant ops.

    Overrides should_allocate()=True so Inductor allocates the output buffer,
    and skips tuple-indexing codegen since .out() writes directly into these buffers.
    """

    def should_allocate(self) -> bool:
        # [FUNCTION DOC] Return True.
        return True

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Emit only size and alignment assertions (unless skipped); no extraction code.
        if not self.skip_size_stride_alignment_checks:
            self.codegen_size_asserts(wrapper)
            self.codegen_alignment_asserts(wrapper)


def _make_out_variant_kernel_name(out_op: torch._ops.OpOverload) -> str:
    # [FUNCTION DOC] HELPER: Build the fully qualified python name torch.ops namespace, op name and overload for an out variant op.
    """Build fully-qualified kernel name for an out-variant op."""
    ns = out_op.namespace
    op_name = out_op._schema.name.split("::")[1]
    overload = out_op._overloadname
    return f"torch.ops.{ns}.{op_name}.{overload}"


class ExternKernelMultiOut(FallbackKernel):
    # [CLASS DOC] FallbackKernel for multi-output out-variant lowering: emits the out call with pre-allocated output buffers and keeps AllocatingMultiOutput child nodes (out_variant_output_nodes) and the out argument names. Uses the out variant's python name and overload.
    """Multi-output .out() variant lowering.

    Subclass of FallbackKernel that emits .out() calls with pre-allocated
    output buffers. Uses AllocatingMultiOutput child nodes for each output.
    """

    out_arg_names: list[str]
    out_variant_output_nodes: list[AllocatingMultiOutput]

    def __init__(
        self,
        *args: Any,
        out_op: torch._ops.OpOverload,
        out_arg_names: list[str],
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Initialize as a FallbackKernel, then record out argument names, an empty output node list, the out variant kernel name and the out variant as the op overload.
        super().__init__(*args, **kwargs)
        self.out_arg_names = out_arg_names
        self.out_variant_output_nodes = []
        self.python_kernel_name = _make_out_variant_kernel_name(out_op)
        self.op_overload = out_op

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Emit the comment and generate the multi-out extern kernel call in the wrapper.
        self.codegen_comment(wrapper)
        wrapper.generate_extern_kernel_multi_out(self)

    @classmethod
    def try_create(
        cls,
        kernel: torch._ops.OpOverload,
        example_output: Any,
        device: torch.device,
        tensor_args: Sequence[IRNode],
        non_tensor_args: Sequence[Any],
        unflatten_args: Callable[..., Any],
        kwargs: dict[str, Any] | None,
        *,
        unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] | None = None,
        has_unaligned_input: bool = False,
    ) -> Sequence[AllocatingMultiOutput] | None:
        # [FUNCTION DOC] Class method creating an out-variant multi-output kernel if possible: requires a functional schema, a tuple or list of tensors as example output, an out variant with the same number of out arguments (otherwise returns None). Builds the packed kernel and one AllocatingMultiOutput per tensor with fixed layouts (marking unaligned buffers as configured), stores them on the kernel and returns them as a tuple or list matching the example output type.
        """Create an ExternKernelMultiOut if the op has a matching .out() variant."""
        from torch._library._out_variant import (
            _is_functional,
            get_out_arg_names,
            to_out_variant,
        )

        if not _is_functional(kernel._schema):
            return None

        if not isinstance(example_output, (tuple, list)):
            return None

        out_op = to_out_variant(kernel)
        if out_op is None:
            return None

        out_arg_names = get_out_arg_names(out_op)
        if not all(isinstance(t, torch.Tensor) for t in example_output):
            return None
        if len(example_output) != len(out_arg_names):
            return None

        packed = cls(
            MultiOutputLayout(device=device),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
            kwargs=kwargs,
            unbacked_bindings=unbacked_bindings,
            out_op=out_op,
            out_arg_names=out_arg_names,
        )

        outputs: list[AllocatingMultiOutput] = []
        for i, tensor_out in enumerate(example_output):
            layout = FixedLayout(
                device=tensor_out.device,
                dtype=tensor_out.dtype,
                size=[*tensor_out.shape],
                stride=[*tensor_out.stride()],
            )
            multi_out = AllocatingMultiOutput(
                layout=layout,
                input=packed,
                indices=[(type(example_output), i)],
            )
            if (
                config.assume_unaligned_fallback_output
                or has_unaligned_input
                or not tensor_is_aligned(tensor_out)
            ):
                V.graph.unaligned_buffers.add(multi_out.name)  # type: ignore[arg-type]
            outputs.append(multi_out)

        packed.out_variant_output_nodes = outputs
        packed.outputs = outputs

        if isinstance(example_output, tuple):
            return tuple(outputs)  # type: ignore[return-value]
        return list(outputs)


# We just use a normal dataclass for MutableBox/TensorBox/StorageBox since
# they're mainly lowering-time constructs that we expect to mutate and such.
@dataclasses.dataclass
class MutableBox(IRNode):
    # [CLASS DOC] Dataclass IR wrapper (data) allowing in-place mutation of the pointed-to node; base of TensorBox and StorageBox. Nearly all queries (device, loader, indexer, strides, name, layout, dtype, reads, mutation names, reduction info, realize, free symbols, codegen reference) delegate to data. Mainly a lowering-time construct.
    """
    TensorBox / StorageBox allow in-place mutation of Tensors
    """

    data: IRNode

    def has_exceeded_max_reads(self) -> bool:
        # [FUNCTION DOC] Delegate to data.
        return self.data.has_exceeded_max_reads()

    def get_device(self) -> torch.device | None:
        # [FUNCTION DOC] Delegate to data.
        return self.data.get_device()

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        # [FUNCTION DOC] Delegate to data.
        return self.data.make_loader()

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        # [FUNCTION DOC] Delegate to data.
        return self.data.make_indexer()

    def get_stride(self) -> Sequence[_IntLike]:
        # [FUNCTION DOC] Delegate to data.
        return self.data.get_stride()

    def get_name(self) -> str:
        # [FUNCTION DOC] Delegate to data.
        return self.data.get_name()

    def has_large_inner_fn(self, threshold: int | None = None) -> bool:
        # [FUNCTION DOC] Delegate to data.
        return self.data.has_large_inner_fn(threshold)

    def mark_reuse(self, users: int, *, graph_reuse: bool = True) -> None:
        # [FUNCTION DOC] Delegate the reuse hint to data.
        return self.data.mark_reuse(users, graph_reuse=graph_reuse)

    def realize_hint(self) -> None:
        # [FUNCTION DOC] Delegate to data.
        return self.data.realize_hint()

    def unwrap_view(self) -> IRNode:
        # [FUNCTION DOC] Delegate to data.
        return self.data.unwrap_view()

    def is_input_buffer(self) -> bool:
        # [FUNCTION DOC] Delegate to data.
        return self.data.is_input_buffer()

    def freeze_layout(self) -> None:
        # [FUNCTION DOC] Delegate to data.
        return self.data.freeze_layout()

    def freeze_layout_with_stride_order(
        self, order: Sequence[int], allow_padding: bool = False
    ) -> None:
        # [FUNCTION DOC] Delegate to data.
        return self.data.freeze_layout_with_stride_order(order, allow_padding)

    def freeze_layout_with_fill_order(self, order: Sequence[int]) -> None:
        # [FUNCTION DOC] Delegate to data.
        return self.data.freeze_layout_with_fill_order(order)

    def freeze_layout_with_same_order(self, stride: Sequence[_IntLike]) -> None:
        # [FUNCTION DOC] Delegate to data.
        return self.data.freeze_layout_with_same_order(stride)

    def freeze_layout_with_exact_strides(
        self, exact_strides: Sequence[_IntLike], allow_padding: bool = False
    ) -> None:
        # [FUNCTION DOC] Delegate to data.
        return self.data.freeze_layout_with_exact_strides(exact_strides, allow_padding)

    def get_read_writes(self) -> dependencies.ReadWrites:
        # [FUNCTION DOC] Delegate to data.
        return self.data.get_read_writes()

    def get_reads(self) -> OrderedSet[Dep]:
        # [FUNCTION DOC] Delegate to data.
        return self.data.get_reads()

    def num_reads(self) -> int:
        # [FUNCTION DOC] Delegate to data.
        return self.data.num_reads()

    def get_storage_numel(self) -> _IntLike:
        # [FUNCTION DOC] Delegate to data.
        return self.data.get_storage_numel()

    def get_reduction_type(self) -> str | None:
        # [FUNCTION DOC] Delegate to data.
        return self.data.get_reduction_type()

    def get_reduction_size(self) -> Sequence[Expr]:
        # [FUNCTION DOC] Delegate to data.
        return self.data.get_reduction_size()

    def is_extern(self) -> bool:
        # [FUNCTION DOC] Delegate to data.
        return self.data.is_extern()

    def is_no_op(self) -> bool:
        # [FUNCTION DOC] Delegate to data.
        return self.data.is_no_op()

    def constant_to_device(self, device: torch.device) -> IRNode:
        # [FUNCTION DOC] Delegate to data.
        return self.data.constant_to_device(device)

    def get_mutation_names(self) -> Sequence[str]:
        # [FUNCTION DOC] Delegate to data.
        return self.data.get_mutation_names()

    def get_operation_name(self) -> str:
        # [FUNCTION DOC] Delegate to data.
        return self.data.get_operation_name()

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        # [FUNCTION DOC] Delegate to data.
        return self.data.get_inputs_that_alias_output()

    def realize(self) -> str | None:
        # [FUNCTION DOC] Realize the wrapped data and return its result.
        return self.data.realize()

    @cache_on_self_and_args("MutableBox")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Delegate to data (cached).
        return self.data.get_free_symbol_uses(unbacked_only)

    def get_read_names(self) -> OrderedSet[str]:
        # [FUNCTION DOC] Delegate to data.
        return self.data.get_read_names()

    def get_defining_op(self) -> Operation | None:
        # [FUNCTION DOC] Delegate to data.
        return self.data.get_defining_op()

    def codegen_reference(self, writer: IndentedBuffer | None = None) -> str:
        # [FUNCTION DOC] Delegate to data, passing the writer.
        return self.data.codegen_reference(writer)

    @property
    def layout(self) -> OutputSpec:
        # [FUNCTION DOC] Property: the output spec of the wrapped data (deliberately the output spec, not the layout, because buffers hold an output spec).
        # we intentionally call get_output_spec (rather than get_layout) since Buffer.layout is an OutputSpec
        return self.data.get_output_spec()

    def get_layout(self) -> Layout:
        # [FUNCTION DOC] Delegate to data.
        return self.data.get_layout()

    def get_output_spec(self) -> OutputSpec:
        # [FUNCTION DOC] Delegate to data.
        return self.data.get_output_spec()

    def get_size(self) -> Sequence[Expr]:
        # [FUNCTION DOC] Delegate to data.
        return self.data.get_size()

    @property
    def dtype(self) -> torch.dtype:
        # [FUNCTION DOC] Property: dtype of the wrapped data.
        return self.data.dtype

    def __str__(self) -> str:
        # [FUNCTION DOC] Return a multi-line repr showing the class name and the indented inner node; for nested boxes both box names are shown.
        if isinstance(self.data, MutableBox):
            line0 = f"{type(self).__name__}({type(self.data).__name__}("
            endl = "))"
            inner = self.data.data
        else:
            line0 = f"{type(self).__name__}("
            inner = self.data
            endl = ")"

        lines = [
            line0,
            indent(str(inner)),
            endl,
        ]
        return "\n".join(lines)

    __repr__ = __str__


class TensorBox(MutableBox):
    # [CLASS DOC] Top-level IR wrapper that lowerings produce and that stands for a torch tensor. Its create static method wraps an IR node in TensorBox over StorageBox (shape constants are returned unwrapped).
    @overload
    @staticmethod
    def create(data: ShapeAsConstantBuffer) -> ShapeAsConstantBuffer: ...
    @overload
    @staticmethod
    def create(data: IRNode) -> TensorBox: ...

    @staticmethod
    def create(data: IRNode):
        # [FUNCTION DOC] Static method: return a ShapeAsConstantBuffer unchanged, otherwise wrap the node as a TensorBox over a StorageBox. (The preceding overload declarations are one-line typing stubs.)
        if isinstance(data, ShapeAsConstantBuffer):
            return data
        return TensorBox(StorageBox(data))


class StorageBox(MutableBox):
    # [CLASS DOC] MutableBox that owns storage semantics: decides when its data is realized into a ComputedBuffer (realize, realize_hint, reuse and read-count heuristics) and exposes graph-input and module-buffer checks.
    """
    StorageBox allow in-place mutation of Tensors
    """

    def is_input_buffer(self) -> bool:
        # [FUNCTION DOC] Return True if the data is an input buffer or reinterpret view whose name is a graph input.
        if isinstance(self.data, (InputBuffer, ReinterpretView)):
            return self.data.get_name() in V.graph.graph_inputs
        return False

    def is_module_buffer(self) -> bool:
        # [FUNCTION DOC] Return True if the data is a ConstantBuffer whose name is a graph constant.
        return (
            isinstance(self.data, (ConstantBuffer))
            and self.data.get_name() in V.graph.constants
        )

    def realize(self) -> str | None:
        # [FUNCTION DOC] Materialize the data and return the buffer name. Already-realized nodes just return their name. Otherwise (must be Pointwise, Reduction, Scan or Sort, else AssertionError) wrap it in a ComputedBuffer with a flexible layout on its device, register it as buffer and operation with the graph, and copy origins, origin node, traceback, stream index and mempool onto it. Mutates this box (replaces data) and graph state.
        if IRNode.is_realized_node(self.data):
            return self.data.get_name()

        if not isinstance(self.data, (Pointwise, Reduction, Scan, Sort)):
            raise AssertionError(type(self.data))
        origin_node = self.data.get_origin_node()
        traceback = self.data.get_traceback()
        device = self.data.get_device()
        if device is None:
            raise AssertionError("Expected device is not None")

        self.data = ComputedBuffer(
            name=None,
            layout=FlexibleLayout(
                device=device,
                dtype=self.data.get_dtype(),
                size=self.data.get_size(),
                is_pinned=False,
            ),
            data=self.data,
        )
        self.data.name = V.graph.register_buffer(self.data)
        V.graph.register_operation(self.data)
        self.data.origins = self.origins
        self.data.origin_node = origin_node
        self.data.traceback = traceback
        self.data.stream_idx = self.data.data.stream_idx
        self.data.mempool = self.data.data.mempool
        return self.data.name

    def realize_hint(self) -> None:
        # [FUNCTION DOC] Called on buffers expected to be forced to realize later: realizes now if the data is a Pointwise or Reduction that reads more than one nontrivial buffer.
        """
        Called on buffers we expect to be forced to realize later.
        """
        if (
            isinstance(self.data, (Pointwise, Reduction))
            and self.data.inner_fn_opcount().nontrivial_read_count > 1
        ):
            self.realize()

    def has_accumulated_enough_reads_by_size(self, threshold: int) -> bool:
        # [FUNCTION DOC] Return True if the sizes of the buffers read are non-empty, total at least threshold, the largest is at most half the total, and all read sizes are equal; non-freeable buffers are ignored.
        from torch._inductor.utils import is_nonfreeable_buffers

        size_of_reads = [
            V.graph.get_dep_size_hint(dep)
            for dep in self.get_reads()
            if not is_nonfreeable_buffers(dep)
        ]
        if not size_of_reads:
            return False
        total_size = sum(size_of_reads)
        max_size = max(size_of_reads)
        min_size = min(size_of_reads)
        return (
            total_size >= threshold
            and total_size / max_size >= 2
            and max_size == min_size
        )

    def has_exceeded_max_reads(self) -> bool:
        # [FUNCTION DOC] Return True for Pointwise data whose read count exceeds the configured accumulated reads threshold (cpu or default; AssertionError if a configured value is not an int), which has a large inner function, or whose reads accumulate enough size per config.
        realize_acc_reads_threshold = config.realize_acc_reads_threshold
        if realize_acc_reads_threshold is None:
            if is_cpu(self.data):
                realize_acc_reads_threshold = config.realize_cpu_acc_reads_threshold
            else:
                realize_acc_reads_threshold = (
                    config._realize_acc_reads_threshold_default
                )
        else:
            if not isinstance(realize_acc_reads_threshold, int):
                raise AssertionError(
                    f"expected int realize_acc_reads_threshold, got {type(realize_acc_reads_threshold)}"
                )
        return isinstance(self.data, Pointwise) and (
            self.num_reads() > realize_acc_reads_threshold
            or self.has_large_inner_fn()
            or (
                config.realize_acc_reads_size_threshold is not None
                and self.has_accumulated_enough_reads_by_size(
                    config.realize_acc_reads_size_threshold
                )
            )
        )

    def should_realize_on_reuse(self, users: int, *, graph_reuse: bool = True) -> bool:
        # [FUNCTION DOC] Heuristic deciding whether a value used by several users should be realized. Only applies to Pointwise or Reduction data with more than one user: realize when inline assembly ops are used; on cpu when heavy math ops are used or (for graph reuse with many users) the op count is high; when the inner function is large; or when graph reuse and the read count exceeds config.realize_reads_threshold.
        """
        A heuristic to decide if we should realize a tensor
        that is used multiple times.
        """
        if users > 1 and isinstance(self.data, (Pointwise, Reduction)):
            opcount = self.data.inner_fn_opcount()
            if "inline_asm_elementwise" in opcount.used_ops:
                return True
            if is_cpu(self.data):
                # Heuristic for realizing reused result of heavy ops on cpu
                heavy_ops = [
                    "exp",
                    "log",
                    "log10",
                    "log1p",
                    "log2",
                    "sigmoid",
                    "tanh",
                ]
                if any(x in opcount.used_ops for x in heavy_ops):
                    return True
                realize_threshold = self.data.get_realize_opcount_threshold()
                if (
                    isinstance(self.data, Pointwise)
                    and graph_reuse
                    and users > config.realize_opusers_threshold
                    and opcount.num_ops > max(0, realize_threshold - 2)
                ):
                    return True
            if self.has_large_inner_fn():
                return True
            return graph_reuse and self.num_reads() > config.realize_reads_threshold
        return False

    def mark_reuse(self, users: int, *, graph_reuse: bool = True) -> None:
        # [FUNCTION DOC] Realize the data if the reuse heuristic says so.
        if self.should_realize_on_reuse(users, graph_reuse=graph_reuse):
            self.realize()

    def num_reads(self) -> int:
        # [FUNCTION DOC] Return the number of reads of the data.
        return self.data.num_reads()


@ir_dataclass(frozen=False)
class Subgraph(IRNode):
    # [CLASS DOC] IR node describing a subgraph for higher-order ops: name, fx graph module, optional inductor config patches applied while lowering, and the lowered graph once created.
    name: str
    graph_module: torch.fx.GraphModule
    inductor_config_patches: dict[str, Any] | None = None
    graph: GraphLowering | None = None


def _has_aliased_buffers(buffers: Sequence[IRNode]) -> bool:
    # [FUNCTION DOC] HELPER: Return True if any two buffers (after unwrapping reinterpret views) are the same node object.
    buffers = [
        buffer.unwrap_view() if isinstance(buffer, ReinterpretView) else buffer
        for buffer in buffers
    ]
    # assuming the same buffer is represented by the same IRNode object
    return len(OrderedSet(id(buffer) for buffer in buffers)) < len(buffers)


@ir_dataclass(frozen=False)
class InvokeSubgraph(ExternKernel):
    # [CLASS DOC] ExternKernel for the invoke_subgraph higher-order op: holds the subgraph, the operands and the output nodes. create lowers the subgraph once and returns MultiOutput nodes (or constants) for its outputs.
    """
    Ir node for the invoke_subgraph HOP.
    """

    subgraph: Subgraph | None = None
    operands: Sequence[IRNode] | None = None
    outputs: Sequence[IRNode] | None = None

    def __init__(
        self, subgraph: Subgraph, operands: Sequence[IRNode], layout: MultiOutputLayout
    ) -> None:
        # [FUNCTION DOC] Initialize as an unnamed extern kernel over the operands with the given multi-output layout; store the subgraph and register buffer and operation.
        super().__init__(
            name=None,
            layout=layout,
            inputs=operands,
        )
        self.subgraph = subgraph
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def get_subgraphs(self) -> list[Subgraph]:
        # [FUNCTION DOC] Return a list with the subgraph, or an empty list.
        return [self.subgraph] if self.subgraph else []

    @classmethod
    def create(
        cls, subgraph: Subgraph, *operands: IRNode
    ) -> list[ShapeAsConstantBuffer | NoneAsConstantBuffer | MultiOutput]:
        # [FUNCTION DOC] Class method lowering an invoke_subgraph call. Determines example operand values from eager input values or node metadata (offsets differ when wrapped in with_effects; AssertionError if the expected target is missing), realizes operands and forces their strides to match the fake tensors (scalars/state objects pass through), takes inductor config patches from the node's nested region config if unset, lowers and runs the subgraph under those patches once, finds a device from the operands (AssertionError if none), creates the node and one MultiOutput per subgraph output (constants pass through) with a fixed layout copied from the subgraph output and size/stride checks skipped. Returns the list of outputs.
        """For each operand, get a realized input, force it to have the same
        strides as the subgraph inputs, then use an InvokeSubgraph"""
        from .lowering import constrain_to_fake_tensor

        # TODO(anijain2305) - Support sym expr as operands in future.
        current_node = V.graph.current_node

        fake_operands = None
        if eager_input_vals := current_node.meta.get("eager_input_vals"):
            # eager_input_vals is (args_values, kwargs_values). We need args for invoke_subgraph
            offset = 2
            if current_node.target is torch.ops.higher_order.with_effects:
                # Aruguments eagerly are (token, subgraph, identifier, *operands)
                if current_node.args[1] is not torch.ops.higher_order.invoke_subgraph:
                    raise AssertionError(
                        "Expected current_node.args[1] is torch.ops.higher_order.invoke_subgraph"
                    )
                offset = 3
            fake_operands = eager_input_vals[0][offset:]
        else:
            offset = 2
            if current_node.target is torch.ops.higher_order.with_effects:
                # with_effects args: (token, invoke_subgraph, subgraph, identifier, *operands)
                if current_node.args[1] is not torch.ops.higher_order.invoke_subgraph:
                    raise AssertionError(
                        "Expected current_node.args[1] is torch.ops.higher_order.invoke_subgraph"
                    )
                offset = 4

            # For the partitioned backward graph, we do not have
            # eager_input_vals. Here, we rely on the recorded example values.
            fx_operands = current_node.args[offset:]
            fake_operands = [x.meta["val"] for x in fx_operands]  # type: ignore[union-attr]

        # Realize the inputs. Also intermediates can have different strides than
        # the inputs of the subgraph. So, force the intermediates to have same
        # strides as that of subgraph inputs.
        # pyrefly: ignore [annotation-mismatch, redefinition]
        operands: list[IRNode] = [cls.realize_input(x) for x in operands]
        new_operands: list[IRNode] = []

        for idx, operand in enumerate(operands):
            if isinstance(
                operand, (ShapeAsConstantBuffer, GeneratorState, OpaqueObjectState)
            ):
                new_operands.append(operand)
            else:
                new_operands.append(
                    constrain_to_fake_tensor(operand, fake_operands[idx])
                )

        # pyrefly: ignore [bad-assignment]
        operands = new_operands

        if subgraph.inductor_config_patches is None:
            nested_config = current_node.meta.get("custom", {}).get(
                "nested_region_config"
            )
            subgraph.inductor_config_patches = getattr(
                nested_config, "inductor_config_patches", None
            )

        if subgraph.graph is None:
            # create and lower subgraphs
            ctx = (
                config.patch(subgraph.inductor_config_patches)
                if subgraph.inductor_config_patches
                else nullcontext()
            )
            with ctx:
                subgraph.graph = V.graph.make_subgraph(
                    gm=subgraph.graph_module,
                    example_inputs=fake_operands,
                    subgraph_name=subgraph.name,
                )
                with V.set_graph_handler(subgraph.graph):
                    subgraph.graph.run(*fake_operands)

        outputs = subgraph.graph.graph_outputs

        # Find the device - operands could be integers from shapes, so we can't
        # use operands[0]
        device = None
        for operand in operands:
            if not isinstance(operand, ShapeAsConstantBuffer):
                device = operand.get_device()
                break
        if device is None:
            raise AssertionError("Expected device is not None")
        invoke_subgraph = InvokeSubgraph(
            subgraph=subgraph,
            operands=operands,
            layout=MultiOutputLayout(device=device),
        )

        def create_output(
            output: IRNode, ind: int
        ) -> ShapeAsConstantBuffer | NoneAsConstantBuffer | MultiOutput:
            # [FUNCTION DOC] HELPER: Return a constant output unchanged, otherwise a MultiOutput over the invoke node with a fixed layout copied from the subgraph output (device required, AssertionError otherwise).
            if isinstance(output, (ShapeAsConstantBuffer, NoneAsConstantBuffer)):
                return output
            else:
                device = output.get_device()
                if device is None:
                    raise AssertionError("Expected device is not None")

                return MultiOutput(
                    FixedLayout(
                        device=device,
                        dtype=output.get_dtype(),
                        size=output.get_size(),
                        stride=output.get_stride(),
                        offset=output.get_layout().offset,
                        is_pinned=output.get_layout().is_pinned,
                    ),
                    invoke_subgraph,  # type: ignore[has-type]
                    [(list, ind)],
                    skip_size_stride_alignment_checks=True,
                )

        outs = [create_output(output, i) for i, output in enumerate(outputs)]
        invoke_subgraph.outputs = outs  # type: ignore[assignment]
        return outs

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Generate wrapper code for the invoke_subgraph call.
        wrapper.codegen_invoke_subgraph(self)


def _maybe_expr(s: int | torch.SymInt) -> int | Expr:
    # [FUNCTION DOC] HELPER: Return an int unchanged or the sympy expression of a SymInt.
    """Convert an int or SymInt to a plain int or sympy Expr for use in IR layouts."""
    if isinstance(s, int):
        return s
    return s.node.expr


@ir_dataclass(frozen=False)
class Switch(ExternKernel):
    # [CLASS DOC] ExternKernel for torch.cond and torch.switch. Fields: selector (scalar tensor: boolean for cond, integer for switch), branches (non-empty subgraphs; exactly two for cond, true then false), operands, is_cond flag and output MultiOutput nodes. Symbolic operands are passed as constant args and tensors as inputs. Supports mutation of operands via MutationOutput nodes and unbacked bindings.
    """
    IR node representing torch.cond and torch.switch.

    For torch.cond: ``selector`` is a boolean scalar tensor, branches contains exactly
    two subgraphs (true branch, false branch), and is_cond=True.
    For torch.switch: ``selector`` is an integer scalar tensor, branches contains one
    subgraph per case, and is_cond=False.

    Attributes:
        selector: A scalar tensor selecting which branch to execute.
        branches: Non-empty sequence of subgraphs representing the individual branches.
        operands: Input tensors passed to the individual subgraphs.
        is_cond: True when this node represents a torch.cond (boolean 2-branch select).
        outputs: MultiOutput nodes representing the node's outputs.
    """

    selector: IRNode | None = None
    branches: Sequence[Subgraph] | None = None
    operands: Sequence[IRNode] | None = None
    is_cond: bool = False
    outputs: Sequence[MultiOutput] | None = None

    def __init__(
        self,
        selector: IRNode,
        branches: Sequence[Subgraph],
        operands: Sequence[IRNode],
        layout: MultiOutputLayout,
        unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] | None,
        is_cond: bool = False,
    ) -> None:
        # [FUNCTION DOC] Store selector, branches, operands and the cond flag; split symbolic from tensor operands and initialize as an unnamed extern kernel; record unbacked bindings if given; register buffer and operation.
        self.selector = selector
        self.branches = branches
        self.operands = operands
        self.is_cond = is_cond

        sym_args, tensor_args = _split_by_sym_type([selector, *operands])

        super().__init__(
            name=None,
            layout=layout,
            inputs=tensor_args,
            constant_args=sym_args,
        )
        if unbacked_bindings is not None:
            self.unbacked_bindings = unbacked_bindings

        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def get_subgraphs(self) -> list[Subgraph]:
        # [FUNCTION DOC] Return the list of branch subgraphs (empty if none).
        return list(self.branches) if self.branches is not None else []

    @classmethod
    def create(
        cls,
        selector: TensorBox,
        branches: list[Subgraph],
        operands: list[TensorBox],
        is_cond: bool = False,
    ) -> list[MultiOutput]:
        # [FUNCTION DOC] Class method lowering a cond or switch. Realizes selector and operands, builds example operand values from fx node metadata (non-node entries pass through), lowers each branch subgraph once with outputs forced to the expected fake tensor strides (branch strides are used when merged strides contain unbacked symbols), checks that no branch has aliased outputs and that all branches agree in output count, device, dtype and layout offset (AssertionError otherwise), resolves the device from operands or the selector, creates the node and fixed-layout MultiOutput nodes sized from the merged fake output, and adds MutationOutput nodes for operands mutated by any branch. Returns the list of outputs.
        """Create a sequence of IRNodes from a switch/cond statement."""
        # pyrefly: ignore [bad-assignment]
        selector = cls.realize_input(selector)
        # pyrefly: ignore [bad-assignment]
        operands = [cls.realize_input(x) for x in operands]
        fx_operands: Argument = V.graph.current_node.args[-1]

        if not isinstance(fx_operands, Sequence):
            raise AssertionError(type(fx_operands))
        # Build fake_operands from FX nodes' metadata
        # For FX Nodes, get the fake tensor from meta["val"]
        # For non-Nodes (e.g., symbolic integers from sym_size lowering), pass directly
        fake_operands: list[Any] = []
        for fx_op in fx_operands:
            if isinstance(fx_op, Node):
                fake_operands.append(fx_op.meta["val"])
            else:
                # Symbolic integer or constant - pass directly
                fake_operands.append(fx_op)
        fake_outputs = V.graph.current_node.meta["val"]

        def _require_exact_strides(
            graph_outputs: Sequence[IRNode],
            fake_tensors: Sequence[torch.Tensor],
            branch_fakes: Sequence[torch.Tensor | int | None],
        ) -> list[IRNode]:
            # [FUNCTION DOC] HELPER: Force each subgraph output to the exact strides of the corresponding fake tensor (or of the branch's own fake if the merged strides contain unbacked symbols); constants pass through.
            ret = []
            for output, fake, branch_fake in zip(
                graph_outputs, fake_tensors, branch_fakes
            ):
                if isinstance(output, ShapeAsConstantBuffer):
                    ret.append(output)
                else:
                    strides = fake.stride()
                    # merged strides can contain unbacked symbols (from mismatched
                    # branch output shapes) undefined inside the subgraph
                    if has_free_unbacked_symbols(strides):
                        # pyrefly: ignore [missing-attribute]
                        strides = branch_fake.stride()
                    ret.append(
                        # pyrefly: ignore [bad-argument-type]
                        ExternKernel.require_exact_strides(
                            TensorBox(output), strides, allow_padding=False
                        )
                    )
            # pyrefly: ignore [bad-return]
            return ret

        for subgraph in branches:
            if subgraph.graph is None:
                # create and lower subgraphs
                subgraph.graph = V.graph.make_subgraph(
                    gm=subgraph.graph_module,
                    example_inputs=fake_operands,
                    subgraph_name=subgraph.name,
                )
                branch_out_args = subgraph.graph_module.graph.output_node().args[0]
                if not isinstance(branch_out_args, Sequence):
                    raise AssertionError(type(branch_out_args))
                branch_fakes: list[Any] = [
                    a.meta["val"] if isinstance(a, Node) else a for a in branch_out_args
                ]
                with V.set_graph_handler(subgraph.graph):
                    subgraph.graph.run(*fake_operands)
                    # Force subgraph outputs to the expected strides from
                    # FakeTensor metadata. Branches share the merged strides
                    # unless those carry an unbacked symbol (mismatched inner
                    # dims), in which case each branch keeps its own strides.
                    subgraph.graph.graph_outputs = _require_exact_strides(
                        subgraph.graph.graph_outputs, fake_outputs, branch_fakes
                    )

        if any(branch.graph is None for branch in branches):
            raise AssertionError("Expected all branch.graph to be not None")

        branch_outputs = [branch.graph.graph_outputs for branch in branches]  # type: ignore[union-attr]

        op_name = "torch.cond" if is_cond else "torch.switch"
        for branch, b_outputs in zip(branches, branch_outputs):
            if _has_aliased_buffers(b_outputs):
                raise AssertionError(
                    f"Output aliasing is currently not supported in compiled {op_name}. "
                    f"The outputs of the {branch.name} subgraph of {op_name} are aliased: {b_outputs}"
                )

        # All branches must produce structurally equivalent outputs (same count,
        # device, dtype, and layout offset).
        ref_outputs = branch_outputs[0]
        for b_outputs in branch_outputs[1:]:
            if len(ref_outputs) != len(b_outputs):
                raise AssertionError((ref_outputs, b_outputs))
            for i, (r_o, b_o) in enumerate(zip(ref_outputs, b_outputs)):
                if r_o.get_device() != b_o.get_device():
                    raise AssertionError((i, r_o, b_o))
                if r_o.get_dtype() != b_o.get_dtype():
                    raise AssertionError((i, r_o, b_o))
                if r_o.get_layout().offset != b_o.get_layout().offset:
                    raise AssertionError((i, r_o, b_o))

        # Determine device from operands and predicate
        # The predicate can be on a different device (e.g., CPU for control flow)
        # while the data operands and outputs should be on the compute device, so
        # using predicate device as a fallback.
        device = next(
            o.get_device()
            for o in operands + [selector]
            if not isinstance(o, ShapeAsConstantBuffer)
        )
        unbacked_bindings = resolve_unbacked_bindings(
            V.graph.sizevars.shape_env,
            V.graph.current_node.meta.get("unbacked_bindings", None),
        )
        if device is None:
            raise AssertionError("cannot determine device")
        switch = Switch(
            selector=selector,
            branches=branches,
            operands=operands,
            layout=MultiOutputLayout(device=device),
            unbacked_bindings=unbacked_bindings,
            is_cond=is_cond,
        )

        outputs = [
            MultiOutput(
                FixedLayout(
                    # pyrefly: ignore [bad-argument-type]
                    device=output.get_device()
                    if output.get_device() is not None
                    else device,  # type: ignore[arg-type]
                    dtype=output.get_dtype(),
                    size=[_maybe_expr(sz) for sz in merged_output.size()],
                    stride=[_maybe_expr(sz) for sz in merged_output.stride()],
                    offset=output.get_layout().offset,
                    is_pinned=output.get_layout().is_pinned,
                ),
                switch,
                [(list, i)],
            )
            # as the true and false outputs are equivalent,
            # we can use either of them here as a "template"
            for i, (output, merged_output) in enumerate(
                zip(ref_outputs, V.graph.current_node.meta["val"])
            )
        ]

        switch.outputs = outputs  # type: ignore[assignment]

        from torch._higher_order_ops.utils import (
            check_input_alias_and_mutation_return_outputs,
        )

        mutated_operand_indices: OrderedSet[int] = OrderedSet()
        for branch in branches:
            (_, _, _, branch_mutated, _) = (
                check_input_alias_and_mutation_return_outputs(branch.graph_module)
            )
            mutated_operand_indices |= OrderedSet(branch_mutated)

        # Create MutationOutput for each mutated operand (for scheduler dependencies)
        switch.mutation_outputs = [
            MutationOutput(operands[idx].layout, operands[idx], switch)  # type: ignore[union-attr]
            for idx in sorted(mutated_operand_indices)
        ]

        return outputs

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Generate wrapper code for the switch and the definitions of unbacked symbols produced by the outputs.
        wrapper.codegen_switch(self)
        wrapper.codegen_unbacked_symbol_defs_for_outputs(
            self.get_name(), self.outputs, getattr(self, "unbacked_bindings", {})
        )

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return the unbacked symbols defined, resolved from the stored bindings via the shape env (empty if none; AssertionError if resolution fails).
        if unbacked_bindings := getattr(self, "unbacked_bindings", None):
            resolved = resolve_unbacked_bindings(
                V.graph.sizevars.shape_env, unbacked_bindings
            )
            if resolved is None:
                raise AssertionError("Expected resolved is not None")
            return OrderedSet(resolved.keys())
        else:
            return OrderedSet()


def _split_by_sym_type(
    args: list[Any],
) -> tuple[list[ShapeAsConstantBuffer], list[Any]]:
    # [FUNCTION DOC] HELPER: Split a list into symbolic expressions (from ShapeAsConstantBuffer entries) and all other entries; returns (symbolic list, other list).
    non_sym_args = []
    sym_args = []
    for arg in args:
        if isinstance(arg, ShapeAsConstantBuffer):
            sym_args.append(arg.expr)
        else:
            non_sym_args.append(arg)

    return sym_args, non_sym_args


@ir_dataclass(frozen=False)
class WhileLoop(ExternKernel):
    # [CLASS DOC] ExternKernel for while_loop and while_loop_stack_output higher-order ops, supporting input mutation. Fields: carried inputs, additional inputs, condition and body subgraphs and output nodes. Symbolic inputs are constant args and tensors are inputs; stack_output selects the stacked-output variant.
    """The IR node for while_loop and while_loop_stack_output. It supports input mutation."""

    carried_inputs: Sequence[IRNode] | None = None
    additional_inputs: Sequence[IRNode] | None = None
    cond_subgraph: Subgraph | None = None
    body_subgraph: Subgraph | None = None
    outputs: Sequence[MultiOutput] | None = None

    def __init__(
        self,
        carried_inputs: Sequence[IRNode],
        additional_inputs: Sequence[IRNode],
        cond_subgraph: Subgraph,
        body_subgraph: Subgraph,
        layout: MultiOutputLayout,
        unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] | None,
        stack_output: bool,
    ) -> None:
        # [FUNCTION DOC] Store the loop parts, split symbolic from tensor inputs and initialize as an unnamed extern kernel; record unbacked bindings and the stack_output flag; register buffer and operation.
        self.carried_inputs = carried_inputs
        self.additional_inputs = additional_inputs
        self.cond_subgraph = cond_subgraph
        self.body_subgraph = body_subgraph

        sym_args, tensor_args = _split_by_sym_type(
            [*carried_inputs, *additional_inputs]
        )
        super().__init__(
            name=None,
            layout=layout,
            inputs=tensor_args,
            constant_args=sym_args,
        )
        if unbacked_bindings is not None:
            self.unbacked_bindings = unbacked_bindings
        self.stack_output = stack_output

        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def get_subgraphs(self) -> list[Subgraph]:
        # [FUNCTION DOC] Return the condition and body subgraphs that are set.
        subgraphs = []
        if self.cond_subgraph:
            subgraphs.append(self.cond_subgraph)
        if self.body_subgraph:
            subgraphs.append(self.body_subgraph)
        return subgraphs

    # Accidental aliasing can be created due to cse, where the empty buffers we
    # allocated for backward to use gets csed into the same buffer in function fx_graph_cse.
    # See test_scan_multiple_layers_gradient for a concrete example.
    @staticmethod
    def _clone_aliased_inputs(carried_inputs: Sequence[IRNode]) -> Sequence[IRNode]:
        # [FUNCTION DOC] HELPER: Static method returning the carried inputs with copies made for any input aliasing an earlier one (aliasing can arise from common-subexpression elimination of empty buffers); returns the input unchanged if no aliasing.
        if not _has_aliased_buffers(carried_inputs):
            return carried_inputs

        # Import clone from lowering module

        # Unwrap views to get the underlying buffers for comparison
        unwrapped_buffers = [
            buffer.unwrap_view() if isinstance(buffer, ReinterpretView) else buffer
            for buffer in carried_inputs
        ]

        # Track which buffers we've seen and their indices
        seen_buffers: OrderedSet[int] = OrderedSet()
        result: list[IRNode | TensorBox] = []

        for original_input, unwrapped_buffer in zip(carried_inputs, unwrapped_buffers):
            if id(unwrapped_buffer) in seen_buffers:
                result.append(ExternKernel.copy_input(original_input))
            else:
                seen_buffers.add(id(unwrapped_buffer))
                result.append(original_input)

        return result

    @staticmethod
    def _maybe_wrap_as_tensor_box(out: IRNode) -> IRNode:
        # [FUNCTION DOC] HELPER: Static method wrapping an output as a TensorBox (TensorBox unchanged, storage boxes and reinterpret views wrapped, MultiOutput created via create); raises RuntimeError for other types.
        if isinstance(out, TensorBox):
            return out
        elif isinstance(out, (StorageBox, ReinterpretView)):
            return TensorBox(out)
        elif isinstance(out, MultiOutput):
            return TensorBox.create(out)
        else:
            raise RuntimeError(f"NYI unsupported output type: {type(out)}")

    @classmethod
    def create(
        cls,
        cond_fn: Subgraph,
        body_fn: Subgraph,
        carried_inputs: Sequence[IRNode],
        additional_inputs: Sequence[IRNode],
        stack_output: bool,
    ) -> IRNode | Sequence[IRNode]:
        # [FUNCTION DOC] Class method lowering a while_loop (stack_output stacks each iteration's outputs, needed for training). Takes example values from fx node metadata, realizes carried and additional inputs (cloning aliased carried inputs) and forces their strides to the fake tensors. Lowers the condition and body subgraphs once; the body outputs are forced to the exact strides of the carried inputs. Validates: body outputs are not aliased, the condition returns a single boolean scalar, there is at least one operand, and carried inputs and body outputs agree in size, stride, device and dtype (AssertionError otherwise; sizes and strides compared via guards). Creates the WhileLoop node on the first input's device, finds input mutations in the body (unsupported with stack_output), and builds outputs: MultiOutput nodes (fixed layouts) for non-mutated positions and MutationOutput nodes for mutated carried inputs. Outputs corresponding to graph inputs are never reused. Returns the list of output nodes.
        """create the while_loop IR node. stack_output controls whether it stack
        each iterations' output, which is necessary for training.
        """
        from torch._higher_order_ops.utils import check_input_alias_and_mutation

        def _require_exact_strides(
            tensor_boxes: Sequence[IRNode],
            fake_tensors: list[int | torch.SymInt | torch.Tensor],
        ) -> list[IRNode]:
            # [FUNCTION DOC] HELPER: For each tensor box paired with a fake tensor, wrap as a TensorBox if needed and require the fake tensor's exact strides; non-tensor entries pass through. AssertionError on length mismatch.
            if len(tensor_boxes) != len(fake_tensors):
                raise AssertionError("Expected len(tensor_boxes) == len(fake_tensors)")
            ret = []
            for tb, fk in zip(tensor_boxes, fake_tensors):
                if isinstance(fk, torch.Tensor):
                    # Subgraph lowering always return StorageBox as graph_outputs because
                    # it realizes the outputs.
                    #
                    # However, require_exact_strides is expecting TensorBox
                    # e.g. in require_exact_strides when an expand happens,
                    # the fake tensor's stride is (0, 0, 0) but the storage
                    # box might have a different stride so lowering.slice_
                    # is used to make the stride consistent and it expects input to
                    # be TensorBox.
                    #
                    # So we wrap the inputs as tensor boxes if they're not yet.
                    new_tb = WhileLoop._maybe_wrap_as_tensor_box(tb)
                    ret.append(
                        ExternKernel.require_exact_strides(
                            new_tb, fk.stride(), allow_padding=False
                        )
                    )
                else:
                    ret.append(tb)
            return ret

        fx_carried_inputs = V.graph.current_node.args[-2]
        fx_additional_inputs = V.graph.current_node.args[-1]
        fx_all_inputs = fx_carried_inputs + fx_additional_inputs  # type: ignore[operator]
        fake_all_inputs = [x.meta["val"] for x in fx_all_inputs]  # type: ignore[union-attr]
        fake_carried_inputs = [x.meta["val"] for x in fx_carried_inputs]  # type: ignore[union-attr]
        fake_additional_inputs = [x.meta["val"] for x in fx_additional_inputs]  # type: ignore[union-attr]

        carried_inputs_ = [cls.realize_input(x) for x in carried_inputs]
        carried_inputs_ = WhileLoop._clone_aliased_inputs(carried_inputs_)
        carried_inputs_ = _require_exact_strides(carried_inputs_, fake_carried_inputs)
        additional_inputs_ = [cls.realize_input(x) for x in additional_inputs]
        additional_inputs_ = _require_exact_strides(
            additional_inputs_, fake_additional_inputs
        )
        all_inputs = carried_inputs_ + additional_inputs_

        for subgraph in (cond_fn, body_fn):
            if subgraph.graph is None:
                # create and lower subgraphs
                if not isinstance(fx_all_inputs, Sequence):
                    raise AssertionError(type(fx_all_inputs))
                subgraph.graph = V.graph.make_subgraph(
                    gm=subgraph.graph_module,
                    example_inputs=fx_all_inputs,  # type: ignore[arg-type]
                    subgraph_name=subgraph.name,
                )
                with V.set_graph_handler(subgraph.graph):
                    subgraph.graph.run(*fake_all_inputs)
                    # For body_fn, we require its output to have the exact same stride
                    # as inputs because the previous output is the input of next iteration.
                    #
                    # This cannot be automatically done in graph lowering because body_fn's graph outputs
                    # are not user-facing so the special handling for strides of user-facing output in graph
                    # lowering is not applicable.
                    if subgraph is body_fn:
                        if len(subgraph.graph.graph_outputs) != len(
                            fake_carried_inputs
                        ):
                            raise AssertionError(
                                "Expected len(subgraph.graph.graph_outputs) == len( fake_carried_inputs )"
                            )
                        subgraph.graph.graph_outputs = _require_exact_strides(  # type: ignore[assignment]
                            subgraph.graph.graph_outputs,
                            fake_carried_inputs,
                        )

        if not (cond_fn.graph and body_fn.graph):
            raise AssertionError("Expected cond_fn.graph and body_fn.graph")
        cond_outputs = cond_fn.graph.graph_outputs
        body_outputs = body_fn.graph.graph_outputs

        if _has_aliased_buffers(body_outputs):
            raise AssertionError(
                "Output aliasing is currently not supported in compiled torch.while_loop. "
                f"The outputs of the body_fn subgraph of torch.while_loop are aliased: {body_outputs}"
            )

        # make sure cond_fn returns a boolean scalar Tensor
        if len(cond_outputs) != 1:
            raise AssertionError(cond_outputs)
        p = cond_outputs[0]
        if not isinstance(p, ShapeAsConstantBuffer):
            if p.get_dtype() != torch.bool:
                raise AssertionError(p)
            if len(p.get_size()) != 0:
                raise AssertionError(p)

        if len(all_inputs) <= 0:
            raise AssertionError(
                "torch.while_loop is assumed to have at least one operand."
            )

        device = all_inputs[0].get_device()

        if device is None:  # to make linter happy
            raise AssertionError("Expected device is not None")
        # make sure carried_inputs_ and body outputs are structurally equivalent
        if len(carried_inputs_) != len(body_outputs):
            raise AssertionError(
                (
                    carried_inputs_,
                    body_outputs,
                )
            )
        for i, (op, bo) in enumerate(zip(carried_inputs_, body_outputs)):

            def _guard_list_equals(
                lhs_exprs: Sequence[int | Expr],
                rhs_exprs: Sequence[int | Expr],
            ) -> None:
                # [FUNCTION DOC] HELPER: Check lists of expressions have equal length (AssertionError) and elementwise equality via the size variable checker (may add guards).
                if len(lhs_exprs) != len(rhs_exprs):
                    raise AssertionError("Expected len(lhs_exprs) == len(rhs_exprs)")
                for lhs, rhs in zip(lhs_exprs, rhs_exprs):
                    V.graph.sizevars.check_equals(lhs, rhs)

            _guard_list_equals(op.get_size(), bo.get_size())
            _guard_list_equals(op.get_stride(), bo.get_stride())
            # assume all carried_inputs_ and outputs are on the same device
            # as the MultiOutputLayout below requires single device
            if op.get_device() != bo.get_device():
                raise AssertionError((i, op, bo, device))
            if op.get_dtype() != bo.get_dtype():
                raise AssertionError((i, op, bo))

        if device is None:
            raise AssertionError("Expected device is not None")

        unbacked_bindings = resolve_unbacked_bindings(
            V.graph.sizevars.shape_env,
            V.graph.current_node.meta.get("unbacked_bindings", None),
        )

        while_loop = WhileLoop(
            carried_inputs=carried_inputs_,
            additional_inputs=additional_inputs_,
            cond_subgraph=cond_fn,
            body_subgraph=body_fn,
            # asserted above that there is at least one operand
            layout=MultiOutputLayout(device=device),
            unbacked_bindings=unbacked_bindings,
            stack_output=stack_output,
        )

        if not (
            body_fn.graph is not None
            and isinstance(body_fn.graph.module, torch.fx.GraphModule)
        ):  # to make linter happy
            raise AssertionError(
                "Expected body_fn.graph is not None and isinstance( body_fn.graph.module, torch.fx.GraphModule )"
            )

        # Handling input mutations
        mutated_idxs = check_input_alias_and_mutation(
            body_fn.graph.module, fake_all_inputs
        )[3]
        mutated_idx_set = OrderedSet(mutated_idxs)
        mutated_inputs = [all_inputs[idx] for idx in mutated_idx_set]

        # Create all outputs first
        mutated_inputs_iter = iter(mutated_inputs)
        all_outputs: list[IRNode] = []
        while_loop.outputs = []
        while_loop.mutation_outputs = []
        if stack_output:
            if len(mutated_idx_set) != 0:
                raise AssertionError("NYI: while_loop_stack_output input mutations.")
            for idx, output in enumerate(V.graph.current_node.meta["val"]):
                # Create MultiOutput for regular outputs
                multi_out = MultiOutput(
                    FixedLayout(
                        device=output.device,  # type: ignore[arg-type]
                        dtype=output.dtype,
                        size=[_maybe_expr(sz) for sz in output.size()],
                        stride=[_maybe_expr(st) for st in output.stride()],
                    ),
                    while_loop,
                    [(list, idx)],
                )
                while_loop.outputs.append(multi_out)
                all_outputs.append(multi_out)
        else:
            for idx, output in enumerate(body_outputs):
                if idx in mutated_idx_set:
                    if idx >= len(carried_inputs):
                        raise AssertionError("only carries can be mutated.")
                    # Create MutationOutput for mutated inputs
                    mutated_input = next(mutated_inputs_iter)
                    while_loop.mutation_outputs.append(
                        MutationOutput(mutated_input.layout, mutated_input, while_loop)  # type: ignore[attr-defined, union-attr]
                    )
                    all_outputs.append(mutated_input)
                else:
                    multi_out = MultiOutput(
                        FixedLayout(
                            device=output.get_device(),  # type: ignore[arg-type]
                            dtype=output.get_dtype(),
                            size=output.get_size(),
                            stride=output.get_stride(),
                            offset=output.get_layout().offset,
                        ),
                        while_loop,
                        [(list, idx)],
                    )
                    while_loop.outputs.append(multi_out)
                    all_outputs.append(multi_out)

        for inp, out in zip(carried_inputs, all_outputs):
            if inp.get_name() in V.graph.graph_inputs:
                # if a carried input of the while_loop is a graph input,
                # it can be returned as is when the number of iterations
                # is zero. due to this, we can't (generally) reuse the
                # output buffers corresponding to the graph inputs, as
                # the inputs may end up being mutated.
                V.graph.never_reuse_buffers.add(out.get_name())
        return all_outputs

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Generate wrapper code for the while loop and the definitions of unbacked symbols produced by the outputs.
        wrapper.codegen_while_loop(self, self.stack_output)
        wrapper.codegen_unbacked_symbol_defs_for_outputs(
            self.get_name(), self.outputs, getattr(self, "unbacked_bindings", {})
        )

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return the unbacked symbols defined, resolved from the stored bindings via the shape env (empty if none; AssertionError if resolution fails).
        if unbacked_bindings := getattr(self, "unbacked_bindings", None):
            resolved = resolve_unbacked_bindings(
                V.graph.sizevars.shape_env, unbacked_bindings
            )
            if resolved is None:
                raise AssertionError("Expected resolved is not None")
            return OrderedSet(resolved.keys())
        else:
            return OrderedSet()


class EffectfulKernel(FallbackKernel):
    # [CLASS DOC] FallbackKernel for ops with an effect type: orders effectful ops of the same kind by chaining a star-dependency read on the previous effectful buffer, registers itself as the latest of its effect type in the graph, and reports side effects.
    def __init__(
        self,
        layout: OutputSpec,
        kernel: _OpOverloads,
        tensor_args: Sequence[IRNode],
        nontensor_args: Sequence[Any],
        unflatten_args: Callable[..., Any],
        kwargs: dict[str, Any] | None = None,
        *,
        unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] | None = None,
    ) -> None:
        # [FUNCTION DOC] Initialize as a FallbackKernel (kwargs are not forwarded; AssertionError if the kernel has no effect type), then record the effect type, the previous effectful buffer for that type and register this node as the newest. KNOWN ISSUE: the kwargs parameter is accepted but ignored.
        super().__init__(
            layout,
            kernel,
            tensor_args,
            nontensor_args,
            unflatten_args,
            kwargs=None,
            unbacked_bindings=unbacked_bindings,
        )

        from torch._higher_order_ops.effects import _get_effect

        effect_type = _get_effect(kernel)
        if effect_type is None:
            raise AssertionError("Expected effect_type is not None")
        self.effect_type = effect_type
        self.prev_effect_buffer = V.graph.effectful_ops.get(effect_type, None)
        V.graph.effectful_ops[effect_type] = self

    def get_read_writes(self) -> dependencies.ReadWrites:
        # [FUNCTION DOC] Return the fallback read/writes plus a star-dependency read on the previous effectful buffer if there is one.
        read_writes = super().get_read_writes()

        if self.prev_effect_buffer is not None:
            read_writes.reads.add(
                dependencies.StarDep(self.prev_effect_buffer.get_name())
            )

        return read_writes

    def has_side_effects(self) -> bool:
        # [FUNCTION DOC] Return True.
        return True


class NonTensorObj(IRNode):
    # [CLASS DOC] Base class for IR nodes that are not tensors (torchbind objects, generator state, opaque objects): no free symbols and realize does nothing.
    @cache_on_self_and_args("NonTensorObj")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return an empty set (cached).
        return OrderedSet()

    def realize(self) -> str | None:
        # [FUNCTION DOC] Return None; nothing to realize.
        return None


@ir_dataclass
class TorchBindObject(NonTensorObj):
    # [CLASS DOC] Non-tensor IR node for a torchbind (script) object with a graph name and a real or fake script object value; can report the real object and its total tensor byte size.
    name: str
    value: FakeScriptObject | torch.ScriptObject

    def get_name(self) -> str:
        # [FUNCTION DOC] Return the graph name.
        return self.name

    def codegen_reference(self, writer: IndentedBuffer | None = None) -> str:
        # [FUNCTION DOC] Return the name as the code reference.
        return self.name

    def get_value(self) -> FakeScriptObject | torch.ScriptObject:
        # [FUNCTION DOC] Return the stored script object value.
        return self.value

    def get_real_obj(self) -> torch.ScriptObject:
        # [FUNCTION DOC] Return the real script object (unwrapping a fake one).
        if isinstance(self.value, torch.ScriptObject):
            return self.value
        else:
            return self.value.real_obj

    def get_buf_bytes(self) -> int:
        # [FUNCTION DOC] Return the total bytes of tensors in the flattened object (zero for custom class objects); AssertionError if the real object cannot be flattened.
        # Returns the sum of all tensors in the flattened object
        real_script_obj = self.get_real_obj()

        if is_custom_class_obj(real_script_obj):
            return 0

        if not hasattr(real_script_obj, "__obj_flatten__"):
            raise AssertionError('Expected hasattr(real_script_obj, "__obj_flatten__")')
        flat_dict = dict(real_script_obj.__obj_flatten__())
        flat_elems = pytree.tree_flatten(flat_dict)[0]
        flat_sizes = [
            x.element_size() * x.numel()
            for x in flat_elems
            if isinstance(x, torch.Tensor)
        ]
        return functools.reduce(operator.add, flat_sizes, 0)


@ir_dataclass
class OpaqueValueTypeConstant(NonTensorObj):
    # [CLASS DOC] Non-tensor IR node for an opaque value-type constant appearing in outputs; its code reference is the inlined repr of the value (registering the needed opaque types on the graph), unlike torchbind objects that reference named constants.
    """IR node for opaque value type constants that appear directly in graph outputs.

    Unlike TorchBindObject (which references named constants loaded at runtime),
    this inlines the value's repr into the generated code since value types are
    reconstructed from their repr.
    """

    value: Any

    def get_name(self) -> str:
        # [FUNCTION DOC] Return the repr of the value as the name.
        return repr(self.value)

    def codegen_reference(self, writer: IndentedBuffer | None = None) -> str:
        # [FUNCTION DOC] Return the repr text for the value and record any opaque value type classes it needs in the graph.
        obj_repr, opaque_types = get_opaque_obj_repr(self.value)
        for n, t in opaque_types.items():
            V.graph.opaque_value_type_classes[n] = t
        return obj_repr


@ir_dataclass
class GeneratorState(NonTensorObj):
    # [CLASS DOC] Non-tensor IR node for random generator state, with a graph name and a device.
    name: str
    device: torch.device

    def get_name(self) -> str:
        # [FUNCTION DOC] Return the graph name.
        return self.name

    def codegen_reference(self, writer: IndentedBuffer | None = None) -> str:
        # [FUNCTION DOC] Return the name as the code reference.
        return self.name


@ir_dataclass
class OpaqueObjectState(NonTensorObj):
    # [CLASS DOC] Non-tensor IR node wrapping an opaque object (e.g. a process group) passed as a graph input, with its placeholder name; the value is kept for reference only.
    """
    Represents an opaque object (e.g., ProcessGroup) that is passed through
    as a graph input. Similar to GeneratorState, this wraps the object with
    its placeholder name so codegen can reference it properly.
    """

    name: str
    value: Any  # The actual opaque object (for reference, not used in codegen)

    def get_name(self) -> str:
        # [FUNCTION DOC] Return the placeholder name.
        return self.name

    def codegen_reference(self, writer: IndentedBuffer | None = None) -> str:
        # [FUNCTION DOC] Return the name as the code reference.
        return self.name


class _CollectiveKernel(FallbackKernel):
    # [CLASS DOC] FallbackKernel base for collective communication ops: does not allocate, has side effects, takes the cpp kernel name from the schema. Provides create_inplace (collective mutating its inputs) and create_out_of_place (collective with fresh outputs); safety rules are in the notes below.
    def should_allocate(self) -> bool:
        # [FUNCTION DOC] Return False.
        return False

    def has_side_effects(self) -> bool:
        # [FUNCTION DOC] Return True.
        return True

    # This is identical to FallbackKernel.set_cpp_kernel(), minus the
    # part that checks against input aliasing and mutation.
    def set_cpp_kernel_name(self, cpp_kernel_name: str | None = None) -> None:
        # [FUNCTION DOC] Set the cpp kernel name (default: the schema name) and the ordered keyword-only argument names; like the fallback version but without alias and mutation checks. Raises AssertionError unless the overload is a plain OpOverload.
        if type(self.op_overload) is not torch._ops.OpOverload:
            raise AssertionError("Setting cpp kernel needs a valid op_overload")
        kernel = self.op_overload
        if cpp_kernel_name is not None:
            self.cpp_kernel_name = cpp_kernel_name
        else:
            self.cpp_kernel_name = kernel._schema.name

        self.ordered_kwargs_for_cpp_kernel = [
            x.name for x in kernel._schema.arguments if x.kwarg_only
        ]

    # NOTE: [In-Place Collective Safety]
    # Between the initiation and completion of an in-place collective, the
    # input buffers are subject to both volatile reads and volatile writes.
    # They must not be read, written to or reused by another kernel. To ensure
    # the constraints, we model collective -> wait_tensor as a two-step
    # mutation of the input buffers.
    @classmethod
    def create_inplace(
        cls,
        kernel: _OpOverloads,
        inputs: IRNode | list[IRNode],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Class method creating an in-place collective: runs process_kernel under fake mode (unbacked bindings not allowed, AssertionError), realizes every tensor input and marks it mutated, finds a device (AssertionError if none), builds the kernel with an empty layout and records MutationOutput nodes and alias names for all inputs and for an out argument if present. Modeled as a two-step mutation until the wait.
        with V.graph.fake_mode:
            result = cls.process_kernel(kernel, inputs, *args, **kwargs)
        tensor_args = result.tensor_args
        non_tensor_args = result.non_tensor_args
        unflatten_args = result.unflatten_args
        if result.unbacked_bindings:
            raise AssertionError(f"{kernel} {result.unbacked_bindings}")
        device = None
        for tensor_arg in tensor_args:
            if isinstance(tensor_arg, NonTensorObj):
                continue
            tensor_arg.realize()
            V.graph.mark_buffer_mutated(tensor_arg.get_name())
            if device is None:
                device = tensor_arg.get_device()
        if device is None:
            raise AssertionError(
                f"In-place collective {kernel} requires at least one tensor "
                f"argument; got only non-tensor IR nodes."
            )
        packed = cls(
            NoneLayout(device=device),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
        )

        inps = pytree.tree_leaves(inputs)
        packed.mutation_outputs.extend(
            [MutationOutput(NoneLayout(device=device), buf, packed) for buf in inps]
        )

        # For inplace collective ops, the input is guaranteed to be alias of the returned value of op.
        packed.alias_names.extend([inp.get_name() for inp in inps])
        if "out" in kwargs:
            packed.mutation_outputs.append(
                MutationOutput(NoneLayout(device=device), kwargs["out"], packed)
            )
            # For out-variant collective ops, the `out=` arg is guaranteed to be alias of the returned value of op.
            packed.alias_names.append(kwargs["out"].get_name())

    # NOTE: [Out-of-Place Collective Safety]
    # Between the initiation and completion of an out-of-place collective:
    #
    # Input buffers:
    # - Are subject to volatile reads
    # - Can be read by another kernel
    # - Must not be written to or reused by another kernel
    #
    # Output buffers:
    # - Are subject to volatile writes
    # - Must not be read, written to or reused by another kernel
    #
    # To ensure the safety of input buffers without sacrificing read
    # availability, we add input buffers as read deps of wait_tensor kernels.
    #
    # To ensure the safety of output buffers, we model wait_tensor as a
    # mutation to the output buffer. Note we also assumes the user program being
    # correct and the output buffer is not consumed by kernels other than
    # wait_tensor.
    #
    # TODO(yifu): add a pre-grad pass to validate the correctness of collective
    # usage in the user program.
    @classmethod
    def create_out_of_place(
        cls,
        kernel: _OpOverloads,
        inputs: TensorBox | list[TensorBox],
        *args: Any,
        **kwargs: Any,
    ) -> list[MultiOutput] | _CollectiveKernel:
        # [FUNCTION DOC] Class method creating an out-of-place collective: runs process_kernel under fake mode (unbacked bindings not allowed), realizes tensor inputs, and for list outputs creates a multi-output kernel with MultiOutput children (marking unaligned ones), otherwise a single kernel with the tensor layout. Returns the list of outputs or the kernel itself. Outputs are made safe by wait_tensor modeling a mutation of the output.
        with V.graph.fake_mode:
            result = cls.process_kernel(kernel, inputs, *args, **kwargs)
        example_output = result.example_output
        tensor_args = result.tensor_args
        non_tensor_args = result.non_tensor_args
        unflatten_args = result.unflatten_args
        if result.unbacked_bindings:
            raise AssertionError(f"{kernel}, {result.unbacked_bindings}")
        for tensor_arg in tensor_args:
            if not isinstance(tensor_arg, TorchBindObject):
                tensor_arg.realize()

        if isinstance(example_output, list):
            device = cls.find_device(tensor_args, example_output)
            if device is None:
                raise AssertionError("Expected device is not None")
            packed = cls(
                MultiOutputLayout(device=device),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
            )
            packed.outputs = [
                MultiOutput(
                    cls.tensor_to_layout(tensor),
                    packed,
                    [(list, i)],
                )
                for i, tensor in enumerate(example_output)
            ]
            for buf, tensor in zip(packed.outputs, example_output):
                if config.assume_unaligned_fallback_output or not tensor_is_aligned(
                    tensor
                ):
                    V.graph.unaligned_buffers.add(buf.name)  # type: ignore[arg-type]
            return packed.outputs
        else:
            packed = cls(
                cls.tensor_to_layout(example_output),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
            )
            if config.assume_unaligned_fallback_output or not tensor_is_aligned(
                example_output
            ):
                V.graph.unaligned_buffers.add(packed.name)  # type: ignore[arg-type]
            packed.outputs = [packed]
            return packed


class _AllReduce_Kernel(_CollectiveKernel):
    # [CLASS DOC] Collective kernel for in-place all reduce on cpu using the AOTI cpu c10d shim; emits an extern alloc call and size asserts.
    def __init__(
        self,
        layout: OutputSpec,
        kernel: _OpOverloads,
        tensor_args: Sequence[IRNode],
        nontensor_args: Sequence[Any],
        unflatten_args: Callable[..., Any],
        kwargs: dict[str, Any] | None = None,
        *,
        unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] | None = None,
    ) -> None:
        # [FUNCTION DOC] Initialize as a collective fallback kernel (kwargs ignored) and set the cpp kernel name to the cpu all reduce in-place shim.
        super().__init__(
            layout,
            kernel,
            tensor_args,
            nontensor_args,
            unflatten_args,
            kwargs=None,
            unbacked_bindings=unbacked_bindings,
        )
        self.set_cpp_kernel_name("aoti_torch_cpu__c10d_functional_all_reduce_")

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Include the cpu shim header, generate the extern kernel call and, for tensor layouts, size asserts.
        wrapper.include_extra_header("torch/csrc/inductor/aoti_torch/c/shim_cpu.h")
        wrapper.generate_extern_kernel_alloc(self)

        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)


class _AllReduceKernel(_CollectiveKernel):
    # [CLASS DOC] Collective kernel for out-of-place all reduce on cpu using the AOTI cpu c10d shim.
    def __init__(
        self,
        layout: OutputSpec,
        kernel: _OpOverloads,
        tensor_args: Sequence[IRNode],
        nontensor_args: Sequence[Any],
        unflatten_args: Callable[..., Any],
        kwargs: dict[str, Any] | None = None,
        *,
        unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] | None = None,
    ) -> None:
        # [FUNCTION DOC] Initialize as a collective fallback kernel (kwargs ignored) and set the cpp kernel name to the cpu all reduce shim.
        super().__init__(
            layout,
            kernel,
            tensor_args,
            nontensor_args,
            unflatten_args,
            kwargs=None,
            unbacked_bindings=unbacked_bindings,
        )
        self.set_cpp_kernel_name("aoti_torch_cpu__c10d_functional_all_reduce")

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Include the cpu shim header, generate the extern kernel call and, for tensor layouts, size asserts.
        wrapper.include_extra_header("torch/csrc/inductor/aoti_torch/c/shim_cpu.h")
        wrapper.generate_extern_kernel_alloc(self)

        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)


class _WaitKernel(_CollectiveKernel):
    # [CLASS DOC] Collective kernel for wait_tensor on cpu: completes a collective by modeling a mutation of its input and adding the collective's inputs as volatile reads.
    def __init__(
        self,
        layout: OutputSpec,
        kernel: _OpOverloads,
        tensor_args: Sequence[IRNode],
        nontensor_args: Sequence[Any],
        unflatten_args: Callable[..., Any],
        kwargs: dict[str, Any] | None = None,
        *,
        unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] | None = None,
    ) -> None:
        # [FUNCTION DOC] Initialize as a collective fallback kernel (kwargs ignored) and set the cpp kernel name to the cpu wait tensor shim.
        super().__init__(
            layout,
            kernel,
            tensor_args,
            nontensor_args,
            unflatten_args,
            kwargs=None,
            unbacked_bindings=unbacked_bindings,
        )
        self.set_cpp_kernel_name("aoti_torch_cpu__c10d_functional_wait_tensor")

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Include the cpu shim header, generate the extern kernel call and, for tensor layouts, size asserts.
        wrapper.include_extra_header("torch/csrc/inductor/aoti_torch/c/shim_cpu.h")
        wrapper.generate_extern_kernel_alloc(self)

        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    def get_volatile_reads(self) -> Sequence[IRNode]:
        # [FUNCTION DOC] Return the input nodes of the collective whose completion this wait ends and that must stay unmodified until then: the collective's first input for an out-of-place single-output collective, the corresponding input for an out-of-place multi-output collective, nothing for in-place collectives. AssertionError if the input is not an IR node.
        inp = self.inputs[0]
        if not isinstance(inp, IRNode):
            raise AssertionError("Expected isinstance(inp, IRNode)")
        if isinstance(inp, _CollectiveKernel):
            # Out-of-place single-output
            i = inp.inputs[0]
            if not isinstance(i, IRNode):
                raise AssertionError(type(i))
            return [i]
        elif isinstance(inp, MultiOutput):
            # This can be two things:
            # 1. Out-of-place multi-output coll
            # 2. In-place coll with inputs coming from another MultiOutput
            coll = inp.inputs[0]
            # Case 1
            if isinstance(coll, _CollectiveKernel):
                _, idx = inp.indices[0]

                return [coll.inputs[idx]]
            # Case 2
            return []
        else:
            # In-place requires no additional deps handling for volatile
            # reads since the inputs are mutated.
            return []

    @classmethod
    def create_wait(cls, kernel: _OpOverloads, inp: TensorBox) -> None:
        # [FUNCTION DOC] Class method creating a wait_tensor node on the input: runs process_kernel under fake mode (no unbacked bindings, AssertionError), creates the kernel with an empty layout, and adds a MutationOutput for the input.
        with V.graph.fake_mode:
            result = cls.process_kernel(kernel, inp)
        if result.unbacked_bindings:
            raise AssertionError(f"{kernel} {result.unbacked_bindings}")
        packed = cls(
            NoneLayout(device=inp.get_device()),
            kernel,
            result.tensor_args,
            result.non_tensor_args,
            result.unflatten_args,
        )
        packed.mutation_outputs.append(
            MutationOutput(NoneLayout(device=inp.get_device()), inp, packed)
        )

    def get_read_writes(self) -> dependencies.ReadWrites:
        # [FUNCTION DOC] Return the fallback read/writes plus star-dependency reads of the volatile read inputs (see the out-of-place collective safety note).
        read_writes = super().get_read_writes()
        # See [Out-of-Place Collective Safety].
        volatile_reads = self.get_volatile_reads()
        for vr in volatile_reads:
            read_writes.reads.add(dependencies.StarDep(vr.get_name()))
        return read_writes


# NB: recursive structure here reflects val_to_arg_str, avoid
# calling free_unbacked_symbols on "exotic" types that don't get pexpr
# treatment
def maybe_free_unbacked_symbols(s: object) -> OrderedSet[Symbol]:
    # [FUNCTION DOC] Return the free unbacked symbols in a value that may be a symbolic type, tuple or list (recursively) or tensor; other values give an empty set. Mirrors the structure handled by argument formatting.
    if isinstance(s, (SymTypes, Expr)):
        # This branch should be impossible in return position
        return free_unbacked_symbols(s)
    elif isinstance(s, (tuple, list)):
        r = OrderedSet[sympy.Symbol]()
        for t in s:
            r |= maybe_free_unbacked_symbols(t)
        return r
    elif isinstance(s, torch.Tensor):
        # This branch is impossible in constant-args position
        return free_unbacked_symbols(s)
    else:
        return OrderedSet()


def maybe_free_symbols(s: object) -> OrderedSet[Symbol]:
    # [FUNCTION DOC] Same as maybe_free_unbacked_symbols but returns all free symbols, not just unbacked ones.
    if isinstance(s, (SymTypes, Expr)):
        # This branch should be impossible in return position
        return free_symbols(s)
    elif isinstance(s, (tuple, list)):
        r = OrderedSet[sympy.Symbol]()
        for t in s:
            r |= maybe_free_symbols(t)
        return r
    elif isinstance(s, torch.Tensor):
        # This branch is impossible in constant-args position
        return free_symbols(s)
    else:
        return OrderedSet()


def assign_origin_node(result: Any, n: torch.fx.Node) -> None:
    # [FUNCTION DOC] Best-effort record of the originating fx node n onto the IR in result: only for a TensorBox over a StorageBox (non-view), setting origin_node on the loop node, buffer, computed buffer data and, for a plain multi-output, its parent buffer. Views are not handled. Mutates the nodes.
    # This is not complete, but it doesn't have to be: origin_node
    # tracking is best effort.  The logic here critically relies on direct
    # TensorBox -> StorageBox denoting a non-view; we don't bother trying
    # to get views to work.  Feel free to add any extra cases as needed.
    #
    # Note: we can't YOLO tree_map over this result, because if there are
    # buffers or a view involved, we might not be able to validly assign
    # the origin_node here.
    if isinstance(result, TensorBox) and isinstance(result.data, StorageBox):
        if isinstance(result.data.data, Loops):
            result.data.data._post_init_setattr("origin_node", n)
        elif isinstance(result.data.data, Buffer):
            result.data.data._post_init_setattr("origin_node", n)
            if isinstance(result.data.data, ComputedBuffer) and isinstance(
                result.data.data.data, Loops
            ):
                result.data.data.data._post_init_setattr("origin_node", n)
            # Not really multi-output, can straightforwardly recurse in
            elif (
                isinstance(result.data.data, MultiOutput)
                and not result.data.data.indices
            ):
                if isinstance(result.data.data.inputs[0], Buffer):
                    result.data.data.inputs[0]._post_init_setattr("origin_node", n)
