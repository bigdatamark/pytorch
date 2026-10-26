# [MODULE DOC] Python wrapper code generator for Inductor. Defines PythonWrapperCodegen (builds the generated host-side "call" function and module source: buffer allocation/free/reuse, kernel definition and launch, extern kernels, subgraphs, streams, cudagraph partitions), plus the WrapperLine family of IR-like line objects that are later lowered to text or to FX, memory planning helpers, and Triton user-kernel source/grid helpers.
# No heavy import-time side effects beyond defining a module-level printer (pexpr) and type aliases; per-compile state lives on the wrapper instances.
# mypy: allow-untyped-defs
from __future__ import annotations

import collections
import contextlib
import dataclasses
import dis
import functools
import inspect
import logging
import operator
import os
import random
import re
import tempfile
from collections.abc import Callable
from enum import Enum
from itertools import chain, count
from typing import Any, Literal, Protocol, TYPE_CHECKING

import sympy
from sympy import Expr

import torch
import torch._ops
import torch.utils._pytree as pytree
from torch import dtype as torch_dtype
from torch._dynamo.utils import counters, dynamo_timed, get_debug_dir
from torch._inductor.codegen.debug_utils import DebugPrinterManager
from torch._inductor.codegen.multi_kernel import MultiKernelState
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._library.opaque_object import get_opaque_obj_repr, is_opaque_constant_type
from torch._logging import trace_structured
from torch.fx.experimental.proxy_tensor import (
    _coor_check_current_accelerator,
    _coor_current_accelerator,
    _coor_enabled,
)
from torch.fx.experimental.symbolic_shapes import (
    CallMethodKey,
    ConvertIntKey,
    DivideByKey,
    resolve_unbacked_bindings,
    SymTypes,
)
from torch.fx.node import _get_qualified_name
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import CleanDiv, Max, Min
from torch.utils._sympy.singleton_int import SingletonInt
from torch.utils._sympy.symbol import symbol_is_type, SymT

from .. import async_compile, config, debug as inductor_debug, ir
from ..codecache import output_code_log
from ..ir import IRNode, ReinterpretView
from ..runtime import triton_heuristics
from ..runtime.hints import DeviceProperties, TritonMeta
from ..stream_constants import DEFAULT_STREAM, DEFAULT_STREAM_IDX, STREAM_NAME_TEMPLATE
from ..stream_utils import (
    COOR_DEVICE_IDX_VAR,
    coor_device_str,
    get_raw_stream_name,
    get_stream_name,
)
from ..utils import (
    cache_on_self,
    DeferredLineBase,
    DelayReplaceLine,
    get_benchmark_name,
    get_dtype_size,
    IndentedBuffer,
    is_codegen_graph_partition_subgraph,
    is_using_cudagraph_partition,
    LineContext,
    make_codegen_buffer,
    sympy_product,
    sympy_str,
    sympy_subs,
    triton_version_uses_attrs_dict,
)
from ..virtualized import V
from .common import (
    ArgName,
    CodeGen,
    DeferredLine,
    PythonPrinter,
    WorkspaceArg,
    WorkspaceZeroMode,
)
from .cpp_utils import cexpr
from .custom_extern_kernel_codegen import CUSTOM_EXTERN_KERNEL_CODEGEN
from .triton_utils import config_of, should_unwrap_unspec_arg, signature_to_meta


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    import triton

    from ..graph import GraphLowering
    from ..ir import ExternKernel
    from ..scheduler import BaseSchedulerNode
    from .wrapper_fxir import FxConverter


log = logging.getLogger(__name__)

pexpr = PythonPrinter().doprint


def _rewrite_symbol_solution_for_int_codegen(expr: sympy.Expr) -> sympy.Expr:
    # [FUNCTION DOC] HELPER: Rewrites a sympy solution expression so it is safe for integer codegen. Combines it over a common denominator; if the denominator is 1 returns the numerator, otherwise returns an exact-division node (CleanDiv) of numerator by denominator. Pure function.
    expr = sympy.together(sympy.sympify(expr))
    numerator, denominator = sympy.fraction(expr)
    if denominator == 1:
        return numerator
    return CleanDiv(numerator, denominator)


def _sanitize_for_repr(obj: Any) -> Any:
    # [FUNCTION DOC] HELPER: Recursively converts a value into one whose repr is valid Python for generated code. Enum members become their underlying value, dicts/lists/tuples are rebuilt with sanitized contents, and namedtuples keep their type. Other objects are returned unchanged. No side effects.
    """Convert Enum values to their underlying value for valid Python repr in code generation."""
    if isinstance(obj, dict):
        return {_sanitize_for_repr(k): _sanitize_for_repr(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_repr(v) for v in obj]
    # For namedtuples (have _fields), reconstruct to preserve the type
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return getattr(type(obj), "_make")(  # noqa: B009
            _sanitize_for_repr(getattr(obj, field)) for field in obj._fields
        )
    if isinstance(obj, tuple):
        return tuple(_sanitize_for_repr(v) for v in obj)
    if isinstance(obj, Enum):
        return _sanitize_for_repr(obj.value)
    return obj


ReuseKey = tuple[torch.device, torch.dtype, str, bool, int, tuple[int, int] | None]
CommBufferReuseKey = tuple[torch.device, torch.dtype, str, "ir.CommBufferType", str]
BufferLike = ir.Buffer | WorkspaceArg
FxConversionFunc = Callable[["WrapperLine"], None]


@dataclasses.dataclass
class BenchmarkStorageGroup:
    # [CLASS DOC] Plain data record describing one group of benchmark storage: buffer name, size in bytes, device, dtype, and a mapping from input name to a pair of integer lists (presumably sizes and strides). No methods.
    buffer_name: str
    nbytes: int
    device: Any
    dtype: Any
    inputs: dict[str, tuple[list[int], list[int]]]


def buffer_reuse_key(node: BufferLike) -> ReuseKey:
    # [FUNCTION DOC] Computes the key used to decide whether a freed buffer may be reused for another. Returns a tuple of device, dtype, symbolic simplified storage size as a string (symbolic on purpose so s0 never reuses s1), alignment flag (buffer not in the unaligned set), scheduler stream and mempool of the buffer. Reads global graph/scheduler state; no mutation.
    storage_size = V.graph.get_allocation_storage_size(node)
    alignment = node.get_name() not in V.graph.unaligned_buffers
    stream = V.graph.scheduler.get_buf_stream(node.get_name())
    mempool = V.graph.scheduler.get_buf_mempool(node.get_name())
    return (
        node.get_device_or_error(),
        node.get_dtype(),
        # NB: this is symbolic so that we don't try to reuse a buffer
        # for s0 for s1, just because they happen to share the same
        # size hint
        sympy_str(V.graph.sizevars.simplify(storage_size)),
        alignment,
        stream,
        mempool,
    )


def comm_buffer_reuse_key(node: BufferLike) -> CommBufferReuseKey:
    # [FUNCTION DOC] Computes the reuse key for communication buffers: device, dtype, simplified symbolic size string, comm buffer type and group name. Raises an assertion error if the node output layout is not a CommBufferLayout. Comm buffers only reuse other comm buffers with an identical key.
    # Comm buffers can only be reused by other comm buffers with the same (device, dtype, size, comm_buffer_type, group_name).
    storage_size = V.graph.get_allocation_storage_size(node)
    layout = node.get_output_spec()
    if not isinstance(layout, ir.CommBufferLayout):
        raise AssertionError(f"expected CommBufferLayout, got {type(layout)}")
    return (
        node.get_device_or_error(),
        node.get_dtype(),
        sympy_str(V.graph.sizevars.simplify(storage_size)),
        layout.comm_buffer_type,
        layout.group_name,
    )


def can_match_buffer_size(input_buf: BufferLike, output_buf: BufferLike):
    # [FUNCTION DOC] Decides whether input_buf can be re-inplaced into output_buf. Returns False when devices or dtypes differ; otherwise True if the simplified allocation sizes are symbolically identical, or if the output size is statically known to lie between 95 percent of the input size and the input size. Reads graph size variables; no mutation.
    # Return True if input_buf can be re-inplaced for output_buf.
    # This differs from `buffer_reuse_key` for general buffer reuse.
    if input_buf.get_device_or_error() != output_buf.get_device_or_error():
        return False

    if input_buf.get_dtype() != output_buf.get_dtype():
        return False

    input_size = V.graph.sizevars.simplify(
        V.graph.get_allocation_storage_size(input_buf)
    )
    output_size = V.graph.sizevars.simplify(
        V.graph.get_allocation_storage_size(output_buf)
    )

    if (
        # NB: this is symbolic so that we don't try to reuse a buffer
        # for s0 for s1, just because they happen to share the same
        # size hint
        sympy_str(input_size) == sympy_str(output_size)
    ) or (
        # statically known that 0.95 * input_size <= output_size <= input_size
        V.graph.sizevars.statically_known_geq(output_size, 0.95 * input_size)
        and V.graph.sizevars.statically_known_leq(output_size, input_size)
    ):
        return True

    return False


def codegen_reinterpret_view_helper(data):
    # [FUNCTION DOC] Walks a chain of TensorBox/StorageBox/ReinterpretView wrappers down to the base Buffer and reports whether the chain can be collapsed. Returns a 5-tuple (size, stride, offset, dtype, collapsible). A bare Buffer is collapsible with its own layout. Returns all None and False if a wrapper has no layout, the chain does not end in a Buffer, or any wrapper offset differs from the base buffer offset; otherwise returns the base layout fields and True.
    """
    Collapse a chain of ReinterpretView <- StorageBox
    <- ReinterpretView <- StorageBox.... <- buffer wrappers if every layer
    has the same offset as the innermost (base) buffer.

    Returns:
        (size, stride, offset, dtype, collapsible: bool)
    """
    if isinstance(data, ir.Buffer):
        lay = data.get_layout()
        return lay.size, lay.stride, lay.offset, lay.dtype, True

    layouts: list[Any] = []
    cur = data
    while isinstance(cur, (ir.TensorBox, ir.StorageBox, ir.ReinterpretView)):
        lay = cur.get_layout()
        if lay is None:
            return None, None, None, None, False
        layouts.append(lay)
        cur = cur.data  # unwrap

    if not isinstance(cur, ir.Buffer):
        return None, None, None, None, False

    # All wrapper offsets must match base offset to be collapsible
    for lay in layouts:
        if lay.offset != cur.get_layout().offset:
            return None, None, None, None, False

    base_lay = cur.get_layout()
    return base_lay.size, base_lay.stride, base_lay.offset, base_lay.dtype, True


# TODO: Move to a well known place
TritonMetaParams = dict[str, int]
TritonGrid = (
    tuple[int | sympy.Expr, ...] | Callable[[TritonMetaParams], tuple[int, ...]]
)


def user_defined_kernel_grid_fn_code(
    name: str,
    configs: list[triton.Config],  # type: ignore[name-defined]
    grids: list[TritonGrid],
    wrapper: PythonWrapperCodegen | None = None,
    original_fxnode_name: str | None = None,
) -> tuple[str, str]:
    # [FUNCTION DOC] Generates source text for a grid wrapper function used by a user-defined Triton kernel. Args: name (kernel name), configs (autotune configs), grids (one grid per config, or a single grid), optional wrapper (to print sympy expressions and record autotune-at-compile-time lines) and original_fxnode_name (to look up example grids). Returns (function name, source). With one grid the function simply returns it; with several it emits conditional returns keyed on config kwargs (AMD specific kwargs ignored, configs ordered by number of kwargs descending, duplicate statements skipped). Side effect: when autotune_at_compile_time is on, also appends example-valued lines to the wrapper kernel_autotune_calls buffer. Raises an assertion error if grid and config counts are inconsistent.
    output = IndentedBuffer()

    def _convert_to_sympy_expr(item: int | sympy.Expr) -> sympy.Expr:
        # [FUNCTION DOC] HELPER: Returns the item as a sympy expression, wrapping plain ints in a sympy Integer.
        return item if isinstance(item, sympy.Expr) else sympy.Integer(item)

    def determine_grid(
        grid: TritonGrid,
        example_grid: TritonGrid | None = None,
    ):
        # [FUNCTION DOC] HELPER: Converts one grid into a pair (real grid code, example grid code). Callable grids, or the absence of a wrapper, are returned as-is for both. Otherwise the grid is converted to sympy and printed as a python shape tuple via the wrapper; the example version uses concrete example argument values and is None unless autotune_at_compile_time is enabled.
        """
        This function returns a tuple of two values: the first one is for the real grid
        which is used in the generated code; the second one is an example grid with
        concrete values which is used in the autotune block to run the generated
        kernels at compile time.
        """
        if wrapper is None or callable(grid):
            # return as-is when used in eager mode or when grid is callable
            return grid, grid
        # Grid contains ints/Expr, so utilize wrapper's expr printer for codegen
        sympy_grid = tuple(_convert_to_sympy_expr(g) for g in grid)
        if not example_grid:
            example_grid = sympy_grid
        return (
            wrapper.codegen_python_shape_tuple(sympy_grid),
            (
                wrapper.codegen_python_shape_tuple(
                    tuple(
                        wrapper.generate_example_arg_value(g, type(g))
                        for g in example_grid  # type: ignore[union-attr]
                    )
                )
                if config.triton.autotune_at_compile_time
                else None
            ),
        )

    def writeline(line: str, example_grid: str | None = None):
        # [FUNCTION DOC] HELPER: Writes a line to the output buffer and, when autotune at compile time is on and this kernel is not yet in the autotuned set, also writes the example-grid variant (or the same line) to the wrapper autotune calls buffer.
        output.writeline(line)
        if (
            wrapper
            and config.triton.autotune_at_compile_time
            and name not in wrapper.kernel_autotune_names
        ):
            wrapper.kernel_autotune_calls.writeline(example_grid or line)

    fn_name = f"grid_wrapper_for_{name}"
    writeline(f"def {fn_name}(meta):")
    kernel_autotune_calls_indent = (
        wrapper.kernel_autotune_calls.indent()
        if wrapper and config.triton.autotune_at_compile_time
        else contextlib.nullcontext()
    )
    with output.indent(), kernel_autotune_calls_indent:
        if (
            config.triton.autotune_at_compile_time
            and original_fxnode_name
            and V.graph.autotuning_grids
            and original_fxnode_name in V.graph.autotuning_grids
        ):
            example_grids = V.graph.autotuning_grids[original_fxnode_name]
        else:
            example_grids = [None] * len(grids)
        if len(grids) == 1:
            grid, example_grid = determine_grid(grids[0], example_grids[0])
            writeline(f"return {grid}", f"return {example_grid}")
        else:
            if len(grids) <= 1:
                raise AssertionError(f"expected more than 1 grid, got {len(grids)}")
            if len(grids) != len(configs):
                raise AssertionError(
                    f"expected len(grids) == len(configs), got {len(grids)} != {len(configs)}"
                )
            seen: OrderedSet[str] = OrderedSet()
            # sort the configs from the largest # of kwargs to the smallest to
            # emit the grids in the order of (approximately) decreasing specificity
            # TODO(aakhundov): the sorting below is generally not sufficient, so
            # maybe we'll need to restrict the supported cases to identical kwarg
            # names in all autotuning configs.
            for grid, c, example_grid in sorted(
                zip(grids, configs, example_grids),
                key=lambda x: len(x[1].kwargs),
                reverse=True,
            ):
                guardslist = []
                if c.kwargs:
                    # Remove AMD specific kwargs.
                    for kwarg in c.kwargs:
                        if kwarg not in [
                            "matrix_instr_nonkdim",
                            "waves_per_eu",
                            "kpack",
                        ]:
                            guardslist.append(f"meta['{kwarg}'] == {c.kwargs[kwarg]}")
                if guardslist:
                    guards = " and ".join(guardslist)
                else:
                    guards = "True"  # for configs with empty kwargs
                grid, example_grid = determine_grid(grid, example_grid)
                statement = f"if {guards}: return {grid}"
                if statement in seen:
                    continue
                seen.add(statement)
                writeline(statement, f"if {guards}: return {example_grid}")

    return fn_name, output.getvalue()


def user_defined_triton_kernel_transitive_closure_source_code(
    kernel, epilogue_fusion: tuple[ir.ComputedBuffer, str] | None = None
) -> str:
    # [FUNCTION DOC] Collects the source of a Triton kernel together with the transitive closure of everything it references, as one string. Starts with the kernel source (or the epilogue-fused source if given as the second element of epilogue_fusion), then recursively appends referenced JITFunctions (decorated as triton.jit), constexpr functions, simple global constants (int/str/bool/constexpr, with type annotations preserved) and imports for names pulled from triton modules. Uses bytecode inspection of global loads and the function globals; tracks already included symbols to avoid duplicates and loops. Returns the assembled source. Requires triton to be importable.
    """
    Given a triton kernel function pointer collect the transitive closure of
    its dependencies

    epilogue_fusion: Optional[(fused epilogue node, modified kerel src code)]
    """
    compile_wrapper = IndentedBuffer()
    kernel_src = kernel.src
    if epilogue_fusion:
        kernel_src = epilogue_fusion[1]
    compile_wrapper.splice(kernel_src, strip=True)

    # Also include any possible kernel being called indirectly
    import triton
    from triton import JITFunction  # type: ignore[name-defined, attr-defined]
    from triton.language import constexpr  # type: ignore[name-defined]
    from triton.language.core import dtype as triton_dtype

    # global constexpr vars handled above
    symbols_included = OrderedSet([kernel.__name__])

    def traverse(cur_kernel):
        # [FUNCTION DOC] HELPER: Recursive worker for the closure collection: inspects a kernel function for global names, and for each not-yet-included name found in its globals emits the right definition into compile_wrapper (jit function, constexpr function with dtype import and alias, constant assignment, or an import from triton) and recurses into functions. Mutates the enclosing buffer and symbol set.
        # here we extract the unqualified names (i.e., not attributes and
        # without prepended module name) loaded in the kernel code, which
        # are matched with the co_names and __globals__ below to codegen
        # the respective imports necessary for the kernel compilation
        unqualified_loads = OrderedSet(
            inst.argval
            for inst in dis.Bytecode(cur_kernel.fn)
            if inst.opname == "LOAD_GLOBAL"
        )
        global_annotations = cur_kernel.fn.__globals__.get("__annotations__", {})
        for symbol_name in cur_kernel.fn.__code__.co_names:
            if symbol_name in symbols_included:
                continue
            if symbol_name in cur_kernel.fn.__globals__:
                symbol = cur_kernel.fn.__globals__[symbol_name]
                if isinstance(symbol, JITFunction):
                    compile_wrapper.newline()
                    compile_wrapper.writeline("@triton.jit")
                    compile_wrapper.splice(symbol.src, strip=True)
                    symbols_included.add(symbol_name)
                    traverse(symbol)
                elif hasattr(triton, "constexpr_function") and isinstance(
                    symbol,
                    triton.runtime.jit.ConstexprFunction,
                ):
                    # Import dtype class if used in type annotations
                    if "dtype" in symbol.src and "dtype" not in symbols_included:
                        dtype_symbol = symbol.fn.__globals__.get("dtype")
                        if (
                            dtype_symbol
                            and hasattr(dtype_symbol, "__module__")
                            and dtype_symbol.__module__.startswith("triton")
                        ):
                            compile_wrapper.writeline(
                                f"from {dtype_symbol.__module__} import dtype as dtype"
                            )
                            symbols_included.add("dtype")
                    compile_wrapper.newline()
                    compile_wrapper.writeline("@triton.constexpr_function")
                    compile_wrapper.splice(symbol.src, strip=True)
                    if symbol_name != symbol.fn.__name__:
                        compile_wrapper.writeline(
                            f"{symbol_name} = {symbol.fn.__name__}"
                        )
                    symbols_included.add(symbol_name)
                    traverse(symbol)
                elif isinstance(symbol, (int, str, bool, constexpr)):
                    compile_wrapper.newline()
                    if isinstance(symbol, constexpr):
                        symbol_str = f"tl.constexpr({symbol.value!r})"
                    else:
                        symbol_str = f"{symbol!r}"
                    if annotation := global_annotations.get(symbol_name):
                        if isinstance(annotation, type):
                            annotation_code = (
                                f": {annotation.__module__}.{annotation.__name__}"
                            )
                        else:
                            annotation_code = f": {annotation!r}"
                        compile_wrapper.writeline(
                            f"{symbol_name}{annotation_code} = {symbol_str}"
                        )
                    else:
                        compile_wrapper.writeline(f"{symbol_name} = {symbol_str}")
                    symbols_included.add(symbol_name)
                elif (
                    symbol_name in unqualified_loads
                    and symbol_name != "tl"  # already imported
                    and hasattr(symbol, "__module__")
                    # only codegen imports from triton; JITFunctions
                    # imported from other modules will be codegened
                    # in the separate branch above
                    and symbol.__module__.startswith("triton")
                ):
                    # a global symbol imported from triton is referenced
                    # without module qualification (i.e., `store` instead
                    # of `tl.store`): need to codegen an import

                    # Triton dtype instances have .name instead of .__name__
                    if isinstance(symbol, triton_dtype):
                        compile_wrapper.writeline(f"{symbol_name} = tl.{symbol.name}")
                    elif hasattr(symbol, "__name__"):
                        compile_wrapper.writeline(
                            f"from {symbol.__module__} import {symbol.__name__} as {symbol_name}"
                        )
                    symbols_included.add(symbol_name)

    traverse(kernel)
    return compile_wrapper.getvalue()


def _escape_triton_kernel_source_for_wrapper(src: str) -> str:
    # [FUNCTION DOC] HELPER: Escapes Triton kernel source so it can be embedded in a triple-single-quoted string literal: doubles backslashes, escapes triple single quotes, and under cpp_wrapper also escapes triple double quotes because the text may be nested in a C++ raw string within a python raw string. Returns the escaped string.
    """Escape src for a '''...''' literal, optionally nested in an r\"\"\"...\"\"\" block."""
    src = src.replace("\\", "\\\\")
    if config.cpp_wrapper:
        # With cpp_wrapper + autotune_at_compile_time=False, the source is
        # further embedded in a C++ raw string inside a Python r"""...""" wrapper.
        # So we need to add backslash here.
        src = src.replace('"""', '\\"\\"\\"')
    return src.replace("'''", "\\'\\'\\'")


@dataclasses.dataclass
class SymbolicCallArg:
    # [CLASS DOC] Dataclass pairing a sympy symbol (inner) used as a call argument with the original symbolic expression (inner_expr) it stands for. Prints as the symbol.
    inner: sympy.Symbol
    # the original symbolic expression represented by inner
    inner_expr: sympy.Expr

    def __str__(self):
        # [FUNCTION DOC] Returns the string form of the inner symbol.
        return str(self.inner)


class MemoryPlanningState:
    # [CLASS DOC] Mutable state for the buffer-reuse memory planning pass. Holds a pool mapping regular reuse keys to lists of FreeIfNotReusedLine candidates, a separate pool for communication buffers (comm-to-comm reuse only), and a running total of statically known CPU allocation size. Invariant: lines stored in or popped from the pools are not already marked as reused.
    def __init__(self):
        # [FUNCTION DOC] Initializes empty regular and comm reuse pools (default-list mappings) and a zero total allocated buffer size.
        super().__init__()
        # Regular buffer reuse pool
        self.reuse_pool: dict[ReuseKey, list[FreeIfNotReusedLine]] = (
            collections.defaultdict(list)
        )
        # Separate pool for comm buffers (comm-comm reuse only)
        self.comm_buffer_reuse_pool: dict[
            CommBufferReuseKey, list[FreeIfNotReusedLine]
        ] = collections.defaultdict(list)
        self.total_allocated_buffer_size: int = 0

    def __contains__(self, key: ReuseKey) -> bool:
        # [FUNCTION DOC] True if the regular pool holds at least one free line for the key.
        return bool(self.reuse_pool.get(key, None))

    def pop(self, key: ReuseKey) -> FreeIfNotReusedLine:
        # [FUNCTION DOC] Removes and returns the most recently pushed free line for the key. Raises an assertion error if that line is already reused; raises a key/index error if the pool is empty.
        item = self.reuse_pool[key].pop()
        if item.is_reused:
            raise AssertionError("expected popped reuse pool item to not be reused")
        return item

    def push(self, key: ReuseKey, item: FreeIfNotReusedLine) -> None:
        # [FUNCTION DOC] Pushes a free line onto the pool for the key. Raises an assertion error if the line is already marked reused.
        if item.is_reused:
            raise AssertionError("expected pushed reuse pool item to not be reused")
        self.reuse_pool[key].append(item)

    def comm_buffer_contains(self, key: CommBufferReuseKey) -> bool:
        # [FUNCTION DOC] True if the comm buffer pool holds at least one free line for the key.
        return bool(self.comm_buffer_reuse_pool.get(key, None))

    def comm_buffer_pop(self, key: CommBufferReuseKey) -> FreeIfNotReusedLine:
        # [FUNCTION DOC] Removes and returns the most recent free line from the comm buffer pool for the key; raises an assertion error if it is already reused.
        item = self.comm_buffer_reuse_pool[key].pop()
        if item.is_reused:
            raise AssertionError(
                "expected popped comm buffer reuse pool item to not be reused"
            )
        return item

    def comm_buffer_push(
        self, key: CommBufferReuseKey, item: FreeIfNotReusedLine
    ) -> None:
        # [FUNCTION DOC] Pushes a free line onto the comm buffer pool for the key; raises an assertion error if it is already reused.
        if item.is_reused:
            raise AssertionError(
                "expected pushed comm buffer reuse pool item to not be reused"
            )
        self.comm_buffer_reuse_pool[key].append(item)


class HasWriteLine(Protocol):
    # [CLASS DOC] HELPER: Structural protocol for objects exposing a writeline method accepting a line context, deferred line or string. Its method is a one-line stub.
    def writeline(self, line: LineContext | DeferredLineBase | str) -> None: ...


class WrapperLine:
    # [CLASS DOC] Base class for all wrapper lines, the intermediate representation of generated wrapper code. Subclasses implement codegen (emit python/C++ text into a buffer) and optionally codegen_fx (return the FxConverter method that lowers this line to FX nodes). The base implementations raise NotImplementedError.
    def codegen(self, code: Any) -> None:
        # [FUNCTION DOC] Emits this line into code; the base version always raises NotImplementedError naming the concrete type.
        raise NotImplementedError(f"Codegen not yet supported for type {type(self)}")

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        # [FUNCTION DOC] Returns the FxConverter handler for this line; the base version always raises NotImplementedError naming the concrete type.
        raise NotImplementedError(f"FX codegen not yet supported for type {type(self)}")


@dataclasses.dataclass
class EnterSubgraphLine(WrapperLine):
    # [CLASS DOC] Marks entry into a subgraph. Holds the wrapper and the graph being entered. On construction saves the wrapper current computed sizes on a stack so the subgraph starts fresh scope; codegen pushes the graph as the current codegened graph and indents.
    wrapper: PythonWrapperCodegen
    graph: GraphLowering

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Pushes the wrapper current computed_sizes onto its saved stack (side effect on the wrapper).
        self.wrapper.push_computed_sizes(self.wrapper.computed_sizes)

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Pushes the subgraph as the currently codegened graph on the wrapper and increases the indentation of the code buffer.
        self.wrapper.push_codegened_graph(self.graph)
        code.do_indent()

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        # [FUNCTION DOC] Returns the FxConverter handler for entering a subgraph.
        return converter._generate_enter_subgraph


@dataclasses.dataclass
class SwitchLine(WrapperLine):
    # [CLASS DOC] Line for a conditional switch node. Holds the wrapper and ir.Switch node. Only supported by the FX backend; text codegen raises NotImplementedError.
    wrapper: PythonWrapperCodegen
    node: ir.Switch

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Always raises NotImplementedError: switch is only supported by FX codegen.
        raise NotImplementedError("Only supports FX codegen")

    @staticmethod
    def codegen_fx(converter: FxConverter) -> FxConversionFunc:
        # [FUNCTION DOC] Returns the FxConverter handler that generates a switch.
        return converter._generate_switch


@dataclasses.dataclass
class CommentLine(WrapperLine):
    # [CLASS DOC] Line holding a comment (a line context) to be written verbatim into the generated code.
    line: LineContext

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Writes the stored comment line into the code buffer.
        code.writeline(self.line)

    @staticmethod
    def codegen_fx(converter: FxConverter) -> FxConversionFunc:
        # [FUNCTION DOC] Returns the FxConverter handler that generates a comment.
        return converter._generate_comment


@dataclasses.dataclass
class DynamicScalarLine(WrapperLine):
    # [CLASS DOC] Line representing a dynamic scalar (unbacked symbol from data) node; delegates generation to the wrapper.
    wrapper: PythonWrapperCodegen
    node: ir.DynamicScalar

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Delegates to the wrapper method that generates the dynamic scalar code for the node.
        self.wrapper._codegen_dynamic_scalar(self.node)

    @staticmethod
    def codegen_fx(converter: FxConverter) -> FxConversionFunc:
        # [FUNCTION DOC] Returns the FxConverter handler for dynamic scalars.
        return converter._generate_dynamic_scalar


@dataclasses.dataclass
class ExitSubgraphLine(WrapperLine):
    # [CLASS DOC] Marks exit from a subgraph. On construction restores the wrapper computed_sizes from the saved stack; codegen pops the current codegened graph and decreases indentation.
    wrapper: PythonWrapperCodegen

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Pops the saved computed sizes from the wrapper and assigns them back to computed_sizes (mutates the wrapper).
        self.wrapper.computed_sizes = self.wrapper.pop_computed_sizes()

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Pops the current codegened graph on the wrapper and decreases the indentation of the code buffer.
        self.wrapper.pop_codegened_graph()
        code.do_unindent()

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        # [FUNCTION DOC] Returns the FxConverter handler for exiting a subgraph.
        return converter._generate_exit_subgraph


# [device-as-parameter] Name of the call()-local variable holding the runtime current
# device index under compile-on-one-rank, so the wrapper is byte-identical across ranks.
_COOR_DEVICE_IDX_VAR = COOR_DEVICE_IDX_VAR


def _coor_device_idx_ref(device_idx: int) -> int | str:
    # [FUNCTION DOC] HELPER: Returns the device index to emit in the wrapper. Under compile-on-one-rank this is the name of the call-local runtime device variable so the wrapper is identical on all ranks; otherwise the literal compile-time index.
    """Device index to emit in the wrapper. Under compile-on-one-rank, the call()-local
    runtime current-device variable (so the wrapper is byte-identical across ranks);
    otherwise the literal compile-time index.

    Safe to ignore ``device_idx`` under CooR: make_fx rejects any operand that is not the
    current accelerator (_coor_check_current_accelerator), so every ``device_idx`` reaching
    codegen is already the current device's index; redirecting it to the runtime variable
    just follows each rank's own device.
    """
    return _COOR_DEVICE_IDX_VAR if _coor_enabled() else device_idx


@dataclasses.dataclass
class EnterDeviceContextManagerLine(WrapperLine):
    # [CLASS DOC] Line that enters a device context in generated code. Holds the device index and the last seen device guard index. Python wrapper emits a with-block on the device guard (indented); cpp wrapper emits stream/device guard declarations or updates the index.
    device_idx: int
    last_seen_device_guard_index: int | None

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Emits device-context entry code. For cpp_wrapper: first use declares an AOTI stream guard (AOT mode) and a device guard (JIT mode); later uses in AOT mode must target the same device, otherwise an assertion error is raised, and JIT/dual mode updates the guard index. For the python wrapper: optionally (compile-on-one-rank) assigns the runtime current device index to a variable, then writes a with-statement using the device guard, indents, and writes the set-device statement.
        if V.graph.cpp_wrapper:
            code.writeline("\n")
            # AOTI stream is bound to a device, so CUDAStreamGuard sets both
            # stream and device, and the device cannot change later.
            if self.last_seen_device_guard_index is None:
                code.writeline_aot(
                    f"{V.graph.device_ops.cpp_aoti_stream_guard()} stream_guard(stream, this->device_idx_);"
                )
                code.writeline_jit(
                    f"{V.graph.device_ops.cpp_aoti_device_guard()} device_guard({self.device_idx});"
                )
            else:
                if V.graph.aot_mode:
                    # AOTI emits a single stream_guard on first use; later
                    # contexts can only re-enter the same device. In dual-wrapper
                    # mode the JIT side still needs device_guard updates.
                    if self.last_seen_device_guard_index != self.device_idx:
                        raise AssertionError(
                            "AOTInductor only supports running on one CUDA device"
                        )
                if not V.graph.aot_mode or V.graph.is_dual_wrapper_mode:
                    code.writeline_jit(f"device_guard.set_index({self.device_idx});")
        else:
            # Note _DeviceGuard has less overhead than device, but only accepts
            # integers
            if _coor_enabled():
                # compile-on-one-rank: resolve the device at runtime so the wrapper is
                # byte-identical across ranks -- a shared artifact follows each rank's
                # current device instead of the compile-time index.
                idx = _COOR_DEVICE_IDX_VAR
                code.writeline(
                    f"{idx} = {V.graph.device_ops.current_device_idx_expr()}"
                )
            else:
                idx = self.device_idx
            code.writeline(f"with {V.graph.device_ops.device_guard(idx)}:")
            code.do_indent()
            code.writeline(V.graph.device_ops.set_device(idx))

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        # [FUNCTION DOC] Returns the FxConverter handler for entering a device context manager.
        return converter._generate_enter_device_context_manager


class ExitDeviceContextManagerLine(WrapperLine):
    def codegen(self, code: IndentedBuffer) -> None:
        # [CLASS DOC] Line that exits the device context: in the python wrapper it unindents; the cpp wrapper emits nothing.
        if not V.graph.cpp_wrapper:
            # [FUNCTION DOC] Unindents the code buffer unless generating a cpp wrapper.
            code.do_unindent()

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_exit_device_context_manager
# [FUNCTION DOC] Returns the FxConverter handler for exiting a device context manager.


@dataclasses.dataclass
class ExternKernelAllocLine(WrapperLine):
    # [CLASS DOC] Line for an ExternKernelAlloc node (extern kernel that allocates its own output). Delegates text generation to the wrapper helper.
    wrapper: PythonWrapperCodegen
    node: ir.ExternKernelAlloc

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Builds the argument list from the node positional and keyword codegen args and calls the wrapper extern-kernel-alloc helper.
        node = self.node
        args = [*node.codegen_args(), *node.codegen_kwargs()]
        self.wrapper._generate_extern_kernel_alloc_helper(self.node, args)

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        # [FUNCTION DOC] Returns the FxConverter handler for extern kernel allocation.
        return converter._generate_extern_kernel_alloc


@dataclasses.dataclass
class ExternKernelOutLine(WrapperLine):
    # [CLASS DOC] Line for an ExternKernelOut node (extern kernel writing into a preallocated out buffer).
    wrapper: PythonWrapperCodegen
    node: ir.ExternKernelOut

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Builds args (keyword args without the out arg), picks the kernel name (special-cased to an AOTI shim name for the cpp mm_plus_mm kernel), determines the device type, and calls the wrapper extern-kernel-out helper with the output reference, optional output view reference and stack traces.
        # KNOWN ISSUE: kernel_name is computed once before the branch and again in the else branch (redundant, same value).
        node = self.node
        args = [*node.codegen_args(), *node.codegen_kwargs(skip_out=True)]
        kernel_name = node.get_kernel_name()
        if (
            V.graph.cpp_wrapper
            and node.cpp_kernel_name == "torch::inductor::_mm_plus_mm"
        ):
            # For https://github.com/pytorch/pytorch/issues/128474
            kernel_name = "aoti_torch__mm_plus_mm_out"
        else:
            kernel_name = node.get_kernel_name()
        device = d.type if (d := node.get_device()) else V.graph.device_type
        self.wrapper._generate_extern_kernel_out_helper(
            kernel_name,
            node.codegen_reference(),
            node.output_view.codegen_reference() if node.output_view else None,
            args,
            device,
            self.node.get_stack_traces(),
        )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        # [FUNCTION DOC] Returns the FxConverter handler for extern kernel out.
        return converter._generate_extern_kernel_out


@dataclasses.dataclass
class ExternKernelMultiOutLine(WrapperLine):
    # [CLASS DOC] Codegen line for multi-output out-variant extern kernel calls: calls the kernel with preallocated output buffers passed as keyword arguments. Text codegen only; no FX handler is defined.
    """Codegen line for multi-output .out() variant calls.

    Generates a kernel call with pre-allocated output buffers passed as
    keyword arguments. E.g. kernel(x, out0=buf0, out1=buf1).
    """

    wrapper: PythonWrapperCodegen
    node: ir.ExternKernelMultiOut

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Writes an assignment of the kernel call result to the node name, with normal args plus one keyword argument per out arg name bound to the corresponding output node name.
        node = self.node
        kernel_name = node.get_kernel_name()

        args = [*node.codegen_args(), *node.codegen_kwargs()]
        for out_name, out_node in zip(
            node.out_arg_names, node.out_variant_output_nodes
        ):
            args.append(f"{out_name}={out_node.get_name()}")

        code.writeline(f"{node.get_name()} = {kernel_name}({', '.join(args)})")


@dataclasses.dataclass
class FreeLine(WrapperLine):
    # [CLASS DOC] Line that frees a buffer (or torch bind object) via the wrapper free statement.
    wrapper: PythonWrapperCodegen
    node: BufferLike | ir.TorchBindObject

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Raises an assertion error if the buffer was already removed; otherwise writes the wrapper free statement for the node.
        if self.node.get_name() in V.graph.removed_buffers:
            raise AssertionError(
                f"buffer {self.node.get_name()} was removed but is being freed"
            )
        code.writeline(self.wrapper.make_buffer_free(self.node))

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        # [FUNCTION DOC] Returns the FxConverter handler for freeing a buffer.
        return converter._generate_free


@dataclasses.dataclass
class KernelCallLine(WrapperLine):
    # [CLASS DOC] Line representing a kernel launch. Stores the kernel name, call args, raw keys/args, arg types, triton flag and meta, inductor meta, device, graph name, original fx node name and optional current stream index. Generation delegates to the wrapper kernel call helper.
    wrapper: PythonWrapperCodegen
    kernel_name: str
    call_args: tuple[Any, ...]
    raw_keys: tuple[Any, ...]
    raw_args: tuple[Any, ...]
    arg_types: list[str]
    triton: bool
    triton_meta: TritonMeta
    inductor_meta: dict[str, Any] | None
    device: torch.device
    graph_name: str
    original_fxnode_name: str
    current_stream_idx: int | None = None

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Calls the wrapper kernel call helper forwarding all stored fields.
        self.wrapper._generate_kernel_call_helper(
            self.kernel_name,
            self.call_args,
            triton=self.triton,
            arg_types=self.arg_types,
            raw_keys=self.raw_keys,
            raw_args=self.raw_args,
            triton_meta=self.triton_meta,
            inductor_meta=self.inductor_meta,
            device=self.device,
            graph_name=self.graph_name,
            original_fxnode_name=self.original_fxnode_name,
            current_stream_idx=self.current_stream_idx,
        )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        # [FUNCTION DOC] Returns the FxConverter handler for kernel calls.
        return converter._generate_kernel_call


@dataclasses.dataclass
class KernelDefinitionLine(WrapperLine):
    # [CLASS DOC] Line representing a kernel definition: kernel name, body, optional metadata, gpu flag and optional cpp definition. Generation delegates to the wrapper define-kernel helper.
    wrapper: PythonWrapperCodegen
    kernel_name: str
    kernel_body: str
    metadata: str | None = None
    gpu: bool = True
    cpp_definition: str | None = None

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Calls the wrapper define-kernel helper with the stored name, body, metadata, gpu flag and cpp definition.
        self.wrapper._define_kernel_helper(
            self.kernel_name,
            self.kernel_body,
            metadata=self.metadata,
            gpu=self.gpu,
            cpp_definition=self.cpp_definition,
        )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        # [FUNCTION DOC] Returns the FxConverter handler for kernel definitions.
        return converter._generate_kernel_definition


@dataclasses.dataclass
class MemoryPlanningLine(WrapperLine):
    # [CLASS DOC] Base class for lines participating in memory planning. Holds the wrapper. The plan pass may replace a line (e.g. allocation by reuse); the codegen pass emits text. String form is one line listing the class name and its fields except the wrapper.
    wrapper: PythonWrapperCodegen

    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        # [FUNCTION DOC] First pass hook: given the planning state returns the line to use in its place; the default returns itself.
        """First pass to find reuse"""
        return self

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Second pass hook that emits code; the default emits nothing.
        """Second pass to output code"""

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns a one-line string of the form ClassName(field=value, ...) over all dataclass fields except the wrapper; buffer-typed fields are shown by name.
        """
        Emits a string representation that fits on one line.
        """
        args: list[str] = []
        for field in dataclasses.fields(self):
            if field.name == "wrapper":
                continue
            val = getattr(self, field.name)
            args.append(
                f"{field.name}={val.get_name() if field.type is ir.Buffer else val}"
            )
        return f"{type(self).__name__}({', '.join(args)})"


@dataclasses.dataclass
class EnterDeviceContextManagerWithStreamInfoLine(EnterDeviceContextManagerLine):
    # [CLASS DOC] Enter-device-context line that additionally sets up streams. Fields: num_streams, mapping from stream index to user object index, and whether to cache the default stream. Extends the base entry behaviour.
    """Enter a device context and retrieve user stream objects.

    Attributes:
        num_streams: Number of streams (determined by user annotations on nodes).
        stream_idx_to_user_obj_idx: Maps stream_idx → user_object_index for
            retrieving user stream objects via _get_stream_by_index.
    """

    num_streams: int = 1
    stream_idx_to_user_obj_idx: dict[int, int] = dataclasses.field(default_factory=dict)
    setup_stream_cache: bool = True

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Emits the device context entry (via the base class) and then stream setup: cpp wrapper uses the wrapper stream info prologue; python wrapper optionally stores the current stream as the default stream variable and, for each extra stream index, assigns a stream variable obtained from the user object index.
        """Generate context switching and stream retrieval code."""
        if V.graph.cpp_wrapper:
            super().codegen(code)
            V.graph.wrapper_code.codegen_stream_info_prologue(
                code, self.num_streams, self.stream_idx_to_user_obj_idx
            )
        else:
            super().codegen(code)

            if self.setup_stream_cache:
                code.writeline(
                    f"{DEFAULT_STREAM} = {V.graph.device_ops.current_stream()}"
                )

            if self.num_streams > 1:
                for i in range(1, self.num_streams):
                    user_obj_idx = self.stream_idx_to_user_obj_idx[i]
                    code.writeline(
                        f"{STREAM_NAME_TEMPLATE.format(stream_idx=i)} "
                        f"= _get_stream_by_index({user_obj_idx})",
                    )


@dataclasses.dataclass
class ExitDeviceContextManagerWithStreamInfoLine(ExitDeviceContextManagerLine):
    # [CLASS DOC] Exit-device-context line paired with the stream-info enter line; records the number of streams allocated, which must match the enter line.
    """Exit a device context.

    Attributes:
        num_streams: Number of streams that were allocated (must match Enter).
    """

    num_streams: int = 1

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Unindents the code buffer unless generating a cpp wrapper.
        """Generate context exit code."""
        if not V.graph.cpp_wrapper:
            code.do_unindent()


@dataclasses.dataclass
class EnterCudaStreamContextLine(WrapperLine):
    # [CLASS DOC] Line entering a block executed on a particular device stream, identified by stream_idx.
    """Enter a context executed by the respective device stream.

    Attributes:
        stream_idx: The index number for the entering device stream context.
    """

    stream_idx: int

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] If there is no wrapper_code on the graph, writes a with-statement on the stream name and indents; otherwise delegates to the wrapper to enter the stream context.
        wrapper_code = getattr(V.graph, "wrapper_code", None)
        if wrapper_code is None:
            code.writeline(f"with {get_stream_name(self.stream_idx)}:")
            code.do_indent()
        else:
            wrapper_code.codegen_enter_cuda_stream_context(code, self.stream_idx)


@dataclasses.dataclass
class ExitCudaStreamContextLine(WrapperLine):
    # [CLASS DOC] Line exiting the current stream context.
    """Generate code to exit the current stream context."""

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Unindents when no wrapper_code exists on the graph; otherwise delegates to the wrapper exit-stream hook.
        wrapper_code = getattr(V.graph, "wrapper_code", None)
        if wrapper_code is None:
            code.do_unindent()
        else:
            wrapper_code.codegen_exit_cuda_stream_context(code)


@dataclasses.dataclass
class EnterCudaMemPoolContextLine(WrapperLine):
    # [CLASS DOC] Line entering a CUDA memory pool context for scheduler node codegen; stores the mempool external-object index and device index.
    """Enter a torch.cuda.use_mem_pool context for scheduler node codegen."""

    mempool_index: int
    device_index: int

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Writes a with-statement using the CUDA mem pool context manager with the external object looked up by index and the device index, then indents.
        code.writeline(
            "with torch.cuda.use_mem_pool("
            f"get_external_object_by_index({self.mempool_index}), "
            f"device={self.device_index}):"
        )
        code.do_indent()


class ExitCudaMemPoolContextLine(WrapperLine):
    # [CLASS DOC] Line exiting the active CUDA mem pool context.
    """Exit the active torch.cuda.use_mem_pool context."""

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Unindents the code buffer.
        code.do_unindent()


class EfficientPeakEstimate:
    # [CLASS DOC] Estimates peak memory of the scheduled graph to decide if buffer reuse would raise the peak. Holds the overall peak and a segmented tree over per-scheduler-node peaks that supports range queries and range updates.
    def __init__(self):
        # [FUNCTION DOC] Computes overall peak memory and per-node peak list from scheduler nodes, graph inputs, and outputs (freeable input buffers), and builds a segmented tree over it with addition for updates and max for summaries.
        from ..memory import estimate_peak_memory, get_freeable_input_buf

        scheduler_nodes = V.graph.scheduler.nodes
        graph_inputs = OrderedSet(V.graph.graph_inputs.keys())
        graph_outputs = OrderedSet(V.graph.get_output_names())
        names_to_freeable_bufs = get_freeable_input_buf(scheduler_nodes, graph_inputs)
        self.overall_peak_memory, peak_by_scheduler_node = estimate_peak_memory(
            scheduler_nodes,
            names_to_freeable_bufs,
            graph_outputs,
        )

        from .segmented_tree import SegmentedTree

        self.segmented_tree = SegmentedTree(
            peak_by_scheduler_node, operator.add, max, 0
        )

    def _get_size(self, node: BufferLike) -> int:
        # [FUNCTION DOC] HELPER: Returns the allocation size of the buffer in bytes (hint size times dtype size; unknown sizes hint as 0).
        return V.graph.sizevars.optimization_hint(
            V.graph.get_allocation_storage_size(node), fallback=0
        ) * get_dtype_size(node.get_dtype())

    def peak_between(self, line_a: FreeIfNotReusedLine, line_b: AllocateLine):
        # [FUNCTION DOC] Returns the peak memory over the scheduler node range strictly between the free line node and the allocate line node.
        return self.segmented_tree.summarize_range(
            line_a.scheduler_node_index + 1, line_b.scheduler_node_index - 1
        )

    def update_peak_between(self, line_a: FreeIfNotReusedLine, line_b: AllocateLine):
        # [FUNCTION DOC] Records that the buffer of line_b stays live over the range strictly between the two lines, by adding its size to that range of the tree. Does nothing if the lines are adjacent.
        if line_a.scheduler_node_index + 1 == line_b.scheduler_node_index:
            return
        self.segmented_tree.update_range(
            line_a.scheduler_node_index + 1,
            line_b.scheduler_node_index - 1,
            self._get_size(line_b.node),
        )


@dataclasses.dataclass
class AllocateLine(MemoryPlanningLine):
    # [CLASS DOC] Memory planning line for allocating a buffer. Fields: the buffer node and a comm_buffer flag. On construction records the index of the current scheduler node (requires one to be set). Planning may turn the allocation into a ReuseLine or drop it for removed buffers.
    """Represents a buffer allocation during memory planning."""

    node: BufferLike
    comm_buffer: bool = False

    def __post_init__(self):
        # [FUNCTION DOC] Records the scheduler node index of the current scheduler node; raises an assertion error if there is no current node.
        if V.graph.scheduler.current_node is None:
            raise AssertionError("expected scheduler.current_node to be set")
        self.scheduler_node_index = V.graph.scheduler.nodes.index(
            V.graph.scheduler.current_node
        )

    def should_reuse_buffer(self, free_line: FreeIfNotReusedLine, size: int) -> bool:
        # [FUNCTION DOC] Decides whether a free line's buffer may be reused: always for comm buffers or when the free was just before this allocation; otherwise only if adding the buffer size to the peak in between does not exceed the overall peak.
        if self.comm_buffer:
            return True
        if free_line.scheduler_node_index + 1 == self.scheduler_node_index:
            return True
        overall_peak_memory = self.wrapper.estimate_peak.overall_peak_memory
        peak_memory_in_range = self.wrapper.estimate_peak.peak_between(free_line, self)
        new_peak_memory = size + peak_memory_in_range
        return new_peak_memory <= overall_peak_memory

    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        # [FUNCTION DOC] Plans this allocation. Removed buffers become NullLine. Comm buffers reuse a pooled comm free line if reuse is allowed. Regular buffers reuse a matching pooled free line if allowed and should_reuse_buffer agrees (marking it reused and updating the peak estimate), else the line is pushed back. Otherwise, for statically-shaped CPU buffers, adds the element count to the total allocated size. Returns the line to use.
        if self.node.get_name() in V.graph.removed_buffers:
            return NullLine(self.wrapper)

        if self.comm_buffer:
            # Comm buffers use separate pool (comm-comm reuse only)
            key = comm_buffer_reuse_key(self.node)
            if config.allow_buffer_reuse and state.comm_buffer_contains(key):
                free_line = state.comm_buffer_pop(key)
                free_line.is_reused = True
                return ReuseLine(
                    self.wrapper, free_line.node, self.node, comm_buffer=True
                )
            return self

        # Regular buffer reuse
        # Stream is part of the key, so cross-stream reuse is naturally prevented.
        key = buffer_reuse_key(self.node)
        if config.allow_buffer_reuse and key in state:
            free_line = state.pop(key)
            size = V.graph.sizevars.optimization_hint(
                V.graph.get_allocation_storage_size(self.node), fallback=0
            ) * get_dtype_size(self.node.get_dtype())
            if self.should_reuse_buffer(free_line, size):
                free_line.is_reused = True
                self.wrapper.estimate_peak.update_peak_between(free_line, self)
                return ReuseLine(self.wrapper, free_line.node, self.node)
            else:
                state.push(key, free_line)
                return self

        if self.node.get_device_or_error().type == "cpu":
            static_shape = self.wrapper.static_shape_for_buffer_or_none(self.node)
            if static_shape is not None:
                state.total_allocated_buffer_size += int(
                    functools.reduce(operator.mul, static_shape, 1)
                )

        return self

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Raises an assertion error if the buffer was removed; delegates comm buffers to the comm-buffer codegen path, otherwise emits the allocation.
        if self.node.get_name() in V.graph.removed_buffers:
            raise AssertionError(
                f"buffer {self.node.get_name()} was removed but is being freed"
            )
        if self.comm_buffer:
            self._codegen_comm_buffer(code)
        else:
            code.writeline(self.wrapper.make_buffer_allocation(self.node))

    def _codegen_comm_buffer(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] HELPER: Emits the allocation statement for a communication buffer. Requires a device with an index and a CommBufferLayout (else assertion error). For symmetric-memory buffers writes an assignment calling the p2p empty-strided allocator with shape, stride, dtype, device (runtime current device under compile-on-one-rank, otherwise the compile-time cuda index), group name and a random allocation id. Any other comm buffer type raises NotImplementedError.
        """Generate allocation code for comm buffers."""
        name = self.node.get_name()
        device = self.node.get_device()
        if not (device is not None and device.index is not None):
            raise AssertionError(
                f"Comm buffer requires a valid CUDA device with index, got {device}"
            )
        dtype = self.node.get_dtype()
        shape = tuple(self.node.get_size())
        stride = tuple(self.node.get_stride())
        layout = self.node.get_output_spec()
        if not isinstance(layout, ir.CommBufferLayout):
            raise AssertionError(f"expected CommBufferLayout, got {type(layout)}")
        comm_buffer_type = layout.comm_buffer_type
        group_name = layout.group_name

        if comm_buffer_type == ir.CommBufferType.SYMM_MEM:
            # [device-as-parameter] under compile-on-one-rank the comm buffer must follow
            # the running rank's device, not the compile-time index.
            if _coor_enabled():
                device_arg = f'torch.device("cuda", {V.graph.device_ops.current_device_idx_expr()})'
            else:
                device_arg = f'torch.device("cuda:{device.index}")'
            line = (
                f"{name} = empty_strided_p2p("
                f"{self.wrapper.codegen_shape_tuple(shape)}, "
                f"{self.wrapper.codegen_shape_tuple(stride)}, "
                f"{dtype}, "
                f"{device_arg}, "
                f'group_name="{group_name}", '
                f"alloc_id={random.randint(0, 2**64 - 1)})"
            )
        else:
            raise NotImplementedError(
                f"Unsupported comm buffer type: {comm_buffer_type}"
            )
        code.writeline(line)

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        # [FUNCTION DOC] Returns the FxConverter handler for allocating a comm buffer or a normal buffer, depending on the comm_buffer flag.
        if self.comm_buffer:
            return converter._generate_comm_buffer_allocate
        return converter._generate_allocate


@dataclasses.dataclass
class FreeIfNotReusedLine(MemoryPlanningLine):
    # [CLASS DOC] Memory planning line that frees a buffer unless it is later reused. Fields: node, is_reused flag (set by a later allocation that takes over the buffer), comm_buffer flag. Records the scheduler node index at creation (requires a current scheduler node). In planning it registers itself in the matching reuse pool.
    node: BufferLike
    is_reused: bool = False
    comm_buffer: bool = False

    def __post_init__(self):
        # [FUNCTION DOC] Records the index of the current scheduler node; raises an assertion error if none is set.
        if V.graph.scheduler.current_node is None:
            raise AssertionError("expected scheduler.current_node to be set")
        self.scheduler_node_index = V.graph.scheduler.nodes.index(
            V.graph.scheduler.current_node
        )

    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        # [FUNCTION DOC] Plans the free. Returns self unchanged for buffers that alias an input, multi-output layouts; raises an assertion error if already reused; returns NullLine for removed buffers. Otherwise, if buffer reuse is enabled, pushes itself on the comm pool or regular pool under the proper reuse key. Returns self.
        if len(self.node.get_inputs_that_alias_output()) > 0:
            return self
        if isinstance(self.node.layout, ir.MultiOutputLayout):
            return self
        if self.is_reused:
            raise AssertionError("expected line to not be reused")
        if self.node.get_name() in V.graph.removed_buffers:
            return NullLine(self.wrapper)
        if config.allow_buffer_reuse:
            if self.comm_buffer:
                # Comm buffers use separate pool (comm-comm reuse only)
                key = comm_buffer_reuse_key(self.node)
                state.comm_buffer_push(key, self)
            else:
                key = buffer_reuse_key(self.node)
                state.push(key, self)
        return self

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Raises an assertion error for removed buffers. If the buffer was not reused, writes the free statement (comm buffers get a trailing comment naming the comm buffer type, and require a CommBufferLayout); otherwise emits nothing.
        if self.node.get_name() in V.graph.removed_buffers:
            raise AssertionError(
                f"buffer {self.node.get_name()} was removed but is being freed"
            )
        if not self.is_reused:
            line = self.wrapper.make_buffer_free(self.node)
            if self.comm_buffer:
                layout = self.node.get_output_spec()
                if not isinstance(layout, ir.CommBufferLayout):
                    raise AssertionError(
                        f"expected CommBufferLayout, got {type(layout)}"
                    )
                code.writeline(f"{line} # {layout.comm_buffer_type.value} buffer free")
            else:
                code.writeline(line)

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        # [FUNCTION DOC] Returns the FxConverter handler for freeing a comm buffer or for a conditional free.
        if self.comm_buffer:
            return converter._generate_comm_buffer_free
        return converter._generate_free_if_not_reused


@dataclasses.dataclass
class ReinterpretLine(MemoryPlanningLine):
    # [CLASS DOC] Memory planning line that reinterprets reused storage: fields node, reused_as and layout. Does not change at planning; codegen requires a NonOwningLayout whose view is a ReinterpretView and defers allocation of the reused buffer through the wrapper.
    node: BufferLike
    reused_as: BufferLike
    layout: ir.Layout

    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        # [FUNCTION DOC] Planning is a no-op; returns self.
        return self

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Asserts the layout is a NonOwningLayout wrapping a ReinterpretView (assertion errors otherwise) and asks the wrapper to generate the deferred allocation for the reused-as buffer using that view.
        if not isinstance(self.layout, ir.NonOwningLayout):
            raise AssertionError(f"expected NonOwningLayout, got {type(self.layout)}")
        if not isinstance(self.layout.view, ir.ReinterpretView):
            raise AssertionError(
                f"expected ReinterpretView, got {type(self.layout.view)}"
            )
        self.wrapper.codegen_deferred_allocation(
            self.reused_as.get_name(), self.layout.view
        )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        # [FUNCTION DOC] Returns the FxConverter handler for reinterpret lines.
        return converter._generate_reinterpret


@dataclasses.dataclass
class ReuseLine(MemoryPlanningLine):
    # [CLASS DOC] Memory planning line meaning buffer reused_as takes over the storage of node. Fields: node, reused_as, delete_old (whether the old name is deleted) and comm_buffer flag.
    node: BufferLike
    reused_as: BufferLike
    delete_old: bool = True
    comm_buffer: bool = False

    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        # [FUNCTION DOC] Returns NullLine if the old buffer was removed (and the new one must also be removed, otherwise assertion error); raises an assertion error if only the new one was removed; otherwise returns self.
        if self.node.get_name() in V.graph.removed_buffers:
            if self.reused_as.get_name() not in V.graph.removed_buffers:
                raise AssertionError(
                    f"reuse target {self.reused_as.get_name()} expected to be removed"
                )
            return NullLine(self.wrapper)
        if self.reused_as.get_name() in V.graph.removed_buffers:
            raise AssertionError(
                f"reuse target {self.reused_as.get_name()} was removed unexpectedly"
            )
        return self

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Raises an assertion error if either buffer was removed; otherwise writes the wrapper buffer-reuse statement for the pair, honoring delete_old.
        if self.node.get_name() in V.graph.removed_buffers:
            raise AssertionError(
                f"buffer {self.node.get_name()} was removed but is being freed"
            )
        if self.reused_as.get_name() in V.graph.removed_buffers:
            raise AssertionError(
                f"reuse target {self.reused_as.get_name()} was removed unexpectedly"
            )
        code.writeline(
            self.wrapper.make_buffer_reuse(self.node, self.reused_as, self.delete_old)
        )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        # [FUNCTION DOC] Returns the FxConverter handler for reuse lines.
        return converter._generate_reuse


class NullLine(MemoryPlanningLine):
    # [CLASS DOC] Memory planning line that emits no code (a placeholder for removed buffers); only has an FX handler producing nothing.
    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        # [FUNCTION DOC] Returns the FxConverter handler for null lines.
        return converter._generate_null


@dataclasses.dataclass
class MultiOutputLine(WrapperLine):
    # [CLASS DOC] Given a buffer with multi-output layout, defines result_name as the element reached by following a path of indices into arg_name. Fields: wrapper, result_name, arg_name and indices (sequence of (container type, index) pairs).
    """
    Given a MultiOutputLayout buffer, indexes actual buffer(s) from the result.
    """

    wrapper: PythonWrapperCodegen
    result_name: str
    arg_name: str
    indices: Sequence[Any]

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Builds the access expression by walking the index path (lists via subscript, tuples via the wrapper tuple access hook so C++ can use std::get, dicts via string keys; any other container type raises an assertion error) and writes a declaration of result_name assigned to it, using the wrapper declare and ending strings.
        def codegen_list_tuple_access(basename, indices):  # type: ignore[no-untyped-def]
            # [FUNCTION DOC] HELPER: Recursively applies the first (container type, index) pair to the base expression and continues with the rest; returns the final expression string. Raises an assertion error for unsupported container types.
            if len(indices) > 0:
                itype, i = indices[0]
                if issubclass(itype, list):
                    return codegen_list_tuple_access(f"{basename}[{i}]", indices[1:])
                elif issubclass(itype, tuple):
                    # cpp wrapper code needs to use std::get<> to access a tuple
                    tuple_access = self.wrapper.codegen_tuple_access(
                        basename, self.result_name, str(i)
                    )
                    return codegen_list_tuple_access(tuple_access, indices[1:])
                elif issubclass(itype, dict):
                    return codegen_list_tuple_access(f"{basename}['{i}']", indices[1:])
                else:
                    raise AssertionError("non supported index type: ", itype)
            else:
                return basename

        value = codegen_list_tuple_access(self.arg_name, self.indices)
        code.writeline(
            f"{self.wrapper.declare}{self.result_name} = {value}{self.wrapper.ending}"
        )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        # [FUNCTION DOC] Returns the FxConverter handler for multi-output lines.
        return converter._generate_multi_output


@dataclasses.dataclass
class IndexPutFallbackLine(WrapperLine):
    # [CLASS DOC] Line for an index_put fallback node: holds wrapper, the IndexPutFallback node and its list of index nodes (None allowed).
    wrapper: PythonWrapperCodegen
    node: ir.IndexPutFallback
    indices: list[ir.IRNode | None]

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Requires node inputs to be a node sequence (else assertion error); takes the first two inputs as self and values, converts index nodes to references (the wrapper none string for missing ones) and calls the wrapper index_put fallback generator with the kernel name, indices and constant args.
        node = self.node
        if not ir.is_node_sequence(node.inputs):
            raise AssertionError("expected node.inputs to be a node sequence")
        (x, values) = (t.codegen_reference() for t in node.inputs[:2])
        indices = [
            idx.codegen_reference() if idx else self.wrapper.none_str
            for idx in self.indices
        ]

        self.wrapper._generate_index_put_fallback(
            node.get_kernel_name(), x, indices, values, *node.codegen_const_args()
        )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        # [FUNCTION DOC] Returns the FxConverter handler for index_put fallback.
        return converter._generate_index_put_fallback


@dataclasses.dataclass
class ScatterFallbackLine(WrapperLine):
    # [CLASS DOC] Line for a scatter fallback node: holds the wrapper and ScatterFallback node.
    wrapper: PythonWrapperCodegen
    node: ir.ScatterFallback

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Requires node inputs to be a node sequence; extracts self, index and src (src is a tensor input or a constant arg), determines device type and calls the wrapper scatter fallback generator with the dim, kernel names, reduce mode and kwargs.
        node = self.node
        if not ir.is_node_sequence(node.inputs):
            raise AssertionError("expected node.inputs to be a node sequence")
        if node.src_is_tensor:
            (x, index, src) = (t.codegen_reference() for t in node.inputs)
        else:
            (x, index) = (t.codegen_reference() for t in node.inputs)
            src = node.constant_args[1]
        device = d.type if (d := node.get_device()) else V.graph.device_type
        self.wrapper._generate_scatter_fallback(
            x,
            [x, node.constant_args[0], index, src],
            node.cpp_kernel_name,
            node.python_kernel_name,
            node.src_is_tensor,
            node.kwargs["reduce"],
            node.codegen_kwargs(),
            device,
        )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        # [FUNCTION DOC] Returns the FxConverter handler for scatter fallback.
        return converter._generate_scatter_fallback


@dataclasses.dataclass
class SymbolicCallArgLine(WrapperLine):
    # [CLASS DOC] Line defining a symbolic call argument in a given graph; delegates to the wrapper helper.
    wrapper: PythonWrapperCodegen
    arg: SymbolicCallArg
    graph: GraphLowering

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Calls the wrapper symbolic call arg helper with the argument and graph.
        self.wrapper._generate_symbolic_call_arg_helper(self.arg, self.graph)

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        # [FUNCTION DOC] Returns the FxConverter handler for symbolic call args.
        return converter._generate_symbolic_call_arg


@dataclasses.dataclass
class UnbackedSymbolDefsLine(WrapperLine):
    # [CLASS DOC] Line defining unbacked symbols bound from the outputs of an operation: holds output name, outputs and the mapping from symbol to key path.
    wrapper: PythonWrapperCodegen
    output_name: str
    outputs: Any
    unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] | None

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Delegates to the wrapper to emit the unbacked symbol definitions for the outputs.
        self.wrapper._codegen_unbacked_symbol_defs_for_outputs(
            self.output_name, self.outputs, self.unbacked_bindings
        )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        # [FUNCTION DOC] Returns the FxConverter handler for unbacked symbol definitions.
        return converter._generate_unbacked_symbol_defs


@dataclasses.dataclass
class AssertSizeStrideLine(WrapperLine):
    # [CLASS DOC] Line asserting size and stride of a named tensor: fields name, size and stride strings, op_name label (default input) and optional dtype.
    wrapper: PythonWrapperCodegen
    name: str
    size: str
    stride: str
    op_name: str = "input"
    dtype: torch.dtype | None = None

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Delegates to the wrapper size/stride assertion emitter writing into the code buffer.
        self.wrapper._codegen_assert_size_stride(
            code, self.name, self.size, self.stride, self.op_name, self.dtype
        )

    @staticmethod
    def codegen_fx(converter: FxConverter) -> FxConversionFunc:
        # [FUNCTION DOC] Returns the FxConverter handler for size/stride assertions.
        return converter._generate_assert_size_stride


@dataclasses.dataclass
class GroupedAssertSizeStrideLine(WrapperLine):
    # [CLASS DOC] Line asserting size and stride for several tensors at once: list of (name, size, stride) strings and an op_name label.
    wrapper: PythonWrapperCodegen
    asserts: list[tuple[str, str, str]]
    op_name: str = "input"

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Delegates to the wrapper grouped assertion emitter.
        self.wrapper._codegen_assert_size_stride_grouped(
            code, self.asserts, self.op_name
        )

    @staticmethod
    def codegen_fx(converter: FxConverter) -> FxConversionFunc:
        # [FUNCTION DOC] Returns the FxConverter handler for size/stride assertions (the same one as the single-tensor line).
        return converter._generate_assert_size_stride


@dataclasses.dataclass
class AssertDivByZeroLine(WrapperLine):
    # [CLASS DOC] Deferred runtime check that a divisor expression is non-zero (AOTI); queued during build and replayed at the position of use. Has no FX handler.
    """Deferred AOTI runtime check that a sizevar divisor is non-zero.

    Queued from CppWrapperCpu.codegen_cpp_sizevar during the build phase
    and replayed at the position of use so the check lands after the
    declarations of any unbacked symbols it references.
    """

    wrapper: PythonWrapperCodegen
    divisor: str
    op_name: str

    def codegen(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Delegates to the wrapper div-by-zero emitter (which raises in the python wrapper).
        self.wrapper._codegen_assert_div_by_zero(code, self.divisor, self.op_name)


BufferName = str
Line = MemoryPlanningLine | LineContext


class PythonWrapperCodegen(CodeGen):
    # [CLASS DOC] Generates the outer python wrapper that calls the kernels. State: several IndentedBuffers (imports, header, prefix, suffix, wrapper_call, kernel declarations, autotune defs/calls, subgraph definitions); the list of lines (MemoryPlanningLine/LineContext/WrapperLine) to be replayed; allocation, free and reuse tracking; kernel and meta caches; pending input assertion and alignment-copy bookkeeping; unbacked symbol declarations; computed sizes and codegened graph stacks; stream and device guard state. Subclasses (cpp, gpu, subgraph wrappers) override emission hooks. Key flow: write header and prefix at construction, collect lines during lowering, plan memory, then generate the final module source.
    """
    Generate outer wrapper in Python that calls the kernels.
    """

    supports_caching: bool = True  # Whether the output code is cacheable.

    def __init__(self):
        # [FUNCTION DOC] Initializes all wrapper state (buffers, caches, tracking sets, flags depending on cpp_wrapper), sets the launcher function name, writes the header, the prefix (unless this is a graph partition subgraph) and autotune defs header, writes constants for non-AOT mode, and prepares import-once and raw-stream caching helpers and the debug printer. Reads global graph (V.graph) state and config; mutates only self.
        super().__init__()
        self._last_default_stream_device: int | None = None
        self._pending_input_asserts: dict[str, tuple[str, str]] = {}
        self._pending_alignment_copies: OrderedSet[str] = OrderedSet()
        # Inputs read on more than one stream get a copy_if_misaligned per
        # consuming stream (each cloning a preserved copy of the original), so
        # every stream is ordered after its own aligned clone.  A single shared
        # copy would leave other streams reading a clone made on -- and ordered
        # only after -- the first reader's stream, a cross-stream race.  These
        # track the reclassified inputs, the saved originals, and the stream
        # each working name is currently aligned for.
        self._multistream_alignment_copies: OrderedSet[str] = OrderedSet()
        self._alignment_orig_saved: OrderedSet[str] = OrderedSet()
        self._alignment_aligned_stream: dict[str, int] = {}
        self._names_iter: Iterator[int] = count()
        self.args_to_buffers: dict[
            str, None | ir.TensorBox | ir.Buffer | ir.TorchBindObject
        ] = {}
        self.imports = IndentedBuffer()
        self.header = make_codegen_buffer()
        self.prefix = make_codegen_buffer()
        self.suffix = IndentedBuffer()
        self.kernel_declarations = IndentedBuffer()
        self.wrapper_call = make_codegen_buffer()
        self.kernel_autotune_defs = IndentedBuffer()
        self.kernel_autotune_calls = IndentedBuffer()
        self.subgraph_definitions = IndentedBuffer()
        self.kernel_autotune_names: OrderedSet[str] = OrderedSet()
        # Map key is the kernel argument name; value is a tuple of the resulting example
        # tensor name with the kernel where that tensor was most recently used.
        self.kernel_autotune_example_args: dict[str, tuple[str, str]] = {}
        self.kernel_autotune_tmp_arg_idx: int = 0
        # If the generated source code is exactly the same, reuse the
        # pre-existing kernel for it
        self.src_to_kernel: dict[str, str] = {}
        self.kernel_numel_expr: OrderedSet[tuple[str, GraphLowering]] = OrderedSet()
        self.lines: list[Line] = []
        self.declare = ""
        self.declare_maybe_reference = ""
        self.ending = ""
        self.comment = "#"
        self.none_str = "None"
        self.move_begin = "std::move(" if V.graph.cpp_wrapper else ""
        self.move_end = ")" if V.graph.cpp_wrapper else ""
        self.last_seen_device_guard_index: int | None = None
        # [device-as-parameter] CooR harness render mode, resolved once and cached:
        # None = unresolved, False = compile-on-one-rank off, else the current accelerator.
        self._coor_current_accelerator: torch.device | Literal[False] | None = None
        self.supports_intermediate_hooks = True
        self.user_defined_kernel_cache: dict[
            tuple[Any, ...], tuple[str, Any, dict[str, Any]]
        ] = {}
        self.unbacked_symbol_decls: OrderedSet[str] = (
            OrderedSet()
        )  # str of sympy.Symbol
        self.computed_sizes: OrderedSet[sympy.Symbol] = OrderedSet()
        self.launcher_fn_name = None
        # This function can be overridden to change the launcher name
        self.set_launcher_fn_name()

        # this is used for tracking which GraphLowering instance---parent graph
        # or (nested) subgraph---is currently codegened; the primary use case is
        # including the graph instance into a cache key to avoid cross-graph
        # caching during lowering of nested subgraphs
        self.codegened_graph_stack = []
        self.computed_sizes_stack = []

        self.write_header()

        if not is_codegen_graph_partition_subgraph(self):
            # See [Note: Removed Graph Partition Arguments]
            self.write_prefix()

        self.write_kernel_autotune_defs_header()

        if not V.graph.aot_mode:
            for name, hashed in V.graph.constant_reprs.items():
                # include a hash so our code cache puts different constants into different files
                self.write_constant(name, hashed)

        self.allocated = OrderedSet[BufferName]()
        self.freed = OrderedSet[BufferName]()

        # maps from reusing buffer to reused buffer
        self.reuses: dict[BufferName, BufferName] = {}

        self.write_get_raw_stream = functools.lru_cache(None)(  # type: ignore[assignment]
            self.write_get_raw_stream
        )

        @functools.cache
        def add_import_once(line: str) -> None:
            # [FUNCTION DOC] HELPER: Writes an import line to the imports buffer once (cached by line text) and also to the autotune calls buffer when autotune at compile time is enabled.
            self.imports.writeline(line)
            if config.triton.autotune_at_compile_time:
                self.kernel_autotune_calls.writeline(line)

        self.add_import_once = add_import_once
        self._metas: dict[str, str] = {}
        self._meta_vars: OrderedSet[str] = OrderedSet()
        self.multi_kernel_state = MultiKernelState()
        self.already_codegened_subgraphs: OrderedSet[str] = OrderedSet()
        self.allocated_workspaces: dict[str, Any] = {}

        # intermediate tensor value printing utility
        self.debug_printer = DebugPrinterManager(
            debug_printer_level=config.aot_inductor.debug_intermediate_value_printer,
            use_array_ref=config.aot_inductor.allow_stack_allocation,
        )

        # Additional files that are dependent to the wrapper (ex. cubin files)
        self.additional_files = []

        # Precompiled kernel .so paths to link into the JIT cpp_wrapper
        # (e.g. CUTLASS, ROCm CK). Populated by the scheduling layer when
        # cpp_wrapper is set and not aot_mode.
        self.external_kernel_libs: OrderedSet[str] = OrderedSet()

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: PythonWrapperCodegen | None,
        partition_signatures: ir.GraphPartitionSignature | None = None,
    ):
        # [FUNCTION DOC] Factory: returns a SubgraphPythonWrapperCodegen for subgraphs (requires subgraph name and parent wrapper, else assertion error) or a plain PythonWrapperCodegen.
        if is_subgraph:
            if subgraph_name is None:
                raise AssertionError("expected subgraph_name to be set")
            if parent_wrapper is None:
                raise AssertionError("expected parent_wrapper to be set")
            return SubgraphPythonWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )
        return PythonWrapperCodegen()

    def set_launcher_fn_name(self) -> None:
        # [FUNCTION DOC] Sets the launcher function name to call; subclasses override to change it.
        # pyrefly: ignore [bad-assignment]
        self.launcher_fn_name = "call"

    def write_constant(self, name: str, hashed: str) -> None:
        # [FUNCTION DOC] Writes a header line defining the constant name as None with a trailing comment containing the hash.
        self.header.writeline(f"{name} = None  # {hashed}")

    def write_header(self) -> None:
        # [FUNCTION DOC] Writes the standard import block into the imports buffer (optionally a debug-utils import and the AOT id comment from the tracing context) and the header definitions (aten aliases, assert and empty_strided helpers, reinterpret_tensor, AsyncCompile instance, optional symmetric memory allocator, optional nvtx and proton profiling setup including output dir creation, finalize hook and atexit registration).
        """Write the header section of the generated Python wrapper code."""
        context = torch._guards.TracingContext.try_get()
        aot_config_comment = ""
        if context is not None and context.aot_graph_name is not None:
            aot_config_comment = f"# AOT ID: {context.aot_graph_name}"
        inductor_debug_utils = ""
        if int(config.aot_inductor.debug_intermediate_value_printer) > 0:
            inductor_debug_utils = "from torch._inductor.codegen.debug_utils import _print_debugging_tensor_value_info"
        elif torch._inductor.config.test_configs.track_memory_lifecycle:
            inductor_debug_utils = "from torch._inductor.runtime.debug_utils import tracked_empty_strided\n"

        self.imports.splice(
            f"""
                {aot_config_comment}
                from ctypes import c_void_p, c_long, c_int
                import torch
                import math
                import random
                import os
                import tempfile
                from math import inf, nan
                from cmath import nanj
                from torch._inductor.hooks import run_intermediate_hooks
                from torch._inductor.utils import maybe_profile
                from torch._inductor.codegen.memory_planning import _align as align
                from torch import device, empty_strided
                from {async_compile.__name__} import AsyncCompile
                from torch._inductor.select_algorithm import extern_kernels
                {inductor_debug_utils}
            """,
            strip=True,
        )
        self.header.splice(
            """
                aten = torch.ops.aten
                inductor_ops = torch.ops.inductor
                _quantized = torch.ops._quantized
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
                assert_size_stride_grouped = torch._C._dynamo.guards.assert_size_stride_grouped
                assert_alignment = torch._C._dynamo.guards.assert_alignment
                empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
                empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
                empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
                empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
                empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
                reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
                alloc_from_pool = torch.ops.inductor._alloc_from_pool
                async_compile = AsyncCompile()
            """,
            strip=True,
        )
        try:
            # Only add empty_strided_p2p() if distributed and SymmetricMemory
            # is available
            from torch._C._distributed_c10d import _SymmetricMemory  # noqa: F401

            self.header.splice(
                """
                empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
                """,
                strip=True,
            )
        except (AttributeError, ImportError):
            pass
        if config.annotate_training:
            self.header.writeline("from torch.cuda import nvtx")
        if config.triton.proton_profiling:
            self.header.writeline("import triton.profiler as proton")
            self.header.writeline("import triton.profiler.language as pl")
            self.header.writeline(
                "from triton.profiler.hooks import HookManager as _ProtonHookManager"
            )
            self.header.writeline("import triton")
            self.header.writeline("import atexit")
            self.header.writeline("import os")
            self.header.writeline(
                "triton.set_allocator(lambda size, align, stream: "
                "torch.empty(size, dtype=torch.uint8, device='cuda'))"
            )
            output_dir = config.triton.proton_output_dir or os.path.join(
                get_debug_dir(), "proton"
            )
            self.header.writeline(f'os.makedirs("{output_dir}", exist_ok=True)')
            proton_name = f'os.path.join("{output_dir}", "inductor")'
            trace_path = f'os.path.join("{output_dir}", "inductor.chrome_trace")'
            group_by_sm = config.triton.proton_group_by_sm
            split_invocations = config.triton.proton_split_invocations
            per_cta_occupancy = config.triton.proton_per_cta_occupancy
            self.header.writeline(
                "from torch._inductor.runtime.proton_utils import process_proton_trace as _proton_process_trace"
            )
            self.header.splice(
                f"""
                def _proton_finalize_and_postprocess():
                    proton.finalize()
                    _trace_path = {trace_path}
                    if os.path.exists(_trace_path):
                        _proton_process_trace(
                            _trace_path,
                            group_by_sm={group_by_sm},
                            split_invocations={split_invocations},
                            per_cta_occupancy={per_cta_occupancy},
                        )
                """
            )
            # Start proton before kernel compilation (instrumentation backend needs to hook JIT)
            self.header.writeline(
                "if not _ProtonHookManager.active_hooks: "
                f'proton.start({proton_name}, backend="instrumentation", data="trace"); '
                "atexit.register(_proton_finalize_and_postprocess)"
            )
            self.header.writeline('pl.enable_semantic("triton")')

    def include_extra_header(self, header: str):
        # [FUNCTION DOC] Hook for adding an extra header; the base implementation does nothing.
        pass

    def write_kernel_autotune_defs_header(self) -> None:
        # [FUNCTION DOC] Writes the imports and helper definitions needed in the compile-time autotune block into kernel_autotune_defs, plus a raw stream getter if the CUDA binding is available.
        self.kernel_autotune_defs.splice(
            f"""
                import torch
                from math import inf, nan
                from torch._dynamo.testing import rand_strided
                from torch._dynamo.utils import preserve_rng_state
                from torch._inductor.select_algorithm import AlgorithmSelectorCache
                from {async_compile.__name__} import AsyncCompile

                async_compile = AsyncCompile()
                generate_example_value = AlgorithmSelectorCache.generate_example_value
                empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
                empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
            """
        )

        try:
            from torch._C import _cuda_getCurrentRawStream  # noqa: F401

            self.kernel_autotune_defs.splice(
                """
                get_raw_stream = torch._C._cuda_getCurrentRawStream
                """,
                strip=True,
            )
        except (ImportError, AttributeError):
            pass

    @cache_on_self
    def write_triton_header_once(self) -> None:
        # [FUNCTION DOC] Once per wrapper (cached): writes triton imports (triton, tl, start/end graph helpers) to the autotune calls buffer when autotuning at compile time, and to imports (plus the device raw stream import) unless cpp_wrapper.
        import_str = f"""
            import triton
            import triton.language as tl
            from {triton_heuristics.__name__} import start_graph, end_graph
            """
        if config.triton.autotune_at_compile_time:
            self.kernel_autotune_calls.splice(import_str)
            self.kernel_autotune_calls.writeline(
                V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
            )
        if not V.graph.cpp_wrapper:
            self.imports.splice(import_str, strip=True)
            self.imports.writeline(
                V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
            )

    def write_get_raw_stream_header(self) -> None:
        # [FUNCTION DOC] Adds the device import of the raw stream getter to the autotune calls and imports buffers (the latter not for cpp_wrapper) if not already present.
        import_get_raw_stream_str = V.graph.device_ops.import_get_raw_stream_as(
            "get_raw_stream"
        )
        if config.triton.autotune_at_compile_time:
            if not self.kernel_autotune_calls.contains(import_get_raw_stream_str):
                self.kernel_autotune_calls.writeline(import_get_raw_stream_str)
        if not V.graph.cpp_wrapper:
            if not self.imports.contains(import_get_raw_stream_str):
                self.imports.writeline(import_get_raw_stream_str)

    @cache_on_self
    def write_get_raw_stream_header_once(self) -> None:
        # [FUNCTION DOC] Cached once-only wrapper around write_get_raw_stream_header.
        self.write_get_raw_stream_header()

    def add_meta_once(self, meta: TritonMetaParams) -> str:
        # [FUNCTION DOC] Registers a triton meta dict once: assigns it a variable name meta<N>, writes its definition to the header (and the autotune calls), and returns the variable name.
        # pyrefly: ignore [bad-assignment]
        meta = repr(meta)
        if meta not in self._metas:
            var = f"meta{len(self._metas)}"
            # pyrefly: ignore [unsupported-operation]
            self._metas[meta] = var
            self.header.writeline(f"{var} = {meta}")
            if config.triton.autotune_at_compile_time:
                self.kernel_autotune_calls.writeline(f"{var} = {meta}")
                self._meta_vars.add(var)
        # pyrefly: ignore [bad-index, index-error]
        return self._metas[meta]

    @cache_on_self
    def get_output_refs(self) -> list[str]:
        # [FUNCTION DOC] Returns (cached) the list of reference expressions for each graph output, generated against wrapper_call.
        return [
            x.codegen_reference(self.wrapper_call) for x in self.get_graph_outputs()
        ]

    def mark_output_type(self) -> None:
        # [FUNCTION DOC] Hook to mark output type; does nothing here.
        return

    def get_graph_inputs(
        self,
    ) -> dict[str, ir.TensorBox | ir.TorchBindObject | sympy.Expr]:
        # [FUNCTION DOC] Returns the mapping of graph input names to inputs from the current graph.
        return V.graph.graph_inputs

    def get_graph_outputs(self) -> list[IRNode]:
        # [FUNCTION DOC] Returns the list of graph output nodes from the current graph.
        return V.graph.graph_outputs

    def codegen_input_size_asserts(self) -> None:
        # [FUNCTION DOC] Records pending size/stride assertions for graph inputs: skips symbolic values, torch bind, generator and opaque objects, inputs not in the graph input names, and zero-sized tensors; stores (size, stride) code strings per name in the pending input asserts mapping.
        for name, buf in self.get_graph_inputs().items():
            if isinstance(
                buf,
                (
                    sympy.Basic,
                    ir.TorchBindObject,
                    ir.GeneratorState,
                    ir.OpaqueObjectState,
                ),
            ):
                continue

            # a graph partition may take an IRNode output from a previous partition
            if name not in V.graph.graph_input_names:
                continue

            # comparing strides for 0 size tensor is tricky. Ignore them for now.
            if sympy_product(buf.get_size()) == 0:
                continue
            size = self.codegen_shape_tuple(buf.get_size())
            stride = self.codegen_shape_tuple(buf.get_stride())
            self._pending_input_asserts[name] = (size, stride)

    def codegen_input_nan_asserts(self) -> None:
        # [FUNCTION DOC] Writes a comment and asserts into the prefix that every non-symbolic, non-torch-bind graph input contains no nan or inf.
        self.prefix.writeline("# make sure graph inputs are not nan/inf")
        for name, buf in self.get_graph_inputs().items():
            if isinstance(buf, (sympy.Basic, ir.TorchBindObject)):
                continue
            line = f"assert not {name}.isnan().any().item()"
            self.prefix.writeline(line)
            line = f"assert not {name}.isinf().any().item()"
            self.prefix.writeline(line)

    def write_async_compile_wait(self) -> None:
        # [FUNCTION DOC] Writes a prefix block that waits for async compilation (with globals) and then deletes the async_compile name.
        self.prefix.splice(
            """

            async_compile.wait(globals())
            del async_compile
            """
        )

    def write_args(self, input_names: list[str]):
        # [FUNCTION DOC] Writes the prefix statement that unpacks the call args into the input names (trailing comma for one name) and then clears the args list.
        lhs = ", ".join(input_names)
        if len(input_names) == 1:
            lhs += ","
        self.prefix.writeline(f"{lhs} = args")
        self.prefix.writeline("args.clear()")

    def write_launcher_fn_call_get_indent(self) -> int:
        # [FUNCTION DOC] Writes the definition of the launcher into the prefix: with graph partition a Runner class (partitions, recursive application of functions, and a call method), otherwise a plain function named by the launcher name. Returns the number of indentation levels the body needs (2 for the class, 1 for a function).
        if config.graph_partition:
            self.prefix.splice(
                """
                class Runner:
                    def __init__(self, partitions):
                        self.partitions = partitions

                    def recursively_apply_fns(self, fns):
                        new_callables = []
                        for fn, c in zip(fns, self.partitions):
                            new_callables.append(fn(c))
                        self.partitions = new_callables

                    def call(self, args):
                """
            )
            prefix_indent = 2
        else:
            self.prefix.splice(
                f"""
                def {self.launcher_fn_name}(args):
                """
            )
            prefix_indent = 1

        return prefix_indent

    def get_graph_input_names(self) -> list[str]:
        # [FUNCTION DOC] Returns the graph input names from the current graph.
        return V.graph.graph_input_names

    def write_prefix(self) -> None:
        # [FUNCTION DOC] Writes the beginning of the generated module into the prefix: async compile wait, launcher definition, optional sync-debug, optional training phase annotation, unpacking of args, input codegen, and size/nan asserts (skipped for the top-level call when cudagraph partition is in use). Raises an assertion error if the launcher name is unset.
        if self.launcher_fn_name is None:
            raise AssertionError("expected launcher_fn_name to be set")
        self.write_async_compile_wait()
        prefix_indent = self.write_launcher_fn_call_get_indent()

        with self.prefix.indent(prefix_indent):
            if config.triton.debug_sync_graph:
                self.generate_debug_sync(self.prefix)
            phase = V.graph.get_training_phase()
            if config.annotate_training:
                self.prefix.writeline(
                    f"training_annotation = nvtx._device_range_start('{phase}')"
                )

            if graph_input_names := self.get_graph_input_names():
                self.write_args(graph_input_names)

            self.codegen_inputs()

            # avoid duplicating asserts for both partition functions and
            # the call function when using cudagraph partition
            if not (
                is_using_cudagraph_partition()
                and (not is_codegen_graph_partition_subgraph(self))
            ):
                self.codegen_input_size_and_nan_asserts()

    def codegen_input_size_and_nan_asserts(self) -> None:
        # [FUNCTION DOC] Emits input size assertions if config.size_asserts and nan assertions if config.nan_asserts.
        if config.size_asserts:
            self.codegen_input_size_asserts()
        if config.nan_asserts:
            self.codegen_input_nan_asserts()

    # Input size/stride assertions are deferred from the top of call() to just
    # before the first kernel that uses each input. This avoids a block of N
    # sequential assert calls (~1 us each) on the critical path before the first
    # GPU kernel launch. Called from the scheduler codegen loop.
    def codegen_deferred_input_asserts(self, input_names: Iterable[str]) -> None:
        # [FUNCTION DOC] Emits deferred size/stride assertions for the given inputs, popping them from the pending set: a single assertion for one input, a grouped assertion for several, nothing if none are pending. Called from the scheduler codegen loop just before the first user kernel.
        grouped_asserts: list[tuple[str, str, str]] = []
        for name in input_names:
            if name in self._pending_input_asserts:
                size, stride = self._pending_input_asserts.pop(name)
                grouped_asserts.append((name, size, stride))
        if len(grouped_asserts) == 1:
            name, size, stride = grouped_asserts[0]
            self.write_assert_size_stride(name, size, stride, "input")
        elif len(grouped_asserts) > 1:
            self.write_assert_size_stride_grouped(grouped_asserts, "input")

    def write_assert_size_stride(
        self,
        name: str,
        size: str,
        stride: str,
        op_name: str,
        dtype: torch.dtype | None = None,
    ) -> None:
        # [FUNCTION DOC] Queues a size/stride assertion line for emission during replay.
        """Queue an assert_size_stride for emission during replay."""
        self.writeline(AssertSizeStrideLine(self, name, size, stride, op_name, dtype))

    def _codegen_assert_size_stride(
        self,
        code: IndentedBuffer,
        name: str,
        size: str,
        stride: str,
        op_name: str,
        dtype: torch.dtype | None = None,
    ) -> None:
        # [FUNCTION DOC] Emits the size/stride assertion for the given tensor into code: a plain assert_size_stride call without dtype, or (adding an import once) an assert_tensor_metadata call with dtype. Subclasses override to change the form.
        """Emit one assert_size_stride line to `code` (replay-phase target).

        Subclasses override to change the emitted form (e.g., C++ assert with
        an AOTI runtime env guard).
        """
        if dtype is None:
            code.writeline(f"assert_size_stride({name}, {size}, {stride}, {op_name!r})")
        else:
            self.add_import_once(
                "from torch._inductor.runtime.runtime_utils import assert_tensor_metadata"
            )
            code.writeline(
                f"assert_tensor_metadata({name}, {size}, {stride}, {dtype}, {op_name!r})"
            )

    def write_assert_div_by_zero(self, divisor_str: str, op_name: str) -> None:
        # [FUNCTION DOC] Queues a divide-by-zero check line for emission during replay.
        """Queue a div-by-zero AOTI check for emission during replay.

        codegen_cpp_sizevar is called in both build and replay phases;
        set_writeline routes the WrapperLine appropriately in either case.
        """
        self.writeline(AssertDivByZeroLine(self, divisor_str, op_name))

    def _codegen_assert_div_by_zero(
        self, code: IndentedBuffer, divisor_str: str, op_name: str
    ) -> None:
        # [FUNCTION DOC] Always raises NotImplementedError: only C++ wrappers emit AOTI div-by-zero checks.
        """Emit one div-by-zero AOTI check to `code` (replay-phase target).

        Only emitted by C++ wrappers; the Python wrapper relies on Python's
        native ZeroDivisionError instead of pre-checking divisors.
        """
        raise NotImplementedError(
            "AOTI div-by-zero check is only emitted by C++ wrappers"
        )

    def write_assert_size_stride_grouped(
        self, asserts: list[tuple[str, str, str]], op_name: str
    ) -> None:
        # [FUNCTION DOC] Queues a grouped size/stride assertion line for emission during replay.
        """Queue a grouped assert_size_stride for emission during replay."""
        self.writeline(GroupedAssertSizeStrideLine(self, asserts, op_name))

    def _codegen_assert_size_stride_grouped(
        self, code: IndentedBuffer, asserts: list[tuple[str, str, str]], op_name: str
    ) -> None:
        # [FUNCTION DOC] Writes one grouped size/stride assertion call listing all names, sizes and strides.
        names = ", ".join(name for name, _, _ in asserts)
        sizes = ", ".join(size for _, size, _ in asserts)
        strides = ", ".join(stride for _, _, stride in asserts)
        code.writeline(
            f"assert_size_stride_grouped(({names}), ({sizes}), ({strides}), {op_name!r})"
        )

    def register_alignment_check_inputs(self) -> None:
        # [FUNCTION DOC] Fills the pending alignment-copy set with the names of inputs that need an alignment check and are not mutated; does nothing for cpp_wrapper or when no inputs need checking. If any are pending, marks the graph as deferring input alignment and adds the copy_if_misaligned import.
        """Populate pending alignment copies for non-mutated inputs.
        Called from the scheduler after mutated_input_idxs is computed."""
        if V.graph.cpp_wrapper:
            return
        inputs_to_check = V.graph.inputs_to_check
        if not inputs_to_check:
            return
        # Mutated inputs are handled separately by the runtime wrapper,
        # which needs to copy back the mutation after the call.
        mutated_idxs = OrderedSet(V.graph.mutated_input_idxs)
        for idx in inputs_to_check:
            if idx not in mutated_idxs:
                name = V.graph.graph_input_names[idx]
                self._pending_alignment_copies.add(name)
        if self._pending_alignment_copies:
            V.graph._defers_input_alignment = True
            self.imports.writeline(
                "from torch._C._dynamo.guards import copy_if_misaligned"
            )

    def codegen_deferred_alignment_copies(
        self, input_names: Iterable[str], stream: int = 0
    ) -> None:
        # [FUNCTION DOC] Emits alignment copies for the inputs read on the given stream (not for cpp_wrapper). Pending inputs get a copy_if_misaligned reassignment once and are removed from the pending set. Inputs read on multiple streams save the original under a new name once and get one aligned clone per stream, remembering the stream last aligned.
        # KNOWN ISSUE: an input reused on a stream after another stream re-clones needlessly (acknowledged by the TODO in the code).
        """Emit alignment checks/clones for the given stream.

        An input with an alignment constraint needs that input to be
        potentially reallocated. This emits the proper copy_if_misaligned calls
        to perform that realignment.

        For inputs read on a simple stream this is trivial: we emit immediately
        before the first reader. Inputs read on multiple streams get 1 copy per
        stream, each reading from a preserved version of the original input.
        This ensures we do not need to worry about ordering between multiple
        streams.
        """
        if V.graph.cpp_wrapper:
            return
        for name in input_names:
            if name in self._pending_alignment_copies:
                self._pending_alignment_copies.discard(name)
                self.writeline(f"{name} = copy_if_misaligned({name})")
            elif name in self._multistream_alignment_copies:
                # TODO: if the same input is read again on a stream after an
                # intervening different stream (e.g. s1, s2, s1) this re-clones
                # on the second s1 use -- correct, but an extra copy.  Tracking
                # stream joins / per-(name, stream) clone names would let the
                # later same-stream use reuse the earlier clone.
                if self._alignment_aligned_stream.get(name) == stream:
                    continue
                if name not in self._alignment_orig_saved:
                    # Preserve the original (still un-rewritten here) so every
                    # stream clones from it, not from another stream's clone.
                    self._alignment_orig_saved.add(name)
                    self.writeline(f"{name}_orig = {name}")
                self.writeline(f"{name} = copy_if_misaligned({name}_orig)")
                self._alignment_aligned_stream[name] = stream

    def mark_multistream_alignment(self, names: Iterable[str]) -> None:
        # [FUNCTION DOC] Moves the given input names that are pending alignment copy from the single pending set into the multi-stream set so they are aligned per consuming stream.
        """Reclassify inputs read on more than one stream so their alignment
        copy is emitted per consuming stream (see codegen_deferred_alignment_copies)
        rather than once at the first reader by line order -- which would place
        the only copy inside one branch's stream and race the other branches."""
        for name in names:
            if name in self._pending_alignment_copies:
                self._pending_alignment_copies.discard(name)
                self._multistream_alignment_copies.add(name)

    # this function (and below) takes the graph name as input so
    # that stream caching happens per graph instance. this
    # is important for nested subgraph codegening.
    def write_get_raw_stream(self, device_idx: int, graph_name: str) -> str:
        # [FUNCTION DOC] Emits a raw-stream getter for the device and returns its variable name. Writes the header import, and under autotune at compile time also writes an assignment of the raw stream (runtime current device under compile-on-one-rank). Graph name is a cache key so caching is per graph instance; the method is wrapped with lru_cache during construction.
        self.write_get_raw_stream_header()
        name = get_raw_stream_name(device_idx)
        if config.triton.autotune_at_compile_time:
            # compile-on-one-rank: resolve at runtime so the autotune block matches the
            # rank-agnostic call() body (see codegen_device_guard_enter).
            autotune_idx = (
                V.graph.device_ops.current_device_idx_expr()
                if _coor_enabled()
                else device_idx
            )
            self.kernel_autotune_calls.writeline(
                f"{name} = get_raw_stream({autotune_idx})"
            )
            if V.graph.cpp_wrapper:
                # For cpp wrapper, no need to continue codegen for the main body
                return name
        self.writeline(f"{name} = get_raw_stream({_coor_device_idx_ref(device_idx)})")
        return name

    def get_codegened_graph(self):
        # [FUNCTION DOC] Returns the graph currently being codegened (top of the codegened graph stack); fails with an index error if the stack is empty.
        return self.codegened_graph_stack[-1]

    def push_codegened_graph(self, graph):
        # [FUNCTION DOC] Pushes a graph onto the codegened graph stack (used to key caches per graph while lowering nested subgraphs).
        self.codegened_graph_stack.append(graph)

    def pop_codegened_graph(self):
        # [FUNCTION DOC] Pops and returns the top graph of the codegened graph stack.
        return self.codegened_graph_stack.pop()

    def push_computed_sizes(self, computed_sizes):
        # [FUNCTION DOC] Pushes a deep copy of the given computed-sizes set onto the computed sizes stack.
        # KNOWN ISSUE: it returns the result of the append, which is always None.
        from copy import deepcopy

        return self.computed_sizes_stack.append(deepcopy(computed_sizes))

    def pop_computed_sizes(self):
        # [FUNCTION DOC] Pops and returns the most recently saved computed-sizes set.
        return self.computed_sizes_stack.pop()

    def next_kernel_suffix(self) -> str:
        # [FUNCTION DOC] Returns the next unique integer from the wrapper counter, as a string, for naming kernels.
        return f"{next(self._names_iter)}"

    def codegen_device_guard_enter(
        self,
        device_idx: int,
        num_streams: int = 1,
        stream_idx_to_user_obj_idx: dict[int, int] | None = None,
    ) -> None:
        # [FUNCTION DOC] Emits entry into a device context. With more than one stream requires a mapping from stream index to user object index (assertion error otherwise), adds the stream-by-index import for the python wrapper and queues a stream-aware enter line (setting up the default stream cache only when the device changed); otherwise queues a plain enter line. Under autotune at compile time also writes a matching device guard with-block and raw stream assignment into the autotune calls (indenting it). Records the device index and number of streams.
        if num_streams > 1:
            if stream_idx_to_user_obj_idx is None:
                raise AssertionError("expected stream_idx_to_user_obj_idx to be set")
            if not V.graph.cpp_wrapper:
                import_line = (
                    "from torch._dynamo.variables.streams import _get_stream_by_index"
                )
                if not self.imports.contains(import_line):
                    self.imports.writeline(import_line)
            setup_stream_cache = self._last_default_stream_device != device_idx
            self._last_default_stream_device = device_idx
            self.writeline(
                EnterDeviceContextManagerWithStreamInfoLine(
                    device_idx,
                    self.last_seen_device_guard_index,
                    num_streams,
                    stream_idx_to_user_obj_idx,
                    setup_stream_cache=setup_stream_cache,
                ),
            )
        else:
            self.writeline(
                EnterDeviceContextManagerLine(
                    device_idx, self.last_seen_device_guard_index
                )
            )
        if config.triton.autotune_at_compile_time:
            # mimic logic of EnterDeviceContextManagerLine.codegen for the autotune code block
            self.write_triton_header_once()
            # compile-on-one-rank: resolve the device at runtime (like the call() body) so
            # the autotune block is byte-identical across ranks, not baked to this rank's idx.
            autotune_idx = (
                V.graph.device_ops.current_device_idx_expr()
                if _coor_enabled()
                else device_idx
            )
            self.kernel_autotune_calls.writeline(
                f"with {V.graph.device_ops.device_guard(autotune_idx)}:"
            )
            self.kernel_autotune_calls.do_indent()
            if is_codegen_graph_partition_subgraph(self):
                # Need get_raw_stream for subgraph
                self.write_get_raw_stream_header()
            self.kernel_autotune_calls.writeline(
                f"{get_raw_stream_name(device_idx)} = get_raw_stream({autotune_idx})"
            )
        self.last_seen_device_guard_index = device_idx
        self._num_streams: int = num_streams

    def codegen_device_guard_exit(self) -> None:
        # [FUNCTION DOC] Queues the exit line matching the last device guard entry (stream-aware if more than one stream) and unindents the autotune calls buffer when autotuning at compile time.
        if hasattr(self, "_num_streams") and self._num_streams > 1:
            self.writeline(
                ExitDeviceContextManagerWithStreamInfoLine(self._num_streams)
            )
        else:
            self.writeline(ExitDeviceContextManagerLine())
        if config.triton.autotune_at_compile_time:
            self.kernel_autotune_calls.do_unindent()

    def codegen_cuda_stream_enter(
        self,
        stream_idx: int,
    ) -> EnterCudaStreamContextLine:
        # [FUNCTION DOC] Queues entry into a stream context and returns the line. Raises ValueError if the scheduler reports a stream context is already active (nested contexts unsupported).
        """Generate data structure for entering a CUDA Stream context.

        Args:
            stream_idx: The index number of the entering CUDA Stream context.

        Raises:
            ValueError: If this function is called while the previous stream context isn't exited.
        """
        # pyre-fixme[16]: scheduler.current_stream_name added in scheduler commit
        if (current_stream_name := V.graph.scheduler.current_stream_name) is not None:
            raise ValueError(
                f"Nested stream context switching: {current_stream_name} -> "
                f"{get_stream_name(stream_idx)}",
            )
        ctx_entrance = EnterCudaStreamContextLine(stream_idx=stream_idx)
        self.writeline(ctx_entrance)
        return ctx_entrance

    def codegen_cuda_stream_exit(self) -> None:
        # [FUNCTION DOC] Queues a line exiting the current stream context.
        """Generate data structure for exiting a CUDA Stream context."""
        self.writeline(ExitCudaStreamContextLine())

    def codegen_enter_cuda_stream_context(
        self, code: IndentedBuffer, stream_idx: int
    ) -> None:
        # [FUNCTION DOC] Writes a with-statement for the stream variable of the given index and indents.
        code.writeline(f"with {get_stream_name(stream_idx)}:")
        code.do_indent()

    def codegen_exit_cuda_stream_context(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Unindents the code buffer to leave the stream context.
        code.do_unindent()

    def codegen_stream_info_prologue(
        self,
        code: IndentedBuffer,
        num_streams: int,
        stream_idx_to_user_obj_idx: dict[int, int],
    ) -> None:
        # [FUNCTION DOC] Hook for emitting stream retrieval at device context entry; the base class raises NotImplementedError (cpp wrappers override).
        raise NotImplementedError

    def codegen_cuda_mempool_enter(self, mempool: tuple[int, int]) -> None:
        # [FUNCTION DOC] Queues entry into a CUDA mem pool context given a (pool index, device index) pair, adding the external-object import once. Raises an assertion error under cpp_wrapper.
        """Generate a CUDA MemPool context around code that may allocate."""
        if V.graph.cpp_wrapper:
            raise AssertionError("CUDA MemPool contexts require Python wrapper")
        import_line = (
            "from torch._dynamo.graph_bytecode_inputs import "
            "get_external_object_by_index"
        )
        if not self.imports.contains(import_line):
            self.imports.writeline(import_line)
        mempool_index, device_index = mempool
        self.writeline(EnterCudaMemPoolContextLine(mempool_index, device_index))

    def codegen_cuda_mempool_exit(self) -> None:
        # [FUNCTION DOC] Queues exit of the current CUDA mem pool context; raises an assertion error under cpp_wrapper.
        """Generate data structure for exiting the current CUDA MemPool context."""
        if V.graph.cpp_wrapper:
            raise AssertionError("CUDA MemPool contexts require Python wrapper")
        self.writeline(ExitCudaMemPoolContextLine())

    def generate_return(self, output_refs: list[str]) -> None:
        # [FUNCTION DOC] Writes the return statement of the wrapper call. With outputs returns them as a tuple (preceded, under nan_asserts, by a loop asserting every tensor output has no nan or inf); with none returns an empty tuple.
        if output_refs:
            if config.nan_asserts:
                self.wrapper_call.writeline(
                    "return_vars = (" + ", ".join(output_refs) + ", )"
                )
                self.wrapper_call.writeline("for var in return_vars:")
                self.wrapper_call.do_indent()
                self.wrapper_call.writeline("if isinstance(var, torch.Tensor):")
                self.wrapper_call.do_indent()
                self.wrapper_call.writeline("assert not var.isnan().any().item()")
                self.wrapper_call.writeline("assert not var.isinf().any().item()")
                self.wrapper_call.do_unindent(2)

            self.wrapper_call.writeline("return (" + ", ".join(output_refs) + ", )")
        else:
            self.wrapper_call.writeline("return ()")

    def generate_before_suffix(self, result: IndentedBuffer) -> None:
        # [FUNCTION DOC] Hook run before the suffix is appended; does nothing here.
        return

    def generate_after_suffix(self, result: IndentedBuffer) -> None:
        # [FUNCTION DOC] With graph partition enabled appends code creating a Runner over all partition names and binding the call and recursive-apply names to its methods; otherwise nothing.
        if config.graph_partition:
            all_partition_name_list = ", ".join(self.all_partition_names) + (
                "," if len(self.all_partition_names) == 1 else ""
            )

            result.splice(
                f"""
                runner = Runner(partitions=[{all_partition_name_list}])
                call = runner.call
                recursively_apply_fns = runner.recursively_apply_fns
                """
            )

    def generate_end(self, result: IndentedBuffer) -> None:
        # [FUNCTION DOC] Hook run at the end of generation; does nothing here.
        return

    def generate_fallback_kernel(self, node: ir.FallbackKernel) -> None:
        # [FUNCTION DOC] Generates a fallback kernel call: uses a registered custom extern kernel codegen for the python kernel name if one exists, otherwise queues an extern-kernel-alloc line.
        # Check if this op has a custom codegen implementation
        op_name = node.python_kernel_name
        if op_name is not None and op_name in CUSTOM_EXTERN_KERNEL_CODEGEN:
            custom_codegen = CUSTOM_EXTERN_KERNEL_CODEGEN[op_name].python
            if custom_codegen is not None:
                custom_codegen(node, self.writeline)
                return
        self.writeline(ExternKernelAllocLine(self, node))

    def generate_extern_kernel_multi_out(self, node: ir.ExternKernelMultiOut) -> None:
        # [FUNCTION DOC] Allocates the pre-allocated output buffers, queues the multi-output out-variant call line, and emits size asserts for outputs with a Layout.
        """Generate .out() call with pre-allocated output buffers."""
        for out_node in node.out_variant_output_nodes:
            self.codegen_allocation(out_node)
        self.writeline(ExternKernelMultiOutLine(self, node))
        for out_node in node.out_variant_output_nodes:
            if isinstance(out_node.layout, ir.Layout):
                out_node.codegen_size_asserts(self)

    def generate_extern_kernel_alloc(self, node: ir.ExternKernelAlloc):
        # [FUNCTION DOC] Emits the node comment, queues an extern kernel alloc line, and adds size asserts when the node has a concrete Layout.
        node.codegen_comment(self)
        self.writeline(ExternKernelAllocLine(self, node))
        if isinstance(node.layout, ir.Layout):
            node.codegen_size_asserts(self)

    def _generate_extern_kernel_alloc_helper(self, extern_kernel, args):
        # [FUNCTION DOC] HELPER: Writes the extern kernel call. Nodes with no-return layout produce a bare call; others assign to the output name. Adds a clone to the result for view_as_complex fallbacks when memory planning is on, optionally routes the call through a profiler wrapper, and, if enabled, writes an intermediate hook call and bumps a counter.
        # If it's a NoneLayout then the extern_kernel should essentially be
        # treated as if it doesn't return anything
        no_return = isinstance(extern_kernel.layout, ir.NoneLayout)
        output_name = extern_kernel.get_name()
        origin_node = extern_kernel.get_origin_node()
        kernel_name = extern_kernel.get_kernel_name()
        ending = self.ending
        if config.memory_planning and "view_as_complex" in kernel_name:
            # view operation fallbacks cause issues since inductor
            # doesn't know the memory is still needed and might reuse it.
            ending = f".clone(){ending}"

        if config.trace.provenance_tracking_to_timeline:
            wrapper_name = self.define_extern_kernel_profile_wrapper(
                kernel_name, self.next_kernel_suffix()
            )
        else:
            wrapper_name = kernel_name

        if no_return:
            self.writeline(f"{self.declare}{wrapper_name}({', '.join(args)}){ending}")
        else:
            self.writeline(
                f"{self.declare}{output_name} = {wrapper_name}({', '.join(args)}){ending}"
            )
            if (
                self.supports_intermediate_hooks
                and config.generate_intermediate_hooks
                and origin_node is not None
            ):
                counters["inductor"]["intermediate_hooks"] += 1
                self.writeline(
                    f"run_intermediate_hooks({origin_node.name!r}, {output_name})"
                )

    def generate_extern_kernel_out(
        self,
        node: ir.ExternKernelOut,
    ) -> None:
        # [FUNCTION DOC] Emits the node comment and queues an extern kernel out line.
        node.codegen_comment(self)
        self.writeline(ExternKernelOutLine(self, node))

    def _generate_extern_kernel_out_helper(
        self,
        kernel: str,
        out: str,
        out_view: str | None,
        args: list[str],
        device: str,
        stack_traces: OrderedSet[str] | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Writes an extern out-kernel call: sets debug printer args, appends the out keyword argument (the view if present, else the buffer), optionally uses a profiler wrapper and writes the call inside the debug printer context.
        # add debug printer code for triton kernel calls at (jit) inductor level
        debug_printer_manager = V.graph.wrapper_code.debug_printer
        debug_printer_manager.set_printer_args(args, kernel, None, None, "extern")
        args.append(f"out={out_view or out}")
        with debug_printer_manager:
            if config.trace.provenance_tracking_to_timeline:
                wrapper_name = self.define_extern_kernel_profile_wrapper(
                    kernel, self.next_kernel_suffix()
                )
            else:
                wrapper_name = kernel
            self.writeline(f"{wrapper_name}({', '.join(args)})")

    def _generate_tma_descriptor_call_experimental(self, desc, apply_size_hints=False):
        # [FUNCTION DOC] HELPER: Returns the code string creating an experimental TMA descriptor (tools.experimental_descriptor create-Nd function) from the tensor data pointer, dims, block dims and element size, optionally replacing sizes by hints.
        dims = desc.dims
        block_dims = desc.block_dims
        if apply_size_hints:
            dims = V.graph.sizevars.optimization_hint(dims)
            block_dims = V.graph.sizevars.optimization_hints(block_dims)

        ptr = f"{desc.tensor.codegen_reference()}.data_ptr()"
        # Explicitly call the Python version of val_to_arg_str
        dims = ", ".join(PythonWrapperCodegen.val_to_arg_str(self, dim) for dim in dims)
        block_dims = ", ".join(
            PythonWrapperCodegen.val_to_arg_str(self, dim) for dim in block_dims
        )
        element_size = PythonWrapperCodegen.val_to_arg_str(self, desc.element_size)
        prefix = "triton.tools.experimental_descriptor"
        fn = f"{prefix}.create_{desc.rank}d_tma_descriptor"
        args = f"{ptr}, {dims}, {block_dims}, {element_size}"
        call = f"{fn}({args})"
        return call

    def _generate_tma_descriptor_call_stable(self, desc, apply_size_hints=False):
        # [FUNCTION DOC] HELPER: Returns the code string creating a stable TensorDescriptor from tensor and block shape (hinted if requested).
        block_shape = desc.block_shape
        if apply_size_hints:
            block_shape = V.graph.sizevars.optimization_hints(block_shape)

        prefix = "triton.tools.tensor_descriptor.TensorDescriptor"
        fn = f"{prefix}.from_tensor"
        args = f"{desc.tensor.codegen_reference()}, {block_shape}"
        call = f"{fn}({args})"
        return call

    def _generate_tma_descriptor_call(self, desc, apply_size_hints=False):
        # [FUNCTION DOC] HELPER: Dispatches to the experimental or stable TMA descriptor code generator by descriptor type; raises an assertion error for other types.
        if isinstance(desc, ir.TMADescriptorExperimental):
            return self._generate_tma_descriptor_call_experimental(
                desc, apply_size_hints
            )
        else:
            if not isinstance(desc, ir.TMADescriptorStable):
                raise AssertionError(f"expected TMADescriptorStable, got {type(desc)}")
            return self._generate_tma_descriptor_call_stable(desc, apply_size_hints)

    def generate_tma_descriptor(self, desc):
        # [FUNCTION DOC] Writes an assignment of the TMA descriptor code to the descriptor name.
        call = self._generate_tma_descriptor_call(desc)
        line = f"{desc.name} = {call}{self.ending}"
        self.writeline(line)

    def generate_scatter_fallback(self, node: ir.ScatterFallback):
        # [FUNCTION DOC] Queues a scatter fallback line.
        self.writeline(ScatterFallbackLine(self, node))

    def _generate_scatter_fallback(
        self,
        output,
        inputs,
        cpp_kernel_name,
        python_kernel_name,
        src_is_tensor,
        reduce,
        kwargs,
        device,
    ):
        # [FUNCTION DOC] HELPER: Writes the scatter kernel call using the python kernel name (optionally via profiler wrapper) with inputs, adding kwargs for scatter_reduce or a reduce keyword when given.
        # KNOWN ISSUE: cpp_kernel_name, src_is_tensor and device are unused by the python version.
        orig_python_kernel_name = python_kernel_name
        if config.trace.provenance_tracking_to_timeline:
            python_kernel_name = self.define_extern_kernel_profile_wrapper(
                python_kernel_name, self.next_kernel_suffix()
            )
        line = f"{python_kernel_name}({','.join(map(str, inputs))}"
        if orig_python_kernel_name.startswith("aten.scatter_reduce"):
            line += ", ".join([""] + kwargs)
        elif reduce:
            line += f", reduce={repr(reduce)}"
        line += ")"
        self.writeline(line)

    def define_extern_kernel_profile_wrapper(self, kernel_name: str, suffix: str):
        # [FUNCTION DOC] Emits into the header a wrapper function that forwards to the extern kernel, under a sanitized name with the suffix, registers the alias with provenance tracking, and returns the new name.
        """Wrap extern calls so profiler events use names with provenance metadata."""
        wrapper_kernel_name = re.sub(
            r"\W",
            "_",
            f"extern_kernels_{kernel_name}_{suffix}",
        )
        inductor_debug.alias_kernel_provenance(kernel_name, wrapper_kernel_name)
        self.header.splice(f"def {wrapper_kernel_name}(*args, **kwargs):")
        with self.header.indent():
            self.header.splice(
                "# Extra indirection is only enabled for profiler timeline provenance."
            )
            self.header.splice(f"return {kernel_name}(*args, **kwargs)")
        return wrapper_kernel_name

    def generate_index_put_fallback(self, node: ir.IndexPutFallback) -> None:
        # [FUNCTION DOC] Builds the list of index nodes aligned with the original indices (None where absent, otherwise the next valid input; non-IRNode raises an assertion error) and queues an index_put fallback line.
        # Collect index tensors into a list.
        indices: list[ir.IRNode | None] = []
        valid_indices = node.inputs[2:]
        iter_valid_indices = iter(valid_indices)
        for i, _ in enumerate(node.indices):
            if node.indices[i] is not None:
                index = next(iter_valid_indices)
                if not isinstance(index, ir.IRNode):
                    raise AssertionError(f"expected IRNode, got {type(index)}")
                indices.append(index)
            else:
                indices.append(None)

        self.writeline(IndexPutFallbackLine(self, node, indices))

    def _generate_index_put_fallback(self, kernel, x, indices, values, accumulate):
        # [FUNCTION DOC] HELPER: Writes the index_put kernel call with self, the list of index names, values and the accumulate flag.
        indices_str = f"[{', '.join(indices)}]"
        args = [x, indices_str, values, accumulate]
        self.writeline(self.wrap_kernel_call(kernel, args))

    def generate_fallback_kernel_with_runtime_lookup(
        self,
        buf_name: str,
        python_kernel_name: str,
        get_args: Callable[[], Sequence[str]],
        op_overload: torch._ops.OpOverload | torch._ops.HigherOrderOperator,
        raw_args: Sequence[Any],
        outputs: Sequence[ir.Buffer],
    ) -> None:
        # [FUNCTION DOC] Writes an assignment of the python kernel call with arguments from get_args to the buffer name (other parameters unused here).
        self.writeline(f"{buf_name} = {python_kernel_name}({', '.join(get_args())})")

    def generate(self, is_inference):
        # [FUNCTION DOC] Public entry: runs generation under a timing context and returns the result of _generate (tuple of source with line map and kernel declarations with line map).
        with dynamo_timed("PythonWrapperCodegen.generate"):
            return self._generate(is_inference)

    def get_wrapper_call_indent(self) -> int:
        # [FUNCTION DOC] Returns the indent level of the call body: 2 with graph partition, else 1.
        if config.graph_partition:
            return 2
        else:
            return 1

    @contextlib.contextmanager
    def set_writeline(
        self,
        target: IndentedBuffer,
        writeline: Callable[..., None],
    ) -> Iterator[Callable[..., None]]:
        # [FUNCTION DOC] Context manager that temporarily replaces self.writeline with a dispatcher: WrapperLine objects are code-generated directly into the target buffer while other lines go to the given writeline function. Restores the original on exit and yields the dispatcher.
        """Temporarily redirect self.writeline to write into `target`.

        Plain strings go through `writeline`.
        WrapperLines are codegen'd into `target` directly.
        """
        old = self.writeline

        def dispatch(line: LineContext | DeferredLineBase | WrapperLine | str) -> None:
            # [FUNCTION DOC] HELPER: Dispatcher used by set_writeline: codegens WrapperLines into the target, forwards everything else to the given writeline.
            if isinstance(line, WrapperLine):
                # pyrefly: ignore [missing-attribute]
                line.codegen(target)
            else:
                writeline(line)

        try:
            self.writeline = dispatch  # type: ignore[method-assign]
            yield dispatch
        finally:
            self.writeline = old  # type: ignore[method-assign]

    def _write_multi_kernel_defs(self) -> None:
        # [FUNCTION DOC] Splices the multi-kernel definitions into the autotune defs when autotuning at compile time, else into the header.
        kernel_defs = self.multi_kernel_state.kernel_defs
        if config.triton.autotune_at_compile_time:
            self.kernel_autotune_defs.splice(kernel_defs)
        else:
            self.header.splice(kernel_defs)

    def _generate(self, is_inference):
        # [FUNCTION DOC] Builds the full module source. Runs the wrapper IR passes (memory planning), replays all lines into the wrapper call buffer (while redirecting writeline), writes multi-kernel defs, output refs, optional profiling/debug/autotune blocks, nvtx end and return, then assembles imports, header, subgraph definitions, prefix, indented wrapper call, suffix and benchmark harness. Returns the source and kernel declaration texts, each with a line map. Side effects: mutates buffers and may execute the autotune block.
        if config.profile_bandwidth:
            self.write_triton_header_once()

        with contextlib.ExitStack() as stack:
            stack.enter_context(self.wrapper_call.indent())
            if config.profiler_mark_wrapper_call:
                self.generate_profiler_mark_wrapper_call(stack)
            if config.profile_bandwidth:
                self.generate_start_graph()

            self.run_wrapper_ir_passes(is_inference)

            if config.triton.store_cubin and not config.triton.autotune_at_compile_time:
                self.generate_reset_kernel_saved_flags()

            # At this point, we shouldn't generate any new memory planning lines.
            # Override writeline to point at the wrapper call, in case it gets called.
            with self.set_writeline(self.wrapper_call, self.wrapper_call.writeline):
                for line in self.lines:
                    if isinstance(line, WrapperLine):
                        # pyrefly: ignore [missing-attribute]
                        line.codegen(self.wrapper_call)
                    else:
                        self.wrapper_call.writeline(line)

            self._write_multi_kernel_defs()

            output_refs = self.get_output_refs()
            self.mark_output_type()
            if config.triton.debug_sync_graph:
                self.generate_debug_sync(self.wrapper_call)

            if config.profile_bandwidth:
                self.generate_end_graph()

            if config.triton.proton_profiling:
                self.generate_proton_finalize()

            if config.triton.store_cubin and not config.triton.autotune_at_compile_time:
                self.generate_save_uncompiled_kernels()

            if config.triton.autotune_at_compile_time:
                self.generate_and_run_autotune_block()

            # cpp_wrapper currently doesn't support nvtx
            if config.annotate_training and not config.cpp_wrapper:
                self.wrapper_call.writeline(
                    "nvtx._device_range_end(training_annotation)"
                )
            self.generate_return(output_refs)

        # Assemble the final code from sections.
        result = IndentedBuffer()
        result.splice(self.imports)
        result.writeline("")
        result.splice(self.header)
        # Intermediate const graphs normally use headers rendered by the main
        # module. In dual-wrapper-mode, the const graph's JIT wrapper is emitted as
        # standalone C++ and needs its own header.
        if (
            V.graph.aot_mode
            and V.graph.cpp_wrapper
            and V.graph.is_const_graph
            and not V.graph.is_dual_wrapper_mode
        ):
            result = IndentedBuffer()

        # Add subgraph definitions to the result
        result.splice(self.subgraph_definitions)
        self.finalize_prefix()
        result.splice(self.prefix)

        wrapper_call_indent = self.get_wrapper_call_indent()

        with result.indent(wrapper_call_indent):
            result.splice(self.wrapper_call)

        self.generate_before_suffix(result)
        result.splice(self.suffix)
        self.generate_after_suffix(result)

        self.generate_end(result)

        self.add_benchmark_harness(result)

        return (
            result.getvaluewithlinemap(),
            self.kernel_declarations.getvaluewithlinemap(),
        )

    def generate_and_run_autotune_block(self):
        # [FUNCTION DOC] Composes the autotune defs and calls into one code block, optionally saved to a temp file in the cache dir when debug logging is on, logged as a structured trace artifact, and executed with the autotuning inputs as globals to trigger Triton compilation and tuning. Raises RuntimeError wrapping any exception raised while running it.
        """
        Compose self.kernel_autotune_defs and self.kernel_autotune_calls into a single block of
        code and execute it to trigger Triton kernel compilation and auto-tuning
        """
        self.kernel_autotune_defs.splice(
            """
            async_compile.wait(globals())
            del async_compile
        """
        )
        scope = {}  # type: ignore[var-annotated]
        if config.triton.autotune_at_compile_time and V.graph.autotuning_inputs:
            scope = {
                self.get_autotuning_input_name(idx): v  # type: ignore[attr-defined]
                for idx, v in enumerate(V.graph.autotuning_inputs)
            }
        tuning_code = (
            self.kernel_autotune_defs.getvalue()
            + "\n"
            + self.kernel_autotune_calls.getvalue()
        )
        if output_code_log.level == logging.DEBUG:
            # Save the autotuning code block into a file
            # Create a temporary file
            with tempfile.NamedTemporaryFile(
                dir=cache_dir(), suffix=".py", delete=False
            ) as f:
                f.write(tuning_code.encode("utf-8"))
                file_path = f.name
            output_code_log.debug(
                "Auto-tuning code written to %s",
                file_path,
            )
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "inductor_autotune_at_compile_time_code",
                "encoding": "string",
            },
            payload_fn=lambda: tuning_code,
        )
        # Execute the code to autotune kernels
        try:
            exec(tuning_code, scope)
        except Exception as e:
            raise RuntimeError(f"Failed to run autotuning code block: {e}") from e

    def memory_plan(self):
        # [FUNCTION DOC] Replaces the lines with the output of the MemoryPlanner (the pooled memory planning path).
        from .memory_planning import MemoryPlanner

        self.lines = MemoryPlanner(self).plan(self.lines)

    def memory_plan_reuse(self):
        # [FUNCTION DOC] Reuse-based memory planning: drops trailing planning lines whose buffers are not outputs, then plans each MemoryPlanningLine against a stack of planning states (new state and peak estimate per subgraph, restored on exit), replacing lines with the planned result. Raises an assertion error if states remain unbalanced.
        # KNOWN ISSUE: the total allocated size computed at the end is unused (marked FIXME in code).
        outputs = self.get_graph_outputs()
        out_names = V.graph._get_output_names(outputs)

        while (
            self.lines
            and isinstance(self.lines[-1], MemoryPlanningLine)
            # TODO: this seems legit, NullLine has no node
            and self.lines[-1].node.name not in out_names  # type: ignore[attr-defined]
        ):
            # these lines will be pointless
            self.lines.pop()

        # codegen allocations in two passes
        planning_states = [MemoryPlanningState()]
        past_planning_states = []
        peak_estimate_stack: list[EfficientPeakEstimate] = []
        for i in range(len(self.lines)):
            line = self.lines[i]
            if isinstance(line, MemoryPlanningLine):
                self.lines[i] = line.plan(planning_states[-1])
            elif isinstance(line, EnterSubgraphLine):
                planning_states.append(MemoryPlanningState())
                if config.allow_buffer_reuse:
                    peak_estimate_stack.append(self.estimate_peak)
                    with V.set_graph_handler(line.graph):
                        self.estimate_peak = EfficientPeakEstimate()
            elif isinstance(line, ExitSubgraphLine):
                past_planning_states.append(planning_states.pop())
                if config.allow_buffer_reuse:
                    self.estimate_peak = peak_estimate_stack.pop()
        past_planning_states.append(planning_states.pop())
        if len(planning_states) != 0:
            raise AssertionError(
                f"expected no remaining planning states, got {len(planning_states)}"
            )

        # conservatively use the sum of all allocated buffer sizes
        # in potentially nested scopes as the total allocated size
        # FIXME(rec): not used
        _total_allocated_buffer_size = sum(
            s.total_allocated_buffer_size for s in past_planning_states
        )

    def run_wrapper_ir_passes(self, is_inference: bool):
        # [FUNCTION DOC] Runs planning passes: pooled memory planning when inference and config.memory_planning, otherwise reuse planning (creating the peak estimate first if buffer reuse is on). Planning is skipped for training because it raises peak memory.
        # We disable planning during training because it presently increases peak memory consumption.
        if is_inference and config.memory_planning:
            self.memory_plan()
        else:
            if config.allow_buffer_reuse:
                self.estimate_peak = EfficientPeakEstimate()
            self.memory_plan_reuse()

    def maybe_emit_replacement_aliases(
        self,
        sym: sympy.Symbol,
        bound_vars: OrderedSet[sympy.Symbol],
    ) -> None:
        # [FUNCTION DOC] For a bound symbol, writes alias assignments to the prefix for the other side of backed-symbol replacements recorded in the shape environment (either source replaced by sym or sym replaced by target), skipping unbacked and already bound symbols, and records them as bound. No-op when there is no shape environment.
        # Deferred runtime asserts and graph input metadata can reference
        # either side of a backed-symbol replacement. Emit aliases so both
        # the pre-replacement and canonical names are defined. Skip unbacked
        # symbols - they are defined separately by the unbacked symbol
        # codegen path.
        sizevars = getattr(V.graph, "sizevars", None)
        shape_env = getattr(sizevars, "shape_env", None)
        if shape_env is None:
            return

        def is_backed_symbol(s: sympy.Symbol) -> bool:
            # [FUNCTION DOC] HELPER: True if the symbol is not an unbacked int or float symbol.
            return not symbol_is_type(s, (SymT.UNBACKED_INT, SymT.UNBACKED_FLOAT))

        for src, tgt in shape_env.replacements.items():
            if (
                tgt == sym
                and isinstance(src, sympy.Symbol)
                and src not in bound_vars
                and is_backed_symbol(src)
                and is_backed_symbol(sym)
            ):
                self.prefix.writeline(f"{src} = {sym}")
                bound_vars.add(src)
            elif (
                src == sym
                and isinstance(tgt, sympy.Symbol)
                and tgt not in bound_vars
                and is_backed_symbol(sym)
                and is_backed_symbol(tgt)
            ):
                self.prefix.writeline(f"{tgt} = {sym}")
                bound_vars.add(tgt)

    def codegen_input_symbol_assignment(
        self,
        name: str,
        value: ir.TensorBox | sympy.Expr,
        bound_vars: OrderedSet[sympy.Symbol],
        deferred_symbol_assignments=None,
    ):
        # [FUNCTION DOC] Emits prefix assignments for symbolic graph inputs: a symbolic expression input is simplified and, if a symbol, assigned from the input name (plus the raw symbol and replacement aliases), tracking bound variables. Tensor, torch bind, generator and opaque inputs need nothing; any other type raises an assertion error unless graph partition is on.
        # KNOWN ISSUE: the deferred_symbol_assignments argument is unused.
        if isinstance(value, sympy.Expr):
            raw_value = value
            value = V.graph.sizevars.simplify(raw_value)
            if isinstance(value, sympy.Symbol) and value not in bound_vars:
                self.prefix.writeline(f"{value} = {name}")
                bound_vars.add(value)
                self.maybe_emit_replacement_aliases(value, bound_vars)
            if isinstance(raw_value, sympy.Symbol):
                if raw_value not in bound_vars:
                    self.prefix.writeline(f"{raw_value} = {name}")
                    bound_vars.add(raw_value)
                return
            if not isinstance(value, sympy.Symbol):
                return
        elif isinstance(
            value,
            (
                ir.TensorBox,
                ir.TorchBindObject,
                ir.GeneratorState,
                ir.OpaqueObjectState,
            ),
        ):
            return
        else:
            if torch._inductor.config.graph_partition:
                pass
            else:
                raise AssertionError(f"Unknown value type: {type(value)}")

    def _retry_deferred_symbol_assignments(self, deferred_symbol_assignments) -> None:
        # [FUNCTION DOC] HELPER: Repeatedly calls each deferred assignment callback (passing a fresh list for retries) until the list is empty or a full pass makes no progress.
        while deferred_symbol_assignments:
            next_deferred_symbol_assignments = []
            progress = False
            for assignment in deferred_symbol_assignments:
                progress = assignment(next_deferred_symbol_assignments) or progress
            if not progress:
                break
            deferred_symbol_assignments = next_deferred_symbol_assignments

    def bind_input_symbol(
        self,
        sym: sympy.Symbol,
        input_name: str,
        kind: Literal["size", "stride"],
        dim: int,
        bound_vars: OrderedSet[sympy.Symbol],
    ) -> None:
        # [FUNCTION DOC] Binds a symbol to the size or stride of an input at a dimension by writing an assignment into the prefix, unless already bound; records it and emits replacement aliases.
        if sym in bound_vars:
            return
        if kind == "size":
            self.prefix.writeline(f"{sym} = {input_name}.size()[{dim}]")
        else:
            self.prefix.writeline(f"{sym} = {input_name}.stride()[{dim}]")
        bound_vars.add(sym)
        self.maybe_emit_replacement_aliases(sym, bound_vars)

    def codegen_inputs(self):
        # [FUNCTION DOC] Emits prefix code assigning symbolic shapes from inputs to wrapper-local variables. Symbol inputs are handled before others. Then binds symbols from the symbolic input sources, and, for named tensor inputs under size_asserts, derives symbols from sizes and strides (solving single unknown expressions, retrying multi-unknown ones later). Finally verifies that every symbol used in a tensor size or stride expression has been defined, raising an assertion error otherwise.
        """Assign input symbolic shapes to wrapper-local variables."""
        bound_vars = OrderedSet[sympy.Symbol]()
        # There is a subtle case in the cpp wrapper codegen which requires generating
        # symbol inputs first followed by non-symbol ones.
        #
        # When a dynamic size constraint specified at the Export time is an expression,
        # we need to solve that expression to proper define a symbol in cpp. Thus we
        # are enforcing this iterating order here to make sure all plain size symbols
        # are defined first.
        graph_inputs = self.get_graph_inputs()
        inputs = [
            (k, v) for k, v in graph_inputs.items() if isinstance(v, sympy.Symbol)
        ] + [(k, v) for k, v in graph_inputs.items() if not isinstance(v, sympy.Symbol)]
        deferred_symbol_assignments = []
        for name, value in inputs:
            self.codegen_input_symbol_assignment(
                name, value, bound_vars, deferred_symbol_assignments
            )
        self._retry_deferred_symbol_assignments(deferred_symbol_assignments)

        @functools.cache
        def sizeof(name):
            # [FUNCTION DOC] HELPER: Cached; writes a prefix line saving the size of the named input to a variable and returns the variable name.
            self.prefix.writeline(f"{name}_size = {name}.size()")
            return f"{name}_size"

        @functools.cache
        def strideof(name):
            # [FUNCTION DOC] HELPER: Cached; writes a prefix line saving the stride of the named input to a variable and returns the variable name.
            self.prefix.writeline(f"{name}_stride = {name}.stride()")
            return f"{name}_stride"

        def bind_input_expr_symbols(
            expr: sympy.Symbol | sympy.Expr,
            input_name: str,
            kind: Literal["size", "stride"],
            dim: int,
            deferred_symbol_assignments=None,
        ) -> bool:
            # [FUNCTION DOC] HELPER: Tries to define symbols from a size or stride expression at one dimension. A lone symbol is assigned directly; an expression with exactly one undefined symbol is solved for it (writing the dimension value and the solved assignment); with several unknowns the attempt is queued for retry. Returns True if something was bound.
            name_fn = sizeof if kind == "size" else strideof
            if isinstance(expr, sympy.Symbol):
                if expr in bound_vars:
                    return False
                self.prefix.writeline(f"{expr} = {name_fn(input_name)}[{dim}]")
                bound_vars.add(expr)
                self.maybe_emit_replacement_aliases(expr, bound_vars)
                return True

            if not isinstance(expr, sympy.Expr):
                return False

            undefined_symbols = [
                sym for sym in expr.free_symbols if sym not in bound_vars
            ]
            if len(undefined_symbols) != 1:
                if (
                    len(undefined_symbols) > 1
                    and deferred_symbol_assignments is not None
                ):

                    def retry(deferred_symbol_assignments):
                        # [FUNCTION DOC] HELPER: Deferred retry callback re-running bind_input_expr_symbols with the new deferred list.
                        return bind_input_expr_symbols(
                            expr,
                            input_name,
                            kind,
                            dim,
                            deferred_symbol_assignments,
                        )

                    deferred_symbol_assignments.append(retry)
                return False

            from torch.utils._sympy.solve import try_solve

            free_symbol = undefined_symbols.pop()
            base_name = name_fn(input_name)
            dim_symbol = sympy.Symbol(f"{base_name}_{dim}", integer=True)
            solution = try_solve(sympy.Eq(expr, dim_symbol), free_symbol)
            if solution is None:
                return False

            self.prefix.writeline(f"{dim_symbol} = {base_name}[{dim}]")
            solution_expr = _rewrite_symbol_solution_for_int_codegen(solution[1])
            self.prefix.writeline(f"{free_symbol} = {pexpr(solution_expr)}")
            bound_vars.add(free_symbol)
            self.maybe_emit_replacement_aliases(free_symbol, bound_vars)
            return True

        for sym, (input_name, kind, dim) in V.graph.symbolic_input_sources.items():
            sym = V.graph.sizevars.simplify(sym)
            if not isinstance(sym, sympy.Symbol):
                continue
            if sym in bound_vars:
                continue
            if input_name not in graph_inputs:
                continue
            self.bind_input_symbol(sym, input_name, kind, dim, bound_vars)

        for input_name, value in inputs:
            if not isinstance(value, ir.TensorBox):
                continue
            is_named_graph_input = input_name in V.graph.graph_input_names
            bind_assert_symbols = config.size_asserts and (
                is_named_graph_input and sympy_product(value.get_size()) != 0
            )
            for dim, raw_size in enumerate(value.get_size()):
                if bind_assert_symbols:
                    bind_input_expr_symbols(
                        raw_size,
                        input_name,
                        "size",
                        dim,
                        deferred_symbol_assignments,
                    )
                size = V.graph.sizevars.simplify(raw_size)
                if bind_assert_symbols and size != raw_size:
                    bind_input_expr_symbols(
                        size,
                        input_name,
                        "size",
                        dim,
                        deferred_symbol_assignments,
                    )
            for dim, raw_stride in enumerate(value.get_stride()):
                if bind_assert_symbols:
                    bind_input_expr_symbols(
                        raw_stride,
                        input_name,
                        "stride",
                        dim,
                        deferred_symbol_assignments,
                    )
                stride = V.graph.sizevars.simplify(raw_stride)
                if bind_assert_symbols and stride != raw_stride:
                    bind_input_expr_symbols(
                        stride,
                        input_name,
                        "stride",
                        dim,
                        deferred_symbol_assignments,
                    )
        self._retry_deferred_symbol_assignments(deferred_symbol_assignments)

        def _verify_input_symbol_assignment(
            input_name: str,
            value: ir.TensorBox,
            bound_vars: OrderedSet[sympy.Symbol],
        ):
            # [FUNCTION DOC] HELPER: For a named tensor input under size_asserts with nonzero size, raises an assertion error if any simplified non-symbol size or stride expression has a free symbol that is not yet bound.
            is_named_graph_input = input_name in V.graph.graph_input_names
            verify_assert_symbols = config.size_asserts and (
                is_named_graph_input and sympy_product(value.get_size()) != 0
            )
            for expr in chain.from_iterable([value.get_size(), value.get_stride()]):
                if not isinstance(expr, Expr) or isinstance(expr, sympy.Symbol):
                    continue
                expr = V.graph.sizevars.simplify(expr)
                if not isinstance(expr, Expr):
                    continue

                undefined_symbols = [
                    sym
                    for sym in expr.free_symbols
                    if sym not in bound_vars and verify_assert_symbols
                ]
                if len(undefined_symbols) > 0:
                    raise AssertionError(
                        f"For {expr}, expected {undefined_symbols} to have been codegen-ed."
                    )

        # For inputs with size/strides which contain sympy expressions, we can
        # encounter symbols that weren't defined yet. Now, let's check each
        # symbol is defined.
        for input_name, value in inputs:
            if not isinstance(value, ir.TensorBox):
                continue
            _verify_input_symbol_assignment(input_name, value, bound_vars)

    def ensure_size_computed(self, sym: sympy.Symbol):
        # [FUNCTION DOC] For a precomputed-size symbol not yet computed, records it and queues a symbolic call arg line defining it from its inverse replacement expression.
        if isinstance(sym, sympy.Symbol) and symbol_is_type(sym, SymT.PRECOMPUTED_SIZE):
            if sym in self.computed_sizes:
                return
            self.computed_sizes.add(sym)
            expr = V.graph.sizevars.inv_precomputed_replacements[sym]
            arg = SymbolicCallArg(sym, expr)
            self.writeline(SymbolicCallArgLine(self, arg, V.graph))

    def finalize_prefix(self):
        # [FUNCTION DOC] Hook to finalize the prefix; does nothing here.
        pass

    def codegen_cpp_sizevar(self, x: Expr, *, simplify: bool = True) -> str:
        # [FUNCTION DOC] Not available in the python wrapper; always raises RuntimeError (only the cpp wrapper implements it).
        raise RuntimeError("codegen_cpp_sizevar is only implemented for cpp_wrapper!")

    def codegen_python_sizevar(self, x: Expr, *, simplify: bool = True) -> str:
        # [FUNCTION DOC] Returns the python expression string for a size expression via the python printer, optionally simplified.
        return pexpr(x, simplify=simplify)

    def codegen_sizevar(self, x: Expr) -> str:
        # [FUNCTION DOC] Returns the size variable code string (python version).
        return self.codegen_python_sizevar(x)

    def codegen_tuple_access(self, basename: str, name: str, index: str) -> str:
        # [FUNCTION DOC] Returns the subscript expression for element index of basename (name is unused; the cpp wrapper overrides).
        return f"{basename}[{index}]"

    def codegen_python_shape_tuple(self, shape: Sequence[Expr]) -> str:
        # [FUNCTION DOC] Returns the python tuple literal string of the printed shape entries, handling empty and single-element tuples.
        parts = [*map(self.codegen_python_sizevar, shape)]
        if len(parts) == 0:
            return "()"
        if len(parts) == 1:
            return f"({parts[0]}, )"
        return f"({', '.join(parts)})"

    def codegen_shape_tuple(self, shape: Sequence[Expr]) -> str:
        # [FUNCTION DOC] Returns the shape tuple code string (python version).
        return self.codegen_python_shape_tuple(shape)

    def codegen_alloc_from_pool(
        self, name, offset, dtype, shape, stride
    ) -> tuple[str, list[str]]:
        # [FUNCTION DOC] Returns an alloc_from_pool call string with name, byte offset, dtype, shape and stride, together with an empty list of extra lines.
        return "alloc_from_pool({})".format(
            ", ".join(
                [
                    name,
                    pexpr(offset),  # bytes not numel
                    str(dtype),
                    self.codegen_python_shape_tuple(shape),
                    self.codegen_python_shape_tuple(stride),
                ]
            )
        ), []

    def codegen_reinterpret_view(
        self,
        data,
        size,
        stride,
        offset,
        writeline: Callable[..., None],
        dtype=None,
    ) -> str:
        # [FUNCTION DOC] Generates a reinterpret view expression for a buffer; begins by finding the innermost buffer layout through the wrapper chain (see codegen_reinterpret_view_helper) so the view is derived from the base buffer rather than an intermediate layout.
        # Get the innermost buffer's layout info to help reinterpret view.
        # Consider a chain of (ReinterpretView <- TensorBox| StorageBox)... <- buffer
        # If we only use x.data to determine the reinterpret, we may get wrong layout.
        # For example:
        # x = ReinterpretView(
        #       Storage(
        #         ReinterpretView(
        #           storage(
        #             Buffer(name='buf0', layout=(size=(2, 5, 10), ...)
        #           ),
        #           layout=(10, 10),
        #         ),
        #       ),
        #       layout=(10, 10),
        #     )
        # In this case, x.data.layout == x.layout is (10, 10), the reinterpret view will return buf0,
        # but buf0 need to be viewed from (2, 5, 10) to (10, 10).
        # So we need to dig into the chain to find the innermost buffer's layout.
        d_size, d_stride, d_offset, d_dtype, collapsible = (
            codegen_reinterpret_view_helper(data)
        )

        def apply_reinterpret(
            name, tgt_size, tgt_stride, tgt_offset, cast_dtype, base_dtype
        ):
            # [FUNCTION DOC] HELPER: Builds the reinterpret_tensor call string for a name with the target size, stride and offset, wrapped in a dtype view if a different cast dtype is requested.
            s = self.codegen_python_shape_tuple(tgt_size)
            st = self.codegen_python_shape_tuple(tgt_stride)
            off = self.codegen_sizevar(tgt_offset)
            expr = f"reinterpret_tensor({name}, {s}, {st}, {off})"
            if cast_dtype is not None and cast_dtype != base_dtype:
                return f"aten.view.dtype({expr}, {cast_dtype})"
            return expr

        name = data.get_name()
        collapsed = collapsible and offset == d_offset
        if collapsed:
            same_layout = size == d_size and stride == d_stride
            base_dtype = d_dtype
        else:
            same_layout = (
                size == data.layout.size
                and stride == data.layout.stride
                and offset == data.layout.offset
            )
            base_dtype = data.dtype

        if same_layout:
            if dtype is not None and dtype != base_dtype:
                return f"aten.view.dtype({name}, {dtype})"
            return f"{name}"

        return apply_reinterpret(name, size, stride, offset, dtype, base_dtype)

    def codegen_device_copy(self, src, dst, non_blocking: bool | str):
        # [FUNCTION DOC] Writes a copy statement copying src into dst with the given non_blocking argument.
        self.writeline(f"{dst}.copy_({src}, {non_blocking})")

    def sync_d2h_copy(self, buffer_name: str) -> None:
        # [FUNCTION DOC] Writes lines that record an event for the device-to-host copy of the buffer and synchronize on it, using an event variable named after the buffer.
        event_var = f"_d2h_event_{buffer_name}"
        self.writeline(f"{event_var} = torch.Event()")
        self.writeline(f"{event_var}.record()")
        self.writeline(f"{event_var}.synchronize()")

    def codegen_multi_output(self, node: ir.MultiOutput):
        # [FUNCTION DOC] Queues a MultiOutputLine defining the node name from its first input by following the node indices.
        result_name = node.get_name()
        arg_name = node.input_name(0)
        self.writeline(MultiOutputLine(self, result_name, arg_name, node.indices))

    def codegen_dynamic_select_index(self, node, clamp):
        # [FUNCTION DOC] Writes the computation of the unbacked offset symbol for a dynamic select: base offset plus base dim stride times the normalized index (negative indices wrap by the size, optionally clamped to zero..size). Records the symbol as declared so it is not declared again.
        index_str = f"{node.index} + {node.size} if {node.index} < 0 else {node.index}"
        if clamp:
            index_str = f"max(0, min({node.size}, {index_str}))"
        self.writeline(
            f"{node.unbacked_offset_symbol} = {node.base_offset} + {node.base_dim_stride} * ({index_str})"
        )
        # record in unbacked_symbol_decls so we won't generate a declaration of the symbol again
        self.unbacked_symbol_decls.add(str(node.unbacked_offset_symbol))

    def codegen_dynamic_slice_size(self, node):
        # [FUNCTION DOC] Writes start and end variables (clamped, with negative wraparound) and the slice length symbol as the non-negative number of elements considering the step; records the size symbol as declared.
        def clamp_index(x):
            # [FUNCTION DOC] HELPER: Returns the code string clamping a slice index to range, treating negatives as offset from the size.
            pos = self.codegen_sizevar(Max(0, Min(x, node.size)))
            neg = self.codegen_sizevar(Max(0, Min(x + node.size, node.size)))
            x_cond = self.codegen_sizevar(x)
            return f"{pos} if {x_cond} >= 0 else {neg}"

        def codegen_with_step(start_var, end_var, step):
            # [FUNCTION DOC] HELPER: Returns the code string for the slice length from start and end variables and a step (ceiling division for step other than 1).
            if step == 1:
                return f"{end_var} - {start_var}"
            step_ = self.codegen_sizevar(step)
            return f"({end_var} - {start_var} + {step_} - 1) // {step_}"

        # codegen start, end
        sym = node.unbacked_size_symbol
        start = clamp_index(node.start)
        end = clamp_index(node.end)
        self.writeline(f"{sym}_start = {start}")
        self.writeline(f"{sym}_end = {end}")
        with_step = codegen_with_step(f"{sym}_start", f"{sym}_end", node.step)
        self.writeline(f"{sym} = max(0, {with_step})")
        self.unbacked_symbol_decls.add(str(node.unbacked_size_symbol))

    def codegen_dynamic_scalar(self, node):
        # [FUNCTION DOC] Queues a DynamicScalarLine for the node.
        self.writeline(DynamicScalarLine(self, node))

    def _codegen_dynamic_scalar(self, node):
        # [FUNCTION DOC] HELPER: Emits code reading a scalar from a tensor into the node symbol. Supports an empty keypath (plain item), an int conversion key (1 if truthy else 0) and a divide-by key (item with a divisibility assertion, then integer division); other keypaths raise an assertion error. Also defines the node buffer name as None.
        (data,) = (t.codegen_reference() for t in node.inputs)
        if len(node.keypath) == 0:
            self.writeline(f"{node.sym} = {data}.item()")
        elif len(node.keypath) == 1 and isinstance(node.keypath[0], ConvertIntKey):
            self.writeline(f"{node.sym} = 1 if {data}.item() else 0")
        elif len(node.keypath) == 1 and isinstance(node.keypath[0], DivideByKey):
            self.writeline(f"{node.sym}_undivided = {data}.item()")
            self.writeline(
                f"assert {node.sym}_undivided % {node.keypath[0].divisor} == 0, "
                f"f'{{{node.sym}_undivided}} not divisible by {node.keypath[0].divisor}'"
            )
            self.writeline(
                f"{node.sym} = {node.sym}_undivided // {node.keypath[0].divisor}"
            )
        else:
            raise AssertionError(f"unrecognized keypath {node.keypath}")
        # No one should ever use this buffer, but for uniformity
        # define the variable and assign it None
        self.writeline(f"{node.get_name()} = None")

    def _coor_device_type_str(self, device: torch.device) -> str:
        # [FUNCTION DOC] HELPER: Renders a device for the benchmark harness without rank-specific index. Resolves (once, cached on the wrapper) the compile-on-one-rank current accelerator; when it is off returns the full device string, otherwise validates the device is the current accelerator and returns only its type.
        """Render a device for the benchmark harness rank-agnostically: the bare device type
        (no rank-specific index), so it is byte-identical across ranks. cpu/meta have no index
        and pass through unchanged; a non-current accelerator violates the single-device
        invariant and is refused. Under compile-on-one-rank the current accelerator is read
        once per compilation and cached on the wrapper; once cached, the per-input/buffer
        harness loops repeat neither the enabled check nor the cudaGetDevice.
        """
        cur = self._coor_current_accelerator
        if cur is None:
            cur = _coor_current_accelerator() if _coor_enabled() else False
            self._coor_current_accelerator = cur
        if cur is False:
            return str(device)
        _coor_check_current_accelerator(device, cur)
        return device.type

    def benchmark_compiled_module(self, output):
        # [FUNCTION DOC] Writes the benchmark support code into output: a get_args function creating random strided tensors for constants, torchbind constants (pickled) and graph inputs (symbolic values use hint 42 or False, generator state and opaque inputs special-cased), sharing one storage among aliased example inputs, and a benchmark_compiled_module function timing the call. Reads graph state; writes only to output.
        """Write out codegen for benchmarking the output code"""

        def add_fake_input(name, shape, stride, device, dtype):
            # [FUNCTION DOC] HELPER: Writes a line creating a random strided tensor with the given shape, stride, device and dtype.
            output.writeline(
                f"{name} = rand_strided("
                f"{self.codegen_python_shape_tuple(shape)}, "
                f"{self.codegen_python_shape_tuple(stride)}, "
                f"device='{self._coor_device_type_str(device)}', dtype={dtype})"
            )

        def add_expr_input(name, val):
            # [FUNCTION DOC] HELPER: Writes a line assigning a value expression to a name.
            output.writeline(f"{name} = {val}")

        def add_torchbind_input(name, value):
            # [FUNCTION DOC] HELPER: Writes a line defining a torchbind input as None or by unpickling it; if pickling fails writes a line that raises TypeError in the generated code.
            if value is None:
                output.writeline(f"{name} = None")
                return

            import pickle

            try:
                output.writeline(f"{name} = pickle.loads({pickle.dumps(value)!r})")
            except (TypeError, AttributeError, pickle.PicklingError) as e:
                output.writeline(
                    f'raise TypeError("Failed to pickle opaque type {type(value)} for variable {name}: {str(e)}")'
                )

        # Preserve shared non-empty input storages so benchmark code does not
        # explode large aliased views into separate allocations.
        storage_groups: dict[StorageWeakRef, BenchmarkStorageGroup] = {}
        aliased_input_specs: dict[str, tuple[str, Any, Any]] = {}
        example_inputs = V.graph.example_inputs

        if example_inputs is not None:
            for name, ex in zip(V.graph.graph_inputs.keys(), example_inputs):
                if not isinstance(ex, torch.Tensor):
                    continue
                storage = ex.untyped_storage()
                nbytes = storage.nbytes()
                storage_key = StorageWeakRef(storage)
                group = storage_groups.get(storage_key)
                if group is None:
                    group = BenchmarkStorageGroup(
                        buffer_name=f"_shared_storage_{len(storage_groups)}",
                        nbytes=nbytes,
                        device=ex.device,
                        dtype=ex.dtype,
                        inputs={},
                    )
                    storage_groups[storage_key] = group

                group.inputs[name] = (list(ex.shape), list(ex.stride()))

        # Generate get_args() to create input tensors separately from benchmarking
        output.writelines(["", "", "def get_args():"])
        with output.indent():
            output.splice(
                """
                from torch._dynamo.testing import rand_strided
                """,
                strip=True,
            )

            for name, value in V.graph.constants.items():
                # all the constants are global variables, that's why we need
                # these 'global var_name' lines
                output.writeline(f"global {name}")
                add_fake_input(
                    name, value.size(), value.stride(), value.device, value.dtype
                )

            if len(V.graph.torchbind_constants) > 0:
                output.writeline("import pickle")
                for name, torchbind_obj in V.graph.torchbind_constants.items():
                    # all the constants are global variables, that's why we need
                    # these 'global var_name' lines
                    output.writeline(f"global {name}")
                    add_torchbind_input(name, torchbind_obj)

            # Generate shared storage buffers for aliased inputs
            for group in storage_groups.values():
                if len(group.inputs) < 2:
                    continue
                numel = group.nbytes // torch._utils._element_size(group.dtype)
                output.writeline(
                    f"{group.buffer_name} = rand_strided(({numel},), (1,), device='{self._coor_device_type_str(group.device)}', dtype={group.dtype})"
                )
                for name, (shape, stride) in group.inputs.items():
                    aliased_input_specs[name] = (group.buffer_name, shape, stride)

            for name, value in V.graph.graph_inputs.items():
                if isinstance(value, sympy.Symbol) and isinstance(
                    V.graph.sizevars.backed_var_to_val.get(value, None), SingletonInt
                ):
                    # Inductor should only work with dense -> dense graph, and
                    # SingletonInts belong to metadata that should only live on
                    # the subclass.
                    continue
                if isinstance(value, ir.TorchBindObject):
                    output.writeline(f"{name} = None")
                elif isinstance(value, sympy.Expr):  # Don't need to add symbolic
                    # TODO: this fallback and those below actually will generate possibly
                    # invalid benchmark code, because it's not guaranteed 42
                    # is actually a valid value for the kernel in question.
                    # See https://github.com/pytorch/pytorch/issues/124686
                    add_expr_input(
                        name, V.graph.sizevars.optimization_hint(value, fallback=42)
                    )
                elif isinstance(value, sympy.Basic):
                    # sympy.Boolean (e.g. StrictLessThan from torch.cond predicates)
                    # is not a sympy.Expr so optimization_hint cannot handle it.
                    # Use False as a fallback for benchmark harness purposes.
                    add_expr_input(name, False)
                elif isinstance(value, ir.GeneratorState):
                    # Harness runs outside the runtime device guard, so resolve the index
                    # with current_device() (the _coor_device_idx var is not in scope here).
                    gen_idx = (
                        V.graph.device_ops.current_device_idx_expr()
                        if _coor_enabled()
                        else value.device.index
                    )
                    add_expr_input(
                        name,
                        f"torch.cuda.default_generators[{gen_idx}].graphsafe_get_state()",
                    )
                elif isinstance(value, ir.OpaqueObjectState):
                    output.writeline(f"{name} = None")
                elif name in aliased_input_specs:
                    buf_name, shape, stride = aliased_input_specs[name]
                    output.writeline(
                        f"{name} = torch.as_strided({buf_name}, "
                        f"{self.codegen_python_shape_tuple(shape)}, "
                        f"{self.codegen_python_shape_tuple(stride)})"
                    )
                else:
                    shape = V.graph.sizevars.optimization_hints(
                        value.get_size(), fallback=42
                    )
                    stride = V.graph.sizevars.optimization_hints(
                        value.get_stride(), fallback=42
                    )

                    add_fake_input(
                        name,
                        shape,
                        stride,
                        value.get_device(),
                        value.get_dtype(),
                    )

            output.writeline(f"return [{', '.join(V.graph.graph_inputs.keys())}]")

        # Generate benchmark_compiled_module() that takes args as parameter
        output.writelines(
            ["", "", "def benchmark_compiled_module(args, times=10, repeat=10):"]
        )
        with output.indent():
            output.splice(
                f"""
                from torch._inductor.utils import print_performance
                fn = lambda: call(list(args))
                return print_performance(fn, times=times, repeat=repeat, device='{V.graph.device_type}')
                """,
                strip=True,
            )

    def add_benchmark_harness(self, output):
        # [FUNCTION DOC] When config.benchmark_harness is on, appends the benchmark module code and a main block that builds args and runs the compiled module benchmark entry point; otherwise does nothing.
        """
        Append a benchmark harness to generated code for debugging
        """
        if not config.benchmark_harness:
            return

        self.benchmark_compiled_module(output)

        output.writelines(["", "", 'if __name__ == "__main__":'])
        with output.indent():
            output.writelines(
                [
                    "from torch._inductor.wrapper_benchmark import compiled_module_main",
                    "args = get_args()",
                    (
                        f"compiled_module_main('{get_benchmark_name()}', "
                        "lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))"
                    ),
                ]
            )

    def define_kernel(
        self,
        kernel_name: str,
        kernel_body: str,
        metadata: str | None = None,
        gpu: bool = True,
        cpp_definition: str | None = None,
    ):
        # [FUNCTION DOC] Queues a KernelDefinitionLine with the kernel name, body, metadata, gpu flag and cpp definition.
        self.writeline(
            KernelDefinitionLine(
                self,
                kernel_name,
                kernel_body,
                metadata=metadata,
                gpu=gpu,
                cpp_definition=cpp_definition,
            )
        )

    @staticmethod
    def _format_kernel_definition(
        kernel_name: str, kernel_body: str, metadata: str | None = None
    ):
        # [FUNCTION DOC] HELPER: Formats a kernel definition as an assignment of the body to the kernel name, preceded by the metadata comment; when autotuning at compile time, C++ style comment starters in the metadata are converted to python style.
        if config.triton.autotune_at_compile_time and metadata:
            # Generating autotune block
            # Need to replace C++ comment starter with Python comment starter
            metadata = re.sub(r"^// ", "# ", metadata, flags=re.MULTILINE)
        metadata_comment = f"{metadata}\n" if metadata else ""
        body = f"\n\n{metadata_comment}{kernel_name} = {kernel_body}"
        return body

    def _define_kernel_helper(
        self,
        kernel_name: str,
        kernel_body: str,
        metadata: str | None = None,
        gpu: bool = True,
        cpp_definition: str | None = None,
    ):
        # [FUNCTION DOC] HELPER: Emits a kernel definition. With autotune at compile time and a gpu kernel it also splices the definition into the autotune defs (and stops there for cpp_wrapper); otherwise splices into the header.
        if config.triton.autotune_at_compile_time and gpu:
            body = self._format_kernel_definition(
                kernel_name, kernel_body, metadata=metadata
            )
            self.kernel_autotune_defs.splice(body)
            if V.graph.cpp_wrapper:
                # For cpp wrapper, no need to continue codegen for the main body
                return

        body = self._format_kernel_definition(
            kernel_name, kernel_body, metadata=metadata
        )
        self.header.splice(body)

    def define_subgraph_launcher_fn(self, name: str, subgraph_code):
        # [FUNCTION DOC] Splices the subgraph launcher code (its value) into the subgraph definitions buffer.
        self.subgraph_definitions.splice(subgraph_code.value)

    @classmethod
    def _get_triton_info_kernel_cls(cls):
        # [FUNCTION DOC] HELPER: Returns the TritonKernel class used to obtain common triton imports and inductor metadata; other backends override it.
        # Other inductor triton backends may subclass from
        # the `TritonKernel` class. An override of this method
        # allows them to set which subclass to use to get information
        # such as common triton imports or inductor metadata
        from .triton import TritonKernel

        return TritonKernel

    def define_user_defined_triton_kernel(
        self,
        kernel,
        configs,
        kwargs,
        restore_value_args,
        reset_to_zero_args,
        grids: list[list[int | sympy.Expr]],
        epilogue_fusion: tuple[ir.ComputedBuffer, str] | None,
        launch_kwargs: tuple[str, ...],
    ):
        # [FUNCTION DOC] Codegens a user-defined Triton kernel and returns (name, triton_meta, inductor_meta, extra launcher call args). Builds the kernel signature and constants from kernel params and kwargs (constexpr, None, equal to 1, TMA descriptors, buffers, reinterpret views and size args), makes triton_meta (signature, device, constants, configs, restore/reset arguments, filtered backend options), and inductor_meta (fixed grid passed as args for one grid or precomputed grids with renamed launcher symbols for several). Computes a cache key from the function id, non-tensor args and metas, returning the cached entry if present. Otherwise assigns a unique name, writes the async_compile.triton wrapper with the user_autotune decorator and escaped kernel source into a kernel definition, and records the cache entry. Raises assertion errors on inconsistent kwargs/grids. Side effects: writes definitions and mutates the kernel cache.
        """Codegen a user-defined Triton kernel and return its cache entry.

        Emits the ``async_compile.triton(...)`` wrapper, assigns a graph-unique
        name (with a leading dunder stripped to avoid Python class-based name
        mangling at the call site), and records the kernel in
        ``user_defined_kernel_cache``. Returns ``(name, triton_meta,
        inductor_meta, extra_launcher_call_args)``; subsequent calls with the
        same ``cache_key`` reuse the previously assigned name.
        """
        from torch._dynamo.device_interface import get_interface_for_device

        from ..runtime.triton_compat import GPUTarget
        from ..runtime.triton_helpers import try_filter_backend_options_for_target
        from ..runtime.triton_heuristics import (
            config_to_dict,
            FixedGrid,
            PrecomputedGrid,
        )
        from .common import (
            ConstexprArg,
            KernelArgType,
            SizeArg,
            TensorArg,
            TMADescriptorArg,
        )

        original_name = kernel.__name__
        signature: list[KernelArgType] = []
        constants: dict[str, Any] = {}
        arg_indices: list[int] = []
        equal_to_1_args: list[str] = []

        def add_to_signature(idx, arg):
            # [FUNCTION DOC] HELPER: Appends an argument and its index to the signature and index lists.
            signature.append(arg)
            arg_indices.append(idx)

        def add_arg(idx, arg, is_constexpr=False, equals_1=False, equals_none=False):
            # [FUNCTION DOC] HELPER: Adds a kernel argument to the signature/constants. Constexpr args are added to the signature only for newer triton versions and recorded as constants if given in kwargs; non-constexpr args must be in kwargs (assertion error otherwise): equal-to-1 args become constants (signature entry as constexpr in newer triton), None args become None constants, others go in the signature as is.
            # KNOWN ISSUE: the two comments describing equal-to-1 handling both say new versions of triton; the second was probably meant for old versions.
            if is_constexpr:
                if triton_version_uses_attrs_dict():
                    # tl.constexpr args appear in the signature in new versions of triton,
                    # but not in old versions of triton.
                    add_to_signature(idx, arg)

                if arg.name in kwargs:
                    # the arg may not appear in kwargs if it is an autotuned arg.
                    # in this case, it will be added in triton_heuristics after autotuning.
                    constants[arg.name] = kwargs[arg.name]

            else:
                # the only case where arg name isn't in kwargs, should be
                # when the arg is a constexpr.
                if arg.name not in kwargs:
                    raise AssertionError(f"expected {arg.name} in kwargs")

                if equals_1:
                    if triton_version_uses_attrs_dict():
                        # new versions of triton: add the equal-to-1 arg in the signature (labeled as "constexpr"),
                        #                         and add the arg as a constant.
                        # new versions of triton: add the equal-to-1 arg in the signature (labeled as, e.g., "i32"),
                        #                         and add the arg as a constant.
                        add_to_signature(idx, ConstexprArg(name=arg.name))
                    else:
                        add_to_signature(idx, arg)
                    constants[arg.name] = 1
                elif equals_none:
                    if triton_version_uses_attrs_dict():
                        # new versions of triton: add the none arg in the signature (as a constexpr arg) and as a constant
                        # old versions of triton: include the none arg as a constant (but not in the signature)
                        add_to_signature(idx, ConstexprArg(name=arg.name))
                    constants[arg.name] = None
                else:
                    add_to_signature(idx, arg)

        arg_names = [p.name for p in kernel.params]
        constexprs = [p.num for p in kernel.params if p.is_constexpr]
        for idx, key in enumerate(arg_names):
            if idx in constexprs:
                add_arg(idx, ConstexprArg(name=key), is_constexpr=True)
                continue

            if key not in kwargs:
                continue

            arg = kwargs[key]

            if kwargs[key] is None:
                add_arg(idx, ConstexprArg(name=key), equals_none=True)
            else:
                if isinstance(arg, ir.TMADescriptor):
                    api_type, block_shape, dtype = (
                        ("stable", arg.block_shape, arg.tensor.get_dtype())
                        if isinstance(arg, ir.TMADescriptorStable)
                        else ("experimental", None, None)
                    )
                    add_arg(
                        idx,
                        TMADescriptorArg(
                            name=key,
                            api_type=api_type,
                            block_shape=block_shape,
                            dtype=dtype,
                        ),
                    )
                elif isinstance(arg, ir.Buffer):
                    add_arg(
                        idx,
                        TensorArg(
                            name=key,
                            buffer=arg.get_name(),
                            dtype=arg.get_dtype(),
                        ),
                    )
                elif isinstance(arg, ir.ReinterpretView):
                    # for ReinterpretView we use the underlying
                    # buffer name and note the (possibly non-zero)
                    # offset relative to the underlying buffer
                    add_arg(
                        idx,
                        TensorArg(
                            name=key,
                            buffer=arg.data.get_name(),
                            dtype=arg.get_dtype(),
                            offset=arg.layout.offset,
                        ),
                    )
                else:
                    equals_1 = isinstance(
                        arg, (int, sympy.Integer)
                    ) and V.graph.sizevars.statically_known_equals(
                        arg,
                        1,  # type: ignore[arg-type]
                    )
                    add_arg(idx, SizeArg(key, arg), equals_1=equals_1)

        triton_signature = signature_to_meta(
            signature,
            size_dtype=None,  # try to infer based on symints
            indices=arg_indices,
            argdefs=[ArgName(x) for x in kernel.arg_names],
        )
        device = V.graph.get_current_device_or_throw()
        device_props = DeviceProperties.create(device)
        triton_meta: TritonMeta = {
            "signature": triton_signature,
            "device": device_props,
            # Triton compiler includes equal_to_1 args into constants even
            # when they are not constexpr. otherwise there may be a segfault
            # during launching the Inductor-compiled Triton kernel.
            # TODO(aakhundov): add None args to constants, too. currently, this
            # causes CUDA errors in test_aot_inductor.test_triton_kernel_with_none_input.
            # https://github.com/pytorch/pytorch/issues/120478#issuecomment-1962822307
            # https://github.com/triton-lang/triton/blob/231efe9ed2d200be0f69a07c298e4342b08efe3d/python/triton/runtime/jit.py#L384
            "constants": {
                **constants,
                **dict.fromkeys(equal_to_1_args, 1),
            },
            "configs": [
                config_of(
                    signature,
                    indices=arg_indices,
                    pointer_range_override=(),
                )
            ],
        }

        if restore_value_args:
            triton_meta["restore_value"] = tuple(restore_value_args)

        if reset_to_zero_args:
            triton_meta["reset_to_zero"] = tuple(reset_to_zero_args)

        backend_option_candidates = {
            name: kwargs[name] for name in launch_kwargs if name in kwargs
        }
        if backend_option_candidates:
            get_interface_for_device(device).raise_if_triton_unavailable(device)
            if GPUTarget is None:
                raise AssertionError("expected GPUTarget to be not None")
            target = GPUTarget(
                device_props.type,
                device_props.cc,
                device_props.warp_size_or_default,
            )
            # HOP capture deliberately keeps an over-approximation of launch kwargs so
            # a concrete kwarg can still have Triton's direct-call dual meaning:
            # kernel parameter plus backend option. Once the target backend is known,
            # only names accepted by parse_options() should be serialized into
            # triton_meta["backend_options"]. Names that are neither kernel parameters
            # nor backend options are invalid launch kwargs, matching eager Triton.
            filtered_backend_options = try_filter_backend_options_for_target(
                target, backend_option_candidates, kernel.arg_names
            )
            if filtered_backend_options:
                triton_meta["backend_options"] = filtered_backend_options

        if len(grids) == 1:
            # compute the grid in the wrapper and pass it in as an arg
            inductor_meta: dict[str, Any] = FixedGrid.setup_grid_as_args()
            extra_launcher_call_args = [*map(sympy.sympify, grids[0])]
        else:

            def rename_sizes_for_launcher(expr: int | sympy.Expr) -> sympy.Expr:
                # [FUNCTION DOC] HELPER: Rewrites a grid expression so its free symbols (sorted by name) are replaced by launcher-local symbols (_launcher_s<N>), recording the mapping in extra_launcher_args. Plain ints become sympy integers; other types raise an assertion error.
                if isinstance(expr, sympy.Expr):
                    symbols = [*expr.free_symbols]
                    if not symbols:
                        return expr
                    symbols.sort(key=str)
                    for sym in symbols:
                        if sym in extra_launcher_args:
                            continue
                        extra_launcher_args[sym] = sympy.Symbol(
                            f"_launcher_s{len(extra_launcher_args)}"
                        )
                    return sympy_subs(expr, extra_launcher_args)
                if not isinstance(expr, int):
                    raise AssertionError(f"expected int, got {type(expr)}")
                return sympy.Integer(expr)

            extra_launcher_args: dict[sympy.Symbol, sympy.Symbol] = {}
            grids = [[*map(rename_sizes_for_launcher, grid)] for grid in grids]

            if not (grids and len(grids) == len(configs)):
                raise AssertionError(
                    f"expected non-empty grids matching configs, got {len(grids)} grids and {len(configs)} configs"
                )
            precomputed_grids = []
            for grid, cfg in sorted(
                zip(grids, configs), key=lambda x: len(x[1].kwargs), reverse=True
            ):
                precomputed_grids.append(
                    {
                        "config": config_to_dict(cfg),
                        "python": [*map(pexpr, grid)],
                        "cpp": [*map(cexpr, grid)],
                        "python_slow": [*map(pexpr, grid)],
                    }
                )
            inductor_meta = {
                "grid_type": PrecomputedGrid.__name__,
                "precomputed_grids": precomputed_grids,
                "extra_launcher_args": [*map(str, extra_launcher_args.values())],
            }
            extra_launcher_call_args = [*extra_launcher_args.keys()]

        if constexprs:
            inductor_meta["declared_constexpr_names"] = [
                arg_names[i] for i in constexprs
            ]

        # Distinguish between different functions using function id
        cache_key: Any = [id(kernel.fn)]
        if len(configs) > 0:
            for arg in kwargs.values():
                # We need to key on non tensor arg only in autotune mode
                if not isinstance(arg, (ir.Buffer, ir.ReinterpretView)):
                    cache_key.append(arg)
        cache_key.append(str(triton_meta))
        cache_key.extend(str(inductor_meta))

        if epilogue_fusion is not None:
            cache_key.append((epilogue_fusion[0].get_name(), epilogue_fusion[1]))

        cache_key = tuple(cache_key)
        if cache_key in self.user_defined_kernel_cache:
            name, triton_meta, cached_inductor_meta = self.user_defined_kernel_cache[
                cache_key
            ]
            return (
                name,
                triton_meta,
                cached_inductor_meta,
                extra_launcher_call_args,
            )

        name = f"{original_name}_{len(self.user_defined_kernel_cache)}"
        # Prevent Python class-based name mangling (``__x`` -> ``_Class__x``)
        # when the generated call site is inside a class body.
        # See https://github.com/pytorch/pytorch/issues/170398
        if name.startswith("__") and not name.endswith("__"):
            name = name[1:]

        compile_wrapper = IndentedBuffer()
        if config.triton.unique_user_kernel_names:
            compile_wrapper.writeline(f"async_compile.triton({name!r}, '''")
        else:
            compile_wrapper.writeline(f"async_compile.triton({original_name!r}, '''")

        inductor_meta["kernel_name"] = name
        triton_info_kernel_cls = self._get_triton_info_kernel_cls()
        inductor_meta.update(triton_info_kernel_cls.inductor_meta_common())

        compile_wrapper.splice(triton_info_kernel_cls.gen_common_triton_imports())
        if config.triton.proton_profiling:
            compile_wrapper.writeline('pl.enable_semantic("triton")')

        # Sanitize triton_meta to convert Enum values for valid Python repr
        sanitized_triton_meta = _sanitize_for_repr(triton_meta)
        compile_wrapper.splice(
            f"""
            @triton_heuristics.user_autotune(
                configs={[*map(config_to_dict, configs)]!r},
                inductor_meta={inductor_meta!r},
                triton_meta={sanitized_triton_meta!r},
                filename=__file__,
                custom_kernel=True,
            )
            @triton.jit
            """
        )
        kernel_src = user_defined_triton_kernel_transitive_closure_source_code(
            kernel, epilogue_fusion
        )
        if config.triton.unique_user_kernel_names:
            # We replace the original_name with the unique name.
            kernel_src = kernel_src.replace(f"def {original_name}(", f"def {name}(")
        kernel_src = _escape_triton_kernel_source_for_wrapper(kernel_src)
        compile_wrapper.splice(kernel_src)

        current_device = V.graph.get_current_device_or_throw()
        compile_wrapper.writeline(f"''', device_str='{current_device.type}')")
        _, lineno = inspect.getsourcelines(kernel.fn)
        srcfile = inspect.getsourcefile(kernel.fn)
        metadata = f"# Original path: {srcfile}:{lineno}"
        self.define_kernel(
            name,
            compile_wrapper.getvalue(),
            metadata,
        )
        # Add to the cache for the next use
        self.user_defined_kernel_cache[cache_key] = (name, triton_meta, inductor_meta)
        return name, triton_meta, inductor_meta, extra_launcher_call_args

    def generate_numel_expr(self, kernel_name: str, tree, suffix: str | None = None):
        # [FUNCTION DOC] Creates the symbolic call argument for the number of elements of a kernel tree (symbol named from kernel name, tree prefix and optional suffix), queues a SymbolicCallArgLine defining it unless this is the benchmark kernel (empty name), and returns the argument.
        sym_name = f"{kernel_name}_{tree.prefix}numel"
        if suffix is not None:
            sym_name += f"_{suffix}"
        sym = sympy.Symbol(sym_name, is_integer=True, is_positive=True)

        # We can get symbolic expressions here, like s0*64
        # It is fine to have them here, but we need to handle them correctly as their own type
        # This is tricky to do, so we wrap in a custom type, distinct from scalars, but also from sympy*
        # scalars as well.
        # This is handled in `generate_args_decl` which has a correct comment of: TODO: only works for
        # constant now, need type info. I agree, this needs type info, and while this is not true type info
        # it suffices as a type hint for the purposes of producing the correct code for this type.
        arg = SymbolicCallArg(sym, tree.numel)

        is_benchmark_kernel = kernel_name == ""
        if not is_benchmark_kernel:
            self.writeline(SymbolicCallArgLine(self, arg, V.graph))

        return arg

    def _generate_symbolic_call_arg_helper(
        self, arg: SymbolicCallArg, graph: GraphLowering
    ) -> None:
        # [FUNCTION DOC] HELPER: Writes an assignment of the argument symbol to the printed expression.
        self.writeline(f"{arg.inner} = {pexpr(arg.inner_expr)}")

    def generate_workspace_allocation(self, ws: WorkspaceArg):
        # [FUNCTION DOC] Emits allocation of a workspace buffer according to its zero mode: uninitialized allocates only, zero-on-call allocates then zeroes, zero-per-graph allocates and zeroes once and later requests only grow the existing allocation to the maximum (assertion error if the prior is not an AllocateLine of a WorkspaceArg); unknown modes raise an assertion error. Under autotune at compile time also writes allocation (and zeroing) into the autotune calls.
        name = ws.get_name()
        line = AllocateLine(self, ws)
        if ws.zero_mode == WorkspaceZeroMode.UNINITIALIZED:
            self.writeline(line)
        elif ws.zero_mode == WorkspaceZeroMode.ZERO_ON_CALL:
            self.writeline(line)
            self.writeline(self.make_zero_buffer(name))
        elif ws.zero_mode == WorkspaceZeroMode.ZERO_PER_GRAPH:
            prior = self.allocated_workspaces.get(name)
            if prior:
                if not (
                    isinstance(prior, AllocateLine)
                    and isinstance(prior.node, WorkspaceArg)
                ):
                    raise AssertionError(
                        "expected prior to be an AllocateLine with a WorkspaceArg node"
                    )
                # expand existing allocation
                prior.node = WorkspaceArg.maximum(prior.node, ws)
            else:
                self.writeline(line)
                self.writeline(self.make_zero_buffer(name))
                self.allocated_workspaces[name] = line
        else:
            raise AssertionError(ws.zero_mode)

        if config.triton.autotune_at_compile_time:
            self.kernel_autotune_calls.writeline(
                PythonWrapperCodegen.make_allocation(
                    self,
                    name,
                    ws.device,
                    ws.dtype,
                    shape=(V.graph.sizevars.optimization_hint(ws.count),),
                    stride=(1,),
                )
            )
            if ws.zero_mode != WorkspaceZeroMode.UNINITIALIZED:
                self.kernel_autotune_calls.writeline(
                    PythonWrapperCodegen.make_zero_buffer(self, name)
                )

    def generate_workspace_deallocation(self, ws: WorkspaceArg):
        # [FUNCTION DOC] Queues a free for the workspace unless its zero mode is zero-per-graph.
        if ws.zero_mode != WorkspaceZeroMode.ZERO_PER_GRAPH:
            self.writeline(FreeIfNotReusedLine(self, ws))

    def make_zero_buffer(self, name):
        # [FUNCTION DOC] Returns the statement zeroing the named buffer.
        return f"{name}.zero_(){self.ending}"

    def wrap_kernel_call(self, name, call_args):
        # [FUNCTION DOC] Returns the call statement text for a kernel name with joined call args and the ending.
        return f"{name}({', '.join(call_args)}){self.ending}"

    def generate_debug_sync(self, buffer):
        # [FUNCTION DOC] Writes a device synchronize statement into the given buffer.
        buffer.writeline(V.graph.device_ops.synchronize())

    def generate_profiler_mark_wrapper_call(self, stack):
        # [FUNCTION DOC] Writes a profiler record_function import and with-statement into wrapper_call and enters its indent on the given exit stack.
        self.wrapper_call.writeline("from torch.profiler import record_function")
        self.wrapper_call.writeline(
            f"with record_function('graph_{V.graph.graph_id}_inductor_wrapper_call'):"
        )
        stack.enter_context(self.wrapper_call.indent())

    def generate_start_graph(self):
        # [FUNCTION DOC] Writes a start_graph call into the wrapper call.
        self.wrapper_call.writeline("start_graph()")

    def generate_end_graph(self):
        # [FUNCTION DOC] Writes an end_graph call with the profile output path into the wrapper call.
        self.wrapper_call.writeline(f"end_graph({config.profile_bandwidth_output!r})")

    def generate_proton_finalize(self):
        # [FUNCTION DOC] Writes a device synchronize into the wrapper call so proton captures all kernel events.
        """Synchronize GPU to ensure proton captures all kernel events."""
        self.wrapper_call.writeline(V.graph.device_ops.synchronize())

    def generate_reset_kernel_saved_flags(self):
        # [FUNCTION DOC] Splices code resetting the saved-kernel flags of all CachingAutotuner kernels in globals into the wrapper call.
        self.wrapper_call.splice(
            f"""
            for kernel in globals().values():
                if isinstance(kernel, {triton_heuristics.__name__}.CachingAutotuner):
                    kernel.cuda_kernel_saved = False
                    kernel.cpu_kernel_saved = False
            """
        )

    def generate_save_uncompiled_kernels(self):
        # [FUNCTION DOC] Splices code into the wrapper call that precompiles and saves (cpu or gpu) artifacts of any autotuner kernels not yet saved, so kernels outside the executed control flow path are saved for later AOT compilation.
        """
        Precompile and save the per-config kernel artifacts (CUBINs on GPU,
        ``.so`` + launcher ``.so`` on CPU) for Triton kernels that haven't
        been precompiled and saved as a side effect of running the generated
        JIT model (Python wrapper). This can happen when the model contains
        control flow: only one pass through the control flow operators covers
        the kernels that are saved, the remaining kernels are not launched,
        hence not saved. The main purpose of this codegen is to compile and
        save the Triton kernels outside the active control flow path for
        subsequent AOTInductor code generation and compilation.
        """
        self.wrapper_call.splice(
            f"""
            for kernel in globals().values():
                if isinstance(kernel, {triton_heuristics.__name__}.CachingAutotuner):
                    if kernel.device_props.type == "cpu":
                        if not kernel.cpu_kernel_saved:
                            if len(kernel.launchers) == 0:
                                kernel.precompile()
                            kernel.save_cpu_kernel(launcher=kernel.launchers[0])
                    else:
                        if not kernel.cuda_kernel_saved:
                            if len(kernel.launchers) == 0:
                                kernel.precompile()
                            kernel.save_gpu_kernel(
                                stream="stream",  # use dummy stream
                                launcher=kernel.launchers[0],
                            )
            """
        )

    def prepare_triton_kernel_call(self, call_args):
        # [FUNCTION DOC] Converts kernel call args to code strings: strings are unwrapped with item when they denote unspecialized 0-d cpu tensors, numbers and symbolic args via str, other values simplified and printed. Returns the list.
        def wrap_arg(arg):
            # [FUNCTION DOC] HELPER: Converts one call arg into its code string.
            if isinstance(arg, str):
                # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
                return arg + ".item()" if should_unwrap_unspec_arg(arg) else arg
            elif isinstance(arg, (int, float, bool, SymbolicCallArg)):
                return str(arg)
            else:
                return pexpr(V.graph.sizevars.simplify(arg))

        return [wrap_arg(arg) for arg in call_args]

    def generate_example_arg_value(self, arg, arg_type, raw_arg=None):
        # [FUNCTION DOC] Produces an example value expression for an autotune argument and returns its name or text. For tensor-typed args (dtype type) it writes a generate_example_value assignment (and for TMA descriptors a descriptor creation) into the autotune calls, using the buffer from args_to_buffers or the raw arg (temp names tmp_arg_N); raises an assertion error if no buffer is found. For symbolic args returns the size hint as a string (meta vars returned as is). Plain str/int/float/bool return their str, lists recurse; other types raise NotImplementedError.
        if isinstance(arg_type, torch_dtype):
            if isinstance(raw_arg, ir.TMADescriptor):
                # first we generate the underlying buffer
                buf_name = raw_arg.get_tensor().get_name()
                buf = self.args_to_buffers[arg]
            elif self.args_to_buffers.get(arg):
                buf_name = arg
                buf = self.args_to_buffers[arg]
            else:
                if raw_arg is None:
                    raise AssertionError(
                        "V.graph.get_buffer(arg) and raw_arg can't be None at the same time"
                    )
                buf_name = f"tmp_arg_{self.kernel_autotune_tmp_arg_idx}"
                buf = raw_arg
                self.kernel_autotune_tmp_arg_idx += 1

            if buf is None:
                raise AssertionError(f"Failed to find a buffer for arg {arg}")
            size = V.graph.sizevars.optimization_hints(buf.get_size())
            allocation_size = V.graph.sizevars.optimization_hints(
                V.graph.get_allocation_size(buf)
            )
            stride = V.graph.sizevars.optimization_hints(buf.get_stride())

            device = buf.get_device()
            dtype = buf.get_dtype()
            offset = V.graph.sizevars.optimization_hint(buf.get_layout().offset)
            value = f"generate_example_value({size}, {stride}, '{coor_device_str(device)}', {dtype}, {offset}, {allocation_size})"
            self.kernel_autotune_calls.writeline(f"{buf_name} = {value}")

            if isinstance(raw_arg, ir.TMADescriptor):
                # generate another line initializing a host-side TMA
                # descriptor from the underlying buffer created above
                value = self._generate_tma_descriptor_call(
                    desc=raw_arg,
                    apply_size_hints=True,
                )
                buf_name = arg
                self.kernel_autotune_calls.writeline(f"{buf_name} = {value}")

            return buf_name
        elif issubclass(arg_type, sympy.Basic) or isinstance(arg, SymbolicCallArg):
            # arg is a symbol or symbolic expression
            if isinstance(arg, str):
                if arg in self._meta_vars:
                    return arg
                if raw_arg is None:
                    return "None"
                arg = raw_arg
            if isinstance(arg, SymbolicCallArg):
                arg = arg.inner_expr
            if arg in V.graph.sizevars.inv_precomputed_replacements:
                arg = V.graph.sizevars.inv_precomputed_replacements[arg]

            return str(V.graph.sizevars.optimization_hint(arg))

        elif isinstance(arg, (str, int, float, bool)):
            return str(arg)
        elif isinstance(arg, list):
            return f"[{', '.join(self.generate_example_arg_value(a, type(a)) for a in arg)}]"
        else:
            raise NotImplementedError(f"Unsupported type {type(arg)}")

    def _grid_dim_str(self, grid_per_dim):
        # [FUNCTION DOC] HELPER: Prints a grid dimension (possibly nested lists) as a code string.
        if isinstance(grid_per_dim, list):
            return (
                "[" + ", ".join(self._grid_dim_str(item) for item in grid_per_dim) + "]"
            )
        else:
            return pexpr(grid_per_dim)

    def generate_kernel_call(
        self,
        kernel_name: str,
        call_args,
        *,
        device=None,
        triton=True,
        arg_types=None,
        raw_keys=None,
        raw_args=None,
        triton_meta: TritonMeta | None = None,
        inductor_meta=None,
        original_fxnode_name=None,
    ):
        # [FUNCTION DOC] Records the buffers behind string call args (for later autotune example generation) and queues a KernelCallLine with the current device, stream index and graph name. The triton flag tells whether the backend uses Triton.
        """
        Generates kernel call code.

        triton: Defines whether the backend uses Triton for codegen. Otherwise it uses the CUDA language when gpu=True,
                and C++ when gpu=False.
        """

        # Store buffers corresponding to each call arg.
        # This is used to generate example args for autotuning later on.
        self.args_to_buffers.update(
            {
                arg: V.graph.try_get_buffer(arg)
                for arg in call_args
                if isinstance(arg, str)
            }
        )

        device = device or V.graph.get_current_device_or_throw()
        current_stream_idx = V.graph.scheduler.current_stream_idx
        self.writeline(
            KernelCallLine(
                self,
                kernel_name=kernel_name,
                call_args=call_args,
                # pyrefly: ignore [bad-argument-type]
                raw_keys=raw_keys,
                # pyrefly: ignore [bad-argument-type]
                raw_args=raw_args,
                # pyrefly: ignore [bad-argument-type]
                arg_types=arg_types,
                triton=triton,
                # pyrefly: ignore [bad-argument-type]
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                device=device,
                graph_name=V.graph.name,
                # pyrefly: ignore [bad-argument-type]
                original_fxnode_name=original_fxnode_name,
                current_stream_idx=current_stream_idx,
            )
        )

    def _generate_kernel_call_helper(
        self,
        kernel_name: str,
        call_args,
        *,
        device=None,
        triton=True,
        arg_types=None,
        raw_keys=None,
        raw_args=None,
        triton_meta: TritonMeta | None = None,
        inductor_meta=None,
        graph_name="",
        original_fxnode_name=None,
        current_stream_idx=None,
    ):
        # [FUNCTION DOC] HELPER: Emits the actual kernel call code. Non-triton kernels on cpu or mps become a plain call; on other non-cuda/xpu devices raises RuntimeError. Otherwise obtains a raw stream (a fresh get_raw_stream when inside a user stream context, else the cached per-graph one), non-triton kernels are called through their namespaced name with a stream pointer, and triton kernels are run with the call args and stream. With autotune at compile time (once per kernel) also writes into the autotune calls a device-guarded run of the kernel with example args (inferred from autotuning inputs, reused args, workspace names, or generated example values, with deletion of temporaries), and stops there for cpp_wrapper. The main-body call is wrapped by the debug printer. Raises an assertion error if call args and arg types or raw args are inconsistent.
        device = device or V.graph.get_current_device_or_throw()
        if not triton and device.type not in ("cuda", "xpu"):
            if device.type == "cpu":
                self.writeline(self.wrap_kernel_call(kernel_name, call_args))
            elif device.type == "mps":
                # TODO: Fix me, MPS does not expose streams now
                self.writeline(self.wrap_kernel_call(kernel_name, call_args))
            else:
                raise RuntimeError(f"device {device.type} nyi")
            return

        call_args_str = self.prepare_triton_kernel_call(call_args)
        call_args_str = ", ".join(call_args_str)
        if current_stream_idx is not None and current_stream_idx != DEFAULT_STREAM_IDX:
            # Inside a user stream context: emit a fresh get_raw_stream call so
            # it picks up the active stream at runtime, rather than reusing the
            # LRU-cached stream0 variable which captured the default stream.
            self.write_get_raw_stream_header()
            stream_name = "raw_stream"
            self.writeline(
                f"{stream_name} = get_raw_stream({_coor_device_idx_ref(device.index)})"
            )
        else:
            stream_name = PythonWrapperCodegen.write_get_raw_stream(
                self, device.index, graph_name
            )
        if not triton:
            stream_ptr = f"c_void_p({stream_name})"
            self.writeline(
                f"{kernel_name}.{kernel_name}({call_args_str}, {stream_ptr})"
            )
            return

        self.write_triton_header_once()

        if (
            config.triton.autotune_at_compile_time
            and kernel_name not in self.kernel_autotune_names
        ):
            # Create example args for autotune in a separate epilogue
            if not (arg_types is not None and len(call_args) == len(arg_types)):
                raise AssertionError("call_args and arg_types do not match")

            autotune_args = None
            if original_fxnode_name and V.graph.autotuning_mapping:
                autotune_args = V.graph.autotuning_mapping.get(
                    original_fxnode_name, None
                )

            _per_kernel = config.aot_inductor.autotune_per_kernel_alloc

            def get_autotune_deletion_call() -> str:
                # [FUNCTION DOC] HELPER: Returns a delete statement for example tensors whose last consumer is this kernel, or an empty string.
                """Returns del for tensors whose last consumer is this kernel."""
                tensors_to_delete = [
                    tensor
                    for tensor, kn in self.kernel_autotune_example_args.values()
                    if kn == kernel_name
                ]
                if tensors_to_delete:
                    return f"del {', '.join(tensors_to_delete)}\n"
                return ""

            def infer_arg_by_inputs(raw_keys, raw_args, idx, reused_args):
                # [FUNCTION DOC] HELPER: Tries to infer an unspecified raw arg from the other raw args that are IR nodes (useful for jagged cases); see the body for the exact search.
                """We try to infer raw_arg (i.e. raw_args[idx]) from remaining raw_args.
                This is particularly useful for jagged cases, where the dimension is often
                being passed in as an input."""

                target_arg = raw_args[idx]
                if target_arg in reused_args:
                    return True

                for i, (raw_key, raw_arg) in enumerate(zip(raw_keys, raw_args)):
                    if i == idx or not isinstance(raw_arg, IRNode):
                        continue

                    triton_input = ""
                    if autotune_args and raw_key in autotune_args:
                        triton_input = self.get_autotuning_input_name(  # type: ignore[attr-defined]
                            autotune_args[raw_key]
                        )
                    if triton_input == "":
                        continue

                    try:
                        layout = raw_arg.get_layout()
                        for dim, s in enumerate(layout.size):
                            if s == target_arg:
                                reused_args[target_arg] = f"{triton_input}.shape[{dim}]"
                                return True
                    except NotImplementedError:
                        # If layout for this IRNode is not implemented, we could just skip.
                        # Only raise for other Error cases.
                        continue
                return False

            all_args = []
            tensor_arg_strs = []  # used only when _per_kernel is True
            if raw_args is None:
                # create a dummy raw_args for uniform behavior in the following loop
                if raw_keys is not None:
                    raise AssertionError("keys are not None but args are")
                raw_keys = [None] * len(call_args)
                raw_args = [None] * len(call_args)
            else:
                if len(raw_args) != len(call_args):
                    raise AssertionError("call_args and raw_args do not match")

            reused_args = {}
            for i, (arg, arg_type, raw_key, raw_arg) in enumerate(
                # pyrefly: ignore [bad-argument-type, no-matching-overload]
                zip(call_args, arg_types, raw_keys, raw_args)
            ):
                key = None
                if isinstance(arg, str) and "=" in str(arg):
                    # arg may be passed in a kwarg style, and then we need to extract its value
                    key, arg = arg.split("=")

                triton_input: str | None = None
                if autotune_args and raw_key in autotune_args:
                    triton_input = self.get_autotuning_input_name(  # type: ignore[attr-defined]
                        autotune_args[raw_key]
                    )

                if triton_input:
                    arg_str = triton_input
                    if not isinstance(arg_type, torch_dtype) and (
                        issubclass(arg_type, sympy.Basic)
                        or isinstance(arg, SymbolicCallArg)
                    ):
                        reused_args[raw_arg] = arg_str
                elif raw_key == "" and infer_arg_by_inputs(
                    raw_keys, raw_args, i, reused_args
                ):
                    # Empty raw_key means this is an arg that's not native to the triton kernel,
                    # and is being added by inductor.
                    arg_str = reused_args[raw_arg]
                elif isinstance(arg_type, torch_dtype):
                    # workspace allocation is already generated by `generate_workspace_allocation()`
                    # in `TritonKernel.call_kernel()`.
                    if re.match(r"^(workspace|semaphore)", arg):
                        arg_str = arg
                    elif _per_kernel:
                        arg_str = self.generate_example_arg_value(
                            arg, arg_type, raw_arg
                        )
                        tensor_arg_strs.append(arg_str)
                    elif arg not in self.kernel_autotune_example_args:
                        arg_str = self.generate_example_arg_value(
                            arg, arg_type, raw_arg
                        )
                    else:
                        arg_str = self.kernel_autotune_example_args[arg][0]
                    if not _per_kernel:
                        self.kernel_autotune_example_args[arg] = (arg_str, kernel_name)
                else:
                    arg_str = self.generate_example_arg_value(arg, arg_type, raw_arg)

                if isinstance(arg, str) and should_unwrap_unspec_arg(arg):
                    arg_str += ".item()"
                all_args.append(arg_str if key is None else f"{key}={arg_str}")

            # Make sure kernel launch under a device guard because models don't always run on device 0
            # The autotune block inlines the current-device expression rather than using the
            # call()-local _coor_device_idx variable, which is not in scope here.
            guard_idx = (
                V.graph.device_ops.current_device_idx_expr()
                if _coor_enabled()
                else device.index
            )
            self.kernel_autotune_calls.writeline(
                f"with {V.graph.device_ops.device_guard(guard_idx)}:"
            )
            self.kernel_autotune_calls.do_indent()
            self.kernel_autotune_calls.writeline(
                f"{kernel_name}.run({', '.join(all_args)}, stream={stream_name})"
            )
            self.kernel_autotune_calls.do_unindent()

            if _per_kernel and tensor_arg_strs:
                self.kernel_autotune_calls.writeline(
                    f"del {', '.join(tensor_arg_strs)}"
                )
            elif not _per_kernel:
                self.kernel_autotune_calls.writeline(
                    DelayReplaceLine(
                        "<del_call>",
                        get_autotune_deletion_call,
                        "<del_call>",
                    )
                )
            self.kernel_autotune_names.add(kernel_name)
            if V.graph.cpp_wrapper:
                # For cpp wrapper, no need to continue codegen for the main body
                return

        # add debug printer code for triton kernel calls at (jit) inductor level
        debug_printer_manager = V.graph.wrapper_code.debug_printer
        debug_printer_manager.set_printer_args(call_args, kernel_name, arg_types, None)
        with debug_printer_manager:
            self.writeline(f"{kernel_name}.run({call_args_str}, stream={stream_name})")
        self.write_triton_header_once()

    def writeline(self, line):
        # [FUNCTION DOC] Appends a line (string, deferred line, or wrapper line object) to the list of lines to be replayed later.
        self.lines.append(line)

    def writelines(self, lines):
        # [FUNCTION DOC] Appends each of the given lines using writeline.
        for line in lines:
            self.writeline(line)

    def enter_context(self, ctx):
        # [FUNCTION DOC] Appends a LineContext wrapping the given context to the lines.
        self.lines.append(LineContext(ctx))

    def val_to_arg_str(self, s, type_=None):
        # [FUNCTION DOC] Converts a value into its code string for generated python. Symbolic ints and sympy expressions are printed with the expression printer; tuples and lists are converted element by element; ops give their qualified name; buffers, views, generator and opaque states give their reference; triton dtypes and everything else use repr; opaque constant types give their repr and register their classes on the graph; under compile-on-one-rank indexed devices are emitted as the bare device type.
        from torch.utils._triton import has_triton_package

        if has_triton_package():
            import triton

        if isinstance(s, SymTypes):
            return pexpr(s.node.expr)
        elif isinstance(s, sympy.Expr):
            return pexpr(s)
        elif isinstance(s, (tuple, list)):

            @dataclasses.dataclass
            class Shim:
                # [CLASS DOC] HELPER: Local dataclass whose repr returns the stored string unquoted, so already-rendered code strings can be placed inside a printed tuple or list.
                ref: Any

                def __repr__(self):
                    # [FUNCTION DOC] HELPER: Returns the reference string as is.
                    return self.ref

            # Explicitly call the Python version of val_to_arg_str
            return repr(
                type(s)(Shim(PythonWrapperCodegen.val_to_arg_str(self, a)) for a in s)
            )
        elif isinstance(s, torch._ops.OpOverload):
            return _get_qualified_name(s)
        elif isinstance(s, (ir.Buffer, ir.MutableBox, ReinterpretView)):
            return s.codegen_reference()
        elif has_triton_package() and isinstance(s, triton.language.dtype):  # type: ignore[possibly-undefined]
            return repr(s)
        elif isinstance(s, (ir.GeneratorState, ir.OpaqueObjectState)):
            return s.codegen_reference()
        elif is_opaque_constant_type(type(s)):
            obj_repr, opaque_types = get_opaque_obj_repr(s)
            for n, t in opaque_types.items():
                V.graph.opaque_value_type_classes[n] = t
            return obj_repr
        elif isinstance(s, torch.device) and _coor_enabled() and s.index is not None:
            # compile-on-one-rank: repr() of an indexed device would bake this rank's
            # index into the wrapper (e.g. an aten fallback's device= arg renders as
            # device(type='cuda', index=0)), which the "cuda:N" checks do not catch.
            # Emit the bare type so the value follows the enclosing runtime device guard.
            return repr(torch.device(s.type))
        else:
            return repr(s)

    # The following methods are for memory management
    def make_buffer_allocation(self, buffer: BufferLike):
        # [FUNCTION DOC] Returns the allocation statement string for a buffer using its device, dtype, size, allocation size, stride and pinned flag.
        device = buffer.get_device()
        dtype = buffer.get_dtype()
        shape = tuple(buffer.get_size())
        allocation_shape = tuple(V.graph.get_allocation_size(buffer))
        stride = tuple(buffer.get_stride())
        is_pinned = buffer.get_is_pinned()
        return self.make_allocation(
            buffer.get_name(), device, dtype, shape, stride, allocation_shape, is_pinned
        )

    @cache_on_self
    def write_memory_track_allocation_once(self):
        # [FUNCTION DOC] Once per wrapper (cached): adds the memory tracking helper imports to the imports buffer unless generating for cpp_wrapper.
        import_str = """
            from torch._inductor.runtime.debug_utils import check_memory_step, track_tensor
            """
        if not V.graph.cpp_wrapper:
            self.imports.splice(import_str, strip=True)

    def make_allocation(
        self, name, device, dtype, shape, stride, allocation_shape=None, is_pinned=False
    ):
        # [FUNCTION DOC] Returns an allocation statement string. Uses tracked_empty_strided when memory lifecycle tracking is enabled; pinned cpu uses the pinned allocator; cpu/cuda/xpu/mtia use the device specific empty_strided helper; other devices use the generic empty_strided with device and dtype. When the allocation shape differs from the logical shape appends an as_strided call to get the logical view.
        if allocation_shape is None:
            allocation_shape = shape

        codegen_shape_tuple = self.codegen_python_shape_tuple(shape)
        codegen_allocation_shape_tuple = self.codegen_python_shape_tuple(
            allocation_shape
        )
        codegen_stride_tuple = self.codegen_python_shape_tuple(stride)
        if torch._inductor.config.test_configs.track_memory_lifecycle:
            out = (
                f"{name} = tracked_empty_strided("
                f"{codegen_allocation_shape_tuple}, "
                f"{codegen_stride_tuple}, "
                f"dtype={dtype}, "
                f"device='{device.type}', "
                f"name='{name}')"
            )
        elif device.type == "cpu" and is_pinned:
            out = (
                f"{name} = empty_strided_cpu_pinned("
                f"{codegen_allocation_shape_tuple}, "
                f"{codegen_stride_tuple}, "
                f"{dtype})"
            )
        elif device.type in ("cpu", "cuda", "xpu", "mtia"):
            # optimized path for faster allocations, saving ~2us versus the stuff below
            out = (
                f"{name} = empty_strided_{device.type}("
                f"{codegen_allocation_shape_tuple}, "
                f"{codegen_stride_tuple}, "
                f"{dtype})"
            )
        # all other devices:
        else:
            out = (
                f"{name} = empty_strided("
                f"{codegen_allocation_shape_tuple}, "
                f"{codegen_stride_tuple}, "
                f"device='{device.type}', dtype={dtype})"
            )
        if codegen_shape_tuple != codegen_allocation_shape_tuple:
            # need an extra as_strided call
            out = out + f".as_strided({codegen_shape_tuple}, {codegen_stride_tuple})"
        return out

    def make_comment(self, line):
        # [FUNCTION DOC] Queues a comment line.
        self.writeline(CommentLine(line))

    def make_tensor_alias(self, new_name, old_name, comment=""):
        # [FUNCTION DOC] Returns an assignment statement aliasing new_name to old_name, with a trailing comment.
        return f"{self.declare}{new_name} = {old_name}{self.ending}  {self.comment} {comment}"

    def make_buffer_free(self, buffer: BufferLike | ir.TorchBindObject):
        # [FUNCTION DOC] Returns the statement deleting the buffer (or torch bind object) by name.
        return f"del {buffer.get_name()}"

    def make_free_by_names(self, names_to_del: list[str]):
        # [FUNCTION DOC] Returns a single delete statement for several names.
        return f"del {', '.join(name for name in names_to_del)}"

    def codegen_exact_buffer_reuse(self, old_name: str, new_name: str, del_line: str):
        # [FUNCTION DOC] Returns the statement binding new_name to old_name followed by the delete suffix, with a reuse comment.
        return f"{self.declare_maybe_reference}{new_name} = {old_name}{del_line}{self.ending}  {self.comment} reuse"

    def write_provenance_debug_handle(
        self,
        kernel_name,
        debug_handle: int | None = None,
    ):
        # [FUNCTION DOC] Writes a comment containing the provenance debug handle for the kernel, if a handle is given.
        if debug_handle is not None:
            self.writeline(
                f"{self.comment} [Provenance debug handles] {kernel_name}:{debug_handle}"
            )

    def make_buffer_reuse(self, old: BufferLike, new: BufferLike, delete_old: bool):
        # [FUNCTION DOC] Returns the statement making new reuse the storage of old (dtypes must match, else assertion error). The old name is also deleted unless it is a graph output or delete_old is false. If sizes and strides match it is a simple rebinding, otherwise a reinterpret view of old with the new size and stride.
        if old.get_dtype() != new.get_dtype():
            raise AssertionError(
                f"expected matching dtypes, got {old.get_dtype()} and {new.get_dtype()}"
            )
        old_name = old.get_name()
        new_name = new.get_name()
        del_line = ";"
        if old_name not in V.graph.get_output_names() and delete_old:
            del_line = f"; {self.make_buffer_free(old)}"

        if old.get_size() == new.get_size() and old.get_stride() == new.get_stride():
            return self.codegen_exact_buffer_reuse(old_name, new_name, del_line)

        reinterpret_view = self.codegen_reinterpret_view(
            old, new.get_size(), new.get_stride(), 0, self.wrapper_call.writeline
        )
        return f"{self.declare}{new_name} = {reinterpret_view}{del_line}  {self.comment} reuse"

    def codegen_deferred_allocation(self, name: str, view: ir.ReinterpretView) -> None:
        # [FUNCTION DOC] Queues a deferred line that defines name as an alias to the view reference, emitted only if the name is used later.
        self.writeline(
            DeferredLine(
                name,
                f"{self.declare}{name} = {view.codegen_reference()}{self.ending}  {self.comment} alias",
            )
        )

    def codegen_allocation(self, buffer: ir.Buffer):
        # [FUNCTION DOC] Queues allocation for a buffer unless it is removed, already allocated, or an input/donated/subgraph buffer; also skips extern-alloc or multi-output buffers that should not allocate, mutation layouts and none layouts. For non-owning layouts it first allocates the underlying buffer (after unwrapping views) and queues a reinterpret line; comm buffer layouts get a comm AllocateLine; otherwise a normal AllocateLine. Marks the buffer allocated. Raises assertion errors on unexpected view or box types.
        name = buffer.get_name()

        if (
            name in V.graph.removed_buffers
            or name in self.allocated
            or isinstance(buffer, (ir.DonatedBuffer, ir.SubgraphBuffer, ir.InputBuffer))
        ):
            return
        self.allocated.add(name)
        if (
            isinstance(
                buffer.get_defining_op(),
                (ir.ExternKernelAlloc, ir.MultiOutput),
            )
            and not buffer.should_allocate()
        ):
            return

        layout = buffer.get_output_spec()
        if isinstance(layout, ir.MutationLayoutSHOULDREMOVE):
            return
        if isinstance(layout, ir.NoneLayout):
            return
        if isinstance(layout, ir.NonOwningLayout):
            if not isinstance(layout.view, ir.ReinterpretView):
                raise AssertionError(f"unexpected {type(layout.view)}: {layout.view}")
            box = layout.view.data
            if not isinstance(box, ir.StorageBox):
                raise AssertionError(type(box))
            input_buffer = box.data
            if not isinstance(input_buffer, (ir.Buffer, ir.ReinterpretView)):
                raise AssertionError(type(input_buffer))
            if isinstance(input_buffer, ir.ReinterpretView):

                def unwrap_views(target) -> ir.Buffer:
                    # [FUNCTION DOC] HELPER: Recursively unwraps views and mutable boxes down to the underlying Buffer; raises an assertion error if it is not a Buffer.
                    if isinstance(target, ir.BaseView):
                        return unwrap_views(target.unwrap_view())
                    if isinstance(target, ir.MutableBox):
                        return unwrap_views(target.data)
                    if not isinstance(target, ir.Buffer):
                        raise AssertionError(type(target))
                    return target

                input_buffer = unwrap_views(input_buffer)
            self.codegen_allocation(input_buffer)
            self.writeline(ReinterpretLine(self, input_buffer, buffer, layout))
            return

        if isinstance(layout, ir.CommBufferLayout):
            self.writeline(AllocateLine(self, buffer, comm_buffer=True))
            return

        self.writeline(AllocateLine(self, buffer))

    def codegen_free(self, buffer):
        # [FUNCTION DOC] Queues freeing of a buffer. Input buffers and torch bind objects get a FreeLine (and the saved original copy of a multistream-aligned input is freed with them). Comm buffers get a comm FreeIfNotReusedLine. Other buffers are skipped if they cannot be reused, otherwise recorded as freed and queued as FreeIfNotReusedLine.
        name = buffer.get_name()

        # can be freed but not reused
        if isinstance(buffer, (ir.InputBuffer, ir.TorchBindObject)):
            self.writeline(FreeLine(self, buffer))
            # A multistream input keeps a preserved copy of its original
            # ({name}_orig, see codegen_deferred_alignment_copies) that each
            # per-stream clone reads from; its lifetime matches the input's, so
            # free it here on the normal last_usage -> codegen_free path.
            if name in self._alignment_orig_saved:
                self._alignment_orig_saved.discard(name)
                self.writeline(self.make_free_by_names([f"{name}_orig"]))
            return

        if isinstance(buffer.get_output_spec(), ir.CommBufferLayout):
            # Comm buffers are not eligible for in-place reuse. Their reuse is
            # achieved exclusively via buffer planning.
            self.writeline(FreeIfNotReusedLine(self, buffer, comm_buffer=True))
            return

        if not self.can_reuse(buffer):
            return
        self.freed.add(name)

        self.writeline(FreeIfNotReusedLine(self, buffer))

    def can_reuse(self, input_buffer, output_buffer=None):
        # [FUNCTION DOC] Returns False if the buffer is removed, is a non-donated graph input, a constant, a torchbind constant, in the never-reuse set, or was already freed; else True. The output_buffer argument is unused.
        name = input_buffer.get_name()
        return not (
            name in V.graph.removed_buffers
            or (
                name in V.graph.graph_inputs
                and not isinstance(
                    V.graph.graph_inputs_original[name], ir.DonatedBuffer
                )
            )
            or name in V.graph.constants
            or name in V.graph.torchbind_constants
            or name in V.graph.never_reuse_buffers
            or name in self.freed
        )

    def did_reuse(self, buffer, reused_buffer):
        # [FUNCTION DOC] Returns True if the buffer was recorded as reused by the given reused buffer; can be consulted from IR codegen to decide whether a copy is needed.
        # Check whether a given buffer was reused by a possible reuser in the wrapper codegen
        # Can be consulted from inside ir codegen, e.g. to determine whether a copy is needed
        return (
            buffer.get_name() in self.reuses
            and self.reuses[buffer.get_name()] == reused_buffer.get_name()
        )

    def codegen_inplace_reuse(self, input_buffer: ir.Buffer, output_buffer: ir.Buffer):
        # [FUNCTION DOC] Records in-place reuse of the input buffer by the output buffer: requires matching sizes (assertion error otherwise), allocates the input if needed, marks the input freed and the output allocated, records the reuse mapping and queues a ReuseLine.
        if not can_match_buffer_size(input_buffer, output_buffer):
            raise AssertionError(
                "expected input_buffer and output_buffer sizes to match"
            )
        self.codegen_allocation(input_buffer)
        self.freed.add(input_buffer.get_name())
        self.allocated.add(output_buffer.get_name())
        self.reuses[output_buffer.get_name()] = input_buffer.get_name()
        self.writeline(ReuseLine(self, input_buffer, output_buffer))

    def codegen_unbacked_symbol_decl(self, symbol):
        # [FUNCTION DOC] Returns the name for an unbacked symbol in a definition: the bare name if it was already declared, otherwise records it and returns the name prefixed by the declaration string.
        name = str(symbol)
        if name in self.unbacked_symbol_decls:
            return name
        else:
            # When in CppWrapperCpu, we should only generate the declaration once
            self.unbacked_symbol_decls.add(name)
            return self.declare + name

    def codegen_unbacked_symbol_defs_for_outputs(
        self,
        output_name: str,
        outputs: Any,
        unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] | None,
    ) -> None:
        # [FUNCTION DOC] Resolves the unbacked bindings against the shape environment and queues an UnbackedSymbolDefsLine for the outputs.
        unbacked_bindings = resolve_unbacked_bindings(
            V.graph.sizevars.shape_env, unbacked_bindings
        )
        self.writeline(
            UnbackedSymbolDefsLine(self, output_name, outputs, unbacked_bindings)
        )

    def _codegen_unbacked_symbol_defs_for_outputs(
        self,
        output_name: str,
        outputs: Any,
        unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] | None,
    ) -> None:
        # [FUNCTION DOC] Emits definitions of unbacked symbols from keypaths into the outputs (nothing if no bindings). For each symbol builds an access expression by walking the keypath (method calls, sequence indexing with tuple access in cpp, divide-by keys) starting from the output name or, for cpp_wrapper, from the output buffer names, and writes the declaration assignment.
        if not unbacked_bindings:
            return

        # This code is designed to generate code expressions from symbolic paths (keypaths)
        # associated with certain symbols (unbacked bindings). These keypaths describe how
        # to access the unbacked symbol in a structured way.
        # For example, we might want to generate "u0 = outs[0].stride(1)"", where s = u0, and the keypath
        # describes the structure of "outs[0].stride(1)", like [SequenceKey(0), CallMethodKey("stride"), SequenceKey[1]].
        for s, keypath in unbacked_bindings.items():
            # `go` recursively constructs a code expression by processing each element of
            # the keypath and construct the expression incrementally.
            # For example, given output name outs and keypath [SequenceKey(0), CallMethodKey("stride", 1)],
            # it generates "outs[0]" based on SequenceKey(0), then recursively go("outs[0]", [CallMethodKey("stride"), ...])
            def go(expr: str, keypath: pytree.KeyPath):
                # [FUNCTION DOC] HELPER: Recursively builds the expression string from a base expression and a key path; raises an assertion error for unrecognized keys.
                # KNOWN ISSUE: the divide-by branch does not assert divisibility (acknowledged by a TODO in the code).
                if keypath == ():
                    return expr

                if (
                    len(keypath) >= 2
                    and isinstance(keypath[0], CallMethodKey)
                    and isinstance(keypath[1], pytree.SequenceKey)
                ):
                    return go(
                        f"{expr}.{keypath[0].name}({keypath[1].idx})", keypath[2:]
                    )
                elif isinstance(keypath[0], CallMethodKey):
                    return go(f"{expr}.{keypath[0].name}()", keypath[1:])
                elif isinstance(keypath[0], pytree.SequenceKey):
                    return (
                        go(f"std::get<{keypath[0].idx}>({expr})", keypath[1:])
                        if V.graph.cpp_wrapper
                        else go(f"{expr}[{keypath[0].idx}]", keypath[1:])
                    )
                elif isinstance(keypath[0], DivideByKey):
                    # TODO: need to assert divisibility
                    if V.graph.cpp_wrapper:
                        return go(f"({expr} / {keypath[0].divisor})", keypath[1:])
                    return go(f"{expr}.__floordiv__({keypath[0].divisor})", keypath[1:])
                else:
                    raise AssertionError(f"unrecognized keypath {keypath}")

            # `go_outer` manages the top-level logic for generating the final expression.
            # It handles special cases for C++ code generation and adjusts
            # the keypath based on the context (e.g., single vs. multiple outputs).
            def go_outer():  # type: ignore[no-untyped-def]
                # [FUNCTION DOC] HELPER: Chooses the starting expression for the keypath walk: for cpp_wrapper uses the single output name (dropping the first key for a single-element multi output) or the output selected by the first sequence key; otherwise uses output_name.
                if V.graph.cpp_wrapper:
                    # Special handling for the top level buffer access,
                    # because self.get_name() is actually never bound; the
                    # individual output arguments are bound by
                    # generate_c_shim_fallback_kernel
                    if len(outputs) == 1:
                        out = outputs[0]
                        # When fallback kernel returns a list consisting of a single tensor,
                        # the output is represented as a MultiOutput with non empty indices.
                        # In this case, we strip the first key path away.
                        return go(
                            outputs[0].get_name(),
                            keypath[1:]
                            if isinstance(out, ir.MultiOutput) and len(out.indices) != 0
                            else keypath,
                        )
                    else:
                        if not isinstance(keypath[0], pytree.SequenceKey):
                            raise AssertionError(
                                f"expected SequenceKey, got {type(keypath[0])}"
                            )
                        return go(outputs[keypath[0].idx].get_name(), keypath[1:])
                else:
                    return go(output_name, keypath)

            self.writeline(
                f"{self.codegen_unbacked_symbol_decl(s)} = {go_outer()}{self.ending}"
            )

    def codegen_subgraph_by_inlining(self, subgraph, outer_inputs, outer_outputs):
        # [FUNCTION DOC] Inlines a subgraph into the output code (older approach, still used for AOT mode and the first cpp wrapper pass): pushes the subgraph as the codegened graph, applies any config patches of the subgraph, writes a comment, assigns outer inputs to the inner inputs, runs the subgraph codegen under its graph handler, assigns inner outputs to outer outputs, and always pops the graph. Raises assertion errors on mismatched counts.
        # TODO (desertfire) - This function is the old way of supporting
        # subgraph codegen by inlining subgraphs in the output code. For python
        # wrapper, we have moved to lifting subgraphs as functions, supported by
        # `codegen_subgraph` function.
        #
        # However this does not work with cpp wrapper. With cpp wrapper, we make
        # two passes and the kernels are shared from the first pass to the next.
        # Therefore, both the Python and CppWrapper need to share the some
        # codegen infra. For now, CppWrapperCpu has not been updated to lift the
        # subgraph as functions. Therefore for cpp_wrapper first pass with
        # PythonWrapper, we still fallback to the old way of inlining subgraphs
        # in the output code. Once we update CppWrapperCpu, we can remove this
        # function.
        def _codegen_subgraph_prefix():
            # [FUNCTION DOC] HELPER: Writes assignments of outer inputs to the subgraph graph inputs; raises an assertion error if counts differ.
            if len(subgraph.graph.graph_inputs) != len(outer_inputs):
                raise AssertionError(
                    f"expected matching input counts, got {len(subgraph.graph.graph_inputs)} != {len(outer_inputs)}"
                )
            for inner_input, outer_input in zip(
                subgraph.graph.graph_inputs, outer_inputs
            ):
                self.writeline(
                    f"{self.declare}{inner_input} = {outer_input}{self.ending}"
                )

        def _codegen_subgraph_suffix():
            # [FUNCTION DOC] HELPER: Writes assignments of subgraph outputs to outer outputs; raises an assertion error if counts differ.
            if len(subgraph.graph.graph_outputs) != len(outer_outputs):
                raise AssertionError(
                    f"expected matching output counts, got {len(subgraph.graph.graph_outputs)} != {len(outer_outputs)}"
                )
            for inner_output, outer_output in zip(
                subgraph.graph.graph_outputs, outer_outputs
            ):
                self.writeline(
                    f"{outer_output} = {inner_output.codegen_reference()}{self.ending}"
                )

        try:
            self.push_codegened_graph(subgraph.graph)
            # Only ir.Subgraph (invoke_subgraph regions) carries nested config
            # patches; other subgraph adapters (e.g. the CodegenGraph used for
            # decompose_k) have none.
            inductor_config_patches = getattr(subgraph, "inductor_config_patches", None)
            ctx = (
                config.patch(inductor_config_patches)
                if inductor_config_patches
                else contextlib.nullcontext()
            )
            with ctx:
                self.writeline(f"{self.comment} subgraph: {subgraph.name}")
                _codegen_subgraph_prefix()
                parent_graph = V.graph
                with V.set_graph_handler(subgraph.graph):
                    subgraph.graph.codegen_subgraph(
                        parent_graph=parent_graph,
                    )
                _codegen_subgraph_suffix()
        finally:
            self.pop_codegened_graph()

    def codegen_partition_call(
        self,
        partition_id: int,
        partition_signatures: ir.GraphPartitionSignature,
    ):
        # [FUNCTION DOC] Writes the call of a graph partition: builds the args list from the partition inputs and symbol inputs, deletes inputs flagged for deallocation, calls the partition via the runner's partitions list with the args, binds the outputs, and deletes the args list.
        """Generate code to call a graph partition"""
        input_deallocation = partition_signatures.input_deallocation
        output_nodes = partition_signatures.output_nodes

        input_names = list(input_deallocation.keys()) + [
            symbol_input.name for symbol_input in partition_signatures.symbol_inputs
        ]

        inputs = ", ".join(input_names) + ("," if len(input_names) == 1 else "")

        output_names = [node.get_name() for node in output_nodes]
        outputs = ", ".join(output_names) + ("," if len(output_nodes) == 1 else "")

        # Create a list of inputs for the subgraph call
        self.writeline(f"partition{partition_id}_args = [{inputs}]")

        names_to_del = [
            name for name, deallocate in input_deallocation.items() if deallocate
        ]
        if names_to_del:
            self.writeline(f"del {', '.join(names_to_del)}")

        # Call the subgraph launcher function
        self.writeline(
            f"({outputs}) = self.partitions[{partition_id}](partition{partition_id}_args)"
        )
        self.writeline(f"del partition{partition_id}_args")

    def get_partition_name(self, partition_id: int) -> str:
        # [FUNCTION DOC] Returns the name used for a partition by id.
        return f"partition_{partition_id}"

    def set_all_partition_names(self, num_partitions: int):
        # [FUNCTION DOC] Stores the list of all partition names for the given number of partitions.
        self.all_partition_names = list(
            map(self.get_partition_name, range(num_partitions))
        )

    def codegen_subgraph_call_with_flattened_outputs(
        self, subgraph, outer_inputs, outer_flattened_outputs
    ):
        # [FUNCTION DOC] Writes code that packs outer inputs into an args list and calls the subgraph launcher function, unpacking the result into the flattened output names (trailing commas for single names).
        # Get the input and output names of the subgraph
        outer_output_names = ", ".join(outer_flattened_outputs) + (
            "," if len(outer_flattened_outputs) == 1 else ""
        )
        outer_input_names = ", ".join(outer_inputs) + (
            "," if len(outer_inputs) == 1 else ""
        )

        self.writeline(f"{subgraph.graph.name}_args = [{outer_input_names}]")

        # Call the subgraph launcher function
        self.writeline(
            f"({outer_output_names}) = {subgraph.graph.name}({subgraph.graph.name}_args)"
        )

    def codegen_subgraph_call(self, subgraph, outer_inputs, outer_buffer_name):
        # [FUNCTION DOC] Writes the args list for the subgraph call, frees buffers whose last use was as args (scheduler free), then writes the assignment of the subgraph launcher call result to the outer buffer name.
        # Get the input and output names of the subgraph
        outer_input_names = ", ".join(outer_inputs) + (
            "," if len(outer_inputs) == 1 else ""
        )

        self.writeline(f"{subgraph.graph.name}_args = [{outer_input_names}]")

        # Since the buffers are already put into the args list, we can free the
        # buffers here.
        V.graph.scheduler.free_buffers()

        # Call the subgraph launcher function
        self.writeline(
            f"{outer_buffer_name} = {subgraph.graph.name}({subgraph.graph.name}_args)"
        )

    def codegen_subgraph_common(self, subgraph):
        # [FUNCTION DOC] Common part of lifting a subgraph into a function: pushes the graph, writes comments, copies cpp_wrapper and fx_wrapper flags from the parent graph, and, if not already done, codegens the subgraph recursively (with its config patches, graph partition disabled) and defines its launcher function; records the subgraph as codegened. Does not pop the graph pushed at the start.
        self.push_codegened_graph(subgraph.graph)
        self.make_comment("")
        self.make_comment(f"{self.comment} subgraph: {subgraph.name}")

        parent_graph = V.graph
        subgraph.graph.cpp_wrapper = parent_graph.cpp_wrapper
        subgraph.graph.fx_wrapper = parent_graph.fx_wrapper

        if subgraph.graph.name not in self.already_codegened_subgraphs:
            # If it is already codegened, the parent wrapper already has
            # subgraph fn by name subgraph.graph.name
            # Only ir.Subgraph (invoke_subgraph regions) carries nested config
            # patches; other subgraph adapters (e.g. the CodegenGraph used for
            # decompose_k) have none.
            inductor_config_patches = getattr(subgraph, "inductor_config_patches", None)
            ctx = (
                config.patch(inductor_config_patches)
                if inductor_config_patches
                else contextlib.nullcontext()
            )
            # do not graph partition inside subgraph bodies
            with (
                ctx,
                config.patch("graph_partition", False),
                V.set_graph_handler(subgraph.graph),
            ):
                # Call the codegen of subgraph recursively
                subgraph_code, _ = subgraph.graph.codegen()
            subgraph_name = subgraph.graph.name
            self.already_codegened_subgraphs.add(subgraph_name)
            self.define_subgraph_launcher_fn(subgraph_name, subgraph_code)

    def codegen_subgraph_with_flattened_outputs(
        self, subgraph, outer_inputs, outer_flattened_outputs
    ):
        # [FUNCTION DOC] Lifts the subgraph into a function and writes the call with flattened outputs.
        self.codegen_subgraph_common(subgraph)
        self.codegen_subgraph_call_with_flattened_outputs(
            subgraph, outer_inputs, outer_flattened_outputs
        )

    def codegen_subgraph(self, subgraph, outer_inputs, outer_buffer_name):
        # [FUNCTION DOC] Lifts the subgraph into a function and writes the call assigning the result to the outer buffer.
        # Codegen subgraph by recursively calling the codegen for the subgraph.
        # This lifts the subgraph as a function in the output code.
        self.codegen_subgraph_common(subgraph)
        self.codegen_subgraph_call(subgraph, outer_inputs, outer_buffer_name)

    def codegen_invoke_subgraph(self, invoke_subgraph):
        # [FUNCTION DOC] Generates code for an invoke_subgraph node: defines a result list of the output count and either inlines the subgraph (AOT mode) with outputs as list elements, or lifts it as a function call.
        name = invoke_subgraph.get_name()

        self.writeline(f"{name} = [None] * {len(invoke_subgraph.outputs)}")
        outer_inputs = [buf.codegen_reference() for buf in invoke_subgraph.inputs]

        if V.graph.aot_mode:
            outer_outputs = [
                f"{name}[{i}]" for i in range(len(invoke_subgraph.outputs))
            ]
            self.codegen_subgraph_by_inlining(
                invoke_subgraph.subgraph, outer_inputs, outer_outputs
            )
        else:
            self.codegen_subgraph(invoke_subgraph.subgraph, outer_inputs, name)

    def codegen_switch(self, node) -> None:
        # [FUNCTION DOC] Generates a host-side conditional for a switch node: evaluates the selector into an int variable (tensor selectors via item, shape expressions directly), defines the list of outputs, then emits if/elif/else branches (first branch index zero, last branch as else) each wrapped in enter/exit subgraph lines and filled by inlining (AOT) or by a function call.
        name = node.get_name()

        outer_inputs = [buf.codegen_reference() for buf in node.operands]

        selector_ref = node.selector.codegen_reference()
        # Evaluate the selector once into a named local.
        selector_var = f"{name}_selector"
        if not isinstance(node.selector, ir.ShapeAsConstantBuffer):
            # move the Tensor selector to host; always as int so the branch loop
            # can use uniform index comparisons (cond: True==1, False==0)
            self.writeline(f"{selector_var} = int({selector_ref}.item())")
        else:
            # ShapeAsConstantBuffer yields a Python bool/int expression; wrap in
            # int() so "== <idx>" comparisons don't form a Python chained comparison.
            self.writeline(f"{selector_var} = int({selector_ref})")

        self.writeline(f"{name} = [None] * {len(node.outputs)}")

        def _emit_branch(keyword: str, condition: str, branch) -> None:
            # [FUNCTION DOC] HELPER: Writes one branch header with its keyword and condition, an enter subgraph line, the branch body, and an exit subgraph line.
            self.writeline(f"{keyword}{condition}:")
            self.writeline(EnterSubgraphLine(self, branch.graph))
            if V.graph.aot_mode:
                outer_outputs = [f"{name}[{i}]" for i in range(len(node.outputs))]
                self.codegen_subgraph_by_inlining(branch, outer_inputs, outer_outputs)
            else:
                self.codegen_subgraph(branch, outer_inputs, name)
            self.writeline(ExitSubgraphLine(self))

        num_branches = len(node.branches)
        for b_idx, branch in enumerate(node.branches):
            if b_idx == 0:
                keyword, condition = "if", f" {selector_var} == 0"
            elif b_idx < num_branches - 1:
                keyword, condition = "elif", f" {selector_var} == {b_idx}"
            else:
                keyword, condition = "else", ""
            _emit_branch(keyword, condition, branch)

    def codegen_while_loop(self, while_loop, stack_output):
        # [FUNCTION DOC] Generates a host-side while loop. Initializes the carried state list (with extra lists for stacked outputs when stack_output), evaluates the condition subgraph before the loop (if false copies carried inputs, unsqueezed when stacking), then loops running the body subgraph, optionally appending states to the stacked lists, re-evaluates the condition at the end of the body, and after the loop stacks the collected lists.
        # KNOWN ISSUE: the re-evaluated loop condition assignment is written with a hardcoded leading four-space indent in the string rather than through the indent mechanism.
        """while_loop is codegened as a host side while_loop"""

        def codegen_subgraph(subgraph, outer_inputs, outer_outputs):
            # [FUNCTION DOC] HELPER: Dispatches to inlining (AOT mode) or the flattened-outputs function call for a subgraph.
            """Helper method to deduplicate subgraph codegen logic"""
            if V.graph.aot_mode:
                self.codegen_subgraph_by_inlining(subgraph, outer_inputs, outer_outputs)
            else:
                self.codegen_subgraph_with_flattened_outputs(
                    subgraph, outer_inputs, outer_outputs
                )

        name = while_loop.get_name()
        outer_carried_inputs = [
            buf.codegen_reference() for buf in while_loop.carried_inputs
        ]
        outer_additional_inputs = [
            buf.codegen_reference() for buf in while_loop.additional_inputs
        ]

        ckp_offset = len(outer_carried_inputs)
        self.writeline(f"{name} = [None] * {len(outer_carried_inputs)}")
        if stack_output:
            self.writeline(
                f"{name}.extend([[] for _ in range({len(outer_carried_inputs)})])"
            )

        for i, inp in enumerate(outer_carried_inputs):
            # set the initial state before the loop
            self.writeline(f"{name}[{i}] = {inp}")

        cond_outer_inputs = [
            *[f"{name}[{i}]" for i in range(len(outer_carried_inputs))],
            *outer_additional_inputs,
        ]
        cond_outer_outputs = [f"{name}_cond_result"]
        body_outer_inputs = list(
            cond_outer_inputs
        )  # same inputs for cond_fn and body_fn
        # Carry over the state from body_fn. Note: We only carry over
        # the carried_inputs part of the inputs, the additional ones
        # are passed in as they're before.
        body_outer_outputs = body_outer_inputs[: len(outer_carried_inputs)]
        # Check condition at the beginning and set up flag
        codegen_subgraph(
            while_loop.cond_subgraph, cond_outer_inputs, cond_outer_outputs
        )
        self.writeline(f"should_loop = {cond_outer_outputs[0]}")
        self.writeline("if not should_loop:")
        if stack_output:
            # Handle the case when loop never executes
            for i, carried_input in enumerate(outer_carried_inputs):
                self.writeline(EnterSubgraphLine(self, while_loop.body_subgraph.graph))
                self.writeline(f"{name}[{i}] = {carried_input}.unsqueeze(0).clone()")
                self.writeline(ExitSubgraphLine(self))
        else:
            for i, carried_input in enumerate(outer_carried_inputs):
                self.writeline(EnterSubgraphLine(self, while_loop.body_subgraph.graph))
                self.writeline(f"{name}[{i}] = {carried_input}.clone()")
                self.writeline(ExitSubgraphLine(self))

        self.writeline("while should_loop:")
        # Body execution
        self.writeline(EnterSubgraphLine(self, while_loop.body_subgraph.graph))
        codegen_subgraph(
            while_loop.body_subgraph, body_outer_inputs, body_outer_outputs
        )
        self.writeline(ExitSubgraphLine(self))

        # Collect outputs if enabled
        if stack_output:
            self.writeline(EnterSubgraphLine(self, while_loop.body_subgraph.graph))
            for i in range(len(outer_carried_inputs)):
                self.writeline(f"{name}[{i + ckp_offset}].append({name}[{i}])")
            self.writeline(ExitSubgraphLine(self))

        # Condition check at end of loop
        self.writeline(EnterSubgraphLine(self, while_loop.cond_subgraph.graph))
        codegen_subgraph(
            while_loop.cond_subgraph, cond_outer_inputs, cond_outer_outputs
        )
        self.writeline(ExitSubgraphLine(self))
        self.writeline(f"    should_loop = {cond_outer_outputs[0]}")

        # Stack outputs after loop completion
        if stack_output:
            self.writeline("# Stack outputs after loop completion")
            for i in range(len(outer_carried_inputs)):
                self.writeline(f"if len({name}[{i + ckp_offset}]) > 0:")
                self.writeline(EnterSubgraphLine(self, while_loop.body_subgraph.graph))
                self.writeline(
                    f"{name}[{i}] = torch.stack({name}[{i + ckp_offset}], dim=0)"
                )
                self.writeline(ExitSubgraphLine(self))

    @staticmethod
    def statically_known_int_or_none(x):
        # [FUNCTION DOC] Returns the integer value of x if it is statically known, else None. Returns None for expressions with free symbols, an int as is, otherwise tries static evaluation in the shape environment; any exception yields None.
        try:
            if getattr(x, "free_symbols", None):
                # _maybe_evaluate_static will return (s0 // (2 // s0)) as 2, but
                # the actual codegen will still generate the full expression here.
                return None
            if isinstance(x, int):
                return x
            val = V.graph._shape_env._maybe_evaluate_static(x)
            if val is None:
                return val
            return int(val)  # type: ignore[call-overload]
        except Exception:
            return None

    @staticmethod
    def statically_known_list_of_ints_or_none(lst):
        # [FUNCTION DOC] Returns the list of statically known ints for the input list, or None if any element is not statically known.
        result = []
        for x in lst:
            num = PythonWrapperCodegen.statically_known_int_or_none(x)
            if num is None:
                return None
            result.append(num)
        return result

    @staticmethod
    def is_statically_known_list_of_ints(lst):
        # [FUNCTION DOC] True if every element of the list is statically known as an int.
        return (
            PythonWrapperCodegen.statically_known_list_of_ints_or_none(lst) is not None
        )

    @staticmethod
    def static_shape_for_buffer_or_none(buffer):
        # [FUNCTION DOC] Returns the list of static sizes of the buffer, or None if any size is not statically known.
        return PythonWrapperCodegen.statically_known_list_of_ints_or_none(
            buffer.get_size()
        )

    @staticmethod
    def can_prove_buffer_has_static_shape(buffer):
        # [FUNCTION DOC] True if the buffer has a fully static shape.
        return PythonWrapperCodegen.static_shape_for_buffer_or_none(buffer) is not None

    def write_kernel_context_guard(
        self,
        kernel_name: str,
        node_schedule: Sequence[BaseSchedulerNode] | ExternKernel,
    ):
        # [FUNCTION DOC] Hook marking a kernel context guard for a kernel; does nothing in the python wrapper.
        return

    def write_kernel_context_guard_begin(
        self,
    ):
        # [FUNCTION DOC] Hook marking the beginning of a kernel context guard; does nothing here.
        """
        Mark the beginning of kernel context guard
        """
        return

    def write_kernel_context_guard_end(
        self,
    ):
        # [FUNCTION DOC] Hook marking the end of a kernel context guard; does nothing here.
        """
        Mark the end of kernel context guard
        """
        return


class SubgraphPythonWrapperCodegen(PythonWrapperCodegen):
    # [CLASS DOC] Wrapper codegen for a subgraph (including graph partitions). Holds the subgraph name, parent wrapper and optional partition signatures. Reuses most behavior from PythonWrapperCodegen but suppresses duplicated pieces (header, async compile wait, benchmark harness, suffix), shares the root wrapper autotune buffers, kernel source cache and user-defined kernel cache, takes inputs/outputs from the partition signature when present, and delegates triton header writing to the parent.
    """
    A wrapper codegen that generates code for a subgraph. For most of the
    methods, we rely on the implementation in the PythonWrapperCodegen. But we
    override a few functions to produce cleaner code (like avoiding writing
    imports twice in the output code)
    """

    def __init__(
        self,
        subgraph_name: str,
        parent_wrapper: PythonWrapperCodegen,
        partition_signatures: ir.GraphPartitionSignature | None = None,
    ):
        # [FUNCTION DOC] Stores the subgraph name, parent wrapper and partition signatures before calling the base initializer (which needs the name for the launcher function), then shares the root wrapper autotune defs/calls, kernel source mapping and user-defined kernel cache.
        # It is necessary to set the subgraph_name before calling super __init__
        # because __init__ calls set_launcher_fn_name
        self.subgraph_name = subgraph_name
        self.parent_wrapper = parent_wrapper
        self.partition_signatures = partition_signatures

        super().__init__()

        root = self.get_root_graph()
        # Only generate auto-tuning block in the main graph
        self.kernel_autotune_defs = root.kernel_autotune_defs
        self.kernel_autotune_calls = root.kernel_autotune_calls
        # Only store kernel src to name mapping in the main graph
        self.src_to_kernel = root.src_to_kernel
        # Same here, only define user-defined Triton kernels in the main graph
        self.user_defined_kernel_cache = root.user_defined_kernel_cache

    def set_launcher_fn_name(self) -> None:
        # [FUNCTION DOC] Sets the launcher function name to the subgraph name.
        # This sets up the name of the function containing the launcher code of
        # the subgraph.
        # pyrefly: ignore [bad-assignment]
        self.launcher_fn_name = self.subgraph_name

    def write_header(self) -> None:
        # [FUNCTION DOC] Writes no header for a subgraph.
        pass

    def add_benchmark_harness(self, output):
        # [FUNCTION DOC] Does nothing: no benchmark harness for subgraphs.
        pass

    def benchmark_compiled_module(self, output):
        # [FUNCTION DOC] Does nothing for subgraphs.
        pass

    def write_async_compile_wait(self):
        # [FUNCTION DOC] Does nothing: only the main graph waits for async compile.
        pass

    def next_kernel_suffix(self) -> str:
        # [FUNCTION DOC] Returns the next kernel suffix from the parent wrapper so subgraph kernel names do not clash.
        # Ensures that subgraphs kernels do not clash with each other
        return self.parent_wrapper.next_kernel_suffix()

    def generate_after_suffix(self, result: IndentedBuffer) -> None:
        # [FUNCTION DOC] Does nothing for subgraphs (no runner creation).
        return

    def write_launcher_fn_call_get_indent(self) -> int:
        # [FUNCTION DOC] Writes the definition of the subgraph launcher function taking args into the prefix and returns an indent of 1.
        self.prefix.splice(
            f"""
            def {self.launcher_fn_name}(args):
            """
        )
        prefix_indent = 1
        return prefix_indent

    def get_wrapper_call_indent(self) -> int:
        # [FUNCTION DOC] Returns the call body indent for subgraphs: always 1.
        return 1

    def get_graph_inputs(
        self,
    ) -> dict[str, ir.TensorBox | ir.TorchBindObject | sympy.Expr | None]:
        # [FUNCTION DOC] Returns the graph inputs: from the partition signature (input nodes plus symbol inputs) if present, else from the current graph.
        if signature := self.partition_signatures:
            inputs = signature.input_nodes | {
                str(s): s for s in signature.symbol_inputs
            }
        else:
            inputs = V.graph.graph_inputs
        return inputs

    def get_graph_input_names(self) -> list[str]:
        # [FUNCTION DOC] Returns the input names: from the partition signature (input nodes then symbol inputs) if present, else from the current graph.
        if signature := self.partition_signatures:
            names = list(signature.input_nodes.keys()) + [
                symbol_input.name for symbol_input in signature.symbol_inputs
            ]
        else:
            names = V.graph.graph_input_names
        return names

    def get_graph_outputs(self) -> list[IRNode]:
        # [FUNCTION DOC] Returns the output nodes from the partition signature if present, else from the current graph.
        if signature := self.partition_signatures:
            outputs = signature.output_nodes
        else:
            outputs = V.graph.graph_outputs
        return outputs

    def codegen_allocation(self, buffer: ir.Buffer):
        # [FUNCTION DOC] Skips allocation for buffers that are inputs of the partition signature; otherwise defers to the base class.
        name = buffer.get_name()
        if (signature := self.partition_signatures) and name in signature.input_nodes:
            # skip allocation if buffer is a subgraph input.
            # This allows reusing an input buffer in graph partition,
            # although this is not allowed in general.
            return

        super().codegen_allocation(buffer)

    def _write_get_raw_stream(
        self, device_idx: int, graph: GraphLowering | None = None
    ) -> str:
        # [FUNCTION DOC] Writes a raw stream variable for the device: ensures the triton header, then uses the scheduler current stream handle if a stream context is active, else a get_raw_stream call for the device. Returns the variable name.
        # KNOWN ISSUE: the base class calls write_get_raw_stream (not this underscore name) and the graph argument is unused, so this method appears to be dead code.
        self.write_triton_header_once()
        # pyre-fixme[16]: scheduler.current_stream_name added in scheduler commit
        if (current_stream_name := V.graph.scheduler.current_stream_name) is not None:
            name = f"{current_stream_name}_raw"
            self.writeline(
                f"{name} = {V.graph.device_ops.stream_handle(current_stream_name)}"
            )
        else:
            name = get_raw_stream_name(device_idx)
            self.writeline(
                f"{name} = get_raw_stream({_coor_device_idx_ref(device_idx)})"
            )
        return name

    def codegen_graph_nvtx_range_push(self, post_grad_graph_id: int) -> None:
        # [FUNCTION DOC] Writes an NVTX range push for the graph id.
        """Generate NVTX range push for graph."""
        self.writeline(f"torch.cuda.nvtx.range_push('graph {post_grad_graph_id}')")

    def codegen_graph_nvtx_range_pop(self) -> None:
        # [FUNCTION DOC] Writes an NVTX range pop.
        """Generate NVTX range pop for graph."""
        self.writeline("torch.cuda.nvtx.range_pop()")

    @cache_on_self
    def write_triton_header_once(self) -> None:
        # [FUNCTION DOC] Once (cached): delegates to the parent wrapper to write the triton header.
        # TODO: Uncomment in future. This will be needed to support subgraph
        # codegen for cpp wrapper.
        # if config.triton.autotune_at_compile_time:
        #     import_str = self.triton_header_str()
        #     self.kernel_autotune_calls.splice(import_str)
        self.parent_wrapper.write_triton_header_once()

    @cache_on_self
    def write_get_raw_stream_header_once(self) -> None:
        # [FUNCTION DOC] Once (cached): delegates to the parent wrapper to write the raw stream import header.
        # TODO: Uncomment in future. This will be needed to support subgraph
        # codegen for cpp wrapper.
        # if config.triton.autotune_at_compile_time:
        #     self.kernel_autotune_calls.writeline(
        #         V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
        #     )
        self.parent_wrapper.write_get_raw_stream_header_once()

    @cache_on_self
    def get_root_graph(self) -> PythonWrapperCodegen:
        # [FUNCTION DOC] Returns (cached) the root PythonWrapperCodegen by following parent wrappers; raises an assertion error if the chain does not end in one.
        root: PythonWrapperCodegen | SubgraphPythonWrapperCodegen = self
        while isinstance(root, SubgraphPythonWrapperCodegen):
            root = root.parent_wrapper

        if not isinstance(root, PythonWrapperCodegen):
            raise AssertionError(f"expected PythonWrapperCodegen, got {type(root)}")
        return root

    def generate_and_run_autotune_block(self):
        # [FUNCTION DOC] Does nothing: the autotune block is only executed in the main graph.
        # Only execute auto-tuning block in the main graph
        pass
