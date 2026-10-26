# [MODULE DOC] Converts Inductor wrapper IR (the list of wrapper Line objects) into a torch.fx GraphModule.
# Main contents: SymbolBuffer and TritonKernel helper dataclasses, replace_floor_div, WrapperFxCodegen (a
# PythonWrapperCodegen backend that emits FX instead of Python text) and FxConverter (does the line-by-line conversion).
# Only Triton kernels, extern kernels and fallbacks are supported. Import-time side effects: defines the aten alias and a logger.
import dataclasses
import functools
import logging
import operator
import textwrap
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import sympy

import torch
from torch._export.passes._node_metadata_hook import (
    _node_metadata_hook,
    _set_node_metadata_hook,
)
from torch._higher_order_ops.triton_kernel_wrap import (
    TraceableTritonKernelWrapper,
    tracing_triton_hopifier_singleton,
    triton_kernel_wrapper_mutation,
)
from torch._inductor.codecache import LambdaFuture, PyCodeCache
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch._inductor.select_algorithm import extern_kernels  # noqa: F401
from torch._inductor.utils import convert_to_symint
from torch._inductor.virtualized import V
from torch._library.triton import wrap_triton
from torch.fx import GraphModule
from torch.fx.experimental.symbolic_shapes import (
    CallMethodKey,
    ConvertIntKey,
    DivideByKey,
)
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.interp import _run_sympy_handler, sympy_interp
from torch.utils._sympy.reference import OptimizedPythonReferenceAnalysis
from torch.utils._sympy.solve import try_solve

from .. import config, ir
from ..runtime.triton_compat import Config
from ..utils import cache_property_on_self, LineContext, ValueWithLineMap
from .common import (
    CodegenSymbol,
    FileBackedGraphModule,
    WorkspaceArg,
    WorkspaceZeroMode,
)
from .wrapper import (
    AllocateLine,
    BufferLike,
    CommentLine,
    DynamicScalarLine,
    EnterDeviceContextManagerLine,
    EnterSubgraphLine,
    ExitDeviceContextManagerLine,
    ExitSubgraphLine,
    ExternKernelAllocLine,
    ExternKernelOutLine,
    FreeIfNotReusedLine,
    FreeLine,
    IndexPutFallbackLine,
    KernelCallLine,
    KernelDefinitionLine,
    Line,
    MultiOutputLine,
    NullLine,
    PythonWrapperCodegen,
    ReinterpretLine,
    ReuseLine,
    ScatterFallbackLine,
    SubgraphPythonWrapperCodegen,
    SwitchLine,
    SymbolicCallArg,
    SymbolicCallArgLine,
    UnbackedSymbolDefsLine,
    WrapperLine,
)


aten = torch.ops.aten
log = logging.getLogger(__name__)


@dataclasses.dataclass
class SymbolBuffer(CodegenSymbol):
    # [CLASS DOC] A named symbolic value (graph input or unbacked symbol) that can sit in the converter symbol table like a buffer.
    # Fields: expr (sympy expression) and optional name; a name is mandatory unless expr is a plain symbol.
    """
    Represents a symbolic graph input. Expressions more complex than a single
    sympy.Symbol require a name.
    """

    expr: sympy.Expr
    name: str | None = None

    def get_name(self) -> str:
        # [FUNCTION DOC] Returns the explicit name if set, otherwise the string form of the symbol.
        # Raises AssertionError if no name is set and the expression is not a single sympy symbol.
        if self.name is not None:
            return self.name
        if not isinstance(self.expr, sympy.Symbol):
            raise AssertionError(f"expression requires a name: {self.expr}")
        return str(self.expr)

    def get_example(self) -> torch.Tensor | torch.SymInt:
        # [FUNCTION DOC] Returns a SymInt example value built from the expression (for node metadata).
        # Raises AssertionError if the conversion does not yield a SymInt.
        sym_int = convert_to_symint(self.expr)
        if not isinstance(sym_int, torch.SymInt):
            raise AssertionError(f"expected torch.SymInt, got {type(sym_int)}")
        return sym_int


CodegenBuffer = BufferLike | SymbolBuffer


@dataclasses.dataclass
class TritonKernel:
    # [CLASS DOC] Pairs a CachingAutotuner (the compiled/tunable Triton kernel) with its traceable wrapper
    # so the converter can emit calls to the Triton higher-order op. Plain data holder.
    """
    Stores metadata about Triton kernels for use in FX.
    """

    tuner: CachingAutotuner
    wrapped: TraceableTritonKernelWrapper


def replace_floor_div(expr: sympy.Expr) -> sympy.Expr:
    # [FUNCTION DOC] Rewrite every floor of a sympy expression into the FloorDiv function so it can be traced to FX.
    # Combines the argument over a common denominator, splits multiplicative factors into numerator and denominator
    # (rational parts and negative-power terms go to the denominator), and builds a FloorDiv of the two.
    # Returns a new expression; pure.
    """
    Replace sympy.floor with FloorDiv.
    """

    def replace(expr: sympy.Expr) -> sympy.Expr:
        # [FUNCTION DOC] HELPER: Replacement callback for one floor term; returns the FloorDiv of the split numerator and denominator.
        expr = sympy.together(expr)

        # Division is represented as a Mul with a Rational factor or a Pow with negative
        # exponent. We convert floor(Mul(...)) to FloorDiv(numerator, denominator) by
        # partitioning factors into the numerator and denominator.
        (numerator, denominator) = (sympy.S.One,) * 2
        for arg in sympy.Mul.make_args(expr):
            if isinstance(arg, sympy.Rational):
                numerator *= arg.numerator
                denominator *= arg.denominator
            elif isinstance(arg, sympy.Pow) and arg.exp.is_negative:
                denominator *= arg.base**-arg.exp
            else:
                numerator *= arg

        return FloorDiv(numerator, denominator)

    return expr.replace(sympy.floor, replace)


class WrapperFxCodegen(PythonWrapperCodegen):
    # [CLASS DOC] Wrapper codegen backend that produces an FX GraphModule instead of Python source.
    # Keeps a mapping of generated subgraph modules by name. supports_caching is False. Key methods: _generate (runs wrapper
    # passes then FxConverter), compile_graph (override hook to compile the produced module), create (factory, with a
    # subgraph variant).
    """
    Backend to generate wrapper code as an FX IR graph.
    """

    supports_caching = False

    def __init__(self, *args: Any, **kwargs: Any):
        # [FUNCTION DOC] Initialize the base wrapper codegen with all arguments and create an empty mapping of subgraph modules.
        super().__init__(*args, **kwargs)
        self.subgms: dict[str, torch.fx.GraphModule] = {}

    def codegen_inputs(self) -> None:
        # [FUNCTION DOC] Intentionally does nothing: symbolic input shapes and strides are handled later by the FX converter.
        """
        This would generate code for symbolic input shapes, strides, etc.
        Since the FX converter handles this, do nothing here.
        """

    def codegen_switch(self, node: ir.Switch) -> None:
        # [FUNCTION DOC] Emit a single switch line for the whole node (instead of several lines), then run the common codegen
        # for every branch subgraph of the node. Side effect: appends to the wrapper lines and generates subgraphs.
        """
        Switch/cond codegen normally emits a number of different wrapper lines.
        Instead, FX conversion uses a dedicated line for the whole node.
        """
        self.writeline(SwitchLine(self, node))
        for subgraph in node.branches:  # pyrefly: ignore [not-iterable]
            self.codegen_subgraph_common(subgraph)

    def define_subgraph_launcher_fn(
        self, name: str, subgraph_code: ValueWithLineMap | FileBackedGraphModule
    ) -> None:
        # [FUNCTION DOC] Record the FX module of a generated subgraph under its name. Raises AssertionError if the code
        # is not a FileBackedGraphModule.
        """
        Record subgms as they're generated.
        """
        if not isinstance(subgraph_code, FileBackedGraphModule):
            raise AssertionError(
                f"expected FileBackedGraphModule, got {type(subgraph_code)}"
            )
        self.subgms[name] = subgraph_code.gm

    @property
    @cache_property_on_self
    def is_subgraph(self) -> bool:
        # [FUNCTION DOC] Property (cached on the instance): True if this wrapper is a subgraph wrapper codegen.
        return isinstance(self, SubgraphPythonWrapperCodegen)

    def get_fx_graph_inputs(
        self,
    ) -> dict[str, ir.TensorBox | ir.TorchBindObject | sympy.Expr | None]:
        # [FUNCTION DOC] Return the mapping of FX placeholder names to input IR nodes (None for unknown inputs).
        # For top-level AOT compilation, follows the placeholders of the original module so signatures match; otherwise
        # returns the wrapper graph inputs.
        """
        Get the input nodes corresponding to FX graph placeholders.
        """

        if V.aot_compilation and not self.is_subgraph:
            # AOT graphs must match the signature of the input module.
            return {
                node.name: V.graph.graph_inputs.get(node.name)
                for node in V.graph.module.graph.find_nodes(op="placeholder")  # type: ignore[operator, union-attr]
            }

        return self.get_graph_inputs()

    def _generate(self, is_inference: bool) -> tuple[FileBackedGraphModule, None]:
        # [FUNCTION DOC] Run wrapper IR passes, build the prologue text from the imports and header buffers, convert lines
        # to an FX graph with FxConverter, compile it and return the file-backed module paired with None (no line map).
        # Args: is_inference flag forwarded to the passes.
        self.run_wrapper_ir_passes(is_inference)

        prologue = "\n".join(
            [
                self.imports.getvalue(),
                self.header.getvalue(),
            ]
        )
        gm = FxConverter(
            lines=self.lines,
            prologue=prologue,
            graph_inputs=self.get_fx_graph_inputs(),
            graph_outputs=self.get_graph_outputs(),
            subgms=self.subgms,
            is_subgraph=self.is_subgraph,
        ).generate()

        compiled_fn = self.compile_graph(gm)

        return FileBackedGraphModule(gm, compiled_fn), None

    def compile_graph(self, gm: GraphModule) -> Callable[..., Any]:
        # [FUNCTION DOC] Turn the graph module into a callable. Default: return its forward (eager interpreter). Subclasses may override.
        """
        Converts the graph module into a runnable function. The default implementation
        is simply an interpreter calling kernels in eager mode. Derived backends can
        override this to do further compilation.
        """
        return gm.forward

    def write_header(self) -> None:
        # [FUNCTION DOC] Force the standard Python wrapper header to be written, even for subgraphs that normally lack one.
        """
        Python subgraphs normally lack headers.
        Override this behavior to generate prologues for FX subgraphs.
        """
        PythonWrapperCodegen.write_header(self)

    def register_alignment_check_inputs(self) -> None:
        # [FUNCTION DOC] Intentionally does nothing: the FX backend does not register inputs for alignment checks.
        """FXIR does not emit deferred alignment copies.
        Alignment is handled by the runtime wrapper."""

    def codegen_deferred_alignment_copies(
        self, input_names: Iterable[str], stream: int = 0
    ) -> None:
        # [FUNCTION DOC] Intentionally does nothing: the FX backend does not emit deferred alignment copies.
        """FXIR does not emit deferred alignment copies."""

    @classmethod
    def create(
        cls: type["WrapperFxCodegen"],
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: PythonWrapperCodegen | None,
        partition_signatures: ir.GraphPartitionSignature | None = None,
    ) -> "WrapperFxCodegen":
        # [FUNCTION DOC] Factory classmethod. For subgraphs, builds (and returns an instance of) a dynamically defined class
        # mixing the given class with SubgraphPythonWrapperCodegen whose compile_graph returns a function that always raises
        # NotImplementedError. For top-level graphs returns a plain instance. Raises AssertionError if a subgraph lacks a
        # name or parent wrapper.
        if is_subgraph:
            if subgraph_name is None:
                raise AssertionError("subgraph_name must not be None for subgraphs")
            if parent_wrapper is None:
                raise AssertionError("parent_wrapper must not be None for subgraphs")

            # Subgraphs override some methods of PythonWrapperCodegen.
            # Apply these overrides to the user-provided class, with priority given to
            # user-provided methods.
            class SubgraphFxWrapperCodegen(cls, SubgraphPythonWrapperCodegen):  # type: ignore[misc,valid-type]
                # [CLASS DOC] HELPER: Subgraph variant of the FX wrapper; subgraph overrides apply, with user-class methods taking priority.
                def compile_graph(self, gm: GraphModule) -> Callable[..., Any]:
                    # [FUNCTION DOC] HELPER: Skips compilation for subgraphs and returns a function that always raises NotImplementedError when called.
                    """
                    Skip graph compilation for subgraphs.
                    """

                    def crash_if_run(*args: Any) -> None:
                        # [FUNCTION DOC] HELPER: Always raises NotImplementedError because a subgraph cannot run in isolation.
                        raise NotImplementedError("Cannot run a subgraph in isolation!")

                    return crash_if_run

            return SubgraphFxWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )

        return cls()


@dataclasses.dataclass
class FxConverter:
    # [CLASS DOC] One-shot converter from wrapper IR lines to an FX GraphModule.
    # Inputs (fields): lines, prologue text, graph_inputs, graph_outputs, subgms, is_subgraph. Built state: the module under
    # construction, a symbol table from buffer names to FX nodes, a table of imported Triton kernels, a map from sympy
    # expressions to FX proxies (for symbolic sizes), and a tracer. generate() is the entry point; each wrapper line type has a
    # matching _generate_* handler.
    """
    Generates FX IR from Wrapper IR. As each instance is only meant to be used once, the
    input and output code are stored as attributes.
    """

    lines: list[Line]
    prologue: str
    graph_inputs: dict[str, ir.TensorBox | ir.TorchBindObject | sympy.Expr | None]
    graph_outputs: list[ir.IRNode]
    subgms: dict[str, torch.fx.GraphModule]
    is_subgraph: bool

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Initialize the empty GraphModule, symbol table, kernel table, unique-id counter, tracer and expression-to-proxy map.
        graph = torch.fx.Graph()
        self.gm = GraphModule({}, graph)  # Wrapper FX IR.
        self.buffer_to_node: dict[
            str | None, torch.fx.Node
        ] = {}  # Symbol table for codegen.
        self.kernels: dict[str, TritonKernel] = {}  # Table to store Triton kernels.
        self._unique_symbol_ids: Counter[str] = Counter()
        self.tracer = torch.fx.proxy.GraphAppendingTracer(graph)
        self.expr_to_proxy: dict[sympy.Expr, torch.fx.Proxy] = {}

    def _import_kernel(self, code: str, kernel_name: str) -> CachingAutotuner:
        # [FUNCTION DOC] HELPER: Load kernel source (prologue plus kernel code) through the code cache and return the named
        # CachingAutotuner, resolving a future if needed and reloading the kernel function in the parent process.
        # Raises NotImplementedError if the object is not a Triton autotuner.
        """
        Imports a kernel from source, possibly autotuning block parameters.
        """
        module_code = "\n".join([self.prologue, code])
        mod = PyCodeCache.load(module_code)
        kernel = getattr(mod, kernel_name)

        if isinstance(kernel, LambdaFuture):
            kernel = kernel.result()

        if not isinstance(kernel, CachingAutotuner):
            raise NotImplementedError(
                textwrap.dedent(f"""
                Unsupported type for kernel {kernel_name}: {type(kernel)}.
                FX conversion only supports Triton kernels.
            """)
            )

        # Parallel compile workers strip the Python function from the pickled
        # JITFunction. FXIR stores the JITFunction in the Triton HOP side table,
        # so reload it in the parent before runtime execution can need it.
        kernel._ensure_kernel_loaded()

        return kernel

    def _create_as_strided(
        self,
        input_node: torch.fx.Node,
        size: tuple[Any, ...],
        stride: tuple[Any, ...],
        offset: int | sympy.Expr,
    ) -> torch.fx.Node:
        # [FUNCTION DOC] HELPER: Emit an as_strided call node on input_node with the given size, stride and offset.
        # Symbolic offsets get floor replaced by FloorDiv; sizes, strides and offset are converted to FX values. Returns the node.
        if isinstance(offset, sympy.Expr):
            offset = replace_floor_div(offset)
        return self.gm.graph.call_function(
            torch.as_strided,
            args=(
                input_node,
                self._generate_sym_nodes(size),
                self._generate_sym_nodes(stride),
                self._generate_sym_node(offset),
            ),
        )

    def _record_allocation(self, buffer: CodegenBuffer, node: torch.fx.Node) -> None:
        # [FUNCTION DOC] HELPER: Record in the symbol table that the buffer's name maps to the node.
        # KNOWN ISSUE: the duplicate check tests whether the node (not the buffer name) is a key of the table, so it never
        # detects re-allocation of an existing name.
        """
        Updates the symbol table to record that an Inductor buffer maps to the result of
        an FX node.
        """
        if node in self.buffer_to_node:
            raise AssertionError(f"node already recorded in buffer_to_node: {node}")
        self.buffer_to_node[buffer.get_name()] = node

    def _free(self, buffer: CodegenBuffer | ir.TorchBindObject) -> None:
        # [FUNCTION DOC] HELPER: Remove the buffer's name from the symbol table (raises KeyError if absent).
        """
        Removes the buffer from the symbol table.
        """
        name = buffer.get_name()
        del self.buffer_to_node[name]

    def _lookup_args(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Map call args to FX values: strings become the FX node of that buffer name, symbolic call args
        # become their inner expression, anything else is unchanged. Returns a tuple.
        """
        Maps call args back to FX nodes.
        """
        return tuple(
            self.buffer_to_node[arg]
            if isinstance(arg, str)
            else arg.inner_expr
            if isinstance(arg, SymbolicCallArg)
            else arg
            for arg in args
        )

    def _get_buffer(self, node: ir.IRNode) -> CodegenBuffer:
        # [FUNCTION DOC] HELPER: Find the underlying buffer of an IR node by unwrapping views and mutable boxes; a sympy symbol
        # becomes a SymbolBuffer. Raises NotImplementedError for other node kinds.
        """
        Extract buffer data from an IR node.
        """
        if isinstance(node, (ir.Buffer, WorkspaceArg)):
            return node
        elif isinstance(node, (ir.BaseView, ir.MutableBox)):
            return self._get_buffer(node.data)
        elif isinstance(node, sympy.Symbol):
            return SymbolBuffer(node)
        else:
            raise NotImplementedError(f"Unable to extract buffer from node: {node}")

    def _generate_size_proxy(
        self, node: torch.fx.Node, expr: sympy.Expr
    ) -> torch.fx.Proxy:
        # [FUNCTION DOC] HELPER: Wrap the node as a proxy, register it as the value of the expression, and return the proxy.
        proxy = torch.fx.Proxy(node, tracer=self.tracer)
        self.expr_to_proxy[expr] = proxy
        return proxy

    def _generate_graph_inputs(self) -> None:
        # [FUNCTION DOC] Create FX placeholders for all graph inputs in order, record them in the symbol table, set example
        # values in node metadata, and register symbolic inputs as sizes. None inputs become dummy placeholders; constants get a
        # fresh integer symbol; compound integer expressions are named by their placeholder.
        # Raises NotImplementedError for non-integer compound expressions.
        """
        Converts graph inputs to FX placeholders.
        """

        for name, ir_node in self.graph_inputs.items():
            if ir_node is None:
                # Create dummy input nodes to match the input signature
                self.gm.graph.placeholder(name)
                continue

            # Introduce a new symbol for constant inputs.
            is_constant = isinstance(ir_node, (int, float, sympy.Integer, sympy.Float))

            # Special handling for dynamic shapes which are not simple symbols.
            is_expr = isinstance(ir_node, sympy.Expr) and not isinstance(
                ir_node, (sympy.Symbol, sympy.Integer, sympy.Float)
            )
            if is_expr and ir_node.is_integer is not True:
                raise NotImplementedError(
                    f"Unsupported non-integer symbolic graph input: {ir_node}"
                )

            if is_constant:
                buffer = SymbolBuffer(sympy.Symbol(name, is_integer=True))
            elif is_expr:
                buffer = SymbolBuffer(ir_node, name=name)
            else:
                buffer = self._get_buffer(ir_node)
            placeholder_node = self.gm.graph.placeholder(buffer.get_name())
            placeholder_node.meta["val"] = (
                ir_node if is_constant else buffer.get_example()
            )
            self._record_allocation(buffer, placeholder_node)

            # Record symbol definitions for dynamic shapes.
            if isinstance(ir_node, sympy.Expr) and not is_constant:
                self._generate_size_proxy(placeholder_node, ir_node)

    def _generate_graph_input_shapes(self) -> None:
        # [FUNCTION DOC] Emit nodes that define the symbolic sizes/strides of tensor inputs (via size and stride calls on the
        # placeholder), solving for one unknown symbol in compound expressions when needed. A second pass recovers symbols that
        # occur only inside compound scalar inputs. Raises NotImplementedError if an expression leaves more than one symbol
        # undefined or cannot be solved.
        """
        Generate nodes creating symints that are part of graph input
        shape/strides.
        """

        def _codegen_symbol(
            sym_or_exp: sympy.Symbol | sympy.Expr,
            base_node: torch.fx.Node,
            target: torch._ops.OpOverload,
            dim: int,
        ) -> None:
            # [FUNCTION DOC] HELPER: Define one size or stride symbol or expression from a dim of the base node (see enclosing function).
            def codegen_proxy() -> torch.fx.Proxy:
                # [FUNCTION DOC] HELPER: Emit a size/stride call for the dimension and register its proxy for the expression; returns the proxy.
                size_node = self.gm.graph.call_function(target, (base_node, dim))
                size_proxy = self._generate_size_proxy(size_node, sym_or_exp)
                return size_proxy

            if isinstance(sym_or_exp, sympy.Symbol):
                if sym_or_exp in self.expr_to_proxy:
                    return
                codegen_proxy()

            elif isinstance(sym_or_exp, sympy.Integer):
                return

            elif isinstance(sym_or_exp, sympy.Expr):
                # Check if we need to solve for an undefined symbol.
                undefined_symbols = [
                    sym
                    for sym in sym_or_exp.free_symbols
                    if sym not in self.expr_to_proxy
                ]
                if len(undefined_symbols) == 0:
                    self._sympy_interp(sym_or_exp)
                    return
                elif len(undefined_symbols) > 1:
                    raise NotImplementedError(
                        f"Underdetermined input expression: {sym_or_exp}"
                    )

                # Define a new symbol for the input size.
                size_proxy = codegen_proxy()
                size_symbol = sympy.Symbol(
                    size_proxy.node.name, integer=True, nonnegative=True
                )
                self.expr_to_proxy[size_symbol] = size_proxy

                self._define_symbol_by_solving(
                    sym_or_exp, size_symbol, undefined_symbols[0]
                )

        for ir_node in self.graph_inputs.values():
            if isinstance(ir_node, ir.TensorBox):
                buffer = self._get_buffer(ir_node)
                placeholder_node = self.buffer_to_node[buffer.get_name()]

                for dim, size in enumerate(ir_node.get_size()):
                    _codegen_symbol(
                        size, placeholder_node, torch.ops.aten.sym_size.int, dim
                    )
                for dim, stride in enumerate(ir_node.get_stride()):
                    _codegen_symbol(
                        stride, placeholder_node, torch.ops.aten.sym_stride.int, dim
                    )

        # A compound input binds its whole expression to one placeholder, so a
        # symbol appearing only inside it has no node of its own. Recover it
        # from that placeholder, after the loop above has defined every symbol
        # the tensor shapes determine.
        for ir_node in self.graph_inputs.values():
            if not isinstance(ir_node, sympy.Expr) or isinstance(ir_node, sympy.Symbol):
                continue
            proxy = self.expr_to_proxy.get(ir_node)
            if proxy is None:
                continue
            undefined = [
                sym for sym in ir_node.free_symbols if sym not in self.expr_to_proxy
            ]
            if len(undefined) == 0:
                continue
            elif len(undefined) > 1:
                # One bound value cannot determine several symbols. This is a
                # branch closing over an expression whose symbols come from
                # tensors it does not take, such as y.shape[0] + z.shape[0].
                raise NotImplementedError(
                    f"Compound input {ir_node} leaves these symbols undefined: "
                    f"{sorted(undefined, key=str)}"
                )

            # A tensor-derived FX node does not reserve its symbol's name, so a
            # Dummy is what keeps the anchor from colliding with one.
            anchor = sympy.Dummy(proxy.node.name, integer=True)
            self.expr_to_proxy[anchor] = proxy
            self._define_symbol_by_solving(ir_node, anchor, undefined[0])

    def _generate_graph_constants(self) -> None:
        # [FUNCTION DOC] Emit get-attribute nodes for all graph constants, store the values on the module and in the symbol table.
        # Side effect: sets attributes on the GraphModule.
        for name, value in V.graph.constants.items():
            node = self.gm.graph.get_attr(name)
            node.meta["val"] = value
            setattr(self.gm, name, value)
            self.buffer_to_node[name] = node

    def _generate_buffer(self, node: ir.IRNode) -> torch.fx.Node | None:
        # [FUNCTION DOC] Return the FX node for an output/argument IR node, generating view code if needed. Shape-as-constant nodes
        # become computed expression nodes; None-constant gives None; reinterpret views get a new unique workspace buffer
        # with an as_strided node. Raises NotImplementedError for unknown node types.
        """
        Generates FX IR for transformations on a buffer, such as ReinterpretView.
        Does nothing if no such transformations are present.
        """

        if isinstance(node, ir.ShapeAsConstantBuffer):
            # Generate FX nodes to compute the shape expression.
            return self._sympy_interp(node.expr).node

        def generate_to_buffer(node: ir.IRNode) -> BufferLike | None:
            # [FUNCTION DOC] HELPER: Resolve an IR node to a buffer (or None), unwrapping boxes and materializing reinterpret views.
            # Raises NotImplementedError for unrecognized nodes and AssertionError when a view buffer has no device.
            if isinstance(node, (ir.Buffer, WorkspaceArg)):
                return node
            elif isinstance(node, ir.NoneAsConstantBuffer):
                return None
            elif isinstance(node, ir.MutableBox):
                return generate_to_buffer(node.data)
            elif isinstance(node, ir.ReinterpretView):
                # We need to introduce a new symbol if the output is a ReinterpretView.
                # Use a WorkspaceArg for this.
                buffer = self._get_buffer(node.data)
                if not isinstance(buffer, (ir.Buffer, WorkspaceArg)):
                    raise AssertionError(
                        f"expected ir.Buffer or WorkspaceArg, got {type(buffer)}"
                    )
                unique_name = self.gm.graph._graph_namespace.create_name(
                    f"{buffer.get_name()}_view", None
                )
                device = buffer.get_device()
                if not device:
                    raise AssertionError(f"buffer has no device: {buffer}")
                reused_as = WorkspaceArg(
                    count=buffer.get_size(),
                    zero_mode=WorkspaceZeroMode.UNINITIALIZED,
                    device=device,
                    outer_name=unique_name,
                    dtype=buffer.get_dtype(),
                )

                # Generate FX IR for the view.
                self._generate_reinterpret_helper(buffer, reused_as, node.layout)

                return reused_as
            else:
                raise NotImplementedError(f"Unrecognized buffer/view node: {node}")

        buffer = generate_to_buffer(node)
        return self.buffer_to_node[buffer.get_name()] if buffer is not None else None

    def _generate_outputs(
        self,
    ) -> torch.fx.Node | None | list[torch.fx.Node | None]:
        # [FUNCTION DOC] Compute the FX node(s) for graph outputs. Returns a single node for a one-output top-level graph,
        # otherwise a list of nodes (entries may be None).
        """
        Generate FX IR for graph outputs.
        """
        output_nodes = [
            self._generate_buffer(node) for idx, node in enumerate(self.graph_outputs)
        ]

        # Parent graphs with single return elements don't use a tuple.
        output_value = (
            output_nodes[0]
            if len(output_nodes) == 1 and not self.is_subgraph
            else output_nodes
        )

        return output_value

    def _generate_subgm_getattrs(self) -> None:
        # [FUNCTION DOC] Add each generated subgraph module as a submodule plus a get-attribute node, and store the mapping in subgm_getattrs.
        """
        Generate getattr nodes for subgms.
        """

        def generate_getattr(name: str, subgm: torch.fx.GraphModule) -> torch.fx.Node:
            # [FUNCTION DOC] HELPER: Register one subgraph module under its name and return its get-attribute node (value stored in metadata).
            self.gm.add_submodule(name, subgm)
            node = self.gm.graph.get_attr(name)
            node.meta["val"] = subgm
            return node

        self.subgm_getattrs = {
            name: generate_getattr(name, subgm) for name, subgm in self.subgms.items()
        }

    def _get_subgm_attr(self, subgraph: ir.Subgraph) -> torch.fx.Node:
        # [FUNCTION DOC] Return the get-attribute node of an IR subgraph by its graph name. Raises AssertionError if the subgraph has no graph.
        """
        Look up the getattr node for a subgraph.
        """
        graph = subgraph.graph
        if graph is None:
            raise AssertionError("subgraph.graph must not be None")
        return self.subgm_getattrs[graph.name]

    def generate(self) -> torch.fx.GraphModule:
        # [FUNCTION DOC] Main entry point. Creates placeholders, constants and subgraph attribute nodes, then under a node
        # metadata hook defines input shapes and converts each line: wrapper lines dispatch to their codegen_fx handler, line
        # contexts are ignored, other types raise NotImplementedError. Finally emits the output, recompiles and returns the module.
        """
        Main entrypoint for FX codegen.
        """
        self._generate_graph_inputs()
        self._generate_graph_constants()
        self._generate_subgm_getattrs()

        with _set_node_metadata_hook(
            self.gm,
            functools.partial(_node_metadata_hook, fake_mode=V.fake_mode),
        ):
            self._generate_graph_input_shapes()

            # Generate FX IR from Wrapper IR lines.
            for line in self.lines:
                if isinstance(line, WrapperLine):
                    line.codegen_fx(self)(line)
                elif isinstance(line, LineContext):
                    # Ignore line context in FX IR.
                    pass
                else:
                    raise NotImplementedError(
                        textwrap.dedent(
                            f"""
                        Found line of unrecognized type '{type(line)}':
                            '{line}'

                        FX conversion only supports Wrapper IR lines.
                        """
                        )
                    )

            output = self._generate_outputs()

        self.gm.graph.output(output)
        self.gm.recompile()
        return self.gm

    def _sympy_interp(self, expr: sympy.Expr) -> torch.fx.Proxy:
        # [FUNCTION DOC] HELPER: Convert a sympy expression to an FX proxy using the Python reference analysis, with hash-consing
        # via the expression-to-proxy map. Base cases (numbers, symbols, boolean atoms) are not cached; others are cached.
        # hash cons
        if expr in self.expr_to_proxy:
            return self.expr_to_proxy[expr]
        # base cases, don't cache
        if isinstance(
            expr,
            (
                sympy.Integer,
                sympy.Number,
                sympy.Symbol,
                sympy.logic.boolalg.BooleanAtom,
            ),
        ):
            return sympy_interp(
                OptimizedPythonReferenceAnalysis, self.expr_to_proxy, expr
            )

        # hash cons on arguments, run expr handler
        self.expr_to_proxy[expr] = _run_sympy_handler(
            OptimizedPythonReferenceAnalysis,
            [self._sympy_interp(arg) for arg in expr.args],
            expr,
        )
        return self.expr_to_proxy[expr]

    def _define_symbol_by_solving(
        self, expr: sympy.Expr, anchor: sympy.Symbol, symbol: sympy.Symbol
    ) -> None:
        # [FUNCTION DOC] HELPER: Define symbol by solving the equation expr equals anchor (anchor already has a proxy).
        # For integer symbols the solution is converted to floor division. Registers the symbol to the solution proxy.
        # Raises NotImplementedError if no solution is found.
        """
        Define symbol by solving expr == anchor, which already maps to a proxy.
        """
        # A returned solution directly defines the symbol in terms of the anchor.
        solution = try_solve(sympy.Eq(expr, anchor), symbol)
        if solution is None:
            raise NotImplementedError(
                f"Cannot solve input expression {expr} for {symbol}"
            )

        symbol_expr = solution[1]
        # If the symbol is an integer, division becomes FloorDiv.
        if symbol.is_integer:
            symbol_expr = replace_floor_div(sympy.floor(symbol_expr))

        # Generate FX for the symbol.
        self._sympy_interp(symbol_expr)
        self.expr_to_proxy[symbol] = self.expr_to_proxy[symbol_expr]

    def _generate_sym_node(self, s: int | sympy.Expr) -> int | torch.fx.Node:
        # [FUNCTION DOC] HELPER: Convert a size/stride/offset value to an int or FX node: integers become ints, symbols are looked
        # up (AssertionError if undefined), expressions are interpreted, nodes pass through. Raises ValueError for other types.
        if isinstance(s, (int, sympy.Integer)):
            return int(s)
        elif isinstance(s, sympy.Symbol):
            if s not in self.expr_to_proxy:
                raise AssertionError(
                    f"Could not find a node corresponding to the symbol {s}"
                )
            return self.expr_to_proxy[s].node
        elif isinstance(s, sympy.Expr):
            return self._sympy_interp(s).node

        elif isinstance(s, torch.fx.Node):
            return s

        else:
            raise ValueError(f"{s} of type {type(s)} is not a valid input")

    def _generate_sym_nodes(
        self, shape: Sequence[sympy.Expr]
    ) -> list[int | torch.fx.Node]:
        # [FUNCTION DOC] HELPER: Convert each element of a shape using the single-value conversion; returns a list.
        return [self._generate_sym_node(s) for s in shape]

    def _generate_allocate(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Handle an allocate line: emit an empty-strided call node with the buffer's size, stride, dtype and device
        # type, name it after the buffer and record it. Raises AssertionError for removed buffers, missing device or empty name.
        if not isinstance(line, AllocateLine):
            raise AssertionError(f"expected AllocateLine, got {type(line)}")
        buffer = line.node
        name = buffer.get_name()
        if name in V.graph.removed_buffers:
            raise AssertionError(f"buffer {name} is in removed_buffers")

        device = buffer.get_device()
        if not device:
            raise AssertionError(f"buffer has no device: {buffer}")
        dtype = buffer.get_dtype()
        shape = self._generate_sym_nodes(buffer.get_size())
        stride = self._generate_sym_nodes(buffer.get_stride())

        node = self.gm.graph.call_function(
            torch.empty_strided,
            args=(shape, stride),
            kwargs={"dtype": dtype, "device": device.type},
        )
        if not name:
            raise AssertionError(f"buffer has an empty name: {buffer}")
        node.name = name
        self._record_allocation(buffer, node)

    def _generate_switch(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Handle a switch line: build FX nodes for selector, operands and branch subgraphs, then emit a call to the
        # higher-order cond (exactly two branches, given in false/true order) or switch op, and record the result.
        # Raises AssertionError if branches or operands are missing or a cond lacks two branches.
        if not isinstance(line, SwitchLine):
            raise AssertionError(f"expected SwitchLine, got {type(line)}")

        ir_node = line.node
        if ir_node.branches is None:
            raise AssertionError("ir_node.branches must not be None")
        if ir_node.operands is None:
            raise AssertionError("ir_node.operands must not be None")

        def generate_buffer(node: ir.IRNode | None) -> torch.fx.Node | None:
            # [FUNCTION DOC] HELPER: Generate the FX node of an IR node; raises AssertionError if the node is None.
            if node is None:
                raise AssertionError("node must not be None")
            return self._generate_buffer(node)

        selector = generate_buffer(ir_node.selector)
        operands = tuple(generate_buffer(arg) for arg in ir_node.operands)
        branch_subgms = [self._get_subgm_attr(branch) for branch in ir_node.branches]

        if ir_node.is_cond:
            # cond expects (selector, true_fn, false_fn, operands) -- branches unpacked positionally.
            if len(branch_subgms) != 2:
                raise AssertionError(
                    f"cond requires exactly 2 branches, got {len(branch_subgms)}"
                )
            # branches are stored as [false_fn, true_fn] in ir.Switch
            false_subgm, true_subgm = branch_subgms
            fx_node = self.gm.graph.call_function(
                torch.ops.higher_order.cond,
                args=(selector, true_subgm, false_subgm, operands),
            )
        else:
            # switch expects (selector, [branch_fn, ...], operands) -- branches passed as a list.
            fx_node = self.gm.graph.call_function(
                torch.ops.higher_order.switch,
                args=(selector, branch_subgms, operands),
            )
        self._record_allocation(ir_node, fx_node)

    def _generate_assert_size_stride(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Intentionally does nothing: size/stride assertion lines are not represented in FX.
        pass

    def _generate_comment(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Ignores comment lines (type checked only).
        if not isinstance(line, CommentLine):
            raise AssertionError(f"expected CommentLine, got {type(line)}")
        # We ignore comments in FX IR.

    def _generate_dynamic_scalar(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Handle a dynamic scalar line: convert the single input to a scalar item call (an empty keypath), or for a
        # single convert-to-int key first map the input to one or zero with a where call. Registers the result under the node's
        # symbol as a buffer and size proxy. Raises NotImplementedError for other keypaths.
        if not isinstance(line, DynamicScalarLine):
            raise AssertionError(f"expected DynamicScalarLine, got {type(line)}")

        ir_node = line.node
        (input_ir_node,) = ir_node.inputs
        if not isinstance(input_ir_node, ir.IRNode):
            raise AssertionError(f"expected ir.IRNode, got {type(input_ir_node)}")
        input_fx_node = self._generate_buffer(input_ir_node)
        keypath = ir_node.keypath
        graph = self.gm.graph

        def generate_item(x: torch.fx.Node | None) -> torch.fx.Node:
            if x is None:
                raise AssertionError("x must not be None")
            return graph.call_function(
                aten.item.default,
                args=(x,),
            )

        if len(keypath) == 0:
            result_fx_node = generate_item(input_fx_node)
        elif len(keypath) == 1 and isinstance(keypath[0], ConvertIntKey):
            where_fx_node = graph.call_function(
                aten.where.Scalar,
                args=(input_fx_node, 1, 0),
            )
            result_fx_node = generate_item(where_fx_node)
        else:
            raise NotImplementedError(f"Unsupported keypath: {keypath}")

        result_symbol = ir_node.sym
        result_buffer = SymbolBuffer(result_symbol)
        self._record_allocation(result_buffer, result_fx_node)
        self._generate_size_proxy(result_fx_node, result_symbol)

    def _generate_enter_device_context_manager(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Ignores enter-device-context lines (type checked only); device contexts are not expressed in FX.
        if not isinstance(line, EnterDeviceContextManagerLine):
            raise AssertionError(
                f"expected EnterDeviceContextManagerLine, got {type(line)}"
            )
        # We ignore the device context in FX IR.

    def _generate_exit_device_context_manager(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Ignores exit-device-context lines (type checked only).
        if not isinstance(line, ExitDeviceContextManagerLine):
            raise AssertionError(
                f"expected ExitDeviceContextManagerLine, got {type(line)}"
            )
        # We ignore the device context in FX IR.

    def _generate_enter_subgraph(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Ignores enter-subgraph lines (type checked only).
        if not isinstance(line, EnterSubgraphLine):
            raise AssertionError(f"expected EnterSubgraphLine, got {type(line)}")
        # We ignore memory planning lines in FX IR.

    def _generate_exit_subgraph(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Ignores exit-subgraph lines (type checked only).
        if not isinstance(line, ExitSubgraphLine):
            raise AssertionError(f"expected ExitSubgraphLine, got {type(line)}")
        # We ignore memory planning lines in FX IR.

    def _generate_free(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Handle a free line: remove the buffer from the symbol table unless it is a placeholder (inputs are never
        # freed).
        if not isinstance(line, FreeLine):
            raise AssertionError(f"expected FreeLine, got {type(line)}")

        buf = line.node

        # No need to free placeholders.
        if self.buffer_to_node[buf.get_name()].op == "placeholder":
            return

        self._free(buf)

    def _generate_free_if_not_reused(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Handle a free-if-not-reused line: remove the buffer from the symbol table when it was not reused.
        # Raises AssertionError if the buffer was already removed from the graph.
        if not isinstance(line, FreeIfNotReusedLine):
            raise AssertionError(f"expected FreeIfNotReusedLine, got {type(line)}")
        buf = line.node
        if buf.get_name() in V.graph.removed_buffers:
            raise AssertionError(f"buffer {buf.get_name()} is in removed_buffers")
        if not line.is_reused:
            self._free(buf)

    def _generate_line_context(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Ignores line-context markers (type checked only).
        if not isinstance(line, LineContext):
            raise AssertionError(f"expected LineContext, got {type(line)}")
        # We ignore line context in FX IR.

    def _generate_reinterpret(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Handle a reinterpret line by delegating to the reinterpret helper with the node, reused buffer and layout.
        if not isinstance(line, ReinterpretLine):
            raise AssertionError(f"expected ReinterpretLine, got {type(line)}")
        self._generate_reinterpret_helper(line.node, line.reused_as, line.layout)

    def _generate_reinterpret_helper(
        self, input_buffer: BufferLike, result_buffer: BufferLike, layout: ir.Layout
    ) -> None:
        # [FUNCTION DOC] HELPER: Emit an as_strided node viewing the input buffer with the layout's size and stride and the sum of the
        # input and layout offsets, name it after the result buffer, and record it. For non-owning layouts the underlying
        # view's layout supplies the offset. Raises AssertionError for an empty name or a non-view layout target.
        input_node = self.buffer_to_node[input_buffer.get_name()]

        # Look up output metadata.
        name = result_buffer.get_name()
        if not name:
            raise AssertionError(f"result_buffer has an empty name: {result_buffer}")
        size = tuple(layout.size)
        stride = tuple(layout.stride)
        if isinstance(layout, ir.NonOwningLayout):
            # Look up the view's layout.
            view = layout.view
            if not isinstance(view, ir.ReinterpretView):
                raise AssertionError(f"unexpected type: {type(view)}")
            layout = view.layout
        offset = input_buffer.get_offset() + layout.offset

        # Map ReinterpretView to as_strided.
        result_node = self._create_as_strided(input_node, size, stride, offset)
        result_node.name = name
        self._record_allocation(result_buffer, result_node)

    def _generate_reuse(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Handle a reuse line: map the new buffer to the old buffer's node, adding an as_strided node if size,
        # stride or offset differ; frees the old buffer if requested, a new tensor was made and it is not a graph output.
        # Raises AssertionError for removed buffers or dtype mismatch.
        if not isinstance(line, ReuseLine):
            raise AssertionError(f"expected ReuseLine, got {type(line)}")
        old = line.node
        new = line.reused_as
        if any(buf.get_name() in V.graph.removed_buffers for buf in (old, new)):
            raise AssertionError("old or new buffer is in removed_buffers")
        if old.get_dtype() != new.get_dtype():
            raise AssertionError(
                f"dtype mismatch: {old.get_dtype()} != {new.get_dtype()}"
            )

        old_node = self.buffer_to_node[old.get_name()]
        result_node = old_node

        # Change shape and stride.
        size = tuple(new.get_size())
        stride = tuple(new.get_stride())
        offset = new.get_offset()
        if (
            tuple(old.get_size()) != size
            or tuple(old.get_stride()) != stride
            or old.get_offset() != offset
        ):
            result_node = self._create_as_strided(old_node, size, stride, offset)

        self._record_allocation(new, result_node)

        # Free the old buffer, if we allocated a new tensor.
        if (
            old.get_name() not in V.graph.get_output_names()
            and line.delete_old
            and result_node is not old_node
        ):
            self._free(old)

    def _generate_multi_output(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Handle a multi-output line: with no indices, alias the result name to the argument's node; otherwise emit a
        # getitem node for the single index and name it after the result. Raises AssertionError for multi-level indices.
        if not isinstance(line, MultiOutputLine):
            raise AssertionError(f"expected MultiOutputLine, got {type(line)}")

        arg_node = self.buffer_to_node[line.arg_name]

        # For non-tuple / non-list outputs, map the
        # output to the same node as the input.
        if len(line.indices) == 0:
            self.buffer_to_node[line.result_name] = arg_node
            return

        # Extract the index for tuple access.
        inds = line.indices[0][1:]
        if len(inds) != 1:
            raise AssertionError(f"Cannot convert {inds} to an index.")
        idx = inds[0]

        node = self.gm.graph.call_function(operator.getitem, args=(arg_node, idx))
        node.name = line.result_name
        self.buffer_to_node[line.result_name] = node

    def _generate_fallback_call(
        self,
        ir_node: ir.ExternKernel,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Emit a call to the fallback op overload with the given args and kwargs, map the node's reference name
        # to the new FX node, and remap every mutated input name to it so later reads see the post-mutation value.
        fx_node = self.gm.graph.call_function(
            ir_node.op_overload,  # type: ignore[arg-type]
            args=args,
            kwargs=kwargs,
        )
        result_buffer = ir_node.codegen_reference()
        self.buffer_to_node[result_buffer] = fx_node
        # For in-place mutation ops (e.g., scatter_reduce_, index_put_),
        # update the buffer mapping for mutated inputs so downstream
        # references to the mutated buffer see the post-mutation node.
        for mutated_name in ir_node.get_mutation_names():
            self.buffer_to_node[mutated_name] = fx_node

    def _generate_index_put_fallback(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Handle an index-put fallback line: build FX nodes for the target, values and index tensors (None allowed),
        # read the accumulate flag from constant args and emit the fallback call.
        if not isinstance(line, IndexPutFallbackLine):
            raise AssertionError(f"expected IndexPutFallbackLine, got {type(line)}")
        ir_node = line.node

        def generate_buffer_or_none(
            x: ir.IRNode | Sequence[ir.IRNode] | None,
        ) -> torch.fx.Node | None:
            # [FUNCTION DOC] HELPER: Return the FX node for an IR node, or None for None. Raises AssertionError for non-IR input.
            """
            Handles None before calling _generate_buffer.
            """
            if x is None:
                return None

            if not isinstance(x, ir.IRNode):
                raise AssertionError(f"expected ir.IRNode, got {type(x)}")
            return self._generate_buffer(x)

        (x, values) = [generate_buffer_or_none(t) for t in ir_node.inputs[:2]]
        indices = tuple(generate_buffer_or_none(t) for t in line.indices)
        accumulate = ir_node.constant_args[0]
        args = (x, indices, values, accumulate)
        self._generate_fallback_call(ir_node, args)

    def _generate_scatter_fallback(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Handle a scatter fallback line: build FX nodes for target, index and source (a constant if the source is not
        # a tensor), pass the dim constant, and forward only those keyword args accepted by the op schema, then emit the call.
        # KNOWN ISSUE: the first kwargs built from the reduce option is immediately overwritten by the schema-filtered
        # mapping, so that earlier code is dead.
        if not isinstance(line, ScatterFallbackLine):
            raise AssertionError(f"expected ScatterFallbackLine, got {type(line)}")
        ir_node = line.node
        if not ir.is_node_sequence(ir_node.inputs):
            raise AssertionError("ir_node.inputs is not a node sequence")
        (x, index, src) = [self._generate_buffer(t) for t in ir_node.inputs] + (
            [] if ir_node.src_is_tensor else [ir_node.constant_args[1]]
        )
        args = (x, ir_node.constant_args[0], index, src)
        kwargs = {}
        if reduce := ir_node.kwargs.get("reduce"):
            kwargs["reduce"] = reduce
        # Only pass kwargs that the op's schema actually accepts, since
        # ScatterFallback stores both reduce and include_self for all
        # scatter variants, but not all ops support them (e.g.,
        # scatter_.value has no kwargs, scatter_reduce_.two has both).
        if not isinstance(ir_node.op_overload, torch._ops.OpOverload):
            raise AssertionError(
                f"expected torch._ops.OpOverload, got {type(ir_node.op_overload)}"
            )
        schema_arg_names = OrderedSet(
            [a.name for a in ir_node.op_overload._schema.arguments]
        )
        kwargs = {k: v for k, v in ir_node.kwargs.items() if k in schema_arg_names}

        self._generate_fallback_call(ir_node, args, kwargs)

    def _generate_null(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Handles null lines by doing nothing.
        if not isinstance(line, NullLine):
            raise AssertionError(f"expected NullLine, got {type(line)}")
        # Does nothing.

    def _generate_comm_buffer_allocate(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Always raises NotImplementedError: communication buffer allocation is unsupported in FX.
        if not (isinstance(line, AllocateLine) and line.comm_buffer):
            raise AssertionError("expected AllocateLine with comm_buffer set")
        raise NotImplementedError("Comm buffer allocation is not yet supported")

    def _generate_comm_buffer_free(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Handle freeing of a communication buffer by removing it from the symbol table.
        if not (isinstance(line, FreeIfNotReusedLine) and line.comm_buffer):
            raise AssertionError("expected FreeIfNotReusedLine with comm_buffer set")
        self._free(line.node)

    def _generate_triton_call(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Handle a Triton kernel call: look up call args and the imported kernel; optionally autotune at compile
        # time (when configured and all unbacked symbols are hinted) with real zero tensors sized by hints; build the full
        # launch args (signature order plus constants, extra launcher args), the grid and keyword args as FX values; store
        # non-graphable kwargs in the side table, and emit a call to the Triton kernel higher-order op (with launch kwargs
        # for backend options, and extra options in node metadata). Side effects: logging, kernel runs when autotuning.
        if not isinstance(line, KernelCallLine):
            raise AssertionError(f"expected KernelCallLine, got {type(line)}")

        # Collect all kwargs, including autotuned block sizes.
        call_args = self._lookup_args(line.call_args)
        kernel = self.kernels[line.kernel_name]
        tuner = kernel.tuner

        def tune_kernel(tuner: CachingAutotuner, call_args: Sequence[Any]) -> None:
            # [FUNCTION DOC] HELPER: Run the autotuner once on freshly allocated tensors (zero-filled, shapes from size hints) and
            # hint integers, using the current Triton device and stream. Imports Triton at call time.
            from triton.runtime import driver

            log.info("Autotuning Triton kernel %s at compile time.", kernel_name)

            device = driver.active.get_current_device()

            stream = driver.active.get_current_stream(device)

            def node_to_tuning_arg(arg: Any) -> Any:
                # [FUNCTION DOC] HELPER: Convert an argument to a real tuning value: non-node args become size hints, FX nodes become
                # zero tensors with hinted shape/strides taken from node metadata on the current device.
                """
                Create real tensors for autotuning arguments, substituting size hints
                for dynamic shapes.
                """

                def to_size_hint_sympy_int(arg: sympy.Expr | int) -> int:
                    # [FUNCTION DOC] HELPER: Return the optimization hint of a size expression as an integer.
                    return V.graph.sizevars.optimization_hint(arg)

                def to_size_hint_list(arg: list[torch.SymInt | int]) -> list[int]:
                    # [FUNCTION DOC] HELPER: Convert a list of ints/SymInts into a list of integer hints.
                    args_sympy = [
                        x.node.expr if isinstance(x, torch.SymInt) else x for x in arg
                    ]
                    return pytree.tree_map(to_size_hint_sympy_int, args_sympy)

                if not isinstance(arg, torch.fx.Node):
                    return to_size_hint_sympy_int(arg)

                fake = arg.meta["val"]
                return torch.empty_strided(
                    to_size_hint_list(fake.shape),
                    to_size_hint_list(fake.stride()),
                    dtype=fake.dtype,
                    device=device,
                ).zero_()

            # call args can be fx nodes or sympy expressions or integers!
            arg_values = [node_to_tuning_arg(arg) for arg in call_args]
            tuner.run(*arg_values, stream=stream)

        # Optionally autotune the kernels.
        # The FX backend currently only supports compile-time tuning.
        kernel_name = tuner.fn.__name__
        if config.triton.autotune_at_compile_time:
            # Skip compile-time autotuning if any unbacked symbol lacks a user-provided
            # optimization hint — autotuning with the generic fallback would
            # produce meaningless results.
            hinted = V.graph.sizevars.all_unbacked_explicitly_hinted
            can_tune = True
            for arg in call_args:
                if isinstance(arg, torch.fx.Node):
                    fake = arg.meta["val"]
                    if not hinted(list(fake.shape) + list(fake.stride())):
                        can_tune = False
                        break
                elif not hinted(arg):
                    can_tune = False
                    break
            if can_tune:
                tune_kernel(tuner, call_args)
            else:
                log.info(
                    "Detected unhinted unbacked symints. Skipping compile-time autotuning for kernel %s.",
                    kernel_name,
                )
        else:
            log.info(
                "Skipping autotuning for kernel %s. Set config.triton.autotune_at_compile_time = True to enable.",
                kernel_name,
            )

        triton_meta = tuner.triton_meta
        signature = triton_meta["signature"]

        def add_constants_to_call_args(
            call_args: Sequence[Any], cfg: Config
        ) -> tuple[Any, ...]:
            # [FUNCTION DOC] HELPER: Build the final positional arg tuple for the launcher: signature-ordered values excluding config
            # kwargs, with Triton metadata constants filled in, then the trailing extra launcher args appended.
            """
            Add constant kwargs to the arg list.
            """
            # Add args from the proper Triton signature.
            # Exclude constants and config kwargs, as those are tracked separately.
            new_call_args = []
            constants = triton_meta["constants"]
            call_kwargs = {
                key: val
                for key, val in zip(signature, call_args)
                # pyrefly: ignore [missing-attribute]
                if key not in constants and key not in cfg.kwargs
            }

            # Add constants stored as Triton metadata, in signature order.
            call_kwargs |= constants
            new_call_args = [
                call_kwargs[key]
                for key in signature
                # pyrefly: ignore [missing-attribute]
                if key not in cfg.kwargs
            ]

            # Add Inductor's extra launcher args to the end.
            if extra_launcher_args := tuner.inductor_meta.get("extra_launcher_args"):
                new_call_args.extend(
                    call_args[len(call_args) - len(extra_launcher_args) :]
                )

            return tuple(new_call_args)

        kernel_config = tuner.compile_results[0].config
        extra_options = getattr(kernel_config, "extra_options", None)
        call_args = add_constants_to_call_args(call_args, kernel_config)
        call_args, grid = tuner._interpret_args_grid(call_args, kernel_config)
        call_kwargs = dict(zip(signature, call_args))
        # pyrefly: ignore [missing-attribute]
        if any(kwarg in kernel_config.kwargs for kwarg in call_kwargs):
            raise AssertionError(f"kwargs overlap config: {call_kwargs}")
        # pyrefly: ignore [missing-attribute]
        call_kwargs.update(kernel_config.kwargs)

        # Replace sympy.floor with FloorDiv, to make the expression traceable.
        grid = [replace_floor_div(x) if isinstance(x, sympy.Expr) else x for x in grid]
        wrapper_grid = [tuple(self._generate_sym_nodes(grid))]
        call_kwargs = {
            name: self._generate_sym_node(val) for name, val in call_kwargs.items()
        }
        backend_options = triton_meta.get("backend_options", {})
        if backend_options:
            # FXIR executes Triton kernels through the HOP, not the already
            # materialized CachingAutotuner launcher. Preserve backend option
            # values in the HOP payload so a direct FXIR run or later Inductor
            # re-lowering re-enters Triton JIT with the same compiler options
            # that were used during FXIR precompile/autotune.
            for name, value in backend_options.items():
                # If the backend option is also a kernel parameter, call_kwargs
                # already contains the launch value reconstructed from the Triton
                # signature/config. Keep that value and only add backend options
                # that are not otherwise represented in the HOP payload.
                call_kwargs.setdefault(name, value)

        # Store non-graphable kwargs in the side table.
        (
            call_kwargs,
            constant_args_idx,
        ) = tracing_triton_hopifier_singleton.store_non_graphable_args(call_kwargs)

        hop_kwargs: dict[str, Any] = {
            "kernel_idx": kernel.wrapped.kernel_idx,
            "constant_args_idx": constant_args_idx,
            "grid": wrapper_grid,
            "tma_descriptor_metadata": {},
            "kwargs": call_kwargs,
        }
        if backend_options:
            # Keep the FXIR HOP call shape unchanged for normal Inductor
            # kernels. Some downstream HOP py_impls have fixed keyword-only
            # signatures and only need launch_kwargs when there are real
            # Triton backend options to preserve.
            hop_kwargs["launch_kwargs"] = tuple(backend_options)

        triton_node = self.gm.graph.call_function(
            triton_kernel_wrapper_mutation,
            kwargs=hop_kwargs,
        )
        if extra_options:
            triton_node.meta["extra_options"] = extra_options

    def _generate_extern_kernel_alloc(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Handle an extern-kernel-alloc line by running the common extern kernel conversion with the node as output.
        if not isinstance(line, ExternKernelAllocLine):
            raise AssertionError(f"expected ExternKernelAllocLine, got {type(line)}")
        node = line.node
        self._generate_extern_kernel_common(node, node)

    def _generate_extern_kernel_out(
        self,
        line: WrapperLine,
    ) -> None:
        # [FUNCTION DOC] Handle an extern-kernel-out line using the output view (if any) as the output node in the common conversion.
        if not isinstance(line, ExternKernelOutLine):
            raise AssertionError(f"expected ExternKernelOutLine, got {type(line)}")
        node = line.node
        out_node = node.output_view if node.output_view else node
        self._generate_extern_kernel_common(node, out_node)

    def _generate_extern_kernel_common(
        self, kernel: ir.ExternKernel, out_ir_node: ir.IRNode
    ) -> None:
        # [FUNCTION DOC] Emit a call to the extern kernel's op overload with positional args (unflattened if supported) and
        # kwargs whose IR nodes are materialized. For out-variants the out kwarg is the existing output node; otherwise the
        # result is named after the kernel and recorded. Raises NotImplementedError for unknown layouts, AssertionError if
        # both a result and out kwarg exist.
        """
        Generates FX IR from either ExternKernelAlloc or ExternKernelOut.
        """

        # Get FX nodes corresponding to the call args.
        if not ir.is_node_sequence(kernel.inputs):
            raise AssertionError("kernel.inputs is not a node sequence")
        tensor_nodes = tuple(self._generate_buffer(arg) for arg in kernel.inputs)
        if hasattr(kernel, "unflatten_args"):
            args, _ = kernel.unflatten_args(tensor_nodes, kernel.constant_args)
        else:
            args = tensor_nodes + tuple(kernel.constant_args)

        # Get the result buffer.
        # Some kernels write to a pre-existing output tensor via the "out" kwarg.
        # Materialize any IR nodes in kwargs to FX nodes (e.g., TensorBox -> Tensor).
        kwargs = {
            k: self._generate_buffer(v) if isinstance(v, ir.IRNode) else v
            for k, v in kernel.kwargs.items()
        }

        result_buffer: str | None = None
        if isinstance(kernel, ir.ExternKernelOut):
            kwargs["out"] = self.buffer_to_node[out_ir_node.codegen_reference()]
        elif isinstance(kernel.layout, (ir.Layout, ir.MultiOutputLayout)):
            result_buffer = kernel.get_name()
        elif isinstance(kernel.layout, ir.NoneLayout):
            pass
        else:
            raise NotImplementedError(f"Unrecognized output layout: {kernel.layout}")

        fx_node = self.gm.graph.call_function(
            kernel.op_overload,  # type: ignore[arg-type]
            args=args,
            kwargs=kwargs,
        )

        # Assign the result to the given name.
        if result_buffer:
            if "out" in kwargs:
                raise AssertionError(
                    f"Extern kernel '{kernel}' has both result and out kwarg. Expected only one."
                )
            fx_node.name = result_buffer
            self.buffer_to_node[result_buffer] = fx_node

    def _generate_kernel_call(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Handle a kernel call line; only Triton kernels are supported (NotImplementedError otherwise).
        if not isinstance(line, KernelCallLine):
            raise AssertionError(f"expected KernelCallLine, got {type(line)}")
        if not line.triton:
            raise NotImplementedError("FX conversion only supports Triton kernels.")

        self._generate_triton_call(line)

    def _generate_kernel_definition(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Handle a kernel definition line: format the kernel source, import it as a Triton autotuner and store it with its
        # wrapped Triton function in the kernel table. Side effect: loads code through the code cache.
        if not isinstance(line, KernelDefinitionLine):
            raise AssertionError(f"expected KernelDefinitionLine, got {type(line)}")

        # Generate code for the kernel.
        kernel_code = PythonWrapperCodegen._format_kernel_definition(
            line.kernel_name, line.kernel_body, metadata=line.metadata
        )

        # Import the module and store the JIT kernel.
        tuner = self._import_kernel(kernel_code, line.kernel_name)
        wrapped = wrap_triton(tuner.fn)
        self.kernels[line.kernel_name] = TritonKernel(tuner, wrapped)

    def _generate_symbolic_call_arg(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Handle a symbolic call arg line: convert its inner expression and register the result for the inner symbol.
        if not isinstance(line, SymbolicCallArgLine):
            raise AssertionError(f"expected SymbolicCallArgLine, got {type(line)}")
        # Store the arg: expr mapping for later use.
        arg = line.arg

        inner_expr_proxy = self._sympy_interp(arg.inner_expr)
        self.expr_to_proxy[arg.inner] = inner_expr_proxy

    def _generate_unbacked_symbol_defs(self, line: WrapperLine) -> None:
        # [FUNCTION DOC] Handle an unbacked symbol definition line: for each unbacked symbol not yet defined, emit FX nodes following its
        # keypath from the kernel output (size, stride, storage offset, sequence index, floor division) and record the
        # result as a symbol. Returns early when there are no bindings. Raises NotImplementedError for unknown key kinds.
        if not isinstance(line, UnbackedSymbolDefsLine):
            raise AssertionError(f"expected UnbackedSymbolDefsLine, got {type(line)}")
        graph = self.gm.graph

        def convert_key(node: torch.fx.Node, path: pytree.KeyPath) -> torch.fx.Node:
            # [FUNCTION DOC] HELPER: Recursively emit FX nodes for each key entry in the path starting from the node; returns the final node.
            """
            Generate FX IR for each key entry.
            """
            # Base case.
            if len(path) == 0:
                return node

            # Process the first entry and recurse.
            entry = path[0]
            if isinstance(entry, CallMethodKey):
                target = {
                    "size": aten.sym_size.int,
                    "stride": aten.sym_stride.int,
                    "storage_offset": aten.sym_storage_offset,
                }[entry.name]
                if not callable(target):
                    raise AssertionError(f"target is not callable: {target}")
                node = graph.call_function(
                    target,
                    args=(
                        (node, path[1].idx)
                        if len(path) > 1 and isinstance(path[1], pytree.SequenceKey)
                        else (node,)
                    ),
                )
                return convert_key(node, path[1 + len(node.args) :])
            elif isinstance(entry, pytree.SequenceKey):
                node = graph.call_function(operator.getitem, args=(node, entry.idx))
                return convert_key(node, path[1:])
            elif isinstance(entry, DivideByKey):
                node = graph.call_function(
                    operator.floordiv, args=(node, entry.divisor)
                )
                return convert_key(node, path[1:])
            else:
                raise NotImplementedError(f"Unrecognized entry type: {type(entry)}")

        unbacked_bindings = line.unbacked_bindings
        if unbacked_bindings is None:
            raise AssertionError("line.unbacked_bindings must not be None")
        # Kernels with no unbacked symbols aren't recorded in buffer_to_node, so
        # return before the output-buffer lookup to avoid a KeyError. Mirrors
        # the non-FX wrapper's early return.
        if not unbacked_bindings:
            return
        root_node = self.buffer_to_node[line.output_name]
        for s, keypath in unbacked_bindings.items():
            # Check if we already generated this symbol.
            if s.name in self.buffer_to_node:
                continue

            node = convert_key(root_node, keypath)
            out_buffer = SymbolBuffer(s)
            self._record_allocation(out_buffer, node)
            self._generate_size_proxy(node, s)
