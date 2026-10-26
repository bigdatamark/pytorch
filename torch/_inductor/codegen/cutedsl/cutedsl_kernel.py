# mypy: allow-untyped-defs
# [MODULE DOC] Kernel-side codegen for CuteDSL templates: CuteDSLTemplateKernel (signature, hooks, call emission),
# ModificationWrapperCuteDSL (inlines score/mask-style modification subgraphs, including captured tensor loads with
# lane analysis and vectorized/gather strategies), small dataclasses for load configuration and index metadata, a
# sympy printer and a kernel runner wrapper. Import-time: creates loggers and a module-level expression printer.
import contextlib
import dataclasses
import logging
import textwrap
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import sympy

import torch
from torch._inductor import config
from torch._inductor.codegen.common import (
    CSE,
    CSEVariable,
    IndentedBuffer,
    Kernel,
    PythonPrinter,
    ValueRanges,
)
from torch._inductor.ir import (
    BaseView,
    Buffer,
    ComputedBuffer,
    ExternKernel,
    InputBuffer,
    MutableBox,
    ReinterpretView,
)
from torch._inductor.ops_handler import StoreMode
from torch._inductor.utils import OrderedSet
from torch._inductor.virtualized import V

from ...utils import sympy_index_symbol
from .aux_scalars import CuteDSLAuxScalarBindings
from .cutedsl_op_overrides import CuteDSLCSEVariable, CuteDSLOpOverrides
from .lane_analysis import classify_lane_expr


# TODO setting the 'main' kernel w/ this suffix. We have 3 should probably just auto generate this
MAIN_SUFFIX = "main"


log = logging.getLogger(__name__)
kernel_code_log = torch._logging.getArtifactLogger(__name__, "kernel_code")


class CuteDSLPrinter(PythonPrinter):
    # [CLASS DOC] Sympy-to-source printer for CuteDSL kernels; extends the Python printer to render integer-to-float conversion as a Float64 cast.
    def _print_ToFloat(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints a ToFloat expression as a Float64 cast of its single argument; raises AssertionError if it does not have exactly one argument.
        if len(expr.args) != 1:
            raise AssertionError("ToFloat expects exactly one argument")
        return f"cutlass.Float64({self.doprint(expr.args[0])})"


cutedsl_pexpr = CuteDSLPrinter().doprint


class CuteDSLKernelWrapper:
    # [CLASS DOC] Thin callable wrapper giving a compiled CuteDSL kernel function a run method (matching other kernel runners). Holds kernel_fn and
    # kernel_path (for logging only).
    """Wrapper to provide .run() interface for CuteDSL kernels"""

    def __init__(self, kernel_fn: Callable[..., Any], kernel_path: str | None = None):
        # [FUNCTION DOC] Stores the kernel function and path and logs the path at info level on the kernel_code artifact logger.
        self.kernel_fn = kernel_fn
        self.kernel_path = kernel_path
        kernel_code_log.info("CuteDSL kernel path: %s", kernel_path)

    def run(self, *args, stream=None, **kwargs):
        # [FUNCTION DOC] Invokes the wrapped kernel function with the given args and kwargs and an explicit stream keyword; returns its result.
        """
        Execute the CuteDSL kernel.

        Args:
            *args: Arguments to pass to the kernel function
            stream: CUDA stream to pass to the kernel function
            **kwargs: Additional keyword arguments for the kernel

        Returns:
            Result of the kernel execution
        """
        return self.kernel_fn(*args, stream=stream, **kwargs)


@dataclasses.dataclass(frozen=True, slots=True)
class CuteDSLVectorLoadConfig:
    # [CLASS DOC] Frozen settings for vectorized loads in captured-tensor accesses: vec_size (lanes), index (the symbol that varies per lane) and dim
    # (tensor dimension to vectorize; negative counts from the end).
    vec_size: int
    index: sympy.Symbol
    dim: int

    @staticmethod
    def from_fixed_inputs(
        fixed_inputs: dict[str, Any],
    ) -> "CuteDSLVectorLoadConfig | None":
        # [FUNCTION DOC] Pops the three vector_load_* keys (vec size, index name, dim) from the mapping (mutating it). Returns None if any is missing,
        # else a config with int-converted size and dim and the index name turned into an index symbol.
        vec_size = fixed_inputs.pop("vector_load_vec_size", None)
        index = fixed_inputs.pop("vector_load_index", None)
        dim = fixed_inputs.pop("vector_load_dim", None)
        if vec_size is None or index is None or dim is None:
            return None
        return CuteDSLVectorLoadConfig(
            vec_size=int(vec_size),
            index=sympy_index_symbol(str(index)),
            dim=int(dim),
        )

    def normalized_dim(self, ndim: int) -> int:
        # [FUNCTION DOC] Returns the vector dimension as a non-negative index for a tensor with ndim dims (adds ndim if negative).
        return self.dim + ndim if self.dim < 0 else self.dim


@dataclasses.dataclass(frozen=True, slots=True)
class CuteDSLIndexFragment:
    # [CLASS DOC] Frozen record describing generated index code for a captured-tensor load: code string, whether it is a static integer, whether lanes are
    # uniform, whether lanes are contiguous (and the proven contiguous_width, if known).
    """Generated index expression metadata for CuteDSL tensor loads.

    Attributes:
        code: Python source expression for the index or index fragment.
        is_static_int: True when code is an inline integer index, not a fragment.
        is_lane_uniform: True when every vector lane has the same index.
        is_contiguous: True when code is aligned and contiguous for the full
            requested vector width.
    """

    code: str
    is_static_int: bool
    is_lane_uniform: bool = False
    is_contiguous: bool = False
    contiguous_width: int | None = None


@dataclasses.dataclass(frozen=True)
class IndexSymbolInfo:
    # [CLASS DOC] Frozen record of what lane analysis may assume for a generated index symbol: the recovered semantic expression (None if unknown)
    # and whether contiguity is still valid (False after negative-index wrapping).
    """What lane analysis may assume about a generated index symbol.

    expr: Recovered semantic index expression, or None when the value's
        semantics are unknown (loaded from a captured tensor, or produced by
        ops outside the traced index algebra). Unknown semantics force a
        per-lane gather: assuming lane-uniform would broadcast lane 0's load
        across the vectorized group (#188878).
    contiguous_ok: False when the value passed through a negative-index wrap
        (idx + size for negative lanes only). Uniform lanes all wrap
        identically, so lane-uniformity survives, but a contiguous window
        straddling 0 jumps from size-1 back to 0 and we can't prove a window
        avoids 0.
    """

    expr: sympy.Expr | None
    contiguous_ok: bool = True


@dataclasses.dataclass
class CuteDSLSubgraphInfo:
    # [CLASS DOC] Dataclass bundling the per-subgraph kernel state that is swapped in and out while rendering a modification: body buffer,
    # template_mask, template_out and an optional CSE cache.
    """Minimal subgraph info for CuteDSL kernels."""

    body: IndentedBuffer
    template_mask: str | None = None
    template_out: str | None = None
    cse: CSE[Any, str] | None = None

    def __post_init__(self):
        # [FUNCTION DOC] Marks the cse field as only overwritten on the kernel when it is not None.
        self.only_copy_if_non_none_fields = ("cse",)

    def to_dict(self):
        # [FUNCTION DOC] Returns a shallow mapping from each dataclass field name to its value.
        return {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }


class CuteDSLTemplateKernel(Kernel):
    # [CLASS DOC] Kernel implementation used to render CuteDSL templates. Holds input/output buffer nodes, optional modification subgraphs, the main body
    # buffer, named subgraph bodies, render hooks (placeholder to deferred generator), CSE state, collected tensor buffer names, aux scalar
    # bindings and captured IR nodes. Template hooks (def_kernel, modification, etc.) are exposed to the template renderer; call_kernel emits the launch.
    """
    Template kernel implementation for CuteDSL (CUTLASS Python DSL).
    Handles code generation and argument management for CuteDSL CUDA kernels.
    Provides CuteDSL-specific functionality for tensor conversion and kernel configuration.
    """

    def __init__(
        self,
        kernel_name: str,
        input_nodes: list[Buffer],
        output_node: Buffer,
        subgraphs: list[Buffer] | None = None,
    ) -> None:
        # [FUNCTION DOC] Initializes kernel state: names, buffers, empty bodies/hooks, name-to-input mapping (using the node name or a positional fallback),
        # a fresh CSE with prefix tmp, empty tensor-buffer list, empty aux scalar bindings and empty capture map.
        # Call parent Kernel constructor
        super().__init__()
        self.kernel_name = kernel_name
        self.input_nodes = input_nodes
        self.output_node = output_node
        self.subgraphs = subgraphs
        self.subgraph_bodies: dict[str, CuteDSLSubgraphInfo] = {}

        # Template attributes
        self.body: IndentedBuffer = IndentedBuffer()
        self.template_mask: str | None = None
        self.template_out: str | None = None
        self.template_indices: list[Any] | None = None
        self.render_hooks: dict[str, Any] = {}

        # TODO Additional attributes needed by template system
        self.prologue_fused_inputs: OrderedSet[str] = OrderedSet()
        self.prologue_fused_inputs_preserve_zero: OrderedSet[str] = OrderedSet()
        self.named_input_nodes: dict[str, Buffer] = {}

        # Create named input nodes mapping
        for i, input_node in enumerate(input_nodes):
            node_name = getattr(input_node, "name", f"input_{i}")
            self.named_input_nodes[node_name] = input_node

        self.cse = CSE(name_prefix="tmp")

        # Track all tensor buffers added during modification processing
        self.collected_tensor_buffers: list[str] = []
        self.aux_scalar_bindings = CuteDSLAuxScalarBindings()

        # Captured IR nodes keyed by buffer name. Used by call_kernel to emit
        # reinterpret_tensor for view captures in the python_argdefs loop.
        self._capture_input_nodes: dict[str, Any] = {}

    def set_capture_input_nodes(self, nodes_by_name: dict[str, Any]) -> None:
        # [FUNCTION DOC] Replaces the mapping of captured input names to IR nodes.
        """Set captured inputs collected while rendering the CuteDSL template."""
        self._capture_input_nodes = nodes_by_name

    def _get_capture_input_node(self, name: str) -> Any | None:
        # [FUNCTION DOC] HELPER: Looks up a captured node by name in this kernel's captures first, then in the graph-level capture mapping (if any); None if absent.
        """Return the IR node for a captured input name, if known."""
        graph_captures = getattr(V.graph, "_cutedsl_capture_nodes", {})
        return self._capture_input_nodes.get(name, graph_captures.get(name))

    @contextlib.contextmanager
    def _patch_get_dtype_for_captures(self):
        # [FUNCTION DOC] HELPER: Context manager that temporarily replaces the graph's dtype lookup so captured synthetic names resolve to the captured node's dtype,
        # falling back to the original lookup. Restores it on exit.
        """Teach python_argdefs to resolve synthetic captured input names."""
        original_get_dtype = V.graph.get_dtype

        def get_dtype(name: str) -> torch.dtype:
            # [FUNCTION DOC] HELPER: Dtype lookup used by the patch: captured node dtype if known, else the original graph lookup.
            capture = self._get_capture_input_node(name)
            if capture is not None:
                return capture.get_dtype()
            return original_get_dtype(name)

        with patch.object(V.graph, "get_dtype", get_dtype):
            yield

    def kexpr(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Renders a sympy expression as CuteDSL source, first replacing any aux-scalar symbols (and their renamed forms) with their tuple lookups.
        """Convert sympy expression to CuteDSL string representation."""
        symbol_codes = self.aux_scalar_bindings.symbol_codes_with_renames(
            self.rename_indexing
        )
        if symbol_codes:
            expr = expr.xreplace(
                {symbol: sympy.Symbol(code) for symbol, code in symbol_codes.items()}
            )
        return cutedsl_pexpr(expr)

    def create_cse_var(self, *args, **kwargs):
        # [FUNCTION DOC] Creates a CuteDSL-specific CSE variable, forwarding all arguments.
        return CuteDSLCSEVariable(*args, **kwargs)

    def gen_imports(self) -> str:
        # [FUNCTION DOC] Returns the fixed import block (torch, cutlass, cute, dlpack, cuda driver bindings, MLIR math dialect, operator and the SSA conversion helpers) prepended to rendered templates.
        """Generate common imports for CuteDSL templates."""
        imports = IndentedBuffer()
        imports.splice(
            """
            import torch
            import cutlass
            import cutlass.cute as cute
            from cutlass.cute.runtime import from_dlpack
            import cuda.bindings.driver as cuda
            from cutlass._mlir.dialects import math as mlir_math
            import operator
            from torch._inductor.codegen.cutedsl._cutedsl_utils import result_to_ssa, ssa_to_fragment, ssa_to_indexable
            """
        )
        return imports.getvalue()

    def gen_defines(self, **kwargs) -> str:
        # [FUNCTION DOC] Returns one line per keyword declaring a compile-time constant parameter with the given value, in keyword order.
        """Generate CuteDSL parameter definitions from kwargs, similar to Triton's gen_defines."""
        params = IndentedBuffer()
        for name, val in kwargs.items():
            params.writeline(f"{name}: cutlass.Constexpr = {val}")
        return params.getvalue()

    def render(self, template, **kwargs):
        # [FUNCTION DOC] Renders the template: saves the kwargs, builds aux scalar bindings from the AUX_SCALAR_SYMBOLS kwarg, exposes helper hooks (def_kernel, gen_defines,
        # get_output, tensor buffer helpers, aux scalars, unpack_buffers, modification, set_cute_hash) to the template, renders it, prepends the imports and returns
        # a PartialRender holding the code and the registered render hooks.
        # KNOWN ISSUE: the string after the local import is not a docstring (it follows a statement), so it is just an unused expression.
        from torch._inductor.select_algorithm import PartialRender

        """Render the kernel using the template, returning PartialRender object with hooks."""
        self._template_kwargs = dict(kwargs)
        self.aux_scalar_bindings = CuteDSLAuxScalarBindings(
            tuple(kwargs.get("AUX_SCALAR_SYMBOLS", ()))
        )

        # Available {{}} hooks for jinja rendering
        template_env = {
            "def_kernel": self.def_kernel,
            "gen_defines": lambda: self.gen_defines(**kwargs),
            "get_output": self.get_output,
            "get_tensor_buffers": self.get_tensor_buffers,
            "add_tensor_inputs": self.add_tensor_inputs,
            "define_aux_scalars": self.define_aux_scalars,
            "unpack_buffers": self.unpack_buffers,
            "modification": self.modification,
            "set_cute_hash": self.set_cute_hash,
        }

        # Render the template with the environment and provided kwargs
        rendered_code = template.render(
            kernel_name=self.kernel_name,
            input_nodes=self.input_nodes,
            output_node=self.output_node,
            **template_env,
            **kwargs,
        )

        # Always prepend the common imports
        imports = self.gen_imports()
        full_code = imports + rendered_code

        return PartialRender(full_code, self.render_hooks)

    @contextlib.contextmanager
    def set_subgraph_body(self, body_name: str):
        # [FUNCTION DOC] Context manager: makes the named subgraph body active by copying its fields (body, mask, out, cse; a None cse is skipped) onto the kernel,
        # creating an empty entry if new. On exit, saves the current field values back into the named entry and restores the previous state.
        # Raises AssertionError if the kernel lacks any subgraph-info field.
        """Set the active subgraph body for template processing."""
        if not all(
            hasattr(self, field.name)
            for field in dataclasses.fields(CuteDSLSubgraphInfo)
        ):
            raise AssertionError(
                "expected all CuteDSLSubgraphInfo fields to be set on self"
            )
        old_state = {
            key.name: getattr(self, key.name)
            for key in dataclasses.fields(CuteDSLSubgraphInfo)
        }

        if body_name not in self.subgraph_bodies:
            self.subgraph_bodies[body_name] = CuteDSLSubgraphInfo(
                body=IndentedBuffer(),
                template_mask=None,
                template_out=None,
                cse=None,
            )

        subgraph = self.subgraph_bodies[body_name]
        for key, value in subgraph.to_dict().items():
            if value is None and key in getattr(
                subgraph, "only_copy_if_non_none_fields", ()
            ):
                continue
            setattr(self, key, value)

        try:
            yield
        finally:
            # Save current state back to subgraph
            self.subgraph_bodies[body_name] = CuteDSLSubgraphInfo(
                **{
                    key.name: getattr(self, key.name)
                    for key in dataclasses.fields(CuteDSLSubgraphInfo)
                }
            )
            # Restore old state
            for key, value in old_state.items():
                setattr(self, key, value)

    @contextlib.contextmanager
    def create_subgraph_body(self, body_name: str, *, clear_cse: bool = False):
        # [FUNCTION DOC] Context manager: registers a new empty subgraph body (with a cloned CSE if clear_cse, else none) then activates it. Raises AssertionError if the name already exists.
        """Create a new subgraph body for template processing."""
        if body_name in self.subgraph_bodies:
            raise AssertionError(f"Subgraph body '{body_name}' already exists")
        new_cse = self.cse.clone() if clear_cse else None
        self.subgraph_bodies[body_name] = CuteDSLSubgraphInfo(
            body=IndentedBuffer(),
            template_mask=None,
            template_out=None,
            cse=new_cse,
        )
        with self.set_subgraph_body(body_name):
            yield

    def _get_reinterpret_view(self, node) -> ReinterpretView | None:
        # [FUNCTION DOC] HELPER: Unwraps mutable boxes; if the result is a view returns its reinterpret-view conversion, else None.
        """Extract or convert to ReinterpretView from a node, handling all views."""
        while isinstance(node, MutableBox):
            node = node.data
        if isinstance(node, BaseView):
            return ExternKernel.convert_to_reinterpret_view(node)
        return None

    def def_kernel(self, *argnames):
        # [FUNCTION DOC] Template hook defining the kernel signature. Registers each input node as an arg named with the prefix arg_ plus the template's name (extra inputs beyond argnames
        # are registered but not named), records them, and registers the output. Installs a deferred hook that renders the def line (template args, then remaining
        # args such as output/sizevars, then stream) plus renames from template names to arg names. Returns the placeholder text; raises AssertionError if already registered.
        """Define kernel function signature for CuteDSL templates.

        When inputs are ReinterpretViews of the same underlying buffer (e.g., Q/K/V
        from fused QKV projection), we generate separate arguments for each input
        even though they share the same underlying buffer.
        """
        renames = IndentedBuffer(initial_indent=1)

        # Track template input args - each input gets its own arg even if buffers are shared
        self._template_input_args: list[tuple[str, Buffer]] = []
        self._seen_input_args: OrderedSet[str] = OrderedSet()

        for i, input_node in enumerate(self.input_nodes):
            buf_name = input_node.get_name()
            # Register with args system (may deduplicate, but we track separately)
            self.args.input(buf_name)

            if i < len(argnames):
                template_name = argnames[i]
                arg_name = f"arg_{template_name}"
                self.args.input_buffers[buf_name] = arg_name
                renames.writeline(f"{template_name} = {arg_name}")
                self._template_input_args.append((arg_name, input_node))
                self._seen_input_args.add(arg_name)

        if self.output_node:
            self.args.output(self.output_node.get_name())

        def hook():
            # [FUNCTION DOC] HELPER: Deferred renderer for the kernel def line: template input arg names, then other arg names not already used, then stream;
            # body holds renames from template names to arg names. Returns the code text.
            # Generate signature with template input args plus additional args (output, sizevars)
            code = IndentedBuffer()
            code.writeline(f"# Kernel function signature: {self.kernel_name}")

            # Start with template input args
            params = [arg_name for arg_name, _ in self._template_input_args]

            # Get additional args from python_argdefs (output, sizevars, etc.)
            with self._patch_get_dtype_for_captures():
                arg_defs, _, _, _ = self.args.python_argdefs()
            for arg_def in arg_defs:
                if arg_def.full_name() not in self._seen_input_args:
                    params.append(arg_def.full_name())

            params.append("stream")
            code.writeline(
                f"def {self.kernel_name}_{MAIN_SUFFIX}({', '.join(params)}):"
            )
            with code.indent():
                code.splice(renames.getvalue())
            return code.getvalue()

        if "<DEF_KERNEL>" in self.render_hooks:
            raise AssertionError("<DEF_KERNEL> hook already registered")
        # Placeholder-based rendering: hook will be called when template encounters "<DEF_KERNEL>"
        self.render_hooks["<DEF_KERNEL>"] = hook
        return "<DEF_KERNEL>"

    def get_output(self):
        # [FUNCTION DOC] Returns the argument name of the output buffer. Raises AssertionError if there is no output node, ValueError if it is not among the output args.
        """Get the actual argument name for the output buffer."""
        if not self.output_node:
            raise AssertionError("Output node must exist to get output buffer name")
        buf_name = self.output_node.get_name()
        output = self.args.output_buffers.get(buf_name, None)
        if output is None:
            raise ValueError(f"Output buffer '{buf_name}' not found in args")
        return output

    def set_cute_hash(self, func_name: str, suffix: str = ""):
        # [FUNCTION DOC] Returns a source line assigning a hash attribute to the named function, using the kernel name (plus suffix if given), so flash-attention can skip runtime hashing.
        """Generate code to set __cute_hash__ on a codegen function.

        This allows hash_callable in flash_attn to skip expensive runtime hashing
        for Inductor-generated functions. The hash is based on the kernel name
        which already contains a unique hash suffix.
        """
        hash_value = f"{self.kernel_name}_{suffix}" if suffix else self.kernel_name
        return f'{func_name}.__cute_hash__ = "{hash_value}"'

    def get_tensor_buffers(self):
        # [FUNCTION DOC] Returns the list of collected tensor buffer arg names.
        """Get list of tensor buffer names that were collected during modifications."""
        return self.collected_tensor_buffers

    def add_tensor_inputs(self, buffers):
        # [FUNCTION DOC] Registers each non-symbolic buffer as a kernel input, records its remapped name once in the collected list, and returns the remapped names in order; sympy entries are skipped.
        """Register extra tensor buffers and return their rendered input names."""
        buffer_names = []
        for buffer in buffers:
            if isinstance(buffer, sympy.Expr):
                continue
            remapped_name = self.args.input(buffer.get_name())
            if remapped_name not in self.collected_tensor_buffers:
                self.collected_tensor_buffers.append(remapped_name)
            buffer_names.append(remapped_name)
        return buffer_names

    def define_aux_scalars(self, symbols):
        # [FUNCTION DOC] Returns the source of the runtime scalar tuple; raises AssertionError if the given symbols differ from those bound at render time.
        """Return the runtime scalar tuple for captured symbolic scalars."""
        if tuple(symbols) != self.aux_scalar_bindings.symbols:
            raise AssertionError(
                f"Expected aux scalar symbols {self.aux_scalar_bindings.symbols}, "
                f"got {tuple(symbols)}"
            )
        return self.aux_scalar_bindings.tuple_expr(self.rename_indexing, cutedsl_pexpr)

    def unpack_buffers(self, buffer_list_name: str, *, indent_width: int = 4):
        # [FUNCTION DOC] Registers a deferred hook (placeholder returned) that expands to one assignment per collected tensor buffer, taking elements by position from the named list,
        # indented by indent_width; empty text if none. Re-registration overwrites.
        """Generate buffer unpacking code via render hook."""

        def hook():
            # [FUNCTION DOC] HELPER: Deferred renderer: returns assignments unpacking each collected tensor buffer from the list by position, or empty text if none.
            tensor_buffers = self.get_tensor_buffers()
            if not tensor_buffers:
                return ""

            # Generate unpacking assignments: in_ptr4 = buffers[0], etc.
            unpacking_lines = []
            for i, buffer_name in enumerate(tensor_buffers):
                # pyrefly: ignore [bad-argument-type]
                unpacking_lines.append(f"{buffer_name} = {buffer_list_name}[{i}]")

            indent = " " * indent_width
            return "\n" + indent + ("\n" + indent).join(unpacking_lines)

        # Register the hook and return placeholder
        placeholder = "<UNPACK_BUFFERS>"
        # TODO: I think double invoking is fine for this specific hook
        # assert placeholder not in self.render_hooks
        self.render_hooks[placeholder] = hook
        return placeholder

    def call_kernel(self, name: str, node=None):
        # [FUNCTION DOC] Emits the kernel call in the wrapper: template inputs first (reinterpret-view reference if the input is a view, else its name) then remaining args from the arg
        # definitions (skipping already-passed inputs; captured view nodes use their view reference), with dtypes. Uses the Triton-style call generation path.
        """Call the kernel function. Simplified version of TritonTemplateKernel.call_kernel.

        For inputs that are ReinterpretViews (e.g., Q/K/V slices from fused QKV),
        we generate reinterpret_tensor() calls to properly handle the views.
        """
        wrapper = V.graph.wrapper_code

        # Build call args matching the signature generated in `def_kernel`
        call_args = []
        arg_types = []

        for _, input_node in self._template_input_args:
            reinterpret_view = self._get_reinterpret_view(input_node)
            if reinterpret_view is not None:
                call_args.append(reinterpret_view.codegen_reference())
            else:
                call_args.append(input_node.get_name())
            arg_types.append(V.graph.get_dtype(input_node.get_name()))

        # Add additional args from python_argdefs (output, sizevars, ..)
        with self._patch_get_dtype_for_captures():
            orig_arg_defs, orig_call_args, _, orig_arg_types = (
                self.args.python_argdefs()
            )
        for arg_def, call_arg, arg_type in zip(
            orig_arg_defs, orig_call_args, orig_arg_types
        ):
            if arg_def.full_name() in self._seen_input_args:
                continue
            capture = self._get_capture_input_node(call_arg)
            if isinstance(capture, ReinterpretView):
                # Pass the original view expression so stride/offset metadata is
                # preserved at the kernel call site.
                call_args.append(capture.codegen_reference())
            else:
                call_args.append(call_arg)
            arg_types.append(arg_type)

        # TODO this karg really should not be called `triton`
        wrapper.generate_kernel_call(name, call_args, triton=True, arg_types=arg_types)

    def _get_subgraph(self, subgraph_number: int):
        # [FUNCTION DOC] HELPER: Validates the subgraph number is an int in range of the subgraph list and that the body is empty (AssertionError otherwise); returns that subgraph.
        """Get subgraph by number for modification processing."""
        if not isinstance(subgraph_number, int):
            raise AssertionError(
                f"expected subgraph_number to be int, got {type(subgraph_number)}"
            )
        if not isinstance(self.subgraphs, list):
            raise AssertionError(
                f"expected self.subgraphs to be list, got {type(self.subgraphs)}"
            )
        if subgraph_number >= len(self.subgraphs):
            raise AssertionError(
                f"Invalid subgraph number provided to create_modification, {subgraph_number} must be < {len(self.subgraphs)}"
            )
        if self.body.getvalue() != "":
            raise AssertionError("Body should be clear before adding a modification")
        return self.subgraphs[subgraph_number]

    def modification(
        self,
        subgraph_number: int,
        output_name: str | None,
        mask: str | None = None,
        **fixed_inputs,
    ) -> str:
        # [FUNCTION DOC] Template hook generating code for a subgraph modification. Picks an unused body name, renders the subgraph pointwise under a ModificationWrapperCuteDSL handler
        # (fixed_inputs substitute placeholders), assigns the result to output_name and collects tensor buffers touched. Returns the body text.
        # Raises AssertionError for non-ComputedBuffer subgraphs or missing result, NotImplementedError for list (scatter) subgraphs or when output_name is None.
        """Generate CuteDSL code for a subgraph modification."""
        # Find unique name to avoid collisions between multiple modifications of same subgraph
        num = 0
        while f"mod_{subgraph_number}_{num}" in self.subgraph_bodies:
            num += 1

        with self.create_subgraph_body(f"mod_{subgraph_number}_{num}", clear_cse=True):
            subgraph = self._get_subgraph(subgraph_number)
            modification_handler = ModificationWrapperCuteDSL(
                self, subgraph_number, fixed_inputs, mask
            )
            with V.set_kernel_handler(self), V.set_ops_handler(modification_handler):
                if not isinstance(subgraph, (ComputedBuffer, list)):
                    raise AssertionError(
                        f"Expected ComputedBuffer or List[ComputedBuffer], got {type(subgraph)}"
                    )

                if isinstance(subgraph, list):
                    raise NotImplementedError(
                        "Scatter graphs are not supported for CuteDSL"
                    )

                if isinstance(subgraph.data, InputBuffer):
                    # grad_score_mod can be InputBuffers
                    out = subgraph.data.make_loader()(())
                else:
                    # Inline a pointwise lowering into the template
                    out = subgraph.data.inner_fn(())

            if output_name is not None:
                if out is None:
                    raise AssertionError(
                        f"Expected computation result for named output {output_name}"
                    )
                self.body.writeline(f"{output_name} = {out.value}")
            else:
                # Side-effect only: no output assignment (currently only for scatter operations)
                raise NotImplementedError(
                    "Side-effect only modifications not yet supported for CuteDSL"
                )

            # Add Buffers that were added during modification
            self.collected_tensor_buffers.extend(modification_handler.tensor_buffers)

            return self.body.getvalue()


class ModificationWrapperCuteDSL(V.WrapperHandler):  # type: ignore[name-defined]
    # [CLASS DOC] Ops handler wrapper used while inlining a modification subgraph: delegates ops to CuteDSLOpOverrides, resolves fixed (template-supplied) inputs on load,
    # emits captured-tensor loads (static, uniform, contiguous-vector or per-lane gather), tracks index symbol semantics for lane analysis and caches repeated
    # captured loads. Read-only: stores are unsupported. Holds kernel, fixed inputs, optional vector load config, mask, tensor_buffers and caches.
    """
    Wrapper handler that enables CuteDSL code generation during subgraph modifications.

    This class sits between the PyTorch IR and CuteDSL code generation, providing:
    1. Operation substitution: converts PyTorch ops to CuteDSL equivalents via CuteDSLOpOverrides
    2. Placeholder handling: resolves fixed_inputs during template processing
    3. Limited operation support: currently restricted to pointwise operations

    """

    def __init__(
        self,
        kernel,
        subgraph_number: int,
        fixed_inputs: dict[str, Any],
        mask: str | None,
    ):
        # [FUNCTION DOC] Sets up the handler with CuteDSL op overrides, a name, the kernel, a copy of fixed inputs (vector-load keys are popped into a config), mask, and empty tracking
        # structures for tensor buffers, index symbol info and the captured-load cache.
        cutedsl_ops = CuteDSLOpOverrides()
        super().__init__(cutedsl_ops)
        self.name = f"CuteDSLPlaceholderSubstitution_{subgraph_number}"
        self.kernel = kernel
        self.fixed_inputs = dict(fixed_inputs)
        self.vector_load_config = CuteDSLVectorLoadConfig.from_fixed_inputs(
            self.fixed_inputs
        )
        self.mask = mask
        # Track tensor buffers that get added during modification processing
        self.tensor_buffers: list[str] = []
        # Maps generated CSE names to their semantic provenance so vector-load
        # analysis can reason about the source indexing semantics.
        self._index_symbol_info: dict[sympy.Symbol, IndexSymbolInfo] = {}
        # Captured modifications are read-only, so repeated loads from the same
        # semantic location can share the generated index and value fragments.
        self._captured_load_cache: dict[tuple[str, sympy.Expr], CuteDSLCSEVariable] = {}

    def _get_input_dtype(self, name: str) -> torch.dtype:
        # [FUNCTION DOC] HELPER: dtype of a named input from the kernel's inputs; otherwise a fallback guess: int32 for the names b, h, m, n, else float32.
        """Get the dtype for an input from the kernel's named_input_nodes."""
        if name in self.kernel.named_input_nodes:
            return self.kernel.named_input_nodes[name].dtype
        # TODO: Fallback for common dimension names - should be replaced with proper dtype tracking
        return torch.float32 if name not in ("b", "h", "m", "n") else torch.int32

    def load(self, name: str, index: sympy.Expr):
        # [FUNCTION DOC] Handles a load. For a non-fixed name (captured tensor): returns the cached value if the same name/index was loaded; otherwise registers the input, builds Int32 index
        # fragments per dimension (hierarchical index splits per dim; scalar buffers have none), emits the fragment load and a load into SSA, upcasts half types to float32 when
        # configured, caches and returns the result. For a fixed input: returns a CSE variable holding the fixed value, tagging integer string values with their index expression.
        """Handle loading from tensor or fixed(template args) input for CuteDSL."""
        from torch._inductor.kernel.flex.flex_flash_attention import HierarchicalIndex

        if name not in self.fixed_inputs:
            cache_key = (name, index)
            cached = self._captured_load_cache.get(cache_key)
            if cached is not None:
                return self.kernel.cse.generate(self.kernel.body, cached)

            var = self._add_kernel_input(name)
            buffer = self.kernel._get_capture_input_node(name)
            if buffer is None:
                buffer = V.graph.get_buffer(name)
            var_dtype = buffer.dtype

            cute_dtype = CuteDSLOpOverrides.TORCH_TO_CUTE_DTYPE.get(
                var_dtype, "cutlass.Float32"
            )
            dim_indices = (
                index.args if isinstance(index, HierarchicalIndex) else (index,)
            )
            dim_sizes = buffer.get_size()
            if len(dim_sizes) == 0:
                dim_indices = ()
            # Per-dimension indices into captured buffers are bounded by dim
            # sizes (not flattened offsets), so Int32 is safe whatever index
            # dtype the consuming kernel passes in; wider index expressions are
            # narrowed at fragment materialization.
            idx_vars = [
                self._emit_index_fragment(
                    dim_index, dim_size, "cutlass.Int32", torch.int32
                )
                for dim_index, dim_size in zip(dim_indices, dim_sizes, strict=True)
            ]

            val_frag = self._emit_tensor_load_fragment(
                var, idx_vars, cute_dtype, var_dtype, buffer
            )
            final_expr = f"{val_frag}.load()"

            if (
                var_dtype in (torch.float16, torch.bfloat16)
                and config.triton.codegen_upcast_to_fp32
            ):
                final_expr = f"({final_expr}).to(cutlass.Float32)"
                var_dtype = torch.float32

            out = self.kernel.cse.generate(
                self.kernel.body,
                final_expr,
                dtype=var_dtype,
                bounds=ValueRanges.unknown(),
                shape=(1,),
            )
            self._captured_load_cache[cache_key] = out
            return out

        value = self.fixed_inputs[name]
        dtype = self._get_input_dtype(name)

        result = self.kernel.cse.generate(
            self.kernel.body, value, bounds=ValueRanges.unknown(), dtype=dtype
        )
        if (
            isinstance(result, CuteDSLCSEVariable)
            and isinstance(value, str)
            and dtype in (torch.int32, torch.int64)
        ):
            result.index_expr = sympy_index_symbol(value)
        return result

    def _emit_tensor_load_fragment(
        self,
        var: str,
        idx_vars: list[CuteDSLIndexFragment],
        cute_dtype: str,
        var_dtype: torch.dtype,
        buffer: Any,
    ) -> str:
        # [FUNCTION DOC] HELPER: Emits code creating a register fragment holding values loaded from a captured tensor and returns its name. Scalar buffers and all-static indices give a single-element
        # fragment; otherwise the fragment is sized to the dynamic index lanes and filled by the contiguous vector load, uniform load, or per-lane gather (first applicable).
        """
        Emit a register fragment load from a captured tensor.

        Static integer indices can be emitted directly as scalar tensor indices.
        Dynamic indices are SSA vectors, which must first be materialized into
        register fragments so each lane can be used for scalar gather loads.  If
        all indices are static, emit a single-element fragment; otherwise size
        the value fragment to match the dynamic index lanes.
        """
        dynamic_index_fragment = next(
            (idx for idx in idx_vars if not idx.is_static_int), None
        )
        # Dynamic index fragments all carry the same vector width, so one
        # representative fragment is enough to size the loaded value fragment.
        val_frag = self.kernel.cse.newvar(dtype=var_dtype)
        if len(buffer.get_size()) == 0:
            self.kernel.body.writeline(
                f"{val_frag} = cute.make_rmem_tensor(1, {cute_dtype})"
            )
            self.kernel.body.writeline(f"{val_frag}[0] = ({var}[()])")
            return str(val_frag)
        if dynamic_index_fragment is None:
            self.kernel.body.writeline(
                f"{val_frag} = cute.make_rmem_tensor(1, {cute_dtype})"
            )
            self.kernel.body.writeline(
                f"{val_frag}[0] = ({var}[{', '.join(idx.code for idx in idx_vars)}])"
            )
            return str(val_frag)

        self.kernel.body.writeline(
            f"{val_frag} = cute.make_rmem_tensor("
            f"cute.size({dynamic_index_fragment.code}.shape), {cute_dtype})"
        )
        if self.vector_load_config is None:
            vector_dim = len(idx_vars) - 1
        else:
            vector_dim = self.vector_load_config.normalized_dim(len(idx_vars))
        if self._can_emit_contiguous_dim_load(idx_vars, buffer, var_dtype, vector_dim):
            self._emit_contiguous_dim_load(
                var, idx_vars, val_frag, cute_dtype, var_dtype, vector_dim
            )
        elif self._can_emit_uniform_load(idx_vars):
            self._emit_uniform_load(var, idx_vars, val_frag)
        else:
            self._emit_per_lane_gather_load(var, idx_vars, val_frag)
        return str(val_frag)

    def _can_emit_contiguous_dim_load(
        self,
        idx_vars: list[CuteDSLIndexFragment],
        buffer: Any,
        var_dtype: torch.dtype,
        vector_dim: int,
    ) -> bool:
        # [FUNCTION DOC] HELPER: True only if an autovectorized copy is safe: non-bool dtype, contiguous lanes in the vector dim, other dims static or uniform, vector width above 1, unit stride, proven
        # contiguous width at least the width, and statically known alignment of size, layout offset and the other strides to the vector size.
        """Return true when vector lanes form a safe contiguous tensor tile.

        This is stricter than lane-contiguity analysis: CuTe autovec also needs
        a real byte-addressable copy type, uniform non-vector dimensions, unit
        vector stride, and a statically aligned full-width tile.
        """
        if var_dtype is torch.bool:
            return False
        if not idx_vars or not idx_vars[vector_dim].is_contiguous:
            return False
        if any(
            not (idx.is_static_int or idx.is_lane_uniform)
            for dim, idx in enumerate(idx_vars)
            if dim != vector_dim
        ):
            return False
        sizes = buffer.get_size()
        strides = buffer.get_stride()
        vector_size = (
            1 if self.vector_load_config is None else self.vector_load_config.vec_size
        )
        contiguous_width = idx_vars[vector_dim].contiguous_width
        has_vector_width = vector_size > 1
        has_unit_stride = V.graph.sizevars.statically_known_equals(
            strides[vector_dim], 1
        )
        has_full_width_contiguity = (
            contiguous_width is not None and contiguous_width >= vector_size
        )
        is_size_aligned = V.graph.sizevars.statically_known_multiple_of(
            sizes[vector_dim], vector_size
        )
        is_offset_aligned = V.graph.sizevars.statically_known_multiple_of(
            buffer.get_layout().offset, vector_size
        )
        are_non_vector_strides_aligned = all(
            V.graph.sizevars.statically_known_multiple_of(stride, vector_size)
            for dim, stride in enumerate(strides)
            if dim != vector_dim
        )
        return (
            has_vector_width
            and has_unit_stride
            and has_full_width_contiguity
            and is_size_aligned
            and is_offset_aligned
            and are_non_vector_strides_aligned
        )

    def _emit_contiguous_dim_load(
        self,
        var: str,
        idx_vars: list[CuteDSLIndexFragment],
        val_frag: str,
        cute_dtype: str,
        var_dtype: torch.dtype,
        vector_dim: int,
    ) -> None:
        # [FUNCTION DOC] HELPER: Emits code for an aligned tile load: takes lane 0 of the vector index (assumed divisible by the width), makes a tile view of the source, rebuilds an aligned pointer/tensor with
        # alignment at most 16 bytes, and copies it into the fragment with autovec copy.
        """Emit an autovec load for lanes contiguous in one tensor dimension."""
        scalar_indices = [
            idx.code if idx.is_static_int else f"{idx.code}[0]" for idx in idx_vars
        ]
        vec_size = f"cute.size({val_frag}.shape)"
        aligned_idx = self.kernel.cse.newvar(dtype=torch.int32)
        source_tile = self.kernel.cse.newvar(dtype=var_dtype)
        aligned_ptr = self.kernel.cse.newvar(dtype=var_dtype)
        aligned_source = self.kernel.cse.newvar(dtype=var_dtype)

        vector_idx = idx_vars[vector_dim].code
        tiler_items = ["1"] * len(idx_vars)
        tiler_items[vector_dim] = vec_size
        coord_items = scalar_indices
        coord_items[vector_dim] = f"{aligned_idx} // {vec_size}"
        source_view_items = ["0"] * len(idx_vars)
        source_view_items[vector_dim] = "None"
        source_view = (
            str(aligned_source)
            if len(idx_vars) == 1
            else f"{aligned_source}[{', '.join(source_view_items)}]"
        )
        tiler = (
            f"({tiler_items[0]},)"
            if len(tiler_items) == 1
            else f"({', '.join(tiler_items)})"
        )
        coord = (
            f"({coord_items[0]},)"
            if len(coord_items) == 1
            else f"({', '.join(coord_items)})"
        )
        self.kernel.body.splice(
            f"""
            {aligned_idx} = cute.assume({vector_idx}[0], divby={vec_size})
            {source_tile} = cute.local_tile({var}, {tiler}, {coord})
            {aligned_ptr} = cute.make_ptr({cute_dtype}, {source_tile}.iterator.toint(), {source_tile}.iterator.memspace, assumed_align=min(16, {vec_size} * {var_dtype.itemsize}))
            {aligned_source} = cute.make_tensor({aligned_ptr}, {source_tile}.layout)
            cute.autovec_copy({source_view}, {val_frag})
            """
        )

    @staticmethod
    def _can_emit_uniform_load(idx_vars: list[CuteDSLIndexFragment]) -> bool:
        # [FUNCTION DOC] HELPER: True if every index is a static integer or lane-uniform.
        """Return true when every vector lane reads the same tensor element."""
        return all(idx.is_static_int or idx.is_lane_uniform for idx in idx_vars)

    def _emit_uniform_load(
        self, var: str, idx_vars: list[CuteDSLIndexFragment], val_frag: str
    ) -> None:
        # [FUNCTION DOC] HELPER: Emits one scalar load (element 0 of non-static indices) and a loop that stores it into every element of the fragment.
        """Emit one scalar load and broadcast it across the value fragment."""
        scalar_value = self.kernel.cse.newvar(dtype=None)
        scalar_indices = [
            idx.code if idx.is_static_int else f"{idx.code}[0]" for idx in idx_vars
        ]
        self.kernel.body.writeline(
            f"{scalar_value} = ({var}[{', '.join(scalar_indices)}])"
        )
        self.kernel.body.writeline(
            f"for load_idx in cutlass.range(cute.size({val_frag}.shape), unroll_full=True):"
        )
        with self.kernel.body.indent():
            self.kernel.body.writeline(f"{val_frag}[load_idx] = {scalar_value}")

    def _emit_per_lane_gather_load(
        self, var: str, idx_vars: list[CuteDSLIndexFragment], val_frag: str
    ) -> None:
        # [FUNCTION DOC] HELPER: Emits a loop over fragment elements doing one scalar load per element; static indices inline, lane-uniform indices use element 0, others use the loop element.
        """Emit scalar per-lane loads for non-contiguous gather indices.

        Lane-uniform index fragments hold a single element, so they are indexed
        at [0] rather than [load_idx] (which would read out of bounds and
        silently corrupt the gathered rows).
        """
        self.kernel.body.writeline(
            f"for load_idx in cutlass.range(cute.size({val_frag}.shape), unroll_full=True):"
        )
        with self.kernel.body.indent():
            load_indices = [
                idx.code
                if idx.is_static_int
                else f"{idx.code}[0]"
                if idx.is_lane_uniform
                else f"{idx.code}[load_idx]"
                for idx in idx_vars
            ]
            self.kernel.body.writeline(
                f"{val_frag}[load_idx] = ({var}[{', '.join(load_indices)}])"
            )

    def _emit_index_fragment(
        self,
        expr: sympy.Expr,
        dim_size: sympy.Expr,
        cute_dtype: str,
        torch_dtype: torch.dtype,
    ) -> CuteDSLIndexFragment:
        # [FUNCTION DOC] HELPER: Produces index metadata for one dimension. A static integer (negative values wrapped by dim size) is inlined. Otherwise the expression is renamed, analyzed for lane
        # uniformity/contiguity, and emitted as a fragment materialized from the SSA value in the given dtype.
        """Materialize an index and preserve enough metadata for load selection."""
        static_expr = self._static_integer_expr(expr)
        if static_expr is not None:
            return CuteDSLIndexFragment(
                self.kernel.kexpr(self._wrap_static_index(static_expr, dim_size)),
                is_static_int=True,
            )

        expr = self.kernel.rename_indexing(expr)
        is_lane_uniform, is_contiguous, contiguous_width = self._analyze_index_fragment(
            expr
        )
        result = self.kernel.cse.newvar(dtype=torch_dtype)
        self.kernel.body.writeline(
            f"{result} = ssa_to_fragment({self.kernel.kexpr(expr)}, {cute_dtype})"
        )
        return CuteDSLIndexFragment(
            str(result),
            is_static_int=False,
            is_lane_uniform=is_lane_uniform,
            is_contiguous=is_contiguous,
            contiguous_width=contiguous_width,
        )

    def _analyze_index_fragment(
        self, expr: sympy.Expr
    ) -> tuple[bool, bool, int | None]:
        # [FUNCTION DOC] HELPER: Returns (lane uniform, lane contiguous, contiguous width) for an index expression. Without a vector config: all unknown/false. Vec size 1: uniform. Unknown symbol semantics or
        # unresolved computed symbols give the pessimistic answer; otherwise lane classification of the recovered semantic expression, with contiguity dropped if a negative-index wrap was involved.
        if self.vector_load_config is None:
            return False, False, None
        if self.vector_load_config.vec_size <= 1:
            return True, False, None

        contiguous_ok = True
        for sym in expr.free_symbols:
            info = self._index_symbol_info.get(sym)
            if info is not None:
                if info.expr is None:
                    # If we dont know the semantics of the index, we cannot determine lane behavior.
                    # So we default to safe/pessimistic behavior: per-lane gather, no uniformity, no contiguity.
                    return False, False, None
                contiguous_ok &= info.contiguous_ok

        semantic_expr = self._semantic_index_expr(expr)
        if self._has_unresolved_index_symbols(semantic_expr):
            # ditto -> we dont know so choose safe/pessimistic behavior
            return False, False, None

        lane_info = classify_lane_expr(
            semantic_expr,
            self.vector_load_config.index,
            max_width=self.vector_load_config.vec_size,
            uniform_symbols=self._lane_uniform_symbols(),
        )
        return (
            lane_info.is_uniform,
            lane_info.is_contiguous and contiguous_ok,
            lane_info.contiguous_width if contiguous_ok else None,
        )

    def _has_unresolved_index_symbols(self, semantic_expr: sympy.Expr) -> bool:
        # [FUNCTION DOC] HELPER: True if the expression still contains a generated CSE-named symbol (prefix followed by digits) other than the lane variable, meaning semantics could not be recovered.
        """Return whether the expression references opaque computed indices.

        Lane analysis treats unknown non-lane symbols as lane-uniform scalars,
        which is correct for sizes and fixed inputs but silently broadcasts
        lane 0's gather when the symbol is actually per-lane runtime data
        (#188878). Recovered semantics substitute away generated CSE names, so
        a surviving symbol in the reserved ``tmpN`` namespace means recovery
        failed and the load must stay a per-lane gather.
        """
        prefix = self.kernel.cse.name_prefix
        lane_var = self.vector_load_config.index if self.vector_load_config else None
        return any(
            sym is not lane_var
            and sym.name.startswith(prefix)
            and sym.name[len(prefix) :].isdigit()
            for sym in semantic_expr.free_symbols
        )

    @staticmethod
    def _static_integer_expr(expr: object) -> int | sympy.Integer | None:
        # [FUNCTION DOC] HELPER: Returns the expression as an integer if it is an int/Integer, a symbol whose name is a (possibly negative) integer literal, or a sympy expression that simplifies to an Integer; else None.
        if isinstance(expr, int | sympy.Integer):
            return expr
        if isinstance(expr, sympy.Symbol) and expr.name.lstrip("-").isdigit():
            return int(expr.name)
        if isinstance(expr, sympy.Expr):
            expr = V.graph.sizevars.simplify(expr)
            if expr.is_Integer:
                return expr
        return None

    @staticmethod
    def _wrap_static_index(
        index: int | sympy.Integer, dim_size: sympy.Expr
    ) -> int | sympy.Expr:
        # [FUNCTION DOC] HELPER: Wraps a negative static index by adding the dimension size (simplified); non-negative indices are returned unchanged.
        if int(index) < 0:
            return V.graph.sizevars.simplify(sympy.Integer(index) + dim_size)
        return index

    def _semantic_index_expr(self, expr: sympy.Expr) -> sympy.Expr:
        # [FUNCTION DOC] HELPER: Substitutes known semantic expressions for generated symbols (from recorded index info and CSE variables carrying an index expression) and simplifies.
        """Recover the original index meaning from generated CSE names."""
        replacements: dict[sympy.Symbol, sympy.Expr] = {
            symbol: info.expr
            for symbol, info in self._index_symbol_info.items()
            if info.expr is not None
        }
        for var in self.kernel.cse._cache.values():
            if isinstance(var, CuteDSLCSEVariable) and var.index_expr is not None:
                replacements[sympy_index_symbol(var.name)] = var.index_expr
        return V.graph.sizevars.simplify(expr.xreplace(replacements))

    def _lane_uniform_symbols(self) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] HELPER: Returns index symbols for all string-keyed CSE cache entries other than the vector lane index, treated as lane-uniform.
        """Return generated CSE source symbols that are uniform across lanes."""
        vector_index_name = (
            None
            if self.vector_load_config is None
            else self.vector_load_config.index.name
        )
        return OrderedSet(
            [
                sympy_index_symbol(source_expr)
                for source_expr in self.kernel.cse._cache
                if isinstance(source_expr, str) and source_expr != vector_index_name
            ]
        )

    def indirect_indexing(self, index_var: str, size, check, wrap_neg=True):
        # [FUNCTION DOC] Converts an index value to a symbolic index. Integer inputs are wrapped (when wrap_neg and negative) and returned as Integer. A CSE variable with a known index expression that may be negative gets a generated conditional
        # wrap (add the size cast to the index dtype when negative) and a recorded info entry with contiguity disabled. Otherwise returns a symbol for the variable's source name and records its semantic expression (or None).
        # The check argument is unused.
        """Convert index variable to symbolic form."""
        if isinstance(index_var, int | sympy.Integer):
            if wrap_neg and index_var < 0:
                return V.graph.sizevars.simplify(sympy.Integer(index_var) + size)
            return sympy.Integer(index_var)
        if (
            wrap_neg
            and isinstance(index_var, CuteDSLCSEVariable)
            and index_var.index_expr is not None
            and not V.graph.sizevars.statically_known_geq(index_var.index_expr, 0)
        ):
            wrapped = self.kernel.cse.newvar(dtype=index_var.dtype)
            # Dynamic sizes render as Int64 aux_scalars; cast to the index dtype so
            # both cute.where branches agree (per-dim size, so narrowing is safe).
            index_cute_dtype = CuteDSLOpOverrides.TORCH_TO_CUTE_DTYPE.get(
                index_var.dtype, "cutlass.Int32"
            )
            size_expr = f"{index_cute_dtype}({self.kernel.kexpr(size)})"
            self.kernel.body.writeline(
                f"{wrapped} = cute.where("
                f"{index_var} < 0, {index_var} + {size_expr}, {index_var})"
            )
            symbol = sympy_index_symbol(str(wrapped))
            # Preserve the pre-wrap semantics so lane analysis sees the true
            # lane structure; the wrap remaps negative lanes individually, so
            # contiguity of the pre-wrap expression does not survive it.
            self._index_symbol_info[symbol] = IndexSymbolInfo(
                index_var.index_expr, contiguous_ok=False
            )
            return symbol
        source_name = None
        for expr, var in self.kernel.cse._cache.items():
            if var is index_var:
                source_name = expr if isinstance(expr, str) else None
                break
        symbol = sympy_index_symbol(source_name or str(index_var))
        index_expr = (
            index_var.index_expr if isinstance(index_var, CuteDSLCSEVariable) else None
        )
        self._index_symbol_info[symbol] = IndexSymbolInfo(index_expr)
        return symbol

    # pyrefly: ignore [bad-override]
    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> str:
        # [FUNCTION DOC] Always raises NotImplementedError: CuteDSL modifications are read-only.
        raise NotImplementedError(
            "Store operations not supported - CuteDSL limited to read-only operations"
        )

    def _add_kernel_input(self, name: str):
        # [FUNCTION DOC] HELPER: Registers the buffer name as a kernel input, records its remapped name once in tensor_buffers, and returns the remapped name.
        """Add name as input to kernel and return input ref."""
        # Get the remapped name that will be used in the kernel
        remapped_name = self.kernel.args.input(name)
        # Track the remapped name for later collection
        if remapped_name not in self.tensor_buffers:
            self.tensor_buffers.append(remapped_name)
        return remapped_name

    def _process_indexing(self, index):
        # [FUNCTION DOC] HELPER: Renames an index expression for the kernel and renders it as source. KNOWN ISSUE: appears unused in this module.
        """Process and rename indexing, adding symbols as kernel inputs."""
        renamed = self.kernel.rename_indexing(index)
        return self.kernel.kexpr(renamed)

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # [FUNCTION DOC] HELPER: Fallback for ops: calls the same-named method on the inner handler; on NotImplementedError raises a new NotImplementedError with a formatted message asking the user to file an issue, including op name, args and kwargs (original chained).
        try:
            return getattr(self._inner, name)(*args, **kwargs)
        except NotImplementedError as e:
            bar = "=" * 80
            msg = textwrap.dedent(f"""
                {bar}
                UNSUPPORTED CUTEDSL OPERATION: '{name}'
                {bar}
                This operation is not yet implemented in Inductor.

                Please open an issue at: https://github.com/pytorch/pytorch/issues
                with the following information:

                Operation: {name}
                Args: {args!r}
                Kwargs: {kwargs!r}

                Title your issue: [CuteDSL] Missing operation: {name}
                {bar}
            """).strip()
            raise NotImplementedError(msg) from e
