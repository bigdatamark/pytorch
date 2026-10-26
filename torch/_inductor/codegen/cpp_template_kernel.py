# mypy: allow-untyped-defs
# [MODULE DOC] Defines CppTemplateKernel, the kernel context used when rendering inductor C++ templates
# (argument registration, buffer/index helpers, epilogue-fused output stores), and CppTemplateCaller,
# the autotuning choice wrapping a rendered template. Also small sympy/IR helper functions. No
# import-time side effects.
import itertools
from collections.abc import Callable, Iterable
from typing import Any
from unittest.mock import patch

import sympy
from sympy.parsing.sympy_parser import parse_expr

import torch
from torch._inductor.utils import do_bench_using_profiling
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.symbol import SymT

from .. import config, cpp_builder, ir, lowering as L
from ..autotune_process import CppBenchmarkRequest
from ..loop_body import LoopBody
from ..select_algorithm import PartialRender
from ..utils import sympy_index_symbol, sympy_index_symbol_with_prefix
from ..virtualized import V
from .common import REMOVED
from .cpp import CppKernel, CppKernelProxy, KernelGroup, ParallelDepth
from .cpp_utils import cexpr_index, DTYPE_TO_CPP, LocalBufferContext


def parse_expr_with_index_symbols(expr):
    # [FUNCTION DOC] HELPER: Converts an expression (or list/tuple of them, recursively) into sympy
    # expressions whose free symbols are replaced by integer index symbols of the same names. Existing
    # sympy expressions are returned unchanged; other values are parsed from their string form.
    if isinstance(expr, sympy.Expr):
        return expr
    elif isinstance(expr, (list, tuple)):
        return [parse_expr_with_index_symbols(e) for e in expr]
    else:
        expr = parse_expr(str(expr))
        int_symbols = {sym: sympy_index_symbol(sym.name) for sym in expr.free_symbols}
        return expr.subs(int_symbols)


def wrap_with_tensorbox(node) -> ir.TensorBox:
    # [FUNCTION DOC] HELPER: Wraps an IR node in a TensorBox, using the create path for buffers and the direct constructor otherwise.
    return (
        ir.TensorBox.create(node) if isinstance(node, ir.Buffer) else ir.TensorBox(node)
    )


class CppTemplateKernel(CppKernel):
    # [CLASS DOC] CppKernel specialization used while rendering C++ templates. State: kernel_name,
    # render_hooks (placeholder to deferred callable, resolved at the end of render) and local_buffers
    # (name to IR buffer for kernel-local scratch buffers). Offers helpers templates call: def_kernel,
    # size/stride/index/dtype queries, slice/select/view/permute, local buffer codegen and
    # store_output(s) for epilogue fusion.
    def __init__(self, kernel_name, num_threads):
        # [FUNCTION DOC] Initializes the CppKernel base with the thread count and stores the kernel
        # name, an empty render-hook mapping and an empty local-buffer mapping.
        super().__init__(None, num_threads)
        self.kernel_name = kernel_name
        self.render_hooks = {}
        self.local_buffers = {}

    def render(self, template, **kwargs):
        # [FUNCTION DOC] Renders the template with this kernel (plus kwargs) and resolves all
        # registered render hooks, returning the final source string.
        return PartialRender(
            template.render(kernel=self, **kwargs), self.render_hooks
        ).finalize_all()

    def def_kernel(
        self,
        inputs: dict[str, ir.Buffer],
        outputs: dict[str, ir.Buffer],
        aliases: dict[str, str] | None = None,
        function_name: str = "",
        extra_sizevars: list[sympy.Expr] | None = None,
        placeholder: str = "<DEF_KERNEL>",
    ) -> str:
        # [FUNCTION DOC] Registers template inputs and outputs (name to buffer mappings) as kernel
        # arguments, adds optional aliases of existing buffers, and collects the sorted free size
        # symbols of inputs, outputs and extra_sizevars as integer size arguments. Registers a render
        # hook under placeholder that later yields the C++ function signature (function_name defaults
        # to the kernel name). Returns the placeholder text. Raises AssertionError if the placeholder
        # is already registered.
        if len(function_name) == 0:
            function_name = str(self.kernel_name)
        for name, inp in inputs.items():
            if inp is not None:
                self.args.input_buffers[inp.get_name()] = name
        for name, out in outputs.items():
            self.args.output_buffers[out.get_name()] = name
        if aliases is not None:
            for alias, orig in aliases.items():
                if orig in self.args.input_buffers:
                    self.args.input_buffers[alias] = self.args.input_buffers[orig]
                if orig in self.args.output_buffers:
                    self.args.output_buffers[alias] = self.args.output_buffers[orig]

        unique_sizevars = OrderedSet(
            s
            for input in inputs.values()
            if input is not None
            for sym in itertools.chain(input.get_size(), input.get_stride())
            if isinstance(sym, sympy.Expr)
            for s in sym.free_symbols
        )
        unique_sizevars.update(
            s
            for sym in extra_sizevars or []
            if isinstance(sym, sympy.Expr)
            for s in sym.free_symbols
        )
        unique_sizevars.update(
            s
            for output in outputs.values()
            for sym in itertools.chain(output.get_size(), output.get_stride())
            if isinstance(sym, sympy.Expr)
            for s in sym.free_symbols
        )
        sizevars = sorted(unique_sizevars, key=str)
        for sizevar in sizevars:
            self.args.sizevars[sizevar] = f"k{sizevar}"

        def hook():
            # [FUNCTION DOC] HELPER: Deferred render hook. Marks every alias output buffer as removed
            # (raises AssertionError if an alias is also an input buffer), then returns the C++
            # function signature string with the final argument definitions.
            # remove all aliases before generate function definition
            if aliases is not None:
                for alias in aliases:
                    if alias in self.args.input_buffers:
                        raise AssertionError(
                            f"input_buffers cannot be removed: {alias}"
                        )
                    if alias in self.args.output_buffers:
                        self.args.output_buffers[alias] = REMOVED
            cpp_argdefs, _, _ = self.args.cpp_argdefs()
            return f"void {function_name}({', '.join(cpp_argdefs)})"

        if placeholder in self.render_hooks:
            raise AssertionError(f"placeholder already registered: {placeholder}")
        self.render_hooks[placeholder] = hook
        return placeholder

    def call_kernel(self, name: str, node: ir.CppTemplateBuffer):
        # [FUNCTION DOC] Emits a non-Triton kernel call named name into the graph's wrapper code,
        # using this kernel's C++ call arguments and argument types. Side effect: writes wrapper code.
        wrapper = V.graph.wrapper_code
        _, call_args, arg_types = self.args.cpp_argdefs()
        wrapper.generate_kernel_call(name, call_args, triton=False, arg_types=arg_types)

    def dtype(self, node: ir.Buffer) -> str:
        # [FUNCTION DOC] Returns the C++ type name for the buffer's dtype.
        return DTYPE_TO_CPP[node.get_dtype()]

    def acc_dtype(self, node: ir.Buffer) -> str:
        # [FUNCTION DOC] Returns the C++ accumulation type name: "float" for float32, bfloat16 and
        # half buffers. Raises NotImplementedError for any other dtype.
        if node.get_dtype() in [torch.float32, torch.bfloat16, torch.half]:
            return "float"
        else:
            raise NotImplementedError(f"Unsupported dtype: {node.get_dtype()}")

    def size(self, node: ir.Buffer, dim: int) -> str:
        # [FUNCTION DOC] Returns the C++ expression string for the size of node along dim, with symbols renamed.
        return cexpr_index(self.rename_indexing(node.get_size()[dim]))

    def stride(self, node: ir.Buffer, dim: int) -> str:
        # [FUNCTION DOC] Returns the C++ expression string for the stride of node along dim, with symbols renamed.
        return cexpr_index(self.rename_indexing(node.get_stride()[dim]))

    def index(self, node: ir.Buffer, indices: list[Any]) -> str:
        # [FUNCTION DOC] Returns a C++ element-access expression for node at the given multi-dim indices,
        # using the node's fixed layout indexer. Uses the buffer's own name if it is a local buffer,
        # else registers it as a kernel input (side effect on args) and uses its argument name.
        indexer = node.get_layout().as_fixed().make_indexer()
        index = indexer(parse_expr_with_index_symbols(indices))
        index = self.rename_indexing(index)
        outer_name = node.get_name()
        inner_name = (
            outer_name
            if outer_name in self.local_buffers
            else self.args.input(node.get_name())
        )
        return f"{inner_name}[{cexpr_index(index)}]"

    def slice_nd(self, node, ranges: list[tuple[Any, Any]]) -> ir.ReinterpretView:
        # [FUNCTION DOC] Slices node along each dim by the matching (start, end) range, skipping dims
        # with an empty range, without clamping. Returns the resulting ReinterpretView. Raises
        # AssertionError if the number of ranges differs from the rank, a range has length other than
        # 0 or 2, or the result is not a ReinterpretView.
        """
        Slice the given node with a list of ranges (start and end) corresponding to its dims.
        The dim is not sliced if the corresponding range is empty.
        """
        if len(ranges) != len(node.get_size()):
            raise AssertionError(f"{ranges=}, {node=}")
        sliced = wrap_with_tensorbox(node)
        for dim, _range in enumerate(ranges):
            if len(_range) == 0:
                continue
            if len(_range) != 2:
                raise AssertionError(f"expected range of length 2, got {len(_range)}")
            start, end = parse_expr_with_index_symbols(_range)
            sliced = L.slice_(sliced, dim, start, end, clamp=False)
        if not isinstance(sliced, ir.TensorBox):
            raise AssertionError(f"expected ir.TensorBox, got {type(sliced)}")
        if not isinstance(sliced.data, ir.ReinterpretView):
            raise AssertionError(sliced.data)
        return sliced.data

    def select(self, node, dim: int, idx: int) -> ir.ReinterpretView:
        # [FUNCTION DOC] Selects index idx (negative allowed) along dim of node by slicing without
        # clamping then squeezing that dim; returns the resulting ReinterpretView. Raises AssertionError
        # if the result is not a ReinterpretView.
        # We avoid using L.select here because we need clamp=False so the dim after slicing
        # is 1 instead of a sympy expression of symbol - dim_size.
        node = wrap_with_tensorbox(node)
        idx = ir.View.handle_negative_index(idx, node.get_size()[dim])
        sliced = L.squeeze(L.slice_(node, dim, idx, idx + 1, clamp=False), dim)
        if not isinstance(sliced.data, ir.ReinterpretView):
            raise AssertionError(sliced.data)
        return sliced.data

    def view(self, node, sizes: list[Any]) -> ir.IRNode:
        # [FUNCTION DOC] Returns the IR node for a view of node reshaped to sizes (parsed into index symbols).
        node = wrap_with_tensorbox(node)
        sizes = parse_expr_with_index_symbols(sizes)
        return L.view(node, sizes).data

    def permute(self, node, dims):
        # [FUNCTION DOC] Returns a ReinterpretView of node with dimensions permuted by dims. Raises
        # AssertionError if the result is not a ReinterpretView.
        node = wrap_with_tensorbox(node)
        permuted = L.permute(node, dims).data
        if not isinstance(permuted, ir.ReinterpretView):
            raise AssertionError(f"expected ir.ReinterpretView, got {type(permuted)}")
        return permuted

    def maybe_codegen_profile(self, prefix_kernel_name: str | None = None) -> str:
        # [FUNCTION DOC] When kernel profiling is enabled in config, returns C++ code declaring an
        # AOTI record-function handle named from the graph id, optional prefix and kernel name;
        # otherwise returns an empty string.
        if config.cpp.enable_kernel_profile:
            graph_id = V.graph.graph_id
            prefix = "graph_" + str(graph_id) + "_" if graph_id is not None else ""
            if prefix and prefix_kernel_name:
                prefix += prefix_kernel_name + "_"
            handle_str = (
                "torch::aot_inductor::RAIIAtenRecordFunctionHandle "
                f'record_{prefix}{self.kernel_name}_("{prefix}{self.kernel_name}", nullptr);'
            )
            return handle_str
        else:
            return ""

    def unroll_pragma(self, unroll):
        # [FUNCTION DOC] Returns the compiler-specific unroll pragma line (GCC form or generic form)
        # for the given unroll factor.
        if cpp_builder.is_gcc():
            return f"#pragma GCC unroll {unroll}"
        else:
            return f"#pragma unroll {unroll}"

    def define_buffer(self, name, sizes: list[Any], dtype=torch.float) -> str:
        # [FUNCTION DOC] Registers a local CPU buffer (name, sizes parsed into index symbols, dtype) in
        # local_buffers and returns C++ code allocating it on the heap via a unique pointer plus a raw
        # pointer variable of the given name.
        """Define kernel local buffer"""
        sizes = parse_expr_with_index_symbols(sizes)
        buf = ir.Buffer(
            name=name, layout=ir.FixedLayout(torch.device("cpu"), dtype, sizes)
        )
        self.local_buffers[name] = buf
        ctype = f"{DTYPE_TO_CPP[dtype]}"
        numel = f"{cexpr_index(buf.get_numel())}"
        return f"auto _{name} = std::make_unique<{ctype}[]>({numel}); auto {name} = _{name}.get();"

    def define_stack_allocated_buffer(
        self, name, sizes: list[Any], dtype=torch.float
    ) -> str:
        # [FUNCTION DOC] Registers a local CPU buffer with the given name, sizes and dtype in
        # local_buffers and returns C++ code declaring a 64-byte aligned stack array plus a pointer to it.
        """Define stack-allocated buffer"""
        sizes = parse_expr_with_index_symbols(sizes)
        buf = ir.Buffer(
            name=name, layout=ir.FixedLayout(torch.device("cpu"), dtype, sizes)
        )
        self.local_buffers[name] = buf
        ctype = f"{DTYPE_TO_CPP[dtype]}"
        numel = f"{cexpr_index(buf.get_numel())}"
        return f"alignas(64) {ctype} _{name}[{numel}]; {ctype}* {name} = _{name};"

    def reinit_buffer_if_null(self, name):
        # [FUNCTION DOC] Returns C++ code that reallocates the named local buffer (previously made by
        # define_buffer) when its owning pointer is null. Raises AssertionError if the name is unknown.
        """Reinit the previously defined local buffer if it is null"""
        if name not in self.local_buffers:
            raise AssertionError(f"unknown local buffer: {name}")
        buf = self.local_buffers[name]
        ctype = f"{DTYPE_TO_CPP[buf.layout.dtype]}"
        numel = f"{cexpr_index(buf.get_numel())}"
        return f"if (_{name} == nullptr) {{ _{name} = std::make_unique<{ctype}[]>({numel}); {name} = _{name}.get(); }}"

    def release_buffer(self, name):
        # [FUNCTION DOC] Returns C++ code releasing ownership of the named local buffer's smart pointer.
        # Raises AssertionError if the name is not a known local buffer.
        """Codegen the code to release the ownership of a local buffer to others"""
        if name not in self.local_buffers:
            raise AssertionError(f"unknown local buffer: {name}")
        return f"_{name}.release()"

    def store_pointwise_nodes(
        self,
        dst: ir.Buffer,
        nodes: list[ir.IRNode],
        offsets: list[sympy.Expr] | None = None,
        reindexers: list[Callable[[list[Any]], list[Any]] | None] | None = None,
    ) -> str:
        # [FUNCTION DOC] Generates C++ loop code that evaluates a chain of pointwise nodes over the
        # shape of dst and stores the last result to dst (earlier nodes store to their own names).
        # Offsets (default zeros, one per dim) shift the indices given to every node; reindexers
        # (default none) optionally remap indices per node. Builds a loop body per node, emits them
        # through a CppKernelProxy sharing this kernel's args, with parallelization disabled.
        # Returns the generated loop code string. Raises AssertionError on non-pointwise nodes or a
        # wrong offsets length. Mutates this kernel's args via buffer registration.
        var_sizes = (tuple(dst.get_size()), ())
        var_ranges = {
            sympy_index_symbol_with_prefix(SymT.INDEX, i): sz
            for i, sz in enumerate(var_sizes[0])
        }
        if not offsets:
            offsets = [sympy.S.Zero] * len(var_sizes[0])
        if not reindexers:
            reindexers = [None] * len(nodes)
        if len(offsets) != len(var_sizes[0]):
            raise AssertionError(
                f"expected {len(var_sizes[0])} offsets, got {len(offsets)}"
            )
        output_index = dst.get_layout().make_indexer()([*var_ranges.keys()])
        kernel_group = KernelGroup()
        kernel_group.args = self.args
        cpp_kernel_proxy = CppKernelProxy(kernel_group)
        bodies = []
        var_sizes_list = []
        for i, node in enumerate(nodes):
            output_name = node.get_name() if i < len(nodes) - 1 else dst.get_name()
            node = node.data if isinstance(node, ir.ComputedBuffer) else node
            if not isinstance(node, ir.Pointwise):
                raise AssertionError(node)

            def fn(*args):
                # [FUNCTION DOC] HELPER: LoopBody body function. Takes the index list and an empty
                # reduction list, adds the per-dim offsets, optionally applies the node's reindexer,
                # then stores the pointwise node's loaded value to the output at the destination index.
                # Raises AssertionError if the argument shapes are unexpected.
                if len(args) != 2:
                    raise AssertionError(f"expected 2 args, got {len(args)}")
                if len(args[0]) != len(var_sizes[0]):
                    raise AssertionError(
                        f"expected {len(var_sizes[0])} indices, got {len(args[0])}"
                    )
                if len(args[1]) != 0:
                    raise AssertionError(
                        f"expected no reduction vars, got {len(args[1])}"
                    )
                new_args = [arg + offset for arg, offset in zip(args[0], offsets)]  # type: ignore[arg-type]
                if reindexers[i] is not None:
                    new_args = reindexers[i](new_args)  # type: ignore[misc]
                V.ops.store(
                    output_name,
                    output_index,
                    node.make_loader()(new_args).value,
                )

            body = LoopBody(
                fn,
                (list(var_ranges.keys()), ()),
                var_ranges,
                list(var_ranges.keys()),
                tuple(),
            )
            bodies.append(body)
            var_sizes_list.append(var_sizes)

        cpp_kernel_proxy.codegen_loop_bodies(bodies, var_sizes_list)

        def max_parallel_depth():
            # [FUNCTION DOC] HELPER: Replacement for the loop nest's max_parallel_depth that reports
            # zero parallel depth so the generated epilogue loops are never parallelized.
            return ParallelDepth(parallel_depth=0, start_depth=0)

        # This loop is not parallelized since it is not the outermost loop.
        with patch.object(
            cpp_kernel_proxy.loop_nest, "max_parallel_depth", max_parallel_depth
        ):
            kernel_group.finalize_kernel(cpp_kernel_proxy, [])
        return kernel_group.loops_code.getvalue()

    def store_grouped_gemm_pointwise_nodes(
        self,
        dst: tuple[ir.Buffer],
        nodes: list[ir.IRNode],
        offsets: list[sympy.Expr],
        reindexers: list[Callable[[list[Any]], list[Any]] | None],
        output_names: list[str],
    ) -> str:
        # [FUNCTION DOC] Grouped-GEMM variant of store_pointwise_nodes. The iteration space is the
        # shape of the first dst; each node gets its own offset list, optional reindexer and explicit
        # output buffer name (parallel lists indexed by node). Emits loop code through a CppKernelProxy
        # sharing this kernel's args with parallelization disabled and returns the code string.
        # Raises AssertionError if offsets are empty, any offset length mismatches the rank, or a
        # node is not pointwise.
        ref_dst = dst[0]
        var_sizes = (tuple(ref_dst.get_size()), ())
        var_ranges = {
            sympy_index_symbol_with_prefix(SymT.INDEX, i): sz
            for i, sz in enumerate(var_sizes[0])
        }
        if not offsets:
            raise AssertionError("offsets should be set outside")
        if not all(len(offset) == len(var_sizes[0]) for offset in offsets):
            raise AssertionError(f"expected all offsets of length {len(var_sizes[0])}")
        output_index = ref_dst.get_layout().make_indexer()([*var_ranges.keys()])
        kernel_group = KernelGroup()
        kernel_group.args = self.args
        cpp_kernel_proxy = CppKernelProxy(kernel_group)
        bodies = []
        var_sizes_list = []
        for i, node in enumerate(nodes):
            output_name = output_names[i]
            node = node.data if isinstance(node, ir.ComputedBuffer) else node
            if not isinstance(node, ir.Pointwise):
                raise AssertionError(node)

            def fn(*args):
                # [FUNCTION DOC] HELPER: LoopBody body function. Takes the index list and an empty
                # reduction list, adds this node's per-dim offsets, optionally applies the node's
                # reindexer, then stores the pointwise node's loaded value to its output name at the
                # reference destination index. Raises AssertionError if argument shapes are unexpected.
                if len(args) != 2:
                    raise AssertionError(f"expected 2 args, got {len(args)}")
                if len(args[0]) != len(var_sizes[0]):
                    raise AssertionError(
                        f"expected {len(var_sizes[0])} indices, got {len(args[0])}"
                    )
                if len(args[1]) != 0:
                    raise AssertionError(
                        f"expected no reduction vars, got {len(args[1])}"
                    )
                new_args = [arg + offset for arg, offset in zip(args[0], offsets[i])]  # type: ignore[arg-type]
                if reindexers[i] is not None:
                    new_args = reindexers[i](new_args)  # type: ignore[misc]
                V.ops.store(
                    output_name,
                    output_index,
                    node.make_loader()(new_args).value,
                )

            body = LoopBody(
                fn,
                (list(var_ranges.keys()), ()),
                var_ranges,
                list(var_ranges.keys()),
                tuple(),
            )
            bodies.append(body)
            var_sizes_list.append(var_sizes)

        cpp_kernel_proxy.codegen_loop_bodies(bodies, var_sizes_list)

        def max_parallel_depth():
            # [FUNCTION DOC] HELPER: Replacement for the loop nest's max_parallel_depth that reports
            # zero parallel depth so the generated epilogue loops are never parallelized.
            return ParallelDepth(parallel_depth=0, start_depth=0)

        # This loop is not parallelized since it is not the outermost loop.
        with patch.object(
            cpp_kernel_proxy.loop_nest, "max_parallel_depth", max_parallel_depth
        ):
            kernel_group.finalize_kernel(cpp_kernel_proxy, [])
        return kernel_group.loops_code.getvalue()

    def store_output(
        self,
        dst: ir.Buffer,
        src: ir.Buffer,
        orig_src: ir.Buffer | None = None,
        epilogue_nodes: list[ir.IRNode] | None = None,
        offsets: list[Any] | None = None,
        reindexers: list[Callable[[list[Any]], list[Any]] | None] | None = None,
    ):
        """
        Store the `src` buffer to the `dst` buffer. The size of `src` and `dst` should match.
        If `epilogue_nodes` is provided, the `src` buffer is firstly computed with the epilogues
        before stored to `dst`. The `epilogues_nodes` are all pointwise.

        Notes:
        1. `src` and `dst` buffer could be the same buffer in which case we are doing in-place compute
           and stores. In case `epilogue_nodes` are not provided, we do nothing.
        2. The `epilogue_nodes`, if exist, have computations on `src` before storing to `dst` but since
           they come from the original Inductor IR, they might need to be adjusted before working with
           `src` and `dst` as outlined below:
           a) `src` or `dst` buffer could be a sub-slice of the ranges the `epilogue_nodes`work on.
              In this case, the `offsets` could be provided to adjust the indices passed to
              `epilogue_nodes` during codegen and the data ranges are also configured according to
              the sizes of `src` and `dst`.
           b) `dst` might be indexed in a different way as the `epilogue_nodes`, hence a `reindexer` is
              needed on the indices to `epilogue_nodes` to match the indexing of `dst`.
           c) If `src` is local, we need to add a local buffer for it and localize the `orig_src` buffer
              in `epilogue_nodes` with `src`.
        """
        # [FUNCTION DOC] Generates C++ code storing src into dst (same size), applying pointwise
        # epilogue nodes first if given (localizing orig_src to src when names differ, with optional
        # index offsets and reindexers). Without epilogues: if names differ, src is treated as a local
        # buffer and copied to dst; if identical, layouts must match and nothing is emitted.
        # Returns the code string ("" for in-place). Raises AssertionError on bad dst type, size
        # mismatch, missing orig_src or layout mismatch.
        if not isinstance(dst, (ir.Buffer, ir.ReinterpretView)):
            raise AssertionError(
                f"expected ir.Buffer or ir.ReinterpretView, got {type(dst)}"
            )
        if dst.get_size() != src.get_size():
            raise AssertionError(f"{dst=}, {src=}")
        if offsets:
            offsets = parse_expr_with_index_symbols(offsets)
        if epilogue_nodes:
            with LocalBufferContext(self.args) as scope:
                if orig_src is None:
                    raise AssertionError("orig_src must not be None")
                if orig_src.get_name() != src.get_name():
                    scope.add_local_buffer(
                        src,
                        [
                            orig_src,
                        ],
                    )
                    epilogue_nodes = scope.localize_nodes(epilogue_nodes)
                return self.store_pointwise_nodes(
                    dst,
                    epilogue_nodes,  # type: ignore[arg-type]
                    offsets,
                    reindexers,
                )
        else:
            if dst.get_name() != src.get_name():
                # src is local
                copy = L.copy(dst, src).data.data
                with LocalBufferContext(self.args) as scope:
                    scope.add_local_buffer(src)

                    return self.store_pointwise_nodes(dst, [copy])
            else:
                if dst.layout != src.layout:
                    raise AssertionError(f"{dst=}, {src=}")
                return ""

    def store_outputs(
        self,
        dst: tuple[ir.Buffer],
        src: tuple[ir.IRNode],
        orig_src: tuple[ir.IRNode] | None = None,
        epilogue_nodes: list[ir.IRNode] | None = None,
        offsets: list[Any] | None = None,
        reindexers: list[Callable[[list[Any]], list[Any]] | None] | None = None,
        multi_output_buffers: tuple[ir.MultiOutput, ...] | None = None,
    ):
        # [FUNCTION DOC] Grouped-GEMM counterpart of store_output: stores each src buffer to the matching
        # dst, optionally applying pointwise epilogue nodes. With epilogues: localizes them, adds local
        # buffers for GEMM outputs that differ from the originals (also localizing MultiOutput buffers
        # read by epilogues), appends copy nodes for outputs no epilogue reads, then generates code via
        # store_grouped_gemm_pointwise_nodes and removes MultiOutput buffers read by epilogues from the
        # kernel args. Without epilogues: copies src to dst if names differ, else checks they match.
        # Returns the generated code string ("" when nothing to store). Raises AssertionError on
        # non-iterable dst, size/name/layout mismatches or missing orig_src.
        if not isinstance(dst, Iterable):
            raise AssertionError(f"expected Iterable, got {type(dst)}")
        if not all(_dst.get_size() == _src.get_size() for _src, _dst in zip(src, dst)):
            raise AssertionError("src and dst sizes must match")
        if offsets:
            offsets = parse_expr_with_index_symbols(offsets)
        gemm_num = len(src)
        final_offsets = []
        output_names = []
        if epilogue_nodes:
            if not reindexers:
                reindexers = [None] * len(epilogue_nodes)
            with LocalBufferContext(self.args) as scope:
                if orig_src is None:
                    raise AssertionError("orig_src must not be None")
                localize_epilogue_nodes = []
                all_read_names = []
                for epilogue in epilogue_nodes:
                    all_read_names.extend(list(epilogue.get_read_names()))
                localize_epilogue_nodes.extend(scope.localize_nodes(epilogue_nodes))
                final_offsets.extend([offsets] * len(localize_epilogue_nodes))
                output_names.extend(
                    [node.get_name() for node in localize_epilogue_nodes]
                )
                for gemm_idx in range(gemm_num):
                    if orig_src[gemm_idx].get_name() != src[gemm_idx].get_name():
                        if orig_src[gemm_idx].get_name() in all_read_names or (
                            multi_output_buffers
                            and multi_output_buffers[gemm_idx].get_name()
                            in all_read_names
                        ):
                            # If any of the Epilogue nodes use this GEMM output, let's localize the GEMM output
                            global_buffers = [orig_src[gemm_idx]]
                            if (
                                multi_output_buffers
                                and multi_output_buffers[gemm_idx].get_name()
                                in all_read_names
                                and orig_src[gemm_idx].get_name() not in all_read_names
                            ):
                                # Epilogue might directly read the MultiOutput, Localize MultiOutput to the local Buffer
                                # if this MultiOutput has not been stored by in-template epilogue
                                # otherwise, use the cse store cache if it will be stored before used
                                global_buffers.append(multi_output_buffers[gemm_idx])
                            scope.add_local_buffer(
                                src[gemm_idx],
                                global_buffers,
                            )
                        else:
                            scope.add_local_buffer(src[gemm_idx])
                            localize_epilogue_nodes.extend(
                                [L.copy(dst[gemm_idx], src[gemm_idx]).data.data]
                            )
                            reindexers.append(None)
                            output_names.append(dst[gemm_idx].get_name())
                            final_offsets.append(
                                [sympy.S.Zero] * len(dst[gemm_idx].get_size())
                            )
                res = self.store_grouped_gemm_pointwise_nodes(
                    dst,
                    localize_epilogue_nodes,
                    final_offsets,
                    reindexers,
                    output_names=output_names,
                )
                for gemm_idx in range(gemm_num):
                    if (
                        multi_output_buffers
                        and multi_output_buffers[gemm_idx].get_name() in all_read_names
                    ):
                        # If the MultiOutput is used in the Epilogue, let's remove it from args
                        multi_output_name = multi_output_buffers[gemm_idx].get_name()
                        if (
                            multi_output_name in self.args.output_buffers
                            and self.args.output_buffers[multi_output_name]
                            is not REMOVED
                        ):
                            self.remove_buffer(multi_output_name)
                return res
        else:
            if dst[0].get_name() != src[0].get_name():
                copy_list = []
                with LocalBufferContext(self.args) as scope:
                    for _src, _dst in zip(src, dst):
                        copy_list.extend([L.copy(_dst, _src).data.data])
                        scope.add_local_buffer(_src)
                        output_names.append(_dst.get_name())
                        final_offsets.append([sympy.S.Zero] * len(_dst.get_size()))
                    reindexers = [None] * len(copy_list)
                    return self.store_grouped_gemm_pointwise_nodes(
                        dst,
                        nodes=copy_list,
                        offsets=final_offsets,
                        reindexers=reindexers,
                        output_names=output_names,
                    )
            else:
                if not all(
                    _src.get_name() == _dst.get_name() for _src, _dst in zip(src, dst)
                ):
                    raise AssertionError("src and dst names must match")
                if not all(
                    _src.get_layout() == _dst.get_layout()
                    for _src, _dst in zip(src, dst)
                ):
                    raise AssertionError("src and dst layouts must match")
                return ""

    def check_bounds(self, expr, size, lower, upper):
        # [FUNCTION DOC] No-op override; bounds checks are not generated for C++ templates.
        # CppTemplateKernel does not need codegen related operations
        return


class CppTemplateCaller(ir.ChoiceCaller):
    # [CLASS DOC] Autotuning choice wrapping a rendered C++ template. Holds category, a factory that
    # builds (kernel, render) for a template buffer, a CppBenchmarkRequest, the owning template and
    # optional info kwargs. Provides precompile, benchmark, hash_key, info_dict and output_node, which
    # materializes a CppTemplateBuffer as the chosen IR node.
    """
    CppTemplateCaller

    This class represents a caller for CPP template kernels. It is a subclass of ir.ChoiceCaller.
    Attributes:
        name (str): The name of the caller.
        category (str): The category of the caller.
        bmreq (CppBenchmarkRequest): The benchmark request for the caller.
        template_buffer (ir.CppTemplateBuffer): The template buffer for the caller.
    """

    def __init__(
        self,
        name: str,
        category: str,
        input_nodes: list[ir.Buffer],
        layout: ir.Layout,
        make_kernel_render: Callable[
            [
                ir.CppTemplateBuffer,
                bool,
                list[ir.IRNode] | None,
            ],
            str,
        ],
        bmreq: CppBenchmarkRequest,
        template: "CppTemplate",  # type: ignore[name-defined]  # noqa: F821
        info_kwargs: dict[str, ir.PrimitiveInfoType | list[ir.PrimitiveInfoType]]
        | None = None,
    ):
        # [FUNCTION DOC] Initializes the ChoiceCaller base and stores category, kernel-render factory,
        # benchmark request, template and optional info kwargs.
        super().__init__(name, input_nodes, layout, description="")
        self.category = category
        self.make_kernel_render = make_kernel_render
        self.bmreq = bmreq
        self.template = template
        self.info_kwargs = info_kwargs

    def precompile(self) -> None:
        # [FUNCTION DOC] Compiles the benchmark request's kernel ahead of time. Raises AssertionError
        # if there is no benchmark request.
        if self.bmreq is None:
            raise AssertionError("bmreq must not be None")
        self.bmreq.precompile()

    def benchmark(self, *args, out) -> float:
        # [FUNCTION DOC] Benchmarks the compiled kernel on the given inputs writing into out and
        # returns the measured time as a float. Uses profiler-based timing when the config flag
        # profile_bandwidth_with_do_bench_using_profiling is set, else the request's own benchmark.
        # Raises AssertionError if there is no benchmark request.
        if self.bmreq is None:
            raise AssertionError("bmreq must not be None")
        if config.profile_bandwidth_with_do_bench_using_profiling:
            algo = self.bmreq.make_run_fn(*args, out=out)
            return do_bench_using_profiling(algo)
        return self.bmreq.benchmark(*args, out=out)

    def hash_key(self) -> str:
        # [FUNCTION DOC] Returns a cache key string: category joined with the benchmark request's hash key.
        return "-".join(
            [
                self.category,
                self.bmreq.hash_key,
            ]
        )

    def info_dict(
        self,
    ) -> dict[str, ir.PrimitiveInfoType | list[ir.PrimitiveInfoType]]:
        # [FUNCTION DOC] Returns a fixed info mapping with backend "CPP" and op_type "unknown".
        return {"backend": "CPP", "op_type": "unknown"}

    def output_node(self) -> ir.TensorBox:
        # [FUNCTION DOC] Creates a CppTemplateBuffer from this caller's layout, inputs, kernel-render
        # factory, template and itself, copies the "ktc" annotation onto it if present, and returns it
        # wrapped in a TensorBox.
        buffer = ir.CppTemplateBuffer(
            layout=self.layout,
            inputs=self.input_nodes,
            make_kernel_render=self.make_kernel_render,
            template=self.template,
            choice=self,
        )
        # Pass KTC annotation to the buffer for encoding
        if "ktc" in self.annotations:
            buffer.annotations["ktc"] = self.annotations["ktc"]
        return ir.TensorBox.create(buffer)
