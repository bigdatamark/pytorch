# mypy: allow-untyped-defs
# [MODULE DOC] Scheduler backend for CUTLASS C++ template kernels. Defines
# CUTLASSScheduling (fusion decisions for pointwise epilogues, kernel source
# registration with the wrapper code, and call-site codegen) and WhyNoFuseNames
# (fusion-rejection logger carrying two node names). Intended to be delegated to by the
# combined CUDA/XPU scheduling classes. Import-time side effects: creates a module logger.
import hashlib
import logging
from collections.abc import Sequence
from typing import cast, TypeGuard

import sympy

from torch._inductor.codegen.cutlass.python_evt import (
    CutlassEVTCodegen,
    MockCutlassHandler,
)
from torch._inductor.utils import Placeholder
from torch.utils._ordered_set import OrderedSet

from ...._dynamo.utils import counters
from ... import config
from ...codecache import code_hash, get_path
from ...ir import Buffer, ComputedBuffer, CUTLASSTemplateBuffer, Pointwise
from ...scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    SchedulerNode,
    WhyNoFuse,
)
from ...utils import get_fused_kernel_name, get_kernel_metadata, sympy_product
from ...virtualized import V
from ..common import BackendFeature, IndentedBuffer


log = logging.getLogger(__name__)


class WhyNoFuseNames(WhyNoFuse):
    # [CLASS DOC] HELPER: Fusion-failure reporter that records the names of the two nodes
    # considered for fusion (name1, name2); calling it logs why fusion was rejected.
    def __init__(self, name1: str, name2: str) -> None:
        # [FUNCTION DOC] HELPER: Stores the two node names.
        self.name1 = name1
        self.name2 = name2


class CUTLASSScheduling(BaseScheduling):
    # [CLASS DOC] Partial scheduling backend for CUTLASS kernels. Stateless itself; uses
    # the global graph (V.graph) for wrapper code and size variables. Decides whether
    # pointwise nodes can be fused as epilogues into a CUTLASS template node
    # (can_fuse_vertical), registers generated kernel source with the wrapper
    # (define_kernel) and emits the template kernel and its call (codegen_template).
    """
    Partial Scheduling implementation for cutlass C++ Kernels.
    This class is intended to be used in combination with TritonScheduling,
    and delegated to by CUDACombinedScheduling/XPUCombinedScheduling.

    It handles fusion decisions and cutlass C++ specific template code generation.
    """

    @classmethod
    def get_backend_features(cls, device) -> OrderedSet[BackendFeature]:
        # [FUNCTION DOC] Returns an empty set of backend features for any device.
        return OrderedSet()

    def group_fn(self, sizes):
        # [FUNCTION DOC] Maps each size list to the simplified product of its dimensions
        # and returns the results as a tuple.
        return tuple(V.graph.sizevars.simplify(sympy_product(s)) for s in sizes)

    @staticmethod
    def is_cutlass_template(node: BaseSchedulerNode) -> TypeGuard[SchedulerNode]:
        # [FUNCTION DOC] True if the node is a plain scheduler node wrapping a CUTLASS
        # template buffer.
        return isinstance(node, SchedulerNode) and isinstance(
            node.node, CUTLASSTemplateBuffer
        )

    def is_cutlass_fused_template(self, node: BaseSchedulerNode) -> bool:
        # [FUNCTION DOC] True if the node is a fused scheduler node that also satisfies
        # the CUTLASS template check.
        # KNOWN ISSUE: the template check requires a plain scheduler node, which a fused
        # node normally is not, so this likely returns False for typical fused nodes
        # unless the fused class derives from the scheduler node class.
        return isinstance(node, FusedSchedulerNode) and self.is_cutlass_template(node)

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # [FUNCTION DOC] Decides whether node2 can be fused vertically after node1 as an
        # epilogue. If node1 is a CUTLASS template node, checks node2 against it with
        # no existing epilogues; if node1 is a fused template node, checks node2 against
        # its template node and already-fused epilogue nodes. Raises AssertionError if a
        # needed node has no IR node. Returns False for any other combination.
        if self.is_cutlass_template(node1) and isinstance(node2, BaseSchedulerNode):
            if not node1.node:
                raise AssertionError("node1.node should not be None")
            return self._can_fuse_epilogue_impl(
                cast(CUTLASSTemplateBuffer, node1.node),
                [],
                node2,  # type: ignore[arg-type]
            )
        elif self.is_cutlass_fused_template(node1) and isinstance(
            node2, BaseSchedulerNode
        ):
            if not node1.node:
                raise AssertionError("node1.node should not be None")
            if not node2.node:
                raise AssertionError("node2.node should not be None")
            fnode1 = cast(FusedSchedulerNode, node1)
            return self._can_fuse_epilogue_impl(
                fnode1.get_template_node(),  # type: ignore[arg-type]
                self._unwrap_epilogue_nodes(fnode1),
                node2,  # type: ignore[arg-type]
            )

        return False

    def define_kernel(self, src_code: str, node_schedule) -> str:
        # [FUNCTION DOC] Registers C++ kernel source with the graph wrapper code and
        # returns the kernel name. Reuses the existing name if the same source was seen.
        # Otherwise derives a name from a short hash of the source (with a descriptive
        # fused-name segment when enabled and not the plain fused name), substitutes the
        # kernel-name placeholder in the source, builds an async-compile wrapper snippet
        # with a metadata comment (kernel path and origins), and defines it on the
        # wrapper. For JIT cpp_wrapper mode (not AOT) also compiles a shared library now
        # and adds it to the wrapper's external kernel libs. Mutates wrapper state.
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            fused_name = (
                get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
                if config.triton.descriptive_names
                else ""
            )

            # use the original src_code as the key
            kernel_hash = hashlib.sha256(src_code.encode("utf-8")).hexdigest()[:8]
            if fused_name == "fused":
                # no EVT kernel, use the original kernel name
                kernel_name = f"cutlass_{kernel_hash}"
            else:
                kernel_name = f"cutlass_{fused_name}_{kernel_hash}"
            wrapper.src_to_kernel[src_code] = kernel_name
            src_code = src_code.replace(str(Placeholder.KERNEL_NAME), kernel_name)

            _, _, kernel_path = get_path(code_hash(src_code), "py")

            compile_wrapper = IndentedBuffer()
            compile_wrapper.writeline(f"async_compile.{V.graph.device_type}(r'''")
            compile_wrapper.splice(src_code, strip=True)
            compile_wrapper.writeline(
                f"''', 'so', aot_compile={str(V.graph.aot_mode)})"
            )

            metadata_comment = f"# kernel path: {kernel_path}"
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment += "\n" + origins + "\n" + detailed_origins
            wrapper.define_kernel(
                kernel_name, compile_wrapper.getvalue(), metadata_comment
            )

            # For JIT cpp_wrapper, the kernel call site emits a bare
            # `extern "C"` symbol reference; compile the .so now so the
            # wrapper compile can link against it via extra_flags.
            # Autotune compiled a .so for the unfused kernel name
            # (e.g. cutlass_<hash>), but at codegen time the kernel name
            # gains a descriptive prefix (cutlass_fused_mm_<hash>), so the
            # exported symbol differs and we need a fresh compile.
            if V.graph.cpp_wrapper and not V.graph.aot_mode:
                from ...codecache import CUDACodeCache, XPUCodeCache

                codecache_cls = (
                    XPUCodeCache if V.graph.device_type == "xpu" else CUDACodeCache
                )
                so_path, _, _ = codecache_cls.compile(src_code, "so")
                wrapper.external_kernel_libs.add(so_path)
        return kernel_name

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ):
        # [FUNCTION DOC] Generates a CUTLASS template kernel and its call. Bumps the
        # epilogue fusion counter, verifies (AssertionError otherwise) that the template
        # node wraps a CUTLASS template buffer with reduction numel one and that all
        # epilogue IR nodes are computed buffers. Builds the kernel and render function,
        # marks all nodes as run, emulates stores/loads so memory planning sees them,
        # renders the source, defines the kernel, sets up debug printing, emits the call
        # and a comment, then merges removed buffers into the graph and frees buffers.
        # Prologue nodes are accepted but unused.
        """
        Codegen a cutlass template, possibly with fused epilogues
        """
        counters["inductor"]["cutlass_epilogue_fusion_counter"] += len(epilogue_nodes)
        if not self.is_cutlass_template(template_node):
            raise AssertionError(
                "Template node passed to CUTLASSScheduling.codegen_template must be a SchedulerNode that wraps a CUTLASSTemplateBuffer"
            )
        _, (_numel, rnumel) = template_node.group
        if rnumel != 1:
            raise AssertionError(f"expected rnumel == 1, got {rnumel}")
        ctb: CUTLASSTemplateBuffer = cast(CUTLASSTemplateBuffer, template_node.node)
        epilogue_ir_nodes: list[Buffer] = [n.node for n in epilogue_nodes]  # type: ignore[misc]
        if not all(isinstance(n, ComputedBuffer) for n in epilogue_ir_nodes):
            raise AssertionError(
                "Epilogue nodes must all be instances of ir.ComputedBuffer"
            )
        kernel, render = ctb.make_kernel_render(  # type: ignore[misc]
            ctb, epilogue_nodes=epilogue_nodes
        )
        with kernel:
            for node in [template_node, *epilogue_nodes]:
                node.mark_run()

            # typically there is a codegen pass which runs after mark_run
            # for this kernel we've already generated the C++ code, but we still
            # need to let the kernel know about loads/stores that occur in the fused
            # kernel for memory planning to properly optimize allocations
            ctb.emulate_store_fn()
            for node in epilogue_ir_nodes:
                with V.set_ops_handler(MockCutlassHandler(V.get_ops_handler())):
                    # Not sure why we need to do this again
                    if not isinstance(node, ComputedBuffer):
                        raise AssertionError(
                            f"expected ComputedBuffer, got {type(node)}"
                        )
                    node.get_store_function()(CutlassEVTCodegen.get_index_vars(node))

        with V.set_kernel_handler(kernel):
            src_code = render()
            node_schedule = [template_node, *epilogue_nodes]
            kernel_name = self.define_kernel(src_code, node_schedule)

        # debug printing values of intermediate tensors
        _, call_args, arg_signatures, _ = kernel.args.python_argdefs()
        debug_printer_manager = V.graph.wrapper_code.debug_printer
        debug_printer_manager.set_printer_args(
            call_args, kernel_name, arg_signatures, kernel
        )
        with debug_printer_manager:
            self.codegen_comment(node_schedule, kernel_name)
            kernel.call_kernel(kernel_name, ctb)

        V.graph.removed_buffers |= kernel.removed_buffers
        self.free_buffers_in_scheduler()

    @staticmethod
    def _unwrap_epilogue_nodes(
        fused_node: FusedSchedulerNode,
    ) -> list[BaseSchedulerNode]:
        # [FUNCTION DOC] HELPER: Returns the nodes of a fused node excluding its template
        # node. Raises AssertionError if any node lacks an IR node.
        nodes = fused_node.get_nodes()
        template_node = fused_node.get_template_node()
        if not all(n.node is not None for n in nodes):
            raise AssertionError("All epilogue nodes should have an IRNode")
        # pyrefly: ignore [redundant-cast]
        return cast(
            list[BaseSchedulerNode], [n for n in nodes if n.node is not template_node]
        )

    @staticmethod
    def _is_compatible_reshape(
        template_size: Sequence[sympy.Expr], node_size: Sequence[sympy.Expr]
    ) -> bool:
        # [FUNCTION DOC] HELPER: True if node_size is a strictly higher-rank split of
        # template_size, i.e. consecutive node dimensions multiply to each template
        # dimension in order and all dimensions are consumed. False if the node rank is
        # not larger or grouping fails.
        """
        Check if node_size is a compatible reshape of template_size.
        This allows cases like template [8192, 3072] with node [16, 512, 3072]
        where [16*512, 3072] == [8192, 3072] (prefix dims multiply to match).
        """
        if len(node_size) <= len(template_size):
            return False
        # Try to merge consecutive node dims to reconstruct template dims
        t_idx = 0
        n_idx = 0

        def _matches_template_dim(lhs: sympy.Expr, rhs: sympy.Expr) -> bool:
            # [FUNCTION DOC] HELPER: True if the two size expressions are equal, using a
            # direct comparison, then a static equality query, then symbolic simplification.
            # Fast path for concrete/static equality before invoking simplify.
            if lhs == rhs:
                return True
            if V.graph.sizevars.statically_known_equals(lhs, rhs):
                return True
            return sympy.simplify(lhs - rhs) == 0

        while t_idx < len(template_size) and n_idx < len(node_size):
            product = node_size[n_idx]
            n_idx += 1
            # Try multiplying consecutive node dims until we match template dim
            while not _matches_template_dim(product, template_size[t_idx]):
                if n_idx >= len(node_size):
                    return False
                product = product * node_size[n_idx]
                n_idx += 1
            t_idx += 1
        return t_idx == len(template_size) and n_idx == len(node_size)

    def _can_fuse_epilogue_impl(
        self,
        cutlass_template_buffer: CUTLASSTemplateBuffer,
        existing_epilogue_nodes: list[BaseSchedulerNode],
        node_to_fuse: BaseSchedulerNode,
    ) -> bool:
        # [FUNCTION DOC] HELPER: Core epilogue fusion legality check. Returns False (with
        # a logged reason) unless every constituent node of node_to_fuse is a pointwise
        # computed buffer with a name and a size equal to, or a compatible reshape of,
        # the template output size; the node has no aliasing or mutation, is not a
        # reduction, config enables cutlass epilogue fusion, and the template supports
        # epilogue fusion. Then tries generating EVT python code for the existing plus
        # new nodes; NotImplementedError (unsupported op or dtype) rejects fusion.
        # Finally rejects external reads that are non-contiguous compatible reshapes.
        # Raises AssertionError for wrong buffer type or if the first epilogue node does
        # not read the template buffer.
        # KNOWN ISSUE: the unsupported-op branch tests for an _op_ prefix on the error
        # text, but the EVT handler raises with the plain op name, so that branch likely
        # never triggers and all such failures report as unsupported dtype.
        """
        Check if the given node can be fused with the epilogue. At the moment, Kernels
        support fusion with Pointwise operations, wrapped in (named) ComputedBuffer nodes.

        Args:
            cutlass_template_buffer : A CUTLASSTemplateBuffer object representing the CUTLASS template and it's result buffer
            existing_epilogue_nodes : List[SchedulerNode]: The list of already fused epilogue nodes.
            node_to_fuse: The SchedulerNode node to be checked if it can be fused with the epilogue.
        Returns:
        - bool: True if the given node can be fused with the epilogue, False otherwise.

        """
        why = WhyNoFuseNames(
            cutlass_template_buffer.get_name(), node_to_fuse.get_name()
        )

        scheduler_nodes_to_fuse = node_to_fuse.get_nodes()

        if not isinstance(cutlass_template_buffer, CUTLASSTemplateBuffer):
            raise AssertionError(
                f"expected CUTLASSTemplateBuffer, got {type(cutlass_template_buffer)}"
            )

        # Checks on constituent nodes
        for s_node in scheduler_nodes_to_fuse:
            node = s_node.node

            if not isinstance(node, ComputedBuffer):
                why(f"{node} is not a ComputedBuffer")
                return False
            elif not isinstance(node.data, Pointwise):
                why(f"{node} is not a Pointwise op")
                return False
            elif not node.get_computed_buffer_name():  # type: ignore[attr-defined]
                why(f"{node} does not have a computed buffer name")
                return False

            name = node.get_computed_buffer_name()  # type: ignore[attr-defined]
            # dtype can differ, and strides can differ as long as they are broadcastable
            # Allow compatible reshapes (e.g., [8192, 3072] vs [16, 512, 3072])
            # where the node's shape is a split of the template's shape
            if node.get_size() != cutlass_template_buffer.get_size():
                if not self._is_compatible_reshape(
                    cutlass_template_buffer.get_size(), node.get_size()
                ):
                    why(
                        f"{name}'s size: {node.get_size()} differs from {cutlass_template_buffer.get_name()}'s \
size: {cutlass_template_buffer.get_size()}"
                    )
                    return False

        if not (
            len(existing_epilogue_nodes)
            or cutlass_template_buffer.get_name()
            in OrderedSet([rd.name for rd in node_to_fuse.read_writes.reads])
        ):
            raise AssertionError(
                "First epilogue node must read from cutlass template buffer"
            )

        if node_to_fuse.has_aliasing_or_mutation():
            why(f"{node_to_fuse.get_name()} has aliasing or mutation")
            return False
        elif node_to_fuse.is_reduction():
            why(
                f"{node_to_fuse.get_name()} is a reduction which is not yet supported by EVT"
            )
            return False
        elif (
            not config.cutlass.cutlass_epilogue_fusion_enabled
            or not config.epilogue_fusion
        ):
            why("cutlass epilogue fusion is not enabled")
            return False
        elif not cutlass_template_buffer.supports_epilogue_fusion:
            why("epilogue fusion is only supported for TMA-enabled gemm ops")
            return False

        try:
            from torch._inductor.codegen.cutlass.python_evt import CutlassEVTCodegen

            read_names, _, _, _ = CutlassEVTCodegen.ir_to_evt_python_code(
                cutlass_template_buffer.get_name(),
                existing_epilogue_nodes + list(node_to_fuse.get_nodes()),
                OrderedSet(),
            )

        except NotImplementedError as e:
            not_implemented_op = str(e)
            if not_implemented_op.startswith("_op_"):
                not_implemented_op = not_implemented_op[4:]
                why(
                    f"Cannot fuse epilogue node {node_to_fuse} into {cutlass_template_buffer.name}, \
likely due to unsupported operation: {not_implemented_op}"
                )
                return False
            else:  # Likely due to unsupported dtype.
                why(
                    f"Cannot fuse epilogue node {node_to_fuse} into {cutlass_template_buffer.name}. \
Reason: {not_implemented_op}"
                )
                return False

        # An external read whose shape is a compatible reshape of the 2D GEMM
        # output is normalized to the template shape during rendering, but only
        # when it is contiguous (the row-major flatten is memory-equivalent).
        # A non-contiguous compatible-reshape read cannot be safely flattened,
        # and EVT would fail shape propagation against the (1, M, N) accumulator,
        # so reject the fusion in that case.
        template_size = cutlass_template_buffer.get_size()
        all_bufs = V.graph.name_to_buffer | V.graph.graph_inputs
        for read_name in read_names:
            buf = all_bufs.get(read_name)
            if (
                buf is not None
                and self._is_compatible_reshape(template_size, buf.get_size())
                and not buf.get_layout().is_contiguous()
            ):
                why(
                    f"external read {read_name} with size {buf.get_size()} is a "
                    f"non-contiguous compatible reshape of template size "
                    f"{template_size}, which CUTLASS EVT cannot align to the "
                    f"accumulator shape"
                )
                return False

        return True
