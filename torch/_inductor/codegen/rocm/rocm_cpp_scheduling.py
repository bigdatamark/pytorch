# mypy: allow-untyped-defs
# [MODULE DOC] Scheduler support for ROCm C++ template kernels. Defines ROCmCPPScheduling, which decides
# fusion (none) and emits kernel definitions and calls into the wrapper code.
import logging
from collections.abc import Sequence
from typing import cast, TypeGuard

from ... import config
from ...codecache import code_hash, get_path
from ...scheduler import BaseSchedulerNode, BaseScheduling, SchedulerNode
from ...utils import get_fused_kernel_name, get_kernel_metadata, sympy_product
from ...virtualized import V
from ..common import IndentedBuffer
from .rocm_template_buffer import ROCmTemplateBuffer


log = logging.getLogger(__name__)


class ROCmCPPScheduling(BaseScheduling):
    # [CLASS DOC] Partial BaseScheduling for ROCm C++ (CK) template kernels, delegated to by the combined CUDA
    # scheduling. Stateless; handles grouping, fusion decisions, kernel definition and template codegen.
    """
    Partial Scheduling implementation for ROCm C++ Kernels.
    This class is intended to be used in combination with TritonScheduling,
    and delegated to by CUDACombinedScheduling.

    It handles fusion decisions and ROCm C++ specific template code generation.
    """

    def group_fn(self, sizes):
        # [FUNCTION DOC] Returns for each size list a tuple of the simplified product of its sizes.
        return tuple(V.graph.sizevars.simplify(sympy_product(s)) for s in sizes)

    @staticmethod
    def is_rocm_cpp_template(node: BaseSchedulerNode) -> TypeGuard[SchedulerNode]:
        # [FUNCTION DOC] Returns True if node is a SchedulerNode wrapping a ROCmTemplateBuffer.
        return isinstance(node, SchedulerNode) and isinstance(
            node.node, ROCmTemplateBuffer
        )

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # [FUNCTION DOC] Always returns False: ROCm C++ templates are never fused vertically here.
        return False

    def define_kernel(self, src_code: str, node_schedule) -> str:
        # [FUNCTION DOC] Registers the kernel source with the wrapper and returns the kernel name. If the source was
        # seen it reuses the name. Otherwise builds a name from a rocm prefix, optional descriptive fused name
        # and a unique suffix, replaces the KERNEL_NAME placeholder in the source, and defines the kernel in the
        # wrapper as an async_compile rocm call with metadata comments (path, origins). For JIT cpp wrapper
        # (not AOT) it compiles the shared library now and records it as an external kernel lib.
        # Side effects: mutates wrapper.src_to_kernel, kernel definitions and external libs.
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            fused_name = (
                get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
                if config.triton.descriptive_names
                else ""
            )
            kernel_name = "_".join(["rocm", fused_name, wrapper.next_kernel_suffix()])
            # use the original src_code as the key
            wrapper.src_to_kernel[src_code] = kernel_name
            src_code = src_code.replace("KERNEL_NAME", kernel_name)

            _, _, kernel_path = get_path(code_hash(src_code), "py")

            compile_wrapper = IndentedBuffer()
            compile_wrapper.writeline("async_compile.rocm(r'''")
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
            # (e.g. rocm_<hash>), but at codegen time the kernel name
            # gains a descriptive prefix (rocm_fused_mm_<hash>), so the
            # exported symbol differs and we need a fresh compile.
            if V.graph.cpp_wrapper and not V.graph.aot_mode:
                from ...codecache import ROCmCodeCache

                so_path, _, _ = ROCmCodeCache.compile(src_code, "so")
                wrapper.external_kernel_libs.add(so_path)
        return kernel_name

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ):
        """
        # [FUNCTION DOC] Generates code for a ROCm template node (epilogue and prologue nodes are accepted but unused).
        # Raises AssertionError if the node is not a scheduler node of a ROCmTemplateBuffer or if its reduction
        # numel is not one. Renders the kernel source, defines it, emits a comment and the kernel call, then
        # merges the kernel's removed buffers into the graph and frees buffers in the scheduler.
        Codegen a ROCm template, possibly with fused epilogues
        """
        if not self.is_rocm_cpp_template(template_node):
            raise AssertionError(
                "Template node passed to ROCmScheduler.codegen_template must be a SchedulerNode that wraps a ROCmTemplateBuffer"
            )
        _, (_numel, rnumel) = template_node.group
        if rnumel != 1:
            raise AssertionError(f"expected rnumel == 1, got {rnumel}")
        ctb: ROCmTemplateBuffer = cast(ROCmTemplateBuffer, template_node.node)
        kernel, render = ctb.make_kernel_render(ctb)  # type: ignore[misc]
        with kernel:
            template_node.mark_run()
            src_code = render()

        with V.set_kernel_handler(kernel):
            node_schedule = [template_node]
            kernel_name = self.define_kernel(src_code, node_schedule)
        self.codegen_comment(node_schedule, kernel_name)
        kernel.call_kernel(kernel_name, ctb)
        V.graph.removed_buffers |= kernel.removed_buffers
        self.free_buffers_in_scheduler()
