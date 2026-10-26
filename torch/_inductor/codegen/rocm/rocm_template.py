# mypy: allow-untyped-defs
# [MODULE DOC] Base class for ROCm C++ kernel templates used by Inductor autotuning. Provides
# ArgInfo (runtime argument name/type), ROCmTemplate (generates a ROCmTemplateCaller with a
# benchmark request and kernel-render factory, plus C++ header/globals snippets). Subclasses
# implement render. No import-time side effects besides creating a logger.
import functools
import itertools
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

from ...autotune_process import TensorMeta
from ...ir import Buffer, IRNode, Layout
from ...utils import IndentedBuffer, unique
from ...virtualized import V
from ..common import KernelTemplate
from .rocm_benchmark_request import ROCmBenchmarkRequest
from .rocm_kernel import ROCmTemplateCaller, ROCmTemplateKernel
from .rocm_template_buffer import ROCmTemplateBuffer
from .rocm_utils import DTYPE_TO_ROCM_TYPE


log = logging.getLogger(__name__)


# FIXME: unify with the CUDA version
@dataclass(frozen=True)
class ArgInfo:
    # [CLASS DOC] Frozen record describing one runtime kernel argument: name and C++ type string ty.
    name: str
    ty: str


class ROCmTemplate(KernelTemplate):
    # [CLASS DOC] Abstract base for ROCm C++ templates. State: input_nodes, output_node (a buffer
    # named buf_out with the given layout), input_reorder, layout. Class attributes: index_counter
    # (shared counter making unique kernel hash names) and gfx9_threads_per_warp (64).
    # Key methods: generate (build caller), header/globals (C++ snippets), render (abstract),
    # get_runtime_arg_info/values (hooks returning empty lists by default).
    index_counter = itertools.count()
    gfx9_threads_per_warp = 64

    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        input_reorder: list[int] | None = None,
    ) -> None:
        # [FUNCTION DOC] Initializes the base template with name and stores input nodes, a fresh
        # output buffer named buf_out using the layout, the optional input reorder list and layout.
        """

        Baseclass for ROCm C++ Templates, derived from KernelTemplate. Not to be instantiated directly.

        Args:
            name (str): The name of the ROCmTemplate object.
            input_nodes (List[IRNode]): A list of input IRNodes.
            layout (Layout): The layout of the output buffer / tensor.
            input_reorder (Optional[List[int]]): An optional list that specifies the order of the input nodes.

        """
        super().__init__(name)
        self.input_nodes = input_nodes
        self.output_node: Buffer = Buffer(name="buf_out", layout=layout)
        self.input_reorder = input_reorder
        self.layout = layout

    def generate(  # type: ignore[override]
        self,
        **kwargs,
    ) -> ROCmTemplateCaller:
        # [FUNCTION DOC] Renders the template once inside a ROCmTemplateKernel (with dtype lookup
        # patched to the output node), logs the code and args, checks that the kernel's call args
        # start with the (reordered, de-duplicated) input names followed by the output name, then
        # builds a ROCmBenchmarkRequest (size args resolved to ints plus runtime args) and returns a
        # ROCmTemplateCaller holding a factory that makes a kernel and render callable for epilogue fusion.
        # Raises AssertionError on call-arg mismatch. Increments the shared index counter.
        """
        Generates the ROCm template caller object for the given GEMM template and operation. This ROCmTemplateCaller
        may be used to call and benchmark the generated ROCm kernel in a standalone manner to enable Autotuning.

        Args:
            kwargs: Additional keyword arguments.

        Returns:
            A ROCmTemplateCaller object representing the generated ROCm template caller.
        """
        kernel_name = f"rocm_{self.name}"
        kernel_hash_name = f"rocm_{self.name}_{next(self.index_counter)}"
        with (
            patch.object(V.graph, "get_dtype", self._fake_get_dtype(self.output_node)),
            ROCmTemplateKernel(
                kernel_name=kernel_name,
                runtime_arg_info=self.get_runtime_arg_info(),
                runtime_arg_values=self.get_runtime_arg_values(**kwargs),
            ) as kernel,
        ):
            code = self.render(kernel=kernel, **kwargs)
            _, call_args, _, _ = kernel.args.python_argdefs()
            log.debug("Autotune key: %s, Generated Code:\n%s", kernel_hash_name, code)
            log.debug(
                "Args: cpp_argdefs: %s, python_argdefs: %s",
                kernel.args.cpp_argdefs(DTYPE_TO_ROCM_TYPE),
                kernel.args.python_argdefs(),
            )

        input_reorder = (
            self.input_reorder
            if self.input_reorder is not None
            else list(range(len(self.input_nodes)))
        )
        expected_args = list(
            unique(self.input_nodes[idx].get_name() for idx in input_reorder)
        )
        expected_args.extend([self.output_node.get_name()])
        if list(call_args)[: len(expected_args)] != expected_args:
            raise AssertionError(
                (
                    call_args,
                    expected_args,
                )
            )

        size_args = (
            self.size_args() if hasattr(self, "size_args") else ()
        )  # subclass should define def size_args()
        # Resolve symbolic sizes to concrete ints for benchmarking only.
        size_args_ints = list(V.graph.sizevars.optimization_hints(size_args))
        # The runtime args come right after the size args
        runtime_args = self.get_runtime_arg_values(**kwargs)
        extra_args = size_args_ints + runtime_args
        bmreq = ROCmBenchmarkRequest(
            kernel_name=kernel_name,
            input_tensor_meta=TensorMeta.from_irnodes(self.input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
            extra_args=extra_args,
            source_code=code,
        )

        def make_kernel_render(
            template_node: ROCmTemplateBuffer,
            epilogue_nodes: Sequence[IRNode] | None = None,
        ):
            # [FUNCTION DOC] HELPER: creates a ROCmTemplateKernel with a placeholder name and a
            # partial of render bound to it, the template buffer node, epilogue nodes and generate kwargs.
            # Returns the (kernel, render) pair.
            kernel = ROCmTemplateKernel(
                kernel_name="KERNEL_NAME",
                runtime_arg_info=self.get_runtime_arg_info(),
                runtime_arg_values=self.get_runtime_arg_values(**kwargs),
            )
            render = functools.partial(
                self.render,
                kernel=kernel,
                template_buffer_node=template_node,
                epilogue_nodes=epilogue_nodes,
                **kwargs,  # includes "op" argument in case of CUTLASSGemmTemplate
            )
            return kernel, render

        return ROCmTemplateCaller(
            kernel_hash_name,
            self.name,
            self.input_nodes,
            self.output_node.get_layout(),
            make_kernel_render,
            bmreq,
            self,
            kwargs,
        )

    def header(self) -> IndentedBuffer:
        # [FUNCTION DOC] Returns an IndentedBuffer with the standard C++ includes (exception,
        # iostream, memory, random, vector).
        res = IndentedBuffer()
        res.splice(
            """
                #include <exception>
                #include <iostream>
                #include <memory>
                #include <random>
                #include <vector>
            """
        )
        return res

    def globals(self) -> IndentedBuffer:
        # [FUNCTION DOC] Returns an IndentedBuffer defining the PT_EXPORT macro (default symbol
        # visibility on GNU compilers, dllexport on Windows, empty otherwise).
        res = IndentedBuffer()
        res.splice(
            """
                // We compile all models with -fvisibility=hidden. Any symbols that need to be
                // exposed in the final shared library must be declared with PT_EXPORT to make
                // them visible.
                #ifdef __GNUC__ // Applies to any compiler with GNU extensions (clang and g++)
                #define PT_EXPORT __attribute__((__visibility__("default")))
                #else
                #ifdef _WIN32
                #define PT_EXPORT __declspec(dllexport)
                #else
                #define PT_EXPORT
                #endif
                #endif
            """
        )
        return res

    def render(self, **kwargs) -> str:
        # [FUNCTION DOC] Abstract hook: subclasses return the kernel C++ source. Always raises NotImplementedError here.
        raise NotImplementedError

    def get_runtime_arg_info(self) -> list[ArgInfo]:
        # [FUNCTION DOC] Hook returning descriptions of extra runtime kernel args; empty by default.
        return []

    def get_runtime_arg_values(self, **kwargs) -> list[Any]:
        # [FUNCTION DOC] Hook returning values for the runtime kernel args; empty by default.
        return []
