# mypy: allow-untyped-defs
# [MODULE DOC] Defines CppTemplate, the base class for inductor C++ code templates. It renders a
# template through CppTemplateKernel, validates call arguments, builds a CppBenchmarkRequest and
# returns a CppTemplateCaller for autotuning. No import-time side effects besides creating a logger.
import ctypes
import functools
import itertools
import logging
import sys
from collections.abc import Callable, Iterable
from unittest.mock import patch

import sympy

from .. import config, ir
from ..autotune_process import CppBenchmarkRequest, TensorMeta
from ..utils import IndentedBuffer, Placeholder, unique
from ..virtualized import V
from .common import KernelTemplate
from .cpp_template_kernel import CppTemplateCaller, CppTemplateKernel


log = logging.getLogger(__name__)


class CppTemplate(KernelTemplate):
    # [CLASS DOC] Base class for C++ kernel templates used by inductor CPU autotuning (e.g. GEMM).
    # State: input_nodes, output_node (a buffer or list of buffers named by a unique per-instance
    # index), layout, num_threads and an optional epilogue_creator. The class-level index_counter
    # makes names unique. generate() produces a CppTemplateCaller; header() gives common includes;
    # render() must be implemented by subclasses.
    index_counter = itertools.count()

    def __init__(
        self,
        name: str,
        input_nodes,
        layout: ir.Layout,
        num_threads: int,
        epilogue_creator: Callable[[ir.Buffer], ir.Pointwise] | None = None,
    ) -> None:
        # [FUNCTION DOC] Stores the input nodes, output layout, thread count and optional epilogue
        # creator; takes a unique index from the class-wide counter (mutating it) and creates the output
        # buffer node named with that index.
        super().__init__(name)
        self.input_nodes = input_nodes
        self.index = next(self.index_counter)
        self.output_node: ir.Buffer | list[ir.Buffer] = ir.Buffer(
            name=f"buf_out{self.index}", layout=layout
        )
        self.layout = layout
        self.num_threads = num_threads
        self.epilogue_creator = epilogue_creator

    def generate(self, **kwargs):
        # [FUNCTION DOC] Renders this template once to C++ source and wraps it as a CppTemplateCaller
        # for autotuning. Under temporary patches (fake dtype lookup for the output node, flexible layout
        # indexing allowed, current device set to the layout device) it renders with a CppTemplateKernel,
        # logs the code and arguments, and checks that the leading kernel call arguments equal the
        # unique input names followed by the output name(s); raises AssertionError otherwise.
        # Remaining (size) arguments are converted via size hints to unsigned 64-bit ctypes values for
        # benchmarking only. Builds a CppBenchmarkRequest from tensor metadata and the source.
        # Returns the CppTemplateCaller with a kernel-render factory; kwargs go to the render call.
        kernel_name = f"cpp_{self.name}"
        with (
            patch.object(V.graph, "get_dtype", self._fake_get_dtype(self.output_node)),
            patch.object(ir.FlexibleLayout, "allow_indexing", True),
            V.graph.set_current_device(self.layout.device),
            CppTemplateKernel(
                kernel_name=kernel_name, num_threads=self.num_threads
            ) as kernel,
        ):
            code = kernel.render(self, **kwargs)
            _, call_args, _, _ = kernel.args.python_argdefs()
            log.debug("Generated Code:\n%s", code)
            log.debug(
                "Args: cpp_argdefs: %s, python_argdefs: %s",
                kernel.args.cpp_argdefs(),
                kernel.args.python_argdefs(),
            )

        expected_args = list(
            unique(input_node.get_name() for input_node in self.input_nodes)
        )
        if isinstance(self.output_node, Iterable):
            expected_args.extend([node.get_name() for node in self.output_node])
        else:
            expected_args.extend([self.output_node.get_name()])
        if list(call_args)[: len(expected_args)] != expected_args:
            raise AssertionError(
                (
                    call_args,
                    expected_args,
                )
            )
        # extra_args are only used for benchmarking, not compiled kernel correctness
        extra_args = V.graph.sizevars.optimization_hints(
            map(sympy.expand, call_args[len(expected_args) :])
        )
        # Cast the size hint from int to ctypes.c_ulonglong explicitly
        # since in cpp kernel, we bind it to C long
        extra_args = tuple(ctypes.c_ulonglong(x) for x in extra_args)

        kernel_hash_name = f"cpp_{self.name}_{self.index}"

        # Create the BenchmarkRequest for CPP
        bmreq = CppBenchmarkRequest(
            kernel_name=kernel_name,
            input_tensor_meta=TensorMeta.from_irnodes(self.input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
            extra_args=extra_args,
            source_code=code,
        )

        def make_kernel_render(
            template_node: ir.CppTemplateBuffer,
            flag_template_buffer_has_other_users: bool,
            epilogue_nodes: list[ir.IRNode] | None = None,
        ):
            # [FUNCTION DOC] HELPER: Callback handed to the CppTemplateCaller. Creates a fresh
            # CppTemplateKernel named by a kernel-name placeholder and a render callable bound to this
            # template, the given template buffer node, a flag saying whether the template buffer has
            # other users, the optional epilogue nodes and the original generate kwargs.
            # Returns the pair (kernel, render callable); the callable is not invoked here.
            kernel = CppTemplateKernel(
                kernel_name=str(Placeholder.KERNEL_NAME), num_threads=self.num_threads
            )
            render = functools.partial(
                kernel.render,
                self,
                template_buffer_node=template_node,
                flag_template_buffer_has_other_users=flag_template_buffer_has_other_users,
                epilogue_nodes=epilogue_nodes,
                **kwargs,
            )
            return kernel, render

        return CppTemplateCaller(
            kernel_hash_name,
            self.name,
            self.input_nodes,
            # pyrefly: ignore [bad-index, index-error]
            self.output_node[0].get_layout()
            if isinstance(self.output_node, Iterable)
            else self.output_node.get_layout(),
            make_kernel_render,
            bmreq,
            self,
        )

    def header(self) -> IndentedBuffer:
        # [FUNCTION DOC] Builds the common C++ include preamble in a new indented buffer: the inductor
        # cpp prefix, the c10 unroll header and the AOTI shim header. When kernel profiling is enabled in
        # config and the platform is linux or win32, also includes the AOTI runtime utils header.
        # Returns the buffer; no global side effects.
        res = IndentedBuffer()
        res.writeline("#include <torch/csrc/inductor/cpp_prefix.h>")
        # TODO: add c10::ForcedUnroll test to test_aoti_abi_check
        res.splice("""#include <c10/util/Unroll.h>""")
        res.splice("""#include <torch/csrc/inductor/aoti_torch/c/shim.h>""")
        enable_kernel_profile = config.cpp.enable_kernel_profile and sys.platform in [
            "linux",
            "win32",
        ]
        if enable_kernel_profile:
            res.writelines(["#include <torch/csrc/inductor/aoti_runtime/utils.h>"])
        return res

    def render(self, **kwargs) -> str:
        # [FUNCTION DOC] Abstract hook: subclasses return the C++ source text of the template.
        # Always raises NotImplementedError in this base class.
        raise NotImplementedError
