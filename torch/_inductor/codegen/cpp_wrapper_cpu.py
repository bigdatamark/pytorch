# mypy: allow-untyped-defs
# [MODULE DOC] C++ wrapper code generator for Inductor on CPU (and base for GPU/ArrayRef variants).
# Defines CppWrapperCpu, a PythonWrapperCodegen subclass that emits C++ source instead of Python: either an
# AOTInductor model class (AOT mode, run_impl/_const_run_impl plus constructor and kernels class) or a JIT
# entry function (inductor_entry_impl) loaded via CppWrapperCodeCache, or both in dual-wrapper mode.
# Also holds DualWrapperCodeLine (replays captured code to JIT/AOT sides) and DeferredCpuTritonCallWrapper
# (emits per-kernel call wrappers for CPU Triton kernels after autotune populates CpuTritonKernelCache).
# Import-time side effects: none beyond imports.
from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import functools
import math
import os
import sys
import textwrap
from itertools import chain, count
from typing import Any, Literal, TYPE_CHECKING

import sympy

import torch
import torch._higher_order_ops.torchbind
import torch._inductor.async_compile
import torch._ops
from torch import dtype as torch_dtype
from torch._inductor.runtime.runtime_utils import dynamo_timed
from torch.fx.experimental.symbolic_shapes import ConvertIntKey, DivideByKey, SymTypes
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import CleanDiv, FloorDiv, Mod, ModularIndexing
from torch.utils._sympy.symbol import symbol_is_type, SymT

from .. import config, ir
from ..ir import ExternKernel
from ..utils import (
    _align,
    AotOnlyBuffer,
    DualIndentedBuffer,
    make_codegen_buffer,
    normalize_name,
)
from ..virtualized import V
from .aoti_hipify_utils import maybe_hipify_code_wrapper
from .common import get_device_op_overrides, IndentedBuffer, Kernel
from .cpp_utils import (
    cexpr,
    DEVICE_TO_ATEN,
    DEVICE_TO_INT,
    DTYPE_TO_ATEN,
    DTYPE_TO_CPP,
    LAYOUT_TO_ATEN,
)
from .wrapper import (
    _rewrite_symbol_solution_for_int_codegen,
    codegen_reinterpret_view_helper,
    EnterSubgraphLine,
    ExitSubgraphLine,
    HasWriteLine,
    PythonWrapperCodegen,
    SymbolicCallArg,
    WrapperLine,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..graph import GraphLowering
    from ..runtime.hints import TritonMeta

    # At most, the list nesting can go one layer deep.
    _OUTPUT_ARGS_TYPE = list[str | None | list[str | None]]

    from ..scheduler import BaseSchedulerNode


@dataclasses.dataclass
class DualWrapperCodeLine(WrapperLine):
    # [CLASS DOC] HELPER: Wrapper line that carries two code snippets, one for the JIT output and one for the
    # AOT-only output. Attributes: jit_code (IndentedBuffer) and aot_code (AotOnlyBuffer).
    jit_code: IndentedBuffer
    aot_code: AotOnlyBuffer

    def codegen(self, code: DualIndentedBuffer) -> None:
        # [FUNCTION DOC] HELPER: Replays the captured snippets into a DualIndentedBuffer: jit_code goes to the JIT side
        # and aot_code to the AOT side. On a plain IndentedBuffer only the JIT part is written. Mutates the buffer.
        # In dual-wrapper replay, these route each captured snippet to the
        # matching side. If replayed on a plain IndentedBuffer, splice_jit writes
        # normally and splice_aot is a no-op, preserving the same selection.
        code.splice_jit(self.jit_code)
        code.splice_aot(self.aot_code)


@dataclasses.dataclass
class DeferredCpuTritonCallWrapper:
    # [CLASS DOC] Deferred emitter of C++ call wrappers for one CPU Triton kernel. Holds the wrapper function name,
    # kernel name, per-argument types, and optional triton/inductor metadata. generate() reads the compiled kernel
    # info from CpuTritonKernelCache (so it must run after the autotune block ran) and writes a static wrapper
    # function into the C++ prefix that loads the kernel and launcher shared objects once and launches them.
    """CPU counterpart of DeferredTritonCallWrapper in cpp_wrapper_gpu.py.

    Deferred so wrapper emission can read `CpuTritonKernelCache` entries
    after the autotune block has populated them.

    NOTE: CPU doesn't actively use autotuning, but we mirror
    the GPU flow for symmetry.
    """

    wrapper_name: str
    kernel_name: str
    arg_types: list[Any]
    triton_meta: TritonMeta | None = None
    inductor_meta: dict[str, Any] | None = None

    def _get_cpp_param_type(self, name: str, arg_type: Any) -> str:
        # [FUNCTION DOC] HELPER: Maps one wrapper argument (name and python type) to a C++ parameter declaration:
        # a tensor dtype becomes a templated const reference, int/sympy/symbolic args become a 64-bit integer, float
        # and bool map to their C++ types. Raises ValueError for any other type.
        if isinstance(arg_type, torch_dtype):
            return f"const {name}_type_& {name}"
        elif issubclass(arg_type, (SymbolicCallArg, sympy.Expr, int)):
            return f"int64_t {name}"
        elif arg_type is float:
            return f"float {name}"
        elif arg_type is bool:
            return f"bool {name}"
        else:
            raise ValueError(
                f"Unexpected arg type {arg_type} for CPU triton wrapper arg '{name}'"
            )

    def _resolve_arg_names(
        self,
    ) -> tuple[list[str], list[str], list[str]]:
        # [FUNCTION DOC] HELPER: Looks up the kernel in CpuTritonKernelCache and returns three name lists: wrapper args
        # (signature names minus declared constexpr names, plus extra launcher grid args), kernel args (signature names
        # whose type is not constexpr) and grid args. Raises AssertionError if the cache has no entry for the kernel.
        """Return (wrapper_arg_names, kernel_arg_names, grid_arg_names).

        Wrapper args exclude *declared* constexpr (the call site doesn't pass
        them); kernel args additionally exclude *value-specialized* constexpr
        (baked into the binary). Grid args are appended for `run_from_nativert`.
        """
        from torch._inductor.codecache import CpuTritonKernelCache

        info = CpuTritonKernelCache.get(self.kernel_name)
        if info is None:
            raise AssertionError(
                f"CpuTritonKernelCache not populated for {self.kernel_name}"
            )
        signature = info["signature"]
        declared_constexpr_names = OrderedSet(
            (self.inductor_meta or {}).get("declared_constexpr_names", [])
        )
        wrapper_arg_names = [
            name for name in signature if name not in declared_constexpr_names
        ]
        kernel_arg_names = [name for name, ty in signature.items() if ty != "constexpr"]
        grid_arg_names = list((self.inductor_meta or {}).get("extra_launcher_args", []))
        return wrapper_arg_names + grid_arg_names, kernel_arg_names, grid_arg_names

    def _write_wrapper_signature(
        self,
        prefix: IndentedBuffer,
        wrapper_arg_names: list[str],
    ) -> None:
        # [FUNCTION DOC] HELPER: Writes the C++ function signature (with optional template header) of the call wrapper
        # into prefix: template params for dtype args (and the kernels type in AOT mode), one parameter per wrapper arg,
        # an AOT kernels reference, and an optional cubin directory. Raises AssertionError if the number of names and
        # arg types differ.
        # `cubin_dir_` name reused from GPU so the same AOTI runtime member feeds both.
        if len(wrapper_arg_names) != len(self.arg_types):
            raise AssertionError(
                (
                    wrapper_arg_names,
                    self.arg_types,
                )
            )
        template_types = [
            f"typename {name}_type_"
            for name, arg_type in zip(wrapper_arg_names, self.arg_types)
            if isinstance(arg_type, torch_dtype)
        ]
        if V.graph.aot_mode:
            template_types.append("typename kernels_type_")
        if template_types:
            prefix.writeline(f"template <{', '.join(template_types)}>")

        param_lines = [
            self._get_cpp_param_type(name, arg_type)
            for name, arg_type in zip(wrapper_arg_names, self.arg_types)
        ]
        if V.graph.aot_mode:
            param_lines.append("kernels_type_& kernels_")
        param_lines.append(
            "const std::optional<std::string>& cubin_dir_ = std::nullopt"
        )

        prefix.writeline(f"static __attribute__((noinline)) void {self.wrapper_name}(")
        with prefix.indent():
            for i, param in enumerate(param_lines):
                comma = "," if i < len(param_lines) - 1 else ""
                prefix.writeline(f"{param}{comma}")
        prefix.writeline("){")

    def _generate_load_kernel(
        self,
        prefix: IndentedBuffer,
        kernel_so: str,
        launcher_so: str,
        kernel_symbol: str,
    ) -> None:
        # [FUNCTION DOC] HELPER: Emits C++ that, under a once flag for thread safety, loads the kernel and launcher
        # shared libraries into the kernels object members for this kernel using the given paths and symbol name.
        # `std::call_once` keeps init thread-safe under concurrent first calls.
        kernel_member = f"kernels_.{self.kernel_name}_kernel_fn"
        launcher_member = f"kernels_.{self.kernel_name}_launcher_fn"
        prefix.splice(f"""
            using namespace torch::aot_inductor;
            static std::once_flag {self.kernel_name}_init_flag;
            std::call_once({self.kernel_name}_init_flag, [&]() {{
                {kernel_member} = loadCpuTritonKernel(
                    "{kernel_so}", "{kernel_symbol}", cubin_dir_);
                {launcher_member} = loadCpuTritonLauncher(
                    "{launcher_so}", cubin_dir_);
            }});
            """)

    def _generate_launch(
        self,
        prefix: IndentedBuffer,
        kernel_arg_names: list[str],
        grid_arg_names: list[str],
        signature: dict[str, Any],
    ) -> None:
        # [FUNCTION DOC] HELPER: Emits the C++ that builds the argument pointer array and calls the launcher. Arguments
        # follow the full signature order: constexpr slots are null pointers, pointer-typed args pass data_ptr, scalars
        # pass their address. Grid x/y/z come from grid_arg_names (default 1) cast to unsigned 32-bit; threads fixed at 1.
        # KNOWN ISSUE: kernel_arg_names is accepted but never used.
        # The launcher uses literal `args[i]` indices over the *original*
        # signature, so constexpr slots must be padded (nullptr) to keep
        # non-constexpr args at the expected positions.
        kernel_member = f"kernels_.{self.kernel_name}_kernel_fn"
        launcher_member = f"kernels_.{self.kernel_name}_launcher_fn"

        args_array_entries: list[str] = []
        for name, ty in signature.items():
            if ty == "constexpr":
                args_array_entries.append("nullptr")
            elif isinstance(ty, str) and ty.startswith("*"):
                args_array_entries.append(f"{name}.data_ptr()")
            else:
                args_array_entries.append(f"&{name}")
        args_arr_str = ", ".join(args_array_entries)

        # Grid args are int64_t in the wrapper but the launcher ABI is uint32_t.
        grid_x = grid_arg_names[0] if len(grid_arg_names) > 0 else "1"
        grid_y = grid_arg_names[1] if len(grid_arg_names) > 1 else "1"
        grid_z = grid_arg_names[2] if len(grid_arg_names) > 2 else "1"

        # TODO: thread `num_cpu_threads` from inductor_meta (autotune support).
        prefix.splice(f"""
            void* args_arr[] = {{ {args_arr_str} }};
            {launcher_member}(
                static_cast<uint32_t>({grid_x}),
                static_cast<uint32_t>({grid_y}),
                static_cast<uint32_t>({grid_z}),
                /*num_cpu_threads=*/1,
                args_arr,
                {kernel_member});
            """)

    def generate(self, wrapper: CppWrapperCpu) -> None:
        # [FUNCTION DOC] Emits the full call wrapper for this kernel into wrapper.prefix (signature, kernel load, launch,
        # closing brace) and registers the kernel and launcher .so paths in the graph wrapper's additional_files so the
        # AOTI packager includes them. Raises RuntimeError if CpuTritonKernelCache has no entry for the kernel.
        from torch._inductor.codecache import CpuTritonKernelCache

        info = CpuTritonKernelCache.get(self.kernel_name)
        if info is None:
            raise RuntimeError(
                f"CpuTritonKernelCache not populated for '{self.kernel_name}': "
                "the autotune block did not save this kernel. Check that "
                "config.triton.autotune_at_compile_time is True and the Triton "
                "CPU backend emits a launcher .so exporting `run_from_nativert`."
            )

        signature = info["signature"]
        wrapper_arg_names, kernel_arg_names, grid_arg_names = self._resolve_arg_names()

        prefix = wrapper.prefix
        self._write_wrapper_signature(prefix, wrapper_arg_names)
        with prefix.indent():
            self._generate_load_kernel(
                prefix,
                info["kernel_so_path"],
                info["launcher_so_path"],
                info["kernel_symbol"],
            )
            self._generate_launch(prefix, kernel_arg_names, grid_arg_names, signature)
        prefix.writeline("}")

        # Register .so artifacts so the AOTI packager picks them up.
        V.graph.wrapper_code.additional_files.append(info["kernel_so_path"])
        V.graph.wrapper_code.additional_files.append(info["launcher_so_path"])


class CppWrapperCpu(PythonWrapperCodegen):
    # [CLASS DOC] Main C++ wrapper code generator (CPU device by default). Extends PythonWrapperCodegen with C++
    # syntax settings (declarations, comment prefix, null string) and state: ids/counters for temporaries, caches of
    # used device/dtype/layout/memory-format constants, included device headers, initialized GEMM kernels, CPU Triton
    # kernel names and their deferred call wrappers, and the AOT output string used in dual-wrapper mode.
    # Key methods: write_header/write_prefix/write_wrapper_decl (preamble), generate and finalize_prefix (assembly),
    # generate_end (closing JIT or AOT code), and the extern/fallback kernel call emitters using the C shim API.
    """
    Generates cpp wrapper for running on CPU and calls cpp kernels
    """

    def __init__(self):
        # [FUNCTION DOC] Initializes C++ wrapper state. Sets a default device of cpu if a subclass has not set one,
        # computes the AOTI model class name (suffixed by model name unless dynamic linkage), calls the base initializer,
        # then configures C++ syntax (auto declarations, semicolons, slash comments, nullptr) and creates counters,
        # caches and sets for generated names, CPU Triton bookkeeping and device op overrides.
        if not hasattr(self, "device"):
            self.device = "cpu"
        # must be initialized prior to calling super().__init__()
        self.included_devices: OrderedSet[str] = OrderedSet()
        self.model_class_name_suffix = (
            ""
            if config.aot_inductor.dynamic_linkage
            else config.aot_inductor.model_name_for_generated_files
        )
        self.aoti_model_class_name = f"AOTInductorModel{self.model_class_name_suffix}"

        super().__init__()

        self.declare = "auto "
        self.declare_maybe_reference = "decltype(auto) "
        self.ending = ";"
        self.comment = "//"
        self.none_str = "nullptr"
        self.supports_intermediate_hooks = False
        self.kernel_callsite_id = count()
        self.int_array_id = count()  # for int array local variable declarations
        self.declared_int_array_vars: OrderedSet[str] = OrderedSet()
        self.tmp_tensor_id = count()  # for tmp tensor local variable declarations
        self.arg_var_id = count()
        self._aot_output: str | None = None
        # Defined kernels drive member declarations; called kernels drive
        # `call_<k>` emission. A kernel may be defined without being called.
        self._cpu_triton_kernel_names: OrderedSet[str] = OrderedSet()
        self._cpu_triton_call_wrappers: dict[str, DeferredCpuTritonCallWrapper] = {}
        self.used_cached_devices: OrderedSet[str] = OrderedSet()
        self.used_cached_dtypes: OrderedSet[str] = OrderedSet()
        self.used_cached_layouts: OrderedSet[str] = OrderedSet()
        self.used_cached_memory_formats: OrderedSet[str] = OrderedSet()
        self.used_switch_selector: OrderedSet[str] = OrderedSet()
        self.cached_output_id = count()
        self.scalar_to_tensor_id = count()
        self.custom_op_wrapper_loaded = False
        # For GEMM kernels that must be initialized and are resolved at linking.
        self.initialized_kernels: dict[str, Kernel] = {}
        self.device_codegen = get_device_op_overrides(self.device)
        self._included_extra_headers: OrderedSet[str] = OrderedSet()
        self.codegen_int_array_var_cache = {}
        self.needs_vec_isa = self.device == "cpu"

    @contextlib.contextmanager
    def _preserve_device_guard_state(self):
    # [FUNCTION DOC] HELPER: Context manager that saves the last seen device guard index and restores it on exit,
    # even if the body raises.
        last_seen_device_guard_index = self.last_seen_device_guard_index
        try:
            yield
        finally:
            self.last_seen_device_guard_index = last_seen_device_guard_index

    @contextlib.contextmanager
    def _preserve_codegen_state(self, graphs):
    # [FUNCTION DOC] HELPER: Context manager that snapshots wrapper bookkeeping (allocated, freed, reuses,
    # workspaces, int array declarations and cache, kernel numel exprs, switch selectors) and per-subgraph removed
    # buffer sets, then restores them on exit even on error. Used to codegen something without lasting side effects.
        wrapper_state = (
            OrderedSet(self.allocated),
            OrderedSet(self.freed),
            dict(self.reuses),
            dict(self.allocated_workspaces),
            OrderedSet(self.declared_int_array_vars),
            dict(self.codegen_int_array_var_cache),
            OrderedSet(self.kernel_numel_expr),
            OrderedSet(self.used_switch_selector),
        )
        subgraph_state = [
            (
                graph,
                OrderedSet(graph.removed_buffers),
                OrderedSet(graph.inplaced_to_remove),
            )
            for graph in graphs
        ]
        try:
            yield
        finally:
            (
                self.allocated,
                self.freed,
                self.reuses,
                self.allocated_workspaces,
                self.declared_int_array_vars,
                self.codegen_int_array_var_cache,
                self.kernel_numel_expr,
                self.used_switch_selector,
            ) = wrapper_state
            for graph, removed_buffers, inplaced_to_remove in subgraph_state:
                graph.removed_buffers = removed_buffers
                graph.inplaced_to_remove = inplaced_to_remove

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: PythonWrapperCodegen | None,
        partition_signatures: ir.GraphPartitionSignature | None = None,
    ):
        # [FUNCTION DOC] Factory used by the wrapper registry: always returns a fresh CppWrapperCpu; the subgraph
        # arguments are ignored because subgraph codegen is not supported by lifting functions yet.
        # TODO - support subgraph codegen by lifting functions. Check the
        # comment at CppWrapperCpu `codegen_subgraph` function.
        return CppWrapperCpu()

    @contextlib.contextmanager
    def _target_buf(self, attr: str, buf: IndentedBuffer):
        # [FUNCTION DOC] HELPER: Context manager that temporarily replaces the named attribute of self with the given
        # buffer and restores the old value on exit.
        """Temporarily redirect self.<attr> to a specific buffer."""
        saved = getattr(self, attr)
        setattr(self, attr, buf)
        try:
            yield
        finally:
            setattr(self, attr, saved)

    @staticmethod
    def _generate_temporary_array_pointer(
        c_type: str, elements: Sequence[str], *, force_mutable: bool = False
    ) -> str:
        # [FUNCTION DOC] HELPER: Returns a C++ expression for a pointer to a temporary std array built from the
        # elements, valid only within the enclosing statement. Returns a mutable data pointer if force_mutable or the
        # element type is already a pointer type; otherwise a pointer to const elements.
        """Get a pointer to an array that only exists for the duration of the C++
        statement it's used in."""
        # If the c_type is already a pointer, return a mutable pointer to the array.
        # Otherwise, return a const pointer.  In the C-shim API, pointer types are only
        # const-qualified with respect to the underlying value, not any nested pointers.
        # e.g. const double** is possible, but not const double* const*.  This means
        # that an array containing pointers must _already_ be properly const-qualified
        # by the c_type, and not add additional const-ness.
        array_expr = f"std::array<{c_type}, {len(elements)}>{{{', '.join(elements)}}}"
        if force_mutable or c_type.endswith("*"):
            return f"{array_expr}.data()"

        return f"static_cast<const {c_type}*>({array_expr}.data())"

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
        # [FUNCTION DOC] HELPER: Emits a kernel call statement. Requires call_args and arg_types of equal length (else
        # AssertionError). For CPU Triton kernels: optionally runs the Python autotune-block call helper once, registers
        # a deferred call wrapper, and writes a call to it (adding kernels and cubin dir args in AOT mode). Otherwise
        # casts pointer-typed args via data_ptr, stringifies scalars, and writes the kernel call under the debug printer.
        # KNOWN ISSUE: graph_name and current_stream_idx parameters are unused; the docstring about triton and cuda is stale.
        """
        Generates kernel call code.

        triton: Defines whether the GPU backend uses Triton for codegen.
                Otherwise it uses the CUDA language for codegen.
                Only valid when cuda == True.
        """
        if not (arg_types is not None and len(call_args) == len(arg_types)):
            raise AssertionError(
                "Mismatch call_args and arg_types in generate_kernel_call:\n"
                f"call_args: {call_args}\n"
                f"arg_types: {arg_types}"
            )

        # CPU Triton kernel: run the autotune block (populates
        # `CpuTritonKernelCache`) and emit `call_<k>(...)`; the body is
        # defined later by `DeferredCpuTritonCallWrapper` in `finalize_prefix`.
        if triton and kernel_name in self._cpu_triton_kernel_names:
            if (
                config.triton.autotune_at_compile_time
                and kernel_name not in self.kernel_autotune_names
            ):
                PythonWrapperCodegen._generate_kernel_call_helper(
                    self,
                    kernel_name,
                    call_args,
                    device=device,
                    triton=triton,
                    arg_types=arg_types,
                    raw_keys=raw_keys,
                    raw_args=raw_args,
                    triton_meta=triton_meta,
                    inductor_meta=inductor_meta,
                    original_fxnode_name=original_fxnode_name,
                )

            wrapper_name = f"call_{kernel_name}"
            if wrapper_name not in self._cpu_triton_call_wrappers:
                self._cpu_triton_call_wrappers[wrapper_name] = (
                    DeferredCpuTritonCallWrapper(
                        wrapper_name=wrapper_name,
                        kernel_name=kernel_name,
                        arg_types=list(arg_types),
                        triton_meta=triton_meta,
                        inductor_meta=inductor_meta,
                    )
                )

            stringified_args = [
                self._stringify_cpu_triton_call_arg(a) for a in call_args
            ]
            if V.graph.aot_mode:
                stringified_args.append("kernels")
                stringified_args.append("this->cubin_dir_")
            self.writeline(self.wrap_kernel_call(wrapper_name, stringified_args))
            return

        new_args = []
        for idx, arg in enumerate(call_args):
            if isinstance(arg_types[idx], str) and "*" in arg_types[idx]:
                new_args.append(f"({arg_types[idx]})({arg}.data_ptr())")
            else:
                # arg is a scalar - ensure it's a string for C++ codegen
                # With Triton support, arg might be a SymPy expression or other type
                new_args.append(str(arg) if not isinstance(arg, str) else arg)
        # debug printer related logic for cpp kernel type.
        debug_printer_manager = V.graph.wrapper_code.debug_printer
        debug_printer_manager.set_printer_args(
            call_args,
            kernel_name,
            None,
            None,
            "cpp",
        )
        with debug_printer_manager:
            self.writeline(self.wrap_kernel_call(kernel_name, new_args))

    def write_constant(self, name, hashed):
        # [FUNCTION DOC] Writes a comment line with the constant name and its hash into the header so the code cache
        # produces distinct files for distinct constants.
        # include a hash so our code cache gives different constants different files
        self.header.writeline(f"// {name} {hashed}")

    @staticmethod
    def get_device_include_path_jit(device: str) -> str:
        # [FUNCTION DOC] Returns the include line for the JIT (cpp_wrapper) header of the given device.
        return f"#include <torch/csrc/inductor/cpp_wrapper/{device}.h>"

    @staticmethod
    def get_device_include_path_aot(device: str) -> str:
        # [FUNCTION DOC] Returns the include line for the AOT (aoti_include) header of the given device.
        return f"#include <torch/csrc/inductor/aoti_include/{device}.h>"

    def add_device_include(self, device: str) -> None:
        # [FUNCTION DOC] Idempotently adds the device headers to the header buffer (JIT and AOT variants) and records
        # the device; also includes the C shim extension header for self.device when it exists in the installed torch.
        if device in self.included_devices:
            return

        self.included_devices.add(device)

        # Add the default header for this device, plus any C-shim extensions that are
        # present.
        self.header.splice_jit(self.get_device_include_path_jit(device))
        self.header.splice_aot(self.get_device_include_path_aot(device))
        extend_aoti_c_shim_include = (
            f"torch/csrc/inductor/aoti_torch/generated/extend/c_shim_{self.device}.h"
        )
        extend_aoti_c_shim_path = os.path.join(
            os.path.dirname(torch.__file__),
            "include",
            extend_aoti_c_shim_include,
        )
        if os.path.exists(extend_aoti_c_shim_path):
            self.header.splice(f"#include <{extend_aoti_c_shim_include}>")

    def _write_aoti_interface_header(self):
        # [FUNCTION DOC] HELPER: Adds the AOTI runtime interface to the AOT header: inlines interface.cpp under dynamic
        # linkage (reads that file), otherwise includes the per-model header; then adds a newline.
        """Emit the AOTI runtime interface include or inline."""
        if config.aot_inductor.dynamic_linkage:
            with open(
                os.path.join(os.path.dirname(__file__), "aoti_runtime", "interface.cpp")
            ) as f:
                self.header.splice_aot(f.read())
        else:
            # we produce a separate model header for each model in static linkage
            self.header.splice_aot(f"""#include \"{self.model_class_name_suffix}.h\"""")
        self.header.splice_aot("\n")

    def write_header(self):
        # [FUNCTION DOC] Writes the header: nothing for the constant graph. In JIT mode starts the Python source that
        # wraps the C++ in a raw string; includes headers for each non-meta device (or the common header if none);
        # adds the AOTI interface in AOT mode and the kernel-context thread-local when kernel profiling and context
        # guard are enabled.
        if V.graph.is_const_graph:
            # We do not write header for constant graph, it will be written by main module.
            return

        if not V.graph.aot_mode:
            self.header.splice('''
                import torch
                from torch._inductor.codecache import CppWrapperCodeCache

                cpp_wrapper_src = (
                r"""
                ''')

        for device in V.graph.device_types:
            if device != "meta":
                self.add_device_include(device)
        if not self.included_devices:
            # When only meta device is present, add_device_include is never
            # called so we need to pull in the common header explicitly.
            if V.graph.aot_mode:
                self.header.writeline(
                    "#include <torch/csrc/inductor/aoti_include/common.h>"
                )
            else:
                self.header.writeline(
                    "#include <torch/csrc/inductor/cpp_wrapper/common.h>"
                )

        if V.graph.aot_mode:
            self._write_aoti_interface_header()

        if config.cpp.enable_kernel_profile and config.cpp.enable_kernel_context_guard:
            self.header.splice(
                "#include <torch/csrc/inductor/aoti_runtime/kernel_context_tls.h>"
            )
            self.header.splice("""
                namespace torch::aot_inductor {
                thread_local KernelContext* tls_kernel_context = nullptr;
                }
                """)

    def include_extra_header(self, header: str):
        # [FUNCTION DOC] Includes the given header once (tracked in a set) into the header buffer.
        # This is needed for cpp to python dtype conversion
        if header in self._included_extra_headers:
            return
        self._included_extra_headers.add(header)
        self.header.splice(f"#include <{header}>")

    def mark_output_type(self):
        # [FUNCTION DOC] Records in output_is_tensor, per graph output index, whether it is a real tensor (false for
        # shape-as-constant outputs that must be unwrapped back to Python scalars).
        # mark output type to unwrap tensor back to python scalar
        from ..ir import ShapeAsConstantBuffer

        output_is_tensor = {}
        for idx, x in enumerate(V.graph.graph_outputs):
            if isinstance(x, ShapeAsConstantBuffer):
                output_is_tensor[idx] = False
            else:
                output_is_tensor[idx] = True

        self.output_is_tensor = output_is_tensor

    def write_prefix(self):
        # [FUNCTION DOC] Writes the prefix: nothing for the constant graph. Emits extern C declarations for configured
        # custom-op C shims, opens the AOT namespace, and generates input size and NaN asserts.
        if V.graph.is_const_graph:
            # We do not write prefix for constant graph, it will be written by main module.
            return
        if config.aot_inductor.custom_ops_to_c_shims:
            # custom_ops_to_c_shims contains declaration of custom ops with C shim.
            # TODO: this could be auto-generated from a passed-in custom op schema
            custom_c_shims = list(
                chain(*config.aot_inductor.custom_ops_to_c_shims.values())
            )
            declarations = "\n".join(
                [f"extern {textwrap.dedent(shim)};" for shim in custom_c_shims]
            )
            self.prefix.splice(f"""
                extern "C" {{
                    {declarations}
                }}
                """)
        self.prefix.writeline_aot("namespace torch::aot_inductor {")
        if not V.graph.is_const_graph:
            self.codegen_input_size_and_nan_asserts()

    def write_input_output_info(
        self,
        info_kind: str,
        idx: int,
        name: str,
    ):
        # [FUNCTION DOC] Writes one line assigning the given name to the idx-th entry of the info array (inputs_info_
        # or outputs_info_) in the prefix.
        self.prefix.writeline(f"""{info_kind}[{idx}].name = "{name}";""")

    def codegen_input_symbol_assignment(  # type: ignore[override]
        self,
        name: str,
        value: ir.TensorBox,
        bound_vars: OrderedSet[sympy.Symbol],
        deferred_symbol_assignments=None,
    ):
        # [FUNCTION DOC] Emits C++ locals binding symbolic shapes. For a sympy symbol input, declares it (double for
        # float symbols, else 64-bit int) from the argument name. For a TensorBox, binds symbols appearing in sizes and
        # strides from the size/stride arrays; compound expressions with one unbound symbol are solved for it and
        # multi-unknown ones are deferred via deferred_symbol_assignments when given. Torchbind, generator and opaque
        # inputs bind nothing. Updates bound_vars and unbacked_symbol_decls; raises AssertionError for unknown types.
        """Assign symbolic shapes to C++ locals, with replacement aliases."""
        code = self.prefix

        @functools.cache
        def sizeof(name):
        # [FUNCTION DOC] HELPER: Cached; declares the size array variable for a tensor name once and returns its name.
            self.codegen_input_size_var_decl(code, name)
            return f"{name}_size"

        @functools.cache
        def strideof(name):
        # [FUNCTION DOC] HELPER: Cached; declares the stride array variable for a tensor name once and returns its name.
            self.codegen_input_stride_var_decl(code, name)
            return f"{name}_stride"

        def codegen_symbol(
            sym_or_exp: sympy.Symbol | sympy.Expr,
            base_name: str,
            name_fn: Callable[[str], str],
            dim: int,
            deferred_symbol_assignments=None,
        ) -> bool:
        # [FUNCTION DOC] HELPER: Binds one size or stride entry. A plain unbound symbol is declared from the array
        # element; a compound expression with exactly one unbound symbol is solved for that symbol (via an extra size
        # variable). Returns True if code was emitted. With several unbound symbols it queues a retry callback.
            if isinstance(sym_or_exp, sympy.Symbol):
                if sym_or_exp in bound_vars:
                    return False
                code.writeline(f"int64_t {sym_or_exp} = {name_fn(base_name)}[{dim}];")
                bound_vars.add(sym_or_exp)
                if symbol_is_type(sym_or_exp, (SymT.UNBACKED_INT, SymT.UNBACKED_FLOAT)):
                    self.unbacked_symbol_decls.add(str(sym_or_exp))
                self.maybe_emit_replacement_aliases(sym_or_exp, bound_vars)
                return True
            elif isinstance(sym_or_exp, sympy.Expr):
                undefined_symbols = [
                    sym for sym in sym_or_exp.free_symbols if sym not in bound_vars
                ]
                if len(undefined_symbols) != 1:
                    # Skip if expression contains no symbols or if multiple
                    # symbols exists since we assume each base symbol is defined
                    # by other codegen_symbol calls.
                    if (
                        len(undefined_symbols) > 1
                        and deferred_symbol_assignments is not None
                    ):

                        def retry(deferred_symbol_assignments):
                        # [FUNCTION DOC] HELPER: Deferred callback that retries binding the same expression later.
                            return codegen_symbol(
                                sym_or_exp,
                                base_name,
                                name_fn,
                                dim,
                                deferred_symbol_assignments,
                            )

                        deferred_symbol_assignments.append(retry)
                    return False

                from torch.utils._sympy.solve import try_solve

                free_symbol = undefined_symbols.pop()
                base_name = name_fn(base_name)
                # Use a size symbol to solve the free symbol
                size_symbol = sympy.Symbol(f"{base_name}_{dim}", integer=True)
                solution = try_solve(sympy.Eq(sym_or_exp, size_symbol), free_symbol)
                if solution is not None:
                    code.writeline(f"int64_t {size_symbol} = {base_name}[{dim}];")
                    expr = _rewrite_symbol_solution_for_int_codegen(solution[1])
                    code.writeline(f"int64_t {free_symbol} = {cexpr(expr)};")
                    bound_vars.add(free_symbol)
                    if symbol_is_type(
                        free_symbol, (SymT.UNBACKED_INT, SymT.UNBACKED_FLOAT)
                    ):
                        self.unbacked_symbol_decls.add(str(free_symbol))
                    self.maybe_emit_replacement_aliases(free_symbol, bound_vars)
                    return True
                return False
            return False

        if isinstance(value, sympy.Expr):
            if not isinstance(value, sympy.Symbol) or value in bound_vars:
                return
            if symbol_is_type(value, (SymT.FLOAT, SymT.UNBACKED_FLOAT)):
                decl = "double"
            else:
                decl = "int64_t"
            code.writeline(f"{decl} {value} = {name};")
            bound_vars.add(value)
            if symbol_is_type(value, (SymT.UNBACKED_INT, SymT.UNBACKED_FLOAT)):
                self.unbacked_symbol_decls.add(str(value))
            self.maybe_emit_replacement_aliases(value, bound_vars)
        elif isinstance(value, ir.TensorBox):
            for dim, size in enumerate(value.get_size()):
                codegen_symbol(size, name, sizeof, dim, deferred_symbol_assignments)
            for dim, stride in enumerate(value.get_stride()):
                codegen_symbol(stride, name, strideof, dim, deferred_symbol_assignments)
        elif isinstance(
            value, (ir.TorchBindObject, ir.GeneratorState, ir.OpaqueObjectState)
        ):
            # Python-only inputs do not define C++ shape symbols.
            pass
        else:
            raise AssertionError(f"Unknown value type: {type(value)}")

    def maybe_emit_replacement_aliases(  # type: ignore[override]
        self,
        sym: sympy.Symbol,
        bound_vars: OrderedSet[sympy.Symbol],
    ) -> None:
        # [FUNCTION DOC] After a symbol is bound, emits aliases for symbols that shape_env replacements say are equal
        # to it (either direction), but only between backed symbols not yet bound. Updates bound_vars; no-op if there
        # is no shape env.
        sizevars = getattr(V.graph, "sizevars", None)
        shape_env = getattr(sizevars, "shape_env", None)
        if shape_env is None:
            return

        def is_backed_symbol(s: sympy.Symbol) -> bool:
        # [FUNCTION DOC] HELPER: Returns true if the symbol is not an unbacked int or float symbol.
            return not symbol_is_type(s, (SymT.UNBACKED_INT, SymT.UNBACKED_FLOAT))

        def decl_type(s: sympy.Symbol) -> str:
        # [FUNCTION DOC] HELPER: Returns the C++ type name for a symbol: double for float symbols, else 64-bit int.
            if symbol_is_type(s, (SymT.FLOAT, SymT.UNBACKED_FLOAT)):
                return "double"
            return "int64_t"

        for src, tgt in shape_env.replacements.items():
            if (
                tgt == sym
                and isinstance(src, sympy.Symbol)
                and src not in bound_vars
                and is_backed_symbol(src)
                and is_backed_symbol(sym)
            ):
                self.prefix.writeline(f"{decl_type(src)} {src} = {sym};")
                bound_vars.add(src)
            elif (
                src == sym
                and isinstance(tgt, sympy.Symbol)
                and tgt not in bound_vars
                and is_backed_symbol(sym)
                and is_backed_symbol(tgt)
            ):
                self.prefix.writeline(f"{decl_type(tgt)} {tgt} = {sym};")
                bound_vars.add(tgt)

    def bind_input_symbol(
        self,
        sym: sympy.Symbol,
        input_name: str,
        kind: Literal["size", "stride"],
        dim: int,
        bound_vars: OrderedSet[sympy.Symbol],
    ) -> None:
        # [FUNCTION DOC] Binds a symbol to the given dimension of an input's sizes or strides (per kind) unless it is
        # already bound, then emits replacement aliases. Mutates bound_vars.
        if sym in bound_vars:
            return
        accessor = "sizes" if kind == "size" else "strides"
        self.prefix.writeline(f"int64_t {sym} = {input_name}.{accessor}()[{dim}];")
        bound_vars.add(sym)
        self.maybe_emit_replacement_aliases(sym, bound_vars)

    def generate_input_output_runtime_checks(self):
        # [FUNCTION DOC] In debug mode, emits AOT-only C++ check functions (one per graph input) verifying dtype,
        # static sizes, symbolic size bounds (lower bound if configured, upper bound), static strides and device type
        # against compile-time sample tensors, throwing at runtime on mismatch; plus a combined checker gated by an
        # environment variable that calls them. Writes into prefix.
        """
        In debug_compile mode, we generate checks to ensure the dtype/shape/stride/device of each
        real input/output tensor match ones provided at compile time via sample
        input/output.
        """

        def gen_check(handle_kind, idx, name, tensor):
        # [FUNCTION DOC] HELPER: Emits the checks for one input tensor handle (dtype, each size, each static stride and,
        # for TensorBox, device type) into the prefix as AOT-only code.
            # Wrap AtenTensorHandle with ConstantHandle for cleaner utility function access
            self.prefix.writeline_aot(
                f"ConstantHandle {name} = ConstantHandle({handle_kind}[{idx}]);"
            )
            self.prefix.writeline_aot(f"int32_t {name}_dtype;")
            self.prefix.writeline_aot(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype({name}, &{name}_dtype));"
            )
            expected_dtype_name = DTYPE_TO_ATEN[tensor.dtype]
            dtype_str = str(tensor.dtype).split(".")[-1]
            self.prefix.splice_aot(f"""
                    int32_t {name}_expected_dtype = aoti_torch_dtype_{dtype_str}();
                    if ({name}_expected_dtype != {name}_dtype) {{
                        std::stringstream ss;
                        ss << "{handle_kind}[{idx}]: unmatched dtype, "
                           << "expected: " << {name}_expected_dtype << "({expected_dtype_name}), "
                           << "but got: " << {name}_dtype << "\\n";
                        throw std::runtime_error(std::move(ss).str());
                    }}
                """)
            self.prefix.writeline_aot(f"auto {name}_size = {name}.sizes();")
            for dim_idx, d in enumerate(tensor.get_size()):
                if isinstance(d, (int, sympy.Integer)):
                    self.prefix.splice_aot(f"""
                            if ({d} != {name}_size[{dim_idx}]) {{
                                std::stringstream ss;
                                ss << "{handle_kind}[{idx}]: unmatched dim value at {dim_idx}, "
                                   << "expected: {d}, " << "but got: " << {name}_size[{dim_idx}]
                                   << "\\n";
                                throw std::runtime_error(std::move(ss).str());
                            }}
                        """)
                else:
                    from torch.utils._sympy.value_ranges import bound_sympy

                    sym_range = bound_sympy(d, V.graph.sizevars.shape_env.var_to_range)
                    if config.aot_inductor.check_lowerbound and not math.isinf(
                        sym_range.lower
                    ):
                        self.prefix.splice_aot(f"""
                                if ({name}_size[{dim_idx}] < {sym_range.lower}) {{
                                    std::stringstream ss;
                                    ss << "{handle_kind}[{idx}]: dim value is too small at {dim_idx}, "
                                       << "expected it to be >= {sym_range.lower}, " << "but got: "
                                       << {name}_size[{dim_idx}] << "\\n";
                                    throw std::runtime_error(std::move(ss).str());
                                }}
                            """)
                    if not math.isinf(sym_range.upper):
                        # Limit upper bound to max C long long value (2^63 - 1)
                        max_long_long = ctypes.c_longlong(2**63 - 1).value
                        upper_bound = min(sym_range.upper, max_long_long)
                        self.prefix.splice_aot(f"""
                                if ({name}_size[{dim_idx}] > {upper_bound}) {{
                                    std::stringstream ss;
                                    ss << "{handle_kind}[{idx}]: dim value is too large at {dim_idx}, "
                                       << "expected to be <= {upper_bound}, " << "but got: "
                                       << {name}_size[{dim_idx}] << "\\n";
                                    throw std::runtime_error(std::move(ss).str());
                                }}
                            """)

            self.prefix.writeline_aot(f"auto {name}_stride = {name}.strides();")
            for stride_idx, s in enumerate(tensor.get_stride()):
                if not isinstance(s, (int, sympy.Integer)):
                    continue
                self.prefix.splice_aot(f"""
                        if ({s} != {name}_stride[{stride_idx}]) {{
                            std::stringstream ss;
                            ss << "{handle_kind}[{idx}]: unmatched stride value at {stride_idx}, "
                               << "expected: {s}, " << "but got: " << {name}_stride[{stride_idx}]
                               << "\\n";
                            throw std::runtime_error(std::move(ss).str());
                        }}
                    """)

            # check input device type
            if isinstance(tensor, ir.TensorBox):
                tensor_device = tensor.get_device()
                if tensor_device is not None:
                    expected_device_type = DEVICE_TO_INT.get(tensor_device.type)
                    if expected_device_type is not None:
                        self.prefix.writeline_aot(f"int32_t {name}_device_type;")
                        self.prefix.writeline_aot(
                            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type({name}, &{name}_device_type));"
                        )
                        device_type_str = tensor_device.type
                        self.prefix.splice_aot(f"""
                                int32_t {name}_expected_device_type = {expected_device_type};
                                if ({name}_expected_device_type != {name}_device_type) {{
                                    std::stringstream ss;
                                    ss << "{handle_kind}[{idx}]: unmatched device type, "
                                    << "expected: " << {name}_expected_device_type << "{expected_device_type}({device_type_str}), "
                                    << "but got: " << {name}_device_type << "\\n";
                                    throw std::runtime_error(std::move(ss).str());
                                }}
                            """)

        # Create a separate function for each input check to avoid "too big to optimize" error
        for idx, (name, tensor) in enumerate(V.graph.graph_inputs.items()):
            self.prefix.splice_aot(f"""
                AOTI_NOINLINE static void check_input_{idx}(
                    AtenTensorHandle* input_handles
                ) {{
                """)
            with self.prefix.indent():
                gen_check("input_handles", idx, name, tensor)
            self.prefix.writeline_aot("}")

        # force noinline to avoid any potential compilation slowdown due to aggressive
        # inline done by the host compiler
        self.prefix.splice_aot("""
            static bool _check_aoti_runtime_check_inputs_env() {
                const static char* env_var_value = getenv("AOTI_RUNTIME_CHECK_INPUTS");
                const static bool result = env_var_value != nullptr && env_var_value[0] != '0';
                return result;
            }

            AOTI_NOINLINE static void __check_inputs_outputs(
                AtenTensorHandle* input_handles,
                AtenTensorHandle* output_handles) {
                if (!_check_aoti_runtime_check_inputs_env()){
                    return;
                }
            """)
        with self.prefix.indent():
            for idx in range(len(V.graph.graph_inputs)):
                self.prefix.writeline_aot(f"check_input_{idx}(input_handles);")
        self.prefix.writeline_aot("}")

    def _write_aoti_entry_point_signature(self):
        # [FUNCTION DOC] HELPER: Emits the AOTI entry point. Includes the const graph's wrapper code (header, prefix,
        # kernel declarations) when present; for the const graph writes the _const_run_impl header, else an empty
        # _const_run_impl if runtime constant folding is off, the input check functions and the run_impl prototype.
        # Raises AssertionError if const wrapper or kernel code is missing.
        """Emit the AOTI entry point: run_impl method and _const_run_impl."""
        self.codegen_additional_funcs()

        if V.graph.const_module:
            # In dual-wrapper-mode, the const graph's JIT output is emitted separately,
            # while its AOTI body is spliced below into the main AOTI source.
            # Keep the const graph header out of the main output to avoid
            # duplicate standalone header content.
            if not V.graph.is_dual_wrapper_mode:
                self.header.splice(V.graph.const_module.wrapper_code.header)
            if V.graph.const_wrapper_code is None:
                raise AssertionError("expected const_wrapper_code to be set")
            self.prefix.splice_aot(V.graph.const_wrapper_code)
            if V.graph.const_kernel_code is None:
                raise AssertionError("expected const_kernel_code to be set")
            self.kernel_declarations.splice(V.graph.const_kernel_code)

        if V.graph.is_const_graph:
            self.prefix.splice_aot(f"""
                void {self.aoti_model_class_name}::_const_run_impl(
                    std::vector<AtenTensorHandle>& output_handles,
                    DeviceStreamType stream,
                    AOTIProxyExecutorHandle proxy_executor
                ) {{
                """)
        else:
            if not config.aot_inductor.use_runtime_constant_folding:
                # If we do not split the constant graph, we'll just create
                # an empty implementation when wrapping the main module.
                self.prefix.splice_aot(f"""
                    void {self.aoti_model_class_name}::_const_run_impl(
                        std::vector<AtenTensorHandle>& output_handles,
                        DeviceStreamType stream,
                        AOTIProxyExecutorHandle proxy_executor
                    ) {{}}

                    """)

            run_impl_proto = f"""
                void {self.aoti_model_class_name}::run_impl(
                    AtenTensorHandle*
                        input_handles, // array of input AtenTensorHandle; handles
                                        // are stolen; the array itself is borrowed
                    AtenTensorHandle*
                        output_handles, // array for writing output AtenTensorHandle; handles
                                        // will be stolen by the caller; the array itself is
                                        // borrowed
                    DeviceStreamType stream,
                    AOTIProxyExecutorHandle proxy_executor
                ) {{
                    __check_inputs_outputs(input_handles, output_handles);
                """

            self.generate_input_output_runtime_checks()
            self.prefix.splice_aot(run_impl_proto)

    def _write_jit_entry_point_signature(self):
        # [FUNCTION DOC] HELPER: Emits the JIT entry function signature (inductor_entry_impl taking input and output
        # handle arrays) into the JIT side of the prefix.
        """Emit the JIT entry point: inductor_entry_impl free function."""
        self.prefix.splice_jit("""
            void inductor_entry_impl(
                AtenTensorHandle*
                    input_handles, // array of input AtenTensorHandle; handles
                                    // are stolen; the array itself is borrowed
                AtenTensorHandle*
                    output_handles  // array for writing output AtenTensorHandle; handles
                                    // will be stolen by the caller; the array itself is
                                    // borrowed)
            ) {
            """)

    def _write_entry_point_signature(self):
        # [FUNCTION DOC] HELPER: Emits the AOT entry signature when in AOT mode and always the JIT entry signature.
        """Emit the entry point function/method signature.

        AOTI: AOTInductorModel::run_impl with stream and proxy_executor params.
        JIT: inductor_entry_impl free function.
        Dual-wrapper-mode: both signatures, to their respective buffers.
        """
        if V.graph.aot_mode:
            self._write_aoti_entry_point_signature()
        self._write_jit_entry_point_signature()

    def _write_input_preamble(self):
        # [FUNCTION DOC] HELPER: Emits input handle stealing. AOT (non-const graph): steals graph-input handles; JIT:
        # releases the GIL and steals handles for graph inputs plus constants.
        """Emit input stealing and (in JIT mode) GIL release."""
        # Weights are promoted in the JIT mode
        jit_num_args = len(V.graph.graph_inputs) + len(V.graph.constants)
        # AOTI const graphs enter through _const_run_impl and have no
        # input_handles. Their JIT entry point still receives constants as
        # appended input handles.
        if not V.graph.is_const_graph:
            aoti_num_args = len(V.graph.graph_inputs)
            self.prefix.splice_aot(
                f"auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, {aoti_num_args});"
            )
        # release GIL to support multiple instances inference (in different threads of the same process)
        self.prefix.splice_jit("py::gil_scoped_release_simple release;")
        self.prefix.splice_jit(
            f"auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, {jit_num_args});"
        )

    def _write_input_unpacking(self):
        # [FUNCTION DOC] HELPER: Emits code that binds each graph input name to its handle (moved from inputs); scalar
        # sympy inputs are unwrapped from tensors with an item call. Also emits debug-print code for inputs whose
        # names start with arg. Raises AssertionError if a scalar input dtype cannot be determined.
        """Emit code to unpack graph inputs from the handles array."""
        inputs_len = len(V.graph.graph_inputs.keys())
        if inputs_len == 0:
            return
        for idx, input_key in enumerate(V.graph.graph_inputs.keys()):
            # unwrap input tensor back to scalar
            if isinstance(V.graph.graph_inputs[input_key], sympy.Expr):
                from ..graph import may_get_constant_buffer_dtype

                dtype = may_get_constant_buffer_dtype(
                    V.graph.graph_inputs[input_key]  # type: ignore[arg-type]
                )
                if dtype is None:
                    raise AssertionError("Fails to get the dtype of the sympy.Expr")
                self.codegen_tensor_item(
                    dtype, f"inputs[{idx}]", input_key, self.prefix
                )
            else:
                self.prefix.writeline(f"auto {input_key} = std::move(inputs[{idx}]);")
        # debug printing for all input args to AOTI model
        debug_printer_manager = V.graph.wrapper_code.debug_printer
        debug_printer_manager.codegen_model_inputs_value_print(
            input_args_to_print=[
                input_key
                for input_key in V.graph.graph_inputs
                if input_key.startswith("arg")
            ]
        )

    def _write_constants_unpacking(self):
        # [FUNCTION DOC] HELPER: Emits code binding each constant name: AOT references the constants array entry, JIT
        # moves it from the extra appended inputs. Raises AssertionError if any constant is not a tensor.
        """Emit code to unpack constants — from constants_ (AOTI) or inputs (JIT)."""
        inputs_len = len(V.graph.graph_inputs.keys())
        if not all(
            isinstance(v, torch.Tensor) for v in list(V.graph.constants.values())
        ):
            raise AssertionError("Expect all constants to be Tensor")
        for idx, constants_key in enumerate(V.graph.constants.keys()):
            # AOTI: weights live in constants_ owned by ConstantHandle; never std::move.
            # JIT: weights are appended as inputs to the graph.
            constants_idx = inputs_len + idx
            self.prefix.writeline_aot(
                f"[[maybe_unused]] auto& {constants_key} = constants_->at({idx});"
            )
            self.prefix.writeline_jit(
                f"[[maybe_unused]] auto {constants_key} = std::move(inputs[{constants_idx}]);"
            )

    def _write_entry_point_postamble(self):
        # [FUNCTION DOC] HELPER: AOT-only emission after unpacking: clears inputs (non-const graph) and defines a
        # reference to the typed kernels object.
        """Emit AOTI-specific post-input setup (inputs.clear, kernels ref)."""
        if not V.graph.is_const_graph:
            self.prefix.writeline_aot("inputs.clear();")
        self.prefix.writeline_aot(
            "[[maybe_unused]] auto& kernels = static_cast<AOTInductorModelKernels&>(*this->kernels_.get());"
        )

    def write_wrapper_decl(self):
        # [FUNCTION DOC] Writes the entry function declaration and, inside its indentation, the prologue, input
        # preamble, input and constant unpacking, input codegen and postamble.
        self._write_entry_point_signature()
        with self.prefix.indent():
            self._codegen_entry_impl_prologue()
            self._write_input_preamble()
            self._write_input_unpacking()
            self._write_constants_unpacking()
            self.codegen_inputs()
            self._write_entry_point_postamble()

    def codegen_tensor_dtype_var_decl(self, code: IndentedBuffer, name):
        # [FUNCTION DOC] Emits a declaration of the dtype variable for the named tensor, fetched via the shim.
        code.writeline(f"int32_t {name}_dtype;")
        code.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype({name}, &{name}_dtype));"
        )

    def codegen_input_size_var_decl(self, code: IndentedBuffer, name):
        # [FUNCTION DOC] Emits a declaration of the size array variable for the named tensor.
        code.writeline(f"auto {name}_size = {name}.sizes();")

    def codegen_input_stride_var_decl(self, code: IndentedBuffer, name):
        # [FUNCTION DOC] Emits a declaration of the stride array variable for the named tensor.
        code.writeline(f"auto {name}_stride = {name}.strides();")

    def codegen_input_device_type_var_decl(self, code: IndentedBuffer, name):
        # [FUNCTION DOC] Emits a declaration of the device type variable for the named tensor, fetched via the shim.
        code.writeline(f"int32_t {name}_device_type;")
        code.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type({name}, &{name}_device_type));"
        )

    def codegen_additional_funcs(self):
        # [FUNCTION DOC] Hook for subclasses to emit extra functions; does nothing here.
        pass

    def codegen_initialized_kernel_decls(self):
        # [FUNCTION DOC] Emits extern C declarations (non-mangled linkage) for each initialized kernel's signature.
        # Raises AssertionError if a kernel lacks a signature method.
        # Tell compiler we need to link with the non-mangled symbols
        for kernel in self.initialized_kernels.values():
            if not hasattr(kernel, "get_signature"):
                raise AssertionError(f"{kernel} must have get_signature implemented")
            signature = kernel.get_signature()
            self.prefix.writeline(f'extern "C" {signature};')

    def codegen_model_kernels(self):
        # [FUNCTION DOC] Emits the AOTInductorModelKernels class: declarations of GEMM kernel signatures, a null-initialized
        # member per defined kernel (user-defined, const-graph and source kernels, minus initialized ones), a kernel and
        # launcher member pair for CPU Triton kernels, and pointer members for initialized kernels. If binary embedding
        # is enabled also declares embedded start (and, with XPU, end) symbols. Raises AssertionError on missing signature.
        self.prefix.writeline("namespace {")
        self.codegen_initialized_kernel_decls()
        self.prefix.writeline(
            "class AOTInductorModelKernels : public AOTInductorModelKernelsBase {"
        )
        self.prefix.writeline("  public:")
        declare_kernel = OrderedSet(self.src_to_kernel.values()) - OrderedSet(
            self.initialized_kernels.keys()
        )
        declare_kernel.update(
            entry[0] for entry in self.user_defined_kernel_cache.values()
        )
        if V.graph.const_module:
            declare_kernel.update(
                V.graph.const_module.wrapper_code.src_to_kernel.values()
            )
        # CPU Triton kernels need a (kernel_fn, launcher_fn) pair, declared
        # separately below; skip them in the default single-`void*` loop.
        cpu_triton_names = self._cpu_triton_kernel_names
        for kernel in sorted(declare_kernel):
            if kernel in cpu_triton_names:
                continue
            self.prefix.writeline(
                maybe_hipify_code_wrapper(
                    f"    {self.device_codegen.cpp_kernel_type()} {kernel}{{nullptr}};"
                )
            )
        for kernel in sorted(cpu_triton_names):
            self.prefix.writeline(f"    void* {kernel}_kernel_fn{{nullptr}};")
            self.prefix.writeline(
                f"    torch::aot_inductor::CpuTritonLauncherFn "
                f"{kernel}_launcher_fn{{nullptr}};"
            )
        for name, kernel in self.initialized_kernels.items():
            if not hasattr(kernel, "get_signature"):
                raise AssertionError(f"{kernel} must have get_signature implemented")
            kernel_ptr = f"(*{name})"
            signature = kernel.get_signature().replace(name, kernel_ptr)
            self.prefix.writeline(f"    {signature} = torch::aot_inductor::{name};")
        self.prefix.writeline("};")
        self.prefix.writeline("}  // namespace\n\n")

        if config.aot_inductor.embed_kernel_binary:
            self.prefix.writeline('extern "C" {')
            for name in sorted(declare_kernel):
                self.prefix.writeline(
                    f"    extern const unsigned char __{name}_start[];"
                )
                if torch.xpu.is_available():
                    self.prefix.writeline(
                        f"    extern const unsigned char __{name}_end[];"
                    )
            self.prefix.writeline("}")

    # MSVC string was longer than the limit of 16380 single-byte characters.
    # https://learn.microsoft.com/en-us/cpp/error-messages/compiler-errors-1/compiler-error-c2026
    MSVC_C2026_MAX_STRING_LENGTH = 16000

    def codegen_write_arg_with_large_length_string(
        self,
        arg_name: str,
        arg_str_val: str,
        max_truncate_length: int = MSVC_C2026_MAX_STRING_LENGTH,
    ):
        # [FUNCTION DOC] Writes an assignment of a string to the named variable as a C++ raw string literal; strings
        # longer than max_truncate_length are split into adjacent chunks to stay under the MSVC literal size limit.
        def truncate_string(s: str, length: int) -> list[str]:
            # [FUNCTION DOC] HELPER: Splits a string into consecutive chunks of at most the given length.
            return [s[i : i + length] for i in range(0, len(s), length)]

        if len(arg_str_val) > max_truncate_length:
            truncated_strs = truncate_string(arg_str_val, max_truncate_length)
            self.prefix.writeline(f"{arg_name} =")
            for truncate_str in truncated_strs:
                self.prefix.writeline(f'R"({truncate_str})"')
            self.prefix.writeline(";")
        else:
            self.prefix.writeline(f'{arg_name} = R"({arg_str_val})";')

    def codegen_model_constructor(self):
        # [FUNCTION DOC] Emits the AOTInductorModel constructor: base-class init with input, output and constant counts
        # and whether weights are included; info entries for each input name; for each constant its name, dtype, device
        # type, offset, aligned data size, folded flag, constant type (folded/tensor constant/parameter/buffer),
        # shape, stride, layout, optional mkldnn metadata and original FQN; map/array updates; in/out specs; output
        # names; and kernels object creation. Raises AssertionError for symbolic inputs/outputs, non-tensor
        # constants, unsupported mixed devices or a missing original FQN.
        # KNOWN ISSUE: the symbolic-output error message formats the name variable left over from earlier loops.
        """
        // Generated code example
        AOTInductorModel::AOTInductorModel()
            : AOTInductorModelBase(4, 1) {
        inputs_info_[0].name = "input0";
        inputs_info_[0].dtype = "torch.float16";
        ...
        constants_info_[0].name = "L__self___weight";
        constants_info_[0].dtype = at::kFloat;
        constants_info_[0].offset = 0;
        constants_info_[0].data_size = 8192;
        constants_info_[0].shape = {64, 32};
        constants_info_[0].stride = {32, 1};
        ...
        outputs_info_[0].name = "output0";
        outputs_info_[0].dtype = "torch.float16";
        }
        """

        num_inputs = len(V.graph.graph_inputs)
        num_outputs = len(V.graph.graph_outputs)
        num_constants = len(V.graph.constants)
        include_weights = (
            "true"
            if config.aot_inductor.package_constants_in_so
            and config.aot_inductor.package_constants_on_disk_format != "binary_blob"
            else "false"
        )
        self.prefix.splice(f"""
            {self.aoti_model_class_name}::{self.aoti_model_class_name}(std::shared_ptr<ConstantMap> constants_map,
                                               std::shared_ptr<std::vector<ConstantHandle>> constants_array,
                                               const std::string& device_str,
                                               std::optional<std::string> cubin_dir)
                : AOTInductorModelBase({num_inputs},
                                       {num_outputs},
                                       {num_constants},
                                       device_str,
                                       std::move(cubin_dir),
                                       {include_weights}) {{
            """)

        with self.prefix.indent():
            for idx, (name, inp) in enumerate(V.graph.graph_inputs.items()):
                if isinstance(inp, sympy.Expr):
                    raise AssertionError(f"input {name=} cannot be symbolic")
                self.write_input_output_info("inputs_info_", idx, name)

            all_cuda = all(
                V.graph.get_original_value_of_constant(name).is_cuda
                for name in V.graph.constants
                if name not in V.graph.folded_constants
            )
            for idx, name in enumerate(V.graph.constants.keys()):
                tensor = V.graph.get_original_value_of_constant(name)
                if not isinstance(tensor, torch.Tensor):
                    raise AssertionError(
                        f"expected constant {name} to be a Tensor, got {type(tensor)}"
                    )
                self.prefix.writeline(f"""constants_info_[{idx}].name = "{name}";""")
                self.prefix.writeline(
                    f"constants_info_[{idx}].dtype = {self.codegen_dtype(tensor.dtype)};"
                )
                # Mixed-device constants are only supported when the secondary device is CPU
                if tensor.device.type != self.device and tensor.device.type != "cpu":
                    raise AssertionError(
                        f"Mixed-device constants are only supported when the secondary "
                        f"device is CPU. Model device is '{self.device}', but constant "
                        f"'{name}' is on device '{tensor.device}'."
                    )
                # device_index is not needed because it can be set at runtime
                device_type, _ = self.codegen_device(tensor.device).split(", ")
                self.prefix.writeline(
                    f"constants_info_[{idx}].device_type = {device_type};"
                )
                self.prefix.writeline(
                    f"constants_info_[{idx}].offset = {tensor.storage_offset()};"
                )

                # If constants to serialize contain cpu tensors, we always align data_size it to 64.
                # When loading the constants, the valid data will depends on the size
                # not the data_size so there won't be correctness issue.
                data_size = (
                    torch.ops.mkldnn._nbytes(tensor)
                    if tensor.is_mkldnn
                    else tensor.untyped_storage().nbytes()
                )
                self.prefix.writeline(
                    f"constants_info_[{idx}].data_size = {data_size if all_cuda else _align(data_size)};"
                )

                from_folded = "true" if name in V.graph.folded_constants else "false"
                self.prefix.writeline(
                    f"constants_info_[{idx}].from_folded = {from_folded};"
                )

                if name in V.graph.folded_constants:
                    constant_type_str = "FoldedConstant"
                elif name.startswith("_tensor_constant"):
                    constant_type_str = "TensorConstant"
                elif any(
                    name == normalize_name(parameter_name)
                    for parameter_name in V.graph.named_parameters
                ):
                    constant_type_str = "Parameter"
                elif any(
                    name == normalize_name(buffer_name)
                    for buffer_name in V.graph.named_buffers
                ):
                    constant_type_str = "Buffer"
                else:
                    constant_type_str = "Unknown"
                self.prefix.writeline(
                    f"constants_info_[{idx}].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::{constant_type_str});"
                )

                size_str = ", ".join([str(s) for s in tensor.size()])
                self.prefix.writeline(f"constants_info_[{idx}].shape = {{{size_str}}};")

                stride_str = ", ".join([str(s) for s in tensor.stride()])
                self.prefix.writeline(
                    f"constants_info_[{idx}].stride = {{{stride_str}}};"
                )
                self.prefix.writeline(
                    f"constants_info_[{idx}].layout = static_cast<int32_t>({self.codegen_layout(tensor.layout)});"
                )

                if tensor.is_mkldnn:
                    opaque_metadata_tensor = torch.ops.mkldnn._get_mkldnn_serialized_md(
                        tensor
                    )
                    if opaque_metadata_tensor.dim() != 1:
                        raise AssertionError("Expect opaque_metadata_tensor to be 1-D")

                    opaque_metadata_list = opaque_metadata_tensor.tolist()
                    opaque_metadata_str = self.codegen_shape_tuple(opaque_metadata_list)
                    self.prefix.writeline(
                        f"constants_info_[{idx}].opaque_metadata = {opaque_metadata_str};"
                    )
                if name in V.graph.dynamo_flat_name_to_original_fqn:
                    original_fqn = V.graph.dynamo_flat_name_to_original_fqn.get(
                        name, name
                    )
                elif name in V.graph.allocated_constant_name:
                    original_fqn = V.graph.allocated_constant_name[name]
                else:
                    raise AssertionError("original_fqn must be set for constant")
                self.prefix.writeline(
                    f"""constants_info_[{idx}].original_fqn = "{original_fqn}";"""
                )
            self.prefix.writeline("update_constants_map(std::move(constants_map));")
            self.prefix.writeline("update_constants_array(std::move(constants_array));")

            def escape_string(x):
            # [FUNCTION DOC] HELPER: Escapes backslashes, double quotes, newlines and tabs in a string. Currently unused.
                return (
                    x.replace("\\", "\\\\")
                    .replace('"', '\\"')
                    .replace("\n", "\\n")
                    .replace("\t", "\\t")
                )

            # Origin code: self.prefix.writeline(f'in_spec_ = R"({config.aot_inductor.serialized_in_spec})";')
            # Fix msvc C2026 error via codegen_write_arg_with_large_length_string
            self.codegen_write_arg_with_large_length_string(
                arg_name="in_spec_", arg_str_val=config.aot_inductor.serialized_in_spec
            )
            # Origin code: self.prefix.writeline(f'out_spec_ = R"({config.aot_inductor.serialized_out_spec})";')
            # Fix msvc C2026 error via codegen_write_arg_with_large_length_string
            self.codegen_write_arg_with_large_length_string(
                arg_name="out_spec_",
                arg_str_val=config.aot_inductor.serialized_out_spec,
            )

            for idx, output in enumerate(V.graph.graph_outputs):
                if isinstance(output, sympy.Expr):
                    raise AssertionError(f"output {name=} cannot be symbolic")
                name = f"output{idx}"
                self.write_input_output_info("outputs_info_", idx, name)

            self.prefix.writeline(
                "this->kernels_ = std::make_unique<AOTInductorModelKernels>();"
            )

        self.prefix.writeline("}")

    def codegen_const_run_driver(self):
        # [FUNCTION DOC] Emits the const_run_impl driver. Without runtime constant folding it warns (unless called for
        # initialization) and returns an empty map. Otherwise maps each folded constant to its output index of the
        # constant-folding graph, seeds output handles from constants, calls _const_run_impl and fills the folded
        # constants map. Raises AssertionError if some constant is not mapped.
        """
        // Generated code example
        std::unordered_map<std::string, AtenTensorHandle> AOTInductorModel::const_run_impl(
            DeviceStreamType stream,
            AOTIProxyExecutorHandle proxy_executor,
            bool initialization
        ) {
            std::unordered_map<std::string, AtenTensorHandle> folded_constants_map;
            std::vector<AtenTensorHandle> output_handles;
            // build up output_handles over here.
            _const_run_impl(output_handles, stream, proxy_executor);
            // build up folded_constants_map
            return folded_constants_map;
        }
        """

        self.prefix.splice(f"""
            std::unordered_map<std::string, AtenTensorHandle> {self.aoti_model_class_name}::const_run_impl(
                DeviceStreamType stream,
                AOTIProxyExecutorHandle proxy_executor,
                bool initialization
            ) {{
            """)
        if not config.aot_inductor.use_runtime_constant_folding:
            self.prefix.splice("""
                    if (!initialization) {
                        std::cerr << "[WARNING] Calling constant_folding in model, but compiled with config: "
                                  << "aot_inductor.use_runtime_constant_folding=False\\n";
                    }
                    return {};
                }
                """)
            return

        with self.prefix.indent():
            # This is a mapping to the index of constant folding graph's output
            const_index_mapping: list[tuple[int, str] | None] = [None] * len(
                V.graph.const_output_index
            )
            for idx, (name, _) in enumerate(V.graph.constants.items()):
                if name in V.graph.const_output_index:
                    const_index_mapping[V.graph.const_output_index[name]] = (idx, name)  # type: ignore[call-overload]
            if None in const_index_mapping:
                raise AssertionError(
                    "Not all constant gets mapped for constant folding graph."
                )

            self.prefix.writeline(f"""
                std::unordered_map<std::string, AtenTensorHandle> folded_constants_map;
                folded_constants_map.reserve({len(const_index_mapping)});
                std::vector<AtenTensorHandle> output_handles({len(const_index_mapping)});
                """)

            self.prefix.splice("""
                // The below assignment of output_handles to constants is not used directly.
                // It's only used to memo the correspondence of handle and constants.
                """)

            for output_idx, (const_idx, _) in enumerate(const_index_mapping):  # type: ignore[misc]
                self.prefix.writeline(
                    f"output_handles[{output_idx}] = constants_->at({const_idx});"
                )

            self.prefix.writeline(
                "_const_run_impl(output_handles, stream, proxy_executor);"
            )

            for output_idx, (_, const_name) in enumerate(const_index_mapping):  # type: ignore[misc]
                self.prefix.writeline(
                    f'folded_constants_map["{const_name}"] = output_handles[{output_idx}];'
                )
            self.prefix.writeline("return folded_constants_map;")

        self.prefix.writeline("}")

    def generate(self, is_inference):
        # [FUNCTION DOC] Generates the full wrapper code (timed): writes the entry declaration, then defers to the base
        # class generate and returns its result.
        with dynamo_timed("CppWrapperCpu.generate", log_pt2_compile_event=True):
            self.write_wrapper_decl()
            return super().generate(is_inference)

    def _codegen_aoti_model_class(self):
        # [FUNCTION DOC] HELPER: Emits the AOTI namespace with the model kernels class, constructor and const_run driver
        # into the prefix, followed by a using directive.
        """Emit AOTI model class, constructor, and const_run driver into self.prefix."""
        self.prefix.writeline("namespace torch::aot_inductor {")
        self.codegen_model_kernels()
        self.codegen_model_constructor()
        self.codegen_const_run_driver()
        self.prefix.writeline("} // namespace torch::aot_inductor")
        self.prefix.writeline("using namespace torch::aot_inductor;")

    def finalize_prefix(self):
        # [FUNCTION DOC] Reorders the prefix at the end of codegen: builds the AOTI model class in a temporary buffer
        # (AOT non-const graph) or emits extern decls for initialized kernels (JIT); then rebuilds the prefix with cache
        # macros for used dtypes/devices/layouts/memory formats, CPU Triton runtime includes and call wrappers, the AOT
        # declarations, and finally the previous prefix content.
        prior = self.prefix

        # Build AOTI model class into a temporary buffer (AOTI-only content).
        # _codegen_aoti_model_class calls methods that write to self.prefix,
        # so redirect self.prefix to the temporary buffer during the call.
        aot_mode_decls = IndentedBuffer()
        if V.graph.aot_mode and not V.graph.is_const_graph:
            with self._target_buf("prefix", aot_mode_decls):
                self._codegen_aoti_model_class()
        elif not V.graph.is_const_graph and self.initialized_kernels:
            # JIT cpp_wrapper: emit extern "C" decls for kernels (CUTLASS,
            # ROCm CK) whose precompiled .so files are linked into the
            # wrapper via extra_flags.
            self.codegen_initialized_kernel_decls()

        # Reset self.prefix and write shared cache decls + CPU triton content.
        self.prefix = make_codegen_buffer()
        for dtype in self.used_cached_dtypes:
            self.prefix.writeline(f"CACHE_TORCH_DTYPE({dtype});")
        for device in self.used_cached_devices:
            self.prefix.writeline(f"CACHE_TORCH_DEVICE({device});")
        for layout in self.used_cached_layouts:
            self.prefix.writeline(f"CACHE_TORCH_LAYOUT({layout});")
        for memory_format in self.used_cached_memory_formats:
            self.prefix.writeline(f"CACHE_TORCH_MEMORY_FORMAT({memory_format});")

        # Includes are needed whenever a kernel was *defined* (the model
        # kernels class references `CpuTritonLauncherFn`); `call_<k>` bodies
        # only for kernels that were actually *called*.
        if self._cpu_triton_kernel_names:
            self._write_cpu_triton_runtime_includes(self.prefix)
        if self._cpu_triton_call_wrappers:
            for wrapper in self._cpu_triton_call_wrappers.values():
                self.prefix.writeline("\n")
                wrapper.generate(self)

        # aot_mode_decls is AOTI-only; splice_aot routes correctly across
        # all three modes (no-op on plain IndentedBuffer in pure-JIT).
        self.prefix.splice_aot(aot_mode_decls)
        self.prefix.splice(prior)

    @staticmethod
    def _write_cpu_triton_runtime_includes(prefix: IndentedBuffer) -> None:
        # [FUNCTION DOC] HELPER: Writes the one-time comment, Windows-unsupported guard, and includes needed by CPU
        # Triton wrapper code.
        """One-time includes/guards needed by emitted CPU Triton wrappers."""
        prefix.writeline("// CPU AOTI Triton kernel wrappers")
        prefix.writeline("#ifdef _WIN32")
        prefix.writeline(
            '#error "CPU AOTI Triton kernels are not supported on Windows"'
        )
        prefix.writeline("#endif")
        prefix.writeline("#include <mutex>")
        prefix.writeline(
            "#include <torch/csrc/inductor/aoti_runtime/cpu_triton_runtime_wrappers.h>"
        )

    @staticmethod
    def _stringify_cpu_triton_call_arg(arg: Any) -> str:
        # [FUNCTION DOC] HELPER: Renders a Triton call argument as C++: strings as is, booleans as true/false, numbers
        # and symbolic call args via string conversion, other expressions simplified and converted with cexpr.
        """Render a Triton kernel call argument as a C++ expression."""
        if isinstance(arg, str):
            return arg
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, float, SymbolicCallArg)):
            return str(arg)
        return cexpr(V.graph.sizevars.simplify(arg))

    def _define_kernel_helper(
        self,
        kernel_name: str,
        kernel_body: str,
        metadata: str | None = None,
        gpu: bool = False,
        cpp_definition: str | None = None,
    ):
        # [FUNCTION DOC] HELPER: Defines a kernel. When gpu (really: is a Triton kernel) is set, records the name as a CPU
        # Triton kernel and, if compile-time autotune is on, lets the Python base class emit its definition. Otherwise
        # splices the C++ definition into the header (and the body into kernel declarations when a separate cpp
        # definition is given, also requiring vector ISA), else splices the body into the header.
        # Misnomer: `gpu` actually means "is this a Triton kernel?"
        if gpu:
            # Register as a CPU Triton kernel and let the autotune block
            # compile it; the C++ wrapper is emitted later in finalize_prefix.
            self._cpu_triton_kernel_names.add(kernel_name)
            if config.triton.autotune_at_compile_time:
                PythonWrapperCodegen._define_kernel_helper(
                    self, kernel_name, kernel_body, metadata, gpu, cpp_definition
                )
            return
        if cpp_definition is not None:
            self.needs_vec_isa = True
            self.header.splice(cpp_definition)
            self.kernel_declarations.splice(f"\n{kernel_body}\n")
        else:
            self.header.splice(f"\n{kernel_body}\n")

    def codegen_scalar_to_tensor(self, output: str):
        # [FUNCTION DOC] Emits a conversion of a scalar C++ value to a tensor handle in the wrapper call, returning the
        # new variable name.
        name = f"scalar_to_tensor_{next(self.scalar_to_tensor_id)}"
        self.wrapper_call.writeline(
            f"RAIIAtenTensorHandle {name} = scalar_to_tensor_handle({output});"
        )
        return name

    def codegen_tensor_item(
        self, dtype: torch.dtype, tensor: str, scalar: str, indented_buffer=None
    ):
        # [FUNCTION DOC] Emits C++ that reads the scalar item of a tensor into a variable of the given dtype (half and
        # bfloat16 go through a temporary and are widened to float). Writes to the given buffer or to self.
        dtype_str = str(dtype).split(".")[-1]
        writer = indented_buffer or self

        if dtype == torch.float16 or dtype == torch.bfloat16:
            scalar_tmp = f"{scalar}_tmp"
            writer.writeline(f"{DTYPE_TO_CPP[dtype]} {scalar_tmp};")
            writer.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_{dtype_str}({tensor}, &{scalar_tmp}));"
            )
            writer.writeline(f"float {scalar} = float({scalar_tmp});")
        else:
            writer.writeline(f"{DTYPE_TO_CPP[dtype]} {scalar};")
            writer.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_{dtype_str}({tensor}, &{scalar}));"
            )

    def generate_return(self, output_refs: list[str]):
        # [FUNCTION DOC] Emits the output assignments for the entry function. Rvalue outputs are first materialized as
        # handles. Scalar (shape) outputs are wrapped into tensors; constant outputs are cloned; repeated outputs alias
        # the first output slot; others release their handle. Null outputs are skipped.
        # KNOWN ISSUE: a comment refers to a NOTE(return_constant) that is not present in this file.
        # KNOWN ISSUE: the output2idx check also records constant/scalar skipped paths inconsistently (those continue first).
        cst_names = V.graph.constants.keys()
        output2idx: dict[str, int] = {}

        # If any output ref represents an rvalue tensor, materialize it to an lvalue
        # RAIIAtenTensorHandle first.  This prevents situations where the code for the
        # rvalue tensor references tensor handles whose contents are modified below.
        output_refs = [
            self.create_tmp_raii_handle_var_if_needed(o, self.wrapper_call)
            for o in output_refs
        ]

        for idx, output in enumerate(output_refs):
            if output == "nullptr":
                continue

            is_constant_buffer = output in cst_names
            output_buffer = V.graph.graph_outputs[idx]
            if isinstance(output_buffer, ir.BaseView):
                output_storage = output_buffer.unwrap_view()
                if not isinstance(output_storage, (ir.BaseView, ir.MutableBox)):
                    raise AssertionError(
                        f"expected output_storage to be BaseView or MutableBox, "
                        f"got {type(output_storage)}"
                    )
                if isinstance(output_storage.data, ir.ConstantBuffer):
                    is_constant_buffer = True

            if isinstance(output_buffer, ir.ShapeAsConstantBuffer):
                # Need to wrap scalar into tensor as the main function returns a vector of tensors
                output_tensor = self.codegen_scalar_to_tensor(output)
                self.wrapper_call.writeline(
                    f"output_handles[{idx}] = {output_tensor}.release();"
                )
                continue

            if is_constant_buffer:
                # See NOTE(return_constant) above.
                self.wrapper_call.writeline(
                    "AOTI_TORCH_ERROR_CODE_CHECK("
                    f"aoti_torch_clone({output}, &output_handles[{idx}]));"
                )
            else:
                if output in output2idx:
                    src_idx = output2idx[output]
                    self.wrapper_call.writeline(
                        f"output_handles[{idx}] = output_handles[{src_idx}];"
                    )
                else:
                    self.wrapper_call.writeline(
                        f"output_handles[{idx}] = {output}.release();"
                    )

            if output not in output2idx:
                output2idx[output] = idx

    def _codegen_entry_impl_prologue(self):
        # [FUNCTION DOC] HELPER: Hook for subclasses to emit code at the top of the entry function, before the GIL is
        # released; here only emits a debug sync when that config is enabled.
        """Hook for subclasses to emit code at the top of inductor_entry_impl,
        before the GIL is released. Default no-op."""
        if config.triton.debug_sync_graph:
            self.generate_debug_sync(self.prefix)

    def generate_before_suffix(self, result):
        # [FUNCTION DOC] Closes the entry function with a closing brace and a comment naming run_impl or _const_run_impl
        # (AOT, non dual mode) or inductor_entry_impl.
        # Close the entry function. In dual-wrapper-mode `result` is the JIT side;
        # the AOTI side close is emitted by _generate_end_aoti.
        if V.graph.aot_mode and not V.graph.is_dual_wrapper_mode:
            entry = "_const_run_impl" if V.graph.is_const_graph else "run_impl"
            result.writeline(f"}} // {self.aoti_model_class_name}::{entry}")
        else:
            result.writeline("} // inductor_entry_impl")

    def _generate_end_aoti(self, result):
        # [FUNCTION DOC] HELPER: Closes the AOTI namespace. In dual-wrapper mode assembles the AOT source (header,
        # prefix, indented wrapper call, closing, suffix) from the AOT buffers into self._aot_output, leaving result
        # untouched; raises AssertionError if buffers are not dual buffers. Otherwise closes the namespace in result
        # unless this is the const graph.
        """Close AOTI namespace. In dual-wrapper-mode, also assemble the
        AOTI source string from `.aot` buffers into self._aot_output; the
        input `result` is the JIT side and is left untouched in that case.
        """
        if V.graph.is_dual_wrapper_mode:
            if not isinstance(self.header, DualIndentedBuffer):
                raise AssertionError(
                    f"expected DualIndentedBuffer, got {type(self.header)}"
                )
            if not isinstance(self.prefix, DualIndentedBuffer):
                raise AssertionError(
                    f"expected DualIndentedBuffer, got {type(self.prefix)}"
                )
            if not isinstance(self.wrapper_call, DualIndentedBuffer):
                raise AssertionError(
                    f"expected DualIndentedBuffer, got {type(self.wrapper_call)}"
                )
            aot_result = IndentedBuffer()
            aot_result.splice(self.header.aot)
            aot_result.splice(self.prefix.aot)
            with aot_result.indent():
                aot_result.splice(self.wrapper_call.aot)
            entry = "_const_run_impl" if V.graph.is_const_graph else "run_impl"
            aot_result.writeline(f"}} // {self.aoti_model_class_name}::{entry}")
            aot_result.splice(self.suffix)
            if not V.graph.is_const_graph:
                aot_result.writeline("} // namespace torch::aot_inductor\n\n\n")
            self._aot_output = aot_result.getvalue()
        elif not V.graph.is_const_graph:
            result.writeline("} // namespace torch::aot_inductor\n\n\n")

    def _generate_end_jit(self, result):
        # [FUNCTION DOC] HELPER: Closes the JIT C++ source string (merging or separating kernel source per config) and
        # appends Python code that loads the compiled entry via CppWrapperCodeCache (with vec ISA, output count and
        # extra link flags) and defines a call function that converts args and constants to tensor handles, calls the
        # entry, and converts handles back to tensors or Python scalars for shape outputs. Raises AssertionError if a
        # constant is not a tensor.
        """Close JIT C++ block and emit Python wrapper for loading/calling."""
        if config.cpp_wrapper_build_separate:
            # Close the wrapper code block, then write any kernel definitions.
            result.splice('"""\n)')
            if self.kernel_declarations:
                result.splice('\nkernel_src = (\nr"""')
                result.splice(self.kernel_declarations.getvalue())
                result.splice('"""\n)')
            else:
                result.splice("""
                    kernel_src = ''
                    """)
        else:
            # Merge main code and kernel code
            result.splice(self.kernel_declarations.getvalue())
            self.kernel_declarations.clear()
            # Close the wrapper code block
            result.splice('"""\n)')

        if config.cpp_wrapper_build_separate:
            kernel_code = "kernel_src"
            needs_vec_isa = "False"
            kernel_needs_vec_isa = str(self.needs_vec_isa)
        else:
            kernel_code = "None"
            needs_vec_isa = str(self.needs_vec_isa)
            kernel_needs_vec_isa = "None"
        # Link any precompiled kernel .so files (CUTLASS, ROCm CK) into the
        # wrapper. The .so files are built without -Wl,-soname, so passing
        # their absolute paths positionally to the linker encodes them as
        # absolute DT_NEEDED entries that resolve at module load (same
        # mechanism HalideCodeCache uses for its standalone runtime).
        extra_flags_repr = repr(tuple(self.external_kernel_libs))
        # Cpp entry function for JIT with cpp wrapper
        result.splice(f"""
            inductor_entry = CppWrapperCodeCache.load_pybinding(
                argtypes=["std::vector<AtenTensorHandle>"],
                main_code=cpp_wrapper_src,
                device_type="{self.device}",
                needs_vec_isa={needs_vec_isa},
                kernel_needs_vec_isa={kernel_needs_vec_isa},
                num_outputs={len(V.graph.graph_outputs)},
                kernel_code={kernel_code},
                extra_flags={extra_flags_repr},
            )
            """)

        input_python_indices = [
            idx
            for idx, value in enumerate(V.graph.graph_inputs.values())
            if isinstance(
                value,
                (ir.TorchBindObject, ir.GeneratorState, ir.OpaqueObjectState),
            )
        ]
        python_only_input_indices = (
            "set()"
            if not input_python_indices
            else "{" + ", ".join(map(str, input_python_indices)) + "}"
        )
        wrapper_body = f"""
                    python_only_input_indices = {python_only_input_indices}
                    input_tensors = []
                    input_handle_positions = []
                    # Preserve graph input slots: Python-only inputs stay None
                    # and are unpacked as nullptrs by the JIT cpp wrapper.
                    input_handles = [None] * len(args)
                    for idx, arg in enumerate(args):
                        if idx in python_only_input_indices:
                            continue
                        input_handle_positions.append(idx)
                        input_tensors.append(
                            arg if isinstance(arg, torch.Tensor) else torch.tensor(arg, device='cpu')
                        )
        """
        if V.graph.constants:
            # Append constants to the input args for cpp wrapper.
            # Python wrapper directly gets the value inside the wrapper call
            # as a global variable passed when calling exec(code, mod.__dict__, mod.__dict__).
            # For cpp wrapper, we need to pass this python value to the inductor_entry_impl function explicitly.
            if not all(
                isinstance(v, torch.Tensor) for v in list(V.graph.constants.values())
            ):
                raise AssertionError("Expect all constants to be Tensor")
            constants_str = f"[{', '.join(V.graph.constants.keys())}]"
            wrapper_body += f"""
                    constants_tensor = {constants_str}
                    constants_start_idx = len(input_handles)
                    input_handles.extend([None] * len(constants_tensor))
                    input_handle_positions.extend(
                        range(constants_start_idx, constants_start_idx + len(constants_tensor))
                    )
                    input_tensors.extend(constants_tensor)
            """
        # Convert vector of at::Tensor to vector of AtenTensorHandle.
        # If we pass at::Tensor, the compilation will be too slow.
        wrapper_body += """
                    packed_input_handles = torch._C._aoti.unsafe_alloc_void_ptrs_from_tensors(input_tensors)
                    for idx, handle in zip(input_handle_positions, packed_input_handles):
                        input_handles[idx] = handle
        """
        # Release the inputs for memory reuse.
        wrapper_body += """
                    args.clear()
                    del input_handle_positions
                    del packed_input_handles
                    del input_tensors
        """

        # unwrap output tensor back to python scalar
        if all(x for x in self.output_is_tensor.values()):
            # If no ShapeAsConstantBuffer in the output, directly return the output as tensors
            outputs_str = "output_tensors"
        else:
            outputs = [
                (
                    f"output_tensors[{i}]"
                    if self.output_is_tensor[i]
                    else f"output_tensors[{i}].item()"
                )
                for i in range(len(V.graph.graph_outputs))
            ]
            outputs_str = f"[{', '.join(outputs)}]"
        wrapper_body += f"""
                    output_handles = f(input_handles)
                    output_tensors = torch._C._aoti.alloc_tensors_by_stealing_from_void_ptrs(output_handles)
                    return {outputs_str}
        """

        # Wrap the func to support setting result._boxed_call = True
        result.splice(f"""
            def _wrap_func(f):
                def g(args):
                    {wrapper_body}
                return g

            call = _wrap_func(inductor_entry)
            """)

    def generate_end(self, result):
        # [FUNCTION DOC] Dispatches to AOT end generation in AOT mode, otherwise to JIT end generation.
        """Generates the end of the code block, and any code needed to call it."""
        if V.graph.aot_mode:
            self._generate_end_aoti(result)
        else:
            self._generate_end_jit(result)

    @staticmethod
    def get_c_shim_func_name(kernel: str, device: str) -> str:
        # [FUNCTION DOC] Returns the C shim function name for a kernel: names starting with the aoti_torch prefix are kept;
        # otherwise builds the aoti_torch_<device>_<op> name from the last (or second to last if last is call) scope
        # component. Raises AssertionError if the name has no scope separator.
        if kernel.startswith("aoti_torch_"):
            return kernel

        if "::" not in kernel:
            raise AssertionError(
                "Cpp kernel name: " + kernel + " does not contain '::'"
            )
        kernel_tokens = kernel.split("::")
        kernel_suffix = kernel_tokens[-1]
        if kernel_suffix == "call":
            kernel_suffix = kernel_tokens[-2]

        shim_fn = f"aoti_torch_{device}_{kernel_suffix}"
        return shim_fn

    def generate_c_shim_extern_kernel_call(
        self,
        kernel: str,
        args: list[str],
        device: str,
        *,
        debug_args: list[str] | None = None,
        stack_traces: OrderedSet[str] | None = None,
    ) -> None:
        # [FUNCTION DOC] Emits a checked call to the C shim function of an extern kernel, wrapped in the debug printer.
        # Adds the device include. If kernel profiling is enabled (linux/win32) wraps the call in a block with a
        # record-function handle and, with the context guard, a kernel context holding stack traces. debug_args
        # substitutes arguments for debug printing only.
        """debug_args kwarg allows CppWrapperCpuArrayRef to pass in wrapped arguments in
        place of args while preserving debug printer output."""
        # We can do this unconditionally, since we cache this call.
        self.add_device_include(device)

        debug_printer_manager = V.graph.wrapper_code.debug_printer
        debug_printer_manager.set_printer_args(
            debug_args if debug_args is not None else args, kernel, None, None, "extern"
        )
        enable_kernel_profile = config.cpp.enable_kernel_profile and sys.platform in [
            "linux",
            "win32",
        ]
        enable_kernel_context_guard = (
            enable_kernel_profile and config.cpp.enable_kernel_context_guard
        )
        with debug_printer_manager:
            shim_fn = self.get_c_shim_func_name(kernel, device)
            shim_fn_codes = [
                f"AOTI_TORCH_ERROR_CODE_CHECK({shim_fn}({', '.join(args)}));"
            ]
            if enable_kernel_profile:
                call_code = shim_fn_codes[0]
                shim_fn_codes = ["{"]
                if enable_kernel_context_guard:
                    stack_trace_str = 'R"('
                    if stack_traces:
                        for stack_trace in stack_traces:
                            for line in stack_trace.split("\n"):
                                stack_trace_str += f"\n{line}"
                            stack_trace_str += "\n"
                    stack_trace_str += ')"'
                    shim_fn_codes.append(
                        f"""KernelContextGuard _ctx("{shim_fn}", {stack_trace_str});"""
                    )
                shim_fn_codes.extend(
                    [
                        f"""RAIIAtenRecordFunctionHandle record_{shim_fn}_("{shim_fn}", nullptr);""",
                        call_code,
                        "}",
                    ]
                )
            self.writelines(shim_fn_codes)

    def generate_c_shim_extern_kernel_alloc(
        self, extern_kernel: ir.ExternKernelAlloc, args: list[str]
    ) -> None:
        # [FUNCTION DOC] Emits a shim call for an extern kernel that allocates its output: declares an output handle
        # unless the op is an inplace view, calls the shim, and wraps the handle in a RAII handle. For all_reduce_
        # and wait_tensor the returned handle is deleted immediately because it is unused.
        # registered output buffer name
        name = extern_kernel.name
        output_handle_name = f"{name}_handle"
        is_inplace = (
            isinstance(extern_kernel.op_overload, torch._ops.OpOverload)
            and torch.Tag.inplace_view in extern_kernel.op_overload.tags
        )

        if not is_inplace:
            self.writeline(f"AtenTensorHandle {output_handle_name};")
            args = [*args, f"&{output_handle_name}"]

        device = d.type if (d := extern_kernel.get_device()) else self.device

        self.generate_c_shim_extern_kernel_call(
            extern_kernel.get_kernel_name(), args, device
        )

        if extern_kernel.python_kernel_name in (
            "torch.ops._c10d_functional.all_reduce_.default",
            "torch.ops._c10d_functional.wait_tensor.default",
        ):
            # all_reduce_ is an inplace op and its returned tensor is not used anywhere.
            # wait_tensor returns its input without any modification and the returned tensor is not used anywhere.
            # In both cases, we can immediately delete the returned AtenTensorHandle to reduce its lifetime.
            self.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_delete_tensor_object({output_handle_name}));"
            )
        elif not is_inplace:
            self.writeline(f"RAIIAtenTensorHandle {name}({output_handle_name});")

    def _generate_extern_kernel_alloc_helper(self, extern_kernel, args):
        # [FUNCTION DOC] HELPER: Uses the fallback-kernel emitter when the extern kernel has multiple outputs, else
        # the single-output alloc emitter.
        if getattr(extern_kernel, "outputs", None):
            # ir.ExternKernelAlloc may have outputs if it returns a tuple
            self.generate_c_shim_fallback_kernel(extern_kernel, args)
        else:
            self.generate_c_shim_extern_kernel_alloc(extern_kernel, args)

    def generate_c_shim_fallback_kernel(
        self, fallback_kernel: ir.FallbackKernel, args: list[str]
    ) -> None:
        # [FUNCTION DOC] Emits a shim call for a fallback kernel with several outputs: tensor outputs get handles,
        # int and symbolic outputs get scalar locals passed by address, None outputs pass null; appends them to args,
        # calls the shim and then wraps tensor handles in RAII handles. Raises AssertionError if a multi-output index
        # mismatches its position and NotImplementedError for unsupported output types.
        output_args = []
        output_raii_handles = []
        output_name_base = fallback_kernel.get_name()
        for idx, output in enumerate(fallback_kernel.outputs):
            if isinstance(output, ir.MultiOutput):
                # TODO: handle integer output (e.g., as in attention)
                name = f"{output.get_name()}"
                output_handle_name = f"{name}_handle"
                if output.indices:
                    if output.indices[0][1] != idx:
                        raise AssertionError(
                            f"expected {output.indices[0][1]=} == {idx=} for {output_name_base=}"
                        )
                self.writeline(f"AtenTensorHandle {output_handle_name};")
                output_args.append(f"&{output_handle_name}")
                output_raii_handles.append(
                    f"RAIIAtenTensorHandle {name}({output_handle_name});"
                )
            elif isinstance(output, int):
                output_name = f"{output_name_base}_{idx}"
                self.writeline(f"int64_t {output_name} = {output};")
                output_args.append(f"&{output_name}")
            elif isinstance(output, sympy.Expr):
                output_name = f"{output_name_base}_{idx}"
                self.writeline(f"auto {output_name} = {cexpr(output)};")
                output_args.append(f"&{output_name}")
            elif output is None:
                output_args.append("nullptr")
            else:
                raise NotImplementedError(f"unsupported type of {output=}")
        args = args + output_args
        device = d.type if (d := fallback_kernel.get_device()) else self.device

        self.generate_c_shim_extern_kernel_call(
            fallback_kernel.cpp_kernel_name,  # type: ignore[arg-type]
            args,
            device,
        )
        for raii_handle in output_raii_handles:
            self.writeline(raii_handle)

    def _generate_extern_kernel_out_helper(
        self,
        kernel: str,
        out: str,
        out_view: str | None,
        args: list[str],
        device: str,
        stack_traces: OrderedSet[str] | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Emits a shim call for an out-variant kernel: prepends the output (or an as-strided
        # view of it declared first) to args and emits the extern call. Mutates args.
        if out_view:
            out_name = f"{out}_as_strided"
            self.writeline(f"auto {out_name} = {out_view};")
            args.insert(0, out_name)
        else:
            args.insert(0, out)

        self.generate_c_shim_extern_kernel_call(
            kernel, args, device, stack_traces=stack_traces
        )

    def _get_scatter_reduce_enum(self, reduce):
        # [FUNCTION DOC] HELPER: Maps scatter reduce names add and multiply to sum and prod; other values unchanged.
        # Follow aten/src/ATen/native/ReductionType.h:get_operator_enum
        get_operator_enum = {"add": "sum", "multiply": "prod"}
        if reduce in get_operator_enum:
            reduce = get_operator_enum[reduce]

        return reduce

    def _generate_scatter_fallback_args(self, inputs: Sequence[Any]) -> list[str]:
        # [FUNCTION DOC] HELPER: Converts scatter fallback inputs to strings; subclasses may wrap them.
        return [str(x) for x in inputs]

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
        # [FUNCTION DOC] HELPER: Emits a checked shim call for scatter ops: uses the out variant of the shim name, passes
        # the output and inputs, appends the kwargs for scatter_reduce, or the reduce argument for tensor src, and
        # raises AssertionError if reduce is set for a scalar src scatter.
        reduce = self._get_scatter_reduce_enum(reduce)

        # call the ABI shim function instead of the ATen one
        self.add_device_include(device)
        cpp_kernel_name = self.get_c_shim_func_name(cpp_kernel_name, device)
        # TODO: consider remove "_out" and add missing inplace variants to fallback_ops.py
        cpp_kernel_name = cpp_kernel_name.replace("__", "_") + "_out"
        # str(output) ensures that CppWrapperCpuArrayRef borrows the output tensor
        args_wrapped = self._generate_scatter_fallback_args((str(output), *inputs))
        # Wrap in AOTI_TORCH_ERROR_CODE_CHECK so a shim failure
        # surfaces instead of being silently swallowed.
        line = f"{cpp_kernel_name}({','.join(args_wrapped)}"

        if python_kernel_name.startswith("aten.scatter_reduce"):
            line += f", {','.join(kwargs)}"
        elif src_is_tensor:
            if reduce:
                line += f", {V.graph.wrapper_code.val_to_arg_str(reduce)}"
        elif reduce is not None:
            raise AssertionError(
                "Expect reduce to be None for aten.scatter_ with scalar src"
            )
        line += ")"
        self.writeline(f"AOTI_TORCH_ERROR_CODE_CHECK({line});")

    def _generate_index_put_fallback(self, kernel, x, indices, values, accumulate):
        # [FUNCTION DOC] HELPER: Emits a checked call of the index_put shim with the tensor as output, a temporary array of
        # index handles, the index count, values and accumulate flag.
        # TODO: update aoti_torch_index_put_out in ir.py to use autogen out version
        # See the comment in codegen_reinterpret_view about why having something like
        # RAIIAtenTensorHandle(tmp_tensor_handle_2) in a tmp array can cause the corresponding
        # tensor prematurely deallocated, thus the temporary array trick here.
        indices_str = self._generate_temporary_array_pointer(
            "AtenTensorHandle", indices
        )
        args = [
            x,
            indices_str,
            str(len(indices)),
            values,
            accumulate,
        ]
        args.insert(0, x)  # set x as the output tensor, this fallback mutates x.
        # Wrap in AOTI_TORCH_ERROR_CODE_CHECK so a shim failure surfaces instead
        # of being silently swallowed.
        self.writeline(f"AOTI_TORCH_ERROR_CODE_CHECK({kernel}({', '.join(args)}));")

    def add_benchmark_harness(self, output):
        # [FUNCTION DOC] Adds the benchmark harness via the base class except in AOT mode, where it does nothing.
        if V.graph.aot_mode:
            return
        super().add_benchmark_harness(output)

    def _extract_divisors_from_expr(
        self, expr: sympy.Expr
    ) -> list[tuple[sympy.Expr, str]]:
        # [FUNCTION DOC] HELPER: Walks a sympy expression and returns unique (divisor, operation name) pairs for floor
        # division, clean division, modulo and modular indexing whose divisor is not a nonzero numeric constant.
        """
        Walk the sympy expression and extract all divisors/modulos from
        FloorDiv, CleanDiv, Mod, and ModularIndexing nodes.

        Returns a list of (divisor_expr, operation_name) tuples for divisors
        that are not constant positive integers (which cannot be zero).
        """
        divisors: list[tuple[sympy.Expr, str]] = []
        seen: OrderedSet[str] = OrderedSet()

        def is_constant_nonzero(e: sympy.Expr) -> bool:
            # [FUNCTION DOC] HELPER: True for a nonzero numeric constant.
            """Check if expression is a constant that cannot be zero."""
            if isinstance(e, sympy.Integer):
                return int(e) != 0
            if isinstance(e, sympy.Number):
                return float(e) != 0
            return False

        def maybe_add_divisor(divisor: sympy.Expr, op_name: str) -> None:
            # [FUNCTION DOC] HELPER: Records a divisor once (deduplicated by string) when it might be zero.
            """Add divisor to list if it could potentially be zero."""
            if not is_constant_nonzero(divisor):
                key = str(divisor)
                if key not in seen:
                    seen.add(key)
                    divisors.append((divisor, op_name))

        def walk(e: sympy.Expr) -> None:
            # [FUNCTION DOC] HELPER: Recursively visits the expression, collecting divisors from division, modulo and
            # modular indexing nodes.
            if isinstance(e, (FloorDiv, CleanDiv)):
                _, div = e.args
                maybe_add_divisor(div, "floor division")
            elif isinstance(e, Mod):
                _, mod = e.args
                maybe_add_divisor(mod, "modulo")
            elif isinstance(e, ModularIndexing):
                _, div, mod = e.args
                maybe_add_divisor(div, "modular indexing division")
                maybe_add_divisor(mod, "modular indexing modulo")

            # Recurse into arguments
            if hasattr(e, "args"):
                for arg in e.args:
                    if isinstance(arg, sympy.Expr):
                        walk(arg)

        walk(expr)
        return divisors

    def codegen_cpp_sizevar(self, x: sympy.Expr, *, simplify: bool = True) -> str:
        # [FUNCTION DOC] Converts a sizevar expression to C++ (optionally simplified first). In AOT mode also emits
        # divide-by-zero asserts for non-constant divisors found in the expression.
        maybe_simplified_x = V.graph.sizevars.simplify(x) if simplify else x
        # In AOT mode, emit runtime checks for potential division/modulo by zero
        # to prevent SIGFPE crashes when symbolic tensor shapes can be 0.
        if V.graph.aot_mode:
            for divisor, op_name in self._extract_divisors_from_expr(
                maybe_simplified_x
            ):
                self.write_assert_div_by_zero(cexpr(divisor), op_name)
        return cexpr(maybe_simplified_x)

    def _codegen_assert_div_by_zero(
        self, code: IndentedBuffer, divisor_str: str, op_name: str
    ) -> None:
        # [FUNCTION DOC] HELPER: Writes one AOT-only divide-by-zero check for the divisor string into the buffer.
        """Emit one div-by-zero AOTI check to `code` (replay-phase target).

        Uses writeline_aot so the check is routed to the AOTI side only in
        dual-wrapper mode; on plain buffers writeline_aot has the expected
        single-stream behavior (AotOnlyBuffer delegates to writeline, plain
        IndentedBuffer is a no-op).
        """
        code.writeline_aot(
            f'AOTI_TORCH_CHECK({divisor_str} != 0, "Integer {op_name} by zero");'
        )

    def codegen_sizevar(self, x: sympy.Expr) -> str:
        # [FUNCTION DOC] Converts a size expression to C++ via codegen_cpp_sizevar.
        return self.codegen_cpp_sizevar(x)

    def codegen_tuple_access(self, basename: str, name: str, index: str) -> str:
        # [FUNCTION DOC] Returns the name unchanged: outputs are returned through arguments so no tuple access is needed.
        # in the abi_compatible mode, outputs are returned via arguments
        return name

    def codegen_shape_tuple(self, shape: Sequence[sympy.Expr]) -> str:
        # [FUNCTION DOC] Formats a shape as a C++ braced list of size expressions; a single dim gets a trailing comma.
        parts = [*map(self.codegen_sizevar, shape)]
        if len(parts) == 0:
            return "{}"
        if len(parts) == 1:
            return f"{{{parts[0]}, }}"
        return f"{{{', '.join(parts)}}}"

    def ensure_size_computed(self, sym: sympy.Symbol):
        # [FUNCTION DOC] Declares a precomputed-size symbol as a 64-bit local from its replacement expression, once.
        if isinstance(sym, sympy.Symbol) and symbol_is_type(sym, SymT.PRECOMPUTED_SIZE):
            if sym in self.computed_sizes:
                return
            self.computed_sizes.add(sym)
            expr = V.graph.sizevars.inv_precomputed_replacements[sym]
            self.writeline(f"int64_t {sym} = {cexpr(expr)};")

    def _generate_symbolic_call_arg_helper(
        self, arg: SymbolicCallArg, graph: GraphLowering
    ) -> None:
        # [FUNCTION DOC] HELPER: Emits the C++ line computing a symbolic call arg's inner value. It declares a 64-bit local
        # the first time per (arg, graph), or always when kernel profiling is on (each kernel call is in its own scope);
        # otherwise it emits a plain reassignment. Records the pair in kernel_numel_expr.
        enable_kernel_profile = config.cpp.enable_kernel_profile and sys.platform in [
            "linux",
            "win32",
        ]
        if enable_kernel_profile or (arg.inner, graph) not in self.kernel_numel_expr:
            # When enable_kernel_profile is on, each kernel call is wrapped in
            # its own {} scope block, so we must redeclare the variable each
            # time since prior declarations are no longer visible.
            self.kernel_numel_expr.add((arg.inner, graph))
            self.writeline(f"int64_t {arg.inner} = {cexpr(arg.inner_expr)};")
        else:
            self.writeline(f"{arg.inner} = {cexpr(arg.inner_expr)};")

    def _codegen_dynamic_scalar(self, node):
        # [FUNCTION DOC] Emits C++ reading a scalar from a tensor input into a raw variable and then defining the node's
        # symbol from it: directly, as a bool-to-int conversion, or divided by the key divisor, depending on the keypath.
        # Registers the symbol in unbacked_symbol_decls. Raises AssertionError for an unrecognized keypath.
        # KNOWN ISSUE: the divide-by-key path has no divisibility check (acknowledged by a TODO).
        (data,) = (t.codegen_reference() for t in node.inputs)
        self.codegen_tensor_item(node.inputs[0].get_dtype(), data, f"{node.sym}_raw")

        if len(node.keypath) == 0:
            self.writeline(f"auto {node.sym} = {node.sym}_raw;")
        elif len(node.keypath) == 1 and isinstance(node.keypath[0], ConvertIntKey):
            self.writeline(f"int64_t {node.sym} = {node.sym}_raw ? 1 : 0;")
        elif len(node.keypath) == 1 and isinstance(node.keypath[0], DivideByKey):
            # TODO: assert divisibility here
            self.writeline(
                f"int64_t {node.sym} = {node.sym}_raw / {node.keypath[0].divisor};"
            )
        else:
            raise AssertionError(f"unrecognized keypath {node.keypath}")

        # record in unbacked_symbol_decls so we won't generate a declaration of the symbol again
        self.unbacked_symbol_decls.add(str(node.sym))

    def codegen_dynamic_select_index(self, node, clamp):
        # [FUNCTION DOC] Emits C++ computing the unbacked offset symbol for a dynamic select: normalizes a negative index
        # by adding the size, optionally clamps to the size when clamp is set, then offset = base offset plus base stride
        # times index. Registers the symbol in unbacked_symbol_decls.
        # KNOWN ISSUE: the clamp expression bounds the index to the size, not size minus one.
        index_cpp_str = self.val_to_arg_str_for_prim_type(node.index, int)
        size_cpp_str = self.val_to_arg_str_for_prim_type(node.size, int)

        # codegen index
        sym = node.unbacked_offset_symbol
        index_str = (
            f"{index_cpp_str} < 0 ? {index_cpp_str} + "
            f"{self.val_to_arg_str_for_prim_type(node.size, int)}: {index_cpp_str}"
        )
        self.writeline(f"auto {sym}_index = {index_str};")
        index_str_clamped = (
            f"{sym}_index < 0 ? 0 : ({sym}_index > {size_cpp_str} ? {size_cpp_str} : {sym}_index)"
            if clamp
            else f"{sym}_index"
        )
        self.writeline(f"auto {sym}_index_clamped = {index_str_clamped};")
        self.writeline(
            f"auto {sym} = {self.val_to_arg_str_for_prim_type(node.base_offset, int)} + "
            f"{self.val_to_arg_str_for_prim_type(node.base_dim_stride, int)} * {sym}_index_clamped;"
        )
        # record in unbacked_symbol_decls so we won't generate a declaration of the symbol again
        self.unbacked_symbol_decls.add(str(sym))

    def codegen_dynamic_slice_size(self, node):
        # [FUNCTION DOC] Emits C++ computing the unbacked size symbol of a dynamic slice: clamps start and end into the
        # range zero to size (after normalizing negatives), computes the length (with ceiling division by step when step
        # is not one), floors it at zero, and registers the symbol in unbacked_symbol_decls.
        start_cpp_str = self.val_to_arg_str_for_prim_type(node.start, int)
        end_cpp_str = self.val_to_arg_str_for_prim_type(node.end, int)
        size_cpp_str = self.val_to_arg_str_for_prim_type(node.size, int)
        step_cpp_str = self.val_to_arg_str_for_prim_type(node.step, int)
        sym = node.unbacked_size_symbol

        def codegen_clamp(index_str, start=True):
            # [FUNCTION DOC] HELPER: Emits two C++ lines producing a normalized (negative adjusted) and a clamped copy of a
            # start or end index for the slice size computation.
            suf = "st" if start else "en"
            index_ = f"{sym}_{suf}_index"
            self.writeline(
                f"int64_t {index_} = {index_str} < 0 ? {index_str} + {size_cpp_str} : {index_str};"
            )
            self.writeline(
                f"int64_t {sym}_{suf}_cl = {index_} < 0 ? 0 : ({index_} > {size_cpp_str} ? {size_cpp_str} : {index_});"
            )

        codegen_clamp(start_cpp_str, start=True)
        codegen_clamp(end_cpp_str, start=False)
        if node.step == 1:
            step_str = f"{sym}_en_cl - {sym}_st_cl"
        else:
            step_str = (
                f"({sym}_en_cl - {sym}_st_cl + {step_cpp_str} - 1) / {step_cpp_str}"
            )
        self.writeline(f"int64_t {sym}_with_step = {step_str};")
        self.writeline(f"int64_t {sym} = {sym}_with_step < 0 ? 0 : {sym}_with_step;")
        self.unbacked_symbol_decls.add(str(sym))

    def make_buffer_free(self, buffer):
        # [FUNCTION DOC] Returns the C++ statement that resets the buffer's handle, or an empty string for multi-output
        # layouts and TMA descriptors.
        return (
            ""
            if isinstance(buffer.get_output_spec(), ir.MultiOutputLayout)
            or isinstance(buffer, ir.TMADescriptor)
            else f"{buffer.get_name()}.reset();"
        )

    def make_free_by_names(self, names_to_del: list[str]):
        # [FUNCTION DOC] Returns one C++ statement string resetting the handle of each given name.
        return " ".join(f"{name}.reset();" for name in names_to_del)

    def codegen_exact_buffer_reuse(self, old_name: str, new_name: str, del_line: str):
        # [FUNCTION DOC] Returns a C++ statement that moves the old buffer into a new name to implement buffer reuse.
        return f"auto {new_name} = std::move({old_name});  // reuse"

    def generate_debug_sync(self, buffer):
        # [FUNCTION DOC] Does nothing: debug sync is not generated for the CPU C++ wrapper (GPU subclasses override).
        pass

    def generate_profiler_mark_wrapper_call(self, stack):
        # [FUNCTION DOC] Writes a C++ record-function handle for the inductor wrapper call into wrapper_call.
        self.wrapper_call.writeline(
            'RAIIAtenRecordFunctionHandle record_inductor_wrapper_call_("inductor_wrapper_call", nullptr);'
        )

    def generate_start_graph(self):
        # [FUNCTION DOC] Does nothing; the C++ wrapper needs no graph start code.
        pass

    def generate_end_graph(self):
        # [FUNCTION DOC] Does nothing; the C++ wrapper needs no graph end code.
        pass

    def generate_inf_and_nan_checker(self, nodes):
        # [FUNCTION DOC] Emits a checked inf/nan shim call for each buffer name of the given node group.
        for buf in nodes.get_names():
            # TODO: Add buf name directly into check_inf_and_nan.
            self.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_check_inf_and_nan({buf}));"
            )

    def codegen_input_nan_asserts(self) -> None:
        # [FUNCTION DOC] Writes a comment plus a checked inf/nan shim call for every graph input into wrapper_call,
        # skipping symbolic expressions and torchbind objects.
        self.wrapper_call.writeline("// make sure graph inputs are not nan/inf")
        for name, buf in self.get_graph_inputs().items():
            if isinstance(buf, (sympy.Expr, ir.TorchBindObject)):
                continue
            self.wrapper_call.writeline(
                f'AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_check_inf_and_nan("{name}", {name}));'
            )

    def _codegen_assert_size_stride(
        self,
        code: IndentedBuffer,
        name: str,
        size: str,
        stride: str,
        op_name: str,
        dtype: torch.dtype | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Emits an assert-size-stride statement for a buffer (optionally with dtype) into code.
        # Skipped for the AOT const graph. In AOT mode the check is guarded by the runtime-check environment flag; in
        # dual-wrapper mode the JIT side gets the plain assert and the AOT side the guarded one.
        if V.graph.aot_mode and V.graph.is_const_graph:
            return
        dtype_args = (
            f', {self.codegen_dtype(dtype)}, "{dtype}"' if dtype is not None else ""
        )
        stmt = f'assert_size_stride({name}, {size}, {stride}, "{op_name}"{dtype_args});'
        if V.graph.aot_mode:
            guarded = f"if (_check_aoti_runtime_check_inputs_env()) {{ {stmt} }}"
            if V.graph.is_dual_wrapper_mode:
                # _check_aoti_runtime_check_inputs_env() is AOTI-only; the JIT
                # side gets the plain assert.
                code.writeline_jit(stmt)
                code.writeline_aot(guarded)
            else:
                code.writeline(guarded)
        else:
            code.writeline(stmt)

    def _codegen_assert_size_stride_grouped(
        self, code: IndentedBuffer, asserts: list[tuple[str, str, str]], op_name: str
    ) -> None:
        # [FUNCTION DOC] HELPER: Emits one size-stride assert per (name, size, stride) tuple.
        for name, size, stride in asserts:
            self._codegen_assert_size_stride(code, name, size, stride, op_name)

    def codegen_device(self, device):
        # [FUNCTION DOC] Returns the C++ "device type constant, index" string for a device and records the device in the
        # cache set. Raises AssertionError if the device type has no ATen mapping.
        if device.type not in DEVICE_TO_ATEN:
            raise AssertionError(device.type + " not found in DEVICE_TO_ATEN")
        device_str = DEVICE_TO_ATEN[device.type][5:].lower()  # remove "at::k"
        self.used_cached_devices.add(device_str)
        return f"cached_torch_device_type_{device_str}, {device.index or 0}"

    def codegen_device_idx(self, device, device_id):
        # [FUNCTION DOC] Returns the device index expression: unchanged in JIT mode. In AOT mode returns the model's
        # device index member, except for explicit indices of the primary device type that differ from the primary
        # device index, which are kept so cross-device copies stay intact.
        device_id = device_id.strip()
        if not V.graph.aot_mode:
            return device_id

        if device.type == V.graph.device_type and device.index is not None:
            # The primary compile-time device is relocatable via device_index at load.
            # Other explicit device indices must stay intact for cross-device copies.
            primary_device_idx = next(iter(V.graph.device_idxs), None)
            if primary_device_idx is not None and device.index != primary_device_idx:
                return device_id

        return "this->device_idx_"

    def codegen_dtype(self, dtype):
        # [FUNCTION DOC] Returns the cached C++ dtype constant name for a dtype and records it as used.
        dtype_str = str(dtype).split(".")[-1]
        self.used_cached_dtypes.add(dtype_str)
        return f"cached_torch_dtype_{dtype_str}"

    def codegen_layout(self, layout):
        # [FUNCTION DOC] Returns the cached C++ layout constant name for a layout and records it as used.
        layout_str = str(layout).split(".")[-1]
        self.used_cached_layouts.add(layout_str)
        return f"cached_torch_layout_{layout_str}"

    def codegen_memory_format(self, memory_format):
        # [FUNCTION DOC] Returns the cached C++ memory-format constant name for a format and records it as used.
        memory_format_str = str(memory_format).split(".")[-1]
        self.used_cached_memory_formats.add(memory_format_str)
        return f"cached_torch_memory_format_{memory_format_str}"

    def codegen_int_array_var(
        self,
        int_array: str,
        writeline: Callable[..., None],
        known_statically=False,
        graph=None,  # for per-graph caching
    ) -> str:
        # [FUNCTION DOC] Returns the name of a C++ int array variable for an array literal string, declaring it through
        # the given writeline on first request. Results are cached per array text, writeline owner, static-ness and graph.
        # Use id(graph) for caching to avoid circular references
        # Bound methods have transient IDs, so we use the ID of the bound object instead.
        writeline_id = (
            id(writeline.__self__) if hasattr(writeline, "__self__") else id(writeline)
        )
        cache_key = (
            int_array,
            writeline_id,
            known_statically,
            id(graph) if graph else None,
        )
        if cache_key not in self.codegen_int_array_var_cache:
            self.codegen_int_array_var_cache[cache_key] = (
                self._codegen_int_array_var_impl(int_array, writeline, known_statically)
            )

        return self.codegen_int_array_var_cache[cache_key]

    def _codegen_int_array_var_impl(
        self,
        int_array: str,
        writeline: Callable[..., None],
        known_statically: bool,
    ) -> str:
        # [FUNCTION DOC] HELPER: Declares a fresh int array variable via the writeline callback and returns its name.
        # An empty array becomes a null pointer; otherwise the array is declared once (static const or constexpr when
        # known statically, per config). Declarations depend on which codegen pass is running, hence the callback.
        # Used for size/stride declaration
        #
        # Because the memory planning is done in two passes (see the implementation
        # of self.generate), the writeline behavior is different in the two passes.
        # As a result, the emitted int array declarations may appear in a later
        # position of the generated code, so the second pass codegen should not
        # reuse int array declarations generated in the first pass.
        # This is why writeline needs to explicitly passed in as a parameter.
        var = f"int_array_{next(self.int_array_id)}"
        ctype = "int64_t"
        if int_array == "{}":
            #  An array of unknown bound cannot be initialized with {}.
            if known_statically:
                if config.cpp.use_constexpr_for_int_array:
                    writeline(f"static constexpr {ctype} *{var}=nullptr;")
                else:
                    writeline(f"static const {ctype} *{var}=nullptr;")
            else:
                writeline(f"const {ctype} *{var}=nullptr;")
        else:
            if var not in self.declared_int_array_vars:
                self.declared_int_array_vars.add(var)
                if known_statically:
                    if config.cpp.use_constexpr_for_int_array:
                        writeline(f"static constexpr {ctype} {var}[] = {int_array};")
                    else:
                        writeline(f"static const {ctype} {var}[] = {int_array};")
                else:
                    writeline(f"const {ctype} {var}[] = {int_array};")
        return var

    def make_buffer_allocation(self, buffer):
        # [FUNCTION DOC] Returns the allocation statement for a buffer, using its name, device, dtype, size, stride,
        # allocation size and pinned flag.
        return self.make_allocation(
            buffer.get_name(),
            buffer.get_device(),
            buffer.get_dtype(),
            buffer.get_size(),
            buffer.get_stride(),
            V.graph.get_allocation_size(buffer),
            buffer.get_is_pinned(),
        )

    def make_allocation(
        self, name, device, dtype, shape, stride, allocation_shape=None, is_pinned=False
    ):
        # [FUNCTION DOC] Emits C++ to allocate a tensor handle (empty strided, pinned variant if requested) into
        # wrapper_call, using int array variables for size, allocation size and stride; AOT and JIT sides get different
        # device index expressions. If the allocation shape differs from the logical shape it emits an as-strided view
        # of the handle. Returns the RAII handle declaration string for the caller to write.
        """Emit C++ to allocate a tensor handle via aoti_torch_empty_strided.

        Returns the RAIIAtenTensorHandle binding string; if the allocation
        shape differs from the logical shape, a follow-up aoti_torch_as_strided
        is emitted to reinterpret the handle.
        """
        if allocation_shape is None:
            allocation_shape = shape

        orig_stride = stride
        device_str = self.codegen_device(device)
        dtype_code = self.codegen_dtype(dtype)
        size = self.codegen_shape_tuple(shape)
        allocation_size = self.codegen_shape_tuple(allocation_shape)
        stride = self.codegen_shape_tuple(orig_stride)

        size_array_var = self.codegen_int_array_var(
            size,
            self.wrapper_call.writeline,
            known_statically=self.is_statically_known_list_of_ints(shape),
            graph=self.get_codegened_graph(),
        )

        if allocation_size != size:
            allocation_size_array_var = self.codegen_int_array_var(
                allocation_size,
                self.wrapper_call.writeline,
                known_statically=self.is_statically_known_list_of_ints(
                    allocation_shape
                ),
                graph=self.get_codegened_graph(),
            )
        else:
            allocation_size_array_var = size_array_var

        stride_array_var = self.codegen_int_array_var(
            stride,
            self.wrapper_call.writeline,
            known_statically=self.is_statically_known_list_of_ints(orig_stride),
            graph=self.get_codegened_graph(),
        )
        device_type, device_id = device_str.split(",")
        handle_name = f"{name}_handle"
        self.wrapper_call.writeline(f"AtenTensorHandle {handle_name};")
        pinned_str = "_pinned" if is_pinned else ""

        def _alloc_line(device_idx: str) -> str:
        # [FUNCTION DOC] HELPER: Builds the checked empty-strided call line for a given device index expression.
            args = ", ".join(
                [
                    str(len(shape)),
                    allocation_size_array_var,
                    stride_array_var,
                    dtype_code,
                    device_type,
                    device_idx,
                    f"&{handle_name}",
                ]
            )
            return f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided{pinned_str}({args}));"

        # AOTI side: codegen_device_idx returns this->device_idx_ for the
        # primary device so the model honors runtime device assignment, but
        # keeps concrete indices for non-primary (cross-device) allocations.
        # JIT side (pure or dual mode's JIT) has no this->device_idx_ in scope,
        # so use the concrete graph device id.
        self.wrapper_call.writeline_aot(
            _alloc_line(self.codegen_device_idx(device, device_id))
        )
        self.wrapper_call.writeline_jit(_alloc_line(device_id))

        if allocation_size != size:
            old_handle_name, handle_name = handle_name, f"{name}_handle_restrided"
            self.wrapper_call.writeline(f"AtenTensorHandle {handle_name};")
            args = [
                old_handle_name,
                size_array_var,
                stride_array_var,
                f"&{handle_name}",
            ]
            self.wrapper_call.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_as_strided({', '.join(args)}));"
            )
            self.wrapper_call.writeline(
                f"wrap_with_raii_handle_if_needed({old_handle_name});"
            )

        return f"RAIIAtenTensorHandle {name}({handle_name});"

    def codegen_alloc_from_pool(
        self, name, offset, dtype, shape, stride
    ) -> tuple[str, list[str]]:
        # [FUNCTION DOC] Builds an allocation from a memory pool at a byte offset. Returns the RAII handle expression and
        # the list of lines to write; the lines are returned instead of written so callers must pair them with the
        # RAII wrapper to avoid leaks.
        size = self.codegen_shape_tuple(shape)
        stride = self.codegen_shape_tuple(stride)
        tmp_name = f"tmp_tensor_handle_{next(self.tmp_tensor_id)}"
        args = [
            name,
            cexpr(offset),  # bytes not numel
            self.codegen_dtype(dtype),
            str(len(shape)),
            self.codegen_int_array_var(
                size, self.wrapper_call.writeline, graph=self.get_codegened_graph()
            ),
            self.codegen_int_array_var(
                stride, self.wrapper_call.writeline, graph=self.get_codegened_graph()
            ),
            f"&{tmp_name}",
        ]
        # We return the lines instead of writing here because writing here is bug prone.
        # If you write aoti_torch__alloc_from_pool lines, you must write the RAIIAtenTensorHandle
        # as well, otherwise you get memory leaks
        allocations_to_write = [
            f"AtenTensorHandle {tmp_name};",
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch__alloc_from_pool({', '.join(args)}));",
        ]
        return f"RAIIAtenTensorHandle({tmp_name})", allocations_to_write

    def codegen_reinterpret_view(
        self,
        data,
        size,
        stride,
        offset,
        writeline: Callable[..., None],
        dtype=None,
    ) -> str:
        # [FUNCTION DOC] Emits code (through the given writeline) for a reinterpreted view of a tensor and returns a
        # temporary RAII handle expression. If the view layout matches the base (after collapsing nested views), it makes
        # a new handle, or a dtype view when dtype differs; otherwise it reinterprets size/stride/offset and then applies a
        # dtype view if needed. The returned handle is temporary; callers must save it for persistent use.
        """Returns a newly-created, temporary RAII tensor handle containing the
        reinterpreted tensor data.  Callers of this function are responsible for saving
        the handle if persistent access is needed."""

        d_size, d_stride, d_offset, d_dtype, collapsible = (
            codegen_reinterpret_view_helper(data)
        )

        dim = str(len(size))
        original_offset = offset
        offset = self.codegen_sizevar(offset)
        call_strs = []
        final_tensor_str = None

        def create_reinterpret_call() -> str:
        # [FUNCTION DOC] HELPER: Builds the reinterpret-tensor wrapper call expression from data, rank, size and stride
        # int arrays and offset.
            args = [
                f"{data.get_name()}",
                dim,
                self.codegen_int_array_var(
                    self.codegen_shape_tuple(size),
                    writeline,
                    known_statically=self.is_statically_known_list_of_ints(size),
                    graph=self.get_codegened_graph(),
                ),
                self.codegen_int_array_var(
                    self.codegen_shape_tuple(stride),
                    writeline,
                    known_statically=self.is_statically_known_list_of_ints(stride),
                    graph=self.get_codegened_graph(),
                ),
                offset,
            ]
            return f"wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper({', '.join(args)}))"

        def create_dtypeview_call(reinterpret_call: str) -> tuple[str, list[str]]:
        # [FUNCTION DOC] HELPER: Returns a new RAII handle expression and the lines that create a dtype view of the given
        # tensor expression using the device-specific shim.
            tmp_AtenTensorHandle = f"tmp_{data.get_name()}_{next(self.tmp_tensor_id)}"
            tmp_call_strs = [f"AtenTensorHandle {tmp_AtenTensorHandle};"]
            device_name = data.layout.device.type
            dtypeview_function = f"aoti_torch_{device_name}_view_dtype"
            tmp_call_strs.append(
                f"AOTI_TORCH_ERROR_CODE_CHECK({dtypeview_function}"
                f"({reinterpret_call}, {self.codegen_dtype(dtype)}, &{tmp_AtenTensorHandle}));"
            )
            return f"RAIIAtenTensorHandle({tmp_AtenTensorHandle})", tmp_call_strs

        def create_new_tensor_handle() -> tuple[str, list[str]]:
        # [FUNCTION DOC] HELPER: Returns a new RAII handle expression and lines creating a new tensor handle aliasing data.
            tmp_AtenTensorHandle = f"tmp_{data.get_name()}_{next(self.tmp_tensor_id)}"
            tmp_call_strs = [
                f"AtenTensorHandle {tmp_AtenTensorHandle};",
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_new_tensor_handle({data.get_name()}, &{tmp_AtenTensorHandle}));",
            ]
            return f"RAIIAtenTensorHandle({tmp_AtenTensorHandle})", tmp_call_strs

        collapsed = collapsible and original_offset == d_offset
        if collapsed:
            same_layout = size == d_size and stride == d_stride
            base_dtype = d_dtype
        else:
            same_layout = (
                size == data.layout.size
                and stride == data.layout.stride
                and original_offset == data.layout.offset
            )
            base_dtype = data.dtype

        if same_layout:
            # pure dtypeview
            if dtype is not None and dtype != base_dtype:
                final_tensor_str, tmp_call_strs = create_dtypeview_call(data.get_name())
            else:
                final_tensor_str, tmp_call_strs = create_new_tensor_handle()
            call_strs.extend(tmp_call_strs)
        else:
            # firstly create reinterpretview
            final_tensor_str = create_reinterpret_call()
            if dtype is not None and dtype != base_dtype:
                # wrap it with dtypeview
                final_tensor_str, tmp_call_strs = create_dtypeview_call(
                    final_tensor_str
                )
                call_strs.extend(tmp_call_strs)

        for line in call_strs:
            writeline(line)

        # NB, the return handle here represents a temporary tensor, which will be automatically
        # released.
        # Here's a sample usage in the cpp wrapper code:
        # ```
        # aoti_torch_addmm_out(
        #     buf1,
        #     arg1_1,
        #     RAIIAtenTensorHandle(tmp_tensor_handle_0),
        #     buf0,
        #     1L,
        #     1L));
        # ```
        # RAIIAtenTensorHandle(tmp_tensor_handle_0) will be released after the call to addmm_out.
        # This could be problematic when it's used in a different pattern, for example:
        # ````
        # AtenTensorHandle tensor_args[] = {RAIIAtenTensorHandle(tmp_tensor_handle_2), buf5, buf6};
        # aoti_torch_proxy_executor_call_function(..., tensor_args);
        # ````
        # RAIIAtenTensorHandle(tmp_tensor_handle_2) will be invalid when it's used in the latter
        # kernel call.
        #
        # This is solved by updating the proxy_executor invocation to
        # ```
        # aoti_torch_proxy_executor_call_function(...,
        #     std::array<AtenTensorHandle, 3>{
        #         RAIIAtenTensorHandle(tmp_tensor_handle_2), buf5, buf6
        #     }.cbegin()
        # );
        # ```
        return final_tensor_str

    def codegen_device_copy(self, src, dst, non_blocking: bool | str):
        # [FUNCTION DOC] Emits a checked copy shim call from src to dst with the non-blocking flag.
        """This function is overridden by cpp_wrapper_cpu_array_ref, so we don't need to
        handle cases where dst is not an AtenTensorHandle."""
        self.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_copy_({dst}, {src}, {non_blocking}));"
        )

    def sync_d2h_copy(self, buffer_name: str) -> None:
        # [FUNCTION DOC] Does nothing; D2H copy synchronization has no AOTI C API yet.
        # TODO: add AOTI C API for event-based D2H copy synchronization
        pass

    def codegen_multi_output(self, node: ir.MultiOutput):
        # [FUNCTION DOC] Does nothing: multi-output results are returned through output pointers by the shim calls.
        # in the abi_compatible mode, outputs are retrieved by passing
        # output pointers, so we skip its codegen here.
        pass

    def codegen_subgraph_prefix(self, subgraph, outer_inputs, outer_outputs):
        # [FUNCTION DOC] Emits, for each tensor subgraph input, a handle copied (assigned) from the corresponding outer
        # input and an RAII wrapper for it. Copying is needed because the outer input may be an rvalue. Raises
        # AssertionError if input counts differ.
        if len(subgraph.graph.graph_inputs) != len(outer_inputs):
            raise AssertionError(
                f"expected len(subgraph.graph.graph_inputs) == len(outer_inputs), "
                f"got {len(subgraph.graph.graph_inputs)} and {len(outer_inputs)}"
            )

        for (inner_input, inner_input_val), outer_input in zip(
            subgraph.graph.graph_inputs.items(), outer_inputs
        ):
            if not isinstance(inner_input_val, ir.TensorBox):
                continue

            # in ABI-compatible mode, we copy the underlying at::Tensor of the conditional
            # input (outer_input) into another at::Tensor to be used as a subgraph input
            # (inner_input) in the nested scope. we can't std::move here, as the codegened
            # outer input may be an expression / rvalue (e.g., reinterpret_view(x)), so we
            # can't necessarily std::move it back to the origin (x).
            self.writeline(f"AtenTensorHandle {inner_input}_handle;")
            self.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_assign_tensors_out({outer_input}, &{inner_input}_handle));"
            )
            self.writeline(f"RAIIAtenTensorHandle {inner_input}({inner_input}_handle);")

    def codegen_subgraph_suffix(self, subgraph, outer_inputs, outer_outputs):
        # [FUNCTION DOC] Emits, per subgraph output, an assignment to the outer output; tensor outputs are moved (after
        # resetting the outer value) since the RAII handle cannot be copied.
        for inner_output, outer_output in zip(
            subgraph.graph.graph_outputs, outer_outputs
        ):
            src = inner_output.codegen_reference()
            if not isinstance(inner_output, ir.ShapeAsConstantBuffer):
                # in ABI-compatible mode, we need to std::move subgraph output (inner_output)
                # to the conditional output (outer_output), as RAIIAtenTensorHandle's copy
                # constructor is deleted.
                src = f"std::move({src})"
                # in case the outer_output carried a value
                # before (e.g., in the while_loop codegen)
                self.writeline(f"{outer_output}.reset();")
            self.writeline(f"{outer_output} = {src};")

    @staticmethod
    def _get_while_loop_carried_output_names(while_loop):
        # [FUNCTION DOC] HELPER: Returns, for each carried input of a while loop, the name of the output holding its
        # updated value: the mutation output when the input is mutated (following chains), otherwise the next non-mutated
        # loop output in order. Raises AssertionError on mutation cycles or mismatched output counts.
        mutation_output_names = {
            mutation_name: mutation_output.get_name()
            for mutation_output in while_loop.mutation_outputs
            for mutation_name in mutation_output.get_mutation_names()
        }

        def resolve_mutation_output_name(name):
        # [FUNCTION DOC] HELPER: Follows the mutation name mapping until reaching a name that is not mutated further.
            seen = OrderedSet()
            while name in mutation_output_names:
                if name in seen:
                    raise AssertionError("while_loop mutation outputs contain a cycle")
                seen.add(name)
                name = mutation_output_names[name]
            return name

        output_names = []
        output_idx = 0
        for carried_input in while_loop.carried_inputs:
            carried_input_name = carried_input.get_name()
            mutation_output_name = resolve_mutation_output_name(carried_input_name)
            if mutation_output_name != carried_input_name:
                output_names.append(mutation_output_name)
                continue

            if output_idx >= len(while_loop.outputs):
                raise AssertionError(
                    "while_loop has fewer non-mutated outputs than carried inputs"
                )
            output_names.append(while_loop.outputs[output_idx].get_name())
            output_idx += 1

        if output_idx != len(while_loop.outputs):
            raise AssertionError("while_loop has unused non-mutated outputs")

        return output_names

    def codegen_invoke_subgraph(self, invoke_subgraph):
        # [FUNCTION DOC] Not supported for the C++ wrapper: always raises NotImplementedError.
        raise NotImplementedError(
            "codegen invoke_subgraph is not implemented for cpp wrapper"
        )

    def codegen_switch(self, node) -> None:
        # [FUNCTION DOC] Emits a C++ if/else-if/else chain for a cond/switch node. Declares outer output handles, reads
        # the selector as a 64-bit scalar (once per name) or casts a shape-expression selector, and emits each branch via
        # codegen_subgraph. In dual-wrapper mode the JIT side visits every branch once (for lazy autotune) in plain
        # blocks, and the AOT side gets the real chain; both are joined in a DualWrapperCodeLine.
        """Emit ABI-compatible C++ for a higher-order cond/switch."""
        outer_inputs = [f"{buf.codegen_reference()}" for buf in node.operands]
        outer_outputs = []
        for out in node.outputs:
            # in ABI-compatible mode, ir.MultiOutput is not codegened,
            # hence pre-declare output variables directly and separately
            self.writeline(f"RAIIAtenTensorHandle {out.get_name()};")
            outer_outputs.append(out.get_name())

        if not isinstance(node.selector, ir.ShapeAsConstantBuffer):
            # use the ABI shim to extract the scalar selector value from the Tensor;
            # always as int64 so the branch loop can use uniform index comparisons
            # (cond: True==1, False==0)
            selector_var = f"{node.selector.get_name()}_scalar"
            if selector_var not in self.used_switch_selector:
                self.codegen_tensor_item(
                    torch.int64,
                    node.selector.codegen_reference(),
                    selector_var,
                )
                self.used_switch_selector.add(selector_var)
        else:
            # ShapeAsConstantBuffer yields a C++ bool/int expression; Cast to int64_t.
            selector_var = f"static_cast<int64_t>({node.selector.codegen_reference()})"

        def _emit_branch(header: str, branch) -> None:
        # [FUNCTION DOC] HELPER: Writes the branch header and the branch's subgraph code with device guard state saved.
            self.writeline(header)
            with self._preserve_device_guard_state():
                self.writeline(EnterSubgraphLine(self, branch.graph))
                self.codegen_subgraph(branch, outer_inputs, outer_outputs)
                self.writeline(ExitSubgraphLine(self))

        def emit_switch_body() -> None:
        # [FUNCTION DOC] HELPER: Emits the if/else chain over all branches, comparing the selector with each index.
            num_branches = len(node.branches)
            for b_idx, branch in enumerate(node.branches):
                if b_idx == 0:
                    header = f"if ({selector_var} == 0) {{"
                elif b_idx < num_branches - 1:
                    header = f"}} else if ({selector_var} == {b_idx}) {{"
                else:
                    header = "} else {"
                _emit_branch(header, branch)
            self.writeline("}")

        if not V.graph.is_dual_wrapper_mode:
            return emit_switch_body()

        graphs = tuple(branch.graph for branch in node.branches)
        jit_code = IndentedBuffer(initial_indent=self.wrapper_call._indent)
        with (
            self._preserve_codegen_state(graphs),
            self._target_buf("wrapper_call", jit_code),
            self.set_writeline(jit_code, jit_code.writeline_jit),
        ):
            # JIT pass: visit each branch once for lazy kernel autotune
            for branch in node.branches:
                self.writeline("{")
                with self._preserve_device_guard_state():
                    self.writeline(EnterSubgraphLine(self, branch.graph))
                    self.codegen_subgraph(branch, outer_inputs, outer_outputs)
                    self.writeline(ExitSubgraphLine(self))
                self.writeline("}")

        aot_code = AotOnlyBuffer(initial_indent=self.wrapper_call._indent)
        with (
            self._target_buf("wrapper_call", aot_code),
            self.set_writeline(aot_code, aot_code.writeline_aot),
        ):
            emit_switch_body()

        self.writeline(DualWrapperCodeLine(jit_code, aot_code))

    def codegen_subgraph(self, subgraph, outer_inputs, outer_outputs):
        # [FUNCTION DOC] Inlines a subgraph into the output: pushes it as the current codegened graph, writes a comment,
        # the input prefix, the subgraph body (via graph lowering under the subgraph graph handler) and the output suffix,
        # and always pops the codegened graph afterwards.
        # TODO (desertfire) - This function is the old way of supporting
        # subgraph codegen by inlining subgraphs in the output code. For python
        # wrapper, we have moved to lifting subgraphs as functions, supported by
        # PythonWrapperCode `codegen_subgraph` function. We should perhaps
        # support lifting of subgraphs as functions for cpp wrapper as well.
        try:
            self.push_codegened_graph(subgraph.graph)
            self.writeline(f"// subgraph: {subgraph.name}")
            self.codegen_subgraph_prefix(subgraph, outer_inputs, outer_outputs)
            parent_graph = V.graph
            with V.set_graph_handler(subgraph.graph):
                subgraph.graph.codegen_subgraph(
                    parent_graph=parent_graph,
                )
            self.codegen_subgraph_suffix(subgraph, outer_inputs, outer_outputs)
        finally:
            self.pop_codegened_graph()

    def codegen_while_loop(self, while_loop, stack_output=False):
        # [FUNCTION DOC] Emits a C++ loop for a while_loop node: declares the condition result (bool or tensor), copies
        # carried inputs into named handles, then loops evaluating the condition subgraph (breaking when false) and the
        # body subgraph. In dual-wrapper mode the JIT side visits condition and body once for autotune and the AOT side
        # gets the real loop. Raises NotImplementedError when stack_output is set.
        """Emit ABI-compatible C++ for a higher-order while_loop.

        In dual-wrapper mode, keep the real loop on the AOTI side while making
        the JIT first pass visit the condition and body once for lazy autotune.
        """
        if stack_output:
            raise NotImplementedError("NYI cpp wrapper for while_loop_stack_output")
        is_bool_pred = isinstance(
            while_loop.cond_subgraph.graph.graph_outputs[0], ir.ShapeAsConstantBuffer
        )
        name = while_loop.get_name()
        outer_carried_inputs = [
            buf.codegen_reference() for buf in while_loop.carried_inputs
        ]
        outer_additional_inputs = [
            buf.codegen_reference() for buf in while_loop.additional_inputs
        ]
        carried_output_names = self._get_while_loop_carried_output_names(while_loop)
        cond_result_name = f"{name}_cond_result"
        if is_bool_pred:
            self.writeline(f"bool {cond_result_name};")
        else:
            self.writeline(f"RAIIAtenTensorHandle {cond_result_name};")

        cond_outer_inputs = []
        for inp, out_name in zip(outer_carried_inputs, carried_output_names):
            # in ABI-compatible mode, the carried inputs are codegened
            # as buffers outside the while loop and set to the initial
            # values. at the end of each while_loop iteration, they
            # will be assigned the carried values.
            self.writeline(f"AtenTensorHandle {out_name}_handle;")
            self.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_assign_tensors_out({inp}, &{out_name}_handle));"
            )
            self.writeline(f"RAIIAtenTensorHandle {out_name}({out_name}_handle);")
            cond_outer_inputs.append(out_name)

        # additional inputs will be assigned within the while_loop
        # iteration directly from the corresponding outer graph buffers
        cond_outer_inputs.extend(outer_additional_inputs)

        cond_outer_outputs = [cond_result_name]
        body_outer_inputs = list(cond_outer_inputs)
        body_outer_outputs = body_outer_inputs[: len(outer_carried_inputs)]

        def codegen_original_while_loop() -> None:
        # [FUNCTION DOC] HELPER: Emits the actual infinite loop with condition check, break, and body subgraph code.
            self.writeline("while (1) {")
            with self._preserve_device_guard_state():
                self.writeline(EnterSubgraphLine(self, while_loop.cond_subgraph.graph))
                self.codegen_subgraph(
                    while_loop.cond_subgraph, cond_outer_inputs, cond_outer_outputs
                )

                if is_bool_pred:
                    cond_result = f"{cond_result_name}"
                else:
                    cond_result = f"{cond_result_name}_scalar"
                    self.codegen_tensor_item(torch.bool, cond_result_name, cond_result)
                self.writeline(f"if (!{cond_result}) break;")

                self.writeline(ExitSubgraphLine(self))
                self.writeline(EnterSubgraphLine(self, while_loop.body_subgraph.graph))
                self.codegen_subgraph(
                    while_loop.body_subgraph, body_outer_inputs, body_outer_outputs
                )
                self.writeline(ExitSubgraphLine(self))
                self.writeline("}")

        if not V.graph.is_dual_wrapper_mode:
            return codegen_original_while_loop()

        graphs = (
            while_loop.cond_subgraph.graph,
            while_loop.body_subgraph.graph,
        )
        jit_code = IndentedBuffer(initial_indent=self.wrapper_call._indent)
        with (
            self._preserve_codegen_state(graphs),
            self._target_buf("wrapper_call", jit_code),
            self.set_writeline(jit_code, jit_code.writeline_jit),
        ):
            self.writeline("{")
            with self._preserve_device_guard_state():
                self.writeline(EnterSubgraphLine(self, while_loop.cond_subgraph.graph))
                self.codegen_subgraph(
                    while_loop.cond_subgraph, cond_outer_inputs, cond_outer_outputs
                )
                self.writeline(ExitSubgraphLine(self))
                self.writeline(EnterSubgraphLine(self, while_loop.body_subgraph.graph))
                self.codegen_subgraph(
                    while_loop.body_subgraph, body_outer_inputs, body_outer_outputs
                )
                self.writeline(ExitSubgraphLine(self))
            self.writeline("}")

        aot_code = AotOnlyBuffer(initial_indent=self.wrapper_call._indent)
        with (
            self._target_buf("wrapper_call", aot_code),
            self.set_writeline(aot_code, aot_code.writeline_aot),
        ):
            codegen_original_while_loop()

        self.writeline(DualWrapperCodeLine(jit_code, aot_code))

    def generate_extern_kernel_args_decl_if_needed(
        self,
        op_overload: torch._ops.OpOverload | torch._ops.HigherOrderOperator,
        raw_args: Sequence[Any],
        output_args: _OUTPUT_ARGS_TYPE,
        raw_outputs: Sequence[ir.Buffer],
    ):
        # [FUNCTION DOC] Classifies the arguments and outputs of an extern kernel call using the op schema (torchbind
        # calls use the schema from the object and method) into tensor argument names and integer argument strings,
        # declaring uninitialized output tensor handles for non-mutated tensor outputs. Returns (tensor args, int args).
        # Raises AssertionError for unexpected types or unsupported fall-through types and NotImplementedError for
        # unsupported return types (SymInt outputs).
        """
        Generates declarations for external kernel arguments if needed, based on the provided
        operator and its arguments. It processes both input and output arguments, categorizing
        them into tensor and integer arguments for further code generation.
        """
        schema = None
        if isinstance(op_overload, torch._higher_order_ops.torchbind.CallTorchBind):
            obj = raw_args[0]
            method = raw_args[1]
            schema = op_overload.schema(obj, method)
        else:
            if not isinstance(op_overload, torch._ops.OpOverload):
                raise AssertionError(type(op_overload))
            schema = op_overload._schema
        if schema is None:
            raise AssertionError("expected schema to be set")
        arg_types = [x.real_type for x in schema.arguments]
        return_types = [x.type for x in schema.returns]

        new_tensor_args = []
        new_int_args = []

        def fill_args(arg, arg_type):
        # [FUNCTION DOC] HELPER: Appends one argument to the tensor or int argument lists by its schema type (tensors,
        # ints, symints, int scalars and lists of them); static types and torchbind objects add nothing. Raises
        # AssertionError for unexpected values or unsupported types.
            static_arg_types = (
                torch.FloatType,
                torch.BoolType,
                torch.StringType,
                torch.Type,
                torch.DeviceObjType,
            )
            inductor_tensor_buffers = (
                ir.Buffer,
                ir.ReinterpretView,
            )

            if isinstance(arg_type, torch.TensorType):
                if not isinstance(arg, inductor_tensor_buffers):
                    raise AssertionError(f"got {type(arg)}")
                new_tensor_args.append(f"{arg.codegen_reference()}")
            elif isinstance(arg_type, torch.IntType):
                # int
                new_int_args.append(str(arg))
            elif isinstance(arg_type, torch.SymIntType):
                # SymInt
                expr = arg.node.expr if isinstance(arg, torch.SymInt) else arg
                new_int_args.append(cexpr(expr))
            elif isinstance(arg_type, torch.NumberType):
                # Scalar of type int
                if not isinstance(arg, (int, float, bool)):
                    raise AssertionError(
                        f"expected arg to be int, float, or bool, got {type(arg)}"
                    )
                # Only treat int Scalar as dynamic
                if isinstance(arg, int):
                    new_int_args.append(str(arg))
            elif isinstance(arg, ir.TorchBindObject):
                # torchbind objects are loaded in proxy executor
                pass
            elif isinstance(arg_type, torch.ListType):
                if not isinstance(arg, (list, tuple)):
                    raise AssertionError(
                        f"expected arg to be list or tuple, got {type(arg)}"
                    )

                # List[Tensor]
                if isinstance(arg_type.getElementType(), torch.TensorType):
                    new_tensor_args.extend([f"{a.codegen_reference()}" for a in arg])
                # List[Optional[Tensor]]
                elif isinstance(
                    arg_type.getElementType(), torch.OptionalType
                ) and isinstance(
                    arg_type.getElementType().getElementType(), torch.TensorType
                ):
                    new_tensor_args.extend(
                        [f"{a.codegen_reference()}" for a in arg if a is not None]
                    )
                # List[int]
                elif isinstance(arg_type.getElementType(), torch.IntType):
                    new_int_args.extend([str(a) for a in arg])
                # List[SymInt]
                elif isinstance(arg_type.getElementType(), torch.SymIntType):
                    expressions = [
                        a.node.expr if isinstance(a, torch.SymInt) else a for a in arg
                    ]
                    new_int_args.extend([cexpr(expr) for expr in expressions])
                # List[Scalar]
                elif isinstance(arg_type.getElementType(), torch.NumberType):
                    # Only treat int Scalar as dynamic
                    is_int_type = [isinstance(a, int) for a in arg]
                    if any(is_int_type):
                        if not all(is_int_type):
                            raise AssertionError(
                                "AOTInductor only supports int scalars of the same type"
                            )
                        new_int_args.extend([str(a) for a in arg])
                else:
                    if not isinstance(
                        arg_type.getElementType(),
                        static_arg_types,  # type: ignore[arg-type]
                    ):
                        raise AssertionError(
                            f"Fall through arguments must be one of static_arg_types, got {type(arg_type)}"
                        )
            else:
                if not isinstance(
                    arg_type,
                    static_arg_types,  # type: ignore[arg-type]
                ):
                    raise AssertionError(
                        f"Fall through arguments must be one of static_arg_types, got {type(arg_type)}"
                    )

        for arg, arg_type in zip(raw_args, arg_types):
            if arg is not None:
                if isinstance(arg_type, torch.OptionalType):
                    fill_args(arg, arg_type.getElementType())
                else:
                    fill_args(arg, arg_type)

        def fill_output_arg(
            arg: str, return_type: torch.JitType, is_mutated_output: bool
        ) -> None:
        # [FUNCTION DOC] HELPER: For a tensor return type, declares a new uninitialized output handle (unless mutated)
        # and records it as a tensor arg; symint returns raise NotImplementedError, others raise AssertionError.
            if isinstance(return_type, torch.TensorType):
                if not is_mutated_output:
                    self.writeline(f"AtenTensorHandle {arg}_handle;  // output buffer")
                    self.writeline(
                        f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_new_uninitialized_tensor(&{arg}_handle));"
                    )
                    self.writeline(f"RAIIAtenTensorHandle {arg}({arg}_handle);")
                new_tensor_args.append(f"{arg}")
            elif isinstance(return_type, torch.SymIntType):
                raise NotImplementedError("NYI support for return type: SymInt")
            elif isinstance(return_type, torch.ListType) and isinstance(
                return_type.getElementType(), torch.SymIntType
            ):
                raise NotImplementedError("NYI support for return type: List[SymInt]")
            else:
                raise AssertionError(f"Unsupported return type found: {return_type}")

        # TODO: Only support None and tensor(s) returns for now, SymInt is not implemented yet
        for return_type in return_types:
            if isinstance(
                return_type, (torch.TensorType, torch.NoneType, torch.IntType)
            ):
                pass
            elif isinstance(return_type, torch.OptionalType):
                if not isinstance(return_type.getElementType(), torch.TensorType):
                    raise AssertionError(
                        f"expected Optional element type to be TensorType, "
                        f"got {return_type.getElementType()}"
                    )
            elif isinstance(return_type, torch.ListType):
                if not isinstance(return_type.getElementType(), torch.TensorType):
                    raise AssertionError(
                        f"expected List element type to be TensorType, "
                        f"got {return_type.getElementType()}"
                    )
            else:
                raise NotImplementedError(
                    f"return type {return_type} is not yet supported."
                )

        for output_arg, raw_output_arg in zip(output_args, raw_outputs):  # type: ignore[arg-type]
            # None output is supported, but Optional return types are not yet supported
            if output_arg is None:
                continue
            elif isinstance(raw_output_arg, int):
                new_int_args.append(str(raw_output_arg))
            elif isinstance(output_arg, list):
                for out in output_arg:
                    if out is None:
                        raise AssertionError(out)
                    fill_output_arg(
                        out,
                        torch.TensorType.get(),
                        isinstance(raw_output_arg, ir.MutationOutput),
                    )
            else:
                fill_output_arg(
                    output_arg,
                    torch.TensorType.get(),
                    isinstance(raw_output_arg, ir.MutationOutput),
                )

        return new_tensor_args, new_int_args

    @staticmethod
    def _compatible_with_stableivalue(op: torch._ops.OpOverload) -> bool:
        # [FUNCTION DOC] HELPER: Returns true if every argument and return of the op schema uses only types handled by
        # the C shim's internal to_ivalue (bool, device, float, int, tensor, optional of those) and none involve symbolic
        # ints, bools or floats (which need the boxed dispatch path).
        """Returns true if op_overload._schema only utilizes types supported by the AOT
        C-shim *internal* function to_ivalue.  to_ivalue is an implementation detail, so
        these types are not guaranteed to be supported long-term.  When generating code
        for cpp_wrapper mode, we don't have to be forward-compatible, so changing this
        function's implementation in future is fine."""
        supported_types = (
            torch.BoolType,
            torch.DeviceObjType,
            torch.FloatType,
            # ScalarTypeType, LayoutType, and MemoryFormatType are seen as IntType
            # when queried via torch.JitType.type.
            torch.IntType,
            torch.TensorType,
        )

        def type_supported(t: torch.JitType) -> bool:
        # [FUNCTION DOC] HELPER: Returns true if the type, unwrapping optionals, is a supported shim type.
            if isinstance(t, torch.OptionalType):
                return type_supported(t.getElementType())
            return isinstance(t, supported_types)

        def uses_symint(t: torch.JitType) -> bool:
        # [FUNCTION DOC] HELPER: Returns true if the type (through optional or list) is symbolic int, bool or float.
            # SymInt/SymBool/SymFloat are reported as Int/Bool/Float by
            # JitType.type, so they pass type_supported above.  But the
            # StableIValue codegen below emits no symbolic-int-aware conversion,
            # so route such ops to the boxed dispatch path, which handles
            # c10::SymInt correctly.  real_type preserves the symbolic types.
            if isinstance(t, (torch.OptionalType, torch.ListType)):
                return uses_symint(t.getElementType())
            return (
                isinstance(t, (torch.SymIntType, torch.SymBoolType))
                or repr(t) == "SymFloat"
            )

        return all(
            type_supported(a.type) and not uses_symint(a.real_type)
            for a in chain(op._schema.arguments, op._schema.returns)
        )

    @staticmethod
    def _compatible_with_cpp_boxed_dispatch(op: torch._ops.OpOverload) -> bool:
        # [FUNCTION DOC] HELPER: Returns true if all arguments (including nested optional/list/tuple) and returns
        # (none, tensor, optional tensor) of the schema can be boxed into IValues directly by the JIT cpp wrapper.
        """Returns true if cpp_wrapper JIT can directly codegen c10::IValue boxing
        for this schema and can unpack the returned values into tensor handles.

        Unlike the StableIValue path, this is only used for JIT cpp_wrapper, where
        we can include ATen/c10 headers and call the dispatcher directly.
        """

        def arg_supported(t: torch.JitType) -> bool:
        # [FUNCTION DOC] HELPER: Recursive predicate for supported argument types.
            if isinstance(t, torch.OptionalType):
                return arg_supported(t.getElementType())
            if isinstance(t, torch.ListType):
                return arg_supported(t.getElementType())
            if isinstance(t, torch.TupleType):
                return all(arg_supported(e) for e in t.elements())
            return isinstance(
                t,
                (
                    torch.BoolType,
                    torch.DeviceObjType,
                    torch.FloatType,
                    torch.IntType,
                    torch.NumberType,
                    torch.StringType,
                    torch.SymBoolType,
                    torch.SymIntType,
                    torch.TensorType,
                ),
            ) or repr(t) in ("Layout", "MemoryFormat", "ScalarType", "SymFloat")

        def return_supported(t: torch.JitType) -> bool:
        # [FUNCTION DOC] HELPER: Predicate for supported return types: none, tensor, or optional tensor.
            if isinstance(t, torch.OptionalType):
                return isinstance(t.getElementType(), torch.TensorType)
            return isinstance(t, (torch.NoneType, torch.TensorType))

        return all(arg_supported(a.real_type) for a in op._schema.arguments) and all(
            return_supported(r.real_type) for r in op._schema.returns
        )

    def generate_fallback_kernel_with_runtime_lookup(
        self,
        buf_name: str,
        python_kernel_name: str,
        get_args: Callable[[], Sequence[str]],
        op_overload: torch._ops.OpOverload | torch._ops.HigherOrderOperator,
        raw_args: Sequence[Any],
        outputs: Sequence[ir.Buffer],
    ) -> None:
        # [FUNCTION DOC] Emits a call to a kernel that is not in the C shim, found at runtime. Computes output variable names
        # mirroring the outputs structure (raises AssertionError for odd mutation outputs or unexpected types), then in
        # AOT mode uses the proxy executor, in JIT mode the dispatcher or Python callback; dual-wrapper mode emits both
        # versions into a DualWrapperCodeLine. Torchbind higher-order ops take their schema from the object and method.
        """Generate a call to a kernel not contained in the C-shim.  This results in
        different code paths for AOT Inductor vs cpp_wrapper Inductor mode."""

        def extract_output_name(
            out: ir.Buffer | Sequence[ir.Buffer] | None,
        ) -> str | None | _OUTPUT_ARGS_TYPE:
        # [FUNCTION DOC] HELPER: Maps a kernel output object to its C++ variable name(s) with the same nesting: None stays
        # None, multi-output/collective/fallback nodes give their name, a mutation output gives its single mutated buffer
        # name, lists recurse, ints become strings. Raises AssertionError for a multi-name mutation or unknown types.
            if out is None:
                return None
            if isinstance(
                out, (ir.MultiOutput, ir._CollectiveKernel, ir.FallbackKernel)
            ):
                return out.get_name()
            if isinstance(out, ir.MutationOutput):
                mutated_buf_names = out.get_mutation_names()
                if not (
                    isinstance(mutated_buf_names, list) and len(mutated_buf_names) == 1
                ):
                    raise AssertionError(
                        "Expect only one mutated buffer in MutationOutput"
                    )
                return mutated_buf_names[0]
            if isinstance(out, (list, tuple)):
                return [extract_output_name(o) for o in out]  # type: ignore[misc]
            if isinstance(out, int):
                return str(out)
            raise AssertionError(f"Unexpected output: {type(out)}")

        if isinstance(op_overload, torch._ops.HigherOrderOperator):
            if not isinstance(
                op_overload, torch._higher_order_ops.torchbind.CallTorchBind
            ):
                raise AssertionError(type(op_overload))
            if len(raw_args) <= 1:
                raise AssertionError(f"expected len(raw_args) > 1, got {len(raw_args)}")
            obj = raw_args[0]
            method = raw_args[1]
            return_schema = op_overload.schema(obj, method).returns
        else:
            return_schema = op_overload._schema.returns

        # output_args has the same pytree structure as outputs
        if not return_schema:
            # kernel does not return a value
            output_args: _OUTPUT_ARGS_TYPE = []
        elif isinstance(output_name := extract_output_name(outputs), str):
            output_args = [output_name]
        else:
            # If the schema indicates a return value, we should have a non-None value by
            # this point.
            if not isinstance(output_name, list):
                raise AssertionError(type(output_name))
            output_args = output_name

        # In AOT mode, we use a ProxyExecutor to run fallback kernels.
        if V.graph.aot_mode and V.graph.is_dual_wrapper_mode:
            jit_code = IndentedBuffer(initial_indent=self.wrapper_call._indent)
            with (
                self._target_buf("wrapper_call", jit_code),
                self.set_writeline(jit_code, jit_code.writeline_jit),
            ):
                self._generate_fallback_kernel_jit(
                    buf_name,
                    python_kernel_name,
                    get_args,
                    op_overload,
                    raw_args,
                    output_args,
                    outputs,
                )

            aot_code = AotOnlyBuffer(initial_indent=self.wrapper_call._indent)
            with (
                self._target_buf("wrapper_call", aot_code),
                self.set_writeline(aot_code, aot_code.writeline_aot),
            ):
                self.generate_fallback_kernel_with_runtime_lookup_aot(
                    op_overload,
                    raw_args,
                    output_args,
                    outputs,
                )

            self.writeline(DualWrapperCodeLine(jit_code, aot_code))
            return

        if V.graph.aot_mode:
            self.generate_fallback_kernel_with_runtime_lookup_aot(
                op_overload,
                raw_args,
                output_args,
                outputs,
            )

        if not V.graph.aot_mode:
            self._generate_fallback_kernel_jit(
                buf_name,
                python_kernel_name,
                get_args,
                op_overload,
                raw_args,
                output_args,
                outputs,
            )

    def _generate_fallback_kernel_jit(
        self,
        buf_name,
        python_kernel_name,
        get_args,
        op_overload,
        raw_args,
        output_args,
        outputs,
    ):
        # [FUNCTION DOC] HELPER: Emits a JIT-mode fallback kernel call, choosing the first path that supports the op schema:
        # the stable-ivalue dispatcher without Python, then direct boxed c10 dispatch, else a callback into Python.
        # Raises AssertionError if op_overload is not an OpOverload or outputs are not simple names.
        """Emit JIT-mode fallback kernel code (no-Python dispatcher or Python callback)."""
        if not isinstance(op_overload, torch._ops.OpOverload):
            raise AssertionError(type(op_overload))
        for output in output_args:
            if not (output is None or isinstance(output, str)):
                raise AssertionError(
                    "fallback kernels with runtime lookup currently only support tensor "
                    "returns, not more complicated types (such as list-of-list-of-tensor)"
                )

        # In non-AOT mode, we use aoti_torch_call_dispatcher if all the inputs and
        # outputs of the op can be represented with StableIValue.  This avoids the
        # overhead of calling back into Python, and covers most remaining fallback ops.
        if self._compatible_with_stableivalue(op_overload):
            self.generate_fallback_kernel_with_runtime_lookup_nopython(
                get_args,
                op_overload,
                output_args,  # type: ignore[arg-type]
                outputs,
            )
            return

        # For cpp_wrapper JIT, fall back to direct boxed dispatcher calls for
        # schemas the StableIValue bridge cannot represent, such as Tensor[].
        if self._compatible_with_cpp_boxed_dispatch(op_overload):
            self.generate_fallback_kernel_with_runtime_lookup_boxed(
                op_overload,
                raw_args,
                output_args,  # type: ignore[arg-type]
                outputs,
            )
            return

        # Otherwise, we call back into Python, which has some extra runtime overhead,
        # but handles situations like list[Tensor] (currently unrepresentable via
        # StableIValue).
        self.generate_fallback_kernel_with_runtime_lookup_python(
            buf_name,
            python_kernel_name,
            op_overload,
            raw_args,
            output_args,  # type: ignore[arg-type]
            outputs,
        )

    def generate_scoped_gil_acquire(self, declarations_before_scope, lines_in_scope):
        # [FUNCTION DOC] Builds the lines for a block that declares variables then runs the given code under a scoped Python
        # GIL acquisition, and returns the list of line strings (not written).
        scoped_lines = IndentedBuffer()
        for declaration in declarations_before_scope:
            scoped_lines.writeline(declaration)

        scoped_lines.writeline("{")
        with scoped_lines.indent():
            scoped_lines.writeline("py::gil_scoped_acquire_simple acquire;")
            scoped_lines.writelines(lines_in_scope.split("\n"))
        scoped_lines.writelines("}")
        return scoped_lines._lines

    def load_custom_op_wrapper(self):
        # [FUNCTION DOC] Emits, once per wrapper, a GIL-scoped block that imports the inductor codecache module and fetches its
        # custom_op_wrapper into a declared holder variable, throwing in C++ if either step fails. Sets a flag.
        # KNOWN ISSUE: a TODO notes this once-only flag does not account for control flow.
        # TODO: need to support control flow
        if self.custom_op_wrapper_loaded:
            return

        lines = """
RAIIPyObject codecache_module(PyImport_ImportModule("torch._inductor.codecache"));
if (!codecache_module) {
    throw std::runtime_error("Failed to load torch._inductor.codecache");
}
custom_op_wrapper = PyObject_GetAttrString(codecache_module, "custom_op_wrapper");
if (!custom_op_wrapper) {
    throw std::runtime_error("Failed to load torch._inductor.codecache.custom_op_wrapper");
}"""

        declarations_before_scope = ["RAIIPyObject custom_op_wrapper;"]
        scope_gil_acquire = self.generate_scoped_gil_acquire(
            declarations_before_scope, lines
        )
        self.writelines(scope_gil_acquire)

        self.custom_op_wrapper_loaded = True

    def generate_float_value(self, val):
        # [FUNCTION DOC] Returns a C++ expression for a float constant, using numeric_limits for infinity and NaN.
        # Raises AssertionError if the value is not a float.
        if not isinstance(val, float):
            raise AssertionError(f"expected val to be float, got {type(val)}")
        if val == float("inf"):
            return "std::numeric_limits<double>::infinity()"
        elif val == float("-inf"):
            return "-std::numeric_limits<double>::infinity()"
        elif math.isnan(val):
            return "std::numeric_limits<double>::quiet_NaN()"
        else:
            return f"{val}"

    def generate_py_arg(self, py_args_var, idx, raw_arg, arg_type):
        # [FUNCTION DOC] Returns C++ lines that convert one operator argument to a Python object and store it at the given
        # index of the Python argument tuple. List arguments (also optional lists) are built into a Python list first.
        # Raises NotImplementedError for unsupported argument types.
        """Generate C++ code that converts a single operator argument into a Python
        object and inserts it into a PyTuple at the given index."""

        def generate_py_arg_inner(lines, raw_arg, arg_type):
        # [FUNCTION DOC] HELPER: Returns a C++ expression creating the Python object for one argument by schema type (or
        # python type for the any type): None (with refcount increment), tensors as capsules of the handle pointer, bool, int,
        # symint, float, string, numbers, device, dtype, layout, memory format. May append setup lines.
        # Raises NotImplementedError for unsupported types.
        # KNOWN ISSUE: strings are embedded in quotes without escaping.
            def handle_scalar(scalar):
            # [FUNCTION DOC] HELPER: Returns the C++ constructor expression for a Python bool, int, float, complex or symbolic
            # scalar. Raises NotImplementedError for other values.
                if isinstance(scalar, bool):
                    return f"PyBool_FromLong({1 if scalar else 0})"
                if isinstance(scalar, int):
                    return f"PyLong_FromLongLong({scalar})"
                if isinstance(scalar, float):
                    return f"PyFloat_FromDouble({self.generate_float_value(scalar)})"
                if isinstance(scalar, complex):
                    real = self.generate_float_value(scalar.real)
                    imag = self.generate_float_value(scalar.imag)
                    return f"PyComplex_FromDoubles({real}, {imag})"
                if isinstance(scalar, SymTypes):
                    scalar_var = cexpr(scalar.node.expr)
                    if isinstance(scalar, torch.SymBool):
                        return f"PyBool_FromLong({scalar_var})"
                    if isinstance(scalar, torch.SymFloat):
                        return f"PyFloat_FromDouble({scalar_var})"
                    return f"PyLong_FromLongLong({scalar_var})"
                raise NotImplementedError(
                    f"scalar {scalar}, {type(scalar)} cannot be handled by handle_scalar"
                )

            is_any = isinstance(arg_type, torch.AnyType)

            def handle_any(types: type | tuple[type, ...]):
            # [FUNCTION DOC] HELPER: True when the schema type is the any type and the raw argument has one of the given types.
                return is_any and isinstance(raw_arg, types)

            if raw_arg is None:
                # Py_None is a singleton, so we have to explicitly incref it here
                lines.append("Py_INCREF(Py_None);\n")
                return "Py_None"
            elif isinstance(arg_type, torch.TensorType) or handle_any(ir.IRNode):
                # In some cases, scalar arguments may be passed in place of tensors.
                if not hasattr(raw_arg, "codegen_reference"):
                    return handle_scalar(raw_arg)

                # Store AtenTensorHandle as void*.  All Python args are constructed in a
                # nested scope, so this handle will self-destruct after the function
                # call.
                base_handle = self.create_tmp_raii_handle_var_if_needed(
                    raw_arg.codegen_reference(), lines
                )
                return f"PyCapsule_New(reinterpret_cast<void*>({base_handle}.get()), NULL, NULL)"
            elif isinstance(arg_type, torch.OptionalType):
                return generate_py_arg_inner(lines, raw_arg, arg_type.getElementType())
            elif isinstance(arg_type, torch.BoolType) or handle_any(bool):
                return f"PyBool_FromLong({1 if raw_arg else 0})"
            elif isinstance(arg_type, torch.IntType) or handle_any(int):
                # int
                return f"PyLong_FromLongLong({raw_arg})"
            elif isinstance(arg_type, torch.SymIntType) or handle_any(torch.SymInt):
                # SymInt
                expr = (
                    raw_arg.node.expr if isinstance(raw_arg, torch.SymInt) else raw_arg
                )
                return f"PyLong_FromLongLong({cexpr(expr)})"
            elif isinstance(arg_type, torch.FloatType) or handle_any(float):
                return f"PyFloat_FromDouble({self.generate_float_value(raw_arg)})"
            elif isinstance(arg_type, torch.StringType) or handle_any(str):
                return f'PyUnicode_FromString("{raw_arg}")'
            elif isinstance(arg_type, torch.NumberType) or handle_any(
                (*SymTypes, torch.types.Number, complex)
            ):
                # Union[bool, int, float, complex]
                # torch/_prims_common/__init__.py
                return handle_scalar(raw_arg)
            elif isinstance(raw_arg, torch.device):
                device_str, device_index = self.codegen_device(raw_arg).split(", ")
                return f"torch_python_thp_device_new({device_str}, {device_index})"
            elif isinstance(raw_arg, torch.dtype):
                return f"torch_python_get_thp_dtype({self.codegen_dtype(raw_arg)})"
            elif isinstance(raw_arg, torch.layout):
                return f"torch_python_get_thp_layout({self.codegen_layout(raw_arg)})"
            elif isinstance(raw_arg, torch.memory_format):
                return (
                    "torch_python_get_thp_memory_format("
                    f"{self.codegen_memory_format(raw_arg)})"
                )
            else:
                raise NotImplementedError(
                    f"arg type {arg_type} is not yet supported by custom_op_wrapper"
                )

        def handle_sequence_arg(raw_arg_, arg_type_, lines_):
        # [FUNCTION DOC] HELPER: Appends lines that create a Python list for a sequence argument, set each converted element
        # and put the list into the argument tuple. Raises AssertionError if the value is not a list or tuple.
        # KNOWN ISSUE: length and error message use the outer raw_arg rather than the raw_arg_ parameter (same object today).
            if not isinstance(raw_arg_, (list, tuple)):
                raise AssertionError(str(raw_arg) + " is not a list")
            lines_.append(
                f"PyObject* {py_args_var}_{idx} = PyList_New({len(raw_arg)});\n"
            )
            for i, elem in enumerate(raw_arg_):
                lines_.append(
                    f"PyList_SetItem({py_args_var}_{idx}, {i}, {generate_py_arg_inner(lines, elem, arg_type_.getElementType())});\n"
                )
            lines_.append(
                f"PyTuple_SetItem({py_args_var}, {idx}, {py_args_var}_{idx});\n"
            )

        lines = []
        if isinstance(arg_type, torch.ListType):
            handle_sequence_arg(raw_arg, arg_type, lines)
        elif isinstance(arg_type, torch.OptionalType) and isinstance(
            arg_type.getElementType(), torch.ListType
        ):
            if raw_arg is None:
                lines.append(
                    f"PyTuple_SetItem({py_args_var}, {idx}, {generate_py_arg_inner(lines, raw_arg, arg_type)});\n"
                )
            else:
                handle_sequence_arg(raw_arg, arg_type.getElementType(), lines)
        else:
            lines.append(
                f"PyTuple_SetItem({py_args_var}, {idx}, {generate_py_arg_inner(lines, raw_arg, arg_type)});\n"
            )
        return "".join(lines)

    def generate_fallback_kernel_with_runtime_lookup_nopython(
        self,
        get_args: Callable[[], Sequence[str]],
        op_overload: torch._ops.OpOverload,
        output_args: Sequence[str | None],
        raw_outputs: Sequence[ir.Buffer],
    ) -> None:
        # [FUNCTION DOC] Emits a JIT fallback call using the stable-ivalue dispatcher without Python: declares RAII output
        # handles (skipping mutation outputs and None), converts each argument to a StableIValue (optionals, temporary
        # handle unwrapping, new handles for tensors that are not temporaries), calls the dispatcher by schema name and
        # overload, and converts results back to tensor handles. Writes the block via writelines.
        """Generate fallback kernel calls with runtime (non-AOT) dispatch.  This can
        only be called in cpp_wrapper mode, and assumes that the input is a non-None
        OpOverload.

        In the future, we may switch over to directly calling c10::Dispatcher if we need
        to support more datatypes."""
        if raw_outputs:
            declarations_before_scope = [
                f"RAIIAtenTensorHandle {output_arg};"
                for output_arg, raw_output_arg in zip(output_args, raw_outputs)  # type: ignore[arg-type]
                if output_arg is not None
                and not isinstance(raw_output_arg, ir.MutationOutput)
            ]
        else:
            declarations_before_scope = [
                f"RAIIAtenTensorHandle {output_arg};"
                for output_arg in output_args  # type: ignore[arg-type]
                if output_arg is not None
            ]

        dispatch_lines = IndentedBuffer()
        dispatch_lines.writelines(declarations_before_scope)
        dispatch_lines.writeline("{")

        with dispatch_lines.indent():
            tmp_var_number = count()

            def parse_arg(arg_type: torch.JitType, codegen_arg: str) -> str:
                # [FUNCTION DOC] HELPER: Converts one generated argument string to a stable-ivalue conversion expression, stripping
                # temporary-reference wrappers, handling optionals via a temporary optional variable, and tensors by creating
                # or releasing handles as needed. Writes helper lines into dispatch_lines.
                # Strip off any temporary references; we're in an indented context, so
                # any saved-off variables will be auto-destroyed.
                new_codegen_arg = codegen_arg.removeprefix("&temporary_reference(")
                if new_codegen_arg != codegen_arg:
                    # If we removed temporary_reference, there's a good chance the
                    # variable ends with get() (which would retrieve an ATenTensorHandle
                    # from a temporary RAII handle).  Strip that off too, since we're
                    # going to save this in a temporary RAII handle.
                    if codegen_arg.endswith(".get())"):
                        codegen_arg = new_codegen_arg.removesuffix(".get())")
                    else:
                        codegen_arg = new_codegen_arg.removesuffix(")")

                if isinstance(arg_type, torch.OptionalType):
                    # If we have a pointer to a variable, strip it off and let
                    # from<std::optional> handle any internal pointers.
                    codegen_arg = codegen_arg.removeprefix("&")

                    if codegen_arg == "nullptr":
                        return "torch::stable::detail::from(std::nullopt)"

                    var_name = f"tmp_var_{next(tmp_var_number)}"
                    dispatch_lines.writeline(
                        f"std::optional {var_name}{{{parse_arg(arg_type.getElementType(), codegen_arg)}}};"
                    )
                    return f"torch::stable::detail::from({var_name})"

                raii_var = self.create_tmp_raii_handle_var_if_needed(
                    codegen_arg, dispatch_lines
                )
                temp_handle = raii_var != codegen_arg

                if isinstance(arg_type, torch.TensorType):
                    if not temp_handle:
                        # If the RAII tensor being referenced _isn't_ a temporary,
                        # scoped to this fallback call, then create a new handle
                        # referencing it which from<AtenTensorHandle> can steal.
                        var_name = f"tmp_var_{next(tmp_var_number)}"
                        dispatch_lines.writeline(f"AtenTensorHandle {var_name};")
                        dispatch_lines.writeline(
                            "AOTI_TORCH_ERROR_CODE_CHECK("
                            f"aoti_torch_new_tensor_handle({raii_var}, &{var_name}));"
                        )
                        return f"torch::stable::detail::from({var_name})"
                    # If the RAII tensor _is_ a temporary scoped to this fallback call,
                    # simply release and steal the handle.
                    return f"torch::stable::detail::from({raii_var}.release())"
                return f"torch::stable::detail::from({codegen_arg})"

            codegen_args = get_args()
            ivalue_args = (
                parse_arg(a.type, c)
                for a, c in zip(op_overload._schema.arguments, codegen_args)
            )
            array_len = max(len(codegen_args), len(output_args))
            dispatch_lines.writeline(
                f"std::array<StableIValue, {array_len}> dispatch_vars{{{', '.join(ivalue_args)}}};"
            )
            dispatch_lines.writeline("AOTI_TORCH_ERROR_CODE_CHECK(")
            with dispatch_lines.indent():
                dispatch_lines.writeline(
                    f'aoti_torch_call_dispatcher("{op_overload._schema.name}", "{op_overload._schema.overload_name}", dispatch_vars.data())'
                )
            dispatch_lines.writeline(");")

            # assign result(s), ignoring None
            for idx, output_arg in enumerate(output_args):
                if output_arg is None:
                    continue
                dispatch_lines.writeline(
                    f"{output_arg} = torch::stable::detail::to<AtenTensorHandle>(dispatch_vars[{idx}]);"
                )

        dispatch_lines.writeline("}")
        self.writelines(dispatch_lines.getvalue().splitlines())

    def generate_fallback_kernel_with_runtime_lookup_boxed(
        self,
        op_overload: torch._ops.OpOverload,
        raw_args: Sequence[Any],
        output_args: Sequence[str | None],
        raw_outputs: Sequence[ir.Buffer],
    ) -> None:
        # [FUNCTION DOC] Emits a JIT-only fallback call through the c10 dispatcher with boxed arguments: adds the needed
        # includes, declares RAII output handles, builds a stack of IValues from each argument by schema type (tensors,
        # lists, tuples, scalars, symbolic values, devices, dtypes, layouts, memory formats), looks up the op once by
        # name, calls it boxed, and converts non-mutated tensor results to handles. Raises AssertionError for unsupported types.
        """Generate a JIT-only fallback call through c10::Dispatcher.

        This covers custom C++ ops whose schemas are outside StableIValue's
        supported subset without routing through Python.
        """
        self.include_extra_header("ATen/core/dispatch/Dispatcher.h")
        self.include_extra_header("ATen/core/ivalue.h")
        self.include_extra_header("ATen/core/jit_type.h")
        self.include_extra_header("torch/csrc/inductor/aoti_torch/utils.h")

        if raw_outputs:
            declarations_before_scope = [
                f"RAIIAtenTensorHandle {output_arg};"
                for output_arg, raw_output_arg in zip(output_args, raw_outputs)
                if output_arg is not None
                and not isinstance(raw_output_arg, ir.MutationOutput)
            ]
        else:
            declarations_before_scope = [
                f"RAIIAtenTensorHandle {output_arg};"
                for output_arg in output_args
                if output_arg is not None
            ]

        dispatch_lines = IndentedBuffer()
        dispatch_lines.writelines(declarations_before_scope)
        dispatch_lines.writeline("{")

        with dispatch_lines.indent():
            tmp_var_number = count()

            def next_tmp(prefix: str) -> str:
            # [FUNCTION DOC] HELPER: Returns a fresh unique temporary variable name with the given prefix.
                return f"{prefix}_{next(tmp_var_number)}"

            def type_expr(t: torch.JitType) -> str:
            # [FUNCTION DOC] HELPER: Returns a C++ expression for the c10 type matching a JIT type (optional, list, tuple,
            # primitives, tensor and special named types), emitting helper vector declarations for tuples. Raises
            # AssertionError for unsupported types.
                if isinstance(t, torch.OptionalType):
                    return f"c10::OptionalType::create({type_expr(t.getElementType())})"
                if isinstance(t, torch.ListType):
                    return f"c10::ListType::create({type_expr(t.getElementType())})"
                if isinstance(t, torch.TupleType):
                    elems = ", ".join(type_expr(e) for e in t.elements())
                    var_name = next_tmp("tmp_type")
                    dispatch_lines.writeline(
                        f"std::vector<c10::TypePtr> {var_name}{{{elems}}};"
                    )
                    return f"c10::TupleType::create(std::move({var_name}))"
                if isinstance(t, torch.BoolType):
                    return "c10::BoolType::get()"
                if isinstance(t, torch.DeviceObjType):
                    return "c10::DeviceObjType::get()"
                if isinstance(t, torch.FloatType):
                    return "c10::FloatType::get()"
                if isinstance(t, torch.IntType):
                    return "c10::IntType::get()"
                if isinstance(t, torch.NumberType):
                    return "c10::NumberType::get()"
                if isinstance(t, torch.StringType):
                    return "c10::StringType::get()"
                if isinstance(t, torch.SymBoolType):
                    return "c10::SymBoolType::get()"
                if isinstance(t, torch.SymIntType):
                    return "c10::SymIntType::get()"
                if isinstance(t, torch.TensorType):
                    return "c10::TensorType::get()"
                type_repr = repr(t)
                if type_repr == "Layout":
                    return "c10::LayoutType::get()"
                if type_repr == "MemoryFormat":
                    return "c10::MemoryFormatType::get()"
                if type_repr == "ScalarType":
                    return "c10::ScalarTypeType::get()"
                if type_repr == "SymFloat":
                    return "c10::SymFloatType::get()"
                raise AssertionError(f"Unsupported list element type: {t}")

            def codegen_memory_format(memory_format: torch.memory_format) -> str:
            # [FUNCTION DOC] HELPER: Maps a torch memory format to the c10 memory format enum name; raises a lookup error for
            # unknown formats.
                return {
                    torch.contiguous_format: "c10::MemoryFormat::Contiguous",
                    torch.channels_last: "c10::MemoryFormat::ChannelsLast",
                    torch.channels_last_3d: "c10::MemoryFormat::ChannelsLast3d",
                    torch.preserve_format: "c10::MemoryFormat::Preserve",
                }[memory_format]

            def codegen_tensor_ivalue(raw_arg: Any, arg_type: torch.JitType) -> str:
            # [FUNCTION DOC] HELPER: Returns an IValue expression for a tensor argument, materializing an RAII handle if needed
            # and dereferencing it to a tensor pointer.
                if hasattr(raw_arg, "codegen_reference"):
                    codegen_arg = raw_arg.codegen_reference()
                else:
                    codegen_arg = self.val_to_arg_str(raw_arg, arg_type)

                raii_var = self.create_tmp_raii_handle_var_if_needed(
                    codegen_arg, dispatch_lines
                )
                return (
                    "c10::IValue(*torch::aot_inductor::"
                    f"tensor_handle_to_tensor_pointer({raii_var}))"
                )

            def codegen_ivalue(raw_arg: Any, arg_type: torch.JitType) -> str:
            # [FUNCTION DOC] HELPER: Returns an IValue expression for any argument given its schema type. None becomes an empty
            # tensor for tensor types or an empty IValue otherwise; also handles optional, device, dtype, layout, memory
            # format, tensors, lists, tuples, bool, int, float, string, symbolic and number types. Raises AssertionError for
            # unsupported types or mismatched values.
                if raw_arg is None:
                    # A None at a non-optional Tensor arg means an absent tensor.
                    # Box it as an undefined at::Tensor (ATen's absent-tensor
                    # sentinel, matching eager value_or(Tensor())); boxing None
                    # would fail to unbox as `const Tensor&`.  Optionals stay None.
                    if isinstance(arg_type, torch.TensorType):
                        return "c10::IValue(at::Tensor())"
                    return "c10::IValue()"

                if isinstance(arg_type, torch.OptionalType):
                    return codegen_ivalue(raw_arg, arg_type.getElementType())

                if isinstance(raw_arg, torch.device):
                    if raw_arg.type not in DEVICE_TO_ATEN:
                        raise AssertionError(
                            raw_arg.type + " not found in DEVICE_TO_ATEN"
                        )
                    return (
                        "c10::IValue(c10::Device("
                        f"{DEVICE_TO_ATEN[raw_arg.type]}, "
                        f"{raw_arg.index if raw_arg.index is not None else 0}))"
                    )
                if isinstance(raw_arg, torch.dtype):
                    return f"c10::IValue({DTYPE_TO_ATEN[raw_arg]})"
                if isinstance(raw_arg, torch.layout):
                    return f"c10::IValue({LAYOUT_TO_ATEN[raw_arg]})"
                if isinstance(raw_arg, torch.memory_format):
                    return f"c10::IValue({codegen_memory_format(raw_arg)})"

                if isinstance(arg_type, torch.TensorType):
                    return codegen_tensor_ivalue(raw_arg, arg_type)

                if isinstance(arg_type, torch.ListType):
                    if not isinstance(raw_arg, (list, tuple)):
                        raise AssertionError(type(raw_arg))
                    list_var = next_tmp("tmp_list")
                    elem_type = arg_type.getElementType()
                    dispatch_lines.writeline(
                        f"c10::impl::GenericList {list_var}({type_expr(elem_type)});"
                    )
                    dispatch_lines.writeline(f"{list_var}.reserve({len(raw_arg)});")
                    for elem in raw_arg:
                        dispatch_lines.writeline(
                            f"{list_var}.emplace_back({codegen_ivalue(elem, elem_type)});"
                        )
                    return f"c10::IValue(std::move({list_var}))"

                if isinstance(arg_type, torch.TupleType):
                    if not isinstance(raw_arg, (list, tuple)):
                        raise AssertionError(type(raw_arg))
                    tuple_var = next_tmp("tmp_tuple")
                    dispatch_lines.writeline(f"std::vector<c10::IValue> {tuple_var};")
                    dispatch_lines.writeline(f"{tuple_var}.reserve({len(raw_arg)});")
                    for elem, elem_type in zip(raw_arg, arg_type.elements()):
                        dispatch_lines.writeline(
                            f"{tuple_var}.emplace_back({codegen_ivalue(elem, elem_type)});"
                        )
                    return (
                        "c10::IValue(c10::ivalue::Tuple::create("
                        f"std::move({tuple_var})))"
                    )

                if isinstance(arg_type, torch.BoolType):
                    return (
                        "c10::IValue(static_cast<bool>("
                        f"{self.val_to_arg_str_for_prim_type(raw_arg, bool)}))"
                    )
                if isinstance(arg_type, torch.IntType):
                    return (
                        "c10::IValue(static_cast<int64_t>("
                        f"{self.val_to_arg_str_for_prim_type(raw_arg, int)}))"
                    )
                if isinstance(arg_type, torch.FloatType):
                    return (
                        "c10::IValue(static_cast<double>("
                        f"{self.val_to_arg_str_for_prim_type(raw_arg, float)}))"
                    )
                if isinstance(arg_type, torch.StringType):
                    return (
                        "c10::IValue(std::string("
                        f"{self.val_to_arg_str_for_prim_type(raw_arg, str)}))"
                    )
                if isinstance(arg_type, torch.SymBoolType):
                    return (
                        "c10::IValue(c10::SymBool(static_cast<bool>("
                        f"{self.val_to_arg_str_for_prim_type(raw_arg, bool)})))"
                    )
                if isinstance(arg_type, torch.SymIntType):
                    return (
                        "c10::IValue(c10::SymInt(static_cast<int64_t>("
                        f"{self.val_to_arg_str_for_prim_type(raw_arg, int)})))"
                    )
                if isinstance(arg_type, torch.NumberType):
                    return f"c10::IValue({self.val_to_arg_str_for_prim_type(raw_arg, None)})"

                arg_type_repr = repr(arg_type)
                if arg_type_repr == "Layout":
                    if not isinstance(raw_arg, torch.layout):
                        raise AssertionError(type(raw_arg))
                    return f"c10::IValue({LAYOUT_TO_ATEN[raw_arg]})"
                if arg_type_repr == "MemoryFormat":
                    if not isinstance(raw_arg, torch.memory_format):
                        raise AssertionError(type(raw_arg))
                    return f"c10::IValue({codegen_memory_format(raw_arg)})"
                if arg_type_repr == "ScalarType":
                    if not isinstance(raw_arg, torch.dtype):
                        raise AssertionError(type(raw_arg))
                    return f"c10::IValue({DTYPE_TO_ATEN[raw_arg]})"
                if arg_type_repr == "SymFloat":
                    return (
                        "c10::IValue(c10::SymFloat(static_cast<double>("
                        f"{self.val_to_arg_str_for_prim_type(raw_arg, float)})))"
                    )

                raise AssertionError(f"Unsupported boxed dispatcher arg: {arg_type}")

            stack_var = next_tmp("dispatch_stack")
            dispatch_lines.writeline(f"std::vector<c10::IValue> {stack_var};")
            dispatch_lines.writeline(
                f"{stack_var}.reserve({len(op_overload._schema.arguments)});"
            )
            for raw_arg, schema_arg in zip(raw_args, op_overload._schema.arguments):
                dispatch_lines.writeline(
                    f"{stack_var}.emplace_back({codegen_ivalue(raw_arg, schema_arg.real_type)});"
                )

            dispatch_lines.writeline(
                "static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("
                f'"{op_overload._schema.name}", "{op_overload._schema.overload_name}");'
            )
            dispatch_lines.writeline(f"op.callBoxed(&{stack_var});")

            stack_idx = 0
            for output_arg, raw_output_arg in zip(output_args, raw_outputs):
                if isinstance(raw_output_arg, ir.MutationOutput):
                    continue
                if output_arg is not None:
                    dispatch_lines.writeline(
                        f"{output_arg} = torch::aot_inductor::new_tensor_handle("
                        f"std::move({stack_var}[{stack_idx}]).toTensor());"
                    )
                stack_idx += 1

        dispatch_lines.writeline("}")
        self.writelines(dispatch_lines.getvalue().splitlines())

    def generate_fallback_kernel_with_runtime_lookup_python(
        self,
        buf_name: str,
        python_kernel_name: str,
        op_overload: torch._ops.OpOverload,
        raw_args: Sequence[Any],
        output_args: Sequence[str | None],
        raw_outputs: Sequence[ir.Buffer],
    ) -> None:
        # [FUNCTION DOC] Emits a JIT fallback call that calls back into Python through custom_op_wrapper: builds a Python
        # tuple of the op name and converted arguments, calls the wrapper (returning from the entry function if a Python
        # error is set, else throwing), then extracts tensor handles from the result capsule or list into the output
        # variables, and wraps everything in a GIL-scoped block with RAII output declarations.
        """Generate fallback kernel calls with runtime (non-AOT) dispatch.  This can
        only be called in cpp_wrapper mode, and assumes that the input is a non-None
        OpOverload.

        This function calls into Python to dispatch, which allows it to handle datatypes
        that cannot be contained in StableIValue, at the cost of some performance."""
        self.load_custom_op_wrapper()

        num_args = len(raw_args)
        py_args_var = f"py_args_{next(self.arg_var_id)}"
        # First arg is always the python op name
        lines = textwrap.dedent(f"""
            RAIIPyObject {py_args_var}(PyTuple_New({num_args + 1}));
            if (!{py_args_var}) {{
                throw std::runtime_error("PyTuple_New {py_args_var} failed");
            }}
            PyTuple_SetItem({py_args_var}, 0, PyUnicode_FromString("{python_kernel_name}"));
            """)

        for idx, (raw_arg, schema_arg) in enumerate(
            zip(raw_args, op_overload._schema.arguments)
        ):
            lines += self.generate_py_arg(
                py_args_var, idx + 1, raw_arg, schema_arg.real_type
            )

        lines += textwrap.dedent(f"""
            // Call the custom op in Python
            RAIIPyObject py_{buf_name}(PyObject_CallObject(custom_op_wrapper, {py_args_var}));
            if (!py_{buf_name}) {{
                if (PyErr_Occurred()) {{
                    return;
                }}
                throw std::runtime_error("PyObject_CallObject {python_kernel_name} failed");
            }}
            """)

        if raw_outputs:
            num_returned_outputs = 0
            returned_output_slots = []
            for output_arg, raw_output_arg in zip(output_args, raw_outputs):
                if isinstance(raw_output_arg, ir.MutationOutput):
                    continue
                if output_arg is not None:
                    returned_output_slots.append((num_returned_outputs, output_arg))
                num_returned_outputs += 1
        else:
            num_returned_outputs = len(output_args)
            returned_output_slots = [
                (idx, output_arg)
                for idx, output_arg in enumerate(output_args)
                if output_arg is not None
            ]

        # custom_op_wrapper returns a bare capsule only for a single, non-list
        # Tensor return. A Tensor[] return (even of length 1) or multiple returns
        # come back as a Python list that must be indexed with PyList_GET_ITEM.
        schema_returns = op_overload._schema.returns
        returns_python_list = len(schema_returns) != 1 or isinstance(
            schema_returns[0].real_type, torch.ListType
        )
        if (
            num_returned_outputs == 1
            and returned_output_slots
            and not returns_python_list
        ):
            # result is a single tensor
            _, output = returned_output_slots[0]
            lines += f"{output} = reinterpret_cast<AtenTensorHandle>(PyCapsule_GetPointer(py_{buf_name}.get(), NULL));\n"
        else:
            # result is a list of tensors (Tensor[] or multiple returns)
            for idx, output_arg in returned_output_slots:
                lines += f"{output_arg} = reinterpret_cast<AtenTensorHandle>(PyCapsule_GetPointer(PyList_GET_ITEM(py_{buf_name}.get(), {idx}), NULL));\n"

        if raw_outputs:
            declarations_before_scope = [
                f"RAIIAtenTensorHandle {output_arg};"
                for output_arg, raw_output_arg in zip(output_args, raw_outputs)  # type: ignore[arg-type]
                if output_arg is not None
                and not isinstance(raw_output_arg, ir.MutationOutput)
            ]
        else:
            declarations_before_scope = [
                f"RAIIAtenTensorHandle {output_arg};"
                for output_arg in output_args  # type: ignore[arg-type]
                if output_arg is not None
            ]
        scope_gil_acquire = self.generate_scoped_gil_acquire(
            declarations_before_scope, lines
        )
        self.writelines(scope_gil_acquire)

    def generate_fallback_kernel_with_runtime_lookup_aot(
        self,
        op_overload: torch._ops.OpOverload | torch._ops.HigherOrderOperator,
        raw_args: Sequence[Any],
        output_args: _OUTPUT_ARGS_TYPE,
        raw_outputs: Sequence[ir.Buffer],
    ) -> None:
        # [FUNCTION DOC] Emits a proxy-executor based call for a fallback kernel in AOT mode: gathers tensor and int
        # arguments, creates mutable temporary arrays for them, and writes a checked proxy executor call using the index of
        # the latest extern kernel node.
        (
            tensor_call_args,
            int_call_args,
        ) = self.generate_extern_kernel_args_decl_if_needed(
            op_overload,
            raw_args,
            output_args,
            raw_outputs,
        )
        # force both temporary arrays to generate mutable data pointers, since the proxy
        # executor signature requires that datatype
        int_call_str = self._generate_temporary_array_pointer(
            "int64_t", int_call_args, force_mutable=True
        )
        tensor_call_args = self.codegen_runtime_lookup_tensor_call_args(
            tensor_call_args
        )
        tensor_call_str = self._generate_temporary_array_pointer(
            "AtenTensorHandle", tensor_call_args, force_mutable=True
        )

        extern_kernel_node_index = len(V.extern_kernel_nodes) - 1
        self.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_proxy_executor_call_function(proxy_executor, "
            f"{extern_kernel_node_index}, "
            f"{len(int_call_args)}, "
            f"{int_call_str}, "
            f"{len(tensor_call_args)}, "
            f"{tensor_call_str}));"
        )

    def codegen_runtime_lookup_tensor_call_args(
        self, tensor_call_args: Sequence[str]
    ) -> Sequence[str]:
        # [FUNCTION DOC] Hook for subclasses to adjust the tensor argument list before the proxy executor call; returns it
        # unchanged here.
        return tensor_call_args

    def generate_reset_kernel_saved_flags(self):
        # [FUNCTION DOC] Does nothing; kernel-saved flags need no reset for the C++ wrapper.
        pass

    def generate_save_uncompiled_kernels(self):
        # [FUNCTION DOC] Does nothing; uncompiled kernels are not saved for the C++ wrapper.
        pass

    def c_type_for_prim_type(self, val, type_) -> str:
        # [FUNCTION DOC] Returns the C type name used for a value of a given schema type: optional adds a pointer, tensor is
        # a handle, ints 64-bit, bool/enums/layout/memory format/dtype 32-bit, floats double, numbers by value type, strings
        # const char pointer. Raises AssertionError for unexpected types or number values.
        if isinstance(type_, torch.OptionalType):
            return f"{self.c_type_for_prim_type(val, type_.getElementType())}*"
        elif isinstance(type_, torch.TensorType):
            return "AtenTensorHandle"
        elif isinstance(type_, (torch.IntType, torch.SymIntType)):
            return "int64_t"
        elif isinstance(
            type_, (torch.BoolType, torch.SymBoolType, torch.EnumType)
        ) or repr(type_) in ("Layout", "MemoryFormat", "ScalarType"):
            return "int32_t"
        elif isinstance(type_, torch.FloatType):
            return "double"
        elif isinstance(type_, torch.NumberType):
            if isinstance(val, bool):
                return "int32_t"
            elif isinstance(val, (int, float)):
                return "double"
            elif val is None:
                # This could happen when val is an optional value
                return "double"
            else:
                raise AssertionError(
                    f"Unexpected type in c_type_for_prim_type: {type_=}"
                )
        elif isinstance(type_, torch.StringType):
            return "const char*"
        else:
            raise AssertionError(f"Unexpected type in c_type_for_prim_type: {type_=}")

    def val_to_arg_str_for_prim_type(self, val, type_) -> str:
        # [FUNCTION DOC] Returns a C++ literal or expression string for a primitive-like value: bool as 0/1, ints with a long
        # suffix per platform, complex as a C++ complex, strings quoted, buffers by their reference, devices, dtypes, layouts and
        # memory formats via cached constants, floats, lists as braced lists, symbolic values via cexpr, else the repr.
        # KNOWN ISSUE: the type argument is not used (marked TODO); strings are quoted without escaping.
        # TODO: not using type_ as the first step of refactoring. Will update this later.
        if isinstance(val, bool):
            return "1" if val else "0"
        elif isinstance(val, int):
            # uint64_t is long on Linux, but long long on MacOS and Windows
            return f"{val}LL" if sys.platform in ["darwin", "win32"] else f"{val}L"
        elif isinstance(val, complex):
            return f"c10::complex<double>{{ {self.generate_float_value(val.real)}, {self.generate_float_value(val.imag)} }}"
        elif isinstance(val, str):
            return f'"{val}"'
        elif isinstance(
            val, (ir.Buffer, ir.ReinterpretView, ir.StorageBox, ir.TensorBox)
        ):
            return val.codegen_reference()
        elif isinstance(val, torch.device):
            return self.codegen_device(val)
        elif isinstance(val, torch.dtype):
            return self.codegen_dtype(val)
        elif isinstance(val, torch.layout):
            return self.codegen_layout(val)
        elif isinstance(val, torch.memory_format):
            return self.codegen_memory_format(val)
        elif isinstance(val, float):
            return self.generate_float_value(val)
        elif isinstance(val, (list, tuple)):
            # FIXME: This happens because type_ is not always properly set to torch.ListType
            return f"{{{', '.join(self.val_to_arg_str(x, None) for x in val)}}}"
        elif isinstance(val, SymTypes):
            return cexpr(val.node.expr)
        elif isinstance(val, sympy.Expr):
            return cexpr(val)
        else:
            return repr(val)

    def val_to_arg_str(self, val, type_=None) -> str:
        # [FUNCTION DOC] Returns the C++ argument string for a value given its schema type. None becomes nullptr (or a
        # null pointer plus zero length for device/list/tuple optionals) or a newly created empty tensor for tensor types.
        # Optional values are passed by address via temporaries; lists and tuples become a temporary array pointer plus
        # length (null and zero when empty); scalars passed for tensor types are converted to tensors. May write
        # declaration lines. Raises AssertionError for None with no known type or value/type mismatches.
        if val is None:
            # None needs special care. It either represent nullopt or an empty tensor
            if type_ is None or isinstance(type_, torch.OptionalType):
                if type_ is not None and isinstance(
                    type_.getElementType(),
                    (
                        torch.DeviceObjType,
                        torch.ListType,
                        torch.TupleType,
                    ),
                ):
                    return "nullptr, 0"
                return "nullptr"

            if isinstance(type_, torch.TensorType):
                # create an empty tensor, the equivalent of at::Tensor()
                var_name = f"var_{next(self.arg_var_id)}"
                self.writeline(f"AtenTensorHandle {var_name}_handle;")
                self.writeline(
                    f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_new_uninitialized_tensor(&{var_name}_handle));"
                )
                self.writeline(f"RAIIAtenTensorHandle {var_name}({var_name}_handle);")
                return var_name

            raise AssertionError("Can not map None to a known data type")

        if isinstance(type_, torch.OptionalType):
            element_type = type_.getElementType()
            arg_str = self.val_to_arg_str(val, element_type)
            # Handle optional iterables as a special case.  Utilize the
            # temporary_reference function to avoid saving them off and increasing
            # memory usage.
            if isinstance(element_type, (torch.ListType, torch.TupleType)):
                main_value, aux = arg_str.rsplit(", ", maxsplit=1)
                return f"&temporary_reference({main_value}), {aux}"

            # Handle optional tensors as a special case, as above.
            if isinstance(element_type, torch.TensorType):
                base_handle = self.val_to_arg_str(val, element_type)
                return f"&temporary_reference({base_handle}.get())"

            var_name = f"var_{next(self.arg_var_id)}"
            if isinstance(element_type, torch.DeviceObjType):
                main_value, aux = arg_str.rsplit(", ", maxsplit=1)
                self.writeline(f"auto {var_name} = {main_value};")
                return f"&{var_name}, {aux}"

            self.writeline(
                f"{self.c_type_for_prim_type(val, element_type)} {var_name} = {arg_str};"
            )
            return f"&{var_name}"

        if isinstance(type_, (torch.ListType, torch.TupleType)):
            if not isinstance(val, (list, tuple)):
                raise AssertionError(f"{val} does not match with arg type {type_}")
            element_type = type_.getElementType()

            if len(val) == 0:
                # Zero-size array is not supported in the C or C++ standard, so return a
                # nullptr.
                return "nullptr, 0"

            result = [self.val_to_arg_str(x, element_type) for x in val]
            if isinstance(element_type, torch.TensorType):
                result = [f"{t}.get()" for t in result]

            c_type = self.c_type_for_prim_type(val[0], element_type)
            # see the comment in self._generate_temporary_array_pointer for an
            # explanation of why this c_type gets modified
            if isinstance(element_type, torch.OptionalType) and not c_type.startswith(
                "const"
            ):
                c_type = f"const {c_type}"

            # need to pass the array length, because we can't use the std::array member
            # function
            return (
                f"{self._generate_temporary_array_pointer(c_type, result)}, {len(val)}"
            )

        val_is_scalar = isinstance(val, (bool, complex, float, int, *SymTypes))
        if isinstance(type_, torch.TensorType) and val_is_scalar:
            val_str = self.val_to_arg_str_for_prim_type(val, None)
            return self.codegen_scalar_to_tensor(val_str)

        return self.val_to_arg_str_for_prim_type(val, type_)

    def create_tmp_raii_handle_var_if_needed(
        self, handle: str, writer: HasWriteLine | list[str] | None = None
    ) -> str:
        # [FUNCTION DOC] If the handle expression is an rvalue RAII tensor construction, writes a named local for it to the
        # writer (a buffer, a list to append to, or self when none) and returns the new name; otherwise returns the
        # handle unchanged.
        """If the input handle is an rvalue RAII tensor, creates an lvalue variable for
        it in writer.  Returns a variable name that can be used to access handle."""
        if not handle.startswith(
            (
                "borrow_arrayref_tensor_as_tensor(",
                "copy_arrayref_tensor_to_tensor(",
                "wrap_with_raii_handle_if_needed(",
                "RAIIAtenTensorHandle(",
            )
        ):
            return handle

        tmp_var_name = f"var_{next(self.arg_var_id)}"
        call_str = f"auto {tmp_var_name} = {handle};"

        if writer is None:
            self.writeline(call_str)
        elif isinstance(writer, list):
            writer.append(call_str)
        else:
            writer.writeline(call_str)

        return tmp_var_name

    def write_kernel_context_guard_begin(
        self,
    ):
        # [FUNCTION DOC] Opens a kernel-context-guarded block by writing an opening brace.
        # Beginning of a kernel context guarded block.
        # The block looks like this:
        # {
        # KernelContextGuard _ctx("{kernel_name}", {stack_trace_str});
        # ... operations...
        # }
        self.writeline("{")

    def write_kernel_context_guard_end(
        self,
    ):
        # [FUNCTION DOC] Closes a kernel-context-guarded block by writing a closing brace.
        # End of a kernel context guarded block.
        self.writeline("}")

    def write_kernel_context_guard(
        self,
        kernel_name: str,
        node_schedule: Sequence[BaseSchedulerNode] | ExternKernel,
    ):
        # [FUNCTION DOC] Writes a C++ kernel context guard for the kernel name, carrying the union of stack traces of the
        # given node schedule or extern kernel as a raw string.
        def aggregate_stack_traces(
            node_schedule: Sequence[BaseSchedulerNode] | ExternKernel,
        ) -> OrderedSet[str]:
        # [FUNCTION DOC] HELPER: Returns the combined stack trace set of a list of scheduler nodes (those with a node),
        # the extern kernel's own traces, or an empty set for other inputs.
            if isinstance(node_schedule, list):
                return functools.reduce(
                    lambda a, b: a | b,
                    [
                        # pyrefly: ignore [missing-attribute]
                        node.node.get_stack_traces()
                        for node in node_schedule
                        if hasattr(node, "node") and node.node
                    ],
                    OrderedSet(),
                )
            elif isinstance(node_schedule, ExternKernel):
                return node_schedule.get_stack_traces()
            else:
                return OrderedSet()

        stack_trace_str = 'R"('
        stack_traces = aggregate_stack_traces(node_schedule)

        for stack_trace in stack_traces:
            for line in stack_trace.split("\n"):
                stack_trace_str += f"\n{line}"
            stack_trace_str += "\n"
        stack_trace_str += ')"'
        self.writeline(f'KernelContextGuard _ctx("{kernel_name}", {stack_trace_str});')
