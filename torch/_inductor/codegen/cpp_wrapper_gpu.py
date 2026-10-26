# mypy: allow-untyped-defs
# [MODULE DOC] C++ wrapper codegen for GPU (CUDA/HIP/XPU) in cpp_wrapper and AOTInductor modes.
# Main content: helpers for C string escaping, TMA descriptor signature handling and the AOTI lazy kernel
# config header; DeferredTritonCallWrapper (generates Triton kernel load/launch C++ once autotuning is done,
# eagerly or lazily); CppWrapperGpu (the wrapper codegen class); UnwrapUnspecArg marker.
# No import-time side effects beyond imports and constants.
from __future__ import annotations

import dataclasses
import os
import re
import sys
from itertools import count, zip_longest
from typing import Any, cast
from typing_extensions import Self

import sympy

import torch
from torch import dtype as torch_dtype
from torch._inductor.codecache import get_cpp_wrapper_cubin_path_name
from torch._inductor.runtime.runtime_utils import dynamo_timed
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..codecache import CudaKernelParamCache
from ..ir import (
    GraphPartitionSignature,
    TensorBox,
    TMADescriptorExperimental,
    TMADescriptorStable,
)
from ..runtime.hints import (
    InductorMeta,
    TRITON_DEFAULT_BLOCK_SIZES,
    TRITON_DEFAULT_RSPLIT,
    TRITON_DEFAULT_RSPLIT_SIZE,
    TritonMeta,
)
from ..stream_utils import (
    AOTI_SUPPORTED_STREAM_OP_NAMES,
    AOTI_UNSUPPORTED_STREAM_OP_REASONS,
    get_stream_name,
)
from ..utils import (
    cache_on_self,
    DeferredLineBase,
    get_gpu_type,
    GPU_ALIGN_BYTES,
    IndentedBuffer,
    make_codegen_buffer,
    XPU_KERNEL_FORMAT,
)
from ..virtualized import V
from .aoti_hipify_utils import maybe_hipify_code_wrapper
from .common import get_device_op_overrides, TritonScratchWorkspace
from .cpp_utils import cexpr
from .cpp_wrapper_cpu import CppWrapperCpu
from .multi_kernel import MultiKernelCall
from .triton_utils import should_unwrap_unspec_arg
from .wrapper import PythonWrapperCodegen, SymbolicCallArg


_cpp_string_literal_escapes = {
    "\\": "\\\\",
    '"': '\\"',
    "\n": "\\n",
    "\t": "\\t",
    "\r": "\\r",
}
_cpp_string_literal_pattern = re.compile(r'["\\\n\t\r]')


def cpp_string_literal(s: str) -> str:
    # [FUNCTION DOC] Return the string as a quoted C++ string literal, escaping backslash, quote, newline, tab and CR.
    escaped = _cpp_string_literal_pattern.sub(
        lambda match: _cpp_string_literal_escapes[match.group(0)], s
    )
    return f'"{escaped}"'


def generate_aoti_kernel_config_header(kernel_names: list[str]) -> str:
    # [FUNCTION DOC] Build the text of a C header with one block of macros per lazily compiled kernel (cubin path, mangled
    # name, warps, shared memory, per-subkernel block sizes, RSPLIT values, precomputed grid config index, scratch
    # sizes), read from CudaKernelParamCache after the JIT first pass. Combo kernel default configs are merged.
    # Raises RuntimeError if a kernel has no cached params.
    # KNOWN ISSUE: the error message prints the whole kernel_names list rather than the missing kernel.
    """Generate a C header defining macros for each lazy-compiled kernel.

    Called after the JIT first-pass runs and populates CudaKernelParamCache.
    The AOTI compilation includes this header so that LazyKernelCompileResult
    structs get compile-time-initialized with the autotuned values.
    """

    def braced(values: list[int]) -> str:
        # [FUNCTION DOC] HELPER: Format a list of ints as a C brace initializer string.
        return "{" + ", ".join(str(v) for v in values) + "}"

    buf = IndentedBuffer()
    buf.splice("""
        #pragma once
        // Auto-generated kernel configurations for AOTInductor lazy compile.
    """)

    for kernel_name in kernel_names:
        params = CudaKernelParamCache.get(kernel_name)
        if params is None:
            raise RuntimeError(
                "When autotune_at_compile_time is False, AOTInductor generates"
                " both JIT code and AOT code. They are expected to have exactly"
                f" the same kernels. However, AOT code contains kernels, {kernel_names},"
                " that is not in the JIT code."
            )

        macro_prefix = kernel_name.upper()
        cubin_path = cpp_string_literal(params[get_cpp_wrapper_cubin_path_name()])
        mangled_name = cpp_string_literal(params["mangled_name"])
        num_warps = params["num_warps"]
        shared_mem = params["shared_mem"]

        # params["config"] is already a dict (from config_to_dict in CachingAutotuner)
        config_dict = params.get("config") or {}

        # For combo/foreach kernels, merge default_config
        inductor_meta = params.get("inductor_meta") or {}
        combo_grid_meta = inductor_meta.get("combo_grid_meta")
        default_config = (
            combo_grid_meta.get("default_config") if combo_grid_meta else None
        )
        if default_config:
            config_dict = {**default_config, **config_dict}

        config_index: int | None = None
        grid_type = inductor_meta.get("grid_type")
        if grid_type == "PrecomputedGrid":
            precomputed_grids = inductor_meta.get("precomputed_grids", [])
            for idx, entry in enumerate(precomputed_grids):
                entry_config = entry.get("config", {})
                if all(config_dict.get(k) == v for k, v in entry_config.items()):
                    config_index = idx
                    break

        # Per-subkernel block sizes for combo kernels, or single-element lists.
        num_kernels = combo_grid_meta.get("num_kernels", 1) if combo_grid_meta else 1
        if num_kernels > 1 and "XBLOCK_0" in config_dict:
            xblocks = [
                config_dict.get(f"XBLOCK_{i}", TRITON_DEFAULT_BLOCK_SIZES["XBLOCK"])
                for i in range(num_kernels)
            ]
            yblocks = [
                config_dict.get(f"YBLOCK_{i}", TRITON_DEFAULT_BLOCK_SIZES["YBLOCK"])
                for i in range(num_kernels)
            ]
            zblocks = [
                config_dict.get(f"ZBLOCK_{i}", TRITON_DEFAULT_BLOCK_SIZES["ZBLOCK"])
                for i in range(num_kernels)
            ]
            r0blocks = [
                config_dict.get(f"R0_BLOCK_{i}", TRITON_DEFAULT_BLOCK_SIZES["R0_BLOCK"])
                for i in range(num_kernels)
            ]
        else:
            xblocks = [config_dict.get("XBLOCK", TRITON_DEFAULT_BLOCK_SIZES["XBLOCK"])]
            yblocks = [config_dict.get("YBLOCK", TRITON_DEFAULT_BLOCK_SIZES["YBLOCK"])]
            zblocks = [config_dict.get("ZBLOCK", TRITON_DEFAULT_BLOCK_SIZES["ZBLOCK"])]
            r0blocks = [
                config_dict.get("R0_BLOCK", TRITON_DEFAULT_BLOCK_SIZES["R0_BLOCK"])
            ]
        rsplit = config_dict.get("RSPLIT", TRITON_DEFAULT_RSPLIT)
        rsplit_size = config_dict.get("RSPLIT_SIZE", TRITON_DEFAULT_RSPLIT_SIZE)
        ci = config_index if config_index is not None else -1
        gs = params.get("global_scratch", -1) or -1
        ps = params.get("profile_scratch", -1) or -1

        buf.writeline("")
        buf.splice(f"""
            // Kernel: {kernel_name}
            #define {macro_prefix}_CUBIN_PATH {cubin_path}
            #define {macro_prefix}_MANGLED_NAME {mangled_name}
            #define {macro_prefix}_NUM_WARPS {num_warps}
            #define {macro_prefix}_SHARED_MEM {shared_mem}
            #define {macro_prefix}_XBLOCKS {braced(xblocks)}
            #define {macro_prefix}_YBLOCKS {braced(yblocks)}
            #define {macro_prefix}_ZBLOCKS {braced(zblocks)}
            #define {macro_prefix}_R0BLOCKS {braced(r0blocks)}
            #define {macro_prefix}_RSPLIT {rsplit}
            #define {macro_prefix}_RSPLIT_SIZE {rsplit_size}
            #define {macro_prefix}_CONFIG_INDEX {ci}
            #define {macro_prefix}_GLOBAL_SCRATCH {gs}
            #define {macro_prefix}_PROFILE_SCRATCH {ps}
        """)

    return buf.getvalue()


TRITON_SIGNATURE_TO_CPP = {
    "i32": "int32_t",
    "i64": "int64_t",
    "fp32": "float",
    "fp64": "double",
}


def signature_is_tma_desc(sig: str | None) -> bool:
    # [FUNCTION DOC] Return True if the Triton signature string denotes a TMA descriptor (nvTmaDesc or tensordesc type).
    """Check if a Triton signature represents a TMA descriptor."""
    if not sig:
        return False
    if sig == "nvTmaDesc":
        return True
    if sig.startswith("tensordesc<"):
        return True
    return False


def _unpack_tma_descriptor_args(var_name: str, sig_type: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Given a variable holding a stable TMA descriptor and its tensordesc signature, return the list
    # of address-of expressions for its kernel args: the descriptor handle, then block shape and strides entries
    # per dimension (rank parsed from the signature). Raises AssertionError if the signature cannot be parsed.
    """Unpack a StableTMADescriptor into kernel launch args.

    Given a variable name holding a StableTMADescriptor and its tensordesc<...>
    signature, returns the list of pointer args: &var.m, &var.block_shape[i]...,
    &var.strides[i]...
    """
    match = re.match(r"tensordesc<[^[]*\[([^\]]*)\]", sig_type)
    if match is None:
        raise AssertionError(f"Cannot parse tensordesc signature: {sig_type}")
    ndim = match.group(1).count(",") + 1
    result = [f"&{var_name}.m"]
    for i in range(ndim):
        result.append(f"&{var_name}.block_shape[{i}]")
    for i in range(ndim):
        result.append(f"&{var_name}.strides[{i}]")
    return result


class _LazyTritonCompileKickoffLine(DeferredLineBase):
    # [CLASS DOC] HELPER: Deferred line that renders its text only if the list of lazy kernel names is non-empty at
    # render time; used to emit the compile kickoff call only when lazy kernels exist.
    def __init__(self, lazy_kernel_names: list[str], line: str):
        # [FUNCTION DOC] HELPER: Store the line text and the (shared, later mutated) list of lazy kernel names.
        super().__init__(line)
        self.lazy_kernel_names = lazy_kernel_names

    def __call__(self) -> str | None:
        # [FUNCTION DOC] HELPER: Return the line if any lazy kernel exists, otherwise None (line dropped).
        return self.line if self.lazy_kernel_names else None

    def _new_line(self, line: str) -> _LazyTritonCompileKickoffLine:
        # [FUNCTION DOC] HELPER: Return a copy with new text sharing the same kernel name list.
        return _LazyTritonCompileKickoffLine(self.lazy_kernel_names, line)


@dataclasses.dataclass
class DeferredTritonCallWrapper:
    # [CLASS DOC] Dataclass describing the deferred C++ wrapper around one Triton kernel (wrapper function name, kernel
    # name, kernel source bodies, argument types, optional triton/inductor meta and TMA tensor arg names).
    # Its generate method writes the wrapper function (kernel load, grid, launch) into the wrapper prefix after
    # autotuning has finished, either from cached params or in lazy-compile (dual JIT/AOTI) form.
    """
    When using cpp wrapper, GPU kernel load and launch needs to wait for Triton kernels
    to be tuned and stored as cubin files, so use a deferred generating the final wrapper around
    the triton kernel until right before the prefix is written.
    """

    wrapper_name: str
    kernel_name: str
    kernel_name_to_body: dict[str, str]
    arg_types: list[Any]
    triton_meta: TritonMeta | None = None
    inductor_meta: dict[str, Any] | None = None
    tma_tensor_args: dict[str, str] = dataclasses.field(default_factory=dict)

    @cache_on_self
    def _get_tma_args(self) -> dict[str, str]:
        # [FUNCTION DOC] HELPER: Return a mapping of TMA descriptor argument name to tensordesc signature type (cached).
        # Raises RuntimeError if the experimental nvTmaDesc API is used, since lazy compile does not support it.
        """Get mapping of TMA descriptor arg names to their signature types."""
        triton_meta = self.triton_meta or {}
        signature = triton_meta.get("signature", {})
        for name, sig_type in signature.items():
            if sig_type == "nvTmaDesc":
                raise RuntimeError(
                    f"nvTmaDesc (experimental TMA API) is not supported in lazy compile "
                    f"for arg '{name}'. Use the stable tensordesc API instead."
                )
        return {
            name: sig_type
            for name, sig_type in signature.items()
            if isinstance(sig_type, str) and sig_type.startswith("tensordesc<")
        }

    def _get_cpp_param_type(
        self, name: str, arg_type: Any, signature: dict[str, str] | None = None
    ) -> str:
        # [FUNCTION DOC] HELPER: Return the C++ parameter declaration for an argument: const template-typed reference for tensors
        # and unwrapped scalars (non-const for TMA descriptors), 64-bit int for integers/symbolic, float, or bool.
        # Raises ValueError for other types.
        """Get the C++ parameter declaration for a given arg type."""
        if isinstance(arg_type, (torch_dtype, UnwrapUnspecArg)):
            # TMA descriptors need non-const references since their fields
            # are passed as void* pointers to kernel launch args
            if signature and signature_is_tma_desc(signature.get(name)):
                return f"{name}_type_& {name}"
            return f"const {name}_type_& {name}"
        elif issubclass(arg_type, (SymbolicCallArg, sympy.Expr, int)):
            return f"int64_t {name}"
        elif arg_type is float:
            return f"float {name}"
        elif arg_type is bool:
            return f"bool {name}"
        else:
            raise ValueError(f"Unexpected arg type {arg_type}")

    def _write_wrapper_signature(
        self,
        prefix: IndentedBuffer,
        wrapper: CppWrapperGpu,
        arg_names: list[str],
        arg_types: list[Any] | None = None,
        signature: dict[str, str] | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Write the template header and signature of the wrapper function, with JIT and AOTI variants
        # (AOTI adds the kernels object; both add device index, stream and optional cubin directory parameters).
        """Write the wrapper function signature including template and parameters."""
        if arg_types is None:
            arg_types = self.arg_types

        template_types = [
            f"typename {name}_type_"
            for name, arg_type in zip(arg_names, arg_types)
            if isinstance(arg_type, (torch_dtype, UnwrapUnspecArg))
        ]
        if template_types:
            prefix.writeline_jit(f"template <{', '.join(template_types)}>")
        prefix.writeline_aot(
            f"template <{', '.join([*template_types, 'typename kernels_type_'])}>"
        )

        cubin_dir_param = "const std::optional<std::string>& cubin_dir_ = std::nullopt"
        kernels_param = "kernels_type_& kernels_"

        shared_params = [
            self._get_cpp_param_type(name, arg_type, signature)
            for name, arg_type in zip(arg_names, arg_types)
        ]
        shared_params.append("int32_t device_idx_")
        shared_params.append(
            maybe_hipify_code_wrapper(
                f"{wrapper.device_codegen.cpp_stream_type()} stream_"
            )
        )

        def emit(writer, params):
            # [FUNCTION DOC] HELPER: Write each parameter followed by a comma except the last.
            for p in params[:-1]:
                writer(f"{p},")
            writer(params[-1])

        prefix.writeline(f"static __attribute__((noinline)) void {self.wrapper_name}(")
        with prefix.indent():
            emit(prefix.writeline_jit, [*shared_params, cubin_dir_param])
            emit(prefix.writeline_aot, [*shared_params, kernels_param, cubin_dir_param])
        prefix.writeline("){")

    def generate(self, wrapper: CppWrapperGpu):
        # [FUNCTION DOC] Write the wrapper function for the kernel into the wrapper prefix: resolve multi-kernel choice, defer to
        # generate_lazy when autotune_at_compile_time is false, otherwise read cached params, emit a static kernel
        # variable (JIT only), signature, grid, kernel load and launch. In AOT mode the original Triton source is
        # included as a comment, and the cubin path is recorded as an additional file unless the binary is embedded.
        # Raises AssertionError if the param cache is missing or extra launcher args are inconsistent.
        """
        Generate the GPU kernel definition, as well as load and launch code.
        """
        prefix = wrapper.prefix
        if self.kernel_name.startswith("multi_kernel_"):
            # MultiKernel will select one kernel after running the autotune block
            self.kernel_name = MultiKernelCall.lookup_choice(self.kernel_name)

        # Defer compilation to runtime if autotune_at_compile_time is False (JIT only).
        # AOTI lazy-compile emission is wired up later in the stack.
        if config.triton.autotune_at_compile_time is False:
            return self.generate_lazy(wrapper)

        params = CudaKernelParamCache.get(self.kernel_name)
        if not params:
            raise AssertionError(
                f"CudaKernelParamCache not populated for {self.kernel_name}"
            )
        def_args = params["def_args"]
        arg_types = self.arg_types
        inductor_meta = params["inductor_meta"]

        if "extra_launcher_args" in inductor_meta and len(def_args) > len(arg_types):
            # extra_launcher_args should already be in def_args
            if len(def_args) != len(arg_types) - len(
                inductor_meta["extra_launcher_args"]
            ):
                raise AssertionError(
                    "expected len(def_args) == len(arg_types) - "
                    f"len(extra_launcher_args), got {len(def_args)}"
                )
            arg_types = arg_types + [SymbolicCallArg] * len(
                inductor_meta["extra_launcher_args"]
            )

        if not V.graph.aot_mode:
            prefix.writeline(
                maybe_hipify_code_wrapper(
                    f"static {wrapper.device_codegen.cpp_kernel_type()} {self.kernel_name} = nullptr;"
                )
            )
            kernel_var_name = self.kernel_name
        else:
            kernel_var_name = f"kernels_.{self.kernel_name}"

        # Write wrapper function signature
        self._write_wrapper_signature(prefix, wrapper, def_args, arg_types)

        with prefix.indent():
            if V.graph.aot_mode:
                # Emit the original Triton kernel for debugging purposes
                prefix.writeline("/*")
                prefix.splice(self.kernel_name_to_body[self.kernel_name])
                prefix.writeline("*/")
            self.generate_grid(prefix, inductor_meta, params)
            self.generate_load_kernel(prefix, kernel_var_name, params)
            self.generate_launch_kernel(prefix, wrapper, kernel_var_name, params)
        prefix.writeline("}")

        if not config.aot_inductor.embed_kernel_binary:
            # Ensure the cubin file is included in the package
            V.graph.wrapper_code.additional_files.append(
                params[get_cpp_wrapper_cubin_path_name()]
            )

    def _resolve_lazy_arg_names(self) -> tuple[list[str], list[str]]:
        # [FUNCTION DOC] HELPER: From the triton signature compute wrapper argument names (excluding internal block/split config
        # params and declared constexprs, plus grid args and TMA tensor args appended) and kernel argument names
        # (non-constexpr only). Raises AssertionError on missing triton_meta or count/order mismatches.
        """Compute wrapper and kernel arg names from triton_meta signature.

        Returns (wrapper_arg_names, kernel_arg_names) where:
        - wrapper_arg_names: params accepted by the C++ wrapper function
        - kernel_arg_names: params passed to the GPU kernel launch (non-constexpr only)
        """
        if self.triton_meta is None:
            raise AssertionError(
                f"triton_meta is required for lazy compile of {self.kernel_name}"
            )
        signature = self.triton_meta.get("signature", {})
        inductor_meta = self.inductor_meta or {}
        extra_launcher_args_count = len(inductor_meta.get("extra_launcher_args", []))
        tma_tensor_args = self.tma_tensor_args
        num_tma_tensor_args = len(tma_tensor_args)

        # Matches internal config params like XBLOCK, RSPLIT_SIZE, and their
        # per-subkernel variants like XBLOCK_0, YBLOCK_1.
        internal_config_re = re.compile(r"(?:BLOCK|RSPLIT_SIZE|RSPLIT)(?:_\d+)?$")
        # Declared constexpr params (tl.constexpr in kernel signature) are excluded
        # from arg_types for user-defined kernels, while value-based constexpr params
        # (e.g. numel=1, arg=None) are still in arg_types.
        declared_constexpr_names = OrderedSet(
            inductor_meta.get("declared_constexpr_names", [])
        )
        wrapper_arg_names = []
        kernel_arg_names = []
        for name, sig_type in signature.items():
            if internal_config_re.search(name):
                continue
            if sig_type != "constexpr":
                kernel_arg_names.append(name)
            if name not in declared_constexpr_names:
                wrapper_arg_names.append(name)

        num_wrapper_args = (
            len(self.arg_types) - extra_launcher_args_count - num_tma_tensor_args
        )
        if num_wrapper_args != len(wrapper_arg_names):
            raise AssertionError(
                f"Mismatch between ({num_wrapper_args}) arg_types and "
                f"{len(wrapper_arg_names)} wrapper_arg_names for {self.kernel_name}."
            )

        # Append grid args: passed to wrapper. Kernel args will handle grids separately.
        for i in range(extra_launcher_args_count):
            wrapper_arg_names.append(f"_grid_{i}")

        # Add TMA tensor args after grid args
        if tma_tensor_args:
            sig_tma_keys = list(self._get_tma_args().keys())
            if list(tma_tensor_args.keys()) != sig_tma_keys:
                raise AssertionError(
                    f"TMA tensor args order mismatch for {self.kernel_name}: "
                    f"{list(tma_tensor_args.keys())} vs signature order {sig_tma_keys}"
                )
        for desc_name in tma_tensor_args:
            wrapper_arg_names.append(f"_tma_tensor_{desc_name}")

        return wrapper_arg_names, kernel_arg_names

    def _generate_lazy_grid(self, prefix: IndentedBuffer) -> None:
        # [FUNCTION DOC] HELPER: Emit grid size computation for lazy kernels: a switch over the runtime config index for
        # precomputed grids, otherwise from GridExpr; returns early from the wrapper when grid_0 is zero.
        """Generate grid computation code for lazy-compiled kernels."""
        kernel_name = self.kernel_name
        grid_type = self.inductor_meta.get("grid_type") if self.inductor_meta else None

        # For PrecomputedGrid, generate switch statement on config_index
        if grid_type == "PrecomputedGrid":
            if self.inductor_meta is None:
                raise AssertionError("inductor_meta is required for PrecomputedGrid")
            precomputed_grids = self.inductor_meta.get("precomputed_grids", [])
            extra_launcher_args = self.inductor_meta.get("extra_launcher_args", [])

            switch_cases = []
            for idx, entry in enumerate(precomputed_grids):
                cpp_grids = list(entry.get("cpp", ["1L", "1L", "1L"]))
                # Replace internal arg names with C++ parameter names
                # e.g., _launcher_s0 -> _grid_0
                for i, arg_name in enumerate(extra_launcher_args):
                    cpp_grids = [g.replace(arg_name, f"_grid_{i}") for g in cpp_grids]
                g0 = cpp_grids[0]
                g1 = cpp_grids[1] if len(cpp_grids) > 1 else "1"
                g2 = cpp_grids[2] if len(cpp_grids) > 2 else "1"
                switch_cases.append(
                    f"case {idx}: grid_0 = {g0}; grid_1 = {g1}; grid_2 = {g2}; break;"
                )
            switch_cases.append("default: grid_0 = 1; grid_1 = 1; grid_2 = 1; break;")
            switch_body = "\n                        ".join(switch_cases)

            prefix.splice(
                f"""\
                uint32_t grid_0, grid_1, grid_2;
                switch ({kernel_name}_result.config_index) {{
                    {switch_body}
                }}
                if (grid_0 == 0) return;
                """
            )
        else:
            from ..runtime.triton_heuristics import GridExpr

            grid = GridExpr.from_meta_lazy(
                cast("InductorMeta | None", self.inductor_meta), kernel_name
            )
            for line in grid.prefix:
                prefix.writeline(line)

            prefix.splice(
                f"""\
                uint32_t grid_0 = {grid.x_grid};
                uint32_t grid_1 = {grid.y_grid};
                uint32_t grid_2 = {grid.z_grid};
                if (grid_0 == 0) return;
                """
            )

    def _generate_lazy_tma_args(
        self,
        prefix: IndentedBuffer,
        call_args_str: str,
        kernel_arg_names: list[str],
        tma_arg_names: OrderedSet[str],
        signature: dict[str, str],
    ) -> str:
        # [FUNCTION DOC] HELPER: Append the unpacked descriptor pointer args of each TMA kernel arg to the kernel call arg
        # string and return the new string.
        """Unpack TMA descriptor args into kernel launch args."""
        for arg_name in kernel_arg_names:
            if arg_name in tma_arg_names:
                tma_parts = _unpack_tma_descriptor_args(arg_name, signature[arg_name])
                tma_str = ", ".join(tma_parts)
                call_args_str = (
                    f"{call_args_str}, {tma_str}" if call_args_str else tma_str
                )
        return call_args_str

    def _generate_lazy_scratch(
        self,
        prefix: IndentedBuffer,
        wrapper: CppWrapperGpu,
        call_args_str: str,
    ) -> str:
        # [FUNCTION DOC] HELPER: Emit scratch tensor allocations (global and profile scratch) sized by the runtime result scratch
        # size times the full launch grid, and append their addresses to the call arg string, which is returned.
        """Generate scratch space allocations with runtime-known sizes."""
        kernel_name = self.kernel_name
        dtype_str = wrapper.codegen_dtype(torch.uint8)
        device_type, _ = wrapper.codegen_device(torch.device(get_gpu_type())).split(
            ", "
        )
        device_ptr_type = wrapper.device_codegen.cpp_device_ptr()
        # Triton reports per-CTA scratch via kernel.metadata.global_scratch_size;
        # the kernel writes its slot at offset (pid * scratch_size). Scale by the
        # full launch grid so concurrent CTAs don't collide.
        grid_extent = "static_cast<int64_t>(grid_0) * grid_1 * grid_2"
        for scratch_name in ("global_scratch", "profile_scratch"):
            size_expr = f"{kernel_name}_result.{scratch_name}"
            var = f"{scratch_name}_ptr"
            prefix.splice(
                maybe_hipify_code_wrapper(
                    f"""\
                int64_t {var}_numel = {size_expr} * {grid_extent};
                RAIIAtenTensorHandle {var}_tensor;
                {device_ptr_type} {var} = allocate_scratch_tensor<{device_ptr_type}>(
                    {var}_numel, {dtype_str}, {device_type}, device_idx_, {var}_tensor);
            """
                )
            )
            call_args_str += f", &{var}"
        return call_args_str

    def _generate_lazy_launch(
        self,
        prefix: IndentedBuffer,
        wrapper: CppWrapperGpu,
        wrapper_arg_names: list[str],
        kernel_arg_names: list[str],
    ) -> None:
        # [FUNCTION DOC] HELPER: Emit the lazy kernel launch: declare non-TMA args, add unpacked TMA args and scratch args, build
        # the kernel args array, then launch (JIT launchKernel; AOTI plain or profiled launch).
        """Generate kernel launch code for lazy-compiled kernels."""
        kernel_name = self.kernel_name
        signature = (self.triton_meta or {}).get("signature", {})
        tma_tensor_args = self.tma_tensor_args
        num_tma_tensor_args = len(tma_tensor_args)

        # wrapper_arg_names may include grid and TMA tensor args at the end;
        # only the leading portion maps 1:1 to kernel signature params.
        num_signature_args = len(wrapper_arg_names) - num_tma_tensor_args
        inductor_meta = self.inductor_meta or {}
        num_signature_args -= len(inductor_meta.get("extra_launcher_args", []))

        arg_type_lookup = dict(
            zip(wrapper_arg_names, self.arg_types[:num_signature_args])
        )

        # Identify TMA args — they are already passed as StableTMADescriptor params,
        # so we just unpack them directly (no need to reconstruct from tensors).
        tma_arg_names = OrderedSet(self._get_tma_args().keys())

        # Non-TMA args go through generate_args_decl
        non_tma_arg_names = [n for n in kernel_arg_names if n not in tma_arg_names]
        non_tma_arg_types = [
            arg_type_lookup[n] for n in non_tma_arg_names if n in arg_type_lookup
        ]
        non_tma_arg_sigs = [signature.get(n) for n in non_tma_arg_names]

        call_args_str = wrapper.generate_args_decl(
            prefix,
            non_tma_arg_names,
            non_tma_arg_types,
            non_tma_arg_sigs,
        )

        call_args_str = self._generate_lazy_tma_args(
            prefix, call_args_str, kernel_arg_names, tma_arg_names, signature
        )
        call_args_str = self._generate_lazy_scratch(prefix, wrapper, call_args_str)

        common_launch_args = (
            f"grid_0, grid_1, grid_2,"
            f" {kernel_name}_result.num_warps,"
            f" {kernel_name}_result.shared_mem,"
            f" kernel_args_, stream_"
        )
        # stream_ comes from the generated wrapper signature on both JIT and
        # AOTI sides.
        launch_kernel_args = [
            "grid_0",
            "grid_1",
            "grid_2",
            f"{kernel_name}_result.num_warps",
            f"{kernel_name}_result.shared_mem",
        ]

        # kernel_args_ is consumed by both JIT and AOT launchKernel calls.
        prefix.writeline(f"void* kernel_args_[] = {{{call_args_str}}};")
        enable_kernel_profile = config.cpp.enable_kernel_profile and sys.platform in [
            "linux",
            "win32",
        ]
        prefix.writeline_jit(f"launchKernel({kernel_name}, {common_launch_args});")
        if enable_kernel_profile:
            profile_arg_types = [arg_type_lookup.get(n) for n in kernel_arg_names]
            profile_arg_sigs = [signature.get(n) for n in kernel_arg_names]
            aot_profile = IndentedBuffer(initial_indent=prefix._indent)
            self.generate_profiled_launch_kernel(
                aot_profile,
                f"kernels_.{kernel_name}",
                kernel_arg_names,
                profile_arg_types,
                profile_arg_sigs,
                [
                    f"kernels_.{kernel_name}",
                    *launch_kernel_args,
                    "kernel_args_",
                    "stream_",
                ],
                num_warps=f"{kernel_name}_result.num_warps",
                shared_mem=f"{kernel_name}_result.shared_mem",
            )
            prefix.splice_aot(aot_profile)
        else:
            prefix.writeline_aot(
                f"launchKernel(kernels_.{kernel_name}, {common_launch_args});"
            )

    def generate_lazy(self, wrapper: CppWrapperGpu):
        # [FUNCTION DOC] Emit the dual JIT/AOTI wrapper for a lazily compiled kernel: records the kernel name, emits the static
        # kernel handle, embedded Triton source and result struct (runtime filled for JIT, compile-time macros for
        # AOTI), the signature, a first-call init (JIT compiles and autotunes then returns after the first run; AOTI
        # loads the cubin), then shared grid and launch code. Scalar tensor args are extracted for autotuning.
        """
        Generate dual-wrapper-mode C++ code for lazy Triton kernel compilation.

        DualIndentedBuffer routes lines into separate JIT and AOTI sources:
        - JIT side: embeds Triton source, compiles at runtime, autotunes
          with real inputs.
        - AOTI side: uses a compile-time-initialized LazyKernelCompileResult
          from a config header generated after the JIT first-pass.

        Grid computation and kernel launch are shared between both sides
        via the LazyKernelCompileResult struct.
        """
        prefix = wrapper.prefix
        kernel_name = self.kernel_name
        macro_prefix = kernel_name.upper()

        # Track kernel names for parallel initialization (JIT only)
        wrapper._lazy_kernel_names.append(kernel_name)

        # Include TMA helpers if any args use TMA descriptors
        tma_signature_types = self._get_tma_args()
        if tma_signature_types:
            wrapper.write_tma_descriptor_helpers_once()

        kernel_type = maybe_hipify_code_wrapper(
            wrapper.device_codegen.cpp_kernel_type()
        )

        # JIT-only: static CUfunction and embedded Triton source
        prefix.writeline_jit(f"static {kernel_type} {kernel_name} = nullptr;")
        kernel_source_str = self.kernel_name_to_body.get(kernel_name, "")
        kernel_body = f'R"TRITON(\n{kernel_source_str}\n)TRITON"'
        prefix.writeline_jit(
            f"static const char* {kernel_name}_source = {kernel_body};"
        )

        # LazyKernelCompileResult: JIT fills at runtime; AOTI uses compile-time
        # init from the config header generated after the first pass.
        prefix.writeline_jit(f"static LazyKernelCompileResult {kernel_name}_result;")
        prefix.splice_aot(
            f"""\
            static LazyKernelCompileResult {kernel_name}_result = {{
                {macro_prefix}_CUBIN_PATH,
                {macro_prefix}_MANGLED_NAME,
                {macro_prefix}_NUM_WARPS,
                {macro_prefix}_SHARED_MEM,
                {macro_prefix}_XBLOCKS,
                {macro_prefix}_YBLOCKS,
                {macro_prefix}_ZBLOCKS,
                {macro_prefix}_R0BLOCKS,
                {macro_prefix}_RSPLIT,
                {macro_prefix}_RSPLIT_SIZE,
                {macro_prefix}_CONFIG_INDEX,
                {macro_prefix}_GLOBAL_SCRATCH,
                {macro_prefix}_PROFILE_SCRATCH,
            }};
            """
        )

        wrapper_arg_names, kernel_arg_names = self._resolve_lazy_arg_names()
        signature = (self.triton_meta or {}).get("signature", {})

        # kernels_type_/kernels_ are routed to the AOTI buffer; if prefix is a
        # plain IndentedBuffer (pure JIT lazy compile) those writes are dropped.
        self._write_wrapper_signature(
            prefix,
            wrapper,
            wrapper_arg_names,
            self.arg_types,
            signature,
        )

        # Build autotune args - for TMA, pass tensors instead of descriptors.
        # Only iterate over signature params and grid args, not the trailing
        # TMA tensor params (those are only in the C++ wrapper signature).
        tma_tensor_args = self.tma_tensor_args
        num_autotune_args = len(wrapper_arg_names) - len(tma_tensor_args)
        autotune_arg_list = []
        # Track which args need scalar extraction for the autotune call.
        # UnwrapUnspecArg args are 0-dim tensors in C++ that Triton expects
        # as Python scalars; we use codegen_tensor_item to extract them.
        scalar_extractions: list[tuple[str, str, torch_dtype]] = []
        for idx, name in enumerate(wrapper_arg_names[:num_autotune_args]):
            if name in tma_signature_types:
                autotune_arg_list.append(f"_tma_tensor_{name}")
            elif isinstance(self.arg_types[idx], UnwrapUnspecArg):
                scalar_var = f"_autotune_scalar_{name}"
                scalar_extractions.append((name, scalar_var, self.arg_types[idx].dtype))
                autotune_arg_list.append(scalar_var)
            else:
                autotune_arg_list.append(name)
        autotune_args = ", ".join(autotune_arg_list)

        with prefix.indent():
            # First-call initialization: JIT lazy compiles, AOTI loads cubin

            # JIT: lazy compile with autotuning on first invocation.
            # Build into temp buffer to avoid DualIndentedBuffer dispatch.
            jit_init = IndentedBuffer(initial_indent=prefix._indent)
            jit_init.writeline(f"if ({kernel_name} == nullptr) {{")
            with jit_init.indent():
                for tensor_name, scalar_var, dtype in scalar_extractions:
                    wrapper.codegen_tensor_item(
                        dtype, tensor_name, scalar_var, indented_buffer=jit_init
                    )
                jit_init.splice(
                    f"""\
                    {kernel_name}_result = runTritonKernelWithAutotune(
                        _module_pending_kernels, "{kernel_name}", stream_, {autotune_args});

                    {kernel_name} = loadKernel(
                        {kernel_name}_result.cubin_path,
                        {kernel_name}_result.mangled_name,
                        {kernel_name}_result.shared_mem);

                    // First invocation already ran the kernel, so return early
                    return;
                    """
                )
            jit_init.writeline("}")
            prefix.splice_jit(jit_init)

            # AOTI: load precompiled cubin from compile-time-initialized result.
            loaded_modules_arg = (
                ",\n                        &kernels_.loaded_modules_"
                if V.graph.device_type != "xpu"
                else ""
            )
            aoti_init = IndentedBuffer(initial_indent=prefix._indent)
            aoti_init.writeline(f"if (kernels_.{kernel_name} == nullptr) {{")
            with aoti_init.indent():
                aoti_init.splice(
                    f"""\
                    kernels_.{kernel_name} = loadKernel(
                        {kernel_name}_result.cubin_path,
                        {kernel_name}_result.mangled_name,
                        {kernel_name}_result.shared_mem,
                        cubin_dir_{loaded_modules_arg});
                    """
                )
            aoti_init.writeline("}")
            prefix.splice_aot(aoti_init)

            # Shared: grid computation and launch using result struct
            self._generate_lazy_grid(prefix)
            self._generate_lazy_launch(
                prefix,
                wrapper,
                wrapper_arg_names,
                kernel_arg_names,
            )
        prefix.writeline("}")

    def generate_grid(
        self,
        prefix: IndentedBuffer,
        inductor_meta: dict[str, Any],
        params: dict[str, Any],
    ):
        # [FUNCTION DOC] Emit grid_0/1/2 computation from GridExpr in cpp mode, plus an early return if any grid dimension is zero.
        from ..runtime.triton_heuristics import GridExpr

        grid = GridExpr.from_meta(
            cast("InductorMeta", inductor_meta), params["config"], mode="cpp"
        )
        for line in grid.prefix:
            prefix.writeline(line)
        prefix.splice(
            f"""\
            uint32_t grid_0 = {grid.x_grid};
            uint32_t grid_1 = {grid.y_grid};
            uint32_t grid_2 = {grid.z_grid};
            """
        )
        prefix.writeline("if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;")

    def generate_load_kernel(self, prefix, kernel_var_name, params):
        # [FUNCTION DOC] Emit a guarded load of the kernel on first use, passing either embedded binary symbols (AOT embed mode,
        # plus XPU end symbol and format flag) or the cubin path, mangled name, shared memory size and cubin dir;
        # in AOT mode on non-XPU devices also passes the loaded modules vector so modules can be unloaded.
        prefix.writeline(f"if ({kernel_var_name} == nullptr) {{")
        with prefix.indent():
            embed_kernel_args = [f"__{params['inductor_meta']['kernel_name']}_start"]
            if torch.xpu.is_available():
                # XPU needs the end address of the kernel to calculate the size of the kernel binary.
                embed_kernel_args.append(
                    f"__{params['inductor_meta']['kernel_name']}_end"
                )

            if V.graph.aot_mode and config.aot_inductor.embed_kernel_binary:
                load_kernel_args = [
                    *embed_kernel_args,
                    cpp_string_literal(params["mangled_name"]),
                    str(params["shared_mem"]),
                ]
                if torch.xpu.is_available():
                    is_spv = "true" if XPU_KERNEL_FORMAT == "spv" else "false"
                    if config.aot_inductor.emit_multi_arch_kernel:
                        is_spv = "true"
                    load_kernel_args.append(is_spv)
            else:
                load_kernel_args = [
                    cpp_string_literal(params[get_cpp_wrapper_cubin_path_name()]),
                    cpp_string_literal(params["mangled_name"]),
                    str(params["shared_mem"]),
                    "cubin_dir_",
                ]

            # In AOTI mode on CUDA/HIP, pass the loaded_modules_ vector so
            # CUmodule handles are tracked and can be unloaded on destruction,
            # preventing GPU code object leaks. XPU is excluded because its
            # loadKernel returns std::unique_ptr<sycl::kernel> and manages
            # cleanup via RAII.
            if V.graph.aot_mode and V.graph.device_type != "xpu":
                load_kernel_args = load_kernel_args + ["&kernels_.loaded_modules_"]

            prefix.writeline(
                f"{kernel_var_name} = loadKernel({', '.join(load_kernel_args)}); "
            )
        prefix.writeline("}")

    def generate_launch_kernel(self, prefix, wrapper, kernel_var_name, params):
        # [FUNCTION DOC] Emit the launch: compute launch args (skipping constants) with arg_types/signatures, scale scratch sizes by
        # num_ctas, declare args, build the kernel args array and call launchKernel, or the profiled variant when kernel
        # profiling is enabled on linux/win32. Raises AssertionError if arg_types and def_args lengths differ.
        """
        Generate the GPU kernel launching code.
        This is where all the call args are sorted out and generated.
        If enable_kernel_profile is enabled, all args related information would be packed in this function.
        """
        triton_meta = params["triton_meta"]
        if len(self.arg_types) != len(params["def_args"]):
            raise AssertionError((self.arg_types, params["def_args"]))
        arg_type_lookup = dict(zip(params["def_args"], self.arg_types))
        # difference between Python and C++ wrapper: C++ wrapper strips out equal_to_1 constants
        call_args = [
            name for name in params["call_args"] if name not in triton_meta["constants"]
        ]
        arg_types = [arg_type_lookup[name] for name in call_args]
        arg_signatures = [triton_meta["signature"][name] for name in call_args]
        num_ctas = params.get("config", {}).get("num_ctas", 1)
        scratch_spaces = {
            name: params[name] * num_ctas
            for name in ["global_scratch", "profile_scratch"]
            if params.get(name, None) is not None
        }
        call_args_str = wrapper.generate_args_decl(
            prefix,
            call_args,
            arg_types,
            arg_signatures,
            scratch_spaces=scratch_spaces,
        )
        prefix.writeline(f"void* kernel_args_[] = {{{call_args_str}}};")
        num_warps = str(params["num_warps"])
        shared_mem = str(params["shared_mem"])
        launch_kernel_args = [
            kernel_var_name,
            "grid_0",
            "grid_1",
            "grid_2",
            num_warps,
            shared_mem,
            "kernel_args_",
            "stream_",
        ]

        enable_kernel_profile = config.cpp.enable_kernel_profile and sys.platform in [
            "linux",
            "win32",
        ]
        if enable_kernel_profile:
            self.generate_profiled_launch_kernel(
                prefix,
                kernel_var_name,
                call_args,
                arg_types,
                arg_signatures,
                launch_kernel_args,
                num_warps=num_warps,
                shared_mem=shared_mem,
            )
        else:
            prefix.writeline(f"launchKernel({', '.join(launch_kernel_args)});")

    def generate_profiled_launch_kernel(
        self,
        prefix: IndentedBuffer,
        kernel_var_name: str,
        call_args: list[str],
        arg_types: list[Any],
        arg_signatures: list[str | None],
        launch_kernel_args: list[str],
        num_warps: str,
        shared_mem: str,
    ) -> None:
        # [FUNCTION DOC] Emit a block that records launch args and input names/shapes as IValues for an AOTI record function
        # scope (non-tensor args as dummy scalars, TMA descriptors skipped), starts the record, then launches the kernel.
        """Wrap a kernel launch in an AOTI record_function profiling scope."""
        normalized_kernel_name = re.sub(r"[^a-zA-Z0-9_]", "_", kernel_var_name)
        prefix.writeline("{")
        with prefix.indent():
            prefix.writelines(
                [
                    f"std::unordered_map<std::string, C10IValueHandle> kwargs_{normalized_kernel_name};",
                    "",
                ]
            )
            # Add launch args info
            record_launch_kernel_args = [
                ("grid_0", "grid_0"),
                ("grid_1", "grid_1"),
                ("grid_2", "grid_2"),
                ("num_warps", num_warps),
                ("shared_mem", shared_mem),
            ]
            for k, v in record_launch_kernel_args:
                arg_name = f"{normalized_kernel_name}_{k}"
                prefix.writelines(
                    [
                        f"// Create c10::IValue for {k}",
                        f"C10IValueHandle tmp_{arg_name};",
                        f"aoti_torch_int64_to_ivalue({v}, &tmp_{arg_name});",
                        f"RAIIC10IValueHandle RAII_{arg_name}(tmp_{arg_name});",
                        f'kwargs_{normalized_kernel_name}.emplace("{k}", RAII_{arg_name});',
                    ]
                )

            # Add input info (This copies the logic from args_decl)
            curr_arg_id = -1
            total_args = []
            ordered_argsname = []

            def write_dummy_scalar_ivalue(arg_name):
                # [FUNCTION DOC] HELPER: Emit a dummy int64 IValue for the current argument and remember its handle.
                # We only care about the shape, therefore we create a dummy scalar here.
                prefix.writelines(
                    [
                        f"// Create c10::IValue for arg_{curr_arg_id}",
                        f"C10IValueHandle tmp_{arg_name};",
                        f"aoti_torch_int64_to_ivalue(0, &tmp_{arg_name});",
                        f"RAIIC10IValueHandle RAII_{arg_name}(tmp_{arg_name});",
                    ]
                )
                # pyrefly: ignore [bad-argument-type]
                total_args.append(f"tmp_{arg_name}")

            def process_args_for_input_shape(arg, arg_type, arg_signature=None):
                # [FUNCTION DOC] HELPER: Advance the arg counter and emit the IValue for one kernel argument by kind (tensor, scalar,
                # unwrapped scalar, or skipped TMA descriptor).
                nonlocal curr_arg_id
                curr_arg_id += 1
                arg_name = f"{normalized_kernel_name}_arg_{curr_arg_id}"
                # ignore tma descriptors, as host-side TMA descriptors need
                # to be passed to the compiled Triton kernel by value
                if isinstance(arg_type, UnwrapUnspecArg) and not signature_is_tma_desc(
                    arg_signature
                ):
                    write_dummy_scalar_ivalue(arg_name)
                elif isinstance(arg_type, torch_dtype) and not signature_is_tma_desc(
                    arg_signature
                ):
                    # This is an at::Tensor.
                    prefix.writelines(
                        [
                            f"// Create c10::IValue for arg_{curr_arg_id}",
                            f"C10IValueHandle tmp_{arg_name};",
                            f"aoti_torch_tensor_to_ivalue({arg}, &tmp_{arg_name});",
                            f"RAIIC10IValueHandle RAII_{arg_name}(tmp_{arg_name});",
                        ]
                    )
                    # pyrefly: ignore [bad-argument-type]
                    total_args.append(f"tmp_{arg_name}")
                elif (
                    isinstance(arg_type, type(SymbolicCallArg))
                    and arg_signature is not None
                    and arg_signature in TRITON_SIGNATURE_TO_CPP
                ) or arg_type in (sympy.Integer, int, sympy.Float, float):
                    write_dummy_scalar_ivalue(arg_name)
                elif arg_signature and arg_signature.startswith("tensordesc<"):
                    # Skip tma related args
                    pass
                else:
                    write_dummy_scalar_ivalue(arg_name)

            # Add input name and shape information
            for arg, arg_type, arg_signature in zip_longest(
                call_args, arg_types, arg_signatures
            ):
                # pyrefly: ignore [bad-argument-type]
                ordered_argsname.append(f'"{arg}"')
                process_args_for_input_shape(arg, arg_type, arg_signature)

            # Add input name into kwargs
            name_var = f"{normalized_kernel_name}_input_names"
            prefix.writelines(
                [
                    "// Create c10::IValue for input names",
                    f"C10IValueHandle tmp_{name_var};",
                    f"std::vector<const char*> {name_var}({{{', '.join(ordered_argsname)}}});",
                    f"aoti_torch_strlist_to_ivalue({name_var}.data(), {len(ordered_argsname)}, &tmp_{name_var});",
                    f"RAIIC10IValueHandle RAII_{name_var}(tmp_{name_var});",
                    f'kwargs_{normalized_kernel_name}.emplace("Input Args", RAII_{name_var});',
                ]
            )

            inputs_info_ = f"{normalized_kernel_name}_inputs_info_"
            # We pass in the non-RAII handles, since C10 doesn't automatically free them.
            # The RAII will make sure they get freed when they are out of scope.
            tmp_args = ",".join(total_args)
            prefix.writelines(
                [
                    "// Aggregate all c10::IValue for inputs",
                    f"std::vector<C10IValueHandle> {inputs_info_}({{{tmp_args}}});",
                ]
            )

            # Start recording Function
            prefix.writelines(
                [
                    "",
                    (
                        "torch::aot_inductor::RAIIAtenRecordFunctionHandle "
                        f"record_{normalized_kernel_name}_"
                        f'("{kernel_var_name}", '
                        f"reinterpret_cast<IValueMapHandle>(&kwargs_{normalized_kernel_name}), "
                        f"{inputs_info_});"
                    ),
                    "",
                    f"launchKernel({', '.join(launch_kernel_args)});",
                ]
            )
        prefix.writeline("}")


class CppWrapperGpu(CppWrapperCpu):
    # [CLASS DOC] Wrapper codegen subclass of CppWrapperCpu for GPU devices. Emits C++ kernel launch code via deferred
    # Triton call wrappers, handles TMA descriptors, input alignment copies, debug sync, and multi-stream AOTI
    # event/stream code. State: device and device_codegen overrides, grid_id counter, kernel body map, triton
    # call wrapper map, lazy kernel names, declared aux stream slots, and stream guard / helper emission flags.
    """
    Generates cpp wrapper for running on GPU and calls CUDA kernels
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Set the GPU device type and its device op overrides, then initialize the base wrapper and GPU state.
        self.device = get_gpu_type()
        self.device_codegen = get_device_op_overrides(self.device)
        super().__init__()
        self.grid_id = count()
        self._kernel_name_to_body: dict[str, str] = {}
        self._triton_call_wrappers: dict[str, DeferredTritonCallWrapper] = {}
        self.autotune_input_prefix = "_REAL_AUTOTUNE_INPUT"
        self._lazy_kernel_names: list[str] = []
        self._declared_aux_stream_slots: OrderedSet[int] = OrderedSet()
        self._aoti_current_stream_guard_declared = False
        self._aoti_stream_helpers_emitted = False

    def generate_debug_sync(self, buffer):
        # [FUNCTION DOC] Emit a device synchronization into the buffer for cuda (driver cuCtxSynchronize, or runtime device sync on
        # ROCm). Raises NotImplementedError for other devices.
        if self.device == "cuda":
            # The fbcode JIT cpp_wrapper CUDA build links only the CUDA driver
            # (libcuda), not libcudart, so the runtime cudaDeviceSynchronize symbol
            # is undefined at dlopen -> use the driver-API cuCtxSynchronize there.
            # On ROCm the driver-context sync hipCtxSynchronize returns
            # hipErrorNotSupported at runtime, so keep the runtime cudaDeviceSynchronize
            # (which hipifies to hipDeviceSynchronize and IS linked in the ROCm build).
            if torch.version.hip is not None:
                buffer.writeline(
                    maybe_hipify_code_wrapper(
                        "AOTI_RUNTIME_CUDA_CHECK(cudaDeviceSynchronize());"
                    )
                )
            else:
                buffer.writeline("CUDA_DRIVER_CHECK(cuCtxSynchronize());")
            return

        raise NotImplementedError(
            f"triton debug sync is not supported with {self.device} cpp_wrapper"
        )

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: PythonWrapperCodegen | None,
        partition_signatures: GraphPartitionSignature | None = None,
    ):
        # [FUNCTION DOC] Factory ignoring its arguments; returns a new CppWrapperGpu.
        # TODO - support subgraph codegen by lifting functions. Check the
        # comment at CppWrapperCpu `codegen_subgraph` function.
        return CppWrapperGpu()

    def write_header(self):
        # [FUNCTION DOC] Write the file header; const graphs skip it unless in dual wrapper mode, in which case only JIT device
        # includes and the kernel driver code are emitted for them. Otherwise adds the (hipified) kernel driver code.
        if V.graph.is_const_graph and not V.graph.is_dual_wrapper_mode:
            # We do not write header for constant graph, it will be written by main module.
            return

        super().write_header()
        kernel_driver = maybe_hipify_code_wrapper(self.device_codegen.kernel_driver())
        if V.graph.is_const_graph and V.graph.is_dual_wrapper_mode:
            # For a dual-wrapper-mode const graph, only the standalone JIT
            # output needs this header content. The AOTI const body is spliced
            # into the main AOTI source, which has its own kernel driver.
            # super().write_header() early-returns for const graphs before it
            # can call add_device_include, so emit the JIT device include here;
            # otherwise the kernel driver's CUfunction/CUmodule/uint32_t types
            # have no declaring header and fail to compile under -nostdinc.
            for device in V.graph.device_types:
                if device != "meta":
                    self.header.splice_jit(self.get_device_include_path_jit(device))
            self.header.splice_jit(kernel_driver)
        else:
            self.header.splice(kernel_driver)

    def _generate(self, is_inference):
        # [FUNCTION DOC] HELPER: Reset per-run auxiliary stream state, then run the base class generation. Stream helper emission
        # flag is deliberately not reset to avoid redefinitions.
        # Per-Run()-function state, reset each generation. Do NOT reset
        # _aoti_stream_helpers_emitted here: the helper structs are spliced into
        # the file-level header once per instance, and _generate can run more
        # than once against the same header (resetting it re-splices and yields
        # a C++ redefinition).
        self._declared_aux_stream_slots.clear()
        self._aoti_current_stream_guard_declared = False
        return super()._generate(is_inference)

    @cache_on_self
    def write_tma_descriptor_helpers_once(self):
        # [FUNCTION DOC] Splice the device TMA descriptor helper code into the header, once per instance (cached).
        self.header.splice(self.device_codegen.tma_descriptor_helpers())

    def write_get_raw_stream(self, device_idx: int, graph_name: str) -> str:
        # [FUNCTION DOC] Return the name of the stream expression: stream parameter in pure AOT mode, a predeclared name in dual
        # mode, or a newly declared variable filled by the get-stream shim (emits lines) otherwise.
        # Pure AOTI receives the stream as a function parameter. JIT and
        # dual-wrapper-mode code use an explicit stream variable so the shared kernel
        # call arguments are valid for the JIT entry point.
        if V.graph.aot_mode and not V.graph.is_dual_wrapper_mode:
            return "stream"

        name = f"stream{device_idx}"
        # In dual-wrapper mode, the JIT stream is declared at the entry function
        # prologue (see _codegen_entry_impl_prologue) so it stays in scope
        # across all kernel call sites.
        if V.graph.is_dual_wrapper_mode:
            return name

        self.writeline(
            maybe_hipify_code_wrapper(
                f"{self.device_codegen.cpp_stream_type()} {name};"
            )
        )
        self.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK({self.device_codegen.aoti_get_stream()}({device_idx}, (void**)&{name}));"
        )
        return name

    def _ensure_aoti_stream_helpers_emitted(self) -> None:
        # [FUNCTION DOC] HELPER: Once per instance, splice the CUDA-only stream/event helper header and thread-local caches into
        # the header; skipped for XPU. Reads streams.h from disk.
        if self._aoti_stream_helpers_emitted:
            return
        # The stream/event helpers in streams.h are CUDA-specific (cudaEvent_t,
        # cudaStream_t, cudaEventRecord, ...). Guarding here on the device type
        # prevents the CUDA-only symbols from being emitted into XPU generated
        # code, where SYCL in-order queues handle event ordering implicitly.
        if self.device == "xpu":
            return
        self._aoti_stream_helpers_emitted = True
        with open(
            os.path.join(os.path.dirname(__file__), "aoti_runtime", "streams.h")
        ) as f:
            self.header.splice(maybe_hipify_code_wrapper(f.read()))
        self.header.splice(
            """
            namespace {

            static thread_local torch::aot_inductor::AOTIPerThreadEventCache
                _aoti_event_cache;
            static thread_local torch::aot_inductor::AOTIPerThreadStreamCache
                _aoti_aux_stream_cache;

            }  // namespace
            """
        )

    def codegen_stream_info_prologue(
        self,
        code: IndentedBuffer,
        num_streams: int,
        stream_idx_to_user_obj_idx: dict[int, int],
    ) -> None:
        # [FUNCTION DOC] Emit setup for multi-stream AOTI: ensure helpers, aux stream cache sizing, a stream guard variable, and
        # declarations for auxiliary streams not yet declared. Does nothing for one stream; raises NotImplementedError
        # when not in AOT mode.
        if num_streams <= 1:
            return
        if not V.graph.aot_mode:
            raise NotImplementedError(
                "Multi-stream cpp_wrapper codegen is only supported for AOTI."
            )
        self._ensure_aoti_stream_helpers_emitted()
        code.writeline(
            f"_aoti_aux_stream_cache.ensure({num_streams}, this->device_idx_, stream);"
        )
        if not self._aoti_current_stream_guard_declared:
            code.writeline(
                f"std::unique_ptr<{V.graph.device_ops.cpp_aoti_stream_guard()}> "
                "_aoti_current_stream_guard;"
            )
            self._aoti_current_stream_guard_declared = True

        stream_type = self.device_codegen.cpp_stream_type()
        for i in range(1, num_streams):
            if i in self._declared_aux_stream_slots:
                continue
            code.writeline(
                maybe_hipify_code_wrapper(
                    f"{stream_type} {get_stream_name(i)} = "
                    f"_aoti_aux_stream_cache.get({i}, this->device_idx_, stream);"
                )
            )
            self._declared_aux_stream_slots.add(i)

    def codegen_enter_cuda_stream_context(
        self, code: IndentedBuffer, stream_idx: int
    ) -> None:
        # [FUNCTION DOC] Emit code entering the stream context for a non-default stream index by creating a stream guard;
        # no-op for index zero.
        if stream_idx == 0:
            return
        code.writeline(
            "_aoti_current_stream_guard = "
            f"std::make_unique<{V.graph.device_ops.cpp_aoti_stream_guard()}>("
            f"{self._stream_expr_for_idx(stream_idx)}, this->device_idx_);"
        )

    def codegen_exit_cuda_stream_context(self, code: IndentedBuffer) -> None:
        # [FUNCTION DOC] Emit code resetting the stream guard to leave the stream context.
        code.writeline("_aoti_current_stream_guard.reset();")

    def _stream_expr_for_idx(self, stream_idx: int) -> str:
        # [FUNCTION DOC] HELPER: Return the stream expression for an index: the main stream for zero, else the cached aux stream.
        if stream_idx == 0:
            return "stream"
        return f"_aoti_aux_stream_cache.get({stream_idx}, this->device_idx_, stream)"

    def _emit_stream_op_inline(self, kernel_name: str | None, args: list[str]) -> bool:
        # [FUNCTION DOC] HELPER: In AOT mode on non-XPU devices, emit inline event record/wait/synchronize calls for supported
        # stream ops and return True; return False if not applicable. Raises NotImplementedError for ops known to be
        # unsupported.
        if kernel_name is None or not V.graph.aot_mode:
            return False
        if self.device == "xpu":
            return False
        if kernel_name in AOTI_UNSUPPORTED_STREAM_OP_REASONS:
            raise NotImplementedError(
                f"{kernel_name} is not supported in AOTI cpp_wrapper. "
                f"{AOTI_UNSUPPORTED_STREAM_OP_REASONS[kernel_name]}"
            )
        op = AOTI_SUPPORTED_STREAM_OP_NAMES.get(kernel_name)
        if op is None:
            return False

        def _parse_idx(arg: str) -> int:
            # [FUNCTION DOC] HELPER: Parse a C integer literal string with optional L suffix into an int.
            return int(arg.rstrip("Ll"))

        self._ensure_aoti_stream_helpers_emitted()
        event_idx = _parse_idx(args[0])
        if op == "record_event":
            stream_idx = _parse_idx(args[1])
            self.writeline(
                maybe_hipify_code_wrapper(
                    "AOTI_RUNTIME_CUDA_CHECK(cudaEventRecord("
                    f"_aoti_event_cache.get({event_idx}, this->device_idx_), "
                    f"{self._stream_expr_for_idx(stream_idx)}));"
                )
            )
            return True
        if op == "wait_event":
            stream_idx = _parse_idx(args[1])
            self.writeline(
                maybe_hipify_code_wrapper(
                    "AOTI_RUNTIME_CUDA_CHECK(cudaStreamWaitEvent("
                    f"{self._stream_expr_for_idx(stream_idx)}, "
                    f"_aoti_event_cache.get({event_idx}, this->device_idx_), 0));"
                )
            )
            return True
        if op == "synchronize_event":
            self.writeline(
                maybe_hipify_code_wrapper(
                    "AOTI_RUNTIME_CUDA_CHECK(cudaEventSynchronize("
                    f"_aoti_event_cache.get({event_idx}, this->device_idx_)));"
                )
            )
            return True
        return False

    def _generate_extern_kernel_alloc_helper(self, extern_kernel, args):
        # [FUNCTION DOC] Emit extern kernel allocation unless it is an inline-able stream op (then drop the extern node record);
        # otherwise delegate to the base class.
        kernel_name = getattr(extern_kernel, "python_kernel_name", None)
        if self._emit_stream_op_inline(kernel_name, args):
            if V.extern_kernel_nodes:
                V.extern_kernel_nodes.pop()
            return
        super()._generate_extern_kernel_alloc_helper(extern_kernel, args)

    def generate_fallback_kernel_with_runtime_lookup(
        self,
        buf_name,
        python_kernel_name,
        get_args,
        op_overload,
        raw_args,
        outputs,
    ):
        # [FUNCTION DOC] Emit a runtime-lookup fallback kernel unless it is an inline stream op (then drop the extern node record);
        # otherwise delegate to the base class.
        if self._emit_stream_op_inline(python_kernel_name, list(get_args())):
            if V.extern_kernel_nodes:
                V.extern_kernel_nodes.pop()
            return
        super().generate_fallback_kernel_with_runtime_lookup(
            buf_name,
            python_kernel_name,
            get_args,
            op_overload,
            raw_args,
            outputs,
        )

    def get_autotuning_input_name(self, idx):
        # [FUNCTION DOC] Return the generated name of the idx-th autotune input.
        return f"{self.autotune_input_prefix}_{idx}"

    def codegen_inputs(self):
        # [FUNCTION DOC] Emit input handling; in AOT mode (outside fbcode) first emit AOT-only code copying inputs that are
        # misaligned at runtime into aligned clones with a warning. Raises AssertionError for non-tensor checked inputs.
        # See Note: [Input Alignment handling in Inductor]
        #
        # JIT Inductor does not guard on input alignment. It relies on copy_misaligned_inputs to
        # copy misaligned inputs to aligned buffers. For AOTInductor, we need to do the same in cpp.

        if config.is_fbcode():
            # TODO: This is added because FC. Remove this once the newly added shim symbols,
            # e.g. aoti_torch_clone_preserve_strides, have landed
            return super().codegen_inputs()

        if V.graph.aot_mode and V.graph.inputs_to_check:
            for idx in V.graph.inputs_to_check:
                input_name = V.graph.graph_input_names[idx]
                if input_name not in V.graph.graph_inputs:
                    raise AssertionError(f"{input_name} not found in graph inputs")
                value = V.graph.graph_inputs[input_name]
                if not isinstance(value, TensorBox):
                    raise AssertionError(
                        f"{input_name} is expected to be tensor but found as {type(value)}"
                    )
                warn_msg = (
                    f"Input {idx} was compiled as {GPU_ALIGN_BYTES}-bytes aligned, "
                    "but it is not aligned at run time. Copying to an aligned tensor "
                    "to guarantee correctness, but expect a performance hit."
                )
                alignment_check = f"""
                    if ((reinterpret_cast<std::uintptr_t>({input_name}.data_ptr()) & ({GPU_ALIGN_BYTES} -1)) != 0) {{
                        AOTI_TORCH_WARN("{warn_msg}");
                        AtenTensorHandle {input_name}_aligned;
                        AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_clone_preserve_strides({input_name}, &{input_name}_aligned));
                        {input_name} = std::move(RAIIAtenTensorHandle({input_name}_aligned));
                    }}
                    """
                self.prefix.splice_aot(alignment_check)

        super().codegen_inputs()

    def _define_kernel_helper(
        self,
        kernel_name: str,
        kernel_body: str,
        metadata: str | None = None,
        gpu: bool = True,
        cpp_definition: str | None = None,
    ):
        # [FUNCTION DOC] HELPER: For GPU kernels store the kernel body, and (when autotuning at compile time) also create the python
        # autotune block; for cpu kernels delegate to the CPU wrapper and return its result.
        if gpu:
            self._kernel_name_to_body[kernel_name] = kernel_body
            if config.triton.autotune_at_compile_time:
                # Call PythonWrapperCodegen to create the autotune code block
                PythonWrapperCodegen._define_kernel_helper(
                    self, kernel_name, kernel_body, metadata, gpu, cpp_definition
                )
        else:
            return CppWrapperCpu._define_kernel_helper(
                self, kernel_name, kernel_body, metadata, gpu, cpp_definition
            )

    def generate(self, is_inference):
        # [FUNCTION DOC] Run base class generation under a timing context and return its result.
        with dynamo_timed("CppWrapperGpu.generate", log_pt2_compile_event=True):
            return super().generate(is_inference)

    def _codegen_entry_impl_prologue(self):
        # [FUNCTION DOC] HELPER: After the base prologue, emit (JIT only) the lazy Triton compile kickoff call, and in dual wrapper
        # mode declare the JIT streams per device index.
        super()._codegen_entry_impl_prologue()
        # ensure_triton_kernel_compiles_started() is JIT-only; AOTI has no
        # Python-dependent lazy compile flow.
        self.prefix.writeline_jit(
            _LazyTritonCompileKickoffLine(
                self._lazy_kernel_names, "ensure_triton_kernel_compiles_started();"
            )
        )
        # In dual-wrapper mode, hoist the JIT-side stream declaration to the entry
        # function prologue. Kernel calls run inside KernelContextGuard
        # scopes, so a per-call declaration would be scoped to the guard
        # and unavailable to other kernel calls in the same function.
        if V.graph.is_dual_wrapper_mode:
            stream_type = maybe_hipify_code_wrapper(
                self.device_codegen.cpp_stream_type()
            )
            get_stream = self.device_codegen.aoti_get_stream()
            for device_idx in sorted(V.graph.device_idxs):
                name = f"stream{device_idx}"
                self.prefix.writeline_jit(f"{stream_type} {name};")
                self.prefix.writeline_jit(
                    f"AOTI_TORCH_ERROR_CODE_CHECK("
                    f"{get_stream}({device_idx}, (void**)&{name}));"
                )

    def finalize_prefix(self):
        # [FUNCTION DOC] Generate the Triton call wrappers into a separate buffer (after autotuning is finished), emit the parallel
        # lazy compile init functions (JIT only) when there are lazy kernels, run the base finalize, and assemble the
        # prefix as base prefix content, Triton wrappers, then old prefix. Mutates self.prefix.
        """Define the triton kernels now that autotuning is finished"""
        old_prefix = self.prefix  # new content should go at start of prefix

        # Generating triton kernel callers can modify the prefix (cached dtypes),
        # so do this before running finalize_prefix(), but put the generated code
        # after the finalize_prefix() code.
        triton_prefix = make_codegen_buffer()
        with self._target_buf("prefix", triton_prefix):
            for kernel in self._triton_call_wrappers.values():
                self.prefix.writeline("\n")
                kernel.generate(self)

            # Generate parallel kernel compilation initialization function.
            # JIT-only since AOTI has no Python dependency.
            if self._lazy_kernel_names:
                start_compile_body = (
                    "loadLazyCompileFuncs();\n"
                    "    _module_pending_kernels = PyDict_New();\n"
                    '    AOTI_TORCH_CHECK(_module_pending_kernels, "Failed to create pending kernels dict");\n'
                    "    "
                    + "\n    ".join(
                        f'startKernelCompile(_module_pending_kernels, "{name}", {name}_source);'
                        for name in self._lazy_kernel_names
                    )
                )
                self.include_extra_header("mutex")
                self.prefix.writeline_jit("")
                self.prefix.splice_jit(
                    f"""\
// Start parallel compilation of all Triton kernels.
static inline void start_all_triton_kernel_compiles() {{
    {start_compile_body}
}}

// inductor_entry_impl calls this on every forward;
// std::call_once makes the first call do the work.
static std::once_flag _triton_kernel_compile_init_flag;
static inline void ensure_triton_kernel_compiles_started() {{
    std::call_once(_triton_kernel_compile_init_flag, [] {{
        start_all_triton_kernel_compiles();
    }});
}}
"""
                )

        self.prefix = make_codegen_buffer()
        super().finalize_prefix()

        self.prefix.splice(triton_prefix)
        self.prefix.writeline("\n")
        self.prefix.splice(old_prefix)

    def generate_tma_descriptor(self, desc):
        # [FUNCTION DOC] Emit code for a TMA descriptor node, using the experimental or stable implementation by type.
        # Raises AssertionError for unknown types.
        self.write_tma_descriptor_helpers_once()

        if isinstance(desc, TMADescriptorExperimental):
            self._generate_experimental_tma_descriptor(desc)
        else:
            if not isinstance(desc, TMADescriptorStable):
                raise AssertionError(f"expected TMADescriptorStable, got {type(desc)}")
            self._generate_stable_tma_descriptor(desc)

    def _generate_experimental_tma_descriptor(self, desc):
        # [FUNCTION DOC] HELPER: Emit a CUtensorMap declaration and a call to the rank-specific init helper with data pointer,
        # dims, block dims and element size.
        # generate data pointer for the source tensor
        source = self.generate_args_decl(
            code=self,
            call_args=[self.val_to_arg_str(desc.tensor)],
            arg_types=[desc.tensor.get_dtype()],
            arg_signatures=[None],
            # these args are passed to initNDTMADescriptor, which is NOT a triton kernel
            is_triton_kernel=False,
        )

        desc_name = desc.name
        self.writeline(f"alignas(64) CUtensorMap {desc_name};")

        # `source` is in the form of `&var_x`, where `var_x` is the data pointer
        # (CUdeviceptr); we dereference `source` and cast to `void*` to pass to
        # the data pointer of the source tensor to the helper function
        # `init{1,2}DTMADescriptor`
        ptr = f"reinterpret_cast<void*>(*({source}))"
        dims = ", ".join(self.val_to_arg_str(dim) for dim in desc.dims)
        block_dims = ", ".join(self.val_to_arg_str(dim) for dim in desc.block_dims)
        element_size = self.val_to_arg_str(desc.element_size)
        fn = f"init{desc.rank}DTMADescriptor"
        args = f"&{desc_name}, {ptr}, {dims}, {block_dims}, {element_size}"
        self.writeline(f"{fn}({args});")

    def _generate_stable_tma_descriptor(self, desc):
        # [FUNCTION DOC] HELPER: Emit a StableTMADescriptor declaration, fill its block shape, global shape and strides, and call
        # initTMADescriptor on it (see AOTI TMA stable handling note).
        source = self.generate_args_decl(
            code=self,
            call_args=[self.val_to_arg_str(desc.tensor)],
            arg_types=[desc.tensor.get_dtype()],
            arg_signatures=[None],
            # these args are passed to initNDTMADescriptor, which is NOT a triton kernel
            is_triton_kernel=False,
        )

        desc_name = desc.name
        # Pack the relevant information into a StableTMADescriptor struct.
        # See [Note: AOTI TMA Stable handling] for more details.
        self.writeline(f"alignas(64) StableTMADescriptor {desc_name};")

        def fill_array(name, values):
            # [FUNCTION DOC] HELPER: Emit assignments of each value into the named C array.
            for i, val in enumerate(values):
                self.writeline(f"{name}[{i}] = {val};")

        ptr = f"reinterpret_cast<void*>(*({source}))"
        rank = len(desc.tensor.get_size())

        fill_array(f"{desc_name}.block_shape", desc.block_shape)
        fill_array(f"{desc_name}.global_shape", desc.tensor.get_size())
        fill_array(f"{desc_name}.strides", desc.tensor.get_stride())

        element_size = self.val_to_arg_str(desc.tensor.get_dtype().itemsize)
        fn = "initTMADescriptor"
        args = ", ".join(
            str(x)
            for x in [
                f"&{desc_name}.m",
                ptr,
                element_size,
                rank,
                f"{desc_name}.block_shape",
                f"{desc_name}.global_shape",
                f"{desc_name}.strides",
            ]
        )
        self.writeline(f"{fn}({args});")

    def generate_args_decl(
        self,
        code: IndentedBuffer | Self,
        call_args,
        arg_types,
        arg_signatures,
        is_triton_kernel=True,
        scratch_spaces: dict[str, int] | None = None,
    ):
        # [FUNCTION DOC] Emit declarations for kernel call arguments (tensor data pointers, extracted scalars, typed symbolic and
        # numeric values, unpacked stable TMA descriptors, and optional scratch pointers for Triton kernels) and
        # return the comma separated list of address-of arg names. Writes lines to code as a side effect.
        """
        Generates any declarations of args to pass into a kernel call, and then returns the arg names.

        In more detail:
        * declarations: e.g. this function has a side effect of generating lines like `auto var_0 = ...;`
        * returns: a string with the list of args, e.g. "var_0, var_1"

        call_args: list of call arguments
        arg_types: list of argument types
        arg_signatures: list with signatures of all the args
        is_triton_kernel: whether these are passed into a triton kernel or not. In particular,
                          calls to triton kernels will have an additional global scratch space
                          arg injected at the front of the arg list.
        """
        new_args: list[str] = []

        def process_tma_stable_arg(arg, arg_type, arg_signature, var_name):
            # [FUNCTION DOC] HELPER: Declare the TMA descriptor variable and return its unpacked kernel arg expressions.
            # [Note: AOTI TMA Stable handling]
            # For most args, a single arg passed to the python triton interface
            # maps to a single arg in the cubin interface. However, for host-side
            # TMA descriptors, a single python arg turns into 1 + 2 * N args in the
            # cubin interface (where N is the rank).
            #
            # To do this: at TMA codegen time (for aoti), we generate a struct
            # (StableTMADescriptor) containing the necessary information; and then
            # when we call the function (i.e. here), we unpack the struct members.
            code.writeline(f"auto {var_name} = {cexpr(arg)};")
            return _unpack_tma_descriptor_args(var_name, arg_signature)

        def process_args(arg, arg_type, arg_signature=None):
            # [FUNCTION DOC] HELPER: Emit the declaration for one argument by type/signature and append its kernel args.
            var_name = f"var_{next(self.arg_var_id)}"
            # ignore tma descriptors, as host-side TMA descriptors need
            # to be passed to the compiled Triton kernel by value
            if isinstance(arg_type, UnwrapUnspecArg) and not signature_is_tma_desc(
                arg_signature
            ):
                self.codegen_tensor_item(
                    arg_type.dtype,
                    arg,
                    var_name,
                    indented_buffer=code,
                )
                new_args.append(f"&{var_name}")
            elif isinstance(arg_type, torch_dtype) and not signature_is_tma_desc(
                arg_signature
            ):
                device_ptr_type = self.device_codegen.cpp_device_ptr()
                code.writeline(
                    maybe_hipify_code_wrapper(
                        f"{device_ptr_type} {var_name} = reinterpret_cast<{device_ptr_type}>({arg}.data_ptr());"
                    )
                )
                new_args.append(f"&{var_name}")
            # For symbolic call arguments, examine the arg signatures from triton meta
            # to explicitly cast to the right type
            # Reason: `auto` can infer unexpected type against kernel input signature.
            elif (
                isinstance(arg_type, type(SymbolicCallArg))
                and arg_signature is not None
                and arg_signature in TRITON_SIGNATURE_TO_CPP
            ):
                code.writeline(
                    f"{TRITON_SIGNATURE_TO_CPP[arg_signature]} {var_name} = {cexpr(arg)};"
                )
                new_args.append(f"&{var_name}")
            elif arg_type in (sympy.Integer, int):
                code.writeline(f"int {var_name} = {cexpr(arg)};")
                new_args.append(f"&{var_name}")
            elif arg_type in (sympy.Float, float):
                # Use signature type if available, otherwise default to float
                cpp_type = TRITON_SIGNATURE_TO_CPP.get(  # pyrefly: ignore[no-matching-overload]
                    arg_signature, "float"
                )
                code.writeline(f"{cpp_type} {var_name} = {cexpr(arg)};")
                new_args.append(f"&{var_name}")
            elif arg_signature and arg_signature.startswith("tensordesc<"):
                new_args.extend(
                    process_tma_stable_arg(arg, arg_type, arg_signature, var_name)
                )
            else:
                code.writeline(f"auto {var_name} = {cexpr(arg)};")
                new_args.append(f"&{var_name}")

        for arg, arg_type, arg_signature in zip_longest(
            call_args, arg_types, arg_signatures
        ):
            process_args(arg, arg_type, arg_signature)

        for scratch_name, workspace_size in (scratch_spaces or {}).items():
            if (
                is_triton_kernel
                and (
                    scratch := self.device_codegen.cpp_scratch(
                        next(self.arg_var_id),
                        workspace=TritonScratchWorkspace(
                            size=workspace_size,
                            generate_dtype_str=(
                                lambda: self.codegen_dtype(torch.uint8)
                            ),
                        ),
                        prefix=scratch_name,
                    )
                )
                is not None
            ):
                scratch_def, scratch_var = scratch
                code.writelines([maybe_hipify_code_wrapper(x) for x in scratch_def])
                new_args.append(f"&{scratch_var}")

        return ", ".join(new_args)

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
        # [FUNCTION DOC] Emit a kernel call. CPU kernels defer to the CPU wrapper. For Triton: optionally create the python autotune
        # block, choose the stream, register a DeferredTritonCallWrapper once per kernel, handle TMA tensors in lazy
        # mode, then write JIT and AOT call lines under the debug printer. For non-Triton kernels, cast args and call
        # the kernel directly. Raises AssertionError for unsupported TMA descriptor types.
        """
        Override the default value of argument 'gpu' to True here.
        generate_kernel_call can still be called with gpu=False because of
        a mix of cpu kernels and gpu kernels.
        """
        device = device or V.graph.get_current_device_or_throw()
        if device.type == "cpu":
            # Even in CppWrapperGpu, we may see cpp kernels
            return CppWrapperCpu._generate_kernel_call_helper(
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
            )

        if (
            triton
            and config.triton.autotune_at_compile_time
            and kernel_name not in self.kernel_autotune_names
        ):
            # Call PythonWrapperCodegen to create the autotune code block
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

        if (
            V.graph.aot_mode
            and current_stream_idx is not None
            and current_stream_idx != 0
        ):
            stream = get_stream_name(current_stream_idx)
        else:
            stream = self.write_get_raw_stream(device.index, graph_name)

        if triton:
            call_args, arg_types = self.prepare_triton_wrapper_args(
                call_args,
                # pyrefly: ignore [bad-argument-type]
                arg_types,
            )

            # For lazy compile mode with TMA, extract underlying tensor names
            tma_tensor_args: dict[str, str] = {}
            is_lazy_compile = config.triton.autotune_at_compile_time is False
            if is_lazy_compile and raw_args and triton_meta:
                signature = triton_meta.get("signature", {})
                raw_keys_list = raw_keys or []
                for key, raw_arg in zip(raw_keys_list, raw_args):
                    sig_type = signature.get(key, "")
                    if isinstance(sig_type, str) and signature_is_tma_desc(sig_type):
                        if isinstance(raw_arg, TMADescriptorStable):
                            # Get the underlying tensor name
                            tensor_name = raw_arg.get_tensor().get_name()
                            tma_tensor_args[key] = tensor_name
                        else:
                            raise AssertionError("Unsupported TMA descriptor type")

            wrapper_name = f"call_{kernel_name}"
            if wrapper_name not in self._triton_call_wrappers:
                self._triton_call_wrappers[wrapper_name] = DeferredTritonCallWrapper(
                    wrapper_name,
                    kernel_name,
                    self._kernel_name_to_body,
                    arg_types,
                    triton_meta=triton_meta,
                    inductor_meta=inductor_meta,
                    tma_tensor_args=tma_tensor_args,
                )

            # For TMA in lazy compile mode, add tensor args to the call
            if is_lazy_compile and tma_tensor_args:
                for tensor_name in tma_tensor_args.values():
                    call_args.append(tensor_name)
                    arg_types.append(
                        torch.float32
                    )  # dtype doesn't matter, just need tensor type

            # AOTI side uses this->device_idx_ so the model honors runtime device
            # assignment; JIT side has no this->device_idx_ in scope, so use the
            # concrete graph device index. Similarly, AOTI side always uses the
            # `stream` function parameter of run_impl, while JIT side uses the
            # locally-declared stream{idx} (see _codegen_entry_impl_prologue).
            aoti_device_idx = (
                "this->device_idx_" if V.graph.aot_mode else str(device.index)
            )
            jit_device_idx = str(device.index)
            jit_call_args = [*call_args, jit_device_idx, stream]
            aot_call_args = [*call_args, aoti_device_idx, "stream"]
            debug_printer_manager = V.graph.wrapper_code.debug_printer
            debug_printer_manager.set_printer_args(
                jit_call_args[: len(arg_types)], kernel_name, arg_types, None
            )
            # DebugPrinterManager is AOTI-only: route its writes through
            # writeline_aot so they're dropped on the JIT side (no-op for the
            # pure-JIT IndentedBuffer; AOT-only for DualIndentedBuffer). Without
            # this, the JIT buffer would receive before/after-launch prints
            # after the JIT launch, reporting post-launch state as pre-launch.
            with self.set_writeline(self.wrapper_call, self.wrapper_call.writeline_aot):
                with debug_printer_manager:
                    self.wrapper_call.writeline_jit(
                        f"{wrapper_name}({', '.join(jit_call_args)});"
                    )
                    self.wrapper_call.writeline_aot(
                        f"{wrapper_name}({', '.join(aot_call_args)}, "
                        f"kernels, this->cubin_dir_);"
                    )
        else:
            casted = []
            # pyrefly: ignore [bad-argument-type, no-matching-overload]
            for arg_type, arg in zip(arg_types, call_args):
                new_arg = arg
                if arg_type.endswith("*") and arg != "nullptr":
                    new_arg = f"{arg}.data_ptr()"
                # pyrefly: ignore [bad-argument-type]
                casted.append(f"({arg_type}){cexpr(new_arg)}")
            call_args_str = ", ".join(casted)
            # AOT: dispatch through AOTInductorModelKernels member.
            # JIT: call the extern "C" symbol directly (resolved at link time
            # via extra_flags pointing at the compiled .so).
            kernel_prefix = "kernels." if V.graph.aot_mode else ""
            self.writeline(f"{kernel_prefix}{kernel_name}({call_args_str}, {stream});")

    def prepare_triton_wrapper_args(
        self, call_args: list[Any], arg_types: list[Any]
    ) -> tuple[list[Any], list[Any]]:
        # [FUNCTION DOC] Normalize Triton call args into strings and matching types: unwrap unspecialized 0d tensors, lower bools,
        # format floats and symbolic values. Raises AssertionError if lengths differ.
        if len(call_args) != len(arg_types):
            raise AssertionError((call_args, arg_types))
        new_args = []
        new_args_types = []
        for arg, arg_type in zip(call_args, arg_types):
            if isinstance(arg, str):
                if isinstance(arg_type, torch_dtype) and should_unwrap_unspec_arg(arg):
                    # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
                    arg_type = UnwrapUnspecArg(dtype=arg_type)
                new_args.append(arg)
            elif isinstance(arg, bool):
                new_args.append(str(arg).lower())
            elif isinstance(arg, (int, float, SymbolicCallArg)):
                if isinstance(arg, float):
                    new_args.append(self.generate_float_value(arg))
                else:
                    new_args.append(str(arg))
            else:
                new_args.append(cexpr(V.graph.sizevars.simplify(arg)))
            new_args_types.append(arg_type)
        return new_args, new_args_types

    def make_zero_buffer(self, name):
        # [FUNCTION DOC] Return C++ code zeroing the named tensor via the aoti zero shim.
        return f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_zero_({name}.get()));"


@dataclasses.dataclass
class UnwrapUnspecArg:
    # [CLASS DOC] Marker dataclass wrapping a dtype: a 0-d CPU tensor argument that must be converted to a scalar (item) before use as a kernel arg.
    """Marker that we need to call .item() on the tensor"""

    dtype: torch_dtype
