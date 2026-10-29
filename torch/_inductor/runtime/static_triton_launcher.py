# [MODULE DOC] Statically launched Triton kernels: wrappers that parse a compiled Triton kernel's
# metadata (cubin/hsaco/zebin, argument type signature, scratch needs) so it can be loaded and
# launched through a precompiled C++ launcher without per-kernel C++ codegen. Provides the
# base class, CUDA/HIP and XPU subclasses, and a device-type factory function.
# No import-time side effects beyond imports.
import functools
import os
from functools import cached_property
from typing import Any
from typing_extensions import Unpack

from ..utils import is_rocm
from .triton_compat import ASTSource, CompiledKernel, knobs as triton_knobs
from .triton_helpers import get_constexprs


@functools.lru_cache(None)
def _tma_arg_helpers():
    # [FUNCTION DOC] HELPER: Cached; lazily imports Triton's nvidia tensor-descriptor argument builder and
    # TensorDescriptor class and returns (make_arg, TensorDescriptor), where make_arg expands
    # a descriptor into a tensor map plus shape and stride values. Raises ImportError if Triton
    # lacks these modules.
    """Cached (make_arg, TensorDescriptor) for host-side TMA arg expansion.
    make_arg(arg, metadata) -> [CUtensorMap, *shape, *strides]; the nvidia
    backend's make_tensordesc_arg ignores its third arg, so we pass None."""
    from triton.backends.nvidia.driver import make_tensordesc_arg
    from triton.tools.tensor_descriptor import TensorDescriptor

    def make_arg(arg, metadata):
        # [FUNCTION DOC] HELPER: Call Triton's tensor-descriptor argument builder with the third (unused) argument
        # set to nothing; returns the expanded argument list.
        return make_tensordesc_arg(arg, metadata, None)

    return make_arg, TensorDescriptor


class StaticallyLaunchedTritonKernel:
    # [CLASS DOC] Base class holding the launch-relevant data of a Triton CompiledKernel: name, cubin path or
    # raw bytes, num_warps, shared memory, a compact argument type string (arg_tys), scratch
    # flags, tensor-descriptor metadata, and the loaded function/module handles (single device)
    # or per-device dicts (device_agnostic mode). Subclasses supply the C++ launcher
    # (C_impl) and current-device lookup. Lifecycle: construct, load_kernel, run, close.
    # Picklable via __getstate__, which drops loaded handles and the cubin path.
    """
    Parses the metadata of a CompiledKernel from Triton into a structure that can
    launch the cuda kernel directly. Only works for triton kernels compiled to cubin.

    Doing this avoids C++ codegen and compilation during compile, since we can use a
    statically compiled library to launch the kernel. To avoid mallocing for the arguments,
    we have a launcher for different numbers of arguments up to a max. StaticCudaLauncher
    only supports # of arguments up until 10 for now.

    Workflow:
    Compile time:
    1. Compile a kernel with triton and get a CompiledKernel
    2. Instantiate kernel = StaticallyLaunchedTritonKernel(triton_kernel)
    3. Write to a cubin file: kernel.write_cubin_to_file(filepath)
    4. Call kernel.load_kernel() (CUDA should be initialized by this point) to load the cubin
    Runtime:
    5. Call kernel.run(grid, stream, args) to launch the kernel

    Note that after step 3, StaticallyLaunchedTritonKernel is fully pickleable/serializable.
    This allows it to be cached by FXGraphCache/TritonBundler, as well as sent from the worker
    to the parent process in inductor.

    There are two main versions of triton that we wish to support: 3.3 and 3.2. Triton makes considerable changes
    to how it handles constants in 3.3, so there's some special logic necessary to handle both versions.
    """

    @cached_property
    def C_impl(self):
        # [FUNCTION DOC] Cached property: the C++ launcher class for the backend. Abstract here; always
        # raises NotImplementedError.
        raise NotImplementedError

    def __init__(self, kernel: CompiledKernel) -> None:
        # [FUNCTION DOC] Extract launch data from a Triton CompiledKernel: function name, cubin path, argument
        # names, declared constexprs, hash, warps, shared memory, scratch-argument flags
        # (global and profile), tensor-descriptor metadata, and the argument type string.
        # Initializes unloaded state (no function/module, not device-agnostic).
        # Raises NotImplementedError if launch enter/exit hooks are set, if a scratch size is
        # positive, or if num_ctas is not 1.
        # pyrefly: ignore [missing-attribute]
        self.name = kernel.src.fn.__name__
        # pyrefly: ignore [missing-attribute]
        self.cubin_path = kernel._cubin_path

        # Used by torch.compile to filter constants in older triton versions
        # pyrefly: ignore [missing-attribute]
        self.arg_names = kernel.src.fn.arg_names

        # Const exprs that are declared by the triton kernel directly
        # Used to generate the kernel launcher's def args
        # pyrefly: ignore [missing-attribute]
        self.declared_constexprs = get_constexprs(kernel.src.fn)

        # pyrefly: ignore [missing-attribute]
        self.hash = kernel.hash

        if triton_knobs is None:
            # pyrefly: ignore [missing-attribute]
            launch_enter = kernel.__class__.launch_enter_hook
            # pyrefly: ignore [missing-attribute]
            launch_exit = kernel.__class__.launch_exit_hook
        else:
            launch_enter = triton_knobs.runtime.launch_enter_hook
            launch_exit = triton_knobs.runtime.launch_exit_hook

        def hook_is_empty(hook: Any) -> bool:
            # [FUNCTION DOC] HELPER: True if the hook is None or a Triton hook chain with no registered calls;
            # any other hook counts as non-empty.
            if hook is None:
                return True
            if (
                triton_knobs
                and (HookChain := getattr(triton_knobs, "HookChain", None)) is not None
                and isinstance(hook, HookChain)
            ):
                # Support hooks after https://github.com/triton-lang/triton/pull/7866
                return len(hook.calls) == 0
            return False

        if not hook_is_empty(launch_enter) or not hook_is_empty(launch_exit):
            raise NotImplementedError(
                "We don't support launch enter or launch exit hooks"
            )
        # pyrefly: ignore [missing-attribute]
        self.num_warps = kernel.metadata.num_warps
        self.shared = (
            # pyrefly: ignore [missing-attribute]
            kernel.shared if hasattr(kernel, "shared") else kernel.metadata.shared
        )

        def needs_scratch_arg(scratch_name: str, param_name: str) -> bool:
            # [FUNCTION DOC] HELPER: True if the kernel metadata has the named scratch-size field (so an extra
            # scratch argument must be passed). Raises NotImplementedError if its size is positive.
            # pyrefly: ignore [missing-attribute]
            if hasattr(kernel.metadata, param_name):
                # pyrefly: ignore [missing-attribute]
                if getattr(kernel.metadata, param_name) > 0:
                    raise NotImplementedError(
                        f"{scratch_name} scratch not yet supported"
                    )
                return True
            return False

        # Newer triton versions pass an extra global scratch parameter to the compiled cuda kernel.
        # Inductor never uses this field or enables it, but we still have to pass
        # an extra None into the set of params if its enabled
        self.has_global_scratch = needs_scratch_arg("Global", "global_scratch_size")
        # same situation for profile scratch - triton-lang/triton#7258
        self.has_profile_scratch = needs_scratch_arg("Profile", "profile_scratch_size")

        # pyrefly: ignore [missing-attribute]
        self.tensordesc_meta = getattr(kernel.metadata, "tensordesc_meta", None)
        self._has_tensordesc = False
        # pyrefly: ignore [missing-attribute]
        self.arg_tys = self.arg_ty_from_signature(kernel.src)
        self.function: int | None = None  # Loaded by load_kernel(on the parent process)
        self.module: int | None = None  # Owns the HIP/CUDA module loaded for function
        # compile-on-one-rank: a device-agnostic kernel (no baked device index) can be
        # launched on more than one device within a single process. A loaded module/
        # function is bound to a single device, so when device_agnostic is set we keep
        # them per device and resolve the current device at launch time. The cubin
        # (device-agnostic) is retained so it can be loaded onto additional devices.
        # These per-device dicts assume one thread per device for a given launcher (the
        # single-process multi-device path is sequential in practice); concurrent
        # first-launch on two new devices would need external locking.
        self.device_agnostic: bool = False
        self.functions: dict[int, int] = {}
        self.modules: dict[int, int] = {}
        num_ctas = 1
        if hasattr(kernel, "num_ctas"):
            num_ctas = kernel.num_ctas
        elif hasattr(kernel, "metadata"):
            num_ctas = kernel.metadata.num_ctas

        if num_ctas != 1:
            raise NotImplementedError(
                "Static cuda launcher only supports num_ctas == 1"
            )

    def reload_cubin_from_raw(self, filepath: str) -> str:
        # [FUNCTION DOC] Ensure a cubin file path exists: if cubin_path is unset, write the retained raw
        # bytes to filepath (creating directories) and record it. Returns the cubin path.
        # Raises AssertionError if both path and raw bytes are unset.
        # KNOWN ISSUE: if cubin_path is already set the file is not rewritten even if missing.
        """
        If the cubin file triton generated gets deleted under us, we can
        reload it from the raw cubin file.
        """
        if self.cubin_path is None:
            if self.cubin_raw is None:
                raise AssertionError("cubin_raw must be set when cubin_path is None")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(self.cubin_raw)
                self.cubin_path = filepath  # pyre-ignore
        return self.cubin_path

    def _agnostic_cubin_path(self) -> str:
        # [FUNCTION DOC] HELPER: For device-agnostic kernels, return a usable cubin path, rewriting the file
        # from the retained raw bytes if it was removed. Raises AssertionError if the path or
        # raw bytes are missing and the file does not exist.
        # The cubin bytes are device-agnostic, so the same file loads onto any device.
        # Keep it available (the single-device path frees it after one load) and rewrite
        # from the retained raw bytes if the file was removed under us.
        if self.cubin_path is not None and os.path.exists(self.cubin_path):
            return self.cubin_path
        if self.cubin_raw is None or self.cubin_path is None:
            raise AssertionError(
                "device-agnostic kernel cannot reload its cubin for a new device"
            )
        os.makedirs(os.path.dirname(self.cubin_path), exist_ok=True)
        with open(self.cubin_path, "wb") as f:
            f.write(self.cubin_raw)
        return self.cubin_path

    def load_kernel(self, device: int) -> None:
        # [FUNCTION DOC] Load the cubin onto the given device through the C++ launcher. In device-agnostic mode,
        # loads once per device and stores module and function in per-device dicts (cubin kept).
        # Otherwise loads once (no-op if already loaded), stores module/function, records
        # register and spill counts, then clears cubin path and raw bytes. Raises AssertionError
        # if the cubin path is missing in single-device mode.
        if self.device_agnostic:
            if device in self.functions:
                return
            (module, function, self.n_regs, self.n_spills) = self.C_impl._load_kernel(
                self._agnostic_cubin_path(), self.name, self.shared, device
            )
            self.modules[device] = module
            self.functions[device] = function
            return

        if self.function is not None:
            return

        if not hasattr(self, "cubin_path"):
            raise AssertionError("cubin_path attribute not set before load_kernel")
        if self.cubin_path is None:
            raise AssertionError("cubin_path must not be None before load_kernel")
        (self.module, self.function, self.n_regs, self.n_spills) = (
            self.C_impl._load_kernel(self.cubin_path, self.name, self.shared, device)
        )
        # Don't need the cubin path anymore now that we've loaded
        self.cubin_path = None
        self.cubin_raw = None

    def _current_device(self) -> int:
        # [FUNCTION DOC] HELPER: Index of the current device for the backend. Abstract; raises NotImplementedError.
        raise NotImplementedError

    def close(self) -> None:
        # [FUNCTION DOC] Release loaded state: clear the function/module handles and per-device dicts first,
        # then unload every module (single and per-device) via the C++ launcher.
        # Clear Python-visible handles first so repeated cleanup is harmless even if
        # the driver reports an error while unloading.
        modules = list(self.modules.values())
        if self.module is not None:
            modules.append(self.module)
        self.module = None
        self.function = None
        self.modules = {}
        self.functions = {}
        for mod in modules:
            self.C_impl._unload_kernel(mod)

    def __del__(self) -> None:
        # [FUNCTION DOC] HELPER: Destructor; calls close and swallows any exception.
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    @functools.lru_cache
    def type_mappings() -> dict[str, str]:
        # [FUNCTION DOC] Cached static method: mapping from Triton scalar type names (integers, unsigned
        # integers, floating types) to single-character launcher type codes (fp16 and bf16 map
        # to the 32-bit float code). TMA descriptor types are not included.
        return {
            "i1": "i",
            "i8": "b",
            "i16": "h",
            "i32": "i",
            "i64": "l",
            "u1": "I",
            "u8": "B",
            "u16": "H",
            "u32": "I",
            "u64": "K",
            "fp16": "f",
            "bf16": "f",
            "fp32": "f",
            "f32": "f",
            "fp64": "d",
            # TODO handle nvTmaDesc/CUtensormap
        }

    def extract_type(self, ty: str) -> str:
        # [FUNCTION DOC] Convert one Triton signature type string to its launcher type code: pointers (leading
        # star) become the object code, scalars use type_mappings. Raises NotImplementedError
        # for TMA descriptor types and KeyError for unknown scalar types.
        """
        Takes a triton type from CompiledKernel.signature and
        converts it into a single char encoding. _StaticCudaLauncher
        will switch on this char to figure out what type the underlying
        value should be passed to the triton kernel as.
        """
        if ty[0] == "*":
            return "O"
        elif ty == "nvTmaDesc" or ty.startswith("tensordesc<"):
            raise NotImplementedError("TMA descriptor kernels are not yet supported")
        return StaticallyLaunchedTritonKernel.type_mappings()[ty]

    def _expand_tensordesc_type(self, ty: str) -> str:
        # [FUNCTION DOC] HELPER: Expand a tensor-descriptor signature entry into the type codes of the
        # kernel parameters it lowers to. Parses the rank from the serialized type string, consumes the next
        # entry of the descriptor metadata (advancing the descriptor counter), and returns:
        # either a tensor-map code (when metadata is present) or a pointer plus 2x rank 64-bit values
        # plus two 32-bit flags, followed by rank 32-bit shape codes and rank 64-bit stride codes.
        # Raises NotImplementedError if the type string cannot be parsed.
        """Expand a tensordesc<dtype[shape]> signature entry into the per-arg
        type chars triton lowers it to (mirrors nvidia driver expand_signature):
        nvTmaDesc path -> CUtensorMap + shape(i32) + strides(i64); host-decomposed
        path -> base ptr + shape/strides(i64) + 2 flags + shape(i32) + strides(i64).

        This mirrors the string parse in triton's own expand_signature: the only
        thing we need is the descriptor's rank, which lives in the shape list of
        the serialized "tensordesc<dtype[d0, d1, ...]>" form. We parse that string
        here rather than importing expand_signature because that helper's argument
        signature differs across triton versions (a 2-arg (signature, meta) form
        in the fb-triton nvidia driver vs a 3-arg (signature, meta,
        descriptor_type) form as backends converge on a shared helper), so
        importing it would break across triton bumps; the serialized string
        format is stable.
        TODO: switch to triton's expand_signature once the versions converge on a
        single argument signature.
        """
        import re

        # Same regex triton's expand_signature uses. We only need the rank, so
        # the captured dtype group is unused (kept to match triton's pattern).
        match = re.match(r"tensordesc<([^\[>]*)\[([^\]]*)\]", ty)
        if match is None:
            raise NotImplementedError(f"Could not parse tensordesc type: {ty}")
        ndim = match.group(2).count(",") + 1
        meta = self.tensordesc_meta
        idx = self._tensordesc_idx
        self._tensordesc_idx += 1
        per_desc_meta = meta[idx] if meta else None
        if per_desc_meta is None:
            chars = ["O"] + ["l"] * (2 * ndim) + ["i", "i"]
        else:
            chars = ["M"]
        chars += ["i"] * ndim + ["l"] * ndim
        return "".join(chars)

    def arg_ty_from_signature(self, src: ASTSource) -> str:
        # [FUNCTION DOC] Build the launcher type string from an ASTSource's signature, in argument index
        # order, skipping constexpr arguments and any argument listed in the source's constants.
        # Tensor-descriptor arguments are expanded and set the has-tensordesc flag. Side effects:
        # sets full_constexprs (all constant indices) and resets the descriptor counter.
        # Raises NotImplementedError for unsupported types.
        def index_key(i: Any) -> int:
            # [FUNCTION DOC] HELPER: Normalize a signature/constants key to an argument index: names are looked
            # up in the function's argument names, tuples use their first element, ints pass through.
            if isinstance(i, str):
                # pyrefly: ignore [missing-attribute]
                return src.fn.arg_names.index(i)
            elif isinstance(i, tuple):
                # In triton 3.3, src.fn.constants has tuples as a key
                return i[0]
            else:
                return i

        # pyrefly: ignore [missing-attribute]
        signature = {index_key(key): value for key, value in src.signature.items()}
        # Triton uses these as the main way to filter out constants passed to their cubin
        constants = [index_key(key) for key in getattr(src, "constants", dict())]
        # This value is always a superset of kernel.fn.constexprs: kernel.fn.constexprs are
        # constants declared by the triton kernel directly, whereas this list can have
        # constants that are unused by the triton kernel that triton figured out during
        # compilation.
        self.full_constexprs = constants
        # Despite requiring them to be passed in, the triton CUDA launcher
        # completely ignores the constexprs passed into it when generating code.
        # So we can ignore them here too
        params = []
        self._tensordesc_idx = 0

        for i in sorted(signature.keys()):
            ty = signature[i]
            # In newer triton versions, constants are passed in to signature with type `constexpr`
            # In older triton versions, there can be constants in src.constants that are not `constexpr` in signature
            # so we check both here
            if ty == "constexpr" or i in constants:
                pass
            elif isinstance(ty, str) and ty.startswith("tensordesc<"):
                self._has_tensordesc = True
                params.append(self._expand_tensordesc_type(ty))
            else:
                # pyrefly: ignore [bad-argument-type]
                params.append(self.extract_type(ty))
        return "".join(params)

    def __getstate__(self) -> dict[str, Any]:
        # [FUNCTION DOC] Pickle support: copy of instance state with loaded function/module handles (single
        # and per-device) cleared and cubin_path set to none, since these are not valid across
        # processes.
        # Remove objects that are no longer valid for pickling
        state = self.__dict__.copy()
        state["function"] = None
        state["module"] = None
        state["functions"] = {}
        state["modules"] = {}
        # Cubin paths aren't consistent across processes, so we clear
        # and reload them.
        state["cubin_path"] = None
        return state

    def _expand_tma_args(self, args: tuple[object, ...]) -> tuple[object, ...]:
        # [FUNCTION DOC] HELPER: Replace each host-side TensorDescriptor argument with its expanded flat
        # parameters (tensor map, shape, strides) using the per-descriptor metadata in order;
        # other arguments pass through. Returns a new tuple.
        """Expand host-side TMA TensorDescriptor args into the flat kernel params
        (CUtensorMap + shape + strides) so they match the expanded type string."""
        make_tensordesc_arg, TensorDescriptor = _tma_arg_helpers()

        meta = self.tensordesc_meta
        out: list[object] = []
        td_idx = 0
        for arg in args:
            if isinstance(arg, TensorDescriptor):
                per_desc_meta = meta[td_idx] if meta else None
                td_idx += 1
                out.extend(make_tensordesc_arg(arg, per_desc_meta))
            else:
                out.append(arg)
        return tuple(out)

    def run(
        self,
        grid_x: int,
        grid_y: int,
        grid_z: int,
        stream: int,
        *args: Unpack[tuple[object, ...]],
    ) -> None:
        # [FUNCTION DOC] Launch the loaded kernel on the stream with the given 3D grid and positional args
        # (hot path). Picks the function for the current device (loading on demand) in
        # device-agnostic mode, expands tensor-descriptor args if present, and appends None
        # scratch arguments: always two on ROCm, otherwise one per enabled scratch flag.
        # Raises AssertionError if the kernel was not loaded or the argument count differs from the
        # type string length.
        """Actually run the kernel at runtime. This function is the hot codepath."""

        if self.device_agnostic:
            # The loaded function is device-bound; pick (loading on demand) the one for
            # the current device so a shared kernel launches on whatever device the
            # caller is using.
            device = self._current_device()
            function = self.functions.get(device)
            if function is None:
                self.load_kernel(device)
                function = self.functions[device]
        else:
            function = self.function

        # Assert load_kernel() has been called and args match
        if function is None:
            raise AssertionError("load_kernel() must be called before run()")

        # TODO: actually, if the args *don't* match, we probably should
        # throw an exception. But if inductor is the only one calling this
        # thing, it should always match.
        # Get rid of constants before passing to cubin launcher

        if self._has_tensordesc:
            args = self._expand_tma_args(args)

        arg_tys = self.arg_tys

        if is_rocm():
            # ROCm/HIP kernel ABI: The Triton HIP backend ALWAYS includes both
            # global_scratch and profile_scratch parameters in the kernel signature,
            # even when the kernel doesn't use them (i.e., when has_*_scratch is False).
            #
            # This differs fundamentally from CUDA, where these parameters are only
            # present in the signature if the corresponding has_*_scratch flag is True.
            #
            # The flags indicate whether memory will be allocated/used:
            # - has_global_scratch: Whether global scratch workspace is needed
            # - has_profile_scratch: Whether profiling instrumentation is enabled
            #
            # However, regardless of flag values, we MUST always pass both parameters
            # to match the HIP kernel ABI. Passing None is safe:
            #
            # - If scratch is not needed (has_*_scratch=False or scratch_size=0):
            #   The None becomes nullptr, which the kernel never dereferences
            #
            # - If scratch is needed (has_*_scratch=True and scratch_size>0):
            #   The None becomes nullptr initially, but the HIP runtime intercepts
            #   the kernel launch, allocates the required scratch memory based on
            #   kernel metadata, and replaces the nullptr with a valid pointer before
            #   the kernel actually executes
            #
            # Not passing both parameters causes segmentation faults because the kernel
            # expects them at specific positions in the argument array.
            arg_tys = arg_tys + "OO"
            args = (*args, None, None)

        else:
            for has_scratch in [self.has_global_scratch, self.has_profile_scratch]:
                if has_scratch:
                    arg_tys = arg_tys + "O"
                    args = (*args, None)
        # pyrefly: ignore [bad-argument-type]
        if len(args) != len(arg_tys):
            raise AssertionError(f"Expected {len(arg_tys)} args, got {len(args)}")

        # TODO: can handle grid functions here or in C++, so
        # that we don't need the grid handler above.
        self.C_impl._launch_kernel(
            function,
            grid_x,
            grid_y,
            grid_z,
            self.num_warps,
            self.shared,
            arg_tys,
            args,
            stream,
        )


class StaticallyLaunchedCudaKernel(StaticallyLaunchedTritonKernel):
    # [CLASS DOC] CUDA and ROCm (HIP) implementation using the static CUDA launcher from torch's C
    # extension; keeps the cubin or hsaco bytes in cubin_raw.
    @cached_property
    def C_impl(self):
        # [FUNCTION DOC] Cached property: the C++ static CUDA launcher class.
        from torch._C import _StaticCudaLauncher

        return _StaticCudaLauncher

    def __init__(self, kernel: CompiledKernel) -> None:
        # [FUNCTION DOC] Store the kernel's binary (hsaco for ROCm, else cubin) as cubin_raw, then run the base
        # initializer. Raises RuntimeError if neither binary is in the kernel's assembly outputs.
        # pyrefly: ignore [missing-attribute]
        if "hsaco" in kernel.asm:
            # pyrefly: ignore [missing-attribute]
            self.cubin_raw = kernel.asm["hsaco"]

        # pyrefly: ignore [missing-attribute]
        elif "cubin" in kernel.asm:
            # pyrefly: ignore [missing-attribute]
            self.cubin_raw = kernel.asm["cubin"]
        else:
            raise RuntimeError(
                "Expected either 'hsaco' (ROCm) or 'cubin' (CUDA) in kernel.asm"
            )
        super().__init__(kernel)

    def _current_device(self) -> int:
        # [FUNCTION DOC] HELPER: Index of the current CUDA device.
        import torch

        return torch.cuda.current_device()


class StaticallyLaunchedXpuKernel(StaticallyLaunchedTritonKernel):
    # [CLASS DOC] Intel XPU implementation using the static XPU launcher; the loaded kernel is a single
    # capsule object (no separate module handle) and the binary is the zebin.
    @cached_property
    def C_impl(self):
        # [FUNCTION DOC] Cached property: the C++ static XPU launcher class.
        from torch._C import _StaticXpuLauncher

        return _StaticXpuLauncher

    def __init__(self, kernel: CompiledKernel) -> None:
        # [FUNCTION DOC] Store the kernel's zebin (or none) as cubin_raw, then run the base initializer.
        # pyrefly: ignore [missing-attribute]
        self.cubin_raw = kernel.asm.get("zebin", None)
        super().__init__(kernel)

    def load_kernel(self, device: int) -> None:
        # [FUNCTION DOC] XPU variant of loading: the launcher returns only a function capsule (plus register and
        # spill counts). Device-agnostic mode loads once per device into the functions dict;
        # otherwise loads once, sets module to none, and clears cubin path and raw bytes.
        # Raises AssertionError if the cubin path is missing in single-device mode.
        # The XPU static launcher returns a PyCapsule for the loaded SYCL kernel,
        # not a separate module/function pair like the CUDA/HIP launcher.
        if self.device_agnostic:
            if device in self.functions:
                return
            (function, self.n_regs, self.n_spills) = self.C_impl._load_kernel(
                self._agnostic_cubin_path(), self.name, self.shared, device
            )
            # XPU has no separate module handle (only the function capsule).
            self.functions[device] = function
            return

        if self.function is not None:
            return

        if not hasattr(self, "cubin_path"):
            raise AssertionError("expected cubin_path attribute to be set")
        if self.cubin_path is None:
            raise AssertionError("expected cubin_path to not be None")
        (self.function, self.n_regs, self.n_spills) = self.C_impl._load_kernel(
            self.cubin_path, self.name, self.shared, device
        )
        self.module = None
        self.cubin_path = None
        self.cubin_raw = None

    def _current_device(self) -> int:
        # [FUNCTION DOC] HELPER: Index of the current XPU device.
        import torch

        return torch.xpu.current_device()

    def close(self) -> None:
        # [FUNCTION DOC] Drop all references (module, function, per-device dicts) so capsule destructors
        # release the kernel; no explicit unload call is made.
        self.module = None
        # Drop the PyCapsule references so their destructors can release sycl::kernel.
        self.function = None
        self.functions = {}
        self.modules = {}


def statically_launched_kernel_by_device(
    kernel: CompiledKernel, device_type: str = "cuda"
) -> StaticallyLaunchedTritonKernel:
    # [FUNCTION DOC] Factory: build the static launcher wrapper for a compiled kernel by device type
    # (cuda or hip gives the CUDA class, xpu gives the XPU class).
    # Raises NotImplementedError for other device types.
    if device_type in ("cuda", "hip"):
        return StaticallyLaunchedCudaKernel(kernel)
    elif device_type == "xpu":
        return StaticallyLaunchedXpuKernel(kernel)
    else:
        raise NotImplementedError(
            f"Device type {device_type} is not supported for static launcher"
        )
