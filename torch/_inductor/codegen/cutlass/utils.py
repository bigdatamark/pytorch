# mypy: allow-untyped-defs
# [MODULE DOC] Shared utilities for the Inductor CUTLASS backend: importing the CUTLASS
# python packages (via symlinks in the cache dir), architecture and toolkit detection for
# CUDA and XPU, generating the library of GEMM operations (cached), dtype mapping and
# matching between torch and CUTLASS, accumulator dtype and alignment inference, a
# context manager capturing compiled sources, and the standalone-runner compile command.
# Import-time side effects: registers an atexit hook that moves the CUTLASS compiled
# cache file into the cache dir, creates a module logger, defines dtype tables, and
# chooses the dtype-to-CUTLASS C++ type table based on whether torch was built with XPU.
import atexit
import functools
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing_extensions import TypeIs

import sympy

import torch
from torch._inductor.runtime.runtime_utils import dynamo_timed
from torch._inductor.utils import clear_on_fresh_cache
from torch.utils._ordered_set import OrderedSet

from ... import config
from ...ir import Layout
from ...runtime.runtime_utils import cache_dir
from ...virtualized import V
from ..cuda.cuda_env import get_cuda_arch, get_cuda_version
from ..xpu.xpu_env import get_xpu_arch, get_xpu_version


log = logging.getLogger(__name__)

CUTLASS_OPERATION_KIND: str = "gemm"
ACCUMULATOR_DTYPES: OrderedSet[torch.dtype] = OrderedSet([torch.float, torch.int32])
XW_DTYPES: OrderedSet[torch.dtype] = OrderedSet(
    [torch.half, torch.bfloat16, torch.float8_e4m3fn, torch.int8, torch.float8_e5m2]
)


@atexit.register
def move_cutlass_compiled_cache() -> None:
    # [FUNCTION DOC] Exit hook. If CUTLASS was ever imported through the cached import
    # check and the cutlass_cppgen module exposes an existing cache file, moves that file
    # into the inductor cache directory (logging success at debug level, or a warning
    # on OS errors). Silently does nothing otherwise. Side effects: filesystem move.
    """Move CUTLASS compiled cache file to the cache directory if it exists."""
    if try_import_cutlass.cache_info().currsize == 0:
        return

    try:
        import cutlass_cppgen  # type: ignore[import-not-found]
    except ImportError:
        return

    # Check if the CACHE_FILE attribute exists in cutlass_cppgen and if the file exists
    if not hasattr(cutlass_cppgen, "CACHE_FILE") or not os.path.exists(
        cutlass_cppgen.CACHE_FILE
    ):
        return

    try:
        filename = os.path.basename(cutlass_cppgen.CACHE_FILE)
        shutil.move(cutlass_cppgen.CACHE_FILE, os.path.join(cache_dir(), filename))
        log.debug("Moved CUTLASS compiled cache file to %s", cache_dir())
    except OSError:
        log.warning("Failed to move CUTLASS compiled cache file", exc_info=True)


def _rename_cutlass_import(content: str, cutlass_modules: list[str]) -> str:
    # [FUNCTION DOC] HELPER: Rewrites import statements in the given text so that each
    # listed module is imported from the cutlass_library package instead. Returns the
    # new text; no other effects.
    for cutlass_module in cutlass_modules:
        content = content.replace(
            f"from {cutlass_module} import ",
            f"from cutlass_library.{cutlass_module} import ",
        )
    return content


@functools.cache
def try_import_cutlass() -> bool:
    # [FUNCTION DOC] Memoized check that the CUTLASS python packages are importable;
    # returns True if so, False (with a debug or warning log) otherwise. In fbcode it
    # just imports cutlass_cppgen and cutlass_library. Otherwise it locates the CUTLASS
    # python directory from the xpu or cutlass config, and if it exists (and the staging
    # directory is not already on the search path) creates symlinks in a staging
    # directory under the cache dir for cutlass_library, cutlass_cppgen, pycute and
    # mock modules for cuda, scipy and pydot, appending the staging directory to the
    # module search path. Raises AssertionError if an existing path is not a symlink or
    # points elsewhere. Side effects: creates directories and symlinks, mutates the
    # module search path, imports CUTLASS modules.
    # KNOWN ISSUE: the docstring says three ways of passing CUTLASS are supported but
    # lists only two.
    """
    We want to support three ways of passing in CUTLASS:
    1. fbcode, handled by the internal build system.
    2. User specifies cutlass_dir. The default is ../third_party/cutlass/,
       which is the directory when developers build from source.
    """
    if config.is_fbcode():
        try:
            import cutlass_cppgen  # type: ignore[import-not-found]
            import cutlass_library  # type: ignore[import-not-found]
        except ImportError as e:
            log.warning(
                "Failed to import CUTLASS packages in fbcode: %s, ignoring the CUTLASS backend.",
                e,
            )
            return False

        return True

    # Copy CUTLASS python scripts to a temp dir and add the temp dir to Python search path.
    # This is a temporary hack to avoid CUTLASS module naming conflicts.
    # TODO(ipiszy): remove this hack when CUTLASS solves Python scripts packaging structure issues.

    # TODO(mlazos): epilogue visitor tree currently lives in python/cutlass,
    # but will be moved to python/cutlass_library in the future (later 2025)
    def path_join(path0, path1):
        # [FUNCTION DOC] HELPER: Returns the absolute normalized path of the two joined parts.
        return os.path.abspath(os.path.join(path0, path1))

    # contains both cutlass and cutlass_library
    # we need cutlass for eVT
    cutlass_dir = (
        config.xpu.cutlass_dir
        if torch.xpu._is_compiled()
        else config.cutlass.cutlass_dir
    )
    cutlass_python_path = path_join(cutlass_dir, "python")
    torch_root = os.path.abspath(os.path.dirname(torch.__file__))
    mock_src_path = os.path.join(
        torch_root,
        "_inductor",
        "codegen",
        "cutlass",
        "lib_extensions",
        "cutlass_mock_imports",
    )

    cutlass_library_src_path = path_join(cutlass_python_path, "cutlass_library")
    cutlass_cppgen_src_path = path_join(cutlass_python_path, "cutlass_cppgen")
    pycute_src_path = path_join(cutlass_python_path, "pycute")

    tmp_cutlass_full_path = os.path.abspath(os.path.join(cache_dir(), "torch_cutlass"))

    dst_link_library = path_join(tmp_cutlass_full_path, "cutlass_library")
    dst_link_cutlass_cppgen = path_join(tmp_cutlass_full_path, "cutlass_cppgen")
    dst_link_pycute = path_join(tmp_cutlass_full_path, "pycute")

    # mock modules to import cutlass
    mock_modules = ["cuda", "scipy", "pydot"]

    if os.path.isdir(cutlass_python_path):
        if tmp_cutlass_full_path not in sys.path:

            def link_and_append(dst_link, src_path, parent_dir):
                # [FUNCTION DOC] HELPER: Ensures dst_link is a symlink to src_path. If the
                # link path exists it must be a symlink resolving to the same target
                # (AssertionError otherwise); if absent, creates the parent directory
                # and the symlink. Then appends the parent directory to the module search
                # path if not already present.
                if os.path.lexists(dst_link):
                    if not os.path.islink(dst_link):
                        raise AssertionError(
                            f"{dst_link} is not a symlink. Try to remove {dst_link} manually and try again."
                        )
                    if os.path.realpath(os.readlink(dst_link)) != os.path.realpath(
                        src_path,
                    ):
                        raise AssertionError(
                            f"Symlink at {dst_link} does not point to {src_path}"
                        )
                else:
                    os.makedirs(parent_dir, exist_ok=True)
                    os.symlink(src_path, dst_link)

                if parent_dir not in sys.path:
                    sys.path.append(parent_dir)

            link_and_append(
                dst_link_library, cutlass_library_src_path, tmp_cutlass_full_path
            )
            link_and_append(
                dst_link_cutlass_cppgen, cutlass_cppgen_src_path, tmp_cutlass_full_path
            )
            link_and_append(dst_link_pycute, pycute_src_path, tmp_cutlass_full_path)

            for module in mock_modules:
                link_and_append(
                    path_join(tmp_cutlass_full_path, module),  # dst_link
                    path_join(mock_src_path, module),  # src_path
                    tmp_cutlass_full_path,  # parent
                )

        try:
            import cutlass_cppgen  # type: ignore[import-not-found]  # noqa: F401
            import cutlass_library.generator
            import cutlass_library.library
            import cutlass_library.manifest  # noqa: F401
            import pycute  # type: ignore[import-not-found]  # noqa: F401

            return True
        except ImportError as e:
            log.debug(
                "Failed to import CUTLASS packages: %s, ignoring the CUTLASS backend.",
                e,
            )
    else:
        log.debug(
            "Failed to import CUTLASS packages: CUTLASS repo does not exist: %s",
            cutlass_python_path,
        )
    return False


def _normalize_xpu_arch(arch: str) -> str:
    # [FUNCTION DOC] HELPER: Strips an Xe prefix if present and returns the rest;
    # otherwise returns the arch string when its integer value is in the range 12 to 50
    # inclusive. Raises NotImplementedError for values outside that range.
    # KNOWN ISSUE: the prefixed form is returned without range validation.
    if arch.startswith("Xe"):
        return arch[2:]
    if 12 <= int(arch) and int(arch) <= 50:
        return arch
    else:
        raise NotImplementedError(f"Unsupported xpu arch: {arch}")


def _normalize_cuda_arch(arch: str) -> str:
    # [FUNCTION DOC] HELPER: Maps a CUDA arch (string with digits, or number) to the
    # nearest supported floor bucket as a string: 107, 103, 100, 90, 80, 75 or 70.
    # Values above 107 are returned as is after a warning. Raises ValueError if a string
    # has no digits and NotImplementedError if below 70.
    arch_num = arch
    if isinstance(arch, str):
        digits = "".join(ch for ch in arch if ch.isdigit())
        if not digits:
            raise ValueError(f"Unrecognized cuda arch: {arch}")
        arch_num = int(digits)
    else:
        arch_num = int(arch)

    if arch_num > 107:
        log.warning("Detected CUDA architecture > 107: %s. Please file an issue.", arch)
        return str(arch_num)
    if arch_num >= 107:
        return "107"
    if arch_num >= 103:
        return "103"
    if arch_num >= 100:
        return "100"
    if arch_num >= 90:
        return "90"
    if arch_num >= 80:
        return "80"
    if arch_num >= 75:
        return "75"
    if arch_num >= 70:
        return "70"
    raise NotImplementedError(f"Unsupported cuda arch: {arch}")


@functools.lru_cache(8)
def cutlass_arch(device_type: str) -> str:
    # [FUNCTION DOC] Memoized. Detects the device architecture for the device type (xpu
    # or otherwise cuda) and returns it normalized for CUTLASS; may raise from the
    # normalizers for unsupported architectures.
    if device_type == "xpu":
        arch = get_xpu_arch()
        return _normalize_xpu_arch(arch)
    else:
        arch = get_cuda_arch()
        return _normalize_cuda_arch(arch)


@functools.lru_cache(1)
def toolkit_version(device_type: str) -> str:
    # [FUNCTION DOC] Memoized. Returns the XPU toolkit version for xpu, otherwise the
    # CUDA version.
    if device_type == "xpu":
        return get_xpu_version()
    else:
        return get_cuda_version()


def get_device_cutlass_config(device_type: str):
    # [FUNCTION DOC] Returns the xpu config module for xpu, otherwise the cutlass config
    # section of the inductor config.
    """Get device-specific CUTLASS config (xpu/cuda overrides general cutlass config)."""
    if device_type == "xpu":
        return config.xpu
    from ...config import cutlass as inductor_cutlass_config

    return inductor_cutlass_config


@dataclass
class CUTLASSArgs:
    # [CLASS DOC] Dataclass mimicking the command line argument object that the CUTLASS
    # library manifest expects. Fields: architectures, toolkit_version, instantiation_level,
    # operations (required except the last two), device_type (default cuda), plus fixed
    # defaults for build directories, kernel filters and arch-compilation flags. The
    # latter fixed attributes are plain class attributes, not dataclass fields.
    """
    CUTLASS args used to initialize a CUTLASS Manifest.
    """

    architectures: str | None = None
    toolkit_version: str | None = None
    instantiation_level: str | None = None
    operations: str | None = None

    build_dir = ""
    curr_build_dir = ""
    generator_target = ""
    kernels = "all"
    ignore_kernels = ""
    exclude_kernels = ""
    # TODO: these three look dead?
    kernel_filter_file: None = None
    selected_kernel_list: None = None
    interface_dir: None = None
    filter_by_cc = True
    disable_full_archs_compilation = False
    device_type: str = "cuda"

    def __post_init__(self):
        # [FUNCTION DOC] Validation: raises RuntimeError if architectures or toolkit
        # version is missing (None).
        if self.architectures is None or self.toolkit_version is None:
            raise RuntimeError(
                f"{self.architectures=} or {self.toolkit_version=} is None!"
            )


@clear_on_fresh_cache
@functools.cache
def _gen_ops_cached(arch: str, version: str, device_type: str) -> dict[Any, Any]:
    # [FUNCTION DOC] HELPER: Memoized (cleared on fresh cache) generation of the CUTLASS
    # operation table for an arch, version and device type. Raises AssertionError if
    # CUTLASS cannot be imported; returns an empty mapping (after logging an error) if
    # arch or version is missing. Builds a manifest from CUTLASSArgs (arch 103 and 107
    # use the SM100 generator with an a-suffixed manifest arch; 107 gets an extra 103a
    # feature), then runs the generator: Intel Xe for xpu, SM100 (if available) plus
    # SM90 for arch 100, or the SM generator for the arch otherwise. Raises
    # NotImplementedError when the generator is missing. Returns the manifest operations
    # mapping and logs generation time.
    # Note: Cache needs to be specific for cuda architecture and version

    # Import cutlass python scripts.
    if not try_import_cutlass():
        raise AssertionError("Failed to import CUTLASS library")
    import cutlass_library.generator as cutlass_generator
    import cutlass_library.manifest as cutlass_manifest

    if arch is None or version is None:
        log.error(
            "Cannot detect cuda arch %s or version %s. "
            "Will discard all cutlass ops. "
            "Please consider setting _inductor.cuda.arch and _inductor.cuda.version configs.",
            arch,
            version,
        )
        return {}

    # SM103 and SM107 reuse the SM100 generator, but the CUTLASS manifest must keep
    # their architecture-specific baseline rather than treating them as SM100.
    if arch in ("103", "107"):
        gen_arch = "100"
        manifest_arch = f"{arch}a"
    else:
        gen_arch = manifest_arch = arch

    instantiation_level: str = config.cutlass.cutlass_instantiation_level
    args = CUTLASSArgs(
        architectures=manifest_arch,
        toolkit_version=version,
        instantiation_level=instantiation_level,
        operations=CUTLASS_OPERATION_KIND,
        device_type=device_type,
    )
    manifest = cutlass_manifest.Manifest(args)
    if arch == "107":
        # CUTLASS uses 103a as the SM100-family feature marker for architectures
        # without INT8 UMMA. Keep the SM107 baseline while sharing that feature.
        manifest.compute_capabilities_feature_set.append("103a")

    start_time = time.time()
    if device_type == "xpu":
        if hasattr(cutlass_generator, "GenerateIntelXe"):
            cutlass_generator.GenerateIntelXe(
                manifest, args.toolkit_version, arch=int(arch)
            )
        else:
            raise NotImplementedError(
                "Arch " + arch + " is not supported by current cutlass lib."
            )

    elif arch == "100":
        if hasattr(cutlass_generator, "GenerateSM100"):
            cutlass_generator.GenerateSM100(manifest, args.toolkit_version)
        cutlass_generator.GenerateSM90(manifest, args.toolkit_version)
    else:
        try:
            func = getattr(cutlass_generator, "GenerateSM" + gen_arch)
            func(manifest, args.toolkit_version)
        except AttributeError as e:
            raise NotImplementedError(
                "Arch " + gen_arch + " is not supported by current cutlass lib."
            ) from e

    log.info(
        "CUTLASS library generated a dict of %d operation kinds in %.2f seconds",
        len(manifest.operations),
        time.time() - start_time,
    )
    return manifest.operations


def gen_ops(device_type: str) -> dict[Any, Any]:
    # [FUNCTION DOC] Returns all supported CUTLASS operations for the current device,
    # detecting arch and toolkit version, under a timing context, via the cached generator.
    """
    Generates all supported CUTLASS operations.
    """
    with dynamo_timed("cutlass_utils.gen_ops"):
        arch = cutlass_arch(device_type)
        version = toolkit_version(device_type)
        return _gen_ops_cached(arch, version, device_type)


from ..cpp_utils import DTYPE_TO_CPP


if torch.xpu._is_compiled():
    DTYPE_TO_CUTLASS_TYPE = {
        **DTYPE_TO_CPP,
        torch.float16: "uint16_t",
        torch.bfloat16: "uint16_t",
        torch.float8_e4m3fn: "uint8_t",
        torch.float8_e5m2: "uint8_t",
    }
else:
    DTYPE_TO_CUTLASS_TYPE = {
        **DTYPE_TO_CPP,
        torch.float16: "__half",
        torch.bfloat16: "__nv_bfloat16",
        torch.float8_e4m3fn: "__nv_fp8_e4m3",
        torch.float8_e5m2: "__nv_fp8_e5m2",
    }


@functools.lru_cache(32)
def torch_dtype_to_cutlass_type(
    torch_dtype: torch.dtype,
) -> "cutlass_library.library.DataType":  # type: ignore[name-defined] # noqa: F821
    # [FUNCTION DOC] Memoized. Maps a torch dtype (float32, float16, bfloat16, float8 e4m3
    # or e5m2) to the CUTLASS library data type enum. Raises AssertionError if CUTLASS
    # cannot be imported and NotImplementedError for other dtypes.
    # Import cutlass python scripts.
    if not try_import_cutlass():
        raise AssertionError("Failed to import CUTLASS library")
    import cutlass_library  # type: ignore[import]

    if torch_dtype == torch.float:
        return cutlass_library.library.DataType.f32
    elif torch_dtype == torch.half:
        return cutlass_library.library.DataType.f16
    elif torch_dtype == torch.bfloat16:
        return cutlass_library.library.DataType.bf16
    elif torch_dtype == torch.float8_e4m3fn:
        return cutlass_library.library.DataType.e4m3
    elif torch_dtype == torch.float8_e5m2:
        return cutlass_library.library.DataType.e5m2
    else:
        raise NotImplementedError(f"Unsupported data type: {torch_dtype=}")


@functools.lru_cache(32)
def dtype_match(
    torch_dtype: torch.dtype | None,
    cutlass_dtype: "cutlass_library.library.DataType",  # type: ignore[name-defined]  # noqa: F821
) -> bool:
    # [FUNCTION DOC] Memoized. True if the CUTLASS data type corresponds to the torch
    # dtype: float32 matches f32 or tf32; half, bfloat16, int8, uint8, int32 and the two
    # float8 types match their one counterpart; any other dtype (including None) gives
    # False. Raises AssertionError if CUTLASS cannot be imported.
    # Import cutlass python scripts.
    if not try_import_cutlass():
        raise AssertionError("Failed to import CUTLASS library")
    import cutlass_library

    if torch_dtype == torch.float:
        return (
            cutlass_dtype == cutlass_library.library.DataType.f32
            or cutlass_dtype == cutlass_library.library.DataType.tf32
        )
    elif torch_dtype == torch.half:
        return cutlass_dtype == cutlass_library.library.DataType.f16
    elif torch_dtype == torch.bfloat16:
        return cutlass_dtype == cutlass_library.library.DataType.bf16
    elif torch_dtype == torch.int8:
        return cutlass_dtype == cutlass_library.library.DataType.s8
    elif torch_dtype == torch.uint8:
        return cutlass_dtype == cutlass_library.library.DataType.u8
    elif torch_dtype == torch.int32:
        return cutlass_dtype == cutlass_library.library.DataType.s32
    elif torch_dtype == torch.float8_e4m3fn:
        return cutlass_dtype == cutlass_library.library.DataType.e4m3
    elif torch_dtype == torch.float8_e5m2:
        return cutlass_dtype == cutlass_library.library.DataType.e5m2
    else:
        return False


def get_accumulator_dtype(
    input_torch_dtypes: list[torch.dtype],
) -> torch.dtype | None:
    # [FUNCTION DOC] Infers the accumulator dtype for GEMM input dtypes. Raises
    # AssertionError if any dtype is not an allowed input dtype or the result is not an
    # allowed accumulator. Returns None unless exactly two dtypes are given. Mixed
    # float8 e5m2/e4m3 gives float32; equal dtypes use that dtype; differing dtypes
    # are only supported as a half/bfloat16 paired with a 8-bit integer (the wider one
    # is used). Floating types map to float32, int8 maps to int32, anything else raises
    # NotImplementedError.
    """
    Given a pair of input torch dtypes, returns the inferred accumulator torch dtype.
    """

    if not (OrderedSet(input_torch_dtypes) <= XW_DTYPES):
        raise AssertionError(f"{input_torch_dtypes=} is not supported")

    if len(input_torch_dtypes) != 2:
        return None

    if OrderedSet(input_torch_dtypes) == OrderedSet(
        [torch.float8_e5m2, torch.float8_e4m3fn]
    ):
        return torch.float

    torch_dtype = None
    if input_torch_dtypes[0] == input_torch_dtypes[1]:
        torch_dtype = input_torch_dtypes[0]
    else:
        size0 = torch.tensor([], dtype=input_torch_dtypes[0]).element_size()
        size1 = torch.tensor([], dtype=input_torch_dtypes[1]).element_size()
        if size0 > size1:
            dtype0, dtype1 = input_torch_dtypes
        else:
            dtype1, dtype0 = input_torch_dtypes
        if dtype0 in [torch.half, torch.bfloat16] and dtype1 in [
            torch.int8,
            torch.uint8,
        ]:
            torch_dtype = dtype0

    if torch_dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        accumulator_dtype = torch.float
    elif torch_dtype == torch.int8:
        accumulator_dtype = torch.int32
    else:
        raise NotImplementedError(f"Unsupported data types: {input_torch_dtypes=}")

    if accumulator_dtype not in ACCUMULATOR_DTYPES:
        raise AssertionError(f"{accumulator_dtype=} is not supported")
    return accumulator_dtype


@functools.lru_cache(32)
def get_alignments(torch_dtype: torch.dtype) -> list[int]:
    # [FUNCTION DOC] Memoized. Returns candidate alignments (in elements, largest first)
    # for the dtype: 16-bit floats 8,4,2,1; float32 and int32 4,2,1; 8-bit types
    # 16,8,4,2. Raises NotImplementedError for other dtypes.
    """
    Returns all possible valid CUTLASS alignments in terms of the number of elements for a given dtype.
    CUTLASS gemm / conv SM80 APIs support 16 bytes max alignment, and 2 bytes min alignment.
    """

    if torch_dtype in (torch.half, torch.bfloat16):
        return [8, 4, 2, 1]
    elif torch_dtype == torch.float:
        return [4, 2, 1]
    elif torch_dtype in (
        torch.uint8,
        torch.int8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        return [16, 8, 4, 2]
    elif torch_dtype == torch.int32:
        return [4, 2, 1]
    else:
        raise NotImplementedError(f"unsupported {torch_dtype=} for alignments")


def get_max_alignment(inductor_layout: Layout) -> int:
    # [FUNCTION DOC] Returns the largest candidate alignment (in elements) valid for the
    # layout: the contiguous (stride one) dimension size and the layout offset must be
    # multiples of it, and every other stride must be too. Returns 1 if no dimension has
    # stride one or no candidate fits. Symbolic values are decided through the graph size
    # variables evaluation, which may add guards.
    """
    Returns the max alignment (in terms of number of elements) for a given Inductor Layout.
    """

    dtype = inductor_layout.dtype
    size = inductor_layout.size
    offset = inductor_layout.offset

    def is_static_int(number: object) -> TypeIs[int | sympy.Integer]:
        # [FUNCTION DOC] HELPER: True if the value is a plain or symbolic-library integer.
        return isinstance(number, (int | sympy.Integer))

    def a_factor_of(x, alignment):
        # [FUNCTION DOC] HELPER: True if x is divisible by alignment. Static integers use
        # a direct remainder test; otherwise asks the graph size variables to evaluate
        # whether the symbolic remainder is zero.
        if is_static_int(x) and is_static_int(alignment):
            return x % alignment == 0
        rem = sympy.Mod(x, alignment)
        return V.graph.sizevars.evaluate_expr(sympy.Eq(rem, 0))

    try:
        contiguous_dim = inductor_layout.stride.index(1)
    except ValueError:
        # No dim with stride 1 found, return 1
        return 1
    alignments = get_alignments(dtype)
    for alignment in alignments:
        if not a_factor_of(size[contiguous_dim], alignment) or not a_factor_of(
            offset, alignment
        ):
            continue
        if all(
            (dim == contiguous_dim)
            or a_factor_of(inductor_layout.stride[dim], alignment)
            for dim in range(len(size))
        ):
            return alignment
    return 1


class CUTLASSCompileSourceCapturingContext:
    # [CLASS DOC] HELPER: Context manager for tests and benchmarks that records every
    # source string passed to the device code cache compile method (CUDA or XPU) while
    # still calling the original compile. State: sources (captured list), device_type
    # and the active mock patch. Entering installs the patch; exiting removes it.
    # Helper class for Benchmarking and Testing CUTLASS Kernels in isolation.
    # Can be used to capture the sourcecode passed to CUDACodeCache.compile

    def __init__(self, device_type: str):
        # [FUNCTION DOC] HELPER: Initializes an empty captured source list and stores the
        # device type; no patch is active yet.
        self.sources = []
        self._compile_patch = None
        self.device_type = device_type

    def __enter__(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Patches the compile method of the XPU or CUDA code cache
        # class with a wrapper that appends the source to the captured list and calls the
        # original. Returns self. Side effect: global monkey patch until exit.
        # KNOWN ISSUE: the wrapper accepts extra_args but does not forward it to the
        # original compile.
        import unittest.mock as mock

        import torch._inductor.codecache

        codecache_cls = (
            torch._inductor.codecache.XPUCodeCache
            if self.device_type == "xpu"
            else torch._inductor.codecache.CUDACodeCache
        )
        _compile_method_orig = codecache_cls.compile

        def my_compile(source_code, dst_file_ext, extra_args: list[str] | None = None):
            # [FUNCTION DOC] HELPER: Replacement compile: records the source, then calls
            # the original compile with the source and extension only.
            self.sources.append(source_code)
            return _compile_method_orig(source_code, dst_file_ext)

        # pyrefly: ignore [bad-assignment]
        self._compile_patch = mock.patch(
            f"torch._inductor.codecache.{codecache_cls.__name__}.compile", my_compile
        )
        self._compile_patch.__enter__(*args, **kwargs)  # type: ignore[union-attr]
        return self

    def __exit__(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Removes the compile patch installed on entry.
        self._compile_patch.__exit__(*args, **kwargs)  # type: ignore[union-attr]


def cutlass_standalone_runner_compile_command(
    device_type: str, srcpath: Path, exepath: Path
):
    # [FUNCTION DOC] Builds the compile command for a captured kernel source file into a
    # standalone executable, using the XPU or CUDA command builder by device type, with
    # a define enabling the standalone runner (plus a trace-level define for non-xpu).
    # Returns the command; nothing is executed.
    # returns command string to compile a (captured) CUDA GEMM Kernel source to a standalone executable that's ready to run
    # Passes the correct preprocessor define to nvcc to ensure the standalone runner is enabled.

    extra_args = ["-DGENERATE_STANDALONE_RUNNER=1"]
    if device_type != "xpu":
        extra_args.append("-DCUTLASS_DEBUG_TRACE_LEVEL=1")
    cutlass_compile_command = (
        torch._inductor.codegen.xpu.compile_utils.xpu_compile_command
        if device_type == "xpu"
        else torch._inductor.codegen.cuda.compile_utils.cuda_compile_command
    )
    compile_command = cutlass_compile_command(
        [str(srcpath)], str(exepath), "exe", extra_args=extra_args
    )
    return compile_command
