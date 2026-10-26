# mypy: allow-untyped-defs
# [MODULE DOC] Builds the ROCm (hipcc/clang) shell command line used to compile generated CK kernels:
# include paths, library options, compiler flags, compiler location, and the final command string.
# Driven by torch._inductor.config.rocm settings. No import-time side effects beyond creating a logger.
import logging
import os

from torch._inductor import config
from torch._inductor.utils import is_linux, try_import_ck_lib


log = logging.getLogger(__name__)


def _rocm_include_paths(dst_file_ext: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns absolute include directories for compilation: Composable Kernel include and
    # library/include dirs, then the ROCm include dir (CK first so it takes priority). The CK root comes
    # from fbcode resources, config.rocm.ck_dir, a discovered CK library (which is then cached back into
    # config.rocm.ck_dir, with a warning if none found), or the ROCm home fallback. For dst_file_ext of exe the
    # CK library utility source dir is appended. Side effect: may mutate config.rocm.ck_dir; logs.
    from torch.utils import cpp_extension

    rocm_include = (
        os.path.join(config.rocm.rocm_home, "include")
        if config.rocm.rocm_home
        else cpp_extension._join_rocm_home("include")
    )

    if config.is_fbcode():
        from libfb.py import parutil

        ck_path = parutil.get_dir_path("composable-kernel-headers")
    else:
        if not config.rocm.ck_dir:
            ck_dir, _, _, _ = try_import_ck_lib()
            if not ck_dir:
                log.warning("Unspecified Composable Kernel directory")
            config.rocm.ck_dir = ck_dir
        ck_path = config.rocm.ck_dir or cpp_extension._join_rocm_home(
            "composable_kernel"
        )

    log.debug("Using ck path %s", ck_path)

    ck_include = os.path.join(ck_path, "include")
    ck_library_include = os.path.join(ck_path, "library", "include")

    # CK has to take priority over ROCm include paths
    # Since CK is potentially more up-to-date
    paths = [
        os.path.realpath(p) for p in (ck_include, ck_library_include, rocm_include)
    ]
    if dst_file_ext == "exe":
        ck_utility_include = os.path.join(ck_path, "library", "src", "utility")
        paths.append(os.path.realpath(ck_utility_include))
    return paths


def _rocm_lib_options(dst_file_ext: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns linker/runtime options: force-include of the clang HIP runtime wrapper header,
    # library search paths for ROCm lib and hip lib, and the amdhip64 library; for exe outputs also
    # links pthread and stdc++. Uses config.rocm.rocm_home or the cpp_extension ROCm home.
    from torch.utils import cpp_extension

    rocm_lib_dir = (
        os.path.join(config.rocm.rocm_home, "lib")
        if config.rocm.rocm_home
        else cpp_extension._join_rocm_home("lib")
    )
    hip_lib_dir = (
        os.path.join(config.rocm.rocm_home, "hip", "lib")
        if config.rocm.rocm_home
        else cpp_extension._join_rocm_home("hip", "lib")
    )

    opts = [
        "-include __clang_hip_runtime_wrapper.h",
        f"-L{os.path.realpath(rocm_lib_dir)}",
        f"-L{os.path.realpath(hip_lib_dir)}",
        "-lamdhip64",
    ]
    if dst_file_ext == "exe":
        opts += ["-lpthread", "-lstdc++"]
    return opts


def _rocm_compiler_options() -> list[str]:
    # [FUNCTION DOC] HELPER: Returns clang flags for HIP compilation: optimization level from config, HIP language,
    # C++20, one offload-arch flag per configured arch (default native), no gpu rdc, PIC, hidden
    # visibility and amdgpu LLVM tuning flags. Conditionally adds debug logging and -g, save-temps,
    # kernel resource usage reports, flush-denormals and fast-math based on config.rocm flags.
    arch_list = config.rocm.arch or ["native"]
    gpu_arch_flags = [f"--offload-arch={arch}" for arch in arch_list]
    opts = [
        config.rocm.compile_opt_level,
        "-x",
        "hip",
        "-std=c++20",
        *gpu_arch_flags,
        "-fno-gpu-rdc",
        "-fPIC",
        "-fvisibility=hidden",
        "-mllvm",
        "-amdgpu-early-inline-all=true",
        "-mllvm",
        "-amdgpu-function-calls=false",
        "-mllvm",
        "-enable-post-misched=0",
    ]
    if config.rocm.is_debug:
        opts += ["-DDEBUG_LOG=1", "-g"]
    if config.rocm.save_temps:
        opts += ["--save-temps=obj"]
    if config.rocm.print_kernel_resource_usage:
        opts += ["-Rpass-analysis=kernel-resource-usage"]
    if config.rocm.flush_denormals:
        opts += ["-fgpu-flush-denormals-to-zero"]
    if config.rocm.use_fast_math:
        opts += ["-ffast-math"]
    return opts


def rocm_compiler() -> str | None:
    # [FUNCTION DOC] Returns the path of the ROCm clang compiler on Linux (under config.rocm.rocm_home or the
    # ROCm home found by cpp_extension), falling back to the bare name clang if no ROCm home can be
    # determined (OSError). Returns None on non-Linux platforms.
    if is_linux():
        if config.rocm.rocm_home:
            return os.path.realpath(
                os.path.join(config.rocm.rocm_home, "llvm", "bin", "clang")
            )
        try:
            from torch.utils import cpp_extension

            return os.path.realpath(
                cpp_extension._join_rocm_home("llvm", "bin", "clang")
            )
        except OSError:
            # neither config.rocm.rocm_home nor env variable ROCM_HOME are set
            return "clang"
    return None


def rocm_compile_command(
    src_files: list[str],
    dst_file: str,
    dst_file_ext: str,
    extra_args: list[str] | None = None,
) -> str:
    # [FUNCTION DOC] Assembles the full compile command string. Args: src_files, dst_file, dst_file_ext (one of o, so,
    # exe), optional extra_args. Combines compiler options, extra args, include flags and lib options,
    # then appends -c, -shared, or the standalone-runner define depending on extension. Returns the
    # command text. Raises NotImplementedError for other extensions.
    include_paths = _rocm_include_paths(dst_file_ext)
    lib_options = _rocm_lib_options(dst_file_ext)
    compiler_options = _rocm_compiler_options()
    compiler = rocm_compiler()
    options = (
        compiler_options
        + (extra_args or [])
        + [f"-I{path}" for path in include_paths]
        + lib_options
    )
    src_file = " ".join(src_files)
    # supported extensions: .o, .so, .exe
    if dst_file_ext == "o":
        options.append("-c")
    elif dst_file_ext == "so":
        options.append("-shared")
    elif dst_file_ext == "exe":
        options.append("-DGENERATE_CK_STANDALONE_RUNNER")
    else:
        raise NotImplementedError(f"Unsupported output file suffix {dst_file_ext}!")
    return f"{compiler} {' '.join(options)} -o {dst_file} {src_file}"
