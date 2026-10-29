#!/usr/bin/env python3
# [MODULE DOC] Command-line script (arguments: input and output directories) that repairs
# PyTorch wheels built in a manylinux container. For each wheel it unpacks via auditwheel,
# bundles libgomp and aarch64 BLAS/ACL libraries, optionally bundles ROCm libraries and
# kernel/db files, rewrites needed-library entries with patchelf, sets RPATHs according to the
# backend (CUDA, XPU, ROCm wheel layout, ROCm OS layout, CPU), retags the platform to
# manylinux_2_28 and repacks. Behavior is driven by USE_CUDA, GPU_ARCH_TYPE, GPU_ARCH_VERSION,
# DESIRED_CUDA, PYTORCH_ROCM_ARCH and ROCM_HOME. Main entry: main. Import-time side effects:
# imports auditwheel (must be installed) and reads the ROCm version file under ROCM_HOME to
# decide whether the hipFile library is appended to the module-level ROCm library list.
"""Repair a PyTorch wheel: bundle libgomp + GPU libs, set RPATHs, retag platform.

Uses auditwheel for unpack/repack (not the `wheel` CLI). `wheel pack`/`wheel tags`
emit an invalid ZIP64 header for archives over 4GB (pypa/wheel#692), which makes
large ROCm wheels fail to install under strict zip parsers such as uv
(pytorch#189748). auditwheel repacks via the stdlib `zipfile`, which writes a
correct ZIP64 record.

Usage: repair_wheel.py <input_dir> <output_dir>

Environment variables:
    DESIRED_CUDA       - cpu, cu126, cu130, xpu, rocm6.4.1, etc.
    GPU_ARCH_TYPE      - cpu, cuda, cuda-aarch64, rocm, xpu
    GPU_ARCH_VERSION   - 12.6, 13.0, 13.2, 6.4.1, etc. (empty for CPU)
    USE_CUDA           - "0" or "1"
    PYTORCH_ROCM_ARCH  - ;-separated gfx targets (ROCm only)
    ROCM_HOME          - /opt/rocm (ROCm only)
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from auditwheel.wheeltools import add_platforms, InWheelCtx


PATCHELF = "/usr/local/bin/patchelf"


def wheel_platform_tags(wheel_name: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Return the list of platform tags from a wheel filename, taken
    # from the last dash-separated field with the extension removed and split on dots.
    """Platform tags encoded in a wheel filename.

    The filename is ``{name}-{version}-{python}-{abi}-{platform}.whl``; the
    platform field is the last ``-``-delimited component and may itself be a
    ``.``-joined set of tags (e.g. ``linux_x86_64`` or
    ``manylinux1_x86_64.manylinux_2_17_x86_64``).
    """
    return wheel_name.removesuffix(".whl").rsplit("-", 1)[-1].split(".")


@dataclass
class BundledLib:
    # [CLASS DOC] HELPER: Plain data record describing a shared library to copy into the
    # wheel's torch lib directory: source path, destination file name, and an optional
    # original soname whose needed entries should be rewritten to the destination name.
    """A shared library to copy into torch/lib/ and (optionally) patchelf-rewrite NEEDED entries for."""

    src: Path
    dest_name: str  # final filename in torch/lib/
    needed_alias: str | None = None  # original SONAME to replace in NEEDED entries


@dataclass
class AuxFile:
    # [CLASS DOC] HELPER: Plain data record for an auxiliary (non-library) file to copy into
    # the wheel's torch directory: source path and destination path relative to torch.
    """Auxiliary content (e.g. MIOpen db, RCCL algos, gfx kernel files) copied into the wheel."""

    src: Path
    rel_dest: str  # relative path under torch/, e.g. "lib/rocblas/library/Tensile..."


def detect_libgomp() -> Path:
    # [FUNCTION DOC] HELPER: Return the expected libgomp path for the host: the multiarch
    # Debian-style path on Ubuntu, otherwise the lib64 path. Reads the os-release file and
    # raises an OS error if it is missing. Does not check that the library exists.
    """libgomp lives in different places on Ubuntu vs RHEL-family images."""
    arch = platform.machine()
    os_release = Path("/etc/os-release").read_text()
    if "Ubuntu" in os_release:
        return Path(f"/usr/lib/{arch}-linux-gnu/libgomp.so.1")
    return Path("/usr/lib64/libgomp.so.1")


def cuda_rpaths(gpu_arch_version: str) -> str:
    # [FUNCTION DOC] Build the colon-separated RPATH string for CUDA wheels from the CUDA
    # version: shared entries for cudnn, nvshmem, nccl and cusparselt, plus a single cu13
    # directory for CUDA 13 or a list of per-component directories otherwise. Returns text.
    # KNOWN ISSUE: the cusparselt entry in the per-component list lacks the nvidia segment
    # unlike all its siblings, which looks like a typo or a legacy path.
    """Build the colon-separated RPATH list for CUDA wheels.

    CUDA 13.x bundles all libs under nvidia/cu13/lib; CUDA 12.x ships them
    in per-component packages.
    """
    base = (
        "$ORIGIN/../../nvidia/cudnn/lib"
        ":$ORIGIN/../../nvidia/nvshmem/lib"
        ":$ORIGIN/../../nvidia/nccl/lib"
        ":$ORIGIN/../../nvidia/cusparselt/lib"
    )
    cuda_major = gpu_arch_version.split(".", 1)[0] if gpu_arch_version else ""
    if cuda_major == "13":
        return base + ":$ORIGIN/../../nvidia/cu13/lib"
    return (
        base
        + ":$ORIGIN/../../nvidia/cublas/lib"
        + ":$ORIGIN/../../nvidia/cuda_cupti/lib"
        + ":$ORIGIN/../../nvidia/cuda_nvrtc/lib"
        + ":$ORIGIN/../../nvidia/cuda_runtime/lib"
        + ":$ORIGIN/../../nvidia/cufft/lib"
        + ":$ORIGIN/../../nvidia/curand/lib"
        + ":$ORIGIN/../../nvidia/cusolver/lib"
        + ":$ORIGIN/../../nvidia/cusparse/lib"
        + ":$ORIGIN/../../cusparselt/lib"
        + ":$ORIGIN/../../nvidia/nvtx/lib"
        + ":$ORIGIN/../../nvidia/cufile/lib"
    )


def rocm_rpaths() -> str:
    # [FUNCTION DOC] HELPER: Return the RPATH string pointing at the pip-installed ROCm SDK
    # directories that sit beside torch in site-packages (used when ROCm is not bundled).
    """RPATH list for the TheRock wheel-based ROCm layout.

    ROCm libs come from the `rocm` pip package, which unpacks under
    <site-packages>/_rocm_sdk_core (a sibling of torch/), so point at it
    $ORIGIN-relatively, mirroring cuda_rpaths(). No ROCm libs are bundled into
    the wheel in this layout.
    """
    return (
        "$ORIGIN/../../_rocm_sdk_core/lib"
        ":$ORIGIN/../../_rocm_sdk_core/lib/rocm_sysdeps/lib"
        ":$ORIGIN/../../_rocm_sdk_libraries/lib"
    )


def aarch64_extra_deps(use_cuda: bool) -> list[Path]:
    # [FUNCTION DOC] Return the existing files among the libraries to bundle on aarch64:
    # libgfortran, ARM Compute Library libraries (if the build directory exists), and either
    # NVPL libraries (CUDA) or OpenBLAS (CPU). Missing files are silently skipped.
    """Libraries to bundle into torch/lib/ on aarch64.

    CPU builds link against OpenBLAS + libgfortran; CUDA builds link against
    NVPL. Both pick up ARM Compute Library (ACL) for oneDNN acceleration.
    """
    deps: list[Path] = []
    candidates: list[Path] = [Path("/usr/lib64/libgfortran.so.5")]
    if Path("/acl/build").is_dir():
        candidates += [
            Path("/acl/build/libarm_compute.so"),
            Path("/acl/build/libarm_compute_graph.so"),
        ]
    if use_cuda:
        candidates += [
            Path(f"/usr/local/lib/{name}")
            for name in (
                "libnvpl_blas_lp64_gomp.so.0",
                "libnvpl_lapack_lp64_gomp.so.0",
                "libnvpl_blas_core.so.0",
                "libnvpl_lapack_core.so.0",
            )
        ]
    else:
        candidates.append(Path("/opt/OpenBLAS/lib/libopenblas.so.0"))
    deps = [p for p in candidates if p.is_file()]
    return deps


# ROCm shared libs to bundle. Discovered under $ROCM_HOME/{lib,lib64}.
ROCM_SO_FILES: list[str] = [
    "libMIOpen.so",
    "libamdhip64.so",
    "libhipblas.so",
    "libhipfft.so",
    "libhiprand.so",
    "libhipsolver.so",
    "libhipsparse.so",
    "libhsa-runtime64.so",
    "libamd_comgr.so",
    "libmagma.so",
    "librccl.so",
    "librocblas.so",
    "librocfft.so",
    "libamd_smi.so",
    "librocrand.so",
    "librocsolver.so",
    "librocsparse.so",
    "libroctracer64.so",
    "libroctx64.so",
    "libhipblaslt.so",
    "libhipsparselt.so",
    "libhiprtc.so",
    "librocprofiler-sdk.so",
    "librocprofiler-register.so",
    "libhsa-amd-aqlprofile64.so",
    "librocm-core.so",
    "librocroller.so",
]

# hipFile only ships with ROCm 7.14 and later, where it is required.
_version_file = Path(os.environ.get("ROCM_HOME", "/opt/rocm")) / ".info" / "version"
_rocm_version = _version_file.read_text().strip() if _version_file.is_file() else ""
if tuple(int(x) for x in _rocm_version.split(".")[:2] if x.isdigit()) >= (7, 14):
    ROCM_SO_FILES.append("libhipfile.so")


def rocm_os_deps() -> list[Path]:
    # [FUNCTION DOC] HELPER: Return the OS-level library paths (numa, elf, tinfo, dw, drm,
    # drm_amdgpu) to bundle with ROCm wheels, with locations depending on Ubuntu versus the
    # RHEL family. Reads the os-release file. Paths are not checked for existence.
    # KNOWN ISSUE: the Ubuntu paths are hardcoded to the x86_64 multiarch directory, so they
    # are wrong on other architectures.
    """OS-side runtime deps that must travel with ROCm wheels."""
    os_release = Path("/etc/os-release").read_text()
    if "Ubuntu" in os_release:
        prefix = "/usr/lib/x86_64-linux-gnu"
        libtinfo = "/lib/x86_64-linux-gnu/libtinfo.so.6"
        libdrm_dir = "/usr/lib/x86_64-linux-gnu"
    else:  # AlmaLinux / CentOS / RHEL
        prefix = "/usr/lib64"
        # CentOS Linux had libtinfo.so.5; AlmaLinux ships .6.
        libtinfo_5 = Path("/usr/lib64/libtinfo.so.5")
        libtinfo = (
            str(libtinfo_5) if libtinfo_5.exists() else "/usr/lib64/libtinfo.so.6"
        )
        libdrm_dir = "/opt/amdgpu/lib64"
    return [
        Path(f"{prefix}/libnuma.so.1"),
        Path(f"{prefix}/libelf.so.1"),
        Path(libtinfo),
        Path(f"{prefix}/libdw.so.1"),
        Path(f"{libdrm_dir}/libdrm.so.2"),
        Path(f"{libdrm_dir}/libdrm_amdgpu.so.1"),
    ]


def find_rocm_lib(rocm_home: Path, basename: str) -> Path | None:
    # [FUNCTION DOC] HELPER: Find a ROCm library file whose name starts with the base name,
    # searching lib, then lib64, then the whole ROCm tree recursively. Returns the first match
    # or None.
    """Locate a ROCm library, falling back from lib/ to lib64/ to a wider search."""
    for sub in ("lib", "lib64"):
        for hit in (rocm_home / sub).rglob(basename + "*"):
            if hit.is_file() and hit.name.startswith(basename):
                return hit
    for hit in rocm_home.rglob(basename + "*"):
        if hit.is_file() and hit.name.startswith(basename):
            return hit
    return None


def rocm_arch_filter(arch_list: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Split a semicolon-separated architecture string into a list,
    # dropping empty entries.
    return [a for a in arch_list.split(";") if a]


def rocm_lib_kernels(
    rocm_home: Path, lib_subdir: str, archs: list[str]
) -> list[AuxFile]:
    # [FUNCTION DOC] HELPER: List kernel files in the library subdirectory of a ROCm library
    # under lib, skipping files whose names mention a gfx target not in the requested list;
    # files without a gfx marker are always included. Returns auxiliary-file records with
    # destinations under lib; returns an empty list if the directory is missing.
    """Per-gfx kernel files under $ROCM_HOME/lib/<lib_subdir>/library/ plus the non-gfx common files."""
    src_dir = rocm_home / "lib" / lib_subdir / "library"
    if not src_dir.is_dir():
        return []
    files: list[AuxFile] = []
    for entry in sorted(src_dir.iterdir()):
        if not entry.is_file():
            continue
        # Pick gfx-specific files matching the arch set, plus common (non-gfx) files.
        name = entry.name
        if "gfx" in name and not any(a in name for a in archs):
            continue
        files.append(AuxFile(src=entry, rel_dest=f"lib/{lib_subdir}/library/{name}"))
    return files


def rocm_bundle(
    rocm_home: Path, gpu_arch_version: str
) -> tuple[list[BundledLib], list[AuxFile]]:
    # [FUNCTION DOC] Build the lists of libraries and auxiliary files to bundle for a ROCm
    # wheel. Libraries come from the fixed ROCm list (plus the smi library when the ROCm
    # version is 7.2 or older), each found under the ROCm home with its soname rewrite
    # requested; OS dependencies that exist are added without rewrite. Auxiliary files are
    # filtered kernel files, MIOpen db files for the architectures, RCCL algorithm files and
    # the amdgpu ids file. Exits the process if a required ROCm library cannot be found.
    """Build the ROCm bundle spec: shared libs and auxiliary kernel/db files.

    Versioned ROCm sonames (libfoo.so.6) get renamed to bare .so to match the
    NEEDED entries that hipcc emits, mirroring the original build_rocm.sh
    fname_without_so_number behaviour. NEEDED entries inside the wheel are
    rewritten to the renamed copies via patchelf in repair_wheel().
    """
    libs: list[BundledLib] = []
    so_files = list(ROCM_SO_FILES)
    # librocm_smi64.so is only needed for ROCm7.2 and earlier
    if gpu_arch_version and tuple(map(int, gpu_arch_version.split(".")[:2])) <= (7, 2):
        so_files.append("librocm_smi64.so")
    for stem in so_files:
        path = find_rocm_lib(rocm_home, stem)
        if path is None:
            sys.exit(f"Required ROCm library not found: {stem}")
        # Strip the SO version: libfoo.so.6.1 -> libfoo.so. The ROCm-built
        # binaries in the wheel link against the bare .so SONAME.
        libs.append(BundledLib(src=path, dest_name=stem, needed_alias=stem))
    for os_lib in rocm_os_deps():
        if os_lib.is_file():
            libs.append(BundledLib(src=os_lib, dest_name=os_lib.name))

    archs = rocm_arch_filter(os.environ.get("PYTORCH_ROCM_ARCH", ""))
    aux: list[AuxFile] = []
    for sub in ("rocblas", "hipblaslt", "hipsparselt"):
        aux += rocm_lib_kernels(rocm_home, sub, archs)
    miopen_db = rocm_home / "share/miopen/db"
    if miopen_db.is_dir():
        for entry in sorted(miopen_db.iterdir()):
            if entry.is_file() and any(a in entry.name for a in archs):
                aux.append(AuxFile(src=entry, rel_dest=f"share/miopen/db/{entry.name}"))
    rccl_dir = rocm_home / "share/rccl/msccl-algorithms"
    if rccl_dir.is_dir():
        for entry in sorted(rccl_dir.iterdir()):
            if entry.is_file():
                aux.append(
                    AuxFile(
                        src=entry,
                        rel_dest=f"share/rccl/msccl-algorithms/{entry.name}",
                    )
                )
    amdgpu_ids = Path("/opt/amdgpu/share/libdrm/amdgpu.ids")
    if amdgpu_ids.is_file():
        aux.append(AuxFile(src=amdgpu_ids, rel_dest="share/libdrm/amdgpu.ids"))
    return libs, aux


def patchelf(*args: str) -> None:
    # [FUNCTION DOC] HELPER: Run the patchelf tool with the given arguments; raises a
    # called-process error on failure.
    subprocess.run([PATCHELF, *args], check=True)


def set_rpath(sofile: Path, rpath: str, force_rpath: bool) -> None:
    # [FUNCTION DOC] HELPER: Set the RPATH of a shared object with patchelf, adding the force
    # option when requested (so it becomes RPATH rather than RUNPATH).
    cmd = ["--set-rpath", rpath]
    if force_rpath:
        cmd.append("--force-rpath")
    cmd.append(str(sofile))
    patchelf(*cmd)


def replace_needed(unpacked_torch: Path, original: str, replacement: str) -> None:
    # [FUNCTION DOC] HELPER: For every shared object under the directory, list its needed
    # entries with patchelf (skipping files patchelf cannot read) and replace each entry equal
    # to the original name or starting with it plus a dot by the replacement name.
    """Rewrite NEEDED entries that match `original*` to `replacement` across the wheel."""
    for sofile in unpacked_torch.rglob("*.so*"):
        if not sofile.is_file():
            continue
        try:
            needed = subprocess.check_output(
                [PATCHELF, "--print-needed", str(sofile)], text=True
            ).splitlines()
        except subprocess.CalledProcessError:
            continue
        for entry in needed:
            if entry == original or entry.startswith(original + "."):
                patchelf("--replace-needed", entry, replacement, str(sofile))


def check_no_dangling_bundled_needed(torch_lib: Path) -> None:
    # [FUNCTION DOC] HELPER: Verify that no shared object in the lib directory needs a
    # versioned name whose bare-stem file is present but whose exact file is absent. Exits the
    # process with a message listing the offending pairs; returns normally otherwise.
    """Fail the build if a lib in torch/lib NEEDs a versioned soname of a
    bundled lib without a file of that name in the wheel (a missed
    replace_needed rewrite). Such a reference only resolves against a system
    ROCm install, so the wheel silently stops being self-contained."""
    names = {f.name for f in torch_lib.iterdir()}
    dangling = []
    for sofile in sorted(torch_lib.glob("*.so*")):
        if not sofile.is_file():
            continue
        try:
            needed = subprocess.check_output(
                [PATCHELF, "--print-needed", str(sofile)], text=True
            ).splitlines()
        except subprocess.CalledProcessError:
            continue
        for entry in needed:
            stem = entry.split(".so", 1)[0] + ".so"
            if stem in names and entry not in names:
                dangling.append(f"{sofile.name} -> {entry}")
    if dangling:
        joined = "\n".join(sorted(set(dangling)))
        sys.exit(f"Dangling NEEDED entries after bundling (missed rewrite?):\n{joined}")


def repair_wheel(
    wheel: Path,
    output_dir: Path,
    platform_tag: str,
    libgomp_path: Path,
    aarch64_deps: list[Path],
    bundled_libs: list[BundledLib],
    aux_files: list[AuxFile],
    c_so_rpath: str,
    lib_so_rpath: str,
    force_rpath: bool,
) -> None:
    # [FUNCTION DOC] Repair one wheel in place of an output copy: open it with the auditwheel
    # context, copy libgomp and the aarch64 dependencies into torch lib, copy bundled libs
    # (adding a bare-name symlink for versioned names), rewrite needed entries and check for
    # dangling ones, copy auxiliary files, set RPATHs on top-level and lib shared objects,
    # then retag the platform (replacing the old tags). The context repacks on exit. Modifies
    # the wheel contents; runs patchelf.
    # KNOWN ISSUE: the libgomp step replaces the needed entry libgomp.so.1 by the identical
    # name, which has no effect.
    # InWheelCtx unpacks via auditwheel's zip2dir on enter and, once out_wheel is
    # set, regenerates RECORD and repacks via dir2zip on exit. dir2zip uses the
    # stdlib zipfile (correct ZIP64), unlike `wheel pack`/`wheel tags` which
    # corrupt >4GB ROCm wheels (pypa/wheel#692, pytorch#189748).
    with InWheelCtx(wheel, output_dir / wheel.name) as ctx:
        torch_dir = ctx.path / "torch"
        torch_lib = torch_dir / "lib"

        # Bundle libgomp and rewrite NEEDED entries to point at our copy
        shutil.copy(libgomp_path, torch_lib / "libgomp.so.1")
        for sofile in torch_dir.glob("*.so*"):
            if sofile.is_file():
                patchelf(
                    "--replace-needed",
                    "libgomp.so.1",
                    "libgomp.so.1",
                    str(sofile),
                )

        # Bundle aarch64 BLAS/LAPACK/ACL dependencies (no-op on x86)
        for dep in aarch64_deps:
            shutil.copy(dep, torch_lib / dep.name)

        # TODO: Remove when switching to ROCm wheels
        # Bundle GPU-specific shared libs (currently only ROCm uses this).
        # Copy follows symlinks so versioned sonames become real files we can
        # rename to their bare .so form to match what the wheel links against.
        for lib in bundled_libs:
            shutil.copy(lib.src, torch_lib / lib.dest_name)
            # Some bundled deps are dlopen'd by their *bare* soname at runtime,
            # not just via NEEDED. In particular rocSHMEM's NUMAWrapper global
            # ctor does dlopen("libnuma.so"). The original build_rocm.sh shipped
            # OS deps under bare names; this pipeline keeps them versioned
            # (e.g. libnuma.so.1), so add a bare-name symlink next to the
            # versioned file. Without it that dlopen fails and rocSHMEM calls
            # exit() at load, tripping a rocprofiler-sdk atexit deadlock that
            # hangs `import torch` on no-GPU/no-kfd hosts. See
            # pytorch/pytorch#189110.
            if ".so." in lib.dest_name:
                bare = lib.dest_name.split(".so.", 1)[0] + ".so"
                bare_path = torch_lib / bare
                if not bare_path.exists():
                    bare_path.symlink_to(lib.dest_name)
        # Rewrite NEEDED entries only after every bundled lib has been copied
        # in: bundled libs reference each other (e.g. libhiprtc.so needs
        # libamd_comgr.so.3), and replace_needed only visits files present in
        # the wheel at call time, so rewriting inside the copy loop misses
        # references from libs bundled after their dependency (#189194).
        for lib in bundled_libs:
            if lib.needed_alias:
                replace_needed(torch_dir, lib.needed_alias, lib.dest_name)
        if bundled_libs:
            check_no_dangling_bundled_needed(torch_lib)

        # Copy auxiliary content (gfx kernel files, MIOpen db, RCCL algos, ...)
        for aux in aux_files:
            dest = torch_dir / aux.rel_dest
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(aux.src, dest)

        # Set RPATH on top-level (_C.so etc.) and lib/ shared objects
        for sofile in torch_dir.glob("*.so*"):
            if sofile.is_file():
                set_rpath(sofile, c_so_rpath, force_rpath)
        for sofile in torch_lib.glob("*.so*"):
            if sofile.is_file():
                set_rpath(sofile, lib_so_rpath, force_rpath)

        # Retag linux_* -> manylinux_2_28_* in both the filename and the WHEEL
        # metadata. add_platforms updates ctx.out_wheel to the new name; RECORD
        # regeneration and repacking happen when the context exits.
        add_platforms(
            ctx, [platform_tag], remove_platforms=wheel_platform_tags(wheel.name)
        )


def main() -> None:
    # [FUNCTION DOC] Entry point. Parses directories, reads backend settings from the
    # environment, locates libgomp (exits if absent), chooses RPATHs, force-rpath and the
    # ROCm bundle by backend (CUDA, XPU, ROCm wheel layout versus OS layout, or CPU), adds
    # aarch64 dependencies on that architecture, then repairs every wheel found (exits if
    # none) with the manylinux_2_28 platform tag and prints the count of output wheels.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    arch = platform.machine()
    use_cuda = os.environ.get("USE_CUDA", "0") == "1"
    gpu_arch_type = os.environ.get("GPU_ARCH_TYPE", "")
    gpu_arch_version = os.environ.get("GPU_ARCH_VERSION", "")
    is_rocm = gpu_arch_type == "rocm" or "rocm" in os.environ.get("DESIRED_CUDA", "")

    libgomp_path = detect_libgomp()
    if not libgomp_path.exists():
        sys.exit(f"libgomp not found at {libgomp_path}")

    bundled_libs: list[BundledLib] = []
    aux_files: list[AuxFile] = []

    if use_cuda:
        rpaths = cuda_rpaths(gpu_arch_version)
        c_so_rpath = f"{rpaths}:$ORIGIN:$ORIGIN/lib"
        lib_so_rpath = f"{rpaths}:$ORIGIN"
        force_rpath = True
    elif gpu_arch_type == "xpu":
        # XPU runtime libs come from pypi packages; set RPATHs like CUDA.
        xpu_rpaths = "$ORIGIN/../../../.."
        c_so_rpath = f"{xpu_rpaths}:$ORIGIN:$ORIGIN/lib"
        lib_so_rpath = f"{xpu_rpaths}:$ORIGIN"
        force_rpath = True
    elif is_rocm:
        rocm_home = Path(os.environ.get("ROCM_HOME", "/opt/rocm"))
        if "_rocm_sdk" in str(rocm_home):
            # TheRock wheel layout (rocm7.14): ROCm ships as the `rocm` pip
            # package (_rocm_sdk_core, a sibling of torch/). Resolve libs via
            # RPATH instead of bundling them, mirroring the CUDA/XPU wheels.
            rpaths = rocm_rpaths()
            c_so_rpath = f"{rpaths}:$ORIGIN:$ORIGIN/lib"
            lib_so_rpath = f"{rpaths}:$ORIGIN"
            force_rpath = True
        else:
            # Legacy OS/tarball layout (/opt/rocm, e.g. rocm7.2): bundle the
            # ROCm libs into the wheel so it stays self-contained.
            bundled_libs, aux_files = rocm_bundle(rocm_home, gpu_arch_version)
            c_so_rpath = "$ORIGIN:$ORIGIN/lib"
            lib_so_rpath = "$ORIGIN"
            force_rpath = True
    else:
        c_so_rpath = "$ORIGIN:$ORIGIN/lib"
        lib_so_rpath = "$ORIGIN"
        force_rpath = False

    aarch64_deps = aarch64_extra_deps(use_cuda) if arch == "aarch64" else []

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wheels = sorted(args.input_dir.glob("*.whl"))
    if not wheels:
        sys.exit(f"No wheels found in {args.input_dir}")

    platform_tag = f"manylinux_2_28_{arch}"
    for whl in wheels:
        repair_wheel(
            whl,
            args.output_dir,
            platform_tag,
            libgomp_path,
            aarch64_deps,
            bundled_libs,
            aux_files,
            c_so_rpath,
            lib_so_rpath,
            force_rpath,
        )

    repaired = list(args.output_dir.glob("*.whl"))
    print(f"Repaired {len(repaired)} wheel(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
