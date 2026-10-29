#!/usr/bin/env python3
# [MODULE DOC] Command line tool that builds a libtorch C++ distribution zip from a
# pre-built PyTorch wheel. It unpacks the wheel, copies libraries, headers, CMake files
# and (on Windows) DLLs into a libtorch directory, optionally bundles CUDA or ROCm
# runtime dependencies and rewrites RPATHs on Linux, writes version and hash metadata,
# splits debug symbols into a separate zip, and zips the result with a latest symlink.
# No import-time side effects; the work happens in main when run as a script.
"""Extract libtorch package from a PyTorch wheel.

Creates a libtorch zip from a pre-built wheel by copying the C++ libraries,
headers, and CMake files. On Linux, optionally splits debug symbols from
libtorch_cpu.so into a separate debug zip.

Usage:
    python extract_libtorch_from_wheel.py \
        --wheel-dir DIR --output-dir DIR --platform linux|macos|windows
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def find_wheel(wheel_dir: str) -> Path:
    # [FUNCTION DOC] Locate the single wheel file in a directory.
    # Args: wheel_dir, directory searched for files ending in .whl (non-recursive).
    # Returns the path of the one wheel. Raises FileNotFoundError when none exist and
    # RuntimeError when more than one exists.
    wheels = glob.glob(os.path.join(wheel_dir, "*.whl"))
    if not wheels:
        raise FileNotFoundError(f"No .whl files found in {wheel_dir}")
    if len(wheels) > 1:
        raise RuntimeError(f"Multiple .whl files found in {wheel_dir}: {wheels}")
    return Path(wheels[0])


def parse_version_from_wheel(wheel_path: Path) -> str:
    # [FUNCTION DOC] Extract the version field from a wheel filename.
    # Splits the file stem on dashes and returns the second field. Raises ValueError
    # if the stem has fewer than three dash separated fields.
    # Wheel filename format: {name}-{version}(-{build})?-{python}-{abi}-{platform}.whl
    name = wheel_path.stem
    parts = name.split("-")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse version from wheel filename: {wheel_path.name}")
    return parts[1]


def extract_wheel(wheel_path: Path, extract_dir: Path) -> Path:
    # [FUNCTION DOC] Unzip a wheel into extract_dir (side effect: writes all members).
    # Returns the path of the torch package directory inside it; raises
    # FileNotFoundError if the extracted tree has no torch directory.
    with zipfile.ZipFile(wheel_path, "r") as zf:
        zf.extractall(extract_dir)
    # Find the torch directory
    torch_dir = extract_dir / "torch"
    if not torch_dir.is_dir():
        raise FileNotFoundError(
            f"No 'torch' directory found in extracted wheel at {extract_dir}"
        )
    return torch_dir


def should_exclude_lib(filename: str) -> bool:
    # [FUNCTION DOC] Decide whether a file name must be left out of libtorch.
    # Excludes names starting with libtorch_python, names matching the _C cpython
    # extension pattern, and Python source or bytecode files. Pure function.
    # KNOWN ISSUE: the check for the name __init__.py is unreachable because every
    # such name already ends with the py suffix handled just before it.
    """Return True for files that should not go into the libtorch package."""
    if filename.startswith("libtorch_python"):
        return True
    if re.match(r"_C\.cpython.*", filename):
        return True
    if filename.endswith((".py", ".pyc")):
        return True
    if filename == "__init__.py":
        return True
    return False


def _is_lib_file(name: str, platform: str) -> bool:
    # [FUNCTION DOC] HELPER: Platform specific test for library file names.
    # Linux accepts names containing .so or ending in .a; macOS accepts dylib and .a;
    # Windows accepts dll, lib and pdb. Unknown platforms yield False.
    """Return True if the file looks like a library or header to include."""
    if platform == "linux":
        return ".so" in name or name.endswith(".a")
    elif platform == "macos":
        return name.endswith((".dylib", ".a"))
    elif platform == "windows":
        return name.endswith((".dll", ".lib", ".pdb"))
    return False


def copy_libraries(torch_dir: Path, libtorch_lib: Path, platform: str) -> None:
    # [FUNCTION DOC] Copy shared libraries from the wheel torch/lib into libtorch/lib.
    # Subdirectories are copied whole; files are copied (with metadata) only if not
    # excluded and recognised as libraries for the platform. On macOS also copies
    # non-excluded dylibs from the .dylibs directory. Raises FileNotFoundError when
    # torch/lib is missing. Side effect: writes into libtorch_lib.
    """Copy libraries from torch/lib/ to libtorch/lib/."""
    torch_lib = torch_dir / "lib"
    if not torch_lib.is_dir():
        raise FileNotFoundError(f"torch/lib/ not found at {torch_lib}")

    for item in torch_lib.iterdir():
        if item.is_dir():
            # Copy subdirectories (e.g. libshm/) as-is
            shutil.copytree(item, libtorch_lib / item.name, dirs_exist_ok=True)
            continue
        if should_exclude_lib(item.name):
            continue
        if _is_lib_file(item.name, platform):
            shutil.copy2(item, libtorch_lib / item.name)

    # On macOS, also copy delocated dylibs from torch/.dylibs/ if present
    if platform == "macos":
        dylibs_dir = torch_dir / ".dylibs"
        if dylibs_dir.is_dir():
            for item in dylibs_dir.iterdir():
                if item.suffix == ".dylib" and not should_exclude_lib(item.name):
                    shutil.copy2(item, libtorch_lib / item.name)


def fix_rpath(libtorch_lib: Path) -> None:
    # [FUNCTION DOC] Rewrite the RPATH of every shared object in libtorch_lib to the
    # origin-relative form using patchelf (forced RPATH, not RUNPATH), printing each
    # fixed file. Raises RuntimeError off Linux or when patchelf fails on a file, and
    # FileNotFoundError when patchelf is not on the path. Modifies files in place.
    # Wheel RPATHs are relative to pip's site-packages layout; rewrite to
    # $ORIGIN for the flat libtorch lib/ directory. Requires patchelf
    # (available in manylinux2_28-builder containers).
    if sys.platform != "linux":
        raise RuntimeError(f"fix_rpath is only supported on Linux, got {sys.platform}")

    patchelf = shutil.which("patchelf")
    if not patchelf:
        raise FileNotFoundError(
            "patchelf not found; the extraction job must run inside a "
            "manylinux2_28-builder container that ships patchelf"
        )

    for item in libtorch_lib.iterdir():
        if item.is_file() and (item.name.endswith(".so") or ".so." in item.name):
            result = subprocess.run(
                [patchelf, "--set-rpath", "$ORIGIN", "--force-rpath", str(item)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"  Fixed rpath: {item.name}")
            else:
                raise RuntimeError(
                    f"patchelf failed on {item.name}: {result.stderr.strip()}"
                )


# CUDA toolkit dirs in the manylinux builder holding the NVIDIA runtime libs the
# wheel needs. Only sonames found here are bundled, so system libs and the driver
# stub (in stubs/, never searched) are left to the runtime environment.
_CUDA_LIB_DIRS = ("/usr/local/cuda/lib64", "/usr/local/cuda/extras/CUPTI/lib64")


def _is_cuda_variant(desired_cuda: str) -> bool:
    # [FUNCTION DOC] HELPER: True when the variant string starts with cu.
    return desired_cuda.startswith("cu")


def _is_rocm_sdk_wheel_variant(desired_cuda: str) -> bool:
    # [FUNCTION DOC] HELPER: True when the variant starts with rocm and the current
    # interpreter has a pip package named rocm installed (checked by running pip show
    # in a subprocess and testing for a zero exit status).
    """True for TheRock pip-SDK ROCm; legacy /opt/rocm bundles deps in-wheel."""
    if not desired_cuda.startswith("rocm"):
        return False
    return (
        subprocess.run(
            [sys.executable, "-m", "pip", "show", "rocm"],
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )


def _needed(patchelf: str, so: Path) -> list[str]:
    # [FUNCTION DOC] HELPER: List the NEEDED sonames of a shared object by running
    # patchelf in print-needed mode. Returns an empty list if patchelf fails.
    r = subprocess.run(
        [patchelf, "--print-needed", str(so)], capture_output=True, text=True
    )
    return r.stdout.split() if r.returncode == 0 else []


def bundle_cuda_deps(libtorch_lib: Path) -> None:
    # [FUNCTION DOC] Make libtorch_lib self-contained for CUDA by copying missing
    # NVIDIA runtime libraries from the CUDA toolkit directories. Walks the NEEDED
    # sonames of every shared object transitively (work queue), copying each soname
    # found in the toolkit dirs and not already present. Unresolvable names are
    # skipped silently. Raises FileNotFoundError without patchelf or toolkit dirs.
    """Copy the NVIDIA runtime deps the small CUDA wheel omits into lib/ so the
    shared-with-deps tarball is self-contained (pytorch/pytorch#183971). Resolves
    NEEDED transitively from the torch libs; fix_rpath then adds $ORIGIN so the
    flat lib/ self-resolves. Must run in a CUDA manylinux builder."""
    patchelf = shutil.which("patchelf")
    search = [Path(d) for d in _CUDA_LIB_DIRS if Path(d).is_dir()]
    if not patchelf or not search:
        raise FileNotFoundError(
            "needs a CUDA manylinux builder (patchelf + /usr/local/cuda)"
        )
    present = {p.name for p in libtorch_lib.iterdir() if p.is_file()}
    queue = [p for p in libtorch_lib.iterdir() if p.is_file() and ".so" in p.name]
    while queue:
        for needed in _needed(patchelf, queue.pop()):
            if needed in present:
                continue
            src = next((d / needed for d in search if (d / needed).exists()), None)
            if not src:
                continue
            dest = libtorch_lib / needed
            shutil.copy2(src, dest)  # follows symlinks -> real file named by soname
            present.add(needed)
            queue.append(dest)
            print(f"  bundled CUDA dep: {needed}")


def _resolve_needed_lib(search: list[Path], needed: str) -> Path | None:
    # [FUNCTION DOC] HELPER: Find the file providing a NEEDED soname in the search
    # directories. First tries an exact file name match; otherwise, for names with a
    # shared object suffix, takes the first (sorted) file whose name starts with the
    # base name plus .so. Returns None when nothing matches.
    for lib_dir in search:
        direct = lib_dir / needed
        if direct.is_file():
            return direct
    if ".so" not in needed:
        return None
    base = needed.split(".so", 1)[0] + ".so"
    for lib_dir in search:
        for hit in sorted(lib_dir.glob(f"{base}*")):
            if hit.is_file():
                # copy2 follows devel symlinks and names the copy by NEEDED soname.
                return hit
    return None


def _rocm_sdk_search_dirs(site_packages: Path) -> list[Path]:
    # [FUNCTION DOC] HELPER: Return the existing library directories of the ROCm pip
    # SDK packages (core, its sysdeps, host-math and llvm subdirs, and libraries)
    # under the given site-packages root, in lookup priority order.
    core = site_packages / "_rocm_sdk_core" / "lib"
    libs = site_packages / "_rocm_sdk_libraries" / "lib"
    candidates = [
        core,
        core / "rocm_sysdeps" / "lib",
        core / "host-math" / "lib",
        core / "llvm" / "lib",
        libs,
    ]
    return [path for path in candidates if path.is_dir()]


def _installed_rocm_site_packages() -> Path:
    # [FUNCTION DOC] HELPER: Run pip show for the rocm package and return the path in
    # its Location line. Raises CalledProcessError if pip fails and FileNotFoundError
    # if no Location line is present.
    """Return the site-packages root for the SDK baked into the ROCm image."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "rocm"],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("Location:"):
            return Path(line.split(":", 1)[1].strip())
    raise FileNotFoundError("Cannot locate the ROCm SDK installed in the builder image")


def bundle_rocm_deps(libtorch_lib: Path) -> None:
    # [FUNCTION DOC] ROCm counterpart of bundle_cuda_deps: copies the transitive
    # closure of missing NEEDED libraries from the pip installed ROCm SDK package
    # directories into libtorch_lib, printing each one. Raises FileNotFoundError when
    # patchelf or the SDK library directories cannot be found. Unresolved names are
    # skipped silently.
    """Copy ROCm runtime deps from TheRock pip packages into lib/ for shared-with-deps.

    Wheel-based ROCm manywheels point RPATH at sibling _rocm_sdk_* packages under
    site-packages; libtorch extraction flattens to $ORIGIN, so copy the transitive
    NEEDED closure from the pip SDK layout first (pytorch/pytorch#192997).
    """
    patchelf = shutil.which("patchelf")
    if not patchelf:
        raise FileNotFoundError(
            "patchelf not found; the extraction job must run inside a "
            "manylinux2_28-builder container that ships patchelf"
        )

    site_packages = _installed_rocm_site_packages()
    search = _rocm_sdk_search_dirs(site_packages)
    if not search:
        raise FileNotFoundError(
            f"No _rocm_sdk_* lib dirs found under {site_packages}; the extraction "
            "job must use the matching ROCm builder image"
        )

    present = {path.name for path in libtorch_lib.iterdir() if path.is_file()}
    queue = [
        path for path in libtorch_lib.iterdir() if path.is_file() and ".so" in path.name
    ]
    while queue:
        for needed in _needed(patchelf, queue.pop()):
            if needed in present:
                continue
            src = _resolve_needed_lib(search, needed)
            if src is None:
                continue
            dest = libtorch_lib / needed
            shutil.copy2(src, dest)
            present.add(needed)
            queue.append(dest)
            print(f"  bundled ROCm dep: {needed}")


def copy_includes(torch_dir: Path, libtorch_include: Path) -> None:
    # [FUNCTION DOC] Copy the C++ headers into libtorch/include. Uses torch/include,
    # falling back to torch/lib/include for older wheels; merges into an existing
    # destination. Raises FileNotFoundError if neither location exists.
    torch_include = torch_dir / "include"
    if not torch_include.is_dir():
        # Some older wheels might have include under torch/lib/include
        torch_include = torch_dir / "lib" / "include"
    if not torch_include.is_dir():
        raise FileNotFoundError("include/ not found in torch directory")
    shutil.copytree(torch_include, libtorch_include, dirs_exist_ok=True)


def copy_cmake(torch_dir: Path, libtorch_share: Path) -> None:
    # [FUNCTION DOC] Copy torch/share/cmake into the cmake subdirectory of the share
    # destination. If the source is missing, prints a warning to stderr and returns
    # without error.
    torch_cmake = torch_dir / "share" / "cmake"
    if not torch_cmake.is_dir():
        print(f"Warning: share/cmake/ not found at {torch_cmake}", file=sys.stderr)
        return
    cmake_dest = libtorch_share / "cmake"
    shutil.copytree(torch_cmake, cmake_dest, dirs_exist_ok=True)


def copy_bin(torch_dir: Path, libtorch_bin: Path, platform: str) -> None:
    # [FUNCTION DOC] On Windows only, copy the non-excluded DLLs from torch/lib into
    # the bin directory; does nothing on other platforms or when torch/lib is absent.
    """Copy binary executables (mainly relevant for Windows)."""
    if platform == "windows":
        torch_lib = torch_dir / "lib"
        if torch_lib.is_dir():
            for item in torch_lib.iterdir():
                if item.suffix == ".dll" and not should_exclude_lib(item.name):
                    shutil.copy2(item, libtorch_bin / item.name)


def write_metadata(libtorch_dir: Path, version: str, git_hash: str) -> None:
    # [FUNCTION DOC] Write the files build-version and build-hash into libtorch_dir,
    # each containing the given string followed by a newline.
    (libtorch_dir / "build-version").write_text(version + "\n")
    (libtorch_dir / "build-hash").write_text(git_hash + "\n")


def get_git_hash(torch_dir: Path) -> str:
    # [FUNCTION DOC] Read the git_version assignment from torch/version.py, parsing
    # the right hand side as a literal. Returns the string "unknown" if the file is
    # missing, no such line exists, or the value cannot be parsed.
    """Read git_version from the wheel's torch/version.py."""
    version_file = torch_dir / "version.py"
    if not version_file.exists():
        return "unknown"
    from ast import literal_eval

    for line in version_file.read_text().splitlines():
        if line.strip().startswith("git_version"):
            try:
                return literal_eval(line.partition("=")[2].strip())
            except Exception:
                pass
    return "unknown"


def split_debug_symbols(
    libtorch_dir: Path, output_dir: Path, zip_prefix: str, version: str
) -> None:
    # [FUNCTION DOC] Move debug info of lib/libtorch_cpu.so into a separate zip.
    # Copies the library to a debug directory next to libtorch_dir, keeps only debug
    # sections in the copy, strips debug info from the original, adds a gnu debuglink
    # to it, derives the CRC from the link section via a bash subprocess (falls back
    # to "unknown"), and writes debug-<prefix>-<version>-<crc>.zip into output_dir.
    # If the library is missing, warns on stderr and returns. Raises on strip or
    # objcopy failure. Modifies the library in place.
    """Split debug symbols from libtorch_cpu.so (Linux only)."""
    libtorch_cpu = libtorch_dir / "lib" / "libtorch_cpu.so"
    if not libtorch_cpu.exists():
        print(
            "Warning: libtorch_cpu.so not found, skipping debug symbol split",
            file=sys.stderr,
        )
        return

    debug_dir = libtorch_dir.parent / "debug"
    debug_dir.mkdir(exist_ok=True)
    dbg_file = debug_dir / "libtorch_cpu.so.dbg"

    # Copy to create debug file
    shutil.copy2(libtorch_cpu, dbg_file)

    # Keep only debug symbols
    subprocess.run(
        ["strip", "--only-keep-debug", str(dbg_file)],
        check=True,
    )

    # Strip debug info from release lib
    subprocess.run(
        ["strip", "--strip-debug", str(libtorch_cpu)],
        check=True,
    )

    # Add debug link
    subprocess.run(
        ["objcopy", str(libtorch_cpu), f"--add-gnu-debuglink={dbg_file}"],
        check=True,
        cwd=str(libtorch_dir / "lib"),
    )

    # Extract CRC32 from the debug link section
    try:
        result = subprocess.run(
            [
                "bash",
                "-c",
                f"objcopy --dump-section .gnu_debuglink=>(tail -c4 | od -t x4 -An | xargs echo) {libtorch_cpu}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        crc32 = result.stdout.strip()
    except subprocess.CalledProcessError:
        crc32 = "unknown"

    # Create debug zip
    debug_zip = output_dir / f"debug-{zip_prefix}-{version}-{crc32}.zip"
    with zipfile.ZipFile(debug_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(dbg_file, "debug/libtorch_cpu.so.dbg")

    print(f"Debug symbols zip: {debug_zip}")


def create_libtorch_zip(
    libtorch_dir: Path,
    output_dir: Path,
    zip_prefix: str,
    version: str,
) -> Path:
    # [FUNCTION DOC] Zip the whole libtorch directory (entries relative to its parent,
    # so paths start with libtorch/) into <prefix>-<version>.zip in output_dir, then
    # create a <prefix>-latest.zip symlink pointing at it. Prints both names and
    # returns the zip path. Raises FileExistsError if the latest symlink exists.
    zip_path = output_dir / f"{zip_prefix}-{version}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(libtorch_dir):
            for f in files:
                filepath = Path(root) / f
                arcname = filepath.relative_to(libtorch_dir.parent)
                zf.write(filepath, arcname)
    # Create latest symlink
    latest_zip = output_dir / f"{zip_prefix}-latest.zip"
    latest_zip.symlink_to(zip_path.name)

    print(f"Libtorch zip: {zip_path}")
    print(f"Libtorch latest zip: {latest_zip}")
    return zip_path


def compute_zip_prefix(
    platform: str, desired_cuda: str, libtorch_variant: str, arch: str
) -> str:
    # [FUNCTION DOC] Build the zip name prefix from platform, variant and arch: macOS
    # is fixed, Windows adds win (and arm64 when arch is arm64), others use libtorch
    # plus the variant. The desired_cuda argument is accepted but unused.
    """Compute the zip filename prefix matching existing naming conventions.

    Linux:  libtorch-shared-with-deps
    macOS:  libtorch-macos-arm64
    Windows: libtorch-win-shared-with-deps (or libtorch-win-arm64-shared-with-deps)

    The arch component keeps the Windows x86_64 and arm64 packages from
    sharing a filename and overwriting each other in the upload bucket.
    """
    if platform == "macos":
        return "libtorch-macos-arm64"
    elif platform == "windows":
        if arch == "arm64":
            return f"libtorch-win-arm64-{libtorch_variant}"
        return f"libtorch-win-{libtorch_variant}"
    else:
        return f"libtorch-{libtorch_variant}"


def main() -> None:
    # [FUNCTION DOC] Entry point. Parses options (wheel dir, output dir, platform,
    # CUDA variant, libtorch variant, arch, optional git hash), extracts the wheel
    # into a temporary directory, assembles the libtorch tree (libs, optional CUDA or
    # ROCm dependency bundling and RPATH fix on Linux, includes, cmake, bin, metadata),
    # splits debug symbols on Linux, writes the zip, and always removes the temporary
    # directory. Creates the output directory; prints progress; errors propagate.
    parser = argparse.ArgumentParser(description="Extract libtorch from PyTorch wheel")
    parser.add_argument(
        "--wheel-dir", required=True, help="Directory containing the .whl file"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for output zip files"
    )
    parser.add_argument(
        "--platform",
        required=True,
        choices=["linux", "macos", "windows"],
        help="Target platform",
    )
    parser.add_argument(
        "--desired-cuda",
        default="cpu",
        help="CUDA variant (cpu, cu126, cu128, rocm7.1, etc.)",
    )
    parser.add_argument(
        "--libtorch-variant",
        default="shared-with-deps",
        help="Libtorch variant (shared-with-deps, etc.)",
    )
    parser.add_argument(
        "--arch",
        default="x86_64",
        choices=["x86_64", "arm64"],
        help="Target architecture (used to disambiguate Windows package names)",
    )
    parser.add_argument(
        "--git-hash",
        default="",
        help="Git hash to use for build-hash (auto-detected if not set)",
    )
    args = parser.parse_args()

    wheel_dir = Path(args.wheel_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find and extract wheel
    wheel_path = find_wheel(str(wheel_dir))
    version = parse_version_from_wheel(wheel_path)
    print(f"Found wheel: {wheel_path}")
    print(f"Version: {version}")

    extract_dir = wheel_dir / "_extract_tmp"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir()

    try:
        torch_dir = extract_wheel(wheel_path, extract_dir)

        # Create libtorch directory structure
        libtorch_dir = extract_dir / "libtorch"
        libtorch_dir.mkdir()
        for subdir in ["lib", "bin", "include", "share"]:
            (libtorch_dir / subdir).mkdir()

        # Copy components
        copy_libraries(torch_dir, libtorch_dir / "lib", args.platform)
        if args.platform == "linux":
            # shared-with-deps must be self-contained: bundle the NVIDIA runtime
            # libs the small wheel omits, then rewrite all RPATHs to $ORIGIN so
            # the flat lib/ resolves both torch and bundled deps.
            with_deps = "with-deps" in args.libtorch_variant
            if with_deps and _is_cuda_variant(args.desired_cuda):
                bundle_cuda_deps(libtorch_dir / "lib")
            elif with_deps and _is_rocm_sdk_wheel_variant(args.desired_cuda):
                bundle_rocm_deps(libtorch_dir / "lib")
            # XPU also uses sibling pip packages, but has no libtorch extract job.
            fix_rpath(libtorch_dir / "lib")
        copy_includes(torch_dir, libtorch_dir / "include")
        copy_cmake(torch_dir, libtorch_dir / "share")
        copy_bin(torch_dir, libtorch_dir / "bin", args.platform)

        # Write metadata
        git_hash = args.git_hash or get_git_hash(torch_dir)
        write_metadata(libtorch_dir, version, git_hash)

        # Compute zip prefix
        zip_prefix = compute_zip_prefix(
            args.platform, args.desired_cuda, args.libtorch_variant, args.arch
        )

        # Split debug symbols on Linux
        if args.platform == "linux":
            split_debug_symbols(libtorch_dir, output_dir, zip_prefix, version)

        # Create the zip
        create_libtorch_zip(libtorch_dir, output_dir, zip_prefix, version)

    finally:
        shutil.rmtree(extract_dir)


if __name__ == "__main__":
    main()
