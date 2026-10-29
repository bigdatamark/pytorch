# [MODULE DOC] Build script for the torch_openreg out-of-tree accelerator backend example.
# Unless invoked with clean or dist_info, it first builds the C++ dependencies with cmake
# (configure, then build and install into the package folder, using PyTorch's install dir), then
# calls setup to build the small C stub extension torch_openreg._C linked against the
# torch_bindings library, package the shared libraries, and register the torch.backends autoload
# entry point. Defines the platform flags and BASE_DIR at import time.
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from distutils.command.clean import clean

from setuptools import Extension, find_packages, setup


# Env Variables
IS_DARWIN = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
RUN_BUILD_DEPS = any(arg in {"clean", "dist_info"} for arg in sys.argv)


def make_relative_rpath_args(path):
    # [FUNCTION DOC] HELPER: Returns the linker arguments that add an rpath relative to the
    # loaded library's location for the given sub path (loader path on macOS, origin elsewhere),
    # or an empty list on Windows.
    if IS_DARWIN:
        return ["-Wl,-rpath,@loader_path/" + path]
    elif IS_WINDOWS:
        return []
    else:
        return ["-Wl,-rpath,$ORIGIN/" + path]


def get_pytorch_dir():
    # [FUNCTION DOC] HELPER: Returns the directory of the installed torch package. Side effect:
    # sets the environment variable that disables device backend autoload before importing
    # torch, to avoid a circular import if the accelerator was already built and installed.
    # Disable autoload of the accelerator

    # We must do this for two reasons:
    # We only need to get the PyTorch installation directory, so whether the accelerator is loaded or not is irrelevant
    # If the accelerator has been previously built and not uninstalled, importing torch will cause a circular import error
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch

    return os.path.dirname(os.path.realpath(torch.__file__))


def build_deps():
    # [FUNCTION DOC] HELPER: Runs cmake to configure the project in a build subfolder (created
    # if needed) with the install prefix at the torch_openreg package folder, the Python include
    # dir and the PyTorch dir, then builds the install target in Release with parallel jobs
    # equal to the CPU count. Raises CalledProcessError if a cmake step fails. Side effects:
    # creates files, runs subprocesses.
    build_dir = os.path.join(BASE_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    cmake_args = [
        "-DCMAKE_INSTALL_PREFIX="
        + os.path.realpath(os.path.join(BASE_DIR, "torch_openreg")),
        "-DPYTHON_INCLUDE_DIR=" + sysconfig.get_paths().get("include"),
        "-DPYTORCH_INSTALL_DIR=" + get_pytorch_dir(),
    ]

    subprocess.check_call(
        ["cmake", BASE_DIR] + cmake_args, cwd=build_dir, env=os.environ
    )

    build_args = [
        "--build",
        ".",
        "--target",
        "install",
        "--config",  # For multi-config generators
        "Release",
        "--",
    ]

    if IS_WINDOWS:
        build_args += ["/m:" + str(multiprocessing.cpu_count())]
    else:
        build_args += ["-j", str(multiprocessing.cpu_count())]

    command = ["cmake"] + build_args
    subprocess.check_call(command, cwd=build_dir, env=os.environ)


class BuildClean(clean):
    # [CLASS DOC] Custom distutils clean command for the openreg build; it replaces the default
    # clean behavior rather than extending it.
    def run(self):
        # [FUNCTION DOC] Removes the build, install and torch_openreg/lib directories if
        # present, then deletes every shared object file found under the torch_openreg package
        # folder. Does not call the base class clean. Side effect: file deletion.
        # KNOWN ISSUE: the base class clean.run is never invoked, so default distutils cleanup
        # is skipped.
        for i in ["build", "install", "torch_openreg/lib"]:
            dirs = os.path.join(BASE_DIR, i)
            if os.path.exists(dirs) and os.path.isdir(dirs):
                shutil.rmtree(dirs)

        for dirpath, _, filenames in os.walk(os.path.join(BASE_DIR, "torch_openreg")):
            for filename in filenames:
                if filename.endswith(".so"):
                    os.remove(os.path.join(dirpath, filename))


def main():
    # [FUNCTION DOC] Entry point: runs build_deps unless clean or dist_info was requested (the
    # constant RUN_BUILD_DEPS is true in exactly that case, despite its name), picks platform
    # specific link and compile flags (MSVC runtime flags on Windows, warning and aliasing flags
    # elsewhere), and calls setup with the C stub extension, shared library package data, the
    # clean command and the torch.backends entry point.
    # KNOWN ISSUE: RUN_BUILD_DEPS is true when the build must be skipped, so its name is
    # inverted.
    if not RUN_BUILD_DEPS:
        build_deps()

    if IS_WINDOWS:
        # /NODEFAULTLIB makes sure we only link to DLL runtime
        # and matches the flags set for protobuf and ONNX
        extra_link_args: list[str] = ["/NODEFAULTLIB:LIBCMT.LIB"] + [
            *make_relative_rpath_args("lib")
        ]
        # /MD links against DLL runtime
        # and matches the flags set for protobuf and ONNX
        # /EHsc is about standard C++ exception handling
        extra_compile_args: list[str] = ["/MD", "/FS", "/EHsc"]
    else:
        extra_link_args = [*make_relative_rpath_args("lib")]
        extra_compile_args = [
            "-Wall",
            "-Wextra",
            "-Wno-strict-overflow",
            "-Wno-unused-parameter",
            "-Wno-missing-field-initializers",
            "-Wno-unknown-pragmas",
            "-fno-strict-aliasing",
        ]

    ext_modules = [
        Extension(
            name="torch_openreg._C",
            sources=["torch_openreg/csrc/stub.c"],
            language="c",
            extra_compile_args=extra_compile_args,
            libraries=["torch_bindings"],
            library_dirs=[os.path.join(BASE_DIR, "torch_openreg/lib")],
            extra_link_args=extra_link_args,
        )
    ]

    package_data = {
        "torch_openreg": [
            "lib/*.so*",
            "lib/*.dylib*",
            "lib/*.dll",
            "lib/*.lib",
        ]
    }

    # LITERALINCLUDE START: SETUP
    setup(
        packages=find_packages(),
        package_data=package_data,
        ext_modules=ext_modules,
        cmdclass={
            "clean": BuildClean,  # type: ignore[misc]
        },
        include_package_data=False,
        entry_points={
            "torch.backends": [
                "torch_openreg = torch_openreg:_autoload",
            ],
        },
    )
    # LITERALINCLUDE END: SETUP


if __name__ == "__main__":
    main()
