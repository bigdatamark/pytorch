# [MODULE DOC] Build script (setuptools) for the libtorch_agn_2_14 test extension, an example of
# a libtorch-agnostic (stable ABI, abi3) extension targeting PyTorch 2.14. It also compiles the
# csrc of the earlier version extensions (2.9 to 2.13) for forward compatibility testing.
# Defines ROOT_DIR and CSRC_DIR, a custom clean command and get_extension. Executing it calls
# setup (BuildExtension with no python ABI suffix, plus the clean command); CUDA sources are
# included only when CUDA is available.
import distutils.command.clean
import shutil
from pathlib import Path

from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    IS_WINDOWS,
)


ROOT_DIR = Path(__file__).parent
CSRC_DIR = ROOT_DIR / "csrc"

# Include csrc from previous versions for forward compatibility testing
PREV_CSRC_DIRS = [
    ROOT_DIR.parent / "libtorch_agn_2_9_extension" / "csrc",
    ROOT_DIR.parent / "libtorch_agn_2_10_extension" / "csrc",
    ROOT_DIR.parent / "libtorch_agn_2_11_extension" / "csrc",
    ROOT_DIR.parent / "libtorch_agn_2_12_extension" / "csrc",
    ROOT_DIR.parent / "libtorch_agn_2_13_extension" / "csrc",
]


class clean(distutils.command.clean.clean):
    # [CLASS DOC] Custom distutils clean command: after the default clean it deletes built
    # shared objects in the package folder and the build, dist and egg-info directories next to
    # the script.
    def run(self):
        # [FUNCTION DOC] Runs the default clean, then unlinks every shared object found under
        # the libtorch_agn_2_14 package folder, then recursively removes the build, dist and
        # egg-info directories if they exist (removal errors ignored). Side effect: file
        # deletion.
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove extension
        for path in (ROOT_DIR / "libtorch_agn_2_14").glob("**/*.so"):
            path.unlink()
        # Remove build and dist and egg-info directories
        dirs = [
            ROOT_DIR / "build",
            ROOT_DIR / "dist",
            ROOT_DIR / "libtorch_agn_2_14.egg-info",
        ]
        for path in dirs:
            if path.exists():
                shutil.rmtree(str(path), ignore_errors=True)


def get_extension():
    # [FUNCTION DOC] Returns two extensions: the op extension _C (this version's csrc plus
    # inherited 2.9 to 2.13 csrc, excluding the interop module source, and excluding mps sources
    # unless MPS is available) and a separate importable _interop abi3 extension built from the
    # interop module source. Both get the target version macro; _C also gets the library name
    # macro, CUDA flags/sources when CUDA is available, and an MPS define when MPS is available.
    common_cxx = ["-DTORCH_TARGET_VERSION=0x020e000000000000"]
    if not IS_WINDOWS:
        common_cxx.append("-fdiagnostics-color=always")

    all_csrc_dirs = [CSRC_DIR, *PREV_CSRC_DIRS]
    mps_available = torch.backends.mps.is_available()

    # Op extension (_C): this version's ops + inherited 2.9-2.13 csrc, minus the
    # interop module. The op extension is loaded via torch.ops.load_library like
    # the other libtorch_agn extensions; the interop module must instead be an
    # importable Python module (its PyMethodDef helpers hold the GIL), so it is
    # built as a separate extension below.
    op_sources = [
        s
        for d in all_csrc_dirs
        for s in d.glob("**/*.cpp")
        if s.name != "pyobject_interop_module.cpp"
        and (s.parent.name != "mps" or mps_available)
    ]

    op_cxx = common_cxx + ["-DSTABLE_LIB_NAME=libtorch_agn_2_14"]
    op_extra = {"cxx": op_cxx}
    op_extension = CppExtension
    # allow including <cuda_runtime.h>
    if torch.cuda.is_available():
        op_cxx.append("-DLAE_USE_CUDA")
        op_extra["nvcc"] = [
            "-O2",
            "-DUSE_CUDA",
            "-DTORCH_TARGET_VERSION=0x020e000000000000",
            "-DSTABLE_LIB_NAME=libtorch_agn_2_14",
        ]
        op_extension = CUDAExtension
        op_sources.extend(CSRC_DIR.glob("**/*.cu"))
        for prev_dir in PREV_CSRC_DIRS:
            op_sources.extend(prev_dir.glob("**/*.cu"))

    if mps_available:
        op_cxx.append("-DUSE_MPS")

    return [
        op_extension(
            "libtorch_agn_2_14._C",
            sources=sorted(str(s) for s in op_sources),
            py_limited_api=True,
            extra_compile_args=op_extra,
            extra_link_args=[],
        ),
        # Interop extension (_interop): an importable abi3 module. Its
        # PyObject<->Tensor helpers are plain module functions, so it is imported
        # (which runs PyInit and exposes them) rather than loaded via
        # torch.ops.load_library like the op extension.
        CppExtension(
            "libtorch_agn_2_14._interop",
            sources=[str(CSRC_DIR / "pyobject_interop_module.cpp")],
            py_limited_api=True,
            extra_compile_args={
                "cxx": common_cxx + ["-DSTABLE_LIB_NAME=libtorch_agn_2_14_interop"]
            },
        ),
    ]


setup(
    name="libtorch_agn_2_14",
    version="0.0",
    author="PyTorch Core Team",
    description="Example of libtorch agnostic extension for PyTorch 2.14+",
    packages=find_packages(exclude=("test",)),
    package_data={"libtorch_agn_2_14": ["*.dll", "*.dylib", "*.so"]},
    install_requires=[
        "torch",
    ],
    ext_modules=get_extension(),
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
        "clean": clean,
    },
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
