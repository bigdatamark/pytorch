# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] Build script for the python_agnostic example extension. Defines a custom
# clean command and an extension factory choosing CUDA (.cu sources) or SYCL/XPU (.sycl
# sources) under python_agnostic/csrc, built with the limited Python API (abi3, cp39).
# Calls setup at import time (it is meant to be run as a script).

import distutils.command.clean
import shutil
from pathlib import Path

from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, SyclExtension


ROOT_DIR = Path(__file__).parent
CSRC_DIR = ROOT_DIR / "python_agnostic" / "csrc"


class clean(distutils.command.clean.clean):
    # [CLASS DOC] HELPER: Setuptools clean command that also deletes built shared libraries
    # and the build, dist and egg-info directories. Stateless.
    def run(self):
        # [FUNCTION DOC] HELPER: Runs the default clean, then removes every .so file under the
        # package directory and the build, dist and egg-info directories (errors ignored for
        # directories). Side effect: deletes files and directories on disk.
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove extension
        for path in (ROOT_DIR / "python_agnostic").glob("**/*.so"):
            path.unlink()
        # Remove build and dist and egg-info directories
        dirs = [
            ROOT_DIR / "build",
            ROOT_DIR / "dist",
            ROOT_DIR / "python_agnostic.egg-info",
        ]
        for path in dirs:
            if path.exists():
                shutil.rmtree(str(path), ignore_errors=True)


def get_extension():
    # [FUNCTION DOC] HELPER: Returns a one-element list with the python_agnostic._C extension.
    # Uses a CUDA extension with .cu sources if CUDA is available, else a SYCL extension with
    # .sycl sources if XPU is available, else raises AssertionError. Sources are sorted and
    # the limited API flag is enabled.
    extra_compile_args = {
        "cxx": ["-fdiagnostics-color=always"],
    }

    if torch.cuda.is_available():
        sources = list(CSRC_DIR.glob("**/*.cu"))
        extension = CUDAExtension
    elif torch.xpu.is_available():
        sources = list(CSRC_DIR.glob("**/*.sycl"))
        extension = SyclExtension
    else:
        raise AssertionError("Expected CUDA or XPU device backend, found none")

    return [
        extension(
            "python_agnostic._C",
            sources=sorted(str(s) for s in sources),
            py_limited_api=True,
            extra_compile_args=extra_compile_args,
            extra_link_args=[],
        )
    ]


setup(
    name="python_agnostic",
    version="0.0",
    author="PyTorch Core Team",
    description="Example of python agnostic extension",
    ext_modules=get_extension(),
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
        "clean": clean,
    },
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
