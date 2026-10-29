# Owner(s): ["module: cpp"]
# [MODULE DOC] Test module verifying that each source file in this (2.10) extension's csrc
# directory requires PyTorch 2.10 or later: it compiles every file with the target version macro
# set to 2.9.0 and expects failure. On non-Windows platforms, defines
# FunctionVersionCompatibilityTest, then at import time discovers all cpp and cu files under
# csrc, attaches one generated test per file to the class, and deletes its temporary names.
# Raises an error at import if csrc is missing. Run with VERSION_COMPAT_DEBUG=1 per its
# docstring to see compile errors. Runnable as a script via run_tests.
# KNOWN ISSUE: the docstring mentions VERSION_COMPAT_DEBUG, but no code in this file reads that
# environment variable (errors are always printed when found).

"""
Unit tests to verify that each function file requires PyTorch 2.10+.

This test suite compiles each .cpp file in the csrc directory with
TORCH_TARGET_VERSION=2.9.0 and expects compilation to fail.
If compilation succeeds, it means that either

(1) The test function works with 2.9.0 and should not be in this directory.
(2) The test function tests APIs that do not have proper TORCH_FEATURE_VERSION
    guards. If this is the case, and you incorrectly move the test function into
    libtorch_agn_2_9_extension the libtorch_agnostic_targetting CI workflow
    will catch this.

Run this script with VERSION_COMPAT_DEBUG=1 to see compilation errors.
"""

import os
import subprocess
import tempfile
from pathlib import Path

from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase
from torch.utils.cpp_extension import (
    CUDA_HOME,
    get_cxx_compiler,
    include_paths as torch_include_paths,
    ROCM_HOME,
)


GPU_HOME = CUDA_HOME or ROCM_HOME

# TODO: Fix this error in Windows:
# numba.cuda.cudadrv.driver:driver.py:384 Call to cuInit results in CUDA_ERROR_NO_DEVICE
if not IS_WINDOWS:

    class FunctionVersionCompatibilityTest(TestCase):
        # [CLASS DOC] Test case (defined only when not on Windows) that checks every C++ or CUDA
        # source in this extension's csrc directory fails to compile when the target version
        # macro is set to 2.9.0, proving it needs 2.10 or later. Class state set in setUpClass:
        # csrc_dir, csrc_dir_2_9, a temporary build_dir, include flags for PyTorch and
        # CUDA/ROCm, and cuda_available. One test method per source file is attached dynamically
        # after the class body, plus two control tests that sources of the 2.9 extension do
        # compile.
        """Test that all function files require PyTorch 2.10+."""

        @classmethod
        def setUpClass(cls):
            # [FUNCTION DOC] Computes the csrc directories of this and the 2.9 extension,
            # creates a temporary build directory, collects PyTorch include flags and the CUDA
            # or ROCm include flag if the include dir exists, and records whether CUDA is
            # available. Side effect: creates a temp directory.
            """Set up test environment once for all tests."""
            ext_dir = Path(__file__).parent
            cls.csrc_dir = ext_dir / "csrc"
            cls.csrc_dir_2_9 = ext_dir.parent / "libtorch_agn_2_9_extension" / "csrc"
            cls.build_dir = Path(tempfile.mkdtemp(prefix="version_check_"))

            cls.pytorch_includes = [
                f"-I{path}" for path in torch_include_paths(device_type="cpu")
            ]
            cls.cuda_includes = []
            if GPU_HOME:
                cuda_include_path = os.path.join(GPU_HOME, "include")
                if os.path.exists(cuda_include_path):
                    cls.cuda_includes = [f"-I{cuda_include_path}"]

            cls.cuda_available = cls._check_cuda_available()

        @classmethod
        def tearDownClass(cls):
            # [FUNCTION DOC] HELPER: Removes the temporary build directory if it exists.
            """Clean up build directory."""
            import shutil

            if cls.build_dir.exists():
                shutil.rmtree(cls.build_dir)

        @staticmethod
        def _check_cuda_available() -> bool:
            # [FUNCTION DOC] HELPER: Returns whether torch reports CUDA as available, or false
            # if torch cannot be imported.
            """Check if CUDA is available."""
            try:
                import torch

                return torch.cuda.is_available()
            except ImportError:
                return False

        def _compile_cpp_file(
            self, source_file: Path, output_file: Path
        ) -> tuple[bool, str]:
            # [FUNCTION DOC] HELPER: Compiles one C++ source to an object file with the system
            # C++ compiler, standard C++20, the target version macro set to 2.9.0, and PyTorch
            # (and, when CUDA is available, CUDA) include paths; 30 second timeout. Returns a
            # pair (succeeded, stderr text, empty on success).
            """
            Compile a C++ file with TORCH_TARGET_VERSION=2.9.0.
            Returns (success, error_message).
            """
            torch_version_2_9 = "0x0209000000000000"

            cmd = [
                get_cxx_compiler(),
                "-c",
                "-std=c++20",
                f"-DTORCH_TARGET_VERSION={torch_version_2_9}",
                f"-I{source_file.parent}",  # For includes in same directory
                *self.pytorch_includes,
            ]

            # Add CUDA flags if available
            if self.cuda_available:
                cmd.extend(self.cuda_includes)

            cmd.extend([str(source_file), "-o", str(output_file)])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                return True, ""
            else:
                return False, result.stderr

        def _compile_cu_file(
            self, source_file: Path, output_file: Path
        ) -> tuple[bool, str]:
            # [FUNCTION DOC] HELPER: Same as the C++ variant but uses nvcc or hipcc from the GPU
            # home with the CUDA define (and the ROCm define on ROCm). Returns a failure pair
            # with an explanatory message when neither CUDA_HOME nor ROCM_HOME is set. 30 second
            # timeout.
            """
            Compile a CUDA file with TORCH_TARGET_VERSION=2.9.0.
            Returns (success, error_message).
            """
            if not GPU_HOME:
                return False, "one of CUDA_HOME and ROCM_HOME should be set but is not"

            torch_version_2_9 = "0x0209000000000000"

            cmd = [
                os.path.join(GPU_HOME, "bin", "nvcc" if CUDA_HOME else "hipcc"),
                "-c",
                "-std=c++20",
                f"-DTORCH_TARGET_VERSION={torch_version_2_9}",
                f"-I{source_file.parent}",  # For includes in same directory
                *self.pytorch_includes,
                *self.cuda_includes,
            ]

            cmd.extend(["-DUSE_CUDA"])
            if ROCM_HOME:
                cmd.extend(["-DUSE_ROCM=1"])

            cmd.extend([str(source_file), "-o", str(output_file)])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                return True, ""
            else:
                return False, result.stderr

        def _test_function_file(self, source_file: Path):
            # [FUNCTION DOC] HELPER: Shared body of the generated tests: compiles the given
            # source with the 2.9.0 target (skips the test for .cu files when CUDA is
            # unavailable), deletes the object file, prints the relevant compiler errors if it
            # failed, and asserts that compilation failed, with a long explanatory message
            # otherwise.
            """Test that a function file fails to compile with TORCH_TARGET_VERSION=2.9.0."""
            func_name = source_file.stem
            obj_file = self.build_dir / f"{func_name}.o"

            # Choose the appropriate compiler based on file extension
            if source_file.suffix == ".cu":
                if not self.cuda_available:
                    self.skipTest(f"CUDA not available, skipping {source_file.name}")
                success, error_msg = self._compile_cu_file(source_file, obj_file)
            else:
                success, error_msg = self._compile_cpp_file(source_file, obj_file)

            obj_file.unlink(missing_ok=True)

            # Print error details for debugging
            if not success:
                relevant_errors = self._extract_relevant_errors(error_msg)
                if relevant_errors:
                    print(f"\n  Compilation errors for {func_name} (requires 2.10+):")
                    for err in relevant_errors:
                        print(f"    {err}")

            self.assertFalse(
                success,
                lambda msg: f"{msg}\nFunction {func_name} compiled successfully with TORCH_TARGET_VERSION=2.9.0. "
                f"This could mean two things.\n\t1. It should run with 2.9.0 and should be "
                "moved to libtorch_agn_2_9_extension\n\t2. The function(s) it tests do not use the "
                "proper TORCH_FEATURE_VERSION guards\n\nThe libtorch_agnostic_targetting CI workflow will "
                "verify if you incorrectly move this to the 2_9 extension instead of adding "
                "the appropriate version guards.",
            )

        def test_kernel_works_with_2_9(self):
            # [FUNCTION DOC] Control test: the 2.9 extension's kernel.cpp must exist and compile
            # successfully with target version 2.9.0; on failure prints the relevant errors and
            # fails the assertion. Validates that the harness can tell compilable from
            # non-compilable sources.
            """Test that the 2.9 extension's kernel.cpp compiles successfully with 2.9.0.

            This is a control test - it ensures that a file we expect to work with 2.9.0
            actually does compile. Every other test in this suite asserts that
            compilation *fails* for sources that require 2.10+, so we have this test to
            validate that our test infra correctly distinguishes when a source can compile
            fine with 2.9.
            """
            cpp_file = self.csrc_dir_2_9 / "kernel.cpp"
            self.assertTrue(cpp_file.exists(), f"{cpp_file} does not exist")

            obj_file = self.build_dir / "kernel.o"
            success, error_msg = self._compile_cpp_file(cpp_file, obj_file)

            # Clean up
            obj_file.unlink(missing_ok=True)

            if not success:
                relevant_errors = self._extract_relevant_errors(error_msg)
                if relevant_errors:
                    print("\n  Unexpected compilation errors for kernel.cpp:")
                    for err in relevant_errors:
                        print(f"{err}")

            self.assertTrue(
                success,
                lambda msg: f"{msg}\nkernel.cpp failed to compile with TORCH_TARGET_VERSION=2.9.0. "
                f"This file is expected to work with 2.9.0 since it doesn't use 2.10+ features. "
                f"Error: {error_msg}",
            )

        def test_cuda_kernel_works_with_2_9(self):
            # [FUNCTION DOC] Control test: the 2.9 extension's cuda_kernel.cu must exist and
            # compile with target version 2.9.0. Skipped when CUDA is unavailable.
            """Test that cuda_kernel.cu compiles successfully with 2.9.0.

            This is a control test - it ensures that a .cu file we expect to work with 2.9.0
            actually does compile. This validates that our test infrastructure correctly
            compiles CUDA files and distinguishes between files that require 2.10+ and those
            that don't.
            """
            if not self.cuda_available:
                self.skipTest("CUDA not available, skipping cuda_kernel.cu test")

            cu_file = self.csrc_dir_2_9 / "cuda_kernel.cu"
            self.assertTrue(cu_file.exists(), f"{cu_file} does not exist")

            obj_file = self.build_dir / "cuda_kernel.o"
            success, error_msg = self._compile_cu_file(cu_file, obj_file)

            # Clean up
            obj_file.unlink(missing_ok=True)

            if not success:
                relevant_errors = self._extract_relevant_errors(error_msg)
                if relevant_errors:
                    print("\n  Unexpected compilation errors for cuda_kernel.cu:")
                    for err in relevant_errors:
                        print(f"{err}")

            self.assertTrue(
                success,
                lambda msg: f"{msg}\ncuda_kernel.cu failed to compile with TORCH_TARGET_VERSION=2.9.0. "
                f"This file is expected to work with 2.9.0 since it doesn't use 2.10+ features. "
                f"Error: {error_msg}",
            )

        @staticmethod
        def _extract_relevant_errors(error_msg: str) -> list[str]:
            # [FUNCTION DOC] HELPER: Static method that splits compiler output into lines and
            # returns the stripped lines mentioning an error, undefined, undeclared or no member
            # named (case-insensitive).
            """Extract the most relevant error messages."""
            error_lines = error_msg.strip().split("\n")
            relevant_errors = []

            for line in error_lines:
                line_lower = line.lower()
                if (
                    "error:" in line_lower
                    or "undefined" in line_lower
                    or "undeclared" in line_lower
                    or "no member named" in line_lower
                ):
                    relevant_errors.append(line.strip())

            return relevant_errors

    # Dynamically create test methods for each .cpp and .cu file

    def _create_test_method_for_file(source_file: Path):
        # [FUNCTION DOC] HELPER: Factory returning a test method that runs the shared
        # compile-must-fail check for the given source file; the method is renamed after the
        # file stem with a requires_2_10 suffix and given a descriptive docstring.
        """Create a test method for a specific source file."""

        def test_method_impl(self):
            # [FUNCTION DOC] HELPER: The generated test body; delegates to the shared
            # compile-must-fail check for the captured source file.
            self._test_function_file(source_file)

        # Set a descriptive name and docstring
        func_name = source_file.stem
        file_ext = source_file.suffix
        test_method_impl.__name__ = f"test_{func_name}_requires_2_10"
        test_method_impl.__doc__ = (
            f"Test that {func_name}{file_ext} requires PyTorch 2.10+"
        )

        return test_method_impl

    # Test discovery: generate a test for each .cpp and .cu file
    _csrc_dir = Path(__file__).parent / "csrc"
    if not _csrc_dir.exists():
        raise AssertionError(f"Expected csrc directory to exist at {_csrc_dir}")
    # Collect both .cpp and .cu files. The control tests defined above compile
    # sources from the 2.9 extension, which is not globbed here.
    _source_files = sorted([*_csrc_dir.rglob("*.cpp"), *_csrc_dir.rglob("*.cu")])

    for _source_file in _source_files:
        _test_method = _create_test_method_for_file(_source_file)
        setattr(FunctionVersionCompatibilityTest, _test_method.__name__, _test_method)

    del (
        _create_test_method_for_file,
        _csrc_dir,
        _source_files,
        _source_file,
        _test_method,
    )

if __name__ == "__main__":
    run_tests()
