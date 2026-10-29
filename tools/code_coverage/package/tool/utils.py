# [MODULE DOC] Small helpers for the code coverage tool: running a C++ test binary and locating
# the llvm tool folder for the current platform.
import subprocess

from ..util.setting import TestPlatform
from ..util.utils import print_error


def run_cpp_test(binary_file: str) -> None:
    # [FUNCTION DOC] Runs the given C++ test binary as a subprocess and waits for it.
    # A non-zero exit status is reported via print_error and not raised. Side effect: process
    # execution and console output.
    # cpp test binary
    try:
        subprocess.check_call(binary_file)
    except subprocess.CalledProcessError:
        print_error(f"Binary failed to run: {binary_file}")


def get_tool_path_by_platform(platform: TestPlatform) -> str:
    # [FUNCTION DOC] Returns the path of the llvm tools folder for the platform. Uses the
    # internal fbcode helper for FBCODE and the OSS helper otherwise; the import is done lazily
    # so only the needed helper module must exist.
    if platform == TestPlatform.FBCODE:
        from caffe2.fb.code_coverage.tool.package.fbcode.utils import (  # type: ignore[import]
            get_llvm_tool_path,
        )

        return get_llvm_tool_path()  # type: ignore[no-any-return]
    else:
        from ..oss.utils import get_llvm_tool_path  # type: ignore[no-redef]

        return get_llvm_tool_path()  # type: ignore[no-any-return]
