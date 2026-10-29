# [MODULE DOC] Path and environment helpers for the OSS coverage flow: locating test binaries,
# shared libraries and the pytorch and llvm folders, detecting the compiler, and handling gcda
# files and python test execution. Reads the CXX, LLVM_TOOL_PATH and PYTORCH_FOLDER variables.

from __future__ import annotations

import os
import subprocess

from ..util.setting import CompilerType, TestType, TOOLS_FOLDER
from ..util.utils import print_error, remove_file


def get_oss_binary_folder(test_type: TestType) -> str:
    # [FUNCTION DOC] Returns the folder holding test programs: build/bin for C++ tests or test
    # for python tests, under the pytorch folder. Raises AssertionError for other test types.
    if test_type not in {TestType.CPP, TestType.PY}:
        raise AssertionError(f"Invalid test_type: {test_type}")
    # TODO: change the way we get binary file -- binary may not in build/bin ?
    return os.path.join(
        get_pytorch_folder(), "build/bin" if test_type == TestType.CPP else "test"
    )


def get_oss_shared_library() -> list[str]:
    # [FUNCTION DOC] Returns the full paths of the dylib files in the build lib folder.
    # KNOWN ISSUE: only matches dylib (macOS) files, so the list is empty for shared objects.
    lib_dir = os.path.join(get_pytorch_folder(), "build", "lib")
    return [
        os.path.join(lib_dir, lib)
        for lib in os.listdir(lib_dir)
        if lib.endswith(".dylib")
    ]


def get_oss_binary_file(test_name: str, test_type: TestType) -> str:
    # [FUNCTION DOC] Returns the command or path to run a test: the binary path,
    # prefixed with the python interpreter command for python tests. Raises
    # AssertionError for other test types.
    if test_type not in {TestType.CPP, TestType.PY}:
        raise AssertionError(f"Invalid test_type: {test_type}")
    binary_folder = get_oss_binary_folder(test_type)
    binary_file = os.path.join(binary_folder, test_name)
    if test_type == TestType.PY:
        # add python to the command so we can directly run the script by using binary_file variable
        binary_file = "python " + binary_file
    return binary_file


def get_llvm_tool_path() -> str:
    # [FUNCTION DOC] Returns the llvm tools folder from LLVM_TOOL_PATH or a default location.
    return os.environ.get(
        "LLVM_TOOL_PATH", "/usr/local/opt/llvm/bin"
    )  # set default as llvm path in dev server, on mac the default may be /usr/local/opt/llvm/bin


def get_pytorch_folder() -> str:
    # [FUNCTION DOC] Returns the absolute pytorch folder, from PYTORCH_FOLDER if set,
    # otherwise two levels above the tools folder.
    # TOOLS_FOLDER in oss: pytorch/tools/code_coverage
    return os.path.abspath(
        os.environ.get("PYTORCH_FOLDER", os.path.dirname(os.path.dirname(TOOLS_FOLDER)))
    )


def detect_compiler_type() -> CompilerType | None:
    # [FUNCTION DOC] Returns the compiler type, from CXX if set (clang, clang++, gcc, g++) or
    # by running the cc compiler with its version flag and checking for clang or gcc. Raises
    # RuntimeError for an unsupported compiler.
    # check if user specifies the compiler type
    user_specify = os.environ.get("CXX", None)
    if user_specify:
        if user_specify in ["clang", "clang++"]:
            return CompilerType.CLANG
        elif user_specify in ["gcc", "g++"]:
            return CompilerType.GCC

        raise RuntimeError(f"User specified compiler is not valid {user_specify}")

    # auto detect
    auto_detect_result = subprocess.check_output(
        ["cc", "-v"], stderr=subprocess.STDOUT
    ).decode("utf-8")
    if "clang" in auto_detect_result:
        return CompilerType.CLANG
    elif "gcc" in auto_detect_result:
        return CompilerType.GCC
    raise RuntimeError(f"Auto detected compiler is not valid {auto_detect_result}")


def clean_up_gcda() -> None:
    # [FUNCTION DOC] Deletes every gcda file found in the build folder.
    gcda_files = get_gcda_files()
    for item in gcda_files:
        remove_file(item)


def get_gcda_files() -> list[str]:
    # [FUNCTION DOC] Returns the paths of gcda files under the build folder (found with the
    # find command), or an empty list when the build folder is missing.
    # KNOWN ISSUE: splitting the output on newlines leaves a trailing empty string entry.
    folder_has_gcda = os.path.join(get_pytorch_folder(), "build")
    if os.path.isdir(folder_has_gcda):
        # TODO use glob
        # output = glob.glob(f"{folder_has_gcda}/**/*.gcda")
        output = subprocess.check_output(["find", folder_has_gcda, "-iname", "*.gcda"])
        return output.decode("utf-8").split("\n")
    else:
        return []


def run_oss_python_test(binary_file: str) -> None:
    # [FUNCTION DOC] Runs a python test command through the shell in the python test folder. A
    # failing exit status is reported with an error message instead of being raised.
    # python test script
    try:
        subprocess.check_call(
            binary_file, shell=True, cwd=get_oss_binary_folder(TestType.PY)
        )
    except subprocess.CalledProcessError:
        print_error(f"Binary failed to run: {binary_file}")
