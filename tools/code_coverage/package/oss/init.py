# [MODULE DOC] Initialization for the OSS coverage flow: creates folders, parses command line
# arguments, discovers the test binaries and scripts, and prints configuration. Defines the
# set of python tests skipped under clang. No import-time side effects besides constants.

from __future__ import annotations

import argparse
import os
from typing import cast

from ..util.setting import (
    CompilerType,
    JSON_FOLDER_BASE_DIR,
    LOG_DIR,
    Option,
    Test,
    TestList,
    TestType,
)
from ..util.utils import (
    clean_up,
    create_folder,
    print_log,
    raise_no_test_found_exception,
    remove_file,
    remove_folder,
)
from ..util.utils_init import add_arguments_utils, create_folders, get_options
from .utils import (
    clean_up_gcda,
    detect_compiler_type,
    get_llvm_tool_path,
    get_oss_binary_folder,
    get_pytorch_folder,
)


BLOCKED_PYTHON_TESTS = {
    "run_test.py",
    "test_dataloader.py",
    "test_multiprocessing.py",
    "test_multiprocessing_spawn.py",
    "test_utils.py",
}


def initialization() -> tuple[Option, TestList, list[str]]:
    # [FUNCTION DOC] Sets up a coverage run: creates the working folders, parses arguments,
    # cleans old data if requested, builds the test list, determines the folders of interest,
    # prints settings and removes the previous log file. Returns the options, test list and
    # interested folders. Side effects: file system changes and console output.
    # create folder if not exists
    create_folders()
    # add arguments
    parser = argparse.ArgumentParser()
    parser = add_arguments_utils(parser)
    parser = add_arguments_oss(parser)
    # parse arguments
    (options, args_interested_folder, args_run_only, arg_clean) = parse_arguments(
        parser
    )
    # clean up
    if arg_clean:
        clean_up_gcda()
        clean_up()
    # get test lists
    test_list = get_test_list(args_run_only)
    # get interested folder -- final report will only over these folders
    interested_folders = empty_list_if_none(args_interested_folder)
    # print initialization information
    print_init_info()
    # remove last time's log
    remove_file(os.path.join(LOG_DIR, "log.txt"))
    return (options, test_list, interested_folders)


def add_arguments_oss(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # [FUNCTION DOC] Adds the OSS specific --run-only option (list of tests to run) to the
    # parser and returns the parser.
    parser.add_argument(
        "--run-only",
        help="only run certain test(s), for example: atest test_nn.py.",
        nargs="*",
        default=None,
    )

    return parser


def parse_arguments(
    parser: argparse.ArgumentParser,
) -> tuple[Option, list[str] | None, list[str] | None, bool | None]:
    # [FUNCTION DOC] Parses command line arguments and returns the options object, interested
    # folders, run-only list and clean flag.
    # parse args
    args = parser.parse_args()
    # get option
    options = get_options(args)
    return (options, args.interest_only, args.run_only, args.clean)


def get_test_list_by_type(run_only: list[str] | None, test_type: TestType) -> TestList:
    # [FUNCTION DOC] Builds the test list for one test type by walking the binary folder and
    # creating a Test for each file, filtered by the run-only list when given. Walks nested
    # folders but uses file names only.
    test_list: TestList = []
    binary_folder = get_oss_binary_folder(test_type)
    g = os.walk(binary_folder)
    for _, _, file_list in g:
        for file_name in file_list:
            if run_only is not None and file_name not in run_only:
                continue
            # target pattern in oss is used in printing report -- which tests we have run
            test: Test = Test(
                name=file_name,
                target_pattern=file_name,
                test_set="",
                test_type=test_type,
            )
            test_list.append(test)
    return test_list


def get_test_list(run_only: list[str] | None) -> TestList:
    # [FUNCTION DOC] Returns the combined list of C++ and python tests to run. Raises an error
    # (via raise_no_test_found_exception) when no tests are found.
    test_list: TestList = []
    # add c++ test list
    test_list.extend(get_test_list_by_type(run_only, TestType.CPP))
    # add python test list
    py_run_only = get_python_run_only(run_only)
    test_list.extend(get_test_list_by_type(py_run_only, TestType.PY))

    # not find any test to run
    if not test_list:
        raise_no_test_found_exception(
            get_oss_binary_folder(TestType.CPP), get_oss_binary_folder(TestType.PY)
        )
    return test_list


def empty_list_if_none(arg_interested_folder: list[str] | None) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns the given list, or an empty list when it is None.
    if arg_interested_folder is None:
        return []
    # if this argument is specified, just return itself
    return arg_interested_folder


def gcc_export_init() -> None:
    # [FUNCTION DOC] HELPER: Recreates the base JSON output folder (removes it, then creates
    # it). Side effect: deletes existing JSON profile data.
    remove_folder(JSON_FOLDER_BASE_DIR)
    create_folder(JSON_FOLDER_BASE_DIR)


def get_python_run_only(args_run_only: list[str] | None) -> list[str]:
    # [FUNCTION DOC] Returns the python test file names to run: the user's run-only list if
    # given; otherwise run_test.py for gcc, or for clang all top-level python files in the
    # test folder except the blocked ones (their intermediate files are too large to merge).
    # if user specifies run-only option
    if args_run_only:
        return args_run_only

    # if not specified, use default setting, different for gcc and clang
    if detect_compiler_type() == CompilerType.GCC:
        return ["run_test.py"]
    else:
        # for clang, some tests will result in too large intermediate files that can't be merged by llvm, we need to skip them
        run_only: list[str] = []
        binary_folder = get_oss_binary_folder(TestType.PY)
        g = os.walk(binary_folder)
        for _, _, file_list in g:
            for file_name in file_list:
                if file_name in BLOCKED_PYTHON_TESTS or not file_name.endswith(".py"):
                    continue
                run_only.append(file_name)
            # only run tests in the first-level folder in test/
            break
        return run_only


def print_init_info() -> None:
    # [FUNCTION DOC] Prints the pytorch folder, test binary and script folders, compiler type
    # and llvm tool folder to the log.
    # KNOWN ISSUE: assumes the compiler type is not None; a None result would fail
    # on the value.
    print_log("pytorch folder: ", get_pytorch_folder())
    print_log("cpp test binaries folder: ", get_oss_binary_folder(TestType.CPP))
    print_log("python test scripts folder: ", get_oss_binary_folder(TestType.PY))
    print_log("compiler type: ", cast(CompilerType, detect_compiler_type()).value)
    print_log(
        "llvm tool folder (only for clang, if you are using gcov please ignore it): ",
        get_llvm_tool_path(),
    )
