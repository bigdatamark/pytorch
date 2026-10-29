# [MODULE DOC] GCC coverage workflow for OSS: run tests and export gcov data as JSON files in
# the profile JSON folder. No import-time side effects.

from __future__ import annotations

import os
import subprocess
import time

# gcc is only used in oss
from ..oss.utils import get_gcda_files, run_oss_python_test
from ..util.setting import JSON_FOLDER_BASE_DIR, TestType
from ..util.utils import print_log, print_time
from .utils import run_cpp_test


def update_gzip_dict(gzip_dict: dict[str, int], file_name: str) -> str:
    # [FUNCTION DOC] HELPER: Counts the file name (lowercased) in the dictionary and returns
    # the name prefixed with its running count, which makes names from different folders
    # unique. Mutates the dictionary.
    file_name = file_name.lower()
    gzip_dict[file_name] = gzip_dict.get(file_name, 0) + 1
    num = gzip_dict[file_name]
    return str(num) + "_" + file_name


def run_target(binary_file: str, test_type: TestType) -> None:
    # [FUNCTION DOC] Runs a C++ or python test binary and prints how long it took. Raises
    # AssertionError for other test types.
    print_log("start run", test_type.value, "test: ", binary_file)
    start_time = time.time()
    if test_type not in {TestType.CPP, TestType.PY}:
        raise AssertionError(f"Invalid test_type: {test_type}")
    if test_type == TestType.CPP:
        run_cpp_test(binary_file)
    else:
        run_oss_python_test(binary_file)

    print_time(" time: ", start_time)


def export() -> None:
    # [FUNCTION DOC] Converts every gcda file to JSON: runs gcov, moves the produced
    # compressed file into the JSON folder under a unique name, and decompresses it. Side
    # effects: runs subprocesses and writes and renames files in the working directory. Prints
    # elapsed time.
    start_time = time.time()
    # collect .gcda files
    gcda_files = get_gcda_files()
    # file name like utils.cpp may have same name in different folder
    gzip_dict: dict[str, int] = {}
    for gcda_item in gcda_files:
        # generate json.gz
        subprocess.check_call(["gcov", "-i", gcda_item])
        # cp json.gz to profile/json folder
        gz_file_name = os.path.basename(gcda_item) + ".gcov.json.gz"
        new_file_path = os.path.join(
            JSON_FOLDER_BASE_DIR, update_gzip_dict(gzip_dict, gz_file_name)
        )
        os.rename(gz_file_name, new_file_path)
        #  unzip json.gz to json
        subprocess.check_output(["gzip", "-d", new_file_path])
    print_time("export take time: ", start_time, summary_time=True)
