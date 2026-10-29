# [MODULE DOC] Shared constants and types for the code coverage tool: folder locations under
# the tool's profile directory, plus enums and small classes for tests, options, platforms and
# compilers.
# Import-time side effects: reads the HOME environment variable (KeyError if unset) and computes
# the folder paths.
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path


# <project folder>
HOME_DIR = os.environ["HOME"]
TOOLS_FOLDER = str(Path(__file__).resolve().parents[2])


# <profile folder>
PROFILE_DIR = os.path.join(TOOLS_FOLDER, "profile")
JSON_FOLDER_BASE_DIR = os.path.join(PROFILE_DIR, "json")
MERGED_FOLDER_BASE_DIR = os.path.join(PROFILE_DIR, "merged")
SUMMARY_FOLDER_DIR = os.path.join(PROFILE_DIR, "summary")

# <log path>
LOG_DIR = os.path.join(PROFILE_DIR, "log")


# test type, DO NOT change the name, it should be consistent with [buck query --output-attribute] result
class TestType(Enum):
    # [CLASS DOC] Enumeration of test kinds; values match the buck query output attribute names
    # (cxx_test for C++ tests, python_test for Python tests).
    CPP = "cxx_test"
    PY = "python_test"


class Test:
    # [CLASS DOC] Plain record describing one test target: its name, buck target pattern,
    # test set label and TestType. Holds no behavior beyond storing the four attributes.
    name: str
    target_pattern: str
    test_set: str  # like __aten__
    test_type: TestType

    def __init__(
        self, name: str, target_pattern: str, test_set: str, test_type: TestType
    ) -> None:
        # [FUNCTION DOC] HELPER: Stores the four constructor arguments as attributes.
        self.name = name
        self.target_pattern = target_pattern
        self.test_set = test_set
        self.test_type = test_type


TestList = list[Test]
TestStatusType = dict[str, set[str]]


# option
class Option:
    # [CLASS DOC] Holder of class-level boolean flags selecting which pipeline stages run
    # (build, run, merge, export, summary, pytest). All default to False and are set globally
    # on the class by the argument handling code; it is never instantiated.
    need_build: bool = False
    need_run: bool = False
    need_merge: bool = False
    need_export: bool = False
    need_summary: bool = False
    need_pytest: bool = False


# test platform
class TestPlatform(Enum):
    # [CLASS DOC] Enumeration of supported platforms: fbcode (internal) and oss (open source).
    FBCODE = "fbcode"
    OSS = "oss"


# compiler type
class CompilerType(Enum):
    # [CLASS DOC] Enumeration of compilers whose coverage output is understood: clang (llvm
    # profile data) and gcc (gcov).
    CLANG = "clang"
    GCC = "gcc"
