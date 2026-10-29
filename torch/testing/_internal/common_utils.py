# mypy: allow-untyped-defs
# [MODULE DOC] Shared infrastructure for PyTorch unit tests: the TestCase base class, the test
# runner, parametrization decorators, skip/xfail decorators, environment-driven flags, numerical
# comparison helpers, subprocess/multiprocess helpers, and many small utilities used by test
# files across the repository. Import-time side effects: reads environment variables to define
# global flags and settings (through TestEnvironment), loads the slow-test and disabled-test
# JSON files named in the environment, disables global torch.backends flags, replaces the call
# methods of TorchScript function and method objects with profiling-aware wrappers, and may
# configure dynamo and inductor when running under compile modes. Importing this file must not
# initialize a CUDA context (see the module docstring).

r"""Importing this file must **not** initialize CUDA context. test_distributed
relies on this assumption to properly run. This means that when this is imported
no CUDA calls shall be made, including torch.cuda.device_count(), etc.

torch.testing._internal.common_cuda.py can freely initialize CUDA context when imported.
"""

import sysconfig
import argparse
import contextlib
import copy
import ctypes
import errno
import functools
import gc
import hashlib
import inspect
import io
import json
import logging
import math
import operator
import os
import pathlib
import platform
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import warnings
from collections.abc import Mapping, Sequence
from contextlib import closing, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import partial, wraps
from itertools import product, chain
from pathlib import Path
from statistics import mean
from typing import (
    Any,
    TypeVar,
)
from collections.abc import Callable
from collections.abc import Iterable, Iterator
from unittest.mock import MagicMock

import expecttest
import numpy as np

import __main__  # type: ignore[import]
import torch
import torch.backends.cudnn
import torch.backends.mkl
import torch.backends.mps
import torch.backends.xnnpack
import torch.cuda
from torch import Tensor
from torch._C import ScriptDict, ScriptList  # type: ignore[attr-defined]
from torch._utils_internal import get_writable_path
from torch._logging.scribe import open_source_signpost
from torch.nn import (
    ModuleDict,
    ModuleList,
    ParameterDict,
    ParameterList,
    Sequential,
)
from torch.onnx import (
    register_custom_op_symbolic,
    unregister_custom_op_symbolic,
)
from torch.testing import make_tensor
from torch.testing._comparison import (
    _unwrap_dtensor_for_comparison,
    BooleanPair,
    NonePair,
    not_close_error_metas,
    NumberPair,
    Pair,
    TensorLikePair,
)
from torch.testing._internal.common_dtype import get_all_dtypes
from torch.utils._import_utils import _check_module_exists
import torch.utils._pytree as pytree
from torch.utils import cpp_extension
from torch._utils import _is_privateuse1_backend_available
try:
    import pytest  # type: ignore[import-not-found]
    has_pytest = True
except ImportError:
    has_pytest = False

SEED = 1234
MI350_ARCH = ("gfx950",)
MI300_ARCH = ("gfx942",)
MI200_ARCH = ("gfx90a",)
NAVI_ARCH = ("gfx1030", "gfx1100", "gfx1101", "gfx1200", "gfx1201")
NAVI3_ARCH = ("gfx1100", "gfx1101")
NAVI4_ARCH = ("gfx1200", "gfx1201")

class ProfilingMode(Enum):
    # [CLASS DOC] Enumeration of the JIT graph executor modes: LEGACY, SIMPLE and PROFILING.
    # Stored in the global GRAPH_EXECUTOR and used to decide whether tests run with the
    # profiling executor enabled.
    LEGACY = 1
    SIMPLE = 2
    PROFILING = 3

class HardwareClassification(Enum):
    # [CLASS DOC] Enumeration of hardware categories a test class can declare via its
    # hw_classification attribute (generic, accelerator, or a specific device). Used with the
    # --hw-classification command line option to select which test classes run.
    """Hardware classification metadata for test classes.

    Test classes declare a ``hw_classification`` class attribute to indicate
    the kind of hardware their tests require.  When ``--hw-classification``
    is passed, only tests whose classification matches one of the requested
    values are executed.  When the flag is not specified, all test discovery
    and execution paths remain unchanged.

    Currently there are three hardware classification categories:

    * ``GENERIC`` – tests that exercise shared, device-agnostic logic
      (e.g. Dynamo dispatcher, FX passes, and other framework internals
      that do not depend on a particular accelerator).  These test classes
      typically do **not** use
      :func:`~torch.testing._internal.common_utils.instantiate_device_type_tests`.

    * ``ACCELERATOR`` – tests that verify behavior which must hold
      across every accelerator (e.g. operator semantics, memory profiling).
      These test classes **are** instantiated via
      :func:`~torch.testing._internal.common_utils.instantiate_device_type_tests`.

    * ``CPU``, ``CUDA``, ``MPS``, ``XPU`` – tests tied to a specific device.
      Use sparingly, and only for device-specific behavior.  These replace
      ``@onlyCPU``, ``@onlyCUDA``, and similar decorators

    Usage::

        class TestFoo(TestCase):
            hw_classification = HardwareClassification.GENERIC

            def test_bar(self):
                ...

    Run only GENERIC and ACCELERATOR tests:

        python test/test_torch.py --hw-classification GENERIC ACCELERATOR

    """
    GENERIC = "generic"
    ACCELERATOR = "accelerator"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    XPU = "xpu"


# Set by parse_cmd_line_args() if called
DISABLED_TESTS_FILE = ""
GRAPH_EXECUTOR : ProfilingMode | None = None
LOG_SUFFIX = ""
PYTEST_SINGLE_TEST = ""
REPEAT_COUNT = 0
RERUN_DISABLED_TESTS = False
RUN_PARALLEL = 0
SHOWLOCALS = False
SLOW_TESTS_FILE = ""
TEST_BAILOUTS = False
TEST_DISCOVER = False
TEST_IN_SUBPROCESS = False
TEST_SAVE_XML = ""
UNITTEST_ARGS : list[str] = []
USE_PYTEST = False
HW_CLASSIFICATION : set[HardwareClassification] | None = None


def get_hw_classification(
    test_case_cls: type[unittest.TestCase],
) -> HardwareClassification | None:
    # [FUNCTION DOC] Return the hardware classification declared by a test class through its
    # hw_classification attribute, or None when the class does not declare one. Raises TypeError
    # if the attribute is present but is not a HardwareClassification.
    requirement = getattr(test_case_cls, "hw_classification", None)
    if requirement is None:
        return None

    if not isinstance(requirement, HardwareClassification):
        raise TypeError(
            f"{test_case_cls.__module__}.{test_case_cls.__name__}."
            "hw_classification must be a HardwareClassification"
        )

    return requirement


def filter_by_hw_classification(
    items: Iterable[Any],
    requirement: set[HardwareClassification],
    get_class: Callable[[Any], type | None],
    *,
    on_match: Callable[[Any], None],
) -> None:
    # [FUNCTION DOC] Walk a collection of items and keep those whose test class has a hardware
    # classification contained in the requested set. The class of each item is obtained with the
    # get_class callable; items with no class are counted separately. Each matching item is
    # passed to the on_match callback. Prints one summary line with the counts of total, passed,
    # unclassified, no-class and mismatched items. Returns None.
    """Filter items by hardware classification and print a summary.

    For each item, `get_class` extracts the associated test class. Items whose
    hardware classification is in `requirement` are passed to `on_match`, if
    provided. Prints a summary of matched, mismatched, unclassified, and
    classless items.
    """
    total = 0
    passed = 0
    no_class = 0
    unclassified = 0
    for item in items:
        total += 1
        cls = get_class(item)
        if cls is None:
            no_class += 1
            continue
        classification = get_hw_classification(cls)
        if classification is None:
            unclassified += 1
        elif classification in requirement:
            passed += 1
            on_match(item)

    mismatched = total - passed - unclassified - no_class
    parts = [
        f"HW classification mode active ({[e.name for e in requirement]}):",
        f"total={total}, passed={passed},",
        f"unclassified={unclassified},",
    ]
    if no_class > 0:
        parts.append(f"no_class={no_class},")
    parts.append(f"mismatched={mismatched}")
    print(" ".join(parts))


def is_navi3_arch():
    # [FUNCTION DOC] Return True if CUDA is available and device 0 is an AMD Navi 3 GPU
    # (architecture name in NAVI3_ARCH), otherwise False. Queries device properties.
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        gfx_arch = prop.gcnArchName.split(":")[0]
        if gfx_arch in NAVI3_ARCH:
            return True
    return False

def freeze_rng_state(*args, **kwargs):
    # [FUNCTION DOC] Thin forwarding wrapper that calls the freeze_rng_state context manager
    # helper from torch.testing._utils with the same arguments and returns its result.
    return torch.testing._utils.freeze_rng_state(*args, **kwargs)


# Class to keep track of test flags configurable by environment variables.
# Flags set here are intended to be read-only and should not be modified after
# definition.
# TODO: Expand this class to handle arbitrary settings in addition to boolean flags?
class TestEnvironment:
    # [CLASS DOC] Registry of test flags and settings that are derived from environment
    # variables. Static methods define a flag or setting, publish it as a module-level global,
    # and remember which environment variables differ from their defaults so a reproduction
    # command prefix can be printed on test failure. State: the class attribute repro_env_vars
    # maps an environment variable name to the value it had when that value differs from the
    # default and is not implied. Flags are meant to be read-only after definition; defining the
    # same name twice raises.
    # Set of env vars to set for the repro command that is output on test failure.
    # Specifically, this includes env vars that are set to non-default values and
    # are not implied. Maps from env var name -> value (int)
    repro_env_vars: dict = {}

    # Defines a flag usable throughout the test suite, determining its value by querying
    # the specified environment variable.
    #
    # Args:
    #     name (str): The name of the flag. A global variable with this name will be set
    #         for convenient access throughout the test suite.
    #     env_var (str): The name of the primary environment variable from which to
    #         determine the value of this flag. If this is None or the environment variable
    #         is unset, the default value will be used unless otherwise implied (see
    #         implied_by_fn). Default: None
    #     default (bool): The default value to use for the flag if unset by the environment
    #         variable and unimplied. Default: False
    #     include_in_repro (bool): Indicates whether this flag should be included in the
    #         repro command that is output on test failure (i.e. whether it is possibly
    #         relevant to reproducing the test failure). Default: True
    #     enabled_fn (Callable): Callable returning whether the flag should be enabled
    #         given the environment variable value and the default value. Default: Lambda
    #         requiring "0" to disable if on by default OR "1" to enable if off by default.
    #     implied_by_fn (Callable): Thunk returning a bool to imply this flag as enabled
    #         by something outside of its primary environment variable setting. For example,
    #         this can be useful if the value of another environment variable implies the flag
    #         as enabled. If the primary env var is set explicitly (to any value, including
    #         "0"), the env var wins and implied_by_fn is not consulted. Default: Lambda
    #         returning False to indicate no implications.
    @staticmethod
    def def_flag(
        name,
        env_var=None,
        default=False,
        include_in_repro=True,
        enabled_fn=lambda env_var_val, default: (
            (env_var_val != "0") if default else (env_var_val == "1")),
        implied_by_fn=lambda: False,
    ):
        # [FUNCTION DOC] Define a boolean test flag and export it as a module-level global.
        # Reads the named environment variable (if any) and decides the value using enabled_fn
        # (by default "0" disables a default-on flag and "1" enables a default-off flag). If the
        # variable is unset, implied_by_fn may switch the flag on. When include_in_repro is true
        # and the flag differs from its default without being implied, the variable is recorded
        # in repro_env_vars. Returns the flag value. Raises AssertionError if a global with this
        # name already exists.
        enabled = default
        env_var_val = None
        if env_var is not None:
            env_var_val = os.getenv(env_var)
            enabled = enabled_fn(env_var_val, default)
        implied = False
        if env_var_val is None:
            implied = implied_by_fn()
            enabled = enabled or implied
        if include_in_repro and (env_var is not None) and (enabled != default) and not implied:
            TestEnvironment.repro_env_vars[env_var] = env_var_val

        # export flag globally for convenience
        if name in globals():
            raise AssertionError(f"duplicate definition of flag '{name}'")
        globals()[name] = enabled
        return enabled

    # Defines a setting usable throughout the test suite, determining its value by querying
    # the specified environment variable. This differs from a flag in that it's not restricted
    # to a boolean value.
    #
    # Args:
    #     name (str): The name of the setting. A global variable with this name will be set
    #         for convenient access throughout the test suite.
    #     env_var (str): The name of the primary environment variable from which to
    #         determine the value of this setting. If this is None or the environment variable
    #         is unset, the default value will be used. Default: None
    #     default (Any): The default value to use for the setting if unset by the environment
    #         variable. Default: None
    #     include_in_repro (bool): Indicates whether this setting should be included in the
    #         repro command that is output on test failure (i.e. whether it is possibly
    #         relevant to reproducing the test failure). Default: True
    #     parse_fn (Callable): Callable parsing the env var string. Default value just uses
    #         the string itself.
    @staticmethod
    def def_setting(
        name,
        env_var=None,
        default=None,
        include_in_repro=True,
        parse_fn=lambda maybe_val_str: maybe_val_str,
    ):
        # [FUNCTION DOC] Define a non-boolean test setting and export it as a module-level
        # global. The raw value is the default when env_var is None, otherwise the environment
        # variable value (possibly None); it is converted with parse_fn. If include_in_repro is
        # true and the parsed value differs from the default, it is recorded in repro_env_vars.
        # Returns the value. Raises AssertionError if the global name is already defined.
        # KNOWN ISSUE: when env_var is None and parse_fn changes the default, the value is
        # recorded in repro_env_vars under a None key.
        value = default if env_var is None else os.getenv(env_var)
        value = parse_fn(value)
        if include_in_repro and (value != default):
            TestEnvironment.repro_env_vars[env_var] = value

        # export setting globally for convenience
        if name in globals():
            raise AssertionError(f"duplicate definition of setting '{name}'")
        globals()[name] = value
        return value

    # Returns a string prefix usable to set environment variables for any test
    # settings that should be explicitly set to match this instantiation of the
    # test suite.
    # Example: "PYTORCH_TEST_WITH_ASAN=1 PYTORCH_TEST_WITH_ROCM=1"
    @staticmethod
    def repro_env_var_prefix() -> str:
        # [FUNCTION DOC] Build a string of NAME=VALUE pairs, separated by spaces, from
        # repro_env_vars. It can be prefixed to a command to reproduce the current test
        # configuration. Returns an empty string when no variables were recorded.
        return " ".join([f"{env_var}={value}"
                         for env_var, value in TestEnvironment.repro_env_vars.items()])


log = logging.getLogger(__name__)
torch.backends.disable_global_flags()

FILE_SCHEMA = "file://"
if sys.platform == 'win32':
    FILE_SCHEMA = "file:///"

# NB: This flag differs semantically from others in that setting the env var to any
# non-empty value will cause it to be true:
#   CI=1, CI="true", CI=0, etc. all set the flag to be true.
#   CI= and an unset CI set the flag to be false.
# GitHub sets the value to CI="true" to enable it.
IS_CI: bool = TestEnvironment.def_flag(
    "IS_CI",
    env_var="CI",
    include_in_repro=False,
    enabled_fn=lambda env_var_value, _: bool(env_var_value),
)
IS_SANDCASTLE: bool = TestEnvironment.def_flag(
    "IS_SANDCASTLE",
    env_var="SANDCASTLE",
    implied_by_fn=lambda: os.getenv("TW_JOB_USER") == "sandcastle",
    include_in_repro=False,
)
IN_RE_WORKER: bool = os.environ.get("INSIDE_RE_WORKER") is not None

_is_fbcode_default = (
    hasattr(torch._utils_internal, "IS_FBSOURCE") and
    torch._utils_internal.IS_FBSOURCE
)

IS_FBCODE: bool = TestEnvironment.def_flag(
    "IS_FBCODE",
    env_var="PYTORCH_TEST_FBCODE",
    default=_is_fbcode_default,
    include_in_repro=False,
)
IS_REMOTE_GPU: bool = TestEnvironment.def_flag(
    "IS_REMOTE_GPU",
    env_var="PYTORCH_TEST_REMOTE_GPU",
    include_in_repro=False,
)

DISABLE_RUNNING_SCRIPT_CHK: bool = TestEnvironment.def_flag(
    "DISABLE_RUNNING_SCRIPT_CHK",
    env_var="PYTORCH_DISABLE_RUNNING_SCRIPT_CHK",
    include_in_repro=False,
)
# NB: enabled by default unless in an fbcode context.
PRINT_REPRO_ON_FAILURE: bool = TestEnvironment.def_flag(
    "PRINT_REPRO_ON_FAILURE",
    env_var="PYTORCH_PRINT_REPRO_ON_FAILURE",
    default=(not IS_FBCODE),
    include_in_repro=False,
)

# possibly restrict OpInfo tests to a single sample input
OPINFO_SAMPLE_INPUT_INDEX: int | None = TestEnvironment.def_setting(
    "OPINFO_SAMPLE_INPUT_INDEX",
    env_var="PYTORCH_OPINFO_SAMPLE_INPUT_INDEX",
    default=None,
    # Don't include the env var value in the repro command because the info will
    # be queried from the tracked sample input instead
    include_in_repro=False,
    parse_fn=lambda val: None if val is None else int(val),
)

# Possibly restrict OpInfo tests to a single DSL runtime.
# Example inputs: "triton", "cutedsl", all possible values
# given by: torch.backends.python_native.all_dsls
OPINFO_RESTRICT_TO_DSL: str | None = TestEnvironment.def_setting(
    "OPINFO_RESTRICT_TO_DSL",
    env_var="OPINFO_RESTRICT_TO_DSL",
    default=None,
    # Don't include the env var value in the repro command because the info will
    # be queried from the tracked sample input instead
    include_in_repro=True,
    parse_fn=lambda val: None if val is None else str(val),
)


DEFAULT_DISABLED_TESTS_FILE = '.pytorch-disabled-tests.json'
DEFAULT_SLOW_TESTS_FILE = 'slow_tests.json'

disabled_tests_dict = {}
slow_tests_dict = {}


def resolve_sandcastle_slow_tests_file() -> str:
    # [FUNCTION DOC] HELPER: On Sandcastle only, return the path of the slow-tests JSON file
    # located through torch._utils_internal if it exists, otherwise an empty string. Returns an
    # empty string when not running on Sandcastle.
    if not IS_SANDCASTLE:
        return ""

    filename = torch._utils_internal.get_file_path("test", DEFAULT_SLOW_TESTS_FILE)
    return filename if os.path.isfile(filename) else ""


def maybe_load_json(filename):
    # [FUNCTION DOC] HELPER: Load and return the parsed JSON content of a file if it exists. If
    # the file does not exist, logs a warning and returns an empty mapping.
    if os.path.isfile(filename):
        with open(filename) as fp:
            return json.load(fp)
    log.warning("Attempted to load json file '%s' but it does not exist.", filename)
    return {}

# set them here in case the tests are running in a subprocess that doesn't call run_tests
slow_tests_file = os.getenv("SLOW_TESTS_FILE", "") or resolve_sandcastle_slow_tests_file()
if slow_tests_file:
    slow_tests_dict = maybe_load_json(slow_tests_file)
    os.environ["SLOW_TESTS_FILE"] = slow_tests_file
if os.getenv("DISABLED_TESTS_FILE", ""):
    disabled_tests_dict = maybe_load_json(os.getenv("DISABLED_TESTS_FILE", ""))

NATIVE_DEVICES = ('cpu', 'cuda', 'xpu', 'meta', 'mps', 'mtia', torch._C._get_privateuse1_backend_name())

# used for managing devices testing for torch profiler UTs
# for now cpu, cuda and xpu are added for testing torch profiler UTs
DEVICE_LIST_SUPPORT_PROFILING_TEST = ('cpu', 'cuda', 'xpu')
ALLOW_XPU_PROFILING_TEST = True

check_names = ['orin', 'concord', 'galen', 'xavier', 'nano', 'jetson', 'tegra', 'thor']
IS_JETSON = any(name in platform.platform() for name in check_names)

def gcIfJetson(fn):
    # [FUNCTION DOC] Decorator that wraps a function so that, on Jetson hosts, garbage
    # collection and a CUDA cache release run before each call. Arguments and the call are
    # forwarded to the wrapped function.
    # KNOWN ISSUE: the wrapper discards the wrapped function's return value.
    # Irregular Jetson host/device memory setup requires cleanup to avoid tests being killed
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper produced by gcIfJetson. On Jetson, collects garbage and
        # empties the CUDA cache, then calls the wrapped function and returns nothing.
        if IS_JETSON:
            gc.collect()
            torch.cuda.empty_cache()
        fn(*args, **kwargs)
    return wrapper

# Tries to extract the current test function by crawling the stack.
# If unsuccessful, return None.
def extract_test_fn() -> Callable | None:
    # [FUNCTION DOC] HELPER: Crawl the call stack looking for a frame whose self is a unittest
    # TestCase, and return the underlying test method function for it (found from the test id
    # and requiring a name starting with test). Returns None if nothing suitable is found or any
    # error occurs; all exceptions are swallowed.
    try:
        stack = inspect.stack()
        for frame_info in stack:
            frame = frame_info.frame
            if "self" not in frame.f_locals:
                continue
            self_val = frame.f_locals["self"]
            if isinstance(self_val, unittest.TestCase):
                test_id = self_val.id()
                *_, cls_name, test_name = test_id.rsplit('.', 2)
                if cls_name == type(self_val).__name__ and test_name.startswith("test"):
                    test_fn = getattr(self_val, test_name).__func__
                    return test_fn
    except Exception:
        pass
    return None

# Contains tracked input data useful for debugging purposes
@dataclass
class TrackedInput:
    # [CLASS DOC] HELPER: Simple record of the most recent input produced while iterating test
    # inputs: its index, the value, and a short description of the input kind. Stored on the
    # test function for debugging failures.
    index: int
    val: Any
    type_desc: str

# Attempt to pull out tracked input information from the test function.
# A TrackedInputIter is used to insert this information.
def get_tracked_input() -> TrackedInput | None:
    # [FUNCTION DOC] HELPER: Return the TrackedInput stored on the currently running test
    # function, or None if there is no test function or nothing was tracked.
    test_fn = extract_test_fn()
    if test_fn is None:
        return None
    return getattr(test_fn, "tracked_input", None)

def clear_tracked_input() -> None:
    # [FUNCTION DOC] HELPER: Reset the tracked input on the currently running test function to
    # None, if a test function is found and it has a tracking slot. Mutates that function.
    test_fn = extract_test_fn()
    if test_fn is None:
        return
    if not hasattr(test_fn, "tracked_input"):
        return
    test_fn.tracked_input = None  # type: ignore[attr-defined]

# Wraps an iterator and tracks the most recent value the iterator produces
# for debugging purposes. Tracked values are stored on the test function.
class TrackedInputIter:
    # [CLASS DOC] Iterator wrapper that yields items from a child iterable while recording the
    # latest produced value as the tracked input on the current test function. State: the
    # enumerated child iterator, a description of the input kind, optional item and track
    # callbacks, the test function found at construction, whether to reseed the RNG before each
    # item, and an optional single index to restrict iteration to.
    def __init__(
        self,
        child_iter,
        input_type_desc,
        item_callback=None,
        track_callback=None,
        set_seed=True,
        restrict_to_index=None
    ):
        # [FUNCTION DOC] Store the child iterable (enumerated) and configuration. item_callback
        # maps (value, index) to the returned item (default: the value itself); track_callback
        # maps a value to what gets tracked (default: the value). Looks up the test function.
        self.child_iter = enumerate(child_iter)
        # Input type describes the things we're tracking (e.g. "sample input", "error input").
        self.input_type_desc = input_type_desc
        # NB: The two types of callbacks below exist because the thing we want to track isn't
        # always the same as the thing we want returned from the iterator. An example of this
        # is ErrorInput, which we want returned from the iterator, but which contains a
        # SampleInput that we want to track.
        # Item callback is run on each (iterated thing, index) to get the thing to return.
        self.item_callback = item_callback
        if self.item_callback is None:
            self.item_callback = lambda x, i: x
        # Track callback is run on each iterated thing to get the thing to track.
        self.track_callback = track_callback
        if self.track_callback is None:
            self.track_callback = lambda x: x
        self.test_fn = extract_test_fn()
        # Indicates whether the random seed should be set before each call to the iterator
        self.set_seed = set_seed
        # Indicates that iteration should be restricted to only the provided index.
        # If None, no restriction is done
        self.restrict_to_index = restrict_to_index

    def __iter__(self):
        # [FUNCTION DOC] HELPER: Return the iterator itself.
        return self

    def __next__(self):
        # [FUNCTION DOC] Produce the next item. If set_seed is enabled, first reseed the RNG
        # using a hash of the test function qualified name (or the default SEED when there is no
        # test function). Skips items whose index differs from restrict_to_index when set.
        # Records the chosen item as the tracked input and returns the item_callback result.
        # StopIteration from the child iterator propagates.
        while True:
            if self.set_seed:
                # use a test-name-specific hash for the seed if possible
                seed = (
                    int.from_bytes(hashlib.sha256(
                        self.test_fn.__qualname__.encode("utf-8")).digest()[:4], 'little')
                    if self.test_fn is not None else SEED
                )
                set_rng_seed(seed)

            # allow StopIteration to bubble up
            input_idx, input_val = next(self.child_iter)
            if (self.restrict_to_index is None) or (input_idx == self.restrict_to_index):
                break

        self._set_tracked_input(
            TrackedInput(
                index=input_idx, val=self.track_callback(input_val), type_desc=self.input_type_desc
            )
        )
        return self.item_callback(input_val, input_idx)

    def _set_tracked_input(self, tracked_input: TrackedInput):
        # [FUNCTION DOC] HELPER: Store the given tracked input on the test function, but only if
        # a test function exists and already carries a tracking slot.
        if self.test_fn is None:
            return
        if not hasattr(self.test_fn, "tracked_input"):
            return
        self.test_fn.tracked_input = tracked_input  # type: ignore[attr-defined]

class _TestParametrizer:
    # [CLASS DOC] Base class for test parametrization decorators. Applying an instance to a
    # test function attaches a parametrize_fn attribute that later generates specialized
    # tests, either through instantiate_device_type_tests or instantiate_parametrized_tests.
    # When applied to a function that already has a parametrize_fn, the two are composed so
    # the generated tests are the product of both parametrizations. Subclasses must
    # implement _parametrize_test.
    """
    Decorator class for parametrizing a test function, yielding a set of new tests spawned
    from the original generic test, each specialized for a specific set of test inputs. For
    example, parametrizing a test across the set of ops will result in a test function per op.

    The decision of how to parametrize / what to parametrize over is intended to be implemented
    by each derived class.

    In the details, the decorator adds a 'parametrize_fn' property to the test function. This function
    is intended to be called later by one of:
      * Device-specific test instantiation via instantiate_device_type_tests(). Note that for this
        case there is no need to explicitly parametrize over device type, as that is handled separately.
      * Device-agnostic parametrized test instantiation via instantiate_parametrized_tests().

    If the decorator is applied to a test function that already has a 'parametrize_fn' property, a new
    composite 'parametrize_fn' will be created that generates tests with the product of the parameters
    generated by the old and new parametrize_fns. This allows for convenient composability of decorators.
    """
    def _parametrize_test(self, test, generic_cls, device_cls):
        # [FUNCTION DOC] Abstract hook: subclasses yield 4-tuples of (test function, name
        # suffix, parameter keyword mapping, decorator function that maps parameters to a list
        # of decorators) for the given test, generic class and device class (None if device
        # agnostic). Raises NotImplementedError in the base class.
        """
        Parametrizes the given test function across whatever dimension is specified by the derived class.
        Tests can be parametrized over any arbitrary dimension or combination of dimensions, such as all
        ops, all modules, or all ops + their associated dtypes.

        Args:
            test (fn): Test function to parametrize over
            generic_cls (class): Generic test class object containing tests (e.g. TestFoo)
            device_cls (class): Device-specialized test class object (e.g. TestFooCPU); set to None
                if the tests are not part of a device-specific set

        Returns:
            Generator object returning 4-tuples of:
                test (fn): Parametrized test function; must support a device arg and args for any params
                test_name (str): Parametrized suffix for the test (e.g. opname_int64); will be appended to
                    the base name of the test
                param_kwargs (dict): Param kwargs to pass to the test (e.g. {'op': 'add', 'dtype': torch.int64})
                decorator_fn (callable): Callable[[Dict], List] for list of decorators to apply given param_kwargs
        """
        raise NotImplementedError

    def __call__(self, fn):
        # [FUNCTION DOC] Attach a parametrize_fn to the test function and return the function.
        # If one is already present, replace it with a composition of the old and the new one.
        # Mutates the function object.
        if hasattr(fn, 'parametrize_fn'):
            # Do composition with the product of args.
            old_parametrize_fn = fn.parametrize_fn
            new_parametrize_fn = self._parametrize_test
            fn.parametrize_fn = compose_parametrize_fns(old_parametrize_fn, new_parametrize_fn)
        else:
            fn.parametrize_fn = self._parametrize_test
        return fn


def compose_parametrize_fns(old_parametrize_fn, new_parametrize_fn):
    # [FUNCTION DOC] Compose two parametrize functions into one that yields the product of their
    # parametrizations. Test names are joined as new name then old name with an underscore when
    # both are non-empty. Returns the composite generator function. The composite raises
    # RuntimeError if both functions parametrize the same parameter.
    """
    Returns a parametrize_fn that parametrizes over the product of the parameters handled
    by the given parametrize_fns. Each given parametrize_fn should each have the signature
    f(test, generic_cls, device_cls).

    The test names will be a combination of the names produced by the parametrize_fns in
    "<new_name>_<old_name>" order. This order is done to match intuition for constructed names
    when composing multiple decorators; the names will be built in top to bottom order when stacking
    parametrization decorators.

    Args:
        old_parametrize_fn (callable) - First parametrize_fn to compose.
        new_parametrize_fn (callable) - Second parametrize_fn to compose.
    """

    def composite_fn(test, generic_cls, device_cls,
                     old_parametrize_fn=old_parametrize_fn,
                     new_parametrize_fn=new_parametrize_fn):
        # [FUNCTION DOC] HELPER: Generator that expands every parametrization of the old
        # function and applies the new function to each result, yielding the merged test, merged
        # name, merged parameter mapping and a merged decorator function. Raises RuntimeError on
        # parameters handled by both.
        old_tests = list(old_parametrize_fn(test, generic_cls, device_cls))
        for (old_test, old_test_name, old_param_kwargs, old_dec_fn) in old_tests:
            for (new_test, new_test_name, new_param_kwargs, new_dec_fn) in \
                    new_parametrize_fn(old_test, generic_cls, device_cls):
                redundant_params = set(old_param_kwargs.keys()).intersection(new_param_kwargs.keys())
                if redundant_params:
                    raise RuntimeError('Parametrization over the same parameter by multiple parametrization '
                                       f'decorators is not supported. For test "{test.__name__}", the following parameters '
                                       f'are handled multiple times: {redundant_params}')
                full_param_kwargs = {**old_param_kwargs, **new_param_kwargs}
                merged_test_name = '{}{}{}'.format(new_test_name,
                                                   '_' if old_test_name != '' and new_test_name != '' else '',
                                                   old_test_name)

                def merged_decorator_fn(param_kwargs, old_dec_fn=old_dec_fn, new_dec_fn=new_dec_fn):
                    # [FUNCTION DOC] HELPER: Return the concatenation of the decorators produced
                    # by the old and new decorator functions for the given parameters.
                    return list(old_dec_fn(param_kwargs)) + list(new_dec_fn(param_kwargs))

                yield (new_test, merged_test_name, full_param_kwargs, merged_decorator_fn)

    return composite_fn


def validate_test_name(name):
    # [FUNCTION DOC] Check a generated test name and raise RuntimeError if it contains a dot,
    # because dotted names cannot be loaded by name through unittest. Returns None otherwise.
    """
    Validates a generated test name at instantiation time. In particular, test names must
    not contain '.': unittest.TestLoader.loadTestsFromName resolves dotted names by
    splitting on '.' and walking with getattr, so a dotted test name is discoverable in a
    full-suite run but breaks any attempt to load the test by name (e.g.
    `python test_foo.py TestClass.test_name`).
    """
    if '.' in name:
        raise RuntimeError(
            f'Test name "{name}" is invalid: it contains a "." character, which breaks '
            'loading the test by name (unittest.TestLoader.loadTestsFromName splits on '
            '"."). This name likely comes from a custom name_fn or an explicit subtest '
            'name that interpolates a value whose string form contains a dot (e.g. '
            'str(torch.bfloat16) == "torch.bfloat16" or a float). Use dtype_name() for '
            'dtypes, or sanitize the generated name (e.g. .replace(".", "_")).')


def instantiate_parametrized_tests(generic_cls):
    # [FUNCTION DOC] Replace each parametrized test on a device-agnostic test class by the
    # generated specialized tests. For every attribute with a parametrize_fn, the generic test
    # is deleted from the class and each parametrization is added under a name made of the test
    # name and suffix, after validating the name and applying the decorators chosen for the
    # parameters. Returns the class so it can be used as a class decorator. Raises
    # AssertionError if a generated name already exists on the class.
    """
    Instantiates tests that have been decorated with a parametrize_fn. This is generally performed by a
    decorator subclass of _TestParametrizer. The generic test will be replaced on the test class by
    parametrized tests with specialized names. This should be used instead of
    instantiate_device_type_tests() if the test class contains device-agnostic tests.

    You can also use it as a class decorator. E.g.

    ```
    @instantiate_parametrized_tests
    class TestFoo(TestCase):
        ...
    ```

    Args:
        generic_cls (class): Generic test class object containing tests (e.g. TestFoo)
    """
    for attr_name in tuple(dir(generic_cls)):
        class_attr = getattr(generic_cls, attr_name)
        if not hasattr(class_attr, 'parametrize_fn'):
            continue

        # Remove the generic test from the test class.
        delattr(generic_cls, attr_name)

        # Add parametrized tests to the test class.
        def instantiate_test_helper(cls, name, test, param_kwargs):
            # [FUNCTION DOC] HELPER: Create a test method that calls the original test with the
            # given parameter mapping and attach it to the class under the given name. Raises
            # AssertionError if the class already has an attribute with that name.
            @wraps(test)
            def instantiated_test(self, param_kwargs=param_kwargs):
                # [FUNCTION DOC] HELPER: Test body that forwards its parameters as keyword
                # arguments to the original test function.
                test(self, **param_kwargs)

            if hasattr(generic_cls, name):
                raise AssertionError(f"Redefinition of test {name}")
            setattr(generic_cls, name, instantiated_test)

        for (test, test_suffix, param_kwargs, decorator_fn) in class_attr.parametrize_fn(
                class_attr, generic_cls=generic_cls, device_cls=None):
            full_name = f'{test.__name__}_{test_suffix}'
            validate_test_name(full_name)

            # Apply decorators based on full param kwargs.
            for decorator in decorator_fn(param_kwargs):
                test = decorator(test)

            instantiate_test_helper(cls=generic_cls, name=full_name, test=test, param_kwargs=param_kwargs)
    return generic_cls


class subtest:
    # [CLASS DOC] Explicit subtest value for use with the parametrize decorator. Holds the
    # argument values, an optional explicit test name, and a list of decorators to apply to the
    # generated test. Uses slots; no behavior beyond storing these fields.
    """
    Explicit subtest case for use with test parametrization.
    Allows for explicit naming of individual subtest cases as well as applying
    decorators to the parametrized test.

    Args:
        arg_values (iterable): Iterable of arg values (e.g. range(10)) or
            tuples of arg values (e.g. [(1, 2), (3, 4)]).
        name (str): Optional name to use for the test.
        decorators (iterable): Iterable of decorators to apply to the generated test.
    """
    __slots__ = ['arg_values', 'name', 'decorators']

    def __init__(self, arg_values, name=None, decorators=None):
        # [FUNCTION DOC] Store the argument values and optional name; decorators default to an
        # empty list when not given.
        self.arg_values = arg_values
        self.name = name
        self.decorators = decorators if decorators else []


class parametrize(_TestParametrizer):
    # [CLASS DOC] Generic parametrization decorator modeled after the pytest parametrize
    # mark. Takes a comma-separated string of argument names, an iterable of values (or
    # subtest objects) and an optional name function. Each value set yields one generated
    # test whose name is derived from the argument names and values unless an explicit name
    # or name function is given. State: the list of argument names, the values iterable and
    # the name function.
    """
    Decorator for applying generic test parametrizations.

    The interface for this decorator is modeled after `@pytest.mark.parametrize`.
    Basic usage between this decorator and pytest's is identical. The first argument
    should be a string containing comma-separated names of parameters for the test, and
    the second argument should be an iterable returning values or tuples of values for
    the case of multiple parameters.

    Beyond this basic usage, the decorator provides some additional functionality that
    pytest does not.

    1. Parametrized tests end up as generated test functions on unittest test classes.
    Since this differs from how pytest works, this decorator takes on the additional
    responsibility of naming these test functions. The default test names consists of
    the test's base name followed by each parameter name + value (e.g. "test_bar_x_1_y_foo"),
    but custom names can be defined using `name_fn` or the `subtest` structure (see below).

    2. The decorator specially handles parameter values of type `subtest`, which allows for
    more fine-grained control over both test naming and test execution. In particular, it can
    be used to tag subtests with explicit test names or apply arbitrary decorators (see examples
    below).

    Examples::

        @parametrize("x", range(5))
        def test_foo(self, x):
            ...

        @parametrize("x,y", [(1, 'foo'), (2, 'bar'), (3, 'baz')])
        def test_bar(self, x, y):
            ...

        @parametrize("x,y", [(1, 'foo'), (2, 'bar'), (3, 'baz')],
                     name_fn=lambda x, y: '{}_{}'.format(x, y))
        def test_bar_custom_names(self, x, y):
            ...

        @parametrize("x, y", [subtest((1, 2), name='double'),
                              subtest((1, 3), name='triple', decorators=[unittest.expectedFailure]),
                              subtest((1, 4), name='quadruple')])
        def test_baz(self, x, y):
            ...

    To actually instantiate the parametrized tests, one of instantiate_parametrized_tests() or
    instantiate_device_type_tests() should be called. The former is intended for test classes
    that contain device-agnostic tests, while the latter should be used for test classes that
    contain device-specific tests. Both support arbitrary parametrizations using the decorator.

    Args:
        arg_str (str): String of arg names separate by commas (e.g. "x,y").
        arg_values (iterable): Iterable of arg values (e.g. range(10)) or
            tuples of arg values (e.g. [(1, 2), (3, 4)]).
        name_fn (Callable): Optional function that takes in parameters and returns subtest name.
    """
    def __init__(self, arg_str, arg_values, name_fn=None):
        # [FUNCTION DOC] Split the argument string on commas into stripped argument names and
        # store them along with the values iterable and the optional name function.
        self.arg_names: list[str] = [s.strip() for s in arg_str.split(',') if s != '']
        self.arg_values = arg_values
        self.name_fn = name_fn

    def _formatted_str_repr(self, idx, name, value):
        # [FUNCTION DOC] HELPER: Produce the piece of a test name for one argument. Dtypes and
        # OpInfo/ModuleInfo objects use their short names, devices use their string form, ints,
        # floats and strings use name plus value with dots replaced by underscores, and any
        # other value uses the name followed by the index.
        """ Returns a string representation for the given arg that is suitable for use in test function names. """
        if isinstance(value, torch.dtype):
            return dtype_name(value)
        elif isinstance(value, torch.device):
            return str(value)
        # Can't use isinstance as it would cause a circular import
        elif type(value).__name__ in {'OpInfo', 'ModuleInfo'}:
            return value.formatted_name
        elif isinstance(value, (int, float, str)):
            return f"{name}_{str(value).replace('.', '_')}"
        else:
            return f"{name}{idx}"

    def _default_subtest_name(self, idx, values):
        # [FUNCTION DOC] HELPER: Join the per-argument name pieces for a value set with
        # underscores to make the default subtest name.
        return '_'.join([self._formatted_str_repr(idx, a, v) for a, v in zip(self.arg_names, values, strict=True)])

    def _get_subtest_name(self, idx, values, explicit_name=None):
        # [FUNCTION DOC] HELPER: Choose the subtest name: the explicit name if given, else the
        # result of calling the name function with the values, else the default name.
        if explicit_name:
            subtest_name = explicit_name
        elif self.name_fn:
            subtest_name = self.name_fn(*values)
        else:
            subtest_name = self._default_subtest_name(idx, values)
        return subtest_name

    def _parametrize_test(self, test, generic_cls, device_cls):
        # [FUNCTION DOC] Generator that yields the parametrizations for a test. With no argument
        # names it yields the test once with an empty name. Otherwise, for each value set it
        # unwraps subtest objects (wrapping the test so decorators apply separately), checks
        # that the number of values matches the number of names, and yields the test, its name,
        # the parameter mapping and a decorator function. Raises RuntimeError on a count
        # mismatch and ValueError if the values iterable is empty.
        if len(self.arg_names) == 0:
            # No additional parameters needed for the test.
            test_name = ''
            yield (test, test_name, {}, lambda _: [])
        else:
            # Each "values" item is expected to be either:
            # * A tuple of values with one for each arg. For a single arg, a single item is expected.
            # * A subtest instance with arg_values matching the previous.
            values = check_exhausted_iterator = object()
            for idx, values in enumerate(self.arg_values):
                maybe_name = None

                decorators: list[Any] = []
                if isinstance(values, subtest):
                    sub = values
                    values = sub.arg_values
                    maybe_name = sub.name

                    @wraps(test)
                    def test_wrapper(*args, **kwargs):
                        # [FUNCTION DOC] HELPER: Wrapper that calls the original test, so that
                        # per-subtest decorators apply to a distinct function object.
                        return test(*args, **kwargs)

                    decorators = sub.decorators
                    gen_test = test_wrapper
                else:
                    gen_test = test

                values = list(values) if len(self.arg_names) > 1 else [values]  # type: ignore[call-overload]
                if len(values) != len(self.arg_names):
                    raise RuntimeError(f'Expected # values == # arg names, but got: {len(values)} '
                                       f'values and {len(self.arg_names)} names for test "{test.__name__}"')

                param_kwargs = dict(zip(self.arg_names, values, strict=True))

                test_name = self._get_subtest_name(idx, values, explicit_name=maybe_name)

                def decorator_fn(_, decorators=decorators):
                    # [FUNCTION DOC] HELPER: Return the pre-bound list of decorators regardless
                    # of parameters.
                    return decorators

                yield (gen_test, test_name, param_kwargs, decorator_fn)

            if values is check_exhausted_iterator:
                raise ValueError(f'{test}: An empty arg_values was passed to @parametrize. '
                                 'Note that this may result from reuse of a generator.')


class reparametrize(_TestParametrizer):
    # [CLASS DOC] Decorator that adapts another parametrizer. For every parametrization the
    # wrapped parametrizer produces, an adapter function receives the test name and the
    # parameter mapping and returns None (drop it) or an iterable of replacement (name,
    # parameter mapping) pairs. State: the wrapped parametrizer and the adapter.
    """
    Decorator for adjusting the way an existing parametrizer operates. This class runs
    the given adapter_fn on each parametrization produced by the given parametrizer,
    allowing for on-the-fly parametrization more flexible than the default,
    product-based composition that occurs when stacking parametrization decorators.

    If the adapter_fn returns None for a given test parametrization, that parametrization
    will be excluded. Otherwise, it's expected that the adapter_fn returns an iterable of
    modified parametrizations, with tweaked test names and parameter kwargs.

    Examples::

        def include_is_even_arg(test_name, param_kwargs):
            x = param_kwargs["x"]
            is_even = x % 2 == 0
            new_param_kwargs = dict(param_kwargs)
            new_param_kwargs["is_even"] = is_even
            is_even_suffix = "_even" if is_even else "_odd"
            new_test_name = f"{test_name}{is_even_suffix}"
            yield (new_test_name, new_param_kwargs)

        ...

        @reparametrize(parametrize("x", range(5)), include_is_even_arg)
        def test_foo(self, x, is_even):
            ...

        def exclude_odds(test_name, param_kwargs):
            x = param_kwargs["x"]
            is_even = x % 2 == 0
            yield None if not is_even else (test_name, param_kwargs)

        ...

        @reparametrize(parametrize("x", range(5)), exclude_odds)
        def test_bar(self, x):
            ...

    """
    def __init__(self, parametrizer, adapter_fn):
        # [FUNCTION DOC] Store the wrapped parametrizer and the adapter function.
        self.parametrizer = parametrizer
        self.adapter_fn = adapter_fn

    def _parametrize_test(self, test, generic_cls, device_cls):
        # [FUNCTION DOC] Generator that runs the wrapped parametrizer and, for each result,
        # calls the adapter. Items the adapter drops are skipped; for the rest it yields the
        # test with the new name and parameters and the original decorator function.
        for (gen_test, test_name, param_kwargs, decorator_fn) in \
                self.parametrizer._parametrize_test(test, generic_cls, device_cls):
            adapted = self.adapter_fn(test_name, param_kwargs)
            if adapted is not None:
                for adapted_item in adapted:
                    if adapted_item is not None:
                        new_test_name, new_param_kwargs = adapted_item
                        yield (gen_test, new_test_name, new_param_kwargs, decorator_fn)


class decorateIf(_TestParametrizer):
    # [CLASS DOC] Parametrizer that conditionally applies a decorator based on the final
    # parameters of each generated test, and composes with other parametrizers. State: the
    # decorator and a predicate taking the parameter mapping and returning a boolean.
    """
    Decorator for applying parameter-specific conditional decoration.
    Composes with other test parametrizers (e.g. @modules, @ops, @parametrize, etc.).

    Examples::

        @decorateIf(unittest.skip, lambda params: params["x"] == 2)
        @parametrize("x", range(5))
        def test_foo(self, x):
            ...

        @parametrize("x,y", [(1, 'foo'), (2, 'bar'), (3, 'baz')])
        @decorateIf(
            unittest.expectedFailure,
            lambda params: params["x"] == 3 and params["y"] == "baz"
        )
        def test_bar(self, x, y):
            ...

        @decorateIf(
            unittest.expectedFailure,
            lambda params: params["op"].name == "add" and params["dtype"] == torch.float16
        )
        @ops(op_db)
        def test_op_foo(self, device, dtype, op):
            ...

        @decorateIf(
            unittest.skip,
            lambda params: params["module_info"].module_cls is torch.nn.Linear and \
                params["device"] == "cpu"
        )
        @modules(module_db)
        def test_module_foo(self, device, dtype, module_info):
            ...

    Args:
        decorator: Test decorator to apply if the predicate is satisfied.
        predicate_fn (Callable): Function taking in a dict of params and returning a boolean
            indicating whether the decorator should be applied or not.
    """
    def __init__(self, decorator, predicate_fn):
        # [FUNCTION DOC] Store the decorator and predicate function.
        self.decorator = decorator
        self.predicate_fn = predicate_fn

    def _parametrize_test(self, test, generic_cls, device_cls):
        # [FUNCTION DOC] Generator that yields a single unmodified (wrapped) test with an empty
        # name and empty parameters, together with a decorator function that applies the stored
        # decorator only when the predicate accepts the parameters.

        # Leave test as-is and return the appropriate decorator_fn.
        def decorator_fn(params, decorator=self.decorator, predicate_fn=self.predicate_fn):
            # [FUNCTION DOC] HELPER: Return a list holding the decorator if the predicate
            # accepts the parameters, otherwise an empty list.
            if predicate_fn(params):
                return [decorator]
            else:
                return []

        @wraps(test)
        def test_wrapper(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper that calls the original test.
            return test(*args, **kwargs)

        test_name = ''
        yield (test_wrapper, test_name, {}, decorator_fn)


def cppProfilingFlagsToProfilingMode():
    # [FUNCTION DOC] Determine the JIT profiling mode implied by the current C++ settings.
    # Temporarily enables the profiling executor and graph optimization to read the previous
    # values, restores them, and returns PROFILING, SIMPLE or LEGACY accordingly. Side effect:
    # calls the C++ setters on the JIT executor flags.
    # KNOWN ISSUE: the graph optimize call used for restoring is a getter-named binding that is
    # used as a setter; the restore of that flag relies on it acting as a setter.
    old_prof_exec_state = torch._C._jit_set_profiling_executor(True)
    old_prof_mode_state = torch._C._get_graph_executor_optimize(True)
    torch._C._jit_set_profiling_executor(old_prof_exec_state)
    torch._C._get_graph_executor_optimize(old_prof_mode_state)

    if old_prof_exec_state:
        if old_prof_mode_state:
            return ProfilingMode.PROFILING
        else:
            return ProfilingMode.SIMPLE
    else:
        return ProfilingMode.LEGACY

@contextmanager
def enable_profiling_mode_for_profiling_tests():
    # [FUNCTION DOC] Context manager that, when the global graph executor is PROFILING, enables
    # the profiling executor and optimization for the body and restores the previous values on
    # exit. Does nothing otherwise. Raises AssertionError if GRAPH_EXECUTOR is unset.
    old_prof_exec_state = False
    old_prof_mode_state = False
    if not GRAPH_EXECUTOR:
        raise AssertionError("GRAPH_EXECUTOR must be set")
    if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
        old_prof_exec_state = torch._C._jit_set_profiling_executor(True)
        old_prof_mode_state = torch._C._get_graph_executor_optimize(True)
    try:
        yield
    finally:
        if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
            torch._C._jit_set_profiling_executor(old_prof_exec_state)
            torch._C._get_graph_executor_optimize(old_prof_mode_state)

@contextmanager
def enable_profiling_mode():
    # [FUNCTION DOC] Context manager that enables the JIT profiling executor and graph
    # optimization for the body and restores the previous settings on exit.
    old_prof_exec_state = torch._C._jit_set_profiling_executor(True)
    old_prof_mode_state = torch._C._get_graph_executor_optimize(True)
    try:
        yield
    finally:
        torch._C._jit_set_profiling_executor(old_prof_exec_state)
        torch._C._get_graph_executor_optimize(old_prof_mode_state)

@contextmanager
def num_profiled_runs(num_runs):
    # [FUNCTION DOC] Context manager that sets the number of profiled runs of the JIT for the
    # body and restores the previous number on exit.
    old_num_runs = torch._C._jit_set_num_profiled_runs(num_runs)
    try:
        yield
    finally:
        torch._C._jit_set_num_profiled_runs(old_num_runs)

func_call = torch._C.ScriptFunction.__call__
meth_call = torch._C.ScriptMethod.__call__

def prof_callable(callable, *args, **kwargs):
    # [FUNCTION DOC] HELPER: Call a TorchScript callable. If the keyword profile_and_replay is
    # given it is removed, and when the graph executor is PROFILING the callable is run twice
    # under profiling mode (the first run profiles, the second returns the result). Otherwise
    # the callable is called once. Returns its result. Raises AssertionError if the keyword is
    # given and GRAPH_EXECUTOR is unset.
    if 'profile_and_replay' in kwargs:
        del kwargs['profile_and_replay']
        if not GRAPH_EXECUTOR:
            raise AssertionError("GRAPH_EXECUTOR must be set")
        if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
            with enable_profiling_mode_for_profiling_tests():
                callable(*args, **kwargs)
                return callable(*args, **kwargs)

    return callable(*args, **kwargs)

def raise_on_run_directly(file_to_call):
    # [FUNCTION DOC] Always raises RuntimeError telling the user that the given test file must
    # not be run directly and showing the correct invocation with a test name.
    raise RuntimeError("This test file is not meant to be run directly, "
                       f"use:\n\n\tpython {file_to_call} TESTNAME\n\n"
                       "instead.")

def prof_func_call(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Replacement for the call method of script functions; forwards to
    # prof_callable with the original call.
    return prof_callable(func_call, *args, **kwargs)

def prof_meth_call(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Replacement for the call method of script methods; forwards to
    # prof_callable with the original call.
    return prof_callable(meth_call, *args, **kwargs)

torch._C.ScriptFunction.__call__ = prof_func_call  # type: ignore[method-assign]
torch._C.ScriptMethod.__call__ = prof_meth_call  # type: ignore[method-assign]

def _get_test_report_path():
    # [FUNCTION DOC] HELPER: Return the relative report directory: test-reports joined with the
    # value of the TEST_REPORT_SOURCE_OVERRIDE environment variable, or python-unittest when it
    # is unset.
    # allow users to override the test file location. We need this
    # because the distributed tests run the same test file multiple
    # times with different configurations.
    override = os.environ.get('TEST_REPORT_SOURCE_OVERRIDE')
    test_source = override if override is not None else 'python-unittest'
    return os.path.join('test-reports', test_source)

def parse_cmd_line_args():
    # [FUNCTION DOC] Parse the test command line (subprocess, accept, jit executor, repeat,
    # bailouts, pytest, XML output, discover, log suffix, parallel, slow/disabled test files,
    # rerun disabled, single pytest test, showlocals, hw classification) and publish the results
    # into the module-level globals. The unrecognized arguments are kept in UNITTEST_ARGS after
    # the program name. If help is requested it also runs unittest's help in a thread. Sets the
    # graph executor mode, the expecttest accept flag, and seeds the RNG through set_rng_seed.
    # Returns None.
    # KNOWN ISSUE: the repeat option defaults to 1, while REPEAT_COUNT starts at 0.
    global DISABLED_TESTS_FILE
    global GRAPH_EXECUTOR
    global LOG_SUFFIX
    global PYTEST_SINGLE_TEST
    global REPEAT_COUNT
    global RERUN_DISABLED_TESTS
    global RUN_PARALLEL
    global SHOWLOCALS
    global SLOW_TESTS_FILE
    global TEST_BAILOUTS
    global TEST_DISCOVER
    global TEST_IN_SUBPROCESS
    global TEST_SAVE_XML
    global UNITTEST_ARGS
    global USE_PYTEST
    global HW_CLASSIFICATION

    is_running_via_run_test = "run_test.py" in getattr(__main__, "__file__", "")
    parser = argparse.ArgumentParser(add_help=not is_running_via_run_test, allow_abbrev=False)
    parser.add_argument('--subprocess', action='store_true',
                        help='whether to run each test in a subprocess')
    parser.add_argument('--accept', action='store_true')
    parser.add_argument('--jit-executor', '--jit_executor', type=str)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--test-bailouts', '--test_bailouts', action='store_true')
    parser.add_argument('--use-pytest', action='store_true')
    parser.add_argument('--save-xml', nargs='?', type=str,
                        const=_get_test_report_path(),
                        default=_get_test_report_path() if IS_CI else None)
    parser.add_argument('--discover-tests', action='store_true')
    parser.add_argument('--log-suffix', type=str, default="")
    parser.add_argument('--run-parallel', type=int, default=1)
    parser.add_argument('--import-slow-tests', type=str, nargs='?', const=DEFAULT_SLOW_TESTS_FILE)
    parser.add_argument('--import-disabled-tests', type=str, nargs='?', const=DEFAULT_DISABLED_TESTS_FILE)
    parser.add_argument('--rerun-disabled-tests', action='store_true')
    parser.add_argument('--pytest-single-test', type=str, nargs=1)
    parser.add_argument('--showlocals', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--hw-classification', nargs='+', choices=[e.name for e in HardwareClassification], type=str.upper, default=None)

# Only run when -h or --help flag is active to display both unittest and parser help messages.
    def run_unittest_help(argv):
        # [FUNCTION DOC] HELPER: Run unittest main with the given argument list (used to print
        # unittest help text; unittest exits the process afterwards).
        unittest.main(argv=argv)

    if '-h' in sys.argv or '--help' in sys.argv:
        help_thread = threading.Thread(target=run_unittest_help, args=(sys.argv,))
        help_thread.start()
        help_thread.join()

    args, remaining = parser.parse_known_args()
    if args.jit_executor == 'legacy':
        GRAPH_EXECUTOR = ProfilingMode.LEGACY
    elif args.jit_executor == 'profiling':
        GRAPH_EXECUTOR = ProfilingMode.PROFILING
    elif args.jit_executor == 'simple':
        GRAPH_EXECUTOR = ProfilingMode.SIMPLE
    else:
        # infer flags based on the default settings
        GRAPH_EXECUTOR = cppProfilingFlagsToProfilingMode()

    RERUN_DISABLED_TESTS = args.rerun_disabled_tests

    SLOW_TESTS_FILE = args.import_slow_tests
    DISABLED_TESTS_FILE = args.import_disabled_tests
    LOG_SUFFIX = args.log_suffix
    RUN_PARALLEL = args.run_parallel
    TEST_BAILOUTS = args.test_bailouts
    USE_PYTEST = args.use_pytest
    PYTEST_SINGLE_TEST = args.pytest_single_test
    TEST_DISCOVER = args.discover_tests
    TEST_IN_SUBPROCESS = args.subprocess
    TEST_SAVE_XML = args.save_xml
    REPEAT_COUNT = args.repeat
    SHOWLOCALS = args.showlocals
    HW_CLASSIFICATION = (
        {HardwareClassification[name] for name in args.hw_classification}
        if args.hw_classification is not None
        else None
    )
    if not getattr(expecttest, "ACCEPT", False):
        expecttest.ACCEPT = args.accept
    UNITTEST_ARGS = [sys.argv[0]] + remaining

    set_rng_seed()


def wait_for_process(p, timeout=None):
    # [FUNCTION DOC] Wait for a subprocess and return its exit status. On KeyboardInterrupt it
    # waits up to five seconds for the child and kills it if still running, then re-raises. On
    # timeout it sends SIGINT, waits five more seconds, and kills the child if needed, then
    # re-raises the timeout. Any other exception kills the child and is re-raised. A final wait
    # always runs.
    # KNOWN ISSUE: in the keyboard interrupt branch the wait with a timeout raises
    # TimeoutExpired rather than returning None, so the kill branch is rarely reached.
    try:
        return p.wait(timeout=timeout)
    except KeyboardInterrupt:
        # Give `p` a chance to handle KeyboardInterrupt. Without this,
        # `pytest` can't print errors it collected so far upon KeyboardInterrupt.
        exit_status = p.wait(timeout=5)
        if exit_status is not None:
            return exit_status
        else:
            p.kill()
            raise
    except subprocess.TimeoutExpired:
        # send SIGINT to give pytest a chance to make xml
        p.send_signal(signal.SIGINT)
        exit_status = None
        try:
            exit_status = p.wait(timeout=5)
        # try to handle the case where p.wait(timeout=5) times out as well as
        # otherwise the wait() call in the finally block can potentially hang
        except subprocess.TimeoutExpired:
            pass
        if exit_status is not None:
            return exit_status
        else:
            p.kill()
        raise
    except:
        p.kill()
        raise
    finally:
        # Always call p.wait() to ensure exit
        p.wait()

def shell(command, cwd=None, env=None, stdout=None, stderr=None, timeout=None):
    # [FUNCTION DOC] Start a subprocess for the given command and wait for it through
    # wait_for_process. Flushes stdout and stderr first. Returns the exit status. Raises
    # AssertionError if the command is a string rather than a list or tuple.
    sys.stdout.flush()
    sys.stderr.flush()
    # The following cool snippet is copied from Py3 core library subprocess.call
    # only the with
    #   1. `except KeyboardInterrupt` block added for SIGINT handling.
    #   2. In Py2, subprocess.Popen doesn't return a context manager, so we do
    #      `p.wait()` in a `final` block for the code to be portable.
    #
    # https://github.com/python/cpython/blob/71b6c1af727fbe13525fb734568057d78cea33f3/Lib/subprocess.py#L309-L323
    if isinstance(command, str):
        raise AssertionError("Command to shell should be a list or tuple of tokens")
    p = subprocess.Popen(command, universal_newlines=True, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
    return wait_for_process(p, timeout=timeout)


def retry_shell(
    command,
    cwd=None,
    env=None,
    stdout=None,
    stderr=None,
    timeout=None,
    retries=1,
    was_rerun=False,
    label="",
) -> tuple[int, bool]:
    # [FUNCTION DOC] Run a command via shell and return a pair (exit code, whether it was
    # rerun). On a nonzero exit or timeout it retries recursively while retries remain, printing
    # a message to stdout. When retries are exhausted on timeout it prints a message with the
    # label and returns exit code 124. Raises AssertionError if retries is negative.
    # Returns exicode + whether it was rerun
    if not (retries >= 0):
        raise AssertionError(
            f"Expecting non negative number for number of retries, got {retries}"
        )
    try:
        exit_code = shell(
            command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, timeout=timeout
        )
        if exit_code == 0 or retries == 0:
            return exit_code, was_rerun
        print(
            f"Got exit code {exit_code}, retrying (retries left={retries})",
            file=stdout,
            flush=True,
        )
    except subprocess.TimeoutExpired:
        if retries == 0:
            # NB: the "Command took >Nmin, returning 124" prefix is load
            # bearing -- the CI log classifier matches on it, so only ever
            # append here. The label names what timed out, so that timeouts
            # group per test file on HUD instead of collapsing into a single
            # fleet-wide bucket (which makes an uptick in one file invisible).
            print(
                f"Command took >{timeout // 60}min, returning 124"
                + (f" ({label})" if label else ""),
                file=stdout,
                flush=True,
            )
            return 124, was_rerun
        print(
            f"Command took >{timeout // 60}min, retrying (retries left={retries})",
            file=stdout,
            flush=True,
        )
    return retry_shell(
        command,
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=stderr,
        timeout=timeout,
        retries=retries - 1,
        was_rerun=True,
        label=label,
    )


def discover_test_cases_recursively(suite_or_case):
    # [FUNCTION DOC] Flatten a unittest suite into a list of its individual TestCase objects by
    # recursion. A TestCase argument returns a one-element list.
    if isinstance(suite_or_case, unittest.TestCase):
        return [suite_or_case]
    rc = []
    for element in suite_or_case:
        rc.extend(discover_test_cases_recursively(element))
    return rc

def get_test_names(test_cases):
    # [FUNCTION DOC] Return, for each test case, a string made of the last two dotted parts of
    # its id (class name and test method name).
    return ['.'.join(case.id().split('.')[-2:]) for case in test_cases]

def _print_test_names():
    # [FUNCTION DOC] HELPER: Load the tests of the main module and print each test name (class
    # and method) on its own line. Used by the discover-tests mode.
    suite = unittest.TestLoader().loadTestsFromModule(__main__)
    test_cases = discover_test_cases_recursively(suite)
    for name in get_test_names(test_cases):
        print(name)

def chunk_list(lst, nchunks):
    # [FUNCTION DOC] Split a list into nchunks interleaved sublists (round robin) and return
    # them as a list of lists.
    return [lst[i::nchunks] for i in range(nchunks)]

# sanitize filename e.g., distributed/pipeline/sync/skip/test_api.py -> distributed.pipeline.sync.skip.test_api
def sanitize_test_filename(filename):
    # [FUNCTION DOC] HELPER: Convert a test file path into a dotted name by removing a trailing
    # .py and replacing slashes with dots.
    strip_py = re.sub(r'.py$', '', filename)
    return re.sub('/', r'.', strip_py)

def lint_test_case_extension(suite):
    # [FUNCTION DOC] Check that every test class in a suite extends the TestCase of this module.
    # Prints a failure message naming each offending class and returns False if any was found,
    # otherwise True. Recurses into nested suites.
    # KNOWN ISSUE: when a nested suite is found as the first test, the function returns
    # immediately after recursing, so later siblings are not checked.
    succeed = True
    for test_case_or_suite in suite:
        test_case = test_case_or_suite
        if isinstance(test_case_or_suite, unittest.TestSuite):
            first_test = test_case_or_suite._tests[0] if len(test_case_or_suite._tests) > 0 else None
            if first_test is not None and isinstance(first_test, unittest.TestSuite):
                return succeed and lint_test_case_extension(test_case_or_suite)
            test_case = first_test

        if test_case is not None:
            if not isinstance(test_case, TestCase):
                test_class = test_case.id().split('.', 1)[1].split('.')[0]
                err = "This test class should extend from torch.testing._internal.common_utils.TestCase but it doesn't."
                print(f"{test_class} - failed. {err}")
                succeed = False
    return succeed


def get_report_path(argv=None, pytest=False):
    # [FUNCTION DOC] Compute and create the directory for XML test reports using the save
    # location, log suffix and sanitized test file name (argv defaults to UNITTEST_ARGS). For
    # pytest it switches the directory name to python-pytest and returns a unique XML file path
    # inside it. Creates directories as a side effect and returns the path.
    if argv is None:
        argv = UNITTEST_ARGS
    test_filename = sanitize_test_filename(argv[0])
    test_report_path = TEST_SAVE_XML + LOG_SUFFIX
    test_report_path = os.path.join(test_report_path, test_filename)
    if pytest:
        test_report_path = test_report_path.replace('python-unittest', 'python-pytest')
        os.makedirs(test_report_path, exist_ok=True)
        test_report_path = os.path.join(test_report_path, f"{test_filename}-{os.urandom(8).hex()}.xml")
        return test_report_path
    os.makedirs(test_report_path, exist_ok=True)
    return test_report_path


def sanitize_pytest_xml(xml_file: str):
    # [FUNCTION DOC] Rewrite a pytest XML report in place so test case entries look like
    # unittest ones: split the class name into a file attribute (with slashes and .py) and a
    # short class name. Test cases with no class name or an unmatched one are left alone.
    # pytext xml is different from unittext xml, this function makes pytest xml more similar to unittest xml
    # consider somehow modifying the XML logger in conftest to do this instead
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_file)
    for testcase in tree.iter('testcase'):
        full_classname = testcase.attrib.get("classname")
        if full_classname is None:
            continue
        # The test prefix is optional
        regex_result = re.search(r"^(test\.)?(?P<file>.*)\.(?P<classname>[^\.]*)$", full_classname)
        if regex_result is None:
            continue
        classname = regex_result.group("classname")
        file = regex_result.group("file").replace(".", "/")
        testcase.set("classname", classname)
        testcase.set("file", f"{file}.py")
    tree.write(xml_file)


def get_pytest_test_cases(argv: list[str]) -> list[str]:
    # [FUNCTION DOC] Collect pytest test node ids without running tests. Runs pytest in
    # collect-only mode with the given argument list (dropping the -vv flag) and a plugin that
    # records each item's id relative to the working directory. Returns the list of ids.
    class TestCollectorPlugin:
        # [CLASS DOC] HELPER: Pytest plugin that records the node ids of collected tests in
        # a list.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Initialize an empty list of collected test ids.
            self.tests: list[Any] = []

        def pytest_collection_finish(self, session):
            # [FUNCTION DOC] HELPER: Pytest hook called after collection; appends the working
            # directory relative node id of every collected item to the list.
            for item in session.items:
                self.tests.append(session.config.cwd_relative_nodeid(item.nodeid))

    test_collector_plugin = TestCollectorPlugin()
    import pytest
    pytest.main(
        [arg for arg in argv if arg != '-vv'] + ['--collect-only', '-qq', '--use-main-module'],
        plugins=[test_collector_plugin]
    )
    return test_collector_plugin.tests


class HardwareClassificationTestLoader(unittest.TestLoader):
    # [CLASS DOC] Test loader that filters the loaded tests by hardware classification. State:
    # the set of requested classifications (None disables filtering). Loading from a module or a
    # name returns a suite containing only tests whose class matches.
    """Unittest TestLoader that filters loaded tests by hw_classification."""
    def __init__(self, hw_classification):
        # [FUNCTION DOC] Initialize the loader and store the requested hardware classifications.
        super().__init__()
        self.hw_classification = hw_classification

    @staticmethod
    def iter_test_cases_recursively(
        suite_or_case: unittest.TestSuite | unittest.TestCase,
    ) -> Iterator[unittest.TestCase]:
        # [FUNCTION DOC] Static generator that yields every individual test case from a suite or
        # case by recursion.
        if isinstance(suite_or_case, unittest.TestCase):
            yield suite_or_case
            return

        _iter = HardwareClassificationTestLoader.iter_test_cases_recursively
        for element in suite_or_case:
            yield from _iter(element)

    def get_filtered_suite(self, tests: unittest.TestSuite) -> unittest.TestSuite:
        # [FUNCTION DOC] Return the given suite unchanged if no classification filter is set,
        # otherwise a new suite with the test cases whose classes match the filter (a summary
        # line is printed by the filtering helper).
        if self.hw_classification is None:
            return tests

        filtered_suite = unittest.TestSuite()
        _iter = HardwareClassificationTestLoader.iter_test_cases_recursively
        filter_by_hw_classification(
            _iter(tests),
            self.hw_classification,
            get_class=lambda tc: tc.__class__,
            on_match=filtered_suite.addTest,
        )
        return filtered_suite

    def loadTestsFromModule(self, module, *args, pattern=None, **kwargs):
        # [FUNCTION DOC] Load the tests of a module using the base loader and return them after
        # applying the hardware classification filter.
        suite = super().loadTestsFromModule(
            module, *args, pattern=pattern, **kwargs
        )
        return self.get_filtered_suite(suite)

    def loadTestsFromName(self, name, module=None, *args, **kwargs):
        # [FUNCTION DOC] Load tests by name with the base loader and return them filtered by
        # hardware classification. A suite that begins with a failed-load placeholder is
        # returned unfiltered so the load error is still reported.
        suite = super().loadTestsFromName(name, module, *args, **kwargs)
        # _FailedTest has no hw_classification attribute, so
        # get_filtered_suite would count it as unclassified and drop it
        # silently. But the original unittest behavior is to surface the
        # error when the test runs (e.g. "test not found" for a typo).
        # Pass it through unfiltered to preserve that error.
        tests = list(suite)
        if tests and isinstance(tests[0], unittest.loader._FailedTest):
            return suite
        return self.get_filtered_suite(suite)


def run_tests(argv=None):
    # [FUNCTION DOC] Main entry point of test files. Parses the command line, loads the slow and
    # disabled test lists (updating globals and environment variables, warning if a file is
    # missing), optionally prints the test names and returns, and checks that all test classes
    # extend the right TestCase (exiting with status 1 otherwise). It then runs the tests in one
    # of several modes: one subprocess per test (retrying once, reporting failures with a
    # reproduction command and raising AssertionError listing failed tests), sharded parallel
    # subprocesses (AssertionError if any shard fails), pytest (exit code 5 treated as success),
    # unittest with the XML runner that also prints skip reasons, a repeated run (exits with -1
    # on a failure), or a plain unittest run. Side effects: global state, environment variables,
    # subprocesses, report files, exits.
    parse_cmd_line_args()
    if argv is None:
        argv = UNITTEST_ARGS

    # import test files.
    if SLOW_TESTS_FILE:
        if os.path.exists(SLOW_TESTS_FILE):
            with open(SLOW_TESTS_FILE) as fp:
                global slow_tests_dict
                slow_tests_dict = json.load(fp)
                # use env vars so pytest-xdist subprocesses can still access them
                os.environ['SLOW_TESTS_FILE'] = SLOW_TESTS_FILE
        else:
            warnings.warn(f'slow test file provided but not found: {SLOW_TESTS_FILE}', stacklevel=2)
    if DISABLED_TESTS_FILE:
        if os.path.exists(DISABLED_TESTS_FILE):
            with open(DISABLED_TESTS_FILE) as fp:
                global disabled_tests_dict
                disabled_tests_dict = json.load(fp)
                os.environ['DISABLED_TESTS_FILE'] = DISABLED_TESTS_FILE
        else:
            warnings.warn(f'disabled test file provided but not found: {DISABLED_TESTS_FILE}', stacklevel=2)
    # Determine the test launch mechanism
    if TEST_DISCOVER:
        _print_test_names()
        return

    # Before running the tests, lint to check that every test class extends from TestCase
    suite = unittest.TestLoader().loadTestsFromModule(__main__)
    if not lint_test_case_extension(suite):
        sys.exit(1)

    testLoader = unittest.loader.defaultTestLoader
    if HW_CLASSIFICATION is not None:
        testLoader = HardwareClassificationTestLoader(HW_CLASSIFICATION)

    if SHOWLOCALS:
        argv = [
            argv[0],
            *(["--showlocals", "--tb=long", "--color=yes"] if USE_PYTEST else ["--locals"]),
            *argv[1:],
        ]

    if TEST_IN_SUBPROCESS:
        suite = testLoader.loadTestsFromModule(__main__)
        other_args = []
        if DISABLED_TESTS_FILE:
            other_args.append("--import-disabled-tests")
        if SLOW_TESTS_FILE:
            other_args.append("--import-slow-tests")
        if USE_PYTEST:
            other_args.append("--use-pytest")
        if RERUN_DISABLED_TESTS:
            other_args.append("--rerun-disabled-tests")
        if TEST_SAVE_XML:
            other_args += ['--save-xml', TEST_SAVE_XML]
        if HW_CLASSIFICATION is not None:
            other_args += ['--hw-classification'] + [req.name for req in HW_CLASSIFICATION]

        test_cases = (
            get_pytest_test_cases(argv) if USE_PYTEST else
            [case.id().split('.', 1)[1] for case in discover_test_cases_recursively(suite)]
        )

        failed_tests = []

        for test_case_full_name in test_cases:

            cmd = (
                [sys.executable] + [argv[0]] + other_args + argv[1:] +
                (["--pytest-single-test"] if USE_PYTEST else []) +
                [test_case_full_name]
            )
            string_cmd = " ".join(cmd)

            timeout = None if RERUN_DISABLED_TESTS else 15 * 60

            exitcode, _ = retry_shell(cmd, timeout=timeout, retries=0 if RERUN_DISABLED_TESTS else 1)

            if exitcode != 0:
                # This is sort of hacky, but add on relevant env variables for distributed tests.
                if 'TestDistBackendWithSpawn' in test_case_full_name:
                    backend = os.environ.get("BACKEND", "")
                    world_size = os.environ.get("WORLD_SIZE", "")
                    env_prefix = f"BACKEND={backend} WORLD_SIZE={world_size}"
                    string_cmd = env_prefix + " " + string_cmd
                # Log the command to reproduce the failure.
                print(f"Test exited with non-zero exitcode {exitcode}. Command to reproduce: {string_cmd}")
                failed_tests.append(test_case_full_name)

        if len(failed_tests) != 0:
            raise AssertionError(
                "{} unit test(s) failed:\n\t{}".format(
                    len(failed_tests), '\n\t'.join(failed_tests)
                )
            )

    elif RUN_PARALLEL > 1:
        suite = testLoader.loadTestsFromModule(__main__)
        test_cases = discover_test_cases_recursively(suite)
        test_batches = chunk_list(get_test_names(test_cases), RUN_PARALLEL)
        processes = []
        for i in range(RUN_PARALLEL):
            command = [sys.executable] + argv + [f'--log-suffix=-shard-{i + 1}'] + test_batches[i]
            processes.append(subprocess.Popen(command, universal_newlines=True))
        failed = False
        for p in processes:
            failed |= wait_for_process(p) != 0
        if failed:
            raise AssertionError("Some test shards have failed")
    elif USE_PYTEST:
        pytest_args = argv + ["--use-main-module"]
        if HW_CLASSIFICATION is not None:
            pytest_args += ['--hw-classification'] + [req.name for req in HW_CLASSIFICATION]
        test_report_path = ""
        if TEST_SAVE_XML:
            test_report_path = get_report_path(pytest=True)
            print(f'Test results will be stored in {test_report_path}')
            pytest_args.append(f'--junit-xml-reruns={test_report_path}')
        if PYTEST_SINGLE_TEST:
            pytest_args = PYTEST_SINGLE_TEST + pytest_args[1:]

        import pytest
        os.environ["NO_COLOR"] = "1"
        exit_code = pytest.main(args=pytest_args)
        if TEST_SAVE_XML:
            sanitize_pytest_xml(test_report_path)

        # exitcode of 5 means no tests were found, which happens since some test configs don't
        # run tests from certain files
        sys.exit(0 if exit_code == 5 else exit_code)
    elif TEST_SAVE_XML:
        # import here so that non-CI doesn't need xmlrunner installed
        import xmlrunner  # type: ignore[import]
        from xmlrunner.result import _XMLTestResult  # type: ignore[import]

        class XMLTestResultVerbose(_XMLTestResult):
            # [CLASS DOC] HELPER: XML test result class that makes the skip message include the
            # skip reason and prints unexpected successes as XPASS in the error summary.
            """
            Adding verbosity to test outputs:
            by default test summary prints 'skip',
            but we want to also print the skip reason.
            GH issue: https://github.com/pytorch/pytorch/issues/69014

            This works with unittest_xml_reporting<=3.2.0,>=2.0.0
            (3.2.0 is latest at the moment)
            """

            def addSkip(self, test, reason):
                # [FUNCTION DOC] HELPER: Record a skip as in the base class, then patch the
                # captured verbose string in the callback closure to read skip followed by
                # the reason.
                super().addSkip(test, reason)
                for c in self.callback.__closure__:
                    if isinstance(c.cell_contents, str) and c.cell_contents == 'skip':
                        # this message is printed in test summary;
                        # it stands for `verbose_str` captured in the closure
                        c.cell_contents = f"skip: {reason}"

            def printErrors(self) -> None:
                # [FUNCTION DOC] HELPER: Print errors as in the base class and additionally
                # print the list of unexpected successes under the XPASS label.
                super().printErrors()
                self.printErrorList("XPASS", self.unexpectedSuccesses)
        test_report_path = get_report_path()
        verbose = '--verbose' in argv or '-v' in argv
        if verbose:
            print(f'Test results will be stored in {test_report_path}')
        unittest.main(argv=argv, testRunner=xmlrunner.XMLTestRunner(
            output=test_report_path,
            verbosity=2 if verbose else 1,
            resultclass=XMLTestResultVerbose), testLoader=testLoader)
    elif REPEAT_COUNT > 1:
        for _ in range(REPEAT_COUNT):
            if not unittest.main(exit=False, argv=argv, testLoader=testLoader).result.wasSuccessful():
                sys.exit(-1)
    else:
        unittest.main(argv=argv, testLoader=testLoader)

IS_LINUX = sys.platform == "linux"
IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_PPC = platform.machine() == "ppc64le"
IS_X86 = platform.machine() in ('x86_64', 'i386')
IS_ARM64 = platform.machine() in ('arm64', 'aarch64', 'ARM64')
IS_S390X = platform.machine() == "s390x"
IS_AVX512_VNNI_SUPPORTED = torch.cpu.get_capabilities().get("avx512_vnni", False)
IS_CPU_EXT_SVE_SUPPORTED = torch.cpu.get_capabilities().get("sve", False)
IS_CPU_CAPABILITY_SVE = torch._C._get_cpu_capability() in ("SVE128", "SVE256")
IS_CPU_CAPABILITY_SVE256 = torch._C._get_cpu_capability() == "SVE256"

if IS_WINDOWS:
    @contextmanager
    def TemporaryFileName(*args, **kwargs):
        # [FUNCTION DOC] Windows variant: context manager that creates a named temporary file,
        # closes it, yields its name, and deletes it afterwards. Arguments go to the temp file
        # constructor; deletion must be disabled (default), otherwise UserWarning is raised.
        # Ideally we would like to not have to manually delete the file, but NamedTemporaryFile
        # opens the file, and it cannot be opened multiple times in Windows. To support Windows,
        # close the file after creation and try to remove it manually
        if 'delete' in kwargs:
            if kwargs['delete'] is not False:
                raise UserWarning("only TemporaryFileName with delete=False is supported on Windows.")
        else:
            kwargs['delete'] = False
        f = tempfile.NamedTemporaryFile(*args, **kwargs)  # noqa:SIM115
        try:
            f.close()
            yield f.name
        finally:
            os.unlink(f.name)
else:
    @contextmanager  # noqa: T484
    def TemporaryFileName(*args, **kwargs):
        # [FUNCTION DOC] Non-Windows variant: context manager that yields the name of a named
        # temporary file created with the given arguments; the file is removed on exit.
        with tempfile.NamedTemporaryFile(*args, **kwargs) as f:
            yield f.name

if IS_WINDOWS:
    @contextmanager
    def TemporaryDirectoryName(suffix=None):
        # [FUNCTION DOC] Windows variant: context manager that creates a temporary directory
        # with mkdtemp, yields its name, and removes it with rmtree on exit.
        # On Windows the directory created by TemporaryDirectory is likely to be removed prematurely,
        # so we first create the directory using mkdtemp and then remove it manually
        try:
            dir_name = tempfile.mkdtemp(suffix=suffix)
            yield dir_name
        finally:
            shutil.rmtree(dir_name)
else:
    @contextmanager  # noqa: T484
    def TemporaryDirectoryName(suffix=None):
        # [FUNCTION DOC] Non-Windows variant: context manager that yields the path of a
        # temporary directory (with the given suffix) which is removed on exit.
        with tempfile.TemporaryDirectory(suffix=suffix) as d:
            yield d


def make_lazy_class(cls):
    # [FUNCTION DOC] Class decorator that turns a class into a lazy value holder. It replaces
    # the initializer with one taking a callback, and installs arithmetic, comparison, bool, int
    # and index special methods that evaluate the callback on first use, cache the value, and
    # delegate to the cached value. Mutates and returns the class.

    def lazy_init(self, cb):
        # [FUNCTION DOC] HELPER: Initializer installed on lazy classes; stores the callback and
        # an empty value.
        self._cb = cb
        self._value = None

    cls.__init__ = lazy_init

    for basename in [
        "add", "sub", "mul", "truediv", "floordiv", "mod", "divmod", "pow",
        "lshift", "rshift", "and", "or", "xor", "neg", "pos", "abs", "invert",
        "eq", "ne", "lt", "le", "gt", "ge", "bool", "int", "index",
    ]:
        name = f"__{basename}__"

        def inner_wrapper(name):
            # [FUNCTION DOC] HELPER: Build the special method for one operator name. The result
            # evaluates the callback once on first call, then calls either the value's own
            # method (for bool and int) or the operator module function of the same name.
            use_operator = basename not in ("bool", "int")

            def wrapped(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: The generated special method: evaluates the pending
                # callback if any (and clears it), then delegates the operation to the
                # cached value.
                if self._cb is not None:
                    self._value = self._cb()
                    self._cb = None
                if not use_operator:
                    return getattr(self._value, name)(*args, **kwargs)
                else:
                    return getattr(operator, name)(self._value, *args, **kwargs)
            return wrapped

        setattr(cls, name, inner_wrapper(name))

    return cls


@make_lazy_class
class LazyVal:
    # [CLASS DOC] Lazily evaluated value (created through make_lazy_class). Wraps a callback
    # that is evaluated on first use of the object in an operator, comparison, bool or int
    # context. Used for expensive availability flags such as TEST_ACCELERATOR.
    pass


IS_FILESYSTEM_UTF8_ENCODING = sys.getfilesystemencoding() == 'utf-8'

TEST_NUMPY = _check_module_exists('numpy')
TEST_FAIRSEQ = _check_module_exists('fairseq')
TEST_SCIPY = _check_module_exists('scipy')
TEST_MKL = torch.backends.mkl.is_available()
TEST_ONEDNN = torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available()
TEST_ACL = torch.backends.mkldnn.is_available() and torch.ops.mkldnn._is_mkldnn_acl_supported()
TEST_MPS = torch.backends.mps.is_available()
MACOS_VERSION = float('.'.join(platform.mac_ver()[0].split('.')[:2]) or -1)
TEST_XPU = torch.xpu.is_available()
TEST_HPU = bool(hasattr(torch, "hpu") and torch.hpu.is_available())
TEST_CUDA = torch.cuda.is_available()
TEST_ACCELERATOR = LazyVal(lambda: torch.accelerator.is_available())  # type: ignore[call-arg]
TEST_MULTIACCELERATOR = LazyVal(lambda: torch.accelerator.device_count() > 1)  # type: ignore[call-arg]
ACCELERATOR_TYPE = LazyVal(lambda: acc.type if (acc := torch.accelerator.current_accelerator(check_available=True)) else None)
custom_device_mod = getattr(torch, torch._C._get_privateuse1_backend_name(), None)
TEST_PRIVATEUSE1 = _is_privateuse1_backend_available()
TEST_PRIVATEUSE1_DEVICE_TYPE = torch._C._get_privateuse1_backend_name()
TEST_NUMBA = _check_module_exists('numba')
TEST_TRANSFORMERS = _check_module_exists('transformers')
TEST_DILL = _check_module_exists('dill')

TEST_LIBROSA = _check_module_exists('librosa') and not IS_ARM64

TEST_OPT_EINSUM = _check_module_exists('opt_einsum')

TEST_Z3 = _check_module_exists('z3')

# DSL availability (lazy evaluation to avoid import overhead)
class LazyDSLCheck:
    # [CLASS DOC] Lazy checker for availability of native DSL backends. It imports the DSL
    # registry only on first use and answers availability questions, so that importing this
    # module does not pay the import cost. State: the registry (None if unavailable) and a flag
    # recording whether an import was attempted.
    """Lazy DSL availability checker to avoid import-time overhead"""
    def __init__(self):
        # [FUNCTION DOC] Initialize with no registry and no import attempted.
        self._registry = None
        self._import_attempted = False

    def _get_registry(self):
        # [FUNCTION DOC] HELPER: Import the DSL registry once; remember failure by keeping None.
        # Returns the registry or None.
        if not self._import_attempted:
            self._import_attempted = True
            try:
                from torch._native.dsl_registry import dsl_registry
                self._registry = dsl_registry
            except ImportError:
                self._registry = None
        return self._registry

    def is_available(self, dsl_name: str) -> bool:
        # [FUNCTION DOC] Return True if the registry reports the named DSL as available, False
        # if there is no registry.
        """Check if specific DSL is available"""
        registry = self._get_registry()
        return registry.is_dsl_available(dsl_name) if registry is not None else False

    def list_available(self) -> list[str]:
        # [FUNCTION DOC] Return a list of the names of available DSLs, empty if there is
        # no registry.
        """Get list of available DSLs"""
        registry = self._get_registry()
        return list(registry.list_available_dsls()) if registry is not None else []

    def list_all(self) -> list[str]:
        # [FUNCTION DOC] Return a list of the names of all registered DSLs, empty if there is
        # no registry.
        """Get list of all registered DSLs"""
        registry = self._get_registry()
        return list(registry.list_all_dsls()) if registry is not None else []

_dsl_checker = LazyDSLCheck()

# Lazy constants to avoid import-time overhead
TEST_TRITON_DSL = LazyVal(lambda: _dsl_checker.is_available('triton'))
TEST_CUTEDSL = LazyVal(lambda: _dsl_checker.is_available('cutedsl'))
TEST_HELION_DSL = LazyVal(lambda: _dsl_checker.is_available('helion'))

def split_if_not_empty(x: str):
    # [FUNCTION DOC] HELPER: Split a string on commas, returning an empty list for an
    # empty string.
    return x.split(",") if len(x) != 0 else []

NOTEST_CPU = "cpu" in split_if_not_empty(os.getenv('PYTORCH_TESTING_DEVICE_EXCEPT_FOR', ''))

skipIfNoDill = unittest.skipIf(not TEST_DILL, "no dill")

# DSL skip decorators (following existing pattern)
skipIfNoTritonDSL = unittest.skipIf(not TEST_TRITON_DSL, "Triton DSL not available")
skipIfNoCuteDSL = unittest.skipIf(not TEST_CUTEDSL, "CuTeDSL not available")
skipIfNoHelionDSL = unittest.skipIf(not TEST_HELION_DSL, "Helion DSL not available")

def skipIfDSLUnavailable(dsl_name: str, reason: str | None = None):
    # [FUNCTION DOC] Return a unittest skip decorator that skips tests when the named DSL is not
    # available. The check happens at decoration time. A custom reason may be given.
    """Skip test if specific DSL is not available"""
    available = _dsl_checker.is_available(dsl_name)
    msg = reason or f"{dsl_name} DSL not available"
    return unittest.skipIf(not available, msg)

def skipUnlessDSLAvailable(dsl_name: str, reason: str | None = None):
    # [FUNCTION DOC] Return a unittest skip decorator that skips tests unless the named DSL is
    # available. The check happens at decoration time. A custom reason may be given.
    """Skip test unless specific DSL is available"""
    available = _dsl_checker.is_available(dsl_name)
    msg = reason or f"{dsl_name} DSL required"
    return unittest.skipUnless(available, msg)

def get_available_dsls() -> list[str]:
    # [FUNCTION DOC] Return the list of available DSL names, for use in test parametrization.
    """Get list of available DSL names for test parameterization"""
    return _dsl_checker.list_available()

def is_dsl_available(dsl_name: str) -> bool:
    # [FUNCTION DOC] Return True if the named DSL is available.
    """Check if specific DSL is available for conditional testing"""
    return _dsl_checker.is_available(dsl_name)

def get_all_dsls() -> list[str]:
    # [FUNCTION DOC] Return the list of all registered DSL names, available or not.
    """Get all registered DSL names (available or not) for comprehensive testing"""
    return _dsl_checker.list_all()


NO_MULTIPROCESSING_SPAWN: bool = False
TEST_WITH_ASAN: bool = TestEnvironment.def_flag(
    "TEST_WITH_ASAN",
    env_var="PYTORCH_TEST_WITH_ASAN",
)
TEST_WITH_DEV_DBG_ASAN: bool = TestEnvironment.def_flag(
    "TEST_WITH_DEV_DBG_ASAN",
    env_var="PYTORCH_TEST_WITH_DEV_DBG_ASAN",
)
TEST_WITH_TSAN: bool = TestEnvironment.def_flag(
    "TEST_WITH_TSAN",
    env_var="PYTORCH_TEST_WITH_TSAN",
)
TEST_WITH_UBSAN: bool = TestEnvironment.def_flag(
    "TEST_WITH_UBSAN",
    env_var="PYTORCH_TEST_WITH_UBSAN",
)
TEST_WITH_ROCM: bool = TestEnvironment.def_flag(
    "TEST_WITH_ROCM",
    env_var="PYTORCH_TEST_WITH_ROCM",
    implied_by_fn=lambda: torch.version.hip is not None,
)
TEST_WITH_MTIA: bool = TestEnvironment.def_flag(
    "TEST_WITH_MTIA",
    env_var="PYTORCH_TEST_WITH_MTIA",
)

# TODO: Remove PYTORCH_MIOPEN_SUGGEST_NHWC once ROCm officially supports NHWC in MIOpen
# See #64427
TEST_WITH_MIOPEN_SUGGEST_NHWC = os.getenv('PYTORCH_MIOPEN_SUGGEST_NHWC', '0') == '1'
# Enables tests that are slow to run (disabled by default)
TEST_WITH_SLOW: bool = TestEnvironment.def_flag(
    "TEST_WITH_SLOW",
    env_var="PYTORCH_TEST_WITH_SLOW",
)

# Disables non-slow tests (these tests enabled by default)
# This is usually used in conjunction with TEST_WITH_SLOW to
# run *only* slow tests.  (I could have done an enum, but
# it felt a little awkward.
TEST_SKIP_FAST: bool = TestEnvironment.def_flag(
    "TEST_SKIP_FAST",
    env_var="PYTORCH_TEST_SKIP_FAST",
)

# Enables crossref tests, in addition to standard tests which
# are being run.  crossref tests work by installing a torch
# function mode that runs extra compute alongside the regular
# computation that happens with the test.  After both computations
# are done, we cross-reference them (thus the name) to check for
# correction, before throwing out the extra compute and proceeding
# as we had before.  By default, we don't run these tests.
TEST_WITH_CROSSREF: bool = TestEnvironment.def_flag(
    "TEST_WITH_CROSSREF",
    env_var="PYTORCH_TEST_WITH_CROSSREF",
)

TEST_SKIP_CUDAGRAPH: bool = TestEnvironment.def_flag(
    "TEST_SKIP_CUDAGRAPH",
    env_var="PYTORCH_TEST_SKIP_CUDAGRAPH",
)
TEST_CUDA_GRAPH = TEST_CUDA and (not TEST_SKIP_CUDAGRAPH) and (
    torch.version.cuda or torch.version.hip
)

TEST_CUDA_CUDSS = TEST_CUDA and torch.version.cuda is not None
TEST_CUDA_GRAPH_CONDITIONAL_NODES = TEST_CUDA_GRAPH and torch.version.cuda is not None

TEST_CUDA_PYTHON_BINDINGS = _check_module_exists("cuda.bindings") and torch.version.cuda is not None
TEST_NVMATH = _check_module_exists("nvmath.bindings") and torch.version.cuda is not None
skipIfNoNvmath = unittest.skipIf(not TEST_NVMATH, "nvmath-python not available")

if TEST_CUDA_PYTHON_BINDINGS:
    def cuda_python_error_check(function_call_output):
        # [FUNCTION DOC] Check the result tuple of a cuda-python runtime call. The first element
        # is the error code; if it is not the success code a ValueError is raised, otherwise the
        # remaining elements are returned as a tuple. Defined only when the bindings exist.
        """Makes calls to cuda-python's cuda runtime functions more
        pythonic by throwing an exception if they return a status
        which is not cudaSuccess
        """
        import cuda.bindings  # type: ignore[import]

        error, *others = function_call_output
        if error != cuda.bindings.runtime.cudaError_t.cudaSuccess:
            raise ValueError(f"CUDA failure! {error}")
        else:
            return tuple(others)
else:
    cuda_python_error_check = None  # type: ignore[assignment]

def allocator_option_enabled_fn(allocator_config, _, option):
    # [FUNCTION DOC] HELPER: Enabled-function for TestEnvironment flags that read allocator
    # configuration. Parses a comma separated list of name and value pairs and returns True
    # only if the named option is present with the value True. Returns False if the
    # configuration is None.
    if allocator_config is None:
        return False
    allocator_config = allocator_config.split(',') if ',' in allocator_config else [allocator_config]
    mapping = dict([var.split(':') for var in allocator_config])

    if option in mapping and mapping[option] == 'True':
        return True
    else:
        return False

EXPANDABLE_SEGMENTS: bool = TestEnvironment.def_flag(
    "EXPANDABLE_SEGMENTS",
    env_var="PYTORCH_CUDA_ALLOC_CONF",
    enabled_fn=functools.partial(allocator_option_enabled_fn, option='expandable_segments'),
)

if TEST_CUDA and 'NUM_PARALLEL_PROCS' in os.environ:
    num_procs = int(os.getenv("NUM_PARALLEL_PROCS", "2"))
    gb_available = torch.cuda.mem_get_info()[1] / 2 ** 30
    # other libraries take up about a little under 1 GB of space per process
    torch.cuda.set_per_process_memory_fraction(round((gb_available - num_procs * .85) / gb_available / num_procs, 2))

requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")


def lazy_skip_if(condition_fn, reason):
    # [FUNCTION DOC] Build a decorator that skips a test function or test class when the
    # condition function returns true. For functions the condition is checked on every run and a
    # SkipTest is raised; for classes it is checked once at decoration time and the standard
    # unittest skip attributes are set. Returns the decorator.
    """Skip a test (function or class) when ``condition_fn()`` is true.

    For function targets the condition is evaluated each time the test
    runs, matching the historical PyTorch convention of checking skip
    flags inside a wrapper. For class targets the condition is evaluated
    once at class-decoration time and the standard ``__unittest_skip__``
    attributes are set, since unittest's TestLoader makes class-level
    skip decisions before instantiation.

    Prefer this helper over hand-rolled ``@wraps + raise SkipTest``
    wrappers, which silently drop classes from discovery when applied at
    class scope.
    """

    def decorator(fn):
        # [FUNCTION DOC] HELPER: Apply the skip logic to a class (mark it skipped if the
        # condition holds) or a function (wrap it so it raises SkipTest when the
        # condition holds).
        if isinstance(fn, type):
            if condition_fn():
                fn.__unittest_skip__ = True  # type: ignore[attr-defined]
                fn.__unittest_skip_why__ = reason  # type: ignore[attr-defined]
            return fn

        @wraps(fn)
        def wrapper(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Raise SkipTest with the reason if the condition holds,
            # otherwise call the wrapped test and return its result.
            if condition_fn():
                raise unittest.SkipTest(reason)
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def skipIfCrossRef(fn):
    # [FUNCTION DOC] Decorator that skips the test when the crossref mode is enabled. Returns
    # the wrapped function.
    # KNOWN ISSUE: the skip reason text reads "test doesn't currently with crossref", a word
    # is missing.
    return lazy_skip_if(lambda: TEST_WITH_CROSSREF, "test doesn't currently with crossref")(fn)

class CrossRefMode(torch.overrides.TorchFunctionMode):
    # [CLASS DOC] Torch function mode used for crossref testing; it simply runs each function
    # call normally and returns its result without extra computation.
    def __torch_function__(self, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] Run the intercepted function with its arguments (keyword arguments
        # default to empty) and return the result unchanged.
        kwargs = kwargs or {}
        r = func(*args, **kwargs)
        return r

# Run PyTorch tests with TorchDynamo
TEST_WITH_TORCHINDUCTOR: bool = TestEnvironment.def_flag(
    "TEST_WITH_TORCHINDUCTOR",
    env_var="PYTORCH_TEST_WITH_INDUCTOR",
)
# AOT_EAGER not tested in ci, useful for debugging
TEST_WITH_AOT_EAGER: bool = TestEnvironment.def_flag(
    "TEST_WITH_AOT_EAGER",
    env_var="PYTORCH_TEST_WITH_AOT_EAGER",
)
TEST_WITH_TORCHDYNAMO: bool = TestEnvironment.def_flag(
    "TEST_WITH_TORCHDYNAMO",
    env_var="PYTORCH_TEST_WITH_DYNAMO",
    implied_by_fn=lambda: TEST_WITH_TORCHINDUCTOR or TEST_WITH_AOT_EAGER,
)
TEST_WITHOUT_COMPILED_AUTOGRAD: bool = TestEnvironment.def_flag(
    "TEST_WITHOUT_COMPILED_AUTOGRAD",
    env_var="PYTORCH_TEST_WITHOUT_COMPILED_AUTOGRAD",
)

if TEST_WITH_TORCHDYNAMO:
    import torch._dynamo
    # Do not spend time on helper functions that are called with different inputs
    torch._dynamo.config.accumulated_recompile_limit = 64
    # Do not log compilation metrics from unit tests
    torch._dynamo.config.log_compilation_metrics = False
    # Silence 3.13.0 guard performance warnings
    torch._dynamo.config.issue_3_13_0_warning = False
    if TEST_WITH_TORCHINDUCTOR:
        import torch._inductor.config
        torch._inductor.config.fallback_random = True
    else:
        # only dynamo for now
        torch._dynamo.config.compiled_autograd = not TEST_WITHOUT_COMPILED_AUTOGRAD


# seems like this is only used in test/torch_np
def xpassIfTorchDynamo_np(func):
    # [FUNCTION DOC] Decorator for numpy-compat tests. Under dynamo with numpy 2 it skips the
    # test; under dynamo otherwise it returns the function; outside dynamo it marks the test as
    # an expected failure. Returns the function or wrapped function.
    # numpy 2.0+ is causing issues
    if TEST_WITH_TORCHDYNAMO and np.__version__[0] == '2':
        return unittest.skip("skipping numpy 2.0+ dynamo-wrapped test")(func)
    return func if TEST_WITH_TORCHDYNAMO else unittest.expectedFailure(func)


requires_mkl = unittest.skipUnless(TEST_MKL, "Test requires MKL")
requires_onednn = unittest.skipUnless(TEST_ONEDNN, "Test requires OneDNN/MKLDNN")

def xfailIfACL(func):
    # [FUNCTION DOC] Mark the function as expected to fail when ACL (Arm Compute Library)
    # support is present, otherwise return it unchanged.
    return unittest.expectedFailure(func) if TEST_ACL else func


def xfailIfTorchDynamo(func):
    # [FUNCTION DOC] Mark the function as expected to fail when running under dynamo, otherwise
    # return it unchanged.
    return unittest.expectedFailure(func) if TEST_WITH_TORCHDYNAMO else func


def xfailIfPy312Plus(func):
    # [FUNCTION DOC] Mark the function as expected to fail on Python 3.12 and newer.
    return unittest.expectedFailure(func) if sys.version_info >= (3, 12) else func


def xfailIfPy314Plus(func):
    # [FUNCTION DOC] Mark the function as expected to fail on Python 3.14 and newer.
    return unittest.expectedFailure(func) if sys.version_info >= (3, 14) else func


def xfailIfPy313AndEarlier(func):
    # [FUNCTION DOC] Mark the function as expected to fail on Python versions before 3.14.
    return unittest.expectedFailure(func) if sys.version_info < (3, 14) else func


def xfailIfLinux(func):
    # [FUNCTION DOC] Mark the function as expected to fail on Linux, except on ROCm and in
    # fbcode builds.
    return unittest.expectedFailure(func) if IS_LINUX and not TEST_WITH_ROCM and not IS_FBCODE else func


def xfailIfWindows(func):
    # [FUNCTION DOC] Mark the function as expected to fail on Windows.
    return unittest.expectedFailure(func) if IS_WINDOWS else func


def xfailIfROCm(func):
    # [FUNCTION DOC] Mark the function as expected to fail on ROCm builds.
    return unittest.expectedFailure(func) if torch.version.hip is not None else func


def _is_cpu_device_type(dev) -> bool:
    # [FUNCTION DOC] HELPER: Return True if the argument is a torch device of type cpu or a
    # string naming cpu (optionally with an index), otherwise False.
    if isinstance(dev, torch.device):
        return dev.type == "cpu"
    if isinstance(dev, str):
        return dev == "cpu" or dev.startswith("cpu:")
    return False


def _device_spec_from_test_call(args: tuple, kwargs: dict):
    # [FUNCTION DOC] HELPER: Return the device or devices keyword argument of a test call,
    # preferring device, or None if neither is present. Positional args are ignored.
    if "device" in kwargs:
        return kwargs["device"]
    if "devices" in kwargs:
        return kwargs["devices"]
    return None


def xfailIfNoAcceleratorTriton(test_func):
    # [FUNCTION DOC] Decorator for a test function or a whole test class. If triton is installed
    # the test runs normally. Otherwise, for CPU devices the test runs and an import error
    # mentioning triton becomes an xfail; for other devices it is xfailed through pytest. For a
    # class, each test method is wrapped. Returns the class or wrapper.
    """Run test normally if triton is present or if running on CPU (which falls back to openmp).
    Otherwise mark as xfail — any accelerator (CUDA, XPU, ROCm, etc.) requires triton.
    Can be applied to a test method or an entire test class."""
    import inspect
    import functools
    from torch.utils._triton import has_triton

    if inspect.isclass(test_func):
        for attr_name in list(vars(test_func)):
            if attr_name.startswith("test"):
                method = getattr(test_func, attr_name)
                if callable(method):
                    setattr(test_func, attr_name, xfailIfNoAcceleratorTriton(method))
        return test_func

    @functools.wraps(test_func)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Run the test if triton exists. Otherwise find the device from
        # the call, run on CPU converting triton import errors into xfail, or call pytest xfail.
        if has_triton():
            return test_func(*args, **kwargs)

        spec = _device_spec_from_test_call(args, kwargs)
        if spec is None and args:
            spec = getattr(args[0], "device_type", None)
        if spec is not None and _is_cpu_device_type(spec):
            try:
                return test_func(*args, **kwargs)
            except ImportError as e:
                # This except block required only for TestUtilsCPU::test_get_device_tflops_cpu
                # test_get_device_tflops imports triton directly in its body — even for CPU
                if "triton" in str(e).lower():
                    import pytest
                    pytest.xfail(f"Triton not available (device={spec!r}): {e}")
                raise

        import pytest
        device_info = f" (device={spec!r})" if spec is not None else ""
        pytest.xfail(f"Triton not available{device_info}")

    return wrapper


def skipIfFreeThreaded(msg="Test doesn't work with free-threaded python"):
    # [FUNCTION DOC] Return a skip decorator for free-threaded Python builds with the given
    # message. Raises AssertionError if the message is not a string (e.g. decorator misuse).
    if not isinstance(msg, str):
        raise AssertionError("Are you using skipIfFreeThreaded correctly?")
    return unittest.skipIf(sysconfig.get_config_var("Py_GIL_DISABLED") == 1, msg)


def skipIfTorchDynamo(msg="test doesn't currently work with dynamo"):
    # [FUNCTION DOC] Return a decorator that skips a test function, or a class, when running
    # under dynamo, with the given message. Raises AssertionError if the message is not a
    # string. Function tests raise SkipTest at call time; classes get skip attributes set.
    """
    Usage:
    @skipIfTorchDynamo(msg)
    def test_blah(self):
        ...
    """
    if not isinstance(msg, str):
        raise AssertionError("Are you using skipIfTorchDynamo correctly?")

    def decorator(fn):
        # [FUNCTION DOC] HELPER: Wrap a function so it raises SkipTest under dynamo, or mark a
        # class as skipped under dynamo. Returns the wrapper or class.
        # KNOWN ISSUE: the second check that fn is a type is redundant, since non-types return
        # earlier; also the wrapper drops the return value of the test.
        if not isinstance(fn, type):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Raise SkipTest under dynamo, otherwise call the test
                # (the return value is not propagated).
                if TEST_WITH_TORCHDYNAMO:
                    raise unittest.SkipTest(msg)
                else:
                    fn(*args, **kwargs)
            return wrapper

        if not isinstance(fn, type):
            raise AssertionError(f"expected fn to be a type, got {type(fn)}")
        if TEST_WITH_TORCHDYNAMO:
            fn.__unittest_skip__ = True  # type: ignore[attr-defined]
            fn.__unittest_skip_why__ = msg  # type: ignore[attr-defined]

        return fn

    return decorator

def skipIfTorchInductor(msg="test doesn't currently work with torchinductor",
                        condition=TEST_WITH_TORCHINDUCTOR):
    # [FUNCTION DOC] Return a decorator that skips a test function or class when condition is
    # true (by default, when running under inductor), with the given message.
    def decorator(fn):
        # [FUNCTION DOC] HELPER: Wrap a function so it raises SkipTest when the condition holds,
        # or mark a class as skipped. Returns the wrapper or class.
        if not isinstance(fn, type):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Raise SkipTest if the condition holds, otherwise call
                # the test (the return value is not propagated).
                if condition:
                    raise unittest.SkipTest(msg)
                else:
                    fn(*args, **kwargs)
            return wrapper

        if not isinstance(fn, type):
            raise AssertionError(f"expected fn to be a type, got {type(fn)}")
        if condition:
            fn.__unittest_skip__ = True  # type: ignore[attr-defined]
            fn.__unittest_skip_why__ = msg  # type: ignore[attr-defined]

        return fn

    return decorator

def runWithoutCompiledAutograd(msg="test doesn't currently work with compiled autograd"):
    # [FUNCTION DOC] Return a decorator that runs the test with compiled autograd disabled.
    # Raises AssertionError if the message is not a string.
    """
    Usage:
    @runWithoutCompiledAutograd(msg)
    def test_blah(self):
        ...
    """
    if not isinstance(msg, str):
        raise AssertionError(f"expected msg to be str, got {type(msg)}")

    def decorator(func):
        # [FUNCTION DOC] HELPER: Wrap the function so it runs inside the compiled autograd
        # disable context.
        @wraps(func)
        def wrapper(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Call the test under the compiled autograd disable context
            # (the return value is not propagated).
            with torch._dynamo.compiled_autograd._disable():
                func(*args, **kwargs)
        return wrapper

    return decorator

def serialTest(condition=True):
    # [FUNCTION DOC] Return a decorator that marks a test with the pytest serial marker when
    # pytest is installed and the condition is true; otherwise the test is unchanged. Raises
    # AssertionError if the condition is not a bool, which catches use without call.
    """
    Decorator for running tests serially.  Requires pytest
    """
    # If one apply decorator directly condition will be callable
    # And test will essentially be essentially skipped, which is undesirable
    if type(condition) is not bool:
        raise AssertionError(f"expected condition to be bool, got {type(condition)}")

    def decorator(fn):
        # [FUNCTION DOC] HELPER: Mark the function serial if pytest exists and the condition is
        # true, else return it unchanged.
        if has_pytest and condition:
            return pytest.mark.serial(fn)
        return fn
    return decorator

def unMarkDynamoStrictTest(cls=None):
    # [FUNCTION DOC] Mark a class as not strict for dynamo by setting its dynamo_strict
    # attribute to False. Usable with or without parentheses: with a class it modifies and
    # returns it, without one it returns a decorator.
    def decorator(cls):
        # [FUNCTION DOC] HELPER: Set the dynamo_strict attribute of the class to False and
        # return it.
        cls.dynamo_strict = False
        return cls

    if cls is None:
        return decorator
    else:
        return decorator(cls)


def markDynamoStrictTest(cls_or_func=None, nopython=False):
    # [FUNCTION DOC] Mark a test class or function as strict for dynamo. For classes it sets the
    # strict attributes (including whether nopython is used). For functions it returns a wrapper
    # that resets dynamo before and after and runs with suppress_errors off. Usable with or
    # without parentheses.
    """
    Marks the test as 'strict'. In strict mode, we reset before and after the
    test, and run without suppress errors.

    Args:
    - nopython: if we should run torch._dynamo.optimize with nopython={True/False}.
    """
    def decorator(cls_or_func):
        # [FUNCTION DOC] HELPER: Apply strict marking to a class (setting attributes) or wrap a
        # function with dynamo reset and suppress_errors disabled. Returns class or wrapper.
        if inspect.isclass(cls_or_func):
            cls_or_func.dynamo_strict = True
            cls_or_func.dynamo_strict_nopython = nopython
            return cls_or_func

        fn = cls_or_func

        @wraps(fn)
        def wrapper(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Reset dynamo, run the test with suppress_errors set to
            # False, reset dynamo again. The return value is not propagated.
            torch._dynamo.reset()
            with unittest.mock.patch("torch._dynamo.config.suppress_errors", False):
                fn(*args, **kwargs)
            torch._dynamo.reset()
        return wrapper

    if cls_or_func is None:
        return decorator
    else:
        return decorator(cls_or_func)


def skipRocmIfTorchInductor(msg="test doesn't currently work with torchinductor on the ROCm stack"):
    # [FUNCTION DOC] Skip decorator for ROCm plus inductor: delegates to skipIfTorchInductor
    # with the condition that both ROCm and inductor testing are enabled.
    return skipIfTorchInductor(msg=msg, condition=TEST_WITH_ROCM and TEST_WITH_TORCHINDUCTOR)

def skipIfLegacyJitExecutor(msg="test doesn't currently work with legacy JIT executor"):
    # [FUNCTION DOC] Return a decorator that skips a function or class when the JIT graph
    # executor is LEGACY, with the given message.
    def decorator(fn):
        # [FUNCTION DOC] HELPER: Wrap a function so it raises SkipTest under the legacy
        # executor, or mark a class as skipped there. Raises AssertionError if GRAPH_EXECUTOR
        # is unset.
        if not isinstance(fn, type):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Raise SkipTest under the legacy executor, otherwise
                # call the test. Raises AssertionError if GRAPH_EXECUTOR is unset.
                if not GRAPH_EXECUTOR:
                    raise AssertionError("GRAPH_EXECUTOR must be set")
                if GRAPH_EXECUTOR == ProfilingMode.LEGACY:
                    raise unittest.SkipTest(msg)
                else:
                    fn(*args, **kwargs)
            return wrapper

        if not isinstance(fn, type):
            raise AssertionError(f"expected fn to be a type, got {type(fn)}")
        if GRAPH_EXECUTOR == ProfilingMode.LEGACY:
            fn.__unittest_skip__ = True  # type: ignore[attr-defined]
            fn.__unittest_skip_why__ = msg  # type: ignore[attr-defined]

        return fn


    return decorator


def make_dynamo_test(
    fn: Callable[..., Any] | None = None
) -> Callable[..., Any]:
    # [FUNCTION DOC] Decorator that turns a test method into a dynamo test. The wrapped test
    # takes the unittest object as its first argument; it is run once in eager mode and once
    # compiled with dynamo, asserting no graph break, with dynamo reset between and after. If
    # called without a function it returns a decorator. Returns the test.
    """
    Decorator function to create a dynamo test case. A function annotate with
    this decorator takes as input a unittest object.
    """
    from torch._dynamo.testing import CompileCounter, reset, optimize_assert
    if fn is None:
        return lambda fn: make_dynamo_test(fn)

    def standard_test(
        self: Any,
        fn: Callable[..., Any],
        kwargs,
    ) -> None:
        # [FUNCTION DOC] HELPER: Run the test function eagerly, reset dynamo, run it again
        # through optimize_assert with a compile counter, then reset dynamo.
        def dummy() -> None:
            # [FUNCTION DOC] HELPER: Call the test function with the test object and the
            # keyword arguments.
            fn(self, **kwargs)

        actual = CompileCounter()

        dummy()
        reset()
        opt_fn = optimize_assert(actual)(dummy)
        opt_fn()
        reset()

    @functools.wraps(fn)
    def test_fn(self: Any, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: The generated test method; forwards keyword arguments to
        # standard_test together with the original function.
        return standard_test(
            self,
            fn=fn,
            kwargs=kwargs,
        )

    return test_fn


# Run PyTorch tests with translation validation on.
TEST_WITH_TV = os.getenv('PYTORCH_TEST_WITH_TV') == '1'

if TEST_WITH_TV:
    torch.fx.experimental._config.translation_validation = True

# Determine whether to enable cuda memory leak check.
# CUDA mem leak check is expensive and thus we don't want to execute it on every
# test case / configuration.
# If this is True then CUDA memory leak checks are skipped. If this is false
#   then CUDA memory leak checks are performed.
# See: https://github.com/pytorch/pytorch/pull/59402#issuecomment-858811135
TEST_CUDA_MEM_LEAK_CHECK: bool = TestEnvironment.def_flag(
    "TEST_CUDA_MEM_LEAK_CHECK",
    env_var="PYTORCH_TEST_CUDA_MEM_LEAK_CHECK",
)


# Dict of NumPy dtype -> torch dtype (when the correspondence exists)
numpy_to_torch_dtype_dict = {
    np.bool_      : torch.bool,
    np.uint8      : torch.uint8,
    np.uint16     : torch.uint16,
    np.uint32     : torch.uint32,
    np.uint64     : torch.uint64,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}


# numpy dtypes like np.float64 are not instances, but rather classes. This leads to rather absurd cases like
# np.float64 != np.dtype("float64") but np.float64 == np.dtype("float64").type.
# Especially when checking against a reference we can't be sure which variant we get, so we simply try both.
def numpy_to_torch_dtype(np_dtype):
    # [FUNCTION DOC] Map a numpy dtype (or scalar type) to the matching torch dtype using the
    # table. If the key is missing, retries with the dtype type attribute; may raise KeyError.
    try:
        return numpy_to_torch_dtype_dict[np_dtype]
    except KeyError:
        return numpy_to_torch_dtype_dict[np_dtype.type]


def has_corresponding_torch_dtype(np_dtype):
    # [FUNCTION DOC] Return True if the numpy dtype has a matching torch dtype, else False.
    try:
        numpy_to_torch_dtype(np_dtype)
        return True
    except KeyError:
        return False


if IS_WINDOWS:
    # Size of `np.intc` is platform defined.
    # It is returned by functions like `bitwise_not`.
    # On Windows `int` is 32-bit
    # https://docs.microsoft.com/en-us/cpp/cpp/data-type-ranges?view=msvc-160
    numpy_to_torch_dtype_dict[np.intc] = torch.int

# Dict of torch dtype -> NumPy dtype
torch_to_numpy_dtype_dict = {value : key for (key, value) in numpy_to_torch_dtype_dict.items()}
torch_to_numpy_dtype_dict.update({
    torch.bfloat16: np.float32,
    torch.complex32: np.complex64
})

def skipIfNNModuleInlined(
    msg="test doesn't currently work with nn module inlining",
    condition=True,
):
    # [FUNCTION DOC] Return a decorator that skips a function or class when nn module inlining
    # is active (condition defaults to true), with the given message.
    def decorator(fn):
        # [FUNCTION DOC] HELPER: Wrap a function so it raises SkipTest when the condition holds,
        # or mark a class as skipped. Returns the wrapper or class.
        # KNOWN ISSUE: the second type check is redundant and the wrapper drops the
        # return value.
        if not isinstance(fn, type):

            @wraps(fn)
            def wrapper(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Raise SkipTest if the condition holds, otherwise call
                # the test (the return value is not propagated).
                if condition:
                    raise unittest.SkipTest(msg)
                else:
                    fn(*args, **kwargs)

            return wrapper

        if not isinstance(fn, type):
            raise AssertionError(f"expected fn to be a type, got {type(fn)}")
        if condition:
            fn.__unittest_skip__ = True  # type: ignore[attr-defined]
            fn.__unittest_skip_why__ = msg  # type: ignore[attr-defined]

        return fn

    return decorator

def skipIfRocm(func=None, *, msg="test doesn't currently work on the ROCm stack"):
    # [FUNCTION DOC] Skip decorator for ROCm builds. Usable bare or with a msg keyword; built on
    # lazy_skip_if so it works for test methods and test classes. Returns the decorated object,
    # or a decorator when no function is given.
    decorator = lazy_skip_if(lambda: TEST_WITH_ROCM, f"skipIfRocm: {msg}")
    return decorator(func) if func is not None else decorator

def skipIfRocm_BUGGY(func=None, *, msg="test doesn't currently work on the ROCm stack"):
    # [FUNCTION DOC] Deprecated variant of skipIfRocm kept for migration. It only wraps
    # functions and so silently drops classes from discovery. Returns the decorated function or
    # a decorator when no function is given.
    """Old skipIfRocm that silently drops classes from discovery. Migrate to skipIfRocm."""
    def dec_fn(fn):
        # [FUNCTION DOC] HELPER: Wrap a function so it raises SkipTest on ROCm builds.
        reason = f"skipIfRocm: {msg}"

        @wraps(fn)
        def wrapper(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Raise SkipTest on ROCm, otherwise call the test and return
            # its result.
            if TEST_WITH_ROCM:
                raise unittest.SkipTest(reason)
            else:
                return fn(*args, **kwargs)
        return wrapper
    if func:
        return dec_fn(func)
    return dec_fn

def getRocmArchName(device_index: int = 0):
    # [FUNCTION DOC] Return the architecture name string (for example gfx942) of the given
    # CUDA/HIP device index (default 0). Queries device properties.
    return torch.cuda.get_device_properties(device_index).gcnArchName

def isRocmArchAnyOf(arch: tuple[str, ...]):
    # [FUNCTION DOC] Return True if this is a HIP build and the device 0 architecture name
    # contains any of the given architecture strings; False on non-HIP builds.
    if not torch.version.hip:
        return False
    rocmArch = getRocmArchName()
    return any(x in rocmArch for x in arch)

def skipIfRocmArch(arch: tuple[str, ...]):
    # [FUNCTION DOC] Return a skip decorator that skips tests on ROCm when the device
    # architecture is one of the given ones. Evaluated lazily at test time.
    return lazy_skip_if(
        lambda: TEST_WITH_ROCM and isRocmArchAnyOf(arch),
        f"skipIfRocm: test skipped on {arch}",
    )

def runOnRocm(fn):
    # [FUNCTION DOC] Decorator that runs a test only on ROCm and raises SkipTest otherwise.
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Run the test on ROCm (return value not propagated) or
        # raise SkipTest.
        if TEST_WITH_ROCM:
            fn(*args, **kwargs)
        else:
            raise unittest.SkipTest("test currently only works on the ROCm stack")
    return wrapper

def runOnRocmArch(arch: tuple[str, ...]):
    # [FUNCTION DOC] Return a decorator that, on ROCm builds only, skips the test unless the
    # device architecture matches one of the given ones. Non-ROCm builds are not skipped.
    def dec_fn(fn):
        # [FUNCTION DOC] HELPER: Wrap a test method with the architecture check.
        @wraps(fn)
        def wrap_fn(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: On ROCm with a non matching architecture raise SkipTest,
            # otherwise call the method and return its result.
            if TEST_WITH_ROCM and not isRocmArchAnyOf(arch):
                reason = f"skipIfRocm: test only runs on {arch}"
                raise unittest.SkipTest(reason)
            return fn(self, *args, **kwargs)
        return wrap_fn
    return dec_fn

def xfailIfS390X(func):
    # [FUNCTION DOC] Mark the function as expected to fail on s390x machines.
    return unittest.expectedFailure(func) if IS_S390X else func

def xfailIf(condition):
    # [FUNCTION DOC] Return a decorator that marks a function as an expected failure when the
    # condition is true and leaves it unchanged otherwise.
    def wrapper(func):
        # [FUNCTION DOC] HELPER: Apply expected failure marking if the condition holds.
        if condition:
            return unittest.expectedFailure(func)
        else:
            return func
    return wrapper

def skipIfXpu(func=None, *, msg="test doesn't currently work on the XPU stack"):
    # [FUNCTION DOC] Skip decorator for XPU builds. Usable bare or with a msg keyword; works for
    # test methods and classes through lazy_skip_if.
    decorator = lazy_skip_if(lambda: TEST_XPU, f"skipIfXpu: {msg}")
    return decorator(func) if func is not None else decorator

def skipIfXpu_BUGGY(func=None, *, msg="test doesn't currently work on the XPU stack"):
    # [FUNCTION DOC] Deprecated variant of skipIfXpu kept for migration; it only wraps functions
    # and so silently drops classes from discovery.
    """Old skipIfXpu that silently drops classes from discovery. Migrate to skipIfXpu."""
    def dec_fn(fn):
        # [FUNCTION DOC] HELPER: Wrap a function so it raises SkipTest when XPU testing is on.
        reason = f"skipIfXpu: {msg}"

        @wraps(fn)
        def wrapper(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Raise SkipTest on XPU, otherwise call the test and return
            # its result.
            if TEST_XPU:
                raise unittest.SkipTest(reason)
            else:
                return fn(*args, **kwargs)
        return wrapper
    if func:
        return dec_fn(func)
    return dec_fn

def skipIfMPS(fn):
    # [FUNCTION DOC] Skip decorator for MPS. A class is marked skipped if MPS is available. A
    # function with a device parameter is skipped only when the device of the test instance is
    # mps; a function without one is skipped whenever MPS is available, and a warning is emitted
    # suggesting device generic tests. Returns the class or wrapped function.
    reason = "test doesn't currently work with MPS"
    # Class-level skip falls back to the global TEST_MPS check; the wrapper
    # below inspects args[0].device_type, which is only available per-method.
    if isinstance(fn, type):
        if TEST_MPS:
            fn.__unittest_skip__ = True  # type: ignore[attr-defined]
            fn.__unittest_skip_why__ = reason  # type: ignore[attr-defined]
        return fn

    sig = inspect.signature(fn)
    has_device_arg = "device" in sig.parameters

    if not has_device_arg:
        warnings.warn(
            f"skipIfMPS applied to {fn.__qualname__} which has no 'device' parameter. "
            "Consider using device-generic tests with instantiate_device_type_tests instead.",
            stacklevel=2,
        )

    @wraps(fn)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Skip (raise SkipTest) when the instance device type is mps, or
        # when no device parameter exists and MPS is available; otherwise call the test.
        if has_device_arg:
            # For device-generic tests, only skip when actually running on MPS
            slf = args[0] if args else None
            if slf is not None:
                device_type = getattr(slf, "device_type", None) or getattr(
                    slf, "device", None
                )
                if isinstance(device_type, str) and device_type == "mps":
                    raise unittest.SkipTest(reason)
        elif TEST_MPS:
            raise unittest.SkipTest(reason)
        return fn(*args, **kwargs)

    return wrapper


def skipIfHpu(fn):
    # [FUNCTION DOC] Skip decorator for HPU builds, applicable to functions and classes.
    return lazy_skip_if(lambda: TEST_HPU, "test doesn't currently work with HPU")(fn)

def skipIfHpu_BUGGY(fn):
    # [FUNCTION DOC] Deprecated variant of skipIfHpu kept for migration; only wraps functions.
    """Old skipIfHpu that silently drops classes from discovery. Migrate to skipIfHpu."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Raise SkipTest if HPU testing is on, otherwise call the test
        # (the return value is not propagated).
        if TEST_HPU:
            raise unittest.SkipTest("test doesn't currently work with HPU")
        else:
            fn(*args, **kwargs)
    return wrapper

def getRocmVersion() -> tuple[int, int]:
    # [FUNCTION DOC] Return the ROCm version as a (major, minor) tuple using the CUDA test
    # utilities. The import is done inside the function to avoid circular imports.
    from torch.testing._internal.common_cuda import _get_torch_rocm_version
    rocm_version = _get_torch_rocm_version()
    return (rocm_version[0], rocm_version[1])

# Skips a test on CUDA if ROCm is available and its version is lower than requested.
def skipIfRocmVersionLessThan(version=None):
    # [FUNCTION DOC] Return a lazy skip decorator that skips on ROCm when the ROCm version is
    # lower than the given version tuple (or when no version is given).
    def _should_skip():
        # [FUNCTION DOC] HELPER: Condition for the decorator: false on non-ROCm builds,
        # otherwise true if the ROCm version is lower than the requested one or none was given.
        if not TEST_WITH_ROCM:
            return False
        rocm_version_tuple = getRocmVersion()
        return (
            rocm_version_tuple is None
            or version is None
            or rocm_version_tuple < tuple(version)
        )
    return lazy_skip_if(_should_skip, f"ROCm version less than {version} required")

# Skips a test on ROCm if the version is at least the given major.minor tuple.
# Use for failures introduced in a ROCm release that pass on older versions,
# e.g. skipIfRocmVersionAtLeast([7, 14]) skips on 7.14+ but runs on 7.2.x.
# Built on lazy_skip_if so it works on both test methods and test classes.
def skipIfRocmVersionAtLeast(version=None):
    # [FUNCTION DOC] Return a lazy skip decorator that skips on ROCm when the ROCm version is at
    # least the given major and minor tuple. Works for methods and classes.
    def _should_skip():
        # [FUNCTION DOC] HELPER: Condition for the decorator: true only on ROCm with a version
        # at least the requested one.
        if not TEST_WITH_ROCM:
            return False
        rocm_version_tuple = getRocmVersion()
        return (
            rocm_version_tuple is not None
            and version is not None
            and rocm_version_tuple >= tuple(version)
        )
    return lazy_skip_if(_should_skip, f"ROCm version at least {version}: known failure")

def skipIfNotMiopenSuggestNHWC(fn):
    # [FUNCTION DOC] Skip the test unless MIOpen NHWC suggestion is enabled through
    # the environment.
    return lazy_skip_if(
        lambda: not TEST_WITH_MIOPEN_SUGGEST_NHWC,
        "test doesn't currently work without MIOpen NHWC activation",
    )(fn)

def skipIfWindows(func=None, *, msg="test doesn't currently work on the Windows stack"):
    # [FUNCTION DOC] Skip decorator for Windows. Usable bare or with a msg keyword.
    decorator = lazy_skip_if(lambda: IS_WINDOWS, f"skipIfWindows: {msg}")
    return decorator(func) if func is not None else decorator

def skipIfWindowsXPU(func=None, *, msg="test doesn't currently work on the Windows stack"):
    # [FUNCTION DOC] Skip decorator for Windows when XPU is available. Usable bare or with a
    # msg keyword.
    decorator = lazy_skip_if(
        lambda: IS_WINDOWS and torch.xpu.is_available(), f"skipIfWindowsXPU: {msg}"
    )
    return decorator(func) if func is not None else decorator

requires_cuda_python_bindings = unittest.skipUnless(TEST_CUDA_PYTHON_BINDINGS, "requires cuda-python (cuda.bindings)")

def requires_cuda_p2p_access():
    # [FUNCTION DOC] Return a skip decorator (pass in sandcastle variant) that skips tests
    # unless there are at least two CUDA devices with compute capability 8.0 or above and all
    # device pairs can access each other as peers. Queries CUDA at call time.
    cuda_p2p_access_available = (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (8, 0)
        and torch.cuda.device_count() >= 2
    )
    num_devices = torch.cuda.device_count()
    for i in range(num_devices - 1):
        for j in range(i + 1, num_devices):
            if not torch.cuda.can_device_access_peer(i, j):
                cuda_p2p_access_available = False
                break
        if not cuda_p2p_access_available:
            break

    return skip_but_pass_in_sandcastle_if(
        not cuda_p2p_access_available,
        "cuda p2p access is not available",
    )

# Reverts the linalg backend back to default to make sure potential failures in one
# test do not affect other tests
def setLinalgBackendsToDefaultFinally(fn):
    # [FUNCTION DOC] Decorator that runs the test and afterwards restores the preferred linalg
    # library backend to its earlier value, even on failure. Return value not propagated.
    @wraps(fn)
    def _fn(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Remember the preferred linalg backend, run the test, and
        # restore it.
        _preferred_backend = torch.backends.cuda.preferred_linalg_library()
        try:
            fn(*args, **kwargs)
        finally:
            torch.backends.cuda.preferred_linalg_library(_preferred_backend)
    return _fn


# Reverts the blas backend back to default to make sure potential failures in one
# test do not affect other tests
def setBlasBackendsToDefaultFinally(fn):
    # [FUNCTION DOC] Decorator that runs the test and afterwards restores the preferred BLAS
    # library backend and, if CUDA is built, resets cuBLAS workspaces. Return value dropped.
    @wraps(fn)
    def _fn(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Remember the preferred BLAS backend, run the test, then restore
        # it and clear cuBLAS and cuBLASLt workspaces when CUDA is built.
        _preferred_backend = torch.backends.cuda.preferred_blas_library()
        try:
            fn(*args, **kwargs)
        finally:
            torch.backends.cuda.preferred_blas_library(_preferred_backend)
            if torch.backends.cuda.is_built():
                torch._C._cuda_resetCublasWorkspaceSize()
                torch._C._cuda_resetCublasLtWorkspaceSize()
                torch._C._cuda_clearCublasWorkspaces()
    return _fn

def setSdpaBackendsToDefaultFinally(fn):
    # [FUNCTION DOC] Decorator that runs the test and afterwards restores the preferred ROCm
    # flash attention backend. Return value is dropped.
    @wraps(fn)
    def _fn(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Remember the preferred ROCm flash attention library, run the
        # test, and restore it.
        _preferred_backend = torch.backends.cuda.preferred_rocm_fa_library()
        try:
            fn(*args, **kwargs)
        finally:
            torch.backends.cuda.preferred_rocm_fa_library(_preferred_backend)
    return _fn

# Context manager for setting deterministic flag and automatically
# resetting it to its original value
class DeterministicGuard:
    # [CLASS DOC] Context manager that sets deterministic algorithm mode (plus warn only and
    # uninitialized memory fill flags) on entry and restores the previous state on exit. State:
    # the three requested settings and, while active, a guard holding the previous state.
    def __init__(self, deterministic, *, warn_only=False, fill_uninitialized_memory=True):
        # [FUNCTION DOC] Store the requested deterministic flag, warn only flag and fill
        # uninitialized memory flag.
        self.deterministic = deterministic
        self.warn_only = warn_only
        self.fill_uninitialized_memory = fill_uninitialized_memory

    @classmethod
    def _current_state(cls):
        # [FUNCTION DOC] HELPER: Class method that returns a guard describing the currently
        # active settings of deterministic mode, warn only, and fill of uninitialized memory.
        return cls(
            torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            fill_uninitialized_memory=torch.utils.deterministic.fill_uninitialized_memory,  # type: ignore[attr-defined]
        )

    def _update(self):
        # [FUNCTION DOC] HELPER: Apply this guard's settings to torch (deterministic algorithms
        # and the fill uninitialized memory global).
        torch.use_deterministic_algorithms(self.deterministic, warn_only=self.warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = self.fill_uninitialized_memory  # type: ignore[attr-defined]

    def __enter__(self):
        # [FUNCTION DOC] Save the current deterministic settings and apply the requested ones.
        self._restore = self._current_state()
        self._update()

    def __exit__(self, exception_type, exception_value, traceback):
        # [FUNCTION DOC] Restore the saved settings. Exceptions from the body are
        # not suppressed.
        self._restore._update()

class AlwaysWarnTypedStorageRemoval:
    # [CLASS DOC] Context manager that sets the global flag to always warn about typed storage
    # removal and restores the previous value on exit. State: the requested boolean.
    def __init__(self, always_warn):
        # [FUNCTION DOC] Store the flag. Raises AssertionError if it is not a bool.
        if not isinstance(always_warn, bool):
            raise AssertionError(f"expected always_warn to be bool, got {type(always_warn)}")
        self.always_warn = always_warn

    def __enter__(self):
        # [FUNCTION DOC] Save the current always warn setting and set the requested one.
        self.always_warn_restore = torch.storage._get_always_warn_typed_storage_removal()
        torch.storage._set_always_warn_typed_storage_removal(self.always_warn)

    def __exit__(self, exception_type, exception_value, traceback):
        # [FUNCTION DOC] Restore the saved always warn setting.
        torch.storage._set_always_warn_typed_storage_removal(self.always_warn_restore)

# Context manager for setting cuda sync debug mode and reset it
# to original value
# we are not exposing it to the core because sync debug mode is
# global and thus not thread safe
class CudaSyncGuard:
    # [CLASS DOC] Context manager that sets the CUDA sync debug mode and restores the
    # previous mode on exit. Kept here rather than in core because the mode is global and not
    # thread safe.
    def __init__(self, sync_debug_mode):
        # [FUNCTION DOC] Store the requested sync debug mode.
        self.mode = sync_debug_mode

    def __enter__(self):
        # [FUNCTION DOC] Save the current sync debug mode and set the requested one.
        self.debug_mode_restore = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode(self.mode)

    def __exit__(self, exception_type, exception_value, traceback):
        # [FUNCTION DOC] Restore the saved sync debug mode.
        torch.cuda.set_sync_debug_mode(self.debug_mode_restore)

# Context manager for setting torch.__future__.set_swap_module_params_on_conversion
# and automatically resetting it to its original value
class SwapTensorsGuard:
    # [CLASS DOC] Context manager that sets the swap module parameters on conversion option
    # (unless None) and restores the previous value on exit.
    def __init__(self, use_swap_tensors):
        # [FUNCTION DOC] Store the requested value (True, False or None meaning
        # leave unchanged).
        self.use_swap_tensors = use_swap_tensors

    def __enter__(self):
        # [FUNCTION DOC] Save the current option and set the requested one when it is not None.
        self.swap_tensors_restore = torch.__future__.get_swap_module_params_on_conversion()
        if self.use_swap_tensors is not None:
            torch.__future__.set_swap_module_params_on_conversion(self.use_swap_tensors)

    def __exit__(self, exception_type, exception_value, traceback):
        # [FUNCTION DOC] Restore the saved option.
        torch.__future__.set_swap_module_params_on_conversion(self.swap_tensors_restore)

# This decorator can be used for API tests that call
# torch.use_deterministic_algorithms().  When the test is finished, it will
# restore the previous deterministic flag setting.
#
# If CUDA >= 10.2, this will set the environment variable
# CUBLAS_WORKSPACE_CONFIG=:4096:8 so that the error associated with that
# setting is not thrown during the test unless the test changes that variable
# on purpose. The previous CUBLAS_WORKSPACE_CONFIG setting will also be
# restored once the test is finished.
#
# Note that if a test requires CUDA to actually register the changed
# CUBLAS_WORKSPACE_CONFIG variable, a new subprocess must be created, because
# CUDA only checks the variable when the runtime initializes. Tests can be
# run inside a subprocess like so:
#
#   import subprocess, sys, os
#   script = '''
#   # Test code should go here
#   '''
#   try:
#       subprocess.check_output(
#           [sys.executable, '-c', script],
#           stderr=subprocess.STDOUT,
#           cwd=os.path.dirname(os.path.realpath(__file__)),
#           env=os.environ.copy())
#   except subprocess.CalledProcessError as e:
#       error_message = e.output.decode('utf-8')
#       # Handle exceptions raised by the subprocess here
#
def wrapDeterministicFlagAPITest(fn):
    # [FUNCTION DOC] Decorator for tests that change the deterministic algorithms flag. The
    # wrapped test runs inside a deterministic guard that restores the old flags afterwards and
    # inside a guard that sets the cuBLAS workspace configuration variable and restores or
    # removes it afterwards. The return value of the test is not propagated.
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Run the test within the deterministic and cuBLAS config guards.
        with DeterministicGuard(
                torch.are_deterministic_algorithms_enabled(),
                warn_only=torch.is_deterministic_algorithms_warn_only_enabled()):
            class CuBLASConfigGuard:
                # [CLASS DOC] HELPER: Context manager that sets the CUBLAS_WORKSPACE_CONFIG
                # environment variable to a fixed value and restores or deletes it on exit.
                cublas_var_name = 'CUBLAS_WORKSPACE_CONFIG'

                def __enter__(self):
                    # [FUNCTION DOC] HELPER: Remember the old value of the variable and set it
                    # to the fixed value.
                    self.cublas_config_restore = os.environ.get(self.cublas_var_name)
                    os.environ[self.cublas_var_name] = ':4096:8'

                def __exit__(self, exception_type, exception_value, traceback):
                    # [FUNCTION DOC] HELPER: Restore the old value, or delete the variable if it
                    # was unset before and is still set.
                    cur_cublas_config = os.environ.get(self.cublas_var_name)
                    if self.cublas_config_restore is None:
                        if cur_cublas_config is not None:
                            del os.environ[self.cublas_var_name]
                    else:
                        os.environ[self.cublas_var_name] = self.cublas_config_restore
            with CuBLASConfigGuard():
                fn(*args, **kwargs)
    return wrapper

# This decorator can be used for API tests that want to safely call
# torch.__future__.set_swap_module_params_on_conversion.  `swap` can be set to
# True, False or None where None indicates that the context manager does not
# set the flag. When the test is finished, it will restore the previous swap
# flag setting.
def wrapSwapTensorsTest(swap=None):
    # [FUNCTION DOC] Return a decorator that runs a test with the swap module parameters on
    # conversion option set to the given value (True, False or None for unchanged) and restores
    # it afterwards.
    def dec_fn(fn):
        # [FUNCTION DOC] HELPER: Wrap the test with the swap tensors guard.
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Run the test inside the guard (return value
            # not propagated).
            with SwapTensorsGuard(swap):
                fn(*args, **kwargs)
        return wrapper
    return dec_fn

# test parametrizer for swapping
class swap(_TestParametrizer):
    # [CLASS DOC] Parametrizer that runs a test once per swap value, wrapping the test with the
    # swap tensors decorator and naming it with the value. State: the list of swap values.
    def __init__(self, swap_values):
        # [FUNCTION DOC] Initialize the base class and store the swap values.
        super().__init__()
        self.swap_values = swap_values

    def _parametrize_test(self, test, generic_cls, device_cls):
        # [FUNCTION DOC] Generator yielding, for each swap value, the test wrapped for that
        # swap value, a name suffix, an empty parameter mapping, and a decorator function that
        # adds none.
        for swap in self.swap_values:
            yield wrapSwapTensorsTest(swap)(test), f'swap_{swap}', {}, lambda _: []

def skipIfCompiledWithoutNumpy(fn):
    # [FUNCTION DOC] Skip decorator: the test is skipped if numpy is missing or if converting a
    # small array to a tensor fails, which indicates a build without numpy support.
    # Even if the numpy module is present, if `USE_NUMPY=0` is used during the
    # build, numpy tests will fail
    numpy_support = TEST_NUMPY
    if numpy_support:
        try:
            torch.from_numpy(np.array([2, 2]))
        except RuntimeError:
            numpy_support = False
    return unittest.skipIf(not numpy_support, "PyTorch was compiled without numpy support")(fn)

def _test_function(fn, device):
    # [FUNCTION DOC] HELPER: Return a one argument test method that calls the given function
    # with the test object and the device.
    def run_test_function(self):
        # [FUNCTION DOC] HELPER: Call the captured function with the test object and device.
        return fn(self, device)
    return run_test_function

def skipIfNoXNNPACK(fn):
    # [FUNCTION DOC] Skip the test when XNNPACK is not enabled.
    return lazy_skip_if(
        lambda: not torch.backends.xnnpack.enabled,  # type: ignore[attr-defined]
        "XNNPACK must be enabled for these tests. Please build with USE_XNNPACK=1.",
    )(fn)

def skipIfNoLapack(fn):
    # [FUNCTION DOC] Skip the test when PyTorch was built without LAPACK.
    return lazy_skip_if(lambda: not torch._C.has_lapack, "PyTorch compiled without Lapack")(fn)

def skipIfNotRegistered(op_name, message):
    # [FUNCTION DOC] Returns a skip decorator unconditionally, saying Caffe2 is not compiled.
    # KNOWN ISSUE: the arguments are ignored and the test is always skipped, although the
    # docstring describes checking whether an operator is registered.
    """Wraps the decorator to hide the import of the `core`.

    Args:
        op_name: Check if this op is registered in `core._REGISTERED_OPERATORS`.
        message: message to fail with.

    Usage:
        @skipIfNotRegistered('MyOp', 'MyOp is not linked!')
            This will check if 'MyOp' is in the caffe2.python.core
    """
    return unittest.skip("Pytorch is compiled without Caffe2")

def skipIfNoSciPy(fn):
    # [FUNCTION DOC] Skip the test when SciPy is not installed.
    return lazy_skip_if(lambda: not TEST_SCIPY, "test require SciPy, but SciPy not found")(fn)

def skip_if_pytest(fn):
    # [FUNCTION DOC] Skip the test when running under pytest (detected by the current test
    # environment variable).
    return lazy_skip_if(
        lambda: "PYTEST_CURRENT_TEST" in os.environ, "does not work under pytest"
    )(fn)

def skipIfNoXPU(fn):
    # [FUNCTION DOC] Skip the test when XPU support is not available.
    return lazy_skip_if(lambda: not TEST_XPU, "test required PyTorched compiled with XPU")(fn)

def skipIfCachingAllocatorDisabled(fn):
    # [FUNCTION DOC] Skip the test when CUDA is available and the caching allocator is disabled
    # (at runtime or through environment variables). CPU only builds are not skipped.
    """Skip if the CUDA/HIP caching allocator is not active. Covers both the
    runtime toggle (``torch.cuda.caching_allocator_enable(False)``) and the
    env-var bypass (``PYTORCH_NO_CUDA_MEMORY_CACHING`` /
    ``PYTORCH_NO_HIP_MEMORY_CACHING``). The CPU-only case (no CUDA built)
    is treated as "allocator irrelevant", so the test is allowed to run."""
    return lazy_skip_if(
        lambda: torch.cuda.is_available()
        and not torch._C._cuda_cudaCachingAllocator_is_enabled(),
        "requires the CUDA/HIP caching allocator (current allocator is uncached)",
    )(fn)

def slowTest(fn):
    # [FUNCTION DOC] Decorator for slow tests. The wrapped test raises SkipTest unless slow
    # tests are enabled, and carries a slow test marker in its dictionary. The return value of
    # the test is not propagated.
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Raise SkipTest if slow tests are not enabled, else call
        # the test.
        if not TEST_WITH_SLOW:
            raise unittest.SkipTest("test is slow; run with PYTORCH_TEST_WITH_SLOW to enable test")
        else:
            fn(*args, **kwargs)
    wrapper.__dict__['slow_test'] = True
    return wrapper


def slowTestIf(condition):
    # [FUNCTION DOC] Return the slowTest decorator if the condition is true, otherwise an
    # identity decorator.
    return slowTest if condition else lambda fn: fn


def skipCUDAMemoryLeakCheckIf(condition):
    # [FUNCTION DOC] Return a decorator that disables the CUDA memory leak check on a test when
    # the condition is true, only if the check is currently enabled for it. Sets an attribute on
    # the function.
    def dec(fn):
        # [FUNCTION DOC] HELPER: Update the leak check attribute on the function and return it.
        if getattr(fn, '_do_cuda_memory_leak_check', True):  # if current True
            fn._do_cuda_memory_leak_check = not condition
        return fn
    return dec

def skipCUDANonDefaultStreamIf(condition):
    # [FUNCTION DOC] Return a decorator that disables running the test on a non default CUDA
    # stream when the condition is true, only if currently enabled. Sets an attribute.
    def dec(fn):
        # [FUNCTION DOC] HELPER: Update the non default stream attribute and return
        # the function.
        if getattr(fn, '_do_cuda_non_default_stream', True):  # if current True
            fn._do_cuda_non_default_stream = not condition
        return fn
    return dec

def suppress_warnings(fn):
    # [FUNCTION DOC] Decorator that runs the test with all warnings ignored.
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Run the test inside a warnings catch block with the
        # ignore filter.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn(*args, **kwargs)
    return wrapper


def to_gpu(obj, type_map=None):
    # [FUNCTION DOC] Recursively copy an object to CUDA. Leaf tensors are copied to the CUDA
    # device with optional dtype mapping and the same requires_grad flag; storages are copied;
    # lists and tuples are processed element by element; other objects are deep copied. Raises
    # AssertionError if a tensor is not a leaf.
    if type_map is None:
        type_map = {}
    if isinstance(obj, torch.Tensor):
        if not obj.is_leaf:
            raise AssertionError("expected obj to be a leaf tensor")
        t = type_map.get(obj.dtype, obj.dtype)
        with torch.no_grad():
            res = obj.to(dtype=t, device="cuda", copy=True)
            res.requires_grad = obj.requires_grad
        return res
    elif torch.is_storage(obj):
        return obj.new().resize_(obj.size()).copy_(obj)  # type: ignore[attr-defined, union-attr]
    elif isinstance(obj, list):
        return [to_gpu(o, type_map) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(to_gpu(o, type_map) for o in obj)
    else:
        return deepcopy(obj)


def get_function_arglist(func):
    # [FUNCTION DOC] Return the list of positional argument names of a function.
    return inspect.getfullargspec(func).args


def set_rng_seed(seed=None):
    # [FUNCTION DOC] Seed the torch, python and (if available) numpy random number generators
    # with the given seed, or the module default seed when None. Mutates global RNG state.
    if seed is None:
        seed = SEED
    torch.manual_seed(seed)
    random.seed(seed)
    if TEST_NUMPY:
        np.random.seed(seed)


@contextlib.contextmanager
def set_default_dtype(dtype):
    # [FUNCTION DOC] Context manager that sets the torch default dtype for the body and restores
    # the previous default afterwards.
    saved_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(saved_dtype)


class set_default_dtype_if_supported:
    # [CLASS DOC] Decorator class like set_default_dtype that only sets the dtype if the primary
    # device of the test instance supports it (checked by trying to create an empty tensor,
    # catching TypeError). The decorated method must belong to a class instantiated by the
    # device type test machinery, since it calls get_primary_device. State: the dtype.
    """Decorator like set_default_dtype, but only sets the dtype if the device
    supports it.

    Calls the test object's `DeviceTypeTestBase.get_primary_device()` method at
    call time, so this only works on test classes that have had
    `instantiate_device_type_tests` applied to them.
    """

    def __init__(self, dtype):
        # [FUNCTION DOC] Store the dtype.
        self._dtype = dtype

    def __call__(self, fn):
        # [FUNCTION DOC] Wrap the test method. Returns the wrapper.
        @functools.wraps(fn)
        def wrapper(test_self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Check whether the primary device supports the dtype, set
            # it as the default if so, run the test and return its result, and restore the
            # old default.
            device = test_self.get_primary_device()
            dtype_supported = True
            try:
                torch.empty(0, device=device, dtype=self._dtype)
            except TypeError:
                dtype_supported = False

            saved_dtype = None

            if dtype_supported:
                saved_dtype = torch.get_default_dtype()
                torch.set_default_dtype(self._dtype)

            try:
                return fn(test_self, *args, **kwargs)
            finally:
                if saved_dtype is not None:
                    torch.set_default_dtype(saved_dtype)

        return wrapper


@contextlib.contextmanager
def set_default_tensor_type(tensor_type):
    # [FUNCTION DOC] Context manager that sets the default tensor type for the body and restores
    # the previous one afterwards.
    saved_tensor_type = torch.tensor([]).type()
    torch.set_default_tensor_type(tensor_type)
    try:
        yield
    finally:
        torch.set_default_tensor_type(saved_tensor_type)

def iter_indices(tensor):
    # [FUNCTION DOC] Return an iterable of indices of a tensor: an empty range for zero
    # dimensions, a range for one dimension, otherwise the cartesian product of the ranges.
    if tensor.dim() == 0:
        return range(0)
    if tensor.dim() == 1:
        return range(tensor.size(0))
    return product(*(range(s) for s in tensor.size()))


def is_iterable(obj):
    # [FUNCTION DOC] Return True if the object can be iterated, False if iterating
    # raises TypeError.
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def is_iterable_of_tensors(iterable, include_empty=False):
    # [FUNCTION DOC] Return True if the argument is an iterable (not a tensor itself) whose
    # elements are all tensors. An empty iterable returns the include_empty value; a non
    # iterable returns False.
    """ Returns True if iterable is an iterable of tensors and False o.w.

        If the iterable is empty, the return value is :attr:`include_empty`
    """
    # Tensor itself is iterable so we check this first
    if isinstance(iterable, torch.Tensor):
        return False

    try:
        if len(iterable) == 0:
            return include_empty

        for t in iter(iterable):
            if not isinstance(t, torch.Tensor):
                return False

    except TypeError:
        return False

    return True


class CudaNonDefaultStream:
    # [CLASS DOC] Context manager that makes every CUDA device use a fresh non default stream on
    # entry (after synchronizing the previous current streams) and restores the previous streams
    # on exit. State: the list of previous streams per device.
    def __enter__(self):
        # [FUNCTION DOC] Save current streams of each device, create and set new streams, and
        # restore the current device.
        # Before starting CUDA test save currently active streams on all
        # CUDA devices and set new non default streams to all CUDA devices
        # to ensure CUDA tests do not use default stream by mistake.
        beforeDevice = torch.cuda.current_device()
        self.beforeStreams = []
        for d in range(torch.cuda.device_count()):
            self.beforeStreams.append(torch.cuda.current_stream(d))
            deviceStream = torch.cuda.Stream(device=d)
            self.beforeStreams[-1].synchronize()
            torch._C._cuda_setStream(stream_id=deviceStream.stream_id,
                                     device_index=deviceStream.device_index,
                                     device_type=deviceStream.device_type)
        torch._C._cuda_setDevice(beforeDevice)

    def __exit__(self, exc_type, exc_value, traceback):
        # [FUNCTION DOC] Restore the saved stream of each device and the current device.
        # After completing CUDA test load previously active streams on all
        # CUDA devices.
        beforeDevice = torch.cuda.current_device()
        for d in range(torch.cuda.device_count()):
            torch._C._cuda_setStream(stream_id=self.beforeStreams[d].stream_id,
                                     device_index=self.beforeStreams[d].device_index,
                                     device_type=self.beforeStreams[d].device_type)
        torch._C._cuda_setDevice(beforeDevice)

class CudaMemoryLeakCheck:
    # [CLASS DOC] Context manager that detects CUDA memory leaks around a test. On entry it
    # records caching allocator and driver memory per device after collecting garbage if needed;
    # on exit (only when no exception occurred) it compares memory, re-checks several times,
    # warns if only the allocator reports growth and raises RuntimeError if the driver confirms.
    # State: the test case, a name, and the recorded before values.
    def __init__(self, testcase, name=None):
        # [FUNCTION DOC] Store the test case and a display name (the test id by default) and
        # initialize the CUDA context and RNG so the first test does not look like a leak.
        self.name = testcase.id() if name is None else name
        self.testcase = testcase

        # initialize context & RNG to prevent false positive detections
        # when the test is the first to initialize those
        from torch.testing._internal.common_cuda import initialize_cuda_context_rng
        initialize_cuda_context_rng()

    # Stores CUDA memory data provided by PyTorch's caching allocator and
    #   the CUDA driver.
    #
    # NOTE: The undocumented torch.cuda.mem_get_info() returns
    #   (#free bytes, #total bytes available) on the GPU
    def __enter__(self):
        # [FUNCTION DOC] Collect garbage and empty caches if any memory is allocated, then
        # record the caching allocator and driver allocated memory of every device.
        self.caching_allocator_befores = []
        self.driver_befores = []

        # Performs a gc if required (required if any CUDA memory is held)
        num_devices = torch.cuda.device_count()
        for i in range(num_devices):
            caching_allocator_mem_allocated = torch.cuda.memory_allocated(i)
            # NOTE: gc is based exclusively on caching allocator memory
            #   because the driver will always have some bytes in use (context size?)
            if caching_allocator_mem_allocated > 0:
                gc.collect()
                torch._C._cuda_clearCublasWorkspaces()
                torch.cuda.empty_cache()
                break

        # Acquires caching allocator and driver statistics before the test is run
        for i in range(num_devices):
            self.caching_allocator_befores.append(torch.cuda.memory_allocated(i))
            bytes_free, bytes_total = torch.cuda.mem_get_info(i)
            driver_mem_allocated = bytes_total - bytes_free
            self.driver_befores.append(driver_mem_allocated)

    def __exit__(self, exc_type, exc_value, traceback):
        # [FUNCTION DOC] Skip when an exception is propagating. Otherwise call the test case
        # hook before the check, collect garbage, compare the allocator memory per device, and
        # if growth persists after cleanup and three repeated queries, warn (allocator only) or
        # raise RuntimeError (confirmed by the driver) naming the test and the memory values.
        # KNOWN ISSUE: a comment mentions an exception on ROCm described below, but no ROCm
        # specific handling exists in this method.
        # Don't check for leaks if an exception was thrown
        if exc_type is not None:
            return

        self.testcase.before_cuda_memory_leak_check()
        gc.collect()
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()

        # Compares caching allocator before/after statistics
        # An increase in allocated memory is a discrepancy indicating a possible
        #   memory leak
        discrepancy_detected = False
        num_devices = torch.cuda.device_count()
        for i in range(num_devices):
            # avoid counting cublasWorkspace allocations
            torch._C._cuda_clearCublasWorkspaces()
            caching_allocator_mem_allocated = torch.cuda.memory_allocated(i)

            if caching_allocator_mem_allocated > self.caching_allocator_befores[i]:
                discrepancy_detected = True
                break

        # Short-circuits if no discrepancy detected
        if not discrepancy_detected:
            return

        # Validates the discrepancy persists after garbage collection and
        #   is confirmed by the driver API

        # NOTE: driver API discrepancies alone are ignored because with the jiterator
        #   some tests may permanently increase the CUDA context size and
        #   that will appear as a driver memory leak but is the expected behavior.

        # GCs and clears the cache
        gc.collect()
        torch.cuda.empty_cache()

        for i in range(num_devices):

            discrepancy_detected = True

            # Query memory multiple items to ensure leak was not transient
            for _ in range(3):
                caching_allocator_mem_allocated = torch.cuda.memory_allocated(i)
                bytes_free, bytes_total = torch.cuda.mem_get_info(i)
                driver_mem_allocated = bytes_total - bytes_free

                caching_allocator_discrepancy = False
                driver_discrepancy = False

                if caching_allocator_mem_allocated > self.caching_allocator_befores[i]:
                    caching_allocator_discrepancy = True

                if driver_mem_allocated > self.driver_befores[i]:
                    driver_discrepancy = True

                if not (caching_allocator_discrepancy or driver_discrepancy):
                    # Leak was false positive, exit loop
                    discrepancy_detected = False
                    break

            if not discrepancy_detected:
                continue

            if caching_allocator_discrepancy and not driver_discrepancy:  # type: ignore[possibly-undefined]
                # Just raises a warning if the leak is not validated by the
                #   driver API
                # NOTE: this may be a problem with how the caching allocator collects its
                #   statistics or a leak too small to trigger the allocation of an
                #   additional block of memory by the CUDA driver
                msg = ("CUDA caching allocator reports a memory leak not "  # type: ignore[possibly-undefined]
                       f"verified by the driver API in {self.name}! "
                       f"Caching allocator allocated memory was {self.caching_allocator_befores[i]} "
                       f"and is now reported as {caching_allocator_mem_allocated} "  # type: ignore[possibly-undefined]
                       f"on device {i}. "
                       f"CUDA driver allocated memory was {self.driver_befores[i]} and is now {driver_mem_allocated}.")  # type: ignore[possibly-undefined]
                warnings.warn(msg, stacklevel=2)
            elif caching_allocator_discrepancy and driver_discrepancy:  # type: ignore[possibly-undefined]
                # A caching allocator discrepancy validated by the driver API is a
                #   failure (except on ROCm, see below)
                msg = (f"CUDA driver API confirmed a leak in {self.name}! "  # type: ignore[possibly-undefined]
                       f"Caching allocator allocated memory was {self.caching_allocator_befores[i]} "
                       f"and is now reported as {caching_allocator_mem_allocated} "  # type: ignore[possibly-undefined]
                       f"on device {i}. "
                       f"CUDA driver allocated memory was {self.driver_befores[i]} and is now {driver_mem_allocated}.")  # type: ignore[possibly-undefined]

                raise RuntimeError(msg)

@contextmanager
def skip_exception_type(exc_type):
    # [FUNCTION DOC] Context manager that converts the given exception type raised in the body
    # into unittest.SkipTest with a not implemented message.
    try:
        yield
    except exc_type as e:
        raise unittest.SkipTest(f"not implemented: {e}") from e

@contextmanager
def print_repro_on_failure(repro_parts):
    # [FUNCTION DOC] Context manager that, on any exception other than SkipTest, appends
    # reproduction instructions to the exception message (including the failing sample input
    # index when known), sends a signpost record, and re-raises. repro_parts is a list of
    # command parts.
    try:
        yield
    except unittest.SkipTest:
        raise
    except Exception as e:
        # Get the index of the sample input that failed the test if possible.
        sample_isolation_prefix = ""
        tracked_input = getattr(e, "_tracked_input", None)
        if tracked_input is not None:
            sample_isolation_prefix = f"PYTORCH_OPINFO_SAMPLE_INPUT_INDEX={tracked_input.index}"

        repro_str = " ".join(filter(None, (sample_isolation_prefix, *repro_parts)))

        open_source_signpost(
            subsystem="test_repros",
            name="test_failure",
            parameters=json.dumps(
                {
                    "repro": " ".join(filter(None, (sample_isolation_prefix, *repro_parts))),
                }
            ),
        )

        repro_msg = f"""
To execute this test, run the following from the base repo dir:
    {repro_str}

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0"""

        # NB: Hacking the exception args is the cleanest way I've found to append
        # failure reproduction info without poisoning the stack trace.
        if len(e.args) >= 1:
            e.args = (f"{e.args[0]}\n{repro_msg}", *e.args[1:])
        raise

#  "min_satisfying_examples" setting has been deprecated in hypothesis
#  3.56.0 and removed in hypothesis 4.x
try:
    import hypothesis

    def settings(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper over hypothesis settings that drops the removed
        # min_satisfying_examples option on hypothesis versions at or above 3.56.0.
        if 'min_satisfying_examples' in kwargs and hypothesis.version.__version_info__ >= (3, 56, 0):
            kwargs.pop('min_satisfying_examples')
        return hypothesis.settings(*args, **kwargs)


    hypothesis.settings.register_profile(
        "pytorch_ci",
        settings(
            derandomize=True,
            suppress_health_check=[hypothesis.HealthCheck.too_slow],
            database=None,
            max_examples=50,
            verbosity=hypothesis.Verbosity.normal))
    hypothesis.settings.register_profile(
        "dev",
        settings(
            suppress_health_check=[hypothesis.HealthCheck.too_slow],
            database=None,
            max_examples=10,
            verbosity=hypothesis.Verbosity.normal))
    hypothesis.settings.register_profile(
        "debug",
        settings(
            suppress_health_check=[hypothesis.HealthCheck.too_slow],
            database=None,
            max_examples=1000,
            verbosity=hypothesis.Verbosity.verbose))

    hypothesis.settings.load_profile(
        "pytorch_ci" if IS_CI else os.getenv('PYTORCH_HYPOTHESIS_PROFILE', 'dev')
    )
except ImportError:
    warnings.warn('Fail to import hypothesis in common_utils, tests are not derandomized', ImportWarning, stacklevel=2)

# Used in check_if_enable to see if a test method should be disabled by an issue,
# sanitizes a test method name from appended suffixes by @dtypes parametrization.
# e.g., an issue with title "DISABLED test_bitwise_ops (__main__.TestBinaryUfuncs)" should
# disabled ALL parametrized test_bitwise_ops tests, such test_bitwise_ops_cuda_int32
def remove_device_and_dtype_suffixes(test_name: str) -> str:
    # [FUNCTION DOC] Strip a trailing dtype suffix, and a device suffix just before it, from a
    # generated test name, so a disabled issue can match all parametrized variants. Imports the
    # device test base machinery lazily to avoid a circular import.
    # import statement is localized to avoid circular dependency issues with common_device_type.py
    from torch.testing._internal.common_device_type import get_device_type_test_bases
    device_suffixes = [x.device_type for x in get_device_type_test_bases()]
    dtype_suffixes = [str(dt)[len("torch."):] for dt in get_all_dtypes()]

    test_name_chunks = test_name.split("_")
    if len(test_name_chunks) > 0 and test_name_chunks[-1] in dtype_suffixes:
        if len(test_name_chunks) > 1 and test_name_chunks[-2] in device_suffixes:
            return "_".join(test_name_chunks[0:-2])
        return "_".join(test_name_chunks[0:-1])
    return test_name


def check_if_enable(test: unittest.TestCase):
    # [FUNCTION DOC] Decide whether a test should be skipped. Marks the test slow and skips if
    # slow tests are disabled and the test appears in the slow test list. Outside Sandcastle,
    # checks the disabled tests list for matching test and class names, filters platforms
    # (printing a message for unknown names), and raises SkipTest if disabled for the current
    # platform (or, in rerun mode, if not disabled). Also skips fast tests when only slow tests
    # are requested.
    classname = str(test.__class__).split("'")[1].split(".")[-1]
    sanitized_testname = remove_device_and_dtype_suffixes(test._testMethodName)

    def matches_test(target: str):
        # [FUNCTION DOC] HELPER: Check whether a disabled test entry (name and class in
        # parentheses) matches this test, allowing class name prefixes and sanitized method
        # names. Returns False for malformed entries.
        target_test_parts = target.split()
        if len(target_test_parts) < 2:
            # poorly formed target test name
            return False
        target_testname = target_test_parts[0]
        target_classname = target_test_parts[1][1:-1].split(".")[-1]
        # if test method name or its sanitized version exactly matches the disabled
        # test method name AND allow non-parametrized suite names to disable
        # parametrized ones (TestSuite disables TestSuiteCPU)
        return classname.startswith(target_classname) and (target_testname in (test._testMethodName, sanitized_testname))

    if any(matches_test(x) for x in slow_tests_dict):
        getattr(test, test._testMethodName).__dict__['slow_test'] = True
        if not TEST_WITH_SLOW:
            raise unittest.SkipTest("test is slow; run with PYTORCH_TEST_WITH_SLOW to enable test")

    if not IS_SANDCASTLE:
        should_skip = False
        skip_msg = ""

        for disabled_test, (issue_url, platforms) in disabled_tests_dict.items():
            if matches_test(disabled_test):
                platform_to_conditional: dict = {
                    "mac": IS_MACOS,
                    "macos": IS_MACOS,
                    "win": IS_WINDOWS,
                    "windows": IS_WINDOWS,
                    "linux": IS_LINUX,
                    "rocm": TEST_WITH_ROCM,
                    "xpu": TEST_XPU,
                    "asan": TEST_WITH_ASAN,
                    "dynamo": TEST_WITH_TORCHDYNAMO,
                    "dynamo_wrapped": TEST_WITH_TORCHDYNAMO,
                    "inductor": TEST_WITH_TORCHINDUCTOR,
                    "slow": TEST_WITH_SLOW,
                }

                invalid_platforms = list(filter(lambda p: p not in platform_to_conditional, platforms))
                if len(invalid_platforms) > 0:
                    invalid_plats_str = ", ".join(invalid_platforms)
                    valid_plats = ", ".join(platform_to_conditional.keys())

                    print(f"Test {disabled_test} is disabled for some unrecognized ",
                          f"platforms: [{invalid_plats_str}]. Please edit issue {issue_url} to fix the platforms ",
                          'assigned to this flaky test, changing "Platforms: ..." to a comma separated ',
                          f"subset of the following (or leave it blank to match all platforms): {valid_plats}")

                    # Sanitize the platforms list so that we continue to disable the test for any valid platforms given
                    platforms = list(filter(lambda p: p in platform_to_conditional, platforms))

                if platforms == [] or any(platform_to_conditional[platform] for platform in platforms):
                    should_skip = True
                    skip_msg = f"Test is disabled because an issue exists disabling it: {issue_url}" \
                        f" for {'all' if platforms == [] else ''}platform(s) {', '.join(platforms)}. " \
                        "If you're seeing this on your local machine and would like to enable this test, " \
                        "please make sure CI is not set and you are not using the flag --import-disabled-tests."
                    break

        if should_skip and not RERUN_DISABLED_TESTS:
            # Skip the disabled test when not running under --rerun-disabled-tests verification mode
            raise unittest.SkipTest(skip_msg)

        if not should_skip and RERUN_DISABLED_TESTS:
            # Probably test has disable issue but not for this platform
            skip_msg = "Test is enabled but --rerun-disabled-tests verification mode is set, so only" \
                " disabled tests are run"
            raise unittest.SkipTest(skip_msg)

    if TEST_SKIP_FAST:
        if hasattr(test, test._testMethodName) and not getattr(test, test._testMethodName).__dict__.get('slow_test', False):
            raise unittest.SkipTest("test is fast; we disabled it with PYTORCH_TEST_SKIP_FAST")


# `TestCase.assertEqual` is very permissive and coerced the inputs into a format that could be compared. This is very
# convenient when writing tests, but not so much while reviewing them. By default, the comparison `Pair` framework of
# `torch.testing._comparison.are_equal`, used for example by the public testing function
# `torch.testing.assert_close`, is more strict. In order to use the same framework and thus reduce the divergence
# between internal and external comparison logic as much as possible, we define some "relaxed" pairs here. They only
# change the supported inputs, but the comparison logic is the same.
# TODO: Revisit the relaxed pairs and check how much work it is to fix the tests that would fail without the relaxation.

class RelaxedBooleanPair(BooleanPair):
    # [CLASS DOC] Comparison pair for boolean like values that, unlike the builtin pair, allows
    # one side to be a number or a single element tensor or array, provided the other side is a
    # bool. Used by TestCase.assertEqual.
    """Pair for boolean-like inputs.

    In contrast to the builtin :class:`BooleanPair`, this class also supports one input being a number or a single
    element tensor-like.
    """
    _supported_number_types = NumberPair(0, 0)._supported_types

    def _process_inputs(self, actual, expected, *, id):
        # [FUNCTION DOC] Verify that at least one input is a boolean and the other is a boolean,
        # number, tensor or array, calling the not supported hook otherwise, and return both
        # inputs converted to bool values.
        # We require only one of the inputs of the inputs to be a boolean and the other can also be a boolean, a
        # number, or a single element tensor or array, whereas in default BooleanPair both inputs have to be booleans.
        tensor_or_array_types: tuple[type, ...] = (torch.Tensor, np.ndarray)
        other_supported_types = (*self._supported_types, *self._supported_number_types, *tensor_or_array_types)
        if not (
            (isinstance(actual, self._supported_types) and isinstance(expected, other_supported_types))
            or (isinstance(expected, self._supported_types) and isinstance(actual, other_supported_types))
        ):
            self._inputs_not_supported()

        return [self._to_bool(input, id=id) for input in (actual, expected)]

    def _to_bool(self, bool_like, *, id):
        # [FUNCTION DOC] Convert a bool like value to bool: numpy numbers and Python numbers are
        # converted directly; tensors and arrays must have a single element (otherwise a
        # ValueError failure is reported); everything else uses the parent conversion.
        if isinstance(bool_like, np.number):
            return bool(bool_like.item())
        elif type(bool_like) in self._supported_number_types:
            return bool(bool_like)
        elif isinstance(bool_like, (torch.Tensor, np.ndarray)):
            numel = bool_like.numel() if isinstance(bool_like, torch.Tensor) else bool_like.size
            if numel > 1:
                self._fail(
                    ValueError,
                    f"Only single element tensor-likes can be compared against a boolean. "
                    f"Got {numel} elements instead.",
                    id=id
                )

            return bool(bool_like.item())
        else:
            return super()._to_bool(bool_like, id=id)


class RelaxedNumberPair(NumberPair):
    # [CLASS DOC] Comparison pair for numbers that also accepts a single element tensor or array
    # or an Enum on one side, ignores dtype checking, uses looser default tolerances and honors
    # tolerance overrides from the precision and tolerance decorators.
    """Pair for number-like inputs.

    In contrast to the builtin :class:`NumberPair`, this class also supports one input being a single element
    tensor-like or a :class:`enum.Enum`. (D)Type checks are disabled, meaning comparing 1 to 1.0 succeeds even when
    ``check_dtype=True`` is passed.

    In addition, this class uses looser default tolerances for :class:`float` and :class:`complex` inputs. Also
    supports overriding the absolute and relative tolerance through the ``@precisionOverride`` and
    ``@toleranceOverride`` decorators.
    """
    _TYPE_TO_DTYPE = {
        int: torch.int64,
        float: torch.float32,
        complex: torch.complex64,
    }

    def __init__(
            self, actual, expected, *, rtol_override=0.0, atol_override=0.0, check_dtype=None, **other_parameters
    ) -> None:
        # [FUNCTION DOC] Initialize the parent with dtype checking disabled, then raise the
        # relative and absolute tolerances to at least the override values.
        # KNOWN ISSUE: the check_dtype argument is accepted but ignored.
        super().__init__(actual, expected, check_dtype=False, **other_parameters)
        self.rtol = max(self.rtol, rtol_override)
        self.atol = max(self.atol, atol_override)

    def _process_inputs(self, actual, expected, *, id):
        # [FUNCTION DOC] Verify that one input is a number and the other is a number, tensor or
        # array, calling the not supported hook otherwise; return both converted to numbers.
        # We require only one of the inputs of the inputs to be a number and the other can also be a number or a single
        # element tensor or array, whereas in default NumberPair both inputs have to be numbers.
        tensor_or_array_types: tuple[type, ...] = (torch.Tensor, np.ndarray)
        other_supported_types = (*self._supported_types, *tensor_or_array_types)
        if not (
                (isinstance(actual, self._supported_types) and isinstance(expected, other_supported_types))
                or (isinstance(expected, self._supported_types) and isinstance(actual, other_supported_types))
        ):
            self._inputs_not_supported()

        return [self._to_number(input, id=id) for input in (actual, expected)]

    def _to_number(self, number_like, *, id):
        # [FUNCTION DOC] Convert a value to a Python number. Single element tensors and arrays
        # use their item (booleans become ints); multi element ones fail with ValueError.
        # Enums become ints. Other values use the parent conversion and must be a supported
        # number type.
        if isinstance(number_like, (torch.Tensor, np.ndarray)):
            numel = number_like.numel() if isinstance(number_like, torch.Tensor) else number_like.size
            if numel > 1:
                self._fail(
                    ValueError,
                    f"Only single element tensor-likes can be compared against a number. "
                    f"Got {numel} elements instead.",
                    id=id
                )
            number = number_like.item()
            if isinstance(number, bool):
                number = int(number)

            return number
        elif isinstance(number_like, Enum):
            return int(number_like)  # type: ignore[call-overload]
        else:
            number = super()._to_number(number_like, id=id)
            if type(number) not in self._TYPE_TO_DTYPE:
                self._inputs_not_supported()
            return number


class TensorOrArrayPair(TensorLikePair):
    # [CLASS DOC] Comparison pair for tensors and numpy arrays: only these two types are
    # accepted, both are converted to tensors, and mixing them is allowed. Supports tolerance
    # overrides from the precision decorators.
    """Pair for tensor-like inputs.

    On the one hand this class is stricter than the builtin :class:`TensorLikePair` since it only allows instances of
    :class:`torch.Tensor` and :class:`numpy.ndarray` rather than allowing any tensor-like than can be converted into a
    tensor. On the other hand this class is looser since it converts all inputs into tensors with no regard of their
    relationship, e.g. comparing a :class:`torch.Tensor` to :class:`numpy.ndarray` is fine.

    In addition, this class supports overriding the absolute and relative tolerance through the ``@precisionOverride``
    and ``@toleranceOverride`` decorators.
    """
    def __init__(self, actual, expected, *, rtol_override=0.0, atol_override=0.0, **other_parameters):
        # [FUNCTION DOC] Initialize the parent and raise the tolerances to at least
        # the overrides.
        super().__init__(actual, expected, **other_parameters)
        self.rtol = max(self.rtol, rtol_override)
        self.atol = max(self.atol, atol_override)

    def _process_inputs(self, actual, expected, *, id, allow_subclasses):
        # [FUNCTION DOC] Check both inputs are tensors or arrays, convert them to tensors, check
        # that the tensors are supported, and return the pair.
        self._check_inputs_isinstance(actual, expected, cls=(torch.Tensor, np.ndarray))

        actual, expected = (self._to_tensor(input) for input in (actual, expected))
        for tensor in (actual, expected):
            self._check_supported(tensor, id=id)
        return actual, expected


class TypedStoragePair(TensorLikePair):
    # [CLASS DOC] Comparison pair for typed storage objects, comparing them as tensors built
    # from their untyped storage, with tolerance overrides.
    """Pair for :class:`torch.storage.TypedStorage` inputs."""
    def __init__(self, actual, expected, *, rtol_override=0.0, atol_override=0.0, **other_parameters):
        # [FUNCTION DOC] Check that both inputs are typed storages, then initialize the parent
        # and raise the tolerances to at least the overrides.
        self._check_inputs_isinstance(actual, expected, cls=torch.storage.TypedStorage)
        super().__init__(actual, expected, **other_parameters)
        self.rtol = max(self.rtol, rtol_override)
        self.atol = max(self.atol, atol_override)

    def _to_tensor(self, typed_storage):
        # [FUNCTION DOC] Convert a typed storage into a tensor on its device, mapping quantized
        # dtypes to the matching plain integer dtypes.
        return torch.tensor(
            typed_storage._untyped_storage,
            dtype={
                torch.quint8: torch.uint8,
                torch.quint4x2: torch.uint8,
                torch.quint2x4: torch.uint8,
                torch.qint32: torch.int32,
                torch.qint8: torch.int8
            }.get(typed_storage.dtype, typed_storage.dtype),
            device=typed_storage.device,
        )


class UnittestPair(Pair):
    # [CLASS DOC] Abstract base for fallback comparison pairs that delegate to the unittest
    # assertEqual. Subclasses set CLS to the supported input class(es) and optionally TYPE_NAME
    # used in the failure message.
    """Fallback ABC pair that handles non-numeric inputs.

    To avoid recreating the mismatch messages of :meth:`unittest.TestCase.assertEqual`, this pair simply wraps it in
    order to use it with the :class:`Pair` "framework" from :func:`are_equal`.

    Define the :attr:`UnittestPair.CLS` in a subclass to indicate which class(es) of the inputs the pair should support.
    """
    CLS: type | tuple[type, ...]
    TYPE_NAME: str | None = None

    def __init__(self, actual, expected, **other_parameters):
        # [FUNCTION DOC] Check that both inputs are instances of CLS and initialize the parent.
        self._check_inputs_isinstance(actual, expected, cls=self.CLS)
        super().__init__(actual, expected, **other_parameters)

    def compare(self):
        # [FUNCTION DOC] Compare the inputs with assertEqual of a fresh unittest TestCase. On
        # mismatch raises (through the fail hook) AssertionError with a message naming the type
        # and containing the original message.
        test_case = unittest.TestCase()

        try:
            return test_case.assertEqual(self.actual, self.expected)
        except test_case.failureException as error:
            msg = str(error)

        type_name = self.TYPE_NAME or (self.CLS if isinstance(self.CLS, type) else self.CLS[0]).__name__
        self._fail(AssertionError, f"{type_name.title()} comparison failed: {msg}")


class StringPair(UnittestPair):
    # [CLASS DOC] Comparison pair for strings and bytes.
    CLS = (str, bytes)
    TYPE_NAME = "string"


class SetPair(UnittestPair):
    # [CLASS DOC] Comparison pair for sets.
    CLS = set


class TypePair(UnittestPair):
    # [CLASS DOC] Comparison pair for type objects.
    CLS = type


class ObjectPair(UnittestPair):
    # [CLASS DOC] Fallback comparison pair for arbitrary objects.
    CLS = object


# This implements a variant of assertRaises/assertRaisesRegex where we first test
# if the exception is NotImplementedError, and if so just skip the test instead
# of failing it.
#
# This is implemented by inheriting from the (private) implementation of
# assertRaises from unittest.case, and slightly tweaking it for this new
# behavior.  The year is 2021: this private class hierarchy hasn't changed since
# 2010, seems low risk to inherit from.
class AssertRaisesContextIgnoreNotImplementedError(unittest.case._AssertRaisesContext):
    # [CLASS DOC] Variant of the assertRaises context from unittest which, if the exception
    # raised is NotImplementedError, skips the test instead of failing. Inherits from a private
    # unittest class.
    def __exit__(self, exc_type, exc_value, tb):
        # [FUNCTION DOC] If the exception is a NotImplementedError, skip the test with a
        # message; otherwise defer to the parent behavior and return its result.
        if exc_type is not None and issubclass(exc_type, NotImplementedError):
            self.test_case.skipTest(f"not_implemented: {exc_value}")  # type: ignore[attr-defined]
        return super().__exit__(exc_type, exc_value, tb)


@contextmanager
def set_warn_always_context(new_val: bool):
    # [FUNCTION DOC] Context manager that sets the global warn always mode for the body and
    # restores the previous value afterwards.
    old_val = torch.is_warn_always_enabled()
    torch.set_warn_always(new_val)
    try:
        yield
    finally:
        torch.set_warn_always(old_val)


class NoTest:
    # [CLASS DOC] HELPER: Marker base class whose test flag is false so pytest does not collect
    # it as a test.
    # causes pytest to not recognize this class as a test
    __test__ = False


class TestCase(expecttest.TestCase):
    # [CLASS DOC] Base class of all PyTorch unit tests, extending the expecttest TestCase. Role:
    # provides global state hygiene (RNG seeding, leak detection of grad mode, default dtype,
    # torch function mode state and fp32 precision flags), automatic skipping of disabled and
    # slow tests, optional dynamo, aot eager and inductor runs with expected failure tables,
    # CUDA memory leak and non default stream policies, reproduction commands on failure, and
    # many assertion helpers (tensor aware assertEqual, expected output files, log, warning and
    # determinism checks, sparse tensor generators). State: class attributes for precision and
    # relative tolerance overrides, flags for the CUDA policies, the default dtype check and
    # NotImplementedError handling, and values saved in setUp for restoration in tearDown. Key
    # methods: setUp, tearDown, run, assertEqual, assertExpectedInline.
    # NOTE: "precision" lets classes and generated tests set minimum
    # atol values when comparing tensors. Used by @precisionOverride and @toleranceOverride, for
    # example.
    # NOTE: "rel_tol" lets classes and generated tests set minimum
    # rtol values when comparing tensors. Used by @toleranceOverride, for example.
    _precision: float = 0
    _rel_tol: float = 0

    # Toggles whether to assert that `torch.get_default_dtype()` returns
    # `torch.float` when `setUp` and `tearDown` are called.
    _default_dtype_check_enabled: bool = False

    _prev_torch_function_mode_stack_len: int = 0
    _prev_torch_function_state = torch._C._TorchFunctionState.ENABLED

    # Always use difflib to print diffs on multi line equality.
    # Undocumented feature in unittest
    _diffThreshold = sys.maxsize
    maxDiff = None

    # checker to early terminate test suite if unrecoverable failure occurs.
    def _should_stop_test_suite(self):
        # [FUNCTION DOC] HELPER: If CUDA has been initialized, synchronize it; return True (and
        # print a message to stderr) when a RuntimeError occurs, which means the suite should be
        # terminated early. Returns False otherwise.
        if torch.cuda.is_initialized():
            # CUDA device side error will cause subsequence test cases to fail.
            # stop entire test suite if catches RuntimeError during torch.cuda.synchronize().
            try:
                torch.cuda.synchronize()
            except RuntimeError as rte:
                print("TEST SUITE EARLY TERMINATION due to torch.cuda.synchronize() failure", file=sys.stderr)
                print(str(rte), file=sys.stderr)
                return True
            return False
        else:
            return False

    @property
    def precision(self) -> float:
        # [FUNCTION DOC] Property returning the class level minimum absolute tolerance used in
        # tensor comparisons.
        return self._precision

    @precision.setter
    def precision(self, prec: float) -> None:
        # [FUNCTION DOC] Property setter that stores a new class or instance level minimum
        # absolute tolerance used in tensor comparisons.
        self._precision = prec

    @property
    def rel_tol(self) -> float:
        # [FUNCTION DOC] Property returning the minimum relative tolerance used in
        # tensor comparisons.
        return self._rel_tol

    @rel_tol.setter
    def rel_tol(self, prec: float) -> None:
        # [FUNCTION DOC] Property setter that stores a new minimum relative tolerance.
        self._rel_tol = prec

    _do_cuda_memory_leak_check = False
    _do_cuda_non_default_stream = False

    # When True, if a test case raises a NotImplementedError, instead of failing
    # the test, skip it instead.
    _ignore_not_implemented_error = False

    def __init__(self, method_name='runTest', methodName='runTest'):
        # [FUNCTION DOC] Construct the test case for a method name (either keyword spelling is
        # accepted for compatibility with testslide; the method name keyword wins when not the
        # default). Wraps the test method with policies according to global settings: a CUDA
        # memory leak check (when enabled, not on Windows), enforcement of a non default CUDA
        # stream, skipping on NotImplementedError when requested, and printing a reproduction
        # command (test file path, class and method plus environment prefix) on failure.
        # Failures to compute the repro command are logged and ignored.
        # KNOWN ISSUE: the log call passes a string as the extra argument, which logging expects
        # to be a mapping, so reporting that failure would itself raise inside logging.
        # methodName is the correct naming in unittest and testslide uses keyword arguments.
        # So we need to use both to 1) not break BC and, 2) support testslide.
        if methodName != "runTest":
            method_name = methodName
        super().__init__(method_name)

        test_method = getattr(self, method_name, None)
        if test_method is not None:
            # Wraps the tested method if we should do CUDA memory check.
            if TEST_CUDA_MEM_LEAK_CHECK:
                self._do_cuda_memory_leak_check &= getattr(test_method, '_do_cuda_memory_leak_check', True)
                # FIXME: figure out the flaky -1024 anti-leaks on windows. See #8044
                if self._do_cuda_memory_leak_check and not IS_WINDOWS:
                    self.wrap_with_cuda_policy(method_name, self.assertLeaksNoCudaTensors)

            # Wraps the tested method if we should enforce non default CUDA stream.
            self._do_cuda_non_default_stream &= getattr(test_method, '_do_cuda_non_default_stream', True)
            if self._do_cuda_non_default_stream and not IS_WINDOWS:
                self.wrap_with_cuda_policy(method_name, self.enforceNonDefaultStream)

            if self._ignore_not_implemented_error:
                self.wrap_with_policy(method_name, lambda: skip_exception_type(NotImplementedError))

            if PRINT_REPRO_ON_FAILURE:
                try:
                    def _get_rel_test_path(abs_test_path):
                        # [FUNCTION DOC] HELPER: Return the path of a test file relative to the
                        # directory containing a path component named test; if none exists,
                        # return only the file name.
                        # Attempt to get relative path based on the "test" dir.
                        # In CI, the working dir is not guaranteed to be the base repo dir so
                        # we can't just compute relative path from that.
                        parts = Path(abs_test_path).parts
                        for i, part in enumerate(parts):
                            if part == "test":
                                base_dir = os.path.join(*parts[:i]) if i > 0 else ''
                                return os.path.relpath(abs_test_path, start=base_dir)

                        # Can't determine containing dir; just return the test filename.
                        # The path isn't strictly correct but it's arguably better than nothing.
                        return os.path.split(abs_test_path)[1]

                    abs_test_path = inspect.getfile(type(self))
                    test_filename = _get_rel_test_path(abs_test_path)
                    class_name = type(self).__name__
                    test_run_cmd = f"python {test_filename} {class_name}.{method_name}"
                    env_var_prefix = TestEnvironment.repro_env_var_prefix()
                    repro_parts = [env_var_prefix, test_run_cmd]
                    self.wrap_with_policy(
                        method_name,
                        lambda repro_parts=repro_parts: print_repro_on_failure(repro_parts))
                except Exception as e:
                    # Don't fail entirely if we can't get the test filename
                    log.info("could not print repro string", extra=str(e))  # type: ignore[arg-type]

    def assertLeaksNoCudaTensors(self, name=None):
        # [FUNCTION DOC] Return a context manager (CudaMemoryLeakCheck) that checks for leaked
        # CUDA memory over its body; name defaults to the test id.
        name = self.id() if name is None else name
        return CudaMemoryLeakCheck(self, name)

    def before_cuda_memory_leak_check(self):
        # [FUNCTION DOC] Hook called before the leak check compares memory; resets dynamo state
        # if dynamo has been imported.
        self._reset_dynamo_if_imported()

    @staticmethod
    def _reset_dynamo_if_imported():
        # [FUNCTION DOC] HELPER: Static method that resets dynamo if its module is already in
        # the module table, without importing it otherwise.
        dynamo = sys.modules.get("torch._dynamo")
        if dynamo is not None:
            reset = getattr(dynamo, "reset", None)
            if reset is not None:
                reset()

    def enforceNonDefaultStream(self):
        # [FUNCTION DOC] Return a context manager that forces non default CUDA streams.
        return CudaNonDefaultStream()

    def _remove_ansi_escape(self, input):
        # [FUNCTION DOC] HELPER: Return the input string with 7-bit ANSI escape
        # sequences removed.
        # 7-bit C1 ANSI sequences
        ansi_escape = re.compile(r'''
            \x1B  # ESC
            (?:   # 7-bit C1 Fe (except CSI)
                [@-Z\\-_]
            |     # or [ for CSI, followed by a control sequence
                \[
                [0-?]*  # Parameter bytes
                [ -/]*  # Intermediate bytes
                [@-~]   # Final byte
            )
        ''', re.VERBOSE)
        return ansi_escape.sub('', input)

    def remove_comment_lines(self, input_string):
        # [FUNCTION DOC] Return the string with lines whose first non blank character is a hash
        # sign removed.
        lines = input_string.split('\n')
        filtered_lines = [line for line in lines if not line.strip().startswith('#')]
        return '\n'.join(filtered_lines)

    def remove_empty_lines(self, input_string):
        # [FUNCTION DOC] Return the string with blank lines removed.
        lines = input_string.split('\n')
        filtered_lines = [line for line in lines if line.strip() != '']
        return '\n'.join(filtered_lines)

    # ignore comments will ignore lines that starts with # after being stripped
    def assertExpectedInline(self, actual, expect, skip=0, ignore_comments=False, ignore_empty_lines=False):
        # [FUNCTION DOC] Compare the string form of the actual value with the expected inline
        # string, after removing ANSI escapes and optionally comment lines and empty lines from
        # both. Delegates to the expecttest implementation (which can rewrite the source in
        # accept mode); skip adds one frame for this wrapper. Returns its result.
        actual = actual if isinstance(actual, str) else str(actual)
        actual = self._remove_ansi_escape(actual)
        expect = self._remove_ansi_escape(expect)
        if ignore_comments:
            actual = self.remove_comment_lines(actual)
            expect = self.remove_comment_lines(expect)

        if ignore_empty_lines:
            actual = self.remove_empty_lines(actual)
            expect = self.remove_empty_lines(expect)

        return super().assertExpectedInline(actual if isinstance(actual, str) else str(actual), expect, skip + 1)

    # Munges exceptions that internally contain stack traces, using munge_exc
    def assertExpectedInlineMunged(
        self, exc_type, callable, expect, *, skip=0, suppress_suffix=True, post_munge=None,
    ):
        # [FUNCTION DOC] Call a callable that is expected to raise exc_type, munge the exception
        # text with munge_exc (removing stack trace specific details), optionally post process
        # it, and compare to the expected inline string. Fails the test if nothing is raised.
        try:
            callable()
        except exc_type as e:
            munged = munge_exc(e, suppress_suffix=suppress_suffix, skip=skip + 1)
            if post_munge:
                munged = post_munge(munged)
            self.assertExpectedInline(
                munged, expect, skip=skip + 1
            )
            return
        self.fail(msg="Did not raise when expected to")

    def assertLogs(self, logger=None, level=None):
        # [FUNCTION DOC] Same as the unittest version, but defaults the logger to the
        # torch logger.
        if logger is None:
            logger = logging.getLogger("torch")
        return super().assertLogs(logger, level)

    def assertNoLogs(self, logger=None, level=None):
        # [FUNCTION DOC] Same as the unittest version, but defaults the logger to the
        # torch logger.
        if logger is None:
            logger = logging.getLogger("torch")
        return super().assertNoLogs(logger, level)

    def wrap_with_cuda_policy(self, method_name, policy):
        # [FUNCTION DOC] Wrap the named test method with the given policy, but only if CUDA is
        # available and the test id mentions gpu or cuda. Imports the CUDA test module, which
        # may initialize CUDA.
        test_method = getattr(self, method_name)
        # the import below may initialize CUDA context, so we do it only if
        # self._do_cuda_memory_leak_check or self._do_cuda_non_default_stream
        # is True.
        # TODO: sure looks like we unconditionally initialize the context here
        # -- ezyang
        from torch.testing._internal.common_cuda import TEST_CUDA
        fullname = self.id().lower()  # class_name.method_name
        if TEST_CUDA and ('gpu' in fullname or 'cuda' in fullname):
            setattr(self, method_name, self.wrap_method_with_policy(test_method, policy))

    def wrap_with_policy(self, method_name, policy):
        # [FUNCTION DOC] Replace the named test method on this instance by a version wrapped
        # with the given policy.
        test_method = getattr(self, method_name)
        setattr(self, method_name, self.wrap_method_with_policy(test_method, policy))

    # A policy is a zero-argument function that returns a context manager.
    # We don't take the context manager directly as it may be necessary to
    # construct it once per test method
    def wrap_method_with_policy(self, method, policy):
        # [FUNCTION DOC] Return a bound method that runs the given method inside the context
        # manager created by calling the policy (a zero argument function). The check is done
        # around each call, not in setUp and tearDown, because exceptions keep objects alive.
        # Assumes that `method` is the tested function in `self`.
        # NOTE: Python Exceptions (e.g., unittest.Skip) keeps objects in scope
        #       alive, so this cannot be done in setUp and tearDown because
        #       tearDown is run unconditionally no matter whether the test
        #       passes or not. For the same reason, we can't wrap the `method`
        #       call in try-finally and always do the check.
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Enter the policy context, call the method and drop
            # its result.
            with policy():
                method(*args, **kwargs)
        return types.MethodType(wrapper, self)

    def wrap_with_cuda_memory_check(self, method):
        # [FUNCTION DOC] Wrap a method with the CUDA memory leak check policy and return it.
        return self.wrap_method_with_policy(method, self.assertLeaksNoCudaTensors)

    def _dynamo_test_key(self):
        # [FUNCTION DOC] HELPER: Return the key class name dot test method name, used to look up
        # expected failures and skips for dynamo and inductor runs.
        return f"{self.__class__.__name__}.{self._testMethodName}"

    def compile_fn(self, fn, backend, nopython):
        # [FUNCTION DOC] Hook that compiles a function with dynamo optimize for the backend and
        # nopython flag; subclasses can override to control compilation.
        # Allows subclasses to control compilation
        return torch._dynamo.optimize(backend, nopython=nopython)(fn)

    def _run_custom(self, result=None):
        # [FUNCTION DOC] HELPER: Run the test, optionally under dynamo, aot eager or inductor.
        # Determines strict mode from environment and class and method attributes, resets dynamo
        # state before and after, patches suppress_errors, can disable inductor size asserts for
        # test_ops, wraps the run with expected failure or skip handling from the failure
        # tables, and runs without compiled autograd where listed. Afterwards, if the suite must
        # stop early (CUDA failure), records a fake failure and stops the result.
        using_unittest = isinstance(result, unittest.TestResult)

        super_run = super().run
        test_cls = super_run.__self__  # type: ignore[attr-defined]

        # Are we compiling?
        compiled = TEST_WITH_TORCHDYNAMO or TEST_WITH_AOT_EAGER or TEST_WITH_TORCHINDUCTOR
        # Is the class strict and compiling?
        strict_default = False
        should_reset_dynamo = False

        # We disable size_asserts for test_ops since some tests fail
        # due to mismatch of strides returned from eager v.s. meta kernels
        # Only some of the ops has this problem, but since tests in
        # test_op.py are parametrized, it's hard to do this specifically
        # for the affected ops.
        # It's not a big deal since these problems are captured by
        # test_torchinductor_opinfo.py as well.
        should_disable_size_asserts = False
        if compiled:
            try:
                path = inspect.getfile(type(test_cls))
                full_path = os.path.abspath(path)
                match = re.match(r".*/test/(.*).py", full_path)
                if match is not None:
                    filename = match.group(1)
                    if TEST_WITH_TORCHINDUCTOR:
                        from .dynamo_test_failures import FIXME_inductor_non_strict
                        strict_default = filename not in FIXME_inductor_non_strict
                        should_reset_dynamo = True

                        if filename == "test_ops":
                            should_disable_size_asserts = True
                    else:
                        strict_default = True
            # inspect.getfile can fail with these
            except (OSError, TypeError):
                pass
            if "STRICT_DEFAULT" in os.environ:
                if os.environ["STRICT_DEFAULT"] == "1":
                    strict_default = True

        strict_mode = False
        if compiled:
            test_method = getattr(self, self._testMethodName)
            if hasattr(test_method, "dynamo_strict"):
                strict_mode = test_method.dynamo_strict
            elif hasattr(test_cls, "dynamo_strict"):
                strict_mode = test_cls.dynamo_strict
            else:
                strict_mode = strict_default
        nopython = getattr(test_cls, "dynamo_strict_nopython", False) and compiled

        if strict_mode or should_reset_dynamo:
            torch._dynamo.reset()
        else:
            # For the non-compiled path there is no optimize() region, so this
            # reset (and the matching one after super_run) is the only place we
            # clear Dynamo state between tests. Resetting inside setUp/tearDown
            # would land inside the compiled region and corrupt compiled
            # autograd state, so do it here instead.
            self._reset_dynamo_if_imported()

        torch.compiler.set_stance("default")

        # TODO: Remove this; this is grandfathered in because we suppressed errors
        # on test suite previously
        # When strict mode is False, suppress_errors is True
        if compiled:
            suppress_errors = not strict_mode
        else:
            suppress_errors = torch._dynamo.config.suppress_errors

        maybe_disable_size_asserts = (
            torch._inductor.config.patch(size_asserts=False)
            if should_disable_size_asserts
            else contextlib.nullcontext()
        )

        with unittest.mock.patch("torch._dynamo.config.suppress_errors", suppress_errors), maybe_disable_size_asserts:
            if TEST_WITH_AOT_EAGER:
                super_run = self.compile_fn(super_run, "aot_eager_decomp_partition", nopython)
            elif TEST_WITH_TORCHDYNAMO or TEST_WITH_TORCHINDUCTOR:
                if TEST_WITH_TORCHINDUCTOR:
                    super_run = self.compile_fn(super_run, "inductor", nopython)
                else:
                    # Assume eager-generated GraphModules will not error out.
                    # If we do, this is probably a Dynamo bug!
                    super_run = self.compile_fn(super_run, "eager_noexcept", nopython)

                key = self._dynamo_test_key()

                def expect_failure(f, file_name):
                    # [FUNCTION DOC] HELPER: Wrap a function so that any failure becomes a skip,
                    # and passing raises RuntimeError asking to remove the expected failure
                    # marker file.
                    @wraps(f)
                    def wrapper(*args, **kwargs):
                        # [FUNCTION DOC] HELPER: Run the function; skip on any exception,
                        # otherwise raise RuntimeError about the unexpected success.
                        try:
                            f(*args, **kwargs)
                        except BaseException as e:
                            self.skipTest(e)
                        raise RuntimeError(f"Unexpected success, please remove `{file_name}`")
                    return wrapper

                if TEST_WITH_TORCHINDUCTOR:
                    subdir = "test/inductor_expected_failures"
                    from .dynamo_test_failures import inductor_expected_failures as expected_failures
                else:
                    subdir = "test/dynamo_expected_failures"
                    from .dynamo_test_failures import dynamo_expected_failures as expected_failures

                if key in expected_failures:
                    method = getattr(self, self._testMethodName)
                    file_name = os.path.join(subdir, key)
                    setattr(self, self._testMethodName, expect_failure(method, file_name))

                def ignore_failure(f, file_name):
                    # [FUNCTION DOC] HELPER: Wrap a function so that failures and passes both
                    # end in a skip with an explanatory message about the skip marker file.
                    @wraps(f)
                    def wrapper(*args, **kwargs):
                        # [FUNCTION DOC] HELPER: Run the function; skip on exception; after
                        # success skip with a message depending on whether the test was marked
                        # as expected failure.
                        try:
                            f(*args, **kwargs)
                        except BaseException as e:
                            self.skipTest(e)
                        method = getattr(self, self._testMethodName)
                        if getattr(method, "__unittest_expecting_failure__", False):
                            self.skipTest("unexpected success")
                        else:
                            self.skipTest(f"This test passed, maybe we can remove `{file_name}`")
                    return wrapper

                if TEST_WITH_TORCHINDUCTOR:
                    subdir = "test/inductor_skips"
                    from .dynamo_test_failures import inductor_skips as skips
                else:
                    subdir = "test/dynamo_skips"
                    from .dynamo_test_failures import dynamo_skips as skips

                if key in skips:
                    method = getattr(self, self._testMethodName)
                    file_name = os.path.join(subdir, key)
                    setattr(self, self._testMethodName, ignore_failure(method, file_name))

                from .dynamo_test_failures import compiled_autograd_skips
                if torch._dynamo.config.compiled_autograd and key in compiled_autograd_skips:
                    # Still run the test, but with compiled autograd disabled
                    super_run = runWithoutCompiledAutograd()(super_run)

            super_run(result=result)

        if strict_mode or should_reset_dynamo:
            torch._dynamo.reset()
        elif torch._dynamo.config.compiled_autograd:
            torch._dynamo.compiled_autograd.reset()
        else:
            self._reset_dynamo_if_imported()

        # Early terminate test if necessary.  If using pytest, use the -x flag instead
        if using_unittest and self._should_stop_test_suite():
            if result.wasSuccessful():
                case = TestCase()
                if TEST_SAVE_XML is not None:
                    # This is a big hacky, XMLRunner modifies expected type from TestCase to TestInfo
                    # Create dummy TestInfo to record results correctly
                    from xmlrunner.result import _TestInfo  # type: ignore[import]
                    case = _TestInfo(result, case)
                    case.output = _TestInfo.ERROR  # type: ignore[attr-defined]
                    case.elapsed_time = 0.0  # type: ignore[attr-defined]
                    case.test_description = "TestSuiteEarlyFailure"  # type: ignore[attr-defined]
                # This shouldn't really happen, but if does add fake failure
                # For more details see https://github.com/pytorch/pytorch/issues/71973
                result.failures.append((case, "TestSuite execution was aborted early"))
                if result.wasSuccessful() is not False:
                    raise AssertionError("expected result.wasSuccessful() to be False after adding failure")
            result.stop()


    def run(self, result=None):
        # [FUNCTION DOC] Run the test inside an exit stack, entering the crossref mode when that
        # mode is enabled, and delegate to the custom runner.
        with contextlib.ExitStack() as stack:
            if TEST_WITH_CROSSREF:
                stack.enter_context(CrossRefMode())
            self._run_custom(
                result=result,
            )

    def setUp(self):
        # [FUNCTION DOC] Prepare for a test: check whether it is disabled or slow (may skip),
        # seed the RNGs, remember and enable sparse tensor invariant checking, check the default
        # dtype if requested, and snapshot grad mode, the torch function mode stack length and
        # state, and the fp32 precision flags. Raises AssertionError on a wrong default dtype.
        check_if_enable(self)
        set_rng_seed()

        # Save global check sparse tensor invariants state that can be
        # restored from tearDown:
        self._check_invariants = torch.sparse.check_sparse_tensor_invariants.is_enabled()

        # Enable invariant checks for all sparse tensors constructions
        # including the unsafe ones. If this is not desired for some
        # test case, use check_invariants=False optional argument to
        # sparse tensor constructors or
        # @torch.sparse.check_sparse_tensor_invariants(False)
        # decorator to disable the invariant checks.
        torch.sparse.check_sparse_tensor_invariants.enable()

        if self._default_dtype_check_enabled:
            if torch.get_default_dtype() != torch.float:
                raise AssertionError(
                    f"expected default dtype to be torch.float, got {torch.get_default_dtype()}"
                )

        # attempt to reset some global state at the end of the test
        self._prev_grad_state = torch.is_grad_enabled()
        self._prev_torch_function_mode_stack_len = torch._C._len_torch_function_stack()
        self._prev_torch_function_state = torch._C._get_torch_function_state()
        self._prev_fp32_precision = _snapshot_fp32_precision()

    def tearDown(self):
        # [FUNCTION DOC] Clean up after a test and detect leaked global state. Restores the
        # sparse invariant checking and grad mode, then raises AssertionError if the default
        # dtype changed (when checked), the torch function mode stack length or state changed
        # (after repairing them), or the fp32 precision flags changed (after restoring them)
        # unless the leak check is disabled with an environment variable.
        # There exists test cases that override TestCase.setUp
        # definition, so we cannot assume that _check_invariants
        # attribute is defined in general.
        if hasattr(self, '_check_invariants'):
            # Restore the global check sparse tensor invariants state
            if self._check_invariants:
                torch.sparse.check_sparse_tensor_invariants.enable()
            else:
                torch.sparse.check_sparse_tensor_invariants.disable()

        if self._default_dtype_check_enabled:
            if torch.get_default_dtype() != torch.float:
                raise AssertionError(
                    f"expected default dtype to be torch.float, got {torch.get_default_dtype()}"
                )

        # attribute may not be defined, per above
        if hasattr(self, '_prev_grad_state'):
            torch.set_grad_enabled(self._prev_grad_state)

        # torch.set_default_device pushes a DeviceContext onto the torch
        # function mode stack, so this check also catches leaked default devices.
        after = torch._C._len_torch_function_stack()
        if after != self._prev_torch_function_mode_stack_len:
            for _ in range(after - self._prev_torch_function_mode_stack_len):
                torch._C._pop_torch_function_stack()
            raise AssertionError(
                f"torch function mode stack was leaked: "
                f"length changed from {self._prev_torch_function_mode_stack_len} to {after}"
            )

        tf_state = torch._C._get_torch_function_state()
        if tf_state != self._prev_torch_function_state:
            torch._C._set_torch_function_state(self._prev_torch_function_state)
            raise AssertionError(
                f"torch function state was leaked: "
                f"changed from {self._prev_torch_function_state} to {tf_state}"
            )

        # Detect leaked mutations to the six fp32 precision flags. Tests that
        # legitimately mutate these globals must restore them themselves (e.g.
        # via recover_orig_fp32_precision or setUpClass/tearDownClass).
        # Escape hatch: PYTORCH_DISABLE_FP32_PRECISION_LEAK_CHECK=1 disables
        # the check globally; goal is zero callers.
        if (
            hasattr(self, '_prev_fp32_precision')
            and os.environ.get('PYTORCH_DISABLE_FP32_PRECISION_LEAK_CHECK') != '1'
        ):
            current = _snapshot_fp32_precision()
            if current != self._prev_fp32_precision:
                _restore_fp32_precision(self._prev_fp32_precision)
                specs = _fp32_precision_flag_specs()
                mismatches = [
                    f"  {label}: was {prev!r}, became {cur!r}"
                    for (label, _, _), prev, cur in zip(specs, self._prev_fp32_precision, current)
                    if prev != cur
                ]
                raise AssertionError(
                    "fp32 precision flag leak detected:\n" + "\n".join(mismatches)
                )

    @staticmethod
    def _make_crow_indices(n_rows, n_cols, nnz,
                           *, device, dtype, random=True):
        # [FUNCTION DOC] HELPER: Static method that builds the compressed row indices of a CSR
        # tensor with the given number of rows and columns and exactly nnz specified elements on
        # the requested device and dtype. Uses a sawtooth model of per row counts so that rows
        # are full, empty or partial, bisecting to size the window, and optionally shuffles the
        # row order. Raises AssertionError if nnz is out of bounds.
        """Return crow_indices of a CSR tensor with size (n_rows, n_cols) and
        the number of specified elements nnz.

        If random is True, the column counts of rows are in random
        order. Otherwise, the column counts of rows are defined by the
        used sampling method.

        Sampling method
        ---------------

        The used sampling method was introduced in
        https://pearu.github.io/csr_sampling.html, and here we give
        only an overall description of the method.

        Notice that crow_indices can be defined as cumsum(counts)
        where counts is a sequence of non-negative integers satisfying
        the following conditions:

          len(counts) == n_rows + 1
          counts.max() <= n_cols

        while counts[i + 1] is interpreted as the number of specified
        elements in the i-th row.

        The used sampling method aims at increasing the diversity of
        CSR samples, that is, a CSR sample should contain (i) rows
        that are all filled, (ii) rows with no elements at all, and
        (iii) rows that are partially filled. At the same time and for
        the given total number of specified elements (nnz), there
        should be minimal preference to rows with a given number of
        elements.  To achieve this, the sampling method is built-up on
        using a sawteeth model for counts. In the simplest case, we
        would have

          counts = arange(n_rows + 1) % (n_cols + 1)

        that has equal number of all possible column counts per row.
        This formula can be used only for specific input values of
        n_rows, n_cols, and nnz. To generalize this model to any
        combinations of inputs, the counts model above is extended
        with an incomplete sawtooth, and the right and lower
        rectangular parts that will guarantee that

          counts.sum() == nnz

        for any combination of n_rows, n_cols, and nnz. Basically,
        we'll find a maximal window in (n_rows + 1, n_cols + 1)-grid
        that is able to hold a sequence of sawteeth and so-called
        final correction, while the external part of the window is
        filled with counts to meet the nnz constraint exactly.
        """
        if not (0 <= nnz <= n_rows * n_cols):
            raise AssertionError(
                f"nnz out of bounds: expected 0 <= nnz <= n_rows * n_cols, got nnz={nnz}, n_rows={n_rows}, n_cols={n_cols}"
            )

        def sawteeth(n, m):
            # [FUNCTION DOC] HELPER: Return the total count of elements in the sequence of
            # sawteeth that fits a window defined by n rows and m columns removed from the grid.
            # return the total number of counts in the sequence of
            # sawteeth where n and m define a window in (n_rows+1,
            # n_cols+1) rectangle where the sequence of sawteeth
            # perfectly fit.
            M = (n_cols - m) * (n_cols - m + 1) // 2
            K = (n_rows - n) % (n_cols - m + 1)
            return M * ((n_rows - n) // (n_cols - m + 1)) + K * (K - 1) // 2

        # Different from the original method description, here counts
        # has leading 0 required by crow_indices:
        counts = torch.zeros(n_rows + 1, dtype=dtype, device=torch.device('cpu'))

        n = m = 0
        N = sawteeth(n, m)
        if N and nnz >= max(N, n_cols):
            # determine the width of the sawteeth window. We use bisection to solve
            #   N(n, 0) == 0 or nnz - n * n_cols < max(N(n, 0), n_cols)
            # for n
            n_left = n
            n_right = n_rows - 1
            N_right = sawteeth(n_right, m)
            while n_right - n_left > 1:
                n_middle = (n_left + n_right) // 2
                N_middle = sawteeth(n_middle, m)
                if N_middle == 0 or nnz - n_middle * n_cols < max(N_middle, n_cols):
                    n_right, N_right = n_middle, N_middle
                else:
                    n_left = n_middle
            n, N = n_right, N_right
            # fill the right rectangle with counts:
            if not n:
                raise AssertionError("n must be non-zero")
            counts[-n:].fill_(n_cols)

        if N and nnz - n * n_cols >= max(N, n_rows - n):
            # determine the height of the sawteeth window. We use bisection to solve
            #   N(n, m) == 0 or nnz - n * n_cols - m * (n_rows - n) < max(N(n, m), n_rows - n)
            # for m.
            m_left = m
            m_right = n_cols - 1
            N_right = sawteeth(n, m_right)
            while m_right - m_left > 1:
                m_middle = (m_left + m_right) // 2
                N_middle = sawteeth(n, m_middle)
                if N_middle == 0 or nnz - n * n_cols - m_middle * (n_rows - n) < max(N_middle, n_rows - n):
                    m_right, N_right = m_middle, N_middle
                else:
                    m_left = m_middle
            m, N = m_right, N_right
            # fill the bottom rectangle with counts:
            if not m:
                raise AssertionError("m must be non-zero")
            counts[1:n_rows - n + 1].fill_(m)

        if N:
            # fill the sawteeth window with counts
            q, r = divmod(nnz - n * n_cols - m * (n_rows - n),
                          (n_cols - m) * (n_cols - m + 1) // 2)
            p = 1 + q * (n_cols - m + 1)
            k = math.isqrt(2 * r)
            if k * (k + 1) > 2 * r:
                k -= 1
            corr = r - k * (k + 1) // 2
            if (p > 1) and (m > 0):
                raise AssertionError(
                    f"full sawteeth are never on top of a bottom rectangle: p={p}, m={m}"
                )
            # sequence of full sawteeth:
            counts[1:p] = torch.arange(p - 1, dtype=dtype, device=counts.device) % (n_cols - m + 1)
            # incomplete sawtooth:
            counts[p:p + k + 1] += torch.arange(k + 1, dtype=dtype, device=counts.device)
        else:
            # given input does not support sawteeth
            p = 1
            corr = nnz - n * n_cols - m * (n_rows - n)

        # correction that will guarantee counts.sum() == nnz:
        counts[p] += corr

        if random:
            # randomize crow_indices by shuffling the sawteeth
            # sequence:
            perm = torch.randperm(n_rows, device=counts.device)
            counts[1:] = counts[1:][perm]

        # compute crow_indices:
        crow_indices = counts
        crow_indices.cumsum_(dim=0)
        return crow_indices.to(device=device)

    def genSparseCompressedTensor(self, size, nnz, *, layout, device, dtype, index_dtype, blocksize=(), dense_dims=0):
        # [FUNCTION DOC] Generate a random sparse compressed tensor (CSR, CSC, BSR or BSC
        # depending on layout) with the given size, number of specified elements, index dtype,
        # optional block size and number of trailing dense dimensions. Batches are generated one
        # by one and stacked. Raises AssertionError on invalid sizes, nnz, or block sizes.
        from operator import mul
        from functools import reduce
        sparse_dim = 2
        if not (all(size[d] > 0 for d in range(len(size))) or nnz == 0):
            raise AssertionError(f"invalid arguments: size={size}, nnz={nnz}")
        if len(size) < sparse_dim:
            raise AssertionError(f"expected len(size) >= {sparse_dim}, got {len(size)}")
        if blocksize:
            if len(blocksize) != 2:
                raise AssertionError(f"expected len(blocksize) == 2, got size={size}, blocksize={blocksize}")
            if size[-2 - dense_dims] % blocksize[0] != 0:
                raise AssertionError(
                    f"size[-2 - dense_dims] must be divisible by blocksize[0]: size={size}, blocksize={blocksize}"
                )
            if size[-1 - dense_dims] % blocksize[1] != 0:
                raise AssertionError(
                    f"size[-1 - dense_dims] must be divisible by blocksize[1]: size={size}, blocksize={blocksize}"
                )
            blocksize0, blocksize1 = blocksize
        else:
            blocksize0 = blocksize1 = 1

        size = tuple(size)
        dense_size = size[(len(size) - dense_dims):]

        def random_sparse_compressed(n_compressed_dims, n_plain_dims, nnz):
            # [FUNCTION DOC] HELPER: Create values, compressed indices and sorted plain indices
            # for a single (non batch) matrix with random column or row positions.
            compressed_indices = self._make_crow_indices(n_compressed_dims, n_plain_dims, nnz, device=device, dtype=index_dtype)
            plain_indices = torch.zeros(nnz, dtype=index_dtype, device=device)
            for i in range(n_compressed_dims):
                count = compressed_indices[i + 1] - compressed_indices[i]
                plain_indices[compressed_indices[i]:compressed_indices[i + 1]], _ = torch.sort(
                    torch.randperm(n_plain_dims, dtype=index_dtype, device=device)[:count])
            low = -1 if dtype != torch.uint8 else 0
            high = 1 if dtype != torch.uint8 else 2
            values = make_tensor((nnz,) + blocksize + dense_size, device=device, dtype=dtype, low=low, high=high)
            return values, compressed_indices, plain_indices

        batch_shape = size[:-2 - dense_dims]
        n_batch = reduce(mul, batch_shape, 1)

        if layout in {torch.sparse_csr, torch.sparse_bsr}:
            n_compressed_dims, n_plain_dims = size[-2 - dense_dims] // blocksize0, size[-1 - dense_dims] // blocksize1
        else:
            n_compressed_dims, n_plain_dims = size[-1 - dense_dims] // blocksize1, size[-2 - dense_dims] // blocksize0
        blocknnz = nnz // (blocksize0 * blocksize1)
        sparse_tensors = [random_sparse_compressed(n_compressed_dims, n_plain_dims, blocknnz) for _ in range(n_batch)]
        sparse_tensors_it = map(list, zip(*sparse_tensors, strict=True))

        values = torch.stack(next(sparse_tensors_it)).reshape(*batch_shape, blocknnz, *blocksize, *dense_size)
        compressed_indices = torch.stack(next(sparse_tensors_it)).reshape(*batch_shape, -1)
        plain_indices = torch.stack(next(sparse_tensors_it)).reshape(*batch_shape, -1)
        return torch.sparse_compressed_tensor(compressed_indices, plain_indices,
                                              values, size=size, dtype=dtype, layout=layout, device=device)

    def genSparseCSRTensor(self, size, nnz, *, device, dtype, index_dtype, dense_dims=0):
        # [FUNCTION DOC] Generate a random CSR tensor by delegating to
        # genSparseCompressedTensor.
        return self.genSparseCompressedTensor(size, nnz, layout=torch.sparse_csr, device=device,
                                              dtype=dtype, index_dtype=index_dtype, blocksize=(), dense_dims=dense_dims)

    def genSparseCSCTensor(self, size, nnz, *, device, dtype, index_dtype, dense_dims=0):
        # [FUNCTION DOC] Generate a random CSC tensor by delegating to
        # genSparseCompressedTensor.
        # KNOWN ISSUE: the dense_dims argument is accepted but a fixed zero is passed on, so
        # hybrid CSC tensors cannot be generated.
        return self.genSparseCompressedTensor(size, nnz, layout=torch.sparse_csc, device=device,
                                              dtype=dtype, index_dtype=index_dtype, blocksize=(), dense_dims=0)

    def genSparseBSRTensor(self, size, blocksize, nnz, *, device, dtype, index_dtype, dense_dims=0):
        # [FUNCTION DOC] Generate a random BSR tensor with the given block size (must have
        # length two, otherwise AssertionError) via genSparseCompressedTensor.
        if len(blocksize) != 2:
            raise AssertionError(f"expected len(blocksize) == 2, got {len(blocksize)}")
        return self.genSparseCompressedTensor(size, nnz, layout=torch.sparse_bsr, device=device,
                                              dtype=dtype, index_dtype=index_dtype, blocksize=blocksize, dense_dims=dense_dims)

    def genSparseBSCTensor(self, size, blocksize, nnz, *, device, dtype, index_dtype, dense_dims=0):
        # [FUNCTION DOC] Generate a random BSC tensor with the given block size (must have
        # length two, otherwise AssertionError) via genSparseCompressedTensor.
        if len(blocksize) != 2:
            raise AssertionError(f"expected len(blocksize) == 2, got {len(blocksize)}")
        return self.genSparseCompressedTensor(size, nnz, layout=torch.sparse_bsc, device=device,
                                              dtype=dtype, index_dtype=index_dtype, blocksize=blocksize, dense_dims=dense_dims)

    def genSparseTensor(self, size, sparse_dim, nnz, is_uncoalesced, device, dtype):
        # [FUNCTION DOC] Generate a random sparse COO tensor of the given size and sparse
        # dimensions with nnz entries (duplicated indices if uncoalesced). Returns a tuple of
        # the tensor, a copy of its indices and a copy of its values. Raises AssertionError on
        # an impossible combination of sizes and nnz.
        # Assert not given impossible combination, where the sparse dims have
        # empty numel, but nnz > 0 makes the indices containing values.
        if not (all(size[d] > 0 for d in range(sparse_dim)) or nnz == 0):
            raise AssertionError(f"invalid arguments: size={size}, sparse_dim={sparse_dim}, nnz={nnz}")

        v_size = [nnz] + list(size[sparse_dim:])
        v = make_tensor(v_size, device=device, dtype=dtype, low=-1, high=1)
        i = torch.rand(sparse_dim, nnz, device=device)
        i.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(i))
        i = i.to(torch.long)
        if is_uncoalesced:
            i1 = i[:, :(nnz // 2), ...]
            i2 = i[:, :((nnz + 1) // 2), ...]
            i = torch.cat([i1, i2], 1)
        x = torch.sparse_coo_tensor(i, v, torch.Size(size), dtype=dtype, device=device)

        if not is_uncoalesced:
            x = x.coalesce()
        else:
            # FIXME: `x` is a sparse view of `v`. Currently rebase_history for
            #        sparse views is not implemented, so this workaround is
            #        needed for inplace operations done on `x`, e.g., copy_().
            #        Remove after implementing something equivalent to CopySlice
            #        for sparse views.
            # NOTE: We do clone() after detach() here because we need to be able to change size/storage of x afterwards
            x = x.detach().clone()._coalesced_(False)
        return x, x._indices().clone(), x._values().clone()

    def generate_simple_inputs(self, layout,
                               device=None,
                               dtype=None,
                               index_dtype=None,
                               pin_memory=None,
                               members_pin_memory=None,
                               enable_batch=True,
                               enable_hybrid=True,
                               enable_zero_sized=True,
                               enable_non_contiguous_indices=True,
                               enable_non_contiguous_values=True,
                               enable_batch_variable_nse=False,
                               output_tensor=True,
                               patterns=None):
        # [FUNCTION DOC] Generator of small, deterministic inputs for tensor constructors of the
        # given layout (strided, COO, CSR, CSC, BSR, BSC). Covers regular, batched, hybrid, zero
        # sized and optionally non contiguous indices or values, all representing the same
        # mathematical tensors across layouts. With output_tensor true, yields tensors;
        # otherwise yields pairs of positional arguments and keyword arguments for the
        # constructors. Raises AssertionError on unexpected layouts or malformed patterns.
        """Generator of simple inputs for tensor constructors of the given layout.

        The generated tensor inputs have the following properties:

        - tensor shapes are minimal but not trivial
        - tensor values are sorted sequences for COO and CSR formats, e.g. [1, 2, 3, 4]
        - the generated tensors represent the same mathematical tensor for all layouts
        - the generated tensors include regular, zero-sized, and optionally, batched or/and hybrid tensors.
        - the generated tensors include contiguous or non-contiguous tensors both in indices and values

        If output_tensor is True, yield tensors with the given
        layout. Otherwise, yield inputs to the corresponding tensor
        constructors:

          - sparse compressed input is defined as
            (compressed_indices, plain_indices, values), dict(size=expected_size_from_shape_inference, device=device, dtype=dtype,
                                                              pin_memory=pin_memory)

          - sparse COO input is defined as
            (indices, values), dict(size=expected_size_from_shape_inference, device=device, dtype=dtype, pin_memory=pin_memory)

          - strided input is defined as
            (values,), dict(device=device, dtype=dtype)
        """
        if index_dtype is None:
            index_dtype = torch.int64

        is_compressed_sparse_layout = layout in {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}

        if output_tensor:
            for args, kwargs in self.generate_simple_inputs(layout, device=device, dtype=dtype, index_dtype=index_dtype,
                                                            pin_memory=pin_memory,
                                                            enable_batch=enable_batch, enable_hybrid=enable_hybrid,
                                                            enable_zero_sized=enable_zero_sized,
                                                            enable_non_contiguous_indices=enable_non_contiguous_indices,
                                                            enable_non_contiguous_values=enable_non_contiguous_values,
                                                            enable_batch_variable_nse=enable_batch_variable_nse,
                                                            output_tensor=False):
                if members_pin_memory:
                    args = tuple(a.pin_memory() for a in args)
                if layout is torch.strided:
                    if len(args) != 1:
                        raise AssertionError(f"expected len(args) == 1 for strided layout, got {len(args)}")
                    size = kwargs.pop('size', None)  # to ensure that a zero-sized tensor has the desired shape
                    if size is None:
                        raise AssertionError("size must not be None for strided layout")
                    if pin_memory:
                        yield args[0].reshape(size).pin_memory()
                    else:
                        yield args[0].reshape(size)
                elif layout is torch.sparse_coo:
                    yield torch.sparse_coo_tensor(*args, **kwargs)
                elif is_compressed_sparse_layout:
                    kwargs.update(layout=layout)
                    yield torch.sparse_compressed_tensor(*args, **kwargs)
                else:
                    raise AssertionError(f"unreachable: unexpected layout {layout}")
            return

        def get_blockpattern(pattern, blocksize):
            # [FUNCTION DOC] HELPER: Reduce a 2-D pattern to a block pattern where each block
            # holds a nonzero element is numbered with a block id; the pattern shape must divide
            # by the block size, otherwise AssertionError.
            basesize = pattern.shape
            if basesize[0] % blocksize[0] != 0:
                raise AssertionError(
                    f"basesize[0] must be divisible by blocksize[0]: basesize={basesize}, blocksize={blocksize}"
                )
            if basesize[1] % blocksize[1] != 0:
                raise AssertionError(
                    f"basesize[1] must be divisible by blocksize[1]: basesize={basesize}, blocksize={blocksize}"
                )
            blockpattern = pattern.reshape(-1,
                                           blocksize[0],
                                           basesize[1] // blocksize[1],
                                           blocksize[1]).transpose(-3, -2).any(-1).any(-1)
            block_ids = torch.arange(1, blockpattern.numel() + 1).reshape(blockpattern.shape)
            return (blockpattern != 0) * block_ids

        def get_sparse_data(pattern):
            # [FUNCTION DOC] HELPER: For a 2-D pattern compute the data of each layout (COO,
            # CSR, CSC and strided) independently from the factory functions, with values
            # numbered from one in row major order. Raises AssertionError if the pattern is not
            # a matrix.
            basesize = pattern.shape
            if len(basesize) != 2:
                raise AssertionError(f"pattern is expected to be a matrix, got shape {basesize}")

            # We cannot use `torch.sparse_xyz_tensor(pattern)` to
            # compute the sparse layout indices and values because
            # generate_simple_inputs is used to generate the inputs to
            # test `torch.sparse_xyz_tensor` factory functions, so
            # we'll compute the indices and values independently of
            # the factory functions.

            indices = torch.where(pattern != 0)
            coo_indices = torch.stack(indices)
            crow_indices = torch.zeros(basesize[0] + 1, dtype=torch.int64)
            crow_indices[1:] = torch.cumsum(coo_indices[0].bincount(minlength=basesize[0]), 0)
            col_indices = coo_indices[1]
            strided_values = torch.zeros(basesize, dtype=torch.int64)

            # the property of `values == range(1, 1+nnz)` is used in
            # get_sparse_data_with_block to relate BSR and BSC values,
            # so, don't change the following line:
            values = torch.arange(1, 1 + len(indices[0]), dtype=torch.int64)
            strided_values[indices] = values

            indices_T = torch.where(pattern.transpose(0, 1) != 0)
            coo_indices_T = torch.stack(indices_T)
            ccol_indices = torch.zeros(basesize[1] + 1, dtype=torch.int64)
            ccol_indices[1:] = torch.cumsum(coo_indices_T[0].bincount(minlength=basesize[1]), 0)
            row_indices = coo_indices_T[1]
            csc_values = strided_values.transpose(0, 1)[indices_T]

            return {torch.sparse_coo: (coo_indices, values),
                    torch.sparse_csr: (crow_indices, col_indices, values),
                    torch.sparse_csc: (ccol_indices, row_indices, csc_values),
                    torch.strided: (strided_values,)}

        def get_sparse_data_with_block(pattern, blocksize):
            # [FUNCTION DOC] HELPER: Compute data for all layouts including BSR and BSC for
            # a pattern and block size, by grouping values into blocks and reordering them
            # for BSC.
            nonblock_data = get_sparse_data(pattern)
            blockpattern = get_blockpattern(pattern, blocksize)
            block_data = get_sparse_data(blockpattern)

            strided_values = nonblock_data[torch.strided][0]
            block_indices = block_data[torch.sparse_coo][0]
            bsr_values = torch.stack([strided_values[bi * blocksize[0]:(bi + 1) * blocksize[0],
                                                     bj * blocksize[1]:(bj + 1) * blocksize[1]]
                                      for bi, bj in block_indices.transpose(0, 1)])

            # here we use the property `values == range(1, 1+nnz)` and
            # `values` relation to `csc_values` (see get_sparse_data)
            # to get BSC blocks via reordering the BSR blocks:
            bsc_values = bsr_values[block_data[torch.sparse_csc][2] - 1]

            return {torch.sparse_bsr: (*block_data[torch.sparse_csr][:2], bsr_values),
                    torch.sparse_bsc: (*block_data[torch.sparse_csc][:2], bsc_values),
                    **nonblock_data}

        def get_batch_sparse_data(pattern, blocksize):
            # [FUNCTION DOC] HELPER: Compute layout data for a pattern that may have batch
            # dimensions by recursing over the leading dimension and concatenating results.
            size = pattern.shape
            if len(size) <= 2:  # non-batch
                return get_sparse_data_with_block(pattern, blocksize)

            # batch data is created recursively:
            batch_data = {}  # type: ignore[var-annotated]
            for i, item in enumerate(pattern):
                for layout, d in get_batch_sparse_data(item, blocksize).items():
                    target = batch_data.get(layout)
                    if layout is torch.sparse_coo:
                        # a "batch COO" means a COO with the leading
                        # sparse dimensions interpreted as batch
                        # dimensions
                        ext_coo_indices1 = torch.cat((torch.full((1, len(d[1])), i, dtype=torch.int64), d[0]))
                        if target is None:
                            target = batch_data[layout] = (ext_coo_indices1, d[1])
                        else:
                            target[0].set_(torch.cat((target[0], ext_coo_indices1), 1))  # type: ignore[call-overload]
                            target[1].set_(torch.cat((target[1], d[1])))
                    else:
                        if target is None:
                            target = batch_data[layout] = tuple(d[j].unsqueeze(0) for j in range(len(d)))
                        else:
                            for j in range(len(d)):
                                target[j].set_(torch.cat((target[j], d[j].unsqueeze(0))))  # type: ignore[call-overload]
            return batch_data

        def generate_values(base, densesize):
            # [FUNCTION DOC] HELPER: Create dense values of the given dense shape from a base
            # value or a batch of base values, such that each element encodes its dense indices
            # in decimal digits plus the base. Unique when every dense size is below ten.
            """Generates a tensor of shape densesize with values equal to

              base + i_1 * 10^0 + ... + i_d * 10^{d - 1}

            at indices i_1, ..., i_d (with 0 <= i_j < densesize[j] for any 1 <= j <=
            len(densesize))

            This mapping produces unique values as long as
            densesize[i] < 10 for all i in range(len(densesize)).
            """

            if not densesize:
                return base
            if not isinstance(base, int) and base.ndim > 0:
                return torch.stack([generate_values(b, densesize) for b in base])
            if base == 0:
                return torch.zeros(densesize, dtype=torch.int64)
            r = torch.arange(densesize[0], dtype=torch.int64)
            for i, d in enumerate(densesize[1:]):
                y = torch.arange(d, dtype=torch.int64) * (10 ** (i + 1))
                r = r[..., None] + y[None, ...]
            r.add_(base)
            return r

        if patterns is None:
            # A pattern is a 3-tuple with the following items:
            #
            # - a list of integers with the depth of two or more. The
            #   integers define the sparsity patterns of the generated
            #   inputs: zero values correspond to unspecified
            #   elements/blocks, and non-zero values to the specified
            #   elements.
            #
            #   For debugging convenience, the elements with the same
            #   value typically belong to the same block. However, it
            #   is not a hard requirement: as long as the shape of a
            #   pattern divides with block sizes, the pattern will be
            #   a valid one.
            #
            #   If the depth of the list is larger than two, inputs
            #   with batch dimensions will be generated.
            #
            # - a list of 2-tuples of block sizes, used to generate
            #   BSR/BSC tensors with various block size parameters
            #
            # - a list of tuples of dense dimensions, used to generate
            #   hybrid tensors with various dense dimensions
            #
            patterns = [
                # a simple 3 x 2 tensor: non-hybrid, hybrid with 1 and 2 dense dimensions
                ([[1, 2, 0],
                  [1, 0, 3]], [(2, 1), (1, 3)], [(), (2,), (4, 5)]),
                # 2 x 3 batch of 3 x 2 tensors: non-hybrid and hybrid with 2 dense dimensions
                ([[[[1, 2, 0],
                    [1, 0, 3]],
                   [[1, 2, 3],
                    [1, 0, 0]],
                   [[1, 0, 0],
                    [1, 2, 3]]],
                  [[[0, 2, 0],
                    [1, 2, 3]],
                   [[1, 0, 3],
                    [1, 2, 0]],
                   [[1, 2, 3],
                    [0, 2, 0]]]], [(2, 1), (2, 3)], [(), (2,)]),
                # tensor with non-trivial blocksize
                ([[0, 1, 0, 2, 0, 2],
                  [0, 1, 0, 0, 2, 0],
                  [3, 3, 3, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 5, 0, 6, 6, 6],
                  [5, 0, 5, 6, 6, 6],
                  [0, 0, 0, 0, 8, 8],
                  [7, 7, 7, 0, 8, 8]], [(2, 3)], [(), (4, 5)]),
                # batch tensor with variable NSE
                # Requires https://github.com/pytorch/pytorch/pull/84843 or similar.
                ([[[1, 2],
                   [3, 4]],
                  [[1, 0],
                   [0, 0]]], [(1, 1)], ([()] if enable_batch_variable_nse else []))]

        def non_contiguous_copy(t, dim=-1, offset=0):
            # [FUNCTION DOC] HELPER: Return a copy of a contiguous tensor that is not contiguous
            # along the given dimension, using the given storage offset. Asserts the input is
            # contiguous, the copy is not, and the values are equal.
            # return a copy of t that is non-contiguous along the
            # given dimension and with the given storage offset
            self.assertTrue(t.is_contiguous())
            if dim < 0:
                dim = dim + t.ndim
            if not (dim >= 0 and dim < t.ndim):
                raise AssertionError(f"dim out of range: expected 0 <= dim < {t.ndim}, got dim={dim}")
            step = max(2, offset + 1)
            tmp = torch.zeros((*t.shape[:dim], t.shape[dim] * step, *t.shape[dim + 1:]), dtype=t.dtype, device=t.device)
            dim_slices = (*((slice(None),) * dim), slice(offset, None, step))
            r = tmp[dim_slices].copy_(t)
            self.assertFalse(r.is_contiguous())
            self.assertEqual(t, r)
            return r

        # the main loop of the method:
        for pattern, blocksizes, densesizes in patterns:
            if not enable_hybrid:
                densesizes = [s for s in densesizes if not s]
            if not (densesizes and blocksizes):
                continue
            pattern = torch.tensor(pattern, dtype=torch.int64)
            if not enable_batch and pattern.ndim > 2:
                continue
            for blocksize in blocksizes:
                data = get_batch_sparse_data(pattern, blocksize)[layout]
                for densesize in densesizes:
                    indices = [a.to(device=device, dtype=index_dtype) for a in data[:-1]]
                    values = generate_values(data[-1], densesize).to(device=device, dtype=dtype)
                    kwargs = dict(device=device, dtype=dtype, size=pattern.shape + densesize)
                    if pin_memory is not None:
                        kwargs.update(pin_memory=pin_memory)

                    yield (*indices, values), kwargs.copy()
                    if enable_non_contiguous_indices and pattern.ndim > 2:
                        # sparse compressed indices can be sliced only along batch dimensions
                        for (dim, offset) in {(0, 1), (-2, 0)}:
                            indices_copy = [non_contiguous_copy(a, dim=dim, offset=offset) for a in indices]
                            yield (*indices_copy, values), kwargs.copy()

                            if enable_non_contiguous_values:
                                values_copy = non_contiguous_copy(values, dim=-1, offset=1)
                                yield (*indices_copy, values_copy), kwargs.copy()

                    if enable_non_contiguous_values:
                        values_copy = non_contiguous_copy(values, dim=-1, offset=1)
                        yield (*indices, values_copy), kwargs.copy()

        # zero-sized tensor inputs, non-batch, non-hybrid/hybrid
        if enable_zero_sized:
            for basesize, blocksizes, densesizes in [
                    ((2, 0), [(1, 2)], [(), (2,), (2, 3)] if enable_hybrid else [()]),
                    ((0, 2), [(1, 2), (2, 1), (3, 2)], [()]),
                    ((0, 0), [(1, 2)], [()]),
            ]:
                for blocksize in blocksizes:
                    for densesize in densesizes:  # type: ignore[attr-defined]
                        if layout == torch.strided:
                            indices = ()  # type: ignore[assignment]
                            values = torch.empty((basesize + densesize), device=device, dtype=dtype)
                        elif layout == torch.sparse_coo:
                            indices = (torch.empty(len(basesize), 0, device=device, dtype=index_dtype),)  # type: ignore[assignment]
                            values = torch.empty((0, *densesize), device=device, dtype=dtype)
                        elif layout == torch.sparse_csr:
                            crow_indices = torch.tensor([0] * (basesize[0] + 1), device=device, dtype=index_dtype)
                            col_indices = torch.empty(0, device=device, dtype=index_dtype)
                            indices = (crow_indices, col_indices)  # type: ignore[assignment]
                            values = torch.empty((0, *densesize), device=device, dtype=dtype)
                        elif layout == torch.sparse_csc:
                            ccol_indices = torch.tensor([0] * (basesize[1] + 1), device=device, dtype=index_dtype)
                            row_indices = torch.empty(0, device=device, dtype=index_dtype)
                            indices = (ccol_indices, row_indices)  # type: ignore[assignment]
                            values = torch.empty((0, *densesize), device=device, dtype=dtype)
                        elif layout == torch.sparse_bsr:
                            crow_indices = torch.tensor([0] * (basesize[0] // blocksize[0] + 1), device=device, dtype=index_dtype)
                            col_indices = torch.empty(0, device=device, dtype=index_dtype)
                            indices = (crow_indices, col_indices)  # type: ignore[assignment]
                            values = torch.empty((0, *blocksize, *densesize), device=device, dtype=dtype)
                        elif layout == torch.sparse_bsc:
                            ccol_indices = torch.tensor([0] * (basesize[1] // blocksize[1] + 1), device=device, dtype=index_dtype)
                            row_indices = torch.empty(0, device=device, dtype=index_dtype)
                            indices = (ccol_indices, row_indices)  # type: ignore[assignment]
                            values = torch.empty((0, *blocksize, *densesize), device=device, dtype=dtype)
                        else:
                            raise AssertionError(f"unreachable: unexpected layout {layout}")
                        kwargs = dict(device=device, dtype=dtype, size=basesize + densesize)
                        if pin_memory is not None:
                            kwargs.update(pin_memory=pin_memory)
                        yield (*indices, values), kwargs

    def safeToDense(self, t):
        # [FUNCTION DOC] Convert a tensor to a dense strided tensor, coalescing it first when it
        # is sparse COO.
        # coalesce is only implemented for COO
        if t.layout == torch.sparse_coo:
            t = t.coalesce()
        return t.to_dense()

    # Compares a torch function with a reference function for a given sample input (object of SampleInput)
    # Note: only values are compared, type comparison is not done here
    def compare_with_reference(self, torch_fn, ref_fn, sample_input, **kwargs):
        # [FUNCTION DOC] Compare a torch function against a reference (numpy) function on a
        # sample input. Calls both with the sample (the reference with its numpy version)
        # and checks equality by values only, ignoring device. Extra keyword arguments go
        # to assertEqual.
        numpy_sample = sample_input.numpy()
        n_inp, n_args, n_kwargs = numpy_sample.input, numpy_sample.args, numpy_sample.kwargs
        t_inp, t_args, t_kwargs = sample_input.input, sample_input.args, sample_input.kwargs

        actual = torch_fn(t_inp, *t_args, **t_kwargs)
        expected = ref_fn(n_inp, *n_args, **n_kwargs)

        self.assertEqual(actual, expected, exact_device=False, **kwargs)

    # Compares the given Torch and NumPy functions on the given tensor-like object.
    # NOTE: both torch_fn and np_fn should be functions that take a single
    #   tensor (array). If the torch and/or NumPy function require additional
    #   arguments then wrap the function in a lambda or pass a partial function.
    # TODO: add args/kwargs for passing to assertEqual (e.g. rtol, atol)
    def compare_with_numpy(self, torch_fn, np_fn, tensor_like,
                           device=None, dtype=None, **kwargs):
        # [FUNCTION DOC] Compare a torch function and a numpy function applied to a tensor like
        # object. A tensor input is converted to a numpy array (bfloat16 via float32); other
        # inputs are built into both an array and a tensor with the given device and dtype. The
        # numpy result is converted to a tensor and compared with assertEqual. Raises
        # AssertionError if numpy is unavailable or device or dtype is given with a tensor.
        if not TEST_NUMPY:
            raise AssertionError("TEST_NUMPY must be True to use compare_with_numpy")

        if isinstance(tensor_like, torch.Tensor):
            if device is not None:
                raise AssertionError("device must be None when tensor_like is a Tensor")
            if dtype is not None:
                raise AssertionError("dtype must be None when tensor_like is a Tensor")
            t_cpu = tensor_like.detach().cpu()
            if t_cpu.dtype is torch.bfloat16:
                t_cpu = t_cpu.float()
            a = t_cpu.numpy()
            t = tensor_like
        else:
            d = copy.copy(torch_to_numpy_dtype_dict)
            d[torch.bfloat16] = np.float32
            a = np.array(tensor_like, dtype=d[dtype])
            t = torch.tensor(tensor_like, device=device, dtype=dtype)

        np_result = np_fn(a)
        torch_result = torch_fn(t).cpu()

        # Converts arrays to tensors
        if isinstance(np_result, np.ndarray):
            try:
                np_result = torch.from_numpy(np_result)
            except Exception:
                # NOTE: copying an array before conversion is necessary when,
                #   for example, the array has negative strides.
                np_result = torch.from_numpy(np_result.copy())
            if t.dtype is torch.bfloat16 and torch_result.dtype is torch.bfloat16 and np_result.dtype is torch.float:
                torch_result = torch_result.to(torch.float)

        self.assertEqual(np_result, torch_result, **kwargs)

    def assertEqualIgnoreType(self, *args, **kwargs) -> None:
        # [FUNCTION DOC] Compare two values with assertEqual without checking dtype. Its use is
        # discouraged and signals a test written wrongly.
        # If you are seeing this function used, that means test is written wrongly
        # and deserves detailed investigation
        return self.assertEqual(*args, exact_dtype=False, **kwargs)

    def assertEqualBroadcasting(self, x, y, *args, **kwargs) -> None:
        # [FUNCTION DOC] Assert that x equals y after broadcasting y to the shape of x. Scalars
        # and sequences are converted to tensors first. Extra arguments go to assertEqual.
        r"""Tests if tensor x equals to y, if y to be broadcast to x.shape.
        """
        if not isinstance(y, Iterable):
            # int, float, etc. or different shape tensors
            y = torch.ones_like(x) * y
        if not isinstance(y, torch.Tensor):
            # iterable, but not a tensor
            y = torch.ones_like(x) * torch.tensor(y)
        return self.assertEqual(x, y, *args, **kwargs)

    def assertEqual(
            self,
            x,
            y,
            msg: str | Callable[[str], str] | None = None,
            *,
            atol: float | None = None,
            rtol: float | None = None,
            equal_nan=True,
            exact_dtype=True,
            # TODO: default this to True
            exact_device=False,
            exact_layout=False,
            exact_stride=False,
            exact_is_coalesced=False
    ):
        # [FUNCTION DOC] Main equality assertion for tests. Normalizes the inputs (numpy arrays
        # with dtypes unknown to torch and sequence versus tensor comparisons are converted,
        # strided nested tensors are unbound, distributed tensors are unwrapped) and compares
        # using the relaxed comparison pair classes. Tolerances default by dtype and are raised
        # to the class level precision and relative tolerance overrides. Flags control checks of
        # dtype, device, layout, stride and coalescing, and NaN equality. msg may be a string or
        # a function that builds the message from the generated one. Raises AssertionError
        # describing the mismatch.
        # Hide this function from `pytest`'s traceback
        __tracebackhide__ = True

        # numpy's dtypes are a superset of what PyTorch supports. In case we encounter an unsupported dtype, we fall
        # back to an elementwise comparison. Note that this has to happen here and not for example in
        # `TensorOrArrayPair`, since at that stage we can no longer split the array into its elements and perform
        # multiple comparisons.
        if any(
            isinstance(input, np.ndarray) and not has_corresponding_torch_dtype(input.dtype) for input in (x, y)
        ):
            def to_list(input):
                # [FUNCTION DOC] HELPER: Convert a tensor, array or iterable into a list of
                # its elements.
                return input.tolist() if isinstance(input, (torch.Tensor, np.ndarray)) else list(input)

            x = to_list(x)
            y = to_list(y)
        # When comparing a sequence of numbers to a tensor, we need to convert the sequence to a tensor here.
        # Otherwise, the pair origination of `are_equal` will fail, because the sequence is recognized as container
        # that should be checked elementwise while the tensor is not.
        elif isinstance(x, torch.Tensor) and isinstance(y, Sequence):
            y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
        elif isinstance(x, Sequence) and isinstance(y, torch.Tensor):
            x = torch.as_tensor(x, dtype=y.dtype, device=y.device)

        # unbind NSTs to compare them; don't do this for NJTs
        if isinstance(x, torch.Tensor) and x.is_nested and x.layout == torch.strided:
            x = x.unbind()
        if isinstance(y, torch.Tensor) and y.is_nested and y.layout == torch.strided:
            y = y.unbind()

        x, y = _unwrap_dtensor_for_comparison(x, y)

        error_metas = not_close_error_metas(
            x,
            y,
            pair_types=(
                NonePair,
                RelaxedBooleanPair,
                RelaxedNumberPair,
                TensorOrArrayPair,
                TypedStoragePair,
                StringPair,
                SetPair,
                TypePair,
                ObjectPair,
            ),
            sequence_types=(
                Sequence,
                Sequential,
                ModuleList,
                ParameterList,
                ScriptList,
                torch.utils.data.dataset.Subset,
            ),
            mapping_types=(Mapping, ModuleDict, ParameterDict, ScriptDict),
            rtol=rtol,
            rtol_override=self.rel_tol,
            atol=atol,
            atol_override=self.precision,
            equal_nan=equal_nan,
            check_device=exact_device,
            check_dtype=exact_dtype,
            check_layout=exact_layout,
            check_stride=exact_stride,
            check_is_coalesced=exact_is_coalesced,
        )

        if error_metas:
            # See [ErrorMeta Cycles]
            error_metas = [error_metas]  # type: ignore[list-item]
            # TODO: compose all metas into one AssertionError
            raise error_metas.pop()[0].to_error(  # type: ignore[index]
                # This emulates unittest.TestCase's behavior if a custom message passed and
                # TestCase.longMessage (https://docs.python.org/3/library/unittest.html#unittest.TestCase.longMessage)
                # is True (default)
                (lambda generated_msg: f"{generated_msg}\n{msg}") if isinstance(msg, str) and self.longMessage else msg
            )

    def assertNotEqual(self, x, y, msg: str | None = None, *,                                       # type: ignore[override]
                       atol: float | None = None, rtol: float | None = None, **kwargs) -> None:
        # [FUNCTION DOC] Assert that two values are not equal, by asserting that assertEqual
        # raises AssertionError. Tolerances and extra arguments are forwarded.
        with self.assertRaises(AssertionError, msg=msg):
            self.assertEqual(x, y, msg, atol=atol, rtol=rtol, **kwargs)

    def _formatMessage(self, msg, standardMsg) -> str:  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Allow a callable message, called with the standard message to
        # produce the final message; otherwise defer to the parent implementation.
        # Allow a callable msg, invoked lazily on failure to build the message.
        if callable(msg):
            return msg(standardMsg)
        return super()._formatMessage(msg, standardMsg)

    def assertEqualTypeString(self, x, y) -> None:
        # [FUNCTION DOC] Assert two tensors have the same device, dtype and sparsity, standing
        # in for the removed comparison of type strings.
        # This API is used simulate deprecated x.type() is y.type()
        self.assertEqual(x.device, y.device)
        self.assertEqual(x.dtype, y.dtype)
        self.assertEqual(x.is_sparse, y.is_sparse)

    def assertObjectIn(self, obj: Any, iterable: Iterable[Any]) -> None:
        # [FUNCTION DOC] Assert that the very same object (compared by identity) occurs in the
        # iterable. Raises AssertionError otherwise.
        for elem in iterable:
            if id(obj) == id(elem):
                return
        raise AssertionError("object not found in iterable")

    # Reimplemented to provide special behavior when
    # _ignore_not_implemented_error is True
    def assertRaises(self, expected_exception, *args, **kwargs):
        # [FUNCTION DOC] Override of assertRaises: when not implemented errors are being
        # ignored, use a context that skips the test on NotImplementedError; otherwise defer to
        # the parent.
        if self._ignore_not_implemented_error:
            context: AssertRaisesContextIgnoreNotImplementedError | None = \
                AssertRaisesContextIgnoreNotImplementedError(expected_exception, self)  # type: ignore[call-arg]
            try:
                return context.handle('assertRaises', args, kwargs)  # type: ignore[union-attr, arg-type]
            finally:
                # see https://bugs.python.org/issue23890
                context = None
        else:
            return super().assertRaises(expected_exception, *args, **kwargs)

    # Reimplemented to provide special behavior when
    # _ignore_not_implemented_error is True
    def assertRaisesRegex(self, expected_exception, expected_regex, *args, **kwargs):
        # [FUNCTION DOC] Override of assertRaisesRegex: for device types that are not native
        # (and not mps) the message is not validated (empty pattern). When not implemented
        # errors are being ignored, a context that skips on NotImplementedError is used.
        # Verifies that an exception with the type expected_exception and message
        # matching the regular expression defined by expected_regex is thrown.
        # If the test is instantiated for a non-native device type (like XLA)
        # then the message is not validated.

        # Checks whether the test is instantiated for a device type by testing
        # if the test class has defined the device_type attribute and,
        # if so, tests whether the instantiated device type is native or not
        if hasattr(self, 'device_type') and self.device_type not in NATIVE_DEVICES and self.device_type != "mps":  # type: ignore[attr-defined]
            # empty string matches any string
            expected_regex = ''

        if self._ignore_not_implemented_error:
            context = AssertRaisesContextIgnoreNotImplementedError(  # type: ignore[call-arg]
                expected_exception, self, expected_regex)
            return context.handle('assertRaisesRegex', args, kwargs)  # type: ignore[attr-defined, arg-type]
        else:
            return super().assertRaisesRegex(expected_exception, expected_regex, *args, **kwargs)

    # Verifies that no unraisable exceptions are raised by callable.  Unlike regular
    # exceptions, these do not actually propagate to the caller and are
    # suppressed.  We must test for them specially.
    def assertNoUnraisable(self, callable, *args, **kwargs):
        # [FUNCTION DOC] Assert that calling the callable raises no unraisable exception (which
        # normally is silently ignored). Temporarily replaces the unraisable hook and disables
        # garbage collection during the call to avoid spurious failures, then asserts that
        # nothing was recorded.
        raised = None

        def record_unraisable(unraisable):
            # [FUNCTION DOC] HELPER: Hook that remembers the last unraisable exception record.
            nonlocal raised
            raised = unraisable

        # Disable GC when running the callable to prevent spurious flakiness
        # from unlucky GCs inside the callable
        prev = gc.isenabled()
        gc.disable()
        try:
            with unittest.mock.patch("sys.unraisablehook", record_unraisable):
                callable(*args, **kwargs)
        finally:
            if prev:
                gc.enable()

        self.assertIsNone(raised)

    # TODO: Support context manager interface
    # NB: The kwargs forwarding to callable robs the 'subname' parameter.
    # If you need it, manually apply your callable in a lambda instead.
    def assertExpectedRaises(self, exc_type, callable, *args, **kwargs):
        # [FUNCTION DOC] Call a callable that must raise the given exception type and compare
        # the exception text to a recorded expect file (using assertExpected). An optional
        # subname keyword is consumed and not forwarded to the callable. Fails the test if
        # nothing is raised.
        subname = None
        if 'subname' in kwargs:
            subname = kwargs['subname']
            del kwargs['subname']
        try:
            callable(*args, **kwargs)
        except exc_type as e:
            self.assertExpected(str(e), subname)
            return
        # Don't put this in the try block; the AssertionError will catch it
        self.fail(msg="Did not raise when expected to")

    def assertNotWarn(self, callable, msg=''):
        # [FUNCTION DOC] Assert that calling the callable produces no warnings, with warn always
        # mode forced on so warn once warnings are also seen.
        r"""
        Test if :attr:`callable` does not raise a warning.
        """
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            with set_warn_always_context(True):
                callable()
            self.assertTrue(len(ws) == 0, msg)

    @contextmanager
    def assertWarnsOnceRegex(self, category, regex=''):
        # [FUNCTION DOC] Context manager for code that must always warn. Forces warn always
        # mode, records the warnings from the body, and fails if none was caught, if none has
        # exactly the given category, or if none has a message matching the regex.
        """Context manager for code that *must always* warn

        This filters expected warnings from the test and fails if
        the expected warning is not caught. It uses set_warn_always() to force
        TORCH_WARN_ONCE to behave like TORCH_WARN
        """
        pattern = re.compile(regex)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            with set_warn_always_context(True):
                yield
            if len(ws) == 0:
                self.fail('no warning caught')
            self.assertTrue(any(type(w.message) is category for w in ws))
            self.assertTrue(
                any(re.match(pattern, str(w.message)) for w in ws),
                f'{pattern}, {[w.message for w in ws if type(w.message) is category]}')

    def assertExpected(self, s, subname=None):
        # [FUNCTION DOC] Compare a string with the content of an expect file named from the test
        # id (and optional unique subname) in an expect directory next to the test script. In
        # accept mode a missing or different file is rewritten (with the ONNX producer version
        # normalized); otherwise a missing file raises RuntimeError with instructions and a
        # difference raises an assertion failure. Raises TypeError if the argument is not a
        # string. Side effects: reads and may write files.
        r"""
        Test that a string matches the recorded contents of a file
        derived from the name of this test and subname.  This file
        is placed in the 'expect' directory in the same directory
        as the test script. You can automatically update the recorded test
        output using --accept.

        If you call this multiple times in a single function, you must
        give a unique subname each time.
        """
        if not isinstance(s, str):
            raise TypeError("assertExpected is strings only")

        def remove_prefix(text, prefix):
            # [FUNCTION DOC] HELPER: Return the text without the given prefix if it starts
            # with it.
            if text.startswith(prefix):
                return text[len(prefix):]
            return text
        # NB: we take __file__ from the module that defined the test
        # class, so we place the expect directory where the test script
        # lives, NOT where test/common_utils.py lives.  This doesn't matter in
        # PyTorch where all test scripts are in the same directory as
        # test/common_utils.py, but it matters in onnx-pytorch
        module_id = self.__class__.__module__
        munged_id = remove_prefix(self.id(), module_id + ".")
        test_file = os.path.realpath(sys.modules[module_id].__file__)  # type: ignore[type-var]
        expected_file = os.path.join(os.path.dirname(test_file),  # type: ignore[type-var, arg-type]
                                     "expect",
                                     munged_id)

        subname_output = ""
        if subname:
            expected_file += "-" + subname
            subname_output = f" ({subname})"
        expected_file += ".expect"
        expected = None

        def accept_output(update_type):
            # [FUNCTION DOC] HELPER: Print a message and write the string, with the producer
            # version replaced by a placeholder, into the expect file.
            print(f"Accepting {update_type} for {munged_id}{subname_output}:\n\n{s}")
            with open(expected_file, 'w') as f:
                # Adjust for producer_version, leave s unmodified
                s_tag = re.sub(r'(producer_version): "[0-9.]*"',
                               r'\1: "CURRENT_VERSION"', s)
                f.write(s_tag)

        try:
            with open(expected_file) as f:
                expected = f.read()
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
            elif expecttest.ACCEPT:
                return accept_output("output")
            else:
                raise RuntimeError(
                      f"I got this output for {munged_id}{subname_output}:\n\n{s}\n\n"
                      "No expect file exists; to accept the current output, run:\n"
                      f"python {__main__.__file__} {munged_id} --accept") from None

        # a hack for JIT tests
        if IS_WINDOWS:
            expected = re.sub(r'CppOp\[(.+?)\]', 'CppOp[]', expected)
            s = re.sub(r'CppOp\[(.+?)\]', 'CppOp[]', s)

        # Adjust for producer_version
        expected = expected.replace(
            'producer_version: "CURRENT_VERSION"',
            f'producer_version: "{torch.onnx.producer_version}"'
        )
        if expecttest.ACCEPT:
            if expected != s:
                return accept_output("updated output")
        else:
            if hasattr(self, "assertMultiLineEqual"):
                # Python 2.7 only
                # NB: Python considers lhs "old" and rhs "new".
                self.assertMultiLineEqual(expected, s)
            else:
                self.assertEqual(s, expected)

    def assertExpectedStripMangled(self, s, subname=None):
        # [FUNCTION DOC] Remove the mangled module names beginning with the torch module marker
        # from the string and then compare with assertExpected.
        s = re.sub(r'__torch__[^ ]+', '', s)
        self.assertExpected(s, subname)

    def assertGreaterAlmostEqual(self, first, second, places=None, msg=None, delta=None):
        # [FUNCTION DOC] Assert that first is greater than or equal to second, or within
        # delta (or a number of decimal places, default seven) below it. Raises TypeError if
        # both delta and places are given, and the failure exception with a formatted
        # message otherwise.
        """Assert that ``first`` is greater than or almost equal to ``second``.

        The equality of ``first`` and ``second`` is determined in a similar way to
        the ``assertAlmostEqual`` function of the standard library.
        """
        if delta is not None and places is not None:
            raise TypeError("specify delta or places not both")

        if first >= second:
            return

        diff = second - first
        if delta is not None:
            if diff <= delta:
                return

            standardMsg = f"{first} not greater than or equal to {second} within {delta} delta"
        else:
            if places is None:
                places = 7

            if round(diff, places) == 0:
                return

            standardMsg = f"{first} not greater than or equal to {second} within {places} places"

        msg = self._formatMessage(msg, standardMsg)
        raise self.failureException(msg)

    def assertAtenOp(self, onnx_model, operator, overload_name=""):
        # [FUNCTION DOC] Assert that an ONNX model contains an ATen operator node with the given
        # operator name and overload name. Fails the test when no ATen node exists or the
        # attributes differ.
        all_aten_nodes = [p for p in onnx_model.graph.node
                          if p.op_type == "ATen" and p.domain == "org.pytorch.aten"]
        self.assertTrue(all_aten_nodes)

        for op in all_aten_nodes:
            attrs = {attr.name: attr.s.decode() for attr in op.attribute}
            if attrs.get("operator") == operator:
                break

        self.assertEqual(attrs["operator"], operator)  # type: ignore[possibly-undefined]
        self.assertEqual(attrs.get("overload_name", ""), overload_name)

    def check_nondeterministic_alert(self, fn, caller_name, should_alert=True):
        # [FUNCTION DOC] Check that fn produces (or, when should_alert is false, does not
        # produce) a nondeterministic alert. Under deterministic mode it expects a RuntimeError
        # mentioning the caller name; under warn only mode it expects a UserWarning with that
        # message. When no alert is expected, it checks neither is produced.
        # KNOWN ISSUE: in the no-alert branch the loop tests whether the recorded warning
        # objects are UserWarning instances, but they are warning message records, so the
        # assertion inside never runs.
        '''Checks that an operation produces a nondeterministic alert when
        expected while `torch.use_deterministic_algorithms(True)` is set.

        Args:
          fn (callable): Function to check for a nondeterministic alert

          caller_name (str): Name of the operation that produces the
              nondeterministic alert. This name is expected to appear in
              the error/warning message.

          should_alert (bool, optional): If True, then the check will only pass
              if calling `fn` produces a nondeterministic error/warning with the
              expected message. If False, then the check will only pass if
              calling `fn` does not produce an error. Default: `True`.
        '''

        alert_message = caller_name + ' does not have a deterministic implementation, but you set'

        # Check that errors are thrown correctly
        with DeterministicGuard(True):
            if should_alert:
                with self.assertRaisesRegex(
                        RuntimeError,
                        alert_message,
                        msg='expected a non-deterministic error, but it was not raised'):
                    fn()

            else:
                # If a nondeterministic error is not expected, make sure
                # that it is not raised
                try:
                    fn()
                except RuntimeError as e:
                    if 'does not have a deterministic implementation' in str(e):
                        self.fail(
                            'did not expect non-deterministic error message, '
                            + 'but got one anyway: "' + str(e) + '"')
                    # Reraise exceptions unrelated to nondeterminism
                    raise

        # Check that warnings are thrown correctly
        with DeterministicGuard(True, warn_only=True):
            if should_alert:
                with self.assertWarnsRegex(
                        UserWarning,
                        alert_message):
                    fn()
            else:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    fn()
                    for warning in w:
                        if isinstance(warning, UserWarning):
                            self.assertTrue(re.search(alert_message, str(warning)) is None)

    # run code in subprocess and capture exceptions.
    @staticmethod
    def run_process_no_exception(code, env=None):
        # [FUNCTION DOC] Static method: run Python code in a new interpreter process with
        # the optional environment, capture output, and return a tuple of stdout bytes and
        # stderr bytes.
        import subprocess

        with subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        ) as p:
            (stdout, stderr) = p.communicate()
            return (stdout, stderr)

    # returns captured stderr
    @staticmethod
    def runWithPytorchAPIUsageStderr(code):
        # [FUNCTION DOC] Static method: run the code in a subprocess with API usage logging to
        # stderr enabled and the CI and show locals variables removed, and return the captured
        # stderr decoded as ascii text.
        env = os.environ.copy()
        env["PYTORCH_API_USAGE_STDERR"] = "1"
        # remove CI flag since this is a wrapped test process.
        # CI flag should be set in the parent process only.
        env.pop("CI", None)
        env.pop("TEST_SHOWLOCALS", None)
        _stdout, stderr = TestCase.run_process_no_exception(code, env=env)
        return stderr.decode('ascii')

    def _attempt_load_from_subprocess(
        self,
        file: pathlib.Path,
        import_string: str,
        expected_failure_message: str | None = None
    ) -> None:
        # [FUNCTION DOC] HELPER: Try to load a checkpoint file in a fresh subprocess with
        # weights only loading, after running an import string. If an expected failure message
        # is given, the load must raise RuntimeError containing it; otherwise it must succeed.
        # Subprocess failures are re-raised as RuntimeError with the output text.
        """
        Attempts weights_only `torch.load` in a subprocess. This is used to test that
        weights_only `torch.load` works as expected without global imports.

        Args:
            file (pathlib.Path): The path to the checkpoint to load.
            import_string (str): import string to add to the script
            expected_failure_message (str, optional): The expected failure message if the
                checkpoint fails to load. If None, the test will pass
        """
        script = f"import torch;{import_string}torch.load(r'{file}', weights_only=True)"
        cm = (
            self.assertRaisesRegex(RuntimeError, re.escape(expected_failure_message))
            if expected_failure_message else contextlib.nullcontext()
        )
        with cm:
            try:
                subprocess.check_output(
                    [sys.executable, "-c", script],
                    # On Windows, opening the subprocess with the default CWD makes `import torch`
                    # fail, so just set CWD to this script's directory
                    cwd=os.path.dirname(os.path.realpath(__file__)),
                    stderr=subprocess.STDOUT,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(e.output.decode("utf-8")) from None


class TestCaseBase(TestCase):
    # [CLASS DOC] HELPER: Empty subclass of TestCase. It exists so dynamically created classes
    # can call the parent implementations of setUp, tearDown and so on by passing this class to
    # super (see the linked discussion).
    # Calls to super() in dynamically created classes are a bit odd.
    # See https://github.com/pytorch/pytorch/pull/118586 for more info
    # Subclassing this class and then calling super(TestCaseBase) will run
    # TestCase's setUp, tearDown etc functions
    pass


def download_file(url, binary=True):
    # [FUNCTION DOC] Download a file from a URL into a data directory next to this module (a
    # writable path) and return its local path; an existing file is reused. A network failure
    # emits a RuntimeWarning and raises unittest.SkipTest. Side effects: network, file.
    from urllib.parse import urlsplit
    from urllib import request, error

    filename = os.path.basename(urlsplit(url)[2])
    data_dir = get_writable_path(os.path.join(os.path.dirname(__file__), 'data'))
    path = os.path.join(data_dir, filename)

    if os.path.exists(path):
        return path
    try:
        with request.urlopen(url, timeout=15) as f1, open(path, 'wb' if binary else 'w') as f2:
            data = f1.read()
            f2.write(data)
        return path
    except error.URLError as e:
        msg = f"could not download test file '{url}'"
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        raise unittest.SkipTest(msg) from e

def find_free_port():
    # [FUNCTION DOC] Find a free TCP port on localhost by binding to port zero, and return its
    # number. The port may be taken by someone else before it is used (race condition).
    """
    Finds an available port and returns that port number.

    NOTE: If this function is being used to allocate a port to Store (or
    indirectly via init_process_group or init_rpc), it should be used
    in conjunction with the `retry_on_connect_failures` decorator as there is a potential
    race condition where the allocated port may become unavailable before it can be used
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('localhost', 0))
        _, port = sock.getsockname()
        return port

# Errors that we can get in c10d initialization for which we should retry tests for.
ADDRESS_IN_USE = "address already in use"
CONNECT_TIMEOUT = "connect() timed out."

def retry_on_connect_failures(func=None, connect_errors=(ADDRESS_IN_USE)):
    # [FUNCTION DOC] Decorator that reruns a test when it raises RuntimeError whose message
    # (also lowercased) contains one of the given error strings, up to ten tries with a random
    # sleep below one second; after that raises RuntimeError. Other errors propagate. Usable
    # with or without arguments.
    # KNOWN ISSUE: the default for connect_errors is a plain string, not a tuple, so the loop
    # iterates over single characters and nearly any error message matches.
    """Reruns a test if the test returns a RuntimeError and the exception
    contains one of the strings in connect_errors."""
    # This if block is executed when using this function as a decorator with arguments.
    if func is None:
        return partial(retry_on_connect_failures, connect_errors=connect_errors)

    @wraps(func)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Retry loop around the test function described above.
        n_retries = 10
        tries_remaining = n_retries
        while True:
            try:
                return func(*args, **kwargs)
            except RuntimeError as error:
                if any(ce in str(error) or ce in str(error).lower() for ce in connect_errors):
                    tries_remaining -= 1
                    if tries_remaining == 0:
                        raise RuntimeError(f"Failing after {n_retries} retries with error: {str(error)}") from error
                    time.sleep(random.random())
                    continue
                raise
    return wrapper


# Decorator to retry upon certain Exceptions.
def retry(ExceptionToCheck, tries=3, delay=3, skip_after_retries=False):
    # [FUNCTION DOC] Return a decorator that retries a function when it raises the given
    # exception type, up to the number of tries with a fixed delay in seconds, printing a
    # message each time. The final failed attempt raises SkipTest.
    # KNOWN ISSUE: the final raise has the form of raising SkipTest from a conditional
    # expression, so SkipTest is raised even when skip_after_retries is false, and the original
    # error is only used as the cause.
    def deco_retry(f):
        # [FUNCTION DOC] HELPER: Wrap the function with the retry logic.
        @wraps(f)
        def f_retry(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Call the function, retrying on the checked exception while
            # more than one try remains, then make one last call that converts the failure into
            # a skip.
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except ExceptionToCheck as e:
                    msg = f"{e}, Retrying in {mdelay:d} seconds..."
                    print(msg)
                    time.sleep(mdelay)
                    mtries -= 1
            try:
                return f(*args, **kwargs)
            except ExceptionToCheck as e:
                raise unittest.SkipTest(f"Skipping after {tries} consecutive {str(e)}") from e if skip_after_retries else e
        return f_retry  # true decorator
    return deco_retry


# FIXME: modernize these to be consistent with make_tensor
#   and review including them in torch.testing
# Methods for matrix generation

def random_square_matrix_of_rank(l, rank, dtype=torch.double, device='cpu'):
    # [FUNCTION DOC] Return a random square matrix of size l by l with the given rank, built
    # from the SVD of a random matrix by zeroing the singular values at and past the rank (and
    # replacing any zero singular value inside the rank by one). Raises AssertionError if rank
    # exceeds l.
    if rank > l:
        raise AssertionError(f"rank must be <= l, got rank={rank}, l={l}")
    A = torch.randn(l, l, dtype=dtype, device=device)
    u, s, vh = torch.linalg.svd(A, full_matrices=False)
    for i in range(l):
        if i >= rank:
            s[i] = 0
        elif s[i] == 0:
            s[i] = 1
    return (u * s.to(dtype).unsqueeze(-2)) @ vh

def random_well_conditioned_matrix(*shape, dtype, device, mean=1.0, sigma=0.001):
    # [FUNCTION DOC] Return a random matrix or batch of matrices of the given shape whose
    # singular values are sampled from a Gaussian with the given mean and standard deviation
    # (small sigma means better conditioned), sorted in descending order.
    """
    Returns a random rectangular matrix (batch of matrices)
    with singular values sampled from a Gaussian with
    mean `mean` and standard deviation `sigma`.
    The smaller the `sigma`, the better conditioned
    the output matrix is.
    """
    primitive_dtype = {
        torch.float: torch.float,
        torch.double: torch.double,
        torch.cfloat: torch.float,
        torch.cdouble: torch.double
    }
    x = torch.rand(shape, dtype=dtype, device=device)
    m = x.size(-2)
    n = x.size(-1)
    u, _, vh = torch.linalg.svd(x, full_matrices=False)
    s = (torch.randn(*(shape[:-2] + (min(m, n),)), dtype=primitive_dtype[dtype], device=device) * sigma + mean) \
        .sort(-1, descending=True).values.to(dtype)
    return (u * s.unsqueeze(-2)) @ vh

# Returns a noncontiguous (tensor with the same shape and values as t
# The noncontiguous tensor is constructed such that elements in the innermost
#   dimension are separated by zeros or (whenever possible) nans
# TODO: consider more complicated noncontiguity schemes
def noncontiguous_like(t):
    # [FUNCTION DOC] Return a non contiguous tensor with the same shape and values as t, made as
    # a slice of a larger tensor where the other elements hold nan, True or 12 depending on
    # dtype. A tensor that is already non contiguous is returned as is. requires_grad is copied.
    # Short-circuits if t is already noncontiguous
    if not t.is_contiguous():
        return t

    # Choose a "weird" value that won't be accessed
    if t.dtype.is_floating_point or t.dtype.is_complex:
        value = math.nan
    elif t.dtype == torch.bool:
        value = True
    else:
        value = 12

    result = t.new_empty(t.shape + (2,))
    result[..., 0] = value
    result[..., 1] = t.detach()
    result = result[..., 1]
    result.requires_grad_(t.requires_grad)
    return result

# TODO: remove this (prefer make_symmetric_matrices below)
def random_symmetric_matrix(l, *batches, **kwargs):
    # [FUNCTION DOC] Return a random symmetric matrix, or batch of them, of size l by l using
    # optional dtype and device keyword arguments (default double and cpu). Deprecated in favor
    # of make_symmetric_matrices.
    dtype = kwargs.get('dtype', torch.double)
    device = kwargs.get('device', 'cpu')
    A = torch.randn(*(batches + (l, l)), dtype=dtype, device=device)
    A = (A + A.mT).div_(2)
    return A

# Creates a symmetric matrix or batch of symmetric matrices
# Shape must be a square matrix or batch of square matrices
def make_symmetric_matrices(*shape, device, dtype):
    # [FUNCTION DOC] Return a random symmetric matrix or batch of matrices with the given shape,
    # device and dtype. Raises AssertionError if the last two dimensions differ.
    if shape[-1] != shape[-2]:
        raise AssertionError(f"expected square matrix, got shape[-1]={shape[-1]} != shape[-2]={shape[-2]}")
    t = make_tensor(shape, device=device, dtype=dtype)
    t = (t + t.mT).div_(2)
    return t

def random_hermitian_matrix(l, *batches, **kwargs):
    # [FUNCTION DOC] Return a random Hermitian matrix, or batch of them, of size l by l using
    # optional dtype and device keyword arguments (default double and cpu).
    dtype = kwargs.get('dtype', torch.double)
    device = kwargs.get('device', 'cpu')
    A = torch.randn(*(batches + (l, l)), dtype=dtype, device=device)
    A = (A + A.mH).div_(2)
    return A


def random_symmetric_psd_matrix(l, *batches, **kwargs):
    # [FUNCTION DOC] Return a batch of random symmetric positive semi-definite matrices of size
    # l by l, computed as a random matrix times its transpose.
    """
    Returns a batch of random symmetric positive-semi-definite matrices.
    The shape of the result is batch_dims + (matrix_size, matrix_size)
    The following example creates a tensor of size 2 x 4 x 3 x 3
    >>> # xdoctest: +SKIP("undefined variables")
    >>> matrices = random_symmetric_psd_matrix(3, 2, 4, dtype=dtype, device=device)
    """
    dtype = kwargs.get('dtype', torch.double)
    device = kwargs.get('device', 'cpu')
    A = torch.randn(*(batches + (l, l)), dtype=dtype, device=device)
    return A @ A.mT


def random_hermitian_psd_matrix(matrix_size, *batch_dims, dtype=torch.double, device='cpu'):
    # [FUNCTION DOC] Return a batch of random Hermitian positive semi-definite matrices of the
    # given size, computed as a random matrix times its conjugate transpose.
    """
    Returns a batch of random Hermitian positive-semi-definite matrices.
    The shape of the result is batch_dims + (matrix_size, matrix_size)
    The following example creates a tensor of size 2 x 4 x 3 x 3
    >>> # xdoctest: +SKIP("undefined variables")
    >>> matrices = random_hermitian_psd_matrix(3, 2, 4, dtype=dtype, device=device)
    """
    A = torch.randn(*(batch_dims + (matrix_size, matrix_size)), dtype=dtype, device=device)
    return A @ A.mH


# TODO: remove this (prefer make_symmetric_pd_matrices below)
def random_symmetric_pd_matrix(matrix_size, *batch_dims, **kwargs):
    # [FUNCTION DOC] Return a batch of random symmetric positive definite matrices, computed as
    # a random matrix times its transpose plus a small multiple of the identity. Deprecated in
    # favor of make_symmetric_pd_matrices.
    dtype = kwargs.get('dtype', torch.double)
    device = kwargs.get('device', 'cpu')
    A = torch.randn(*(batch_dims + (matrix_size, matrix_size)),
                    dtype=dtype, device=device)
    return torch.matmul(A, A.mT) \
        + torch.eye(matrix_size, dtype=dtype, device=device) * 1e-5


# Creates a symmetric positive-definite matrix or batch of
#   such matrices
def make_symmetric_pd_matrices(*shape, device, dtype):
    # [FUNCTION DOC] Return a random symmetric positive definite matrix or batch of matrices
    # with the given shape (random matrix times its transpose plus a small multiple of the
    # identity). Raises AssertionError if the last two dimensions differ.
    if shape[-1] != shape[-2]:
        raise AssertionError(f"expected square matrix, got shape[-1]={shape[-1]} != shape[-2]={shape[-2]}")
    t = make_tensor(shape, device=device, dtype=dtype)
    i = torch.eye(shape[-1], device=device, dtype=dtype) * 1e-5
    return t @ t.mT + i

def random_hermitian_pd_matrix(matrix_size, *batch_dims, dtype, device):
    # [FUNCTION DOC] Return a batch of random Hermitian positive definite matrices, computed as
    # a random matrix times its conjugate transpose plus the identity.
    """
    Returns a batch of random Hermitian positive-definite matrices.
    The shape of the result is batch_dims + (matrix_size, matrix_size)
    The following example creates a tensor of size 2 x 4 x 3 x 3
    >>> # xdoctest: +SKIP("undefined variables")
    >>> matrices = random_hermitian_pd_matrix(3, 2, 4, dtype=dtype, device=device)
    """
    A = torch.randn(*(batch_dims + (matrix_size, matrix_size)),
                    dtype=dtype, device=device)
    return A @ A.mH + torch.eye(matrix_size, dtype=dtype, device=device)

# Creates a full rank matrix with distinct singular values or
#   a batch of such matrices
def make_fullrank_matrices_with_distinct_singular_values(*shape, device, dtype, requires_grad=False):
    # [FUNCTION DOC] Return a full rank matrix or batch of matrices with distinct singular
    # values in the range of about 2/3 to 3/2 (alternating signs), built from the SVD of a
    # random tensor. The result is optionally marked as requiring grad.
    with torch.no_grad():
        t = make_tensor(shape, device=device, dtype=dtype)
        u, _, vh = torch.linalg.svd(t, full_matrices=False)
        real_dtype = t.real.dtype if t.dtype.is_complex else t.dtype
        k = min(shape[-1], shape[-2])
        # We choose the singular values to be "around one"
        # This is to make the matrix well conditioned
        # s = [2, 3, ..., k+1]
        s = torch.arange(2, k + 2, dtype=real_dtype, device=device)
        # s = [2, -3, 4, ..., (-1)^k k+1]
        s[1::2] *= -1.
        # 1 + 1/s so that the singular values are in the range [2/3, 3/2]
        # This gives a condition number of 9/4, which should be good enough
        s.reciprocal_().add_(1.)
        # Note that the singular values need not be ordered in an SVD so
        # we don't need to sort S
        x = (u * s.to(u.dtype)) @ vh
    x.requires_grad_(requires_grad)
    return x

def random_matrix(rows, columns, *batch_dims, **kwargs):
    # [FUNCTION DOC] Return a random rectangular matrix (or batch) with singular values between
    # about 1/(k+1) and 1, built from an SVD. Keyword arguments: dtype, device, silent (returns
    # ones if LAPACK is missing), and singular (zero out the smallest singular value or values
    # to make it singular). An empty matrix is returned unchanged.
    """Return rectangular matrix or batches of rectangular matrices.

    Parameters:
      dtype - the data type
      device - the device kind
      singular - when True, the output will be singular
    """
    dtype = kwargs.get('dtype', torch.double)
    device = kwargs.get('device', 'cpu')
    silent = kwargs.get("silent", False)
    singular = kwargs.get("singular", False)
    if silent and not torch._C.has_lapack:
        return torch.ones(rows, columns, dtype=dtype, device=device)

    A = torch.randn(batch_dims + (rows, columns), dtype=dtype, device=device)
    if A.numel() == 0:
        return A
    u, _, vh = torch.linalg.svd(A, full_matrices=False)
    k = min(rows, columns)
    s = torch.linspace(1 / (k + 1), 1, k, dtype=dtype, device=device)
    if singular:
        # make matrix singular
        s[k - 1] = 0
        if k > 2:
            # increase the order of singularity so that the pivoting
            # in LU factorization will be non-trivial
            s[0] = 0
    return (u * s.unsqueeze(-2)) @ vh


def random_matrix_with_scaled_reduction_dim(rows, columns, *batch_dims, **kwargs):
    # [FUNCTION DOC] Return a random rectangular matrix (or batch) of normally distributed
    # entries scaled by the square root of the size of the reduction dimension, so a product
    # along that dimension has unit variance. Keyword arguments: dtype, device, requires_grad
    # and reduction_dim (minus one or minus two).
    """Return rectangular matrix or batches of rectangular matrices
    with entries being iid and sampled from N(0, sigma^2) such that
    the variance of (A @ A.T)[..., i, j] is 1 if reduction_dim=-1, or
    the variance of (A.T @ A)[..., i, j] is 1 if reduction_dim=-2.

    Parameters:
      dtype - the data type
      device - the device kind
      requires_grad - whether output requires grad
      reduction_dim - the row/column dimension to re-scale.
                    Expected to be either -1 (columns) or -2 (rows).
    """
    dtype = kwargs.get('dtype', torch.double)
    device = kwargs.get('device', 'cpu')
    requires_grad = kwargs.get('requires_grad', False)
    reduction_dim = kwargs.get('reduction_dim', -1)

    shape = (*batch_dims, rows, columns)
    red_scale = math.sqrt(shape[reduction_dim])
    res = torch.randn(*shape, dtype=dtype, device=device) / red_scale
    res.requires_grad_(requires_grad)
    return res


def random_lowrank_matrix(rank, rows, columns, *batch_dims, **kwargs):
    # [FUNCTION DOC] Return a rectangular matrix (or batch) of the given rank as the product of
    # two random matrices; keyword arguments are forwarded to random_matrix.
    """Return rectangular matrix or batches of rectangular matrices with
    given rank.
    """
    B = random_matrix(rows, rank, *batch_dims, **kwargs)
    C = random_matrix(rank, columns, *batch_dims, **kwargs)
    return B.matmul(C)


def _generate_indices_prefer_all_rows(rows: int, cols: int, num_indices: int) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Generate a tensor of at most num_indices row and column index pairs
    # for a matrix, choosing the same number of random columns in each row (so each row gets an
    # entry if possible); pairs can overlap.
    """Generate indices for a row x cols matrix, preferring at least one index per row if possible."""
    indices = []  # type: ignore[var-annotated]
    n_per_row = math.ceil(num_indices / rows)
    col_indices = list(range(cols))

    for r in range(rows):
        # Note that this can yield overlapping indices
        indices.extend((r, c) for c in random.choices(col_indices, k=n_per_row))

    return torch.tensor(indices[:num_indices])


def random_sparse_matrix(rows, columns, density=0.01, **kwargs):
    # [FUNCTION DOC] Return a coalesced random sparse COO matrix with approximately the given
    # density but at least min(rows, columns) elements, with values decaying away from the
    # diagonal. Keyword arguments: dtype and device.
    """Return rectangular random sparse matrix within given density.

    The density of the result approaches to given density as the size
    of the matrix is increased and a relatively small value of density
    is specified but higher than min(rows, columns)/(rows * columns)
    for non-singular matrices.
    """
    dtype = kwargs.get('dtype', torch.double)
    device = kwargs.get('device', 'cpu')

    nonzero_elements = max(min(rows, columns), int(rows * columns * density))
    indices = _generate_indices_prefer_all_rows(rows, columns, nonzero_elements)
    values = torch.randn(nonzero_elements, dtype=dtype, device=device)

    # ensure that the diagonal dominates
    values *= torch.tensor([-float(i - j)**2 for i, j in indices], dtype=dtype, device=device).exp()
    A = torch.sparse_coo_tensor(indices.t(), values, (rows, columns), device=device)
    return A.coalesce()


def random_sparse_pd_matrix(matrix_size, density=0.01, **kwargs):
    # [FUNCTION DOC] Return a random sparse positive definite matrix with eigenvalues spread
    # between 1/n and 1. Starts from a diagonal matrix and repeatedly applies random plane
    # rotations (on both sides) until the number of nonzero entries reaches the target density.
    # Keyword arguments: torch (module override), dtype and device.
    """Return random sparse positive-definite matrix with given density.

    The eigenvalues of the matrix are defined as::
      arange(1, matrix_size+1)/matrix_size

    Algorithm:
      A = diag(arange(1, matrix_size+1)/matrix_size)
      while <A density is smaller than required>:
          <choose random i, j in range(matrix_size), theta in [0, 2*pi]>
          R = <rotation matrix (i,j,theta)>
          A = R^T A R
    """
    import math
    torch = kwargs.get('torch', globals()['torch'])
    dtype = kwargs.get('dtype', torch.double)
    device = kwargs.get('device', 'cpu')
    data = {(i, i): float(i + 1) / matrix_size
            for i in range(matrix_size)}


    def multiply(data, N, i, j, cs, sn, left=True):
        # [FUNCTION DOC] HELPER: Apply a plane rotation with the given cosine and sine to rows
        # (or columns) i and j of a sparse matrix stored as a mapping from index pairs to
        # values, removing entries that become zero. Mutates the mapping.
        for k in range(N):
            if left:
                ik, jk = (k, i), (k, j)
            else:
                ik, jk = (i, k), (j, k)
            aik, ajk = data.get(ik, 0), data.get(jk, 0)
            aik, ajk = cs * aik + sn * ajk, -sn * aik + cs * ajk
            if aik:
                data[ik] = aik
            else:
                data.pop(ik, None)
            if ajk:
                data[jk] = ajk
            else:
                data.pop(jk, None)

    target_nnz = density * matrix_size * matrix_size
    while len(data) < target_nnz:
        i = random.randint(0, matrix_size - 1)
        j = random.randint(0, matrix_size - 1)
        if i != j:
            theta = random.uniform(0, 2 * math.pi)
            cs = math.cos(theta)
            sn = math.sin(theta)
            multiply(data, matrix_size, i, j, cs, sn, left=True)
            multiply(data, matrix_size, i, j, cs, sn, left=False)
    icoords, jcoords, values = [], [], []
    for (i, j), v in sorted(data.items()):
        icoords.append(i)
        jcoords.append(j)
        values.append(v)
    indices_tensor = torch.tensor([icoords, jcoords])
    return torch.sparse_coo_tensor(indices_tensor, values, (matrix_size, matrix_size), dtype=dtype, device=device)

# FIXME: remove this by updating test suites using it
def do_test_dtypes(self, dtypes, layout, device):
    # [FUNCTION DOC] Legacy test helper: for each dtype except float16, create a zero tensor
    # with the layout and device and check the dtype, layout and device of the result.
    for dtype in dtypes:
        if dtype != torch.float16:
            out = torch.zeros((2, 3), dtype=dtype, layout=layout, device=device)
            self.assertIs(dtype, out.dtype)
            self.assertIs(layout, out.layout)
            self.assertEqual(device, out.device)

# FIXME: remove this by updating test suites using it
def do_test_empty_full(self, dtypes, layout, device):
    # [FUNCTION DOC] Legacy test helper: check dtype, layout, device, requires_grad and filled
    # values for the results of empty, full and their like and new variants, over the given
    # dtypes and layout and with or without requires_grad.
    shape = torch.Size([2, 3])

    def check_value(tensor, dtype, layout, device, value, requires_grad):
        # [FUNCTION DOC] HELPER: Assert the shape, dtype, layout, requires_grad, device (for
        # CUDA tensors) and, if a value is given, the filled content of a tensor.
        self.assertEqual(shape, tensor.shape)
        self.assertIs(dtype, tensor.dtype)
        self.assertIs(layout, tensor.layout)
        self.assertEqual(tensor.requires_grad, requires_grad)
        if tensor.is_cuda and device is not None:
            self.assertEqual(device, tensor.device)
        if value is not None:
            fill = tensor.new(shape).fill_(value)
            self.assertEqual(tensor, fill)

    def get_int64_dtype(dtype):
        # [FUNCTION DOC] HELPER: Return the int64 dtype from the same module as the given dtype
        # (for example the quantized or default namespace).
        module = '.'.join(str(dtype).split('.')[1:-1])
        if not module:
            return torch.int64
        return operator.attrgetter(module)(torch).int64

    default_dtype = torch.get_default_dtype()
    check_value(torch.empty(shape), default_dtype, torch.strided, -1, None, False)
    check_value(torch.full(shape, -5.), default_dtype, torch.strided, -1, None, False)
    for dtype in dtypes:
        for rg in {dtype.is_floating_point, False}:
            int64_dtype = get_int64_dtype(dtype)
            v = torch.empty(shape, dtype=dtype, device=device, layout=layout, requires_grad=rg)
            check_value(v, dtype, layout, device, None, rg)
            out = v.new()
            check_value(torch.empty(shape, out=out, device=device, layout=layout, requires_grad=rg),
                        dtype, layout, device, None, rg)
            check_value(v.new_empty(shape), dtype, layout, device, None, False)
            check_value(v.new_empty(shape, dtype=int64_dtype, device=device, requires_grad=False),
                        int64_dtype, layout, device, None, False)
            check_value(torch.empty_like(v), dtype, layout, device, None, False)
            check_value(torch.empty_like(v, dtype=int64_dtype, layout=layout, device=device, requires_grad=False),
                        int64_dtype, layout, device, None, False)

            if dtype is not torch.float16 and layout != torch.sparse_coo:
                fv = 3
                v = torch.full(shape, fv, dtype=dtype, layout=layout, device=device, requires_grad=rg)
                check_value(v, dtype, layout, device, fv, rg)
                check_value(v.new_full(shape, fv + 1), dtype, layout, device, fv + 1, False)
                out = v.new()
                check_value(torch.full(shape, fv + 2, out=out, device=device, layout=layout, requires_grad=rg),
                            dtype, layout, device, fv + 2, rg)
                check_value(v.new_full(shape, fv + 3, dtype=int64_dtype, device=device, requires_grad=False),
                            int64_dtype, layout, device, fv + 3, False)
                check_value(torch.full_like(v, fv + 4), dtype, layout, device, fv + 4, False)
                check_value(torch.full_like(v, fv + 5,
                                            dtype=int64_dtype, layout=layout, device=device, requires_grad=False),
                            int64_dtype, layout, device, fv + 5, False)

# FIXME: improve load_tests() documentation here
running_script_path = None  # type: ignore[var-annotated]
def set_running_script_path():
    # [FUNCTION DOC] Record the absolute resolved path of the running script (the first command
    # line argument) in the global, only if it ends in .py. Errors are ignored.
    global running_script_path
    try:
        running_file = os.path.abspath(os.path.realpath(sys.argv[0]))
        if running_file.endswith('.py'):  # skip if the running file is not a script
            running_script_path = running_file
    except Exception:
        pass

def check_test_defined_in_running_script(test_case):
    # [FUNCTION DOC] Raise AssertionError if the class of the given test case is defined in a
    # file other than the running script, which indicates an accidental import of a test case
    # from another file. Does nothing if the running script path is not known.
    if running_script_path is None:
        return
    test_case_class_file = os.path.abspath(os.path.realpath(inspect.getfile(test_case.__class__)))
    if test_case_class_file != running_script_path:
        raise AssertionError(
            f'Class of loaded TestCase "{test_case.id()}" '
            f'is not defined in the running script "{running_script_path}", but in "{test_case_class_file}". Did you '
            "accidentally import a unittest.TestCase from another file?"
        )

def load_tests(loader, tests, pattern):
    # [FUNCTION DOC] The load_tests protocol hook. Records the running script and returns a
    # suite with the non empty test groups, checking (unless disabled by the environment) that
    # each test is defined in the running script.
    set_running_script_path()
    test_suite = unittest.TestSuite()
    for test_group in tests:
        if not DISABLE_RUNNING_SCRIPT_CHK:
            for test in test_group:
                check_test_defined_in_running_script(test)
        if test_group._tests:
            test_suite.addTest(test_group)
    return test_suite

# FIXME: document this and move it to test_serialization
class BytesIOContext(io.BytesIO):
    # [CLASS DOC] HELPER: In-memory bytes buffer that can be used in a with statement without
    # being closed on exit (the exit method does nothing).
    def __enter__(self):
        # [FUNCTION DOC] HELPER: Return the buffer itself.
        return self

    def __exit__(self, *args):
        # [FUNCTION DOC] HELPER: Do nothing, leaving the buffer open.
        pass

# Tentative value for nondet_tol for gradcheck when backward implementation
# relies on nondeterministic operations, i.e., those listed here:
# https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
#
# For more information see https://github.com/pytorch/pytorch/issues/56202
GRADCHECK_NONDET_TOL = 1e-12

TEST_WITH_SLOW_GRADCHECK: bool = TestEnvironment.def_flag(
    "TEST_WITH_SLOW_GRADCHECK",
    env_var="PYTORCH_TEST_WITH_SLOW_GRADCHECK",
)

skipIfSlowGradcheckEnv = unittest.skipIf(
    TEST_WITH_SLOW_GRADCHECK,
    "Tests that don't use gradcheck don't need to run on slow_gradcheck CI",
)


def gradcheck(fn, inputs, **kwargs):
    # [FUNCTION DOC] Wrapper around autograd gradcheck that turns on batched grad checking and
    # fast mode by default (fast mode off when slow gradcheck is enabled); values passed as None
    # are replaced by these defaults. Returns the result of gradcheck.
    # Wrapper around gradcheck that enables certain keys by default.
    # Use this testing-internal gradcheck instead of autograd.gradcheck so that new features like vmap and
    # forward-mode AD are tested by default. We create this wrapper because we'd like to keep new checks
    # to be disabled to default for the public-facing api to avoid breaking user code.
    #
    # All PyTorch devs doing testing should use this wrapper instead of autograd.gradcheck.
    default_values = {
        "check_batched_grad": True,
        "fast_mode": True,
    }

    if TEST_WITH_SLOW_GRADCHECK:
        default_values["fast_mode"] = False

    for key, value in default_values.items():
        # default value override values explicitly set to None
        k = kwargs.get(key)
        kwargs[key] = k if k is not None else value

    return torch.autograd.gradcheck(fn, inputs, **kwargs)

def gradgradcheck(fn, inputs, grad_outputs=None, **kwargs):
    # [FUNCTION DOC] Wrapper around autograd gradgradcheck with the same defaults as gradcheck
    # (batched grad check on, fast mode unless slow gradcheck is enabled). Returns its result.
    # Wrapper around gradgradcheck that enables certain keys by default
    # See gradcheck above for an explanation of why we need something like this.
    #
    # All PyTorch devs doing testing should use this wrapper instead of autograd.gradgradcheck
    default_values = {
        "check_batched_grad": True,
        "fast_mode": True,
    }

    if TEST_WITH_SLOW_GRADCHECK:
        default_values["fast_mode"] = False

    for key, value in default_values.items():
        # default value override values explicitly set to None
        k = kwargs.get(key)
        kwargs[key] = k if k is not None else value

    return torch.autograd.gradgradcheck(fn, inputs, grad_outputs, **kwargs)


def _assertGradAndGradgradChecks(test_case, apply_fn, inputs, **kwargs):
    # [FUNCTION DOC] HELPER: Assert that both gradcheck and gradgradcheck pass for the function
    # and inputs, using the test case assertions.
    # call assert function rather than returning a bool since it's nicer
    # if we get whether this failed on the gradcheck or the gradgradcheck.
    test_case.assertTrue(gradcheck(apply_fn, inputs, **kwargs))
    test_case.assertTrue(gradgradcheck(apply_fn, inputs, **kwargs))


@contextmanager
def set_cwd(path: str) -> Iterator[None]:
    # [FUNCTION DOC] Context manager that changes the working directory to the given path for
    # the body and restores the previous directory afterwards.
    old_cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_cwd)


# FIXME: delete this
# Using @toleranceOverride specific to your test is the recommended way
# of doing this. These are just some values that worked for test_nn.
dtype2prec_DONTUSE = {torch.float: 1e-5,
                      torch.double: 1e-5,
                      torch.half: 1e-2,
                      torch.bfloat16: 1e-1}

# FIXME: move to test_sparse or sparse utils
# This is a wrapper that wraps a test to run this test twice, one with
# coalesced=True, another with coalesced=False for coalesced/uncoalesced sparse tensors.
def coalescedonoff(f):
    # [FUNCTION DOC] Decorator for sparse tests: runs the test twice, once with the coalesced
    # keyword true and once false.
    @wraps(f)
    def wrapped(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Call the test with coalesced true then false.
        f(self, *args, **kwargs, coalesced=True)
        f(self, *args, **kwargs, coalesced=False)
    return wrapped


def is_coalesced_indices(s):
    # [FUNCTION DOC] Return True if the indices of a sparse COO tensor are sorted in
    # lexicographic order and have no duplicates, using a hash of each index column.
    indices = s._indices()
    hash_coeffs = (1,) + s.shape[s.sparse_dim() - 1:0:-1]
    hash_indices = torch.tensor(hash_coeffs, device=s.device).cumprod(-1).flip(-1)
    if s.sparse_dim() > 1:
        hash_indices.unsqueeze_(-1)
        hash_indices = (indices * hash_indices).sum(0)
    else:
        hash_indices = indices * hash_indices

    # check if indices are sorted
    res = torch.allclose(hash_indices, hash_indices.sort()[0])

    # check if there are no repeated indices
    res = res and torch.allclose(hash_indices, hash_indices.unique())

    return res


@contextlib.contextmanager
def disable_gc():
    # [FUNCTION DOC] Context manager that disables garbage collection for the body if it is
    # enabled, and re-enables it afterwards.
    if gc.isenabled():
        try:
            gc.disable()
            yield
        finally:
            gc.enable()
    else:
        yield


def find_library_location(lib_name: str) -> Path:
    # [FUNCTION DOC] Return the path of a shared library: the file in the torch lib directory if
    # it exists, otherwise the file in the build lib directory of the source tree.
    # return the shared library file in the installed folder if exist,
    # else the file in the build folder
    torch_root = Path(torch.__file__).resolve().parent
    path = torch_root / 'lib' / lib_name
    if os.path.exists(path):
        return path
    torch_root = Path(__file__).resolve().parents[2]
    return torch_root / 'build' / 'lib' / lib_name

def skip_but_pass_in_sandcastle(reason):
    # [FUNCTION DOC] Return a decorator like unittest skip, except on Sandcastle where the test
    # instead runs as an empty pass after printing a message to stderr.
    """
    Similar to unittest.skip, however in the sandcastle environment it just
    "passes" the test instead to avoid creating tasks complaining about tests
    skipping continuously.
    """
    def decorator(func):
        # [FUNCTION DOC] HELPER: Mark the function as skipped (outside Sandcastle) or wrap it so
        # it prints a message and returns nothing (on Sandcastle).
        if not IS_SANDCASTLE:
            func.__unittest_skip__ = True
            func.__unittest_skip_why__ = reason
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Print the skip message on stderr and return
            # without running.
            print(f'Skipping {func.__name__} on sandcastle for following reason: {reason}', file=sys.stderr)
            return
        return wrapper

    return decorator

def mock_wrapper(method):
    # [FUNCTION DOC] Return a function that calls the real method and also records the call on a
    # mock object.
    """
    Returns a function that calls the real implementation of a method
    in addition to passing args to a mock object.
    """
    mock = MagicMock()

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Record the call arguments on the mock (without the instance),
        # then call the real method with the instance and arguments and return its result.
        mock(*args, **kwargs)
        return method(self, *args, **kwargs)
    wrapper.mock = mock  # type: ignore[attr-defined]
    return wrapper

def get_tensors_from(args, kwargs):
    # [FUNCTION DOC] Return a set of all Tensor objects found among the positional arguments and
    # the keyword argument values (not searching inside containers).
    """ Returns a set of all Tensor objects in the given args and kwargs. """
    return set([arg for arg in args if isinstance(arg, Tensor)] +
               [v for v in kwargs.values() if isinstance(v, Tensor)])


# Returns scalar tensor representation of a list of integer byte values
def bytes_to_scalar(byte_list: list[int], dtype: torch.dtype, device: torch.device):
    # [FUNCTION DOC] Build a scalar tensor from a list of byte values (each from 0 to 255)
    # interpreted in native byte order as the C type matching the dtype; complex dtypes need
    # twice the bytes (real part then imaginary part). Raises AssertionError on a wrong number
    # of bytes or out of range bytes, and KeyError for unsupported dtypes.
    dtype_to_ctype: dict[torch.dtype, Any] = {
        torch.int8: ctypes.c_int8,
        torch.uint8: ctypes.c_uint8,
        torch.uint16: ctypes.c_uint16,
        torch.uint32: ctypes.c_uint32,
        torch.uint64: ctypes.c_uint64,
        torch.int16: ctypes.c_int16,
        torch.int32: ctypes.c_int32,
        torch.int64: ctypes.c_int64,
        torch.bool: ctypes.c_bool,
        torch.float32: ctypes.c_float,
        torch.complex64: ctypes.c_float,
        torch.float64: ctypes.c_double,
        torch.complex128: ctypes.c_double,
    }
    ctype = dtype_to_ctype[dtype]
    num_bytes = ctypes.sizeof(ctype)

    def check_bytes(byte_list):
        # [FUNCTION DOC] HELPER: Raise AssertionError if any value is outside the range 0
        # to 255.
        for byte in byte_list:
            if not (0 <= byte <= 255):
                raise AssertionError(f"byte value out of range: expected 0 <= byte <= 255, got {byte}")

    if dtype.is_complex:
        if len(byte_list) != (num_bytes * 2):
            raise AssertionError(
                f"expected len(byte_list) == {num_bytes * 2} for complex dtype, got {len(byte_list)}"
            )
        check_bytes(byte_list)
        real = ctype.from_buffer((ctypes.c_byte * num_bytes)(
            *byte_list[:num_bytes])).value
        imag = ctype.from_buffer((ctypes.c_byte * num_bytes)(
            *byte_list[num_bytes:])).value
        res = real + 1j * imag
    else:
        if len(byte_list) != num_bytes:
            raise AssertionError(
                f"expected len(byte_list) == {num_bytes}, got {len(byte_list)}"
            )
        check_bytes(byte_list)
        res = ctype.from_buffer((ctypes.c_byte * num_bytes)(
            *byte_list)).value

    return torch.tensor(res, device=device, dtype=dtype)


def copy_func(f):
    # [FUNCTION DOC] Return a new function object sharing the code, globals, defaults and
    # closure of the given function, with its metadata and keyword defaults copied. Needed so
    # marking a copy as expected failure does not alter the original.
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__  # type: ignore[attr-defined]
    return g


def xfail_inherited_tests(tests):
    # [FUNCTION DOC] Return a class decorator that marks the listed inherited test methods as
    # expected failures on the decorated class, by replacing each with an expected failure copy
    # of the inherited function. Returns the class.
    """
    Given a list of test names which are defined by a superclass of the
    class this decorates, mark them as expected failure.  This is useful
    if you are doing poor man's parameterized tests by subclassing a generic
    test class.
    """
    def deco(cls):
        # [FUNCTION DOC] HELPER: Apply the expected failure marking to each named test of
        # the class.
        for t in tests:
            # NB: expectedFailure operates by mutating the method in question,
            # which is why you have to copy the function first
            setattr(cls, t, unittest.expectedFailure(copy_func(getattr(cls, t))))
        return cls
    return deco


def skip_but_pass_in_sandcastle_if(condition, reason):
    # [FUNCTION DOC] Return a decorator like unittest skipIf which, on Sandcastle, replaces a
    # test that would be skipped by a function that prints a message to stderr and passes.
    # Outside Sandcastle it marks the function skipped with the reason.
    """
    Similar to unittest.skipIf, however in the sandcastle environment it just
    "passes" the test instead to avoid creating tasks complaining about tests
    skipping continuously.
    """
    def decorator(func):
        # [FUNCTION DOC] HELPER: If the condition holds, on Sandcastle return a printing no-op
        # wrapper, otherwise mark the function skipped; return the function unchanged otherwise.
        if condition:
            if IS_SANDCASTLE:
                @wraps(func)
                def wrapper(*args, **kwargs):
                    # [FUNCTION DOC] HELPER: Print the skip message to stderr and return without
                    # running the test.
                    print(f'Skipping {func.__name__} on sandcastle for following reason: {reason}', file=sys.stderr)
                return wrapper
            else:
                func.__unittest_skip__ = True
                func.__unittest_skip_why__ = reason

        return func

    return decorator

def dtype_name(dtype):
    # [FUNCTION DOC] Return the short name of a dtype, the part of its string form after the
    # module prefix (for example int64).
    """ Returns the pretty name of the dtype (e.g. torch.int64 -> int64). """
    return str(dtype).split('.')[1]


def _cpu_sleep(cycles: int) -> None:
    # [FUNCTION DOC] HELPER: Busy loop for the given number of iterations, standing in for a
    # device sleep on CPU.
    """Spin-wait for approximately the given number of cycles."""
    for _ in range(cycles):
        pass


def device_sleep(device: str, cycles: int) -> None:
    # [FUNCTION DOC] Sleep for a number of cycles on a device. For CPU, temporarily installs a
    # spin loop as the cpu sleep function and removes or restores it afterwards; for other
    # devices calls the sleep function of the device module.
    """Sleep for the given number of cycles on the specified device.

    For CPU, temporarily patches torch.cpu._sleep if needed.
    For CUDA/other devices, uses torch.get_device_module(device)._sleep.
    """
    if device == "cpu":
        orig = getattr(torch.cpu, "_sleep", None)
        torch.cpu._sleep = _cpu_sleep
        try:
            torch.cpu._sleep(cycles)
        finally:
            if orig is None:
                delattr(torch.cpu, "_sleep")
            else:
                torch.cpu._sleep = orig
    else:
        torch.get_device_module(device)._sleep(cycles)


@functools.lru_cache
def get_cycles_per_ms(device: str = "cuda") -> float:
    # [FUNCTION DOC] Measure and return the approximate number of device sleep cycles per
    # millisecond for a device type. Takes ten measurements (timed with wall clock for CPU, with
    # device events otherwise; a fixed estimate when the device has no sleep), drops the two
    # smallest and two largest, and averages the rest. The result is cached per device.
    """Measure and return approximate number of cycles per millisecond for device _sleep.

    Args:
        device: Device type to measure cycles for ("cuda" or "cpu").

    Works for both CUDA (torch.cuda._sleep) and CPU (torch.cpu._sleep).
    """
    test_cycles = 1000000

    if device == "cpu":
        import time

        def measure() -> float:
            # [FUNCTION DOC] HELPER: Time a CPU spin loop and return cycles per millisecond.
            start = time.perf_counter()
            _cpu_sleep(test_cycles)
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            cycles_per_ms = test_cycles / elapsed_ms if elapsed_ms > 0 else 1000000
            return cycles_per_ms
    else:
        def measure() -> float:
            # [FUNCTION DOC] HELPER: Time a device sleep with events and return cycles per
            # millisecond, or a fixed default value of one million when the device module has no
            # sleep function.
            if hasattr(torch.get_device_module(device), "_sleep"):
                start = torch.Event(enable_timing=True)
                end = torch.Event(enable_timing=True)
                start.record()
                torch.get_device_module(device)._sleep(test_cycles)
                end.record()
                end.synchronize()
                cycles_per_ms = test_cycles / start.elapsed_time(end)
                return cycles_per_ms
            else:
                cycles_per_ms = 1000000.0
                return cycles_per_ms

    # Get 10 values and remove the 2 max and 2 min and return the avg.
    # This is to avoid system disturbance that skew the results, e.g.
    # the very first cuda call likely does a bunch of init, which takes
    # much longer than subsequent calls.
    #
    # Tested on both Tesla V100, Quadro GP100, Titan RTX, RTX 3090 GPUs
    # and seems to return stable values. Therefore, we enable caching
    # using lru_cache decorator above.
    num = 10
    vals = [measure() for _ in range(num)]
    vals = sorted(vals)
    return mean(vals[2 : num - 2])


# OpInfo utils

T = TypeVar('T')
def first_sample(self: unittest.TestCase, samples: Iterable[T]) -> T:
    # [FUNCTION DOC] Return the first item of an iterable of samples. Raises unittest.SkipTest
    # when the iterable is empty. The test case argument is unused.
    """
    Returns the first sample from an iterable of samples, like those returned by OpInfo.
    The test will be skipped if no samples are available.
    """
    try:
        return next(iter(samples))
    except StopIteration as e:
        raise unittest.SkipTest('Skipped! Need at least 1 sample input') from e

# this helper method is to recursively
# clone the tensor-type input of operators tested by OpInfo
def clone_input_helper(input):
    # [FUNCTION DOC] Recursively clone tensor inputs of an OpInfo test: tensors are cloned,
    # sequences become tuples of cloned elements, other values are returned unchanged.
    if isinstance(input, torch.Tensor):
        return torch.clone(input)

    if isinstance(input, Sequence):
        return tuple(map(clone_input_helper, input))

    return input

@contextmanager
def custom_op(opname, symbolic_fn, opset_version):
    # [FUNCTION DOC] Context manager that registers a custom ONNX symbolic function for an
    # operator at an opset version for the body, and unregisters it afterwards (even on error).
    """Context manager/decorator to test ONNX export with custom operator"""
    try:
        register_custom_op_symbolic(opname, symbolic_fn, opset_version)
        yield
    finally:
        unregister_custom_op_symbolic(opname, opset_version)


def outs_and_grads(fn, graph_inps, inps):
    # [FUNCTION DOC] Run a function on the given inputs, backpropagate a sum of each output that
    # requires grad (retaining the graph), collect the gradients of the tensor leaves of the
    # second inputs argument, clear those gradients, and return a pair of outputs and gradients.
    # Mutates the grad attributes of the inputs.
    outs = fn(*graph_inps)
    for out in pytree.tree_leaves(outs):
        if isinstance(out, torch.Tensor) and out.requires_grad:
            out.sum().backward(retain_graph=True)
    grads = [inp.grad for inp in pytree.tree_leaves(inps) if isinstance(inp, torch.Tensor)]
    for inp in pytree.tree_leaves(inps):
        if isinstance(inp, torch.Tensor):
            inp.grad = None
    return outs, grads

def compare_equal_outs_and_grads(test, m1, m2, inps):
    # [FUNCTION DOC] Run two callables on the same inputs through outs_and_grads and assert
    # (with the test case) that their outputs and gradients are equal.
    r1, g1 = outs_and_grads(m1, inps, inps)
    r2, g2 = outs_and_grads(m2, inps, inps)
    test.assertEqual(r1, r2)
    test.assertEqual(g1, g2)

class TestGradients(TestCase):
    # [CLASS DOC] Test case base class for gradient tests over OpInfo operators. Provides the
    # shared helper that runs gradcheck, gradgradcheck or forward over reverse checks for every
    # sample input of an operator and variant, plus skip helpers. State: the class attribute
    # exact_dtype set to true.
    exact_dtype = True

    # Copies inputs to inplace operations to avoid inplace modifications
    #   to leaves requiring gradient
    def _get_safe_inplace(self, inplace_variant):
        # [FUNCTION DOC] HELPER: Wrap an in-place variant so it operates on a clone of its first
        # argument, protecting leaf tensors that require grad.
        @wraps(inplace_variant)
        def _fn(t, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Call the in-place variant on a clone of the tensor with the
            # other arguments and return the result.
            return inplace_variant(t.clone(), *args, **kwargs)

        return _fn

    def _check_helper(self, device, dtype, op, variant, check, *, check_forward_ad=False, check_backward_ad=True,
                      check_batched_grad=None, check_batched_forward_grad=False):
        # [FUNCTION DOC] HELPER: Run one of the gradient checks (gradcheck, bwgrad_bwgrad, or
        # fwgrad_bwgrad) for every sample input of the operator for the variant, device and
        # dtype. Skips the test if the variant is missing or the dtype is unsupported, skips
        # in-place variants for broadcasting samples, and fails if a sample tensor already
        # carries a gradient. Builds a function over just the tensors requiring grad that
        # recomposes the original input. Raises AssertionError for an unknown check name.
        if check not in ('gradcheck', 'bwgrad_bwgrad', 'fwgrad_bwgrad'):
            raise AssertionError(
                f"check must be one of ('gradcheck', 'bwgrad_bwgrad', 'fwgrad_bwgrad'), got {check!r}"
            )
        # NB: check_backward_ad does not affect gradgradcheck (always True)
        if variant is None:
            self.skipTest("Skipped! Variant not implemented.")
        if not op.supports_dtype(dtype, torch.device(device).type):
            self.skipTest(f"Skipped! {op.name} does not support dtype {str(dtype)}")

        def is_inplace(variant):
            # [FUNCTION DOC] HELPER: Return True if the variant (or the function it wraps) is
            # the operator's in-place variant.
            if hasattr(variant, "__wrapped__"):
                return variant.__wrapped__ is op.get_inplace()
            return variant is op.get_inplace()

        include_conjugated_inputs = op.test_conjugated_samples and dtype.is_complex

        samples = op.sample_inputs(device, dtype, requires_grad=True, include_conjugated_inputs=include_conjugated_inputs,
                                   small_inputs_only=TEST_WITH_SLOW_GRADCHECK)

        for sample in samples:
            if sample.broadcasts_input and is_inplace(variant):
                continue

            # Gradcheck expects tensors as its input, but autograd actually supports tensorlists
            #   and tensors passed as kwargs. The following creates a function that accepts just
            #   the tensors that require grad as varargs, and then recomposes them back into the
            #   original input.

            # Creates gradcheck inputs by identifying tensors requiring grad
            all_args = None
            if is_iterable_of_tensors(sample.input):
                all_args = chain(sample.input, sample.args, sample.kwargs.values())
            else:
                all_args = tuple(chain((sample.input,), sample.args, sample.kwargs.values()))  # type: ignore[assignment]
            gradcheck_args = tuple(x for x in all_args if (isinstance(x, torch.Tensor) and x.requires_grad))  # type: ignore[union-attr]

            # Verifies sample input tensors should have no grad
            # This may happen if the same tensor is used in two different SampleInputs
            for t in gradcheck_args:
                self.assertIsNone(t.grad,
                                  "A sampled input has a gradient before running autograd. "
                                  "This usually means that (at least) one input tensor is reused "
                                  "across different SampleInputs. "
                                  "Please create a new tensor for each SampleInput.")

            def _input_recomposition_helper(inputs, inp, input_idx):
                # [FUNCTION DOC] HELPER: Take the next tensors requiring grad from the flat
                # inputs to rebuild one original argument (a single tensor or a list of
                # tensors); return the rebuilt argument and the updated index.
                if is_iterable_of_tensors(inp):
                    tensor_list = []
                    for x in inp:
                        if isinstance(x, torch.Tensor) and x.requires_grad:
                            tensor_list.append(inputs[input_idx])
                            input_idx = input_idx + 1
                        else:
                            tensor_list.append(x)
                    return tensor_list, input_idx
                elif isinstance(inp, torch.Tensor) and inp.requires_grad:
                    return inputs[input_idx], input_idx + 1
                else:
                    return inp, input_idx

            def fn(*inputs):
                # [FUNCTION DOC] HELPER: The function under test for the gradient check: puts
                # the flat tensor inputs back into positional and keyword arguments of the
                # sample, calls the operator's wrapper on the variant, and applies the sample's
                # output processing function when present.
                # Puts inputs back into sample properly
                positional_args = []
                input_idx = 0
                inp, input_idx = _input_recomposition_helper(inputs, sample.input, input_idx)
                positional_args.append(inp)

                for x in sample.args:
                    inp, input_idx = _input_recomposition_helper(inputs, x, input_idx)
                    positional_args.append(inp)

                # Recreates kwargs
                kwargs = {}
                for k, v in sample.kwargs.items():
                    inp, input_idx = _input_recomposition_helper(inputs, v, input_idx)
                    kwargs[k] = inp

                output = op.gradcheck_wrapper(variant, *positional_args, **kwargs)
                if sample.output_process_fn_grad is not None:
                    return sample.output_process_fn_grad(output)
                return output

            if check == 'gradcheck':
                if check_batched_grad is None:
                    check_batched_grad = op.check_batched_grad
                self.assertTrue(gradcheck(fn, gradcheck_args,
                                          check_batched_grad=check_batched_grad,
                                          check_grad_dtypes=True,
                                          nondet_tol=op.gradcheck_nondet_tol,
                                          fast_mode=op.gradcheck_fast_mode,
                                          check_forward_ad=check_forward_ad,
                                          check_backward_ad=check_backward_ad,
                                          check_undefined_grad=True,
                                          check_batched_forward_grad=check_batched_forward_grad))
            elif check in ('bwgrad_bwgrad', 'fwgrad_bwgrad'):  # gradgrad check
                self.assertFalse(check_forward_ad, msg="Cannot run forward AD check for gradgradcheck")
                for gen_non_contig_grad_outputs in (False, True):
                    kwargs = {
                        "gen_non_contig_grad_outputs": gen_non_contig_grad_outputs,
                        "check_batched_grad": op.check_batched_gradgrad,
                        "check_grad_dtypes": True,
                        "nondet_tol": op.gradcheck_nondet_tol,
                        "fast_mode": op.gradcheck_fast_mode
                    }
                    if check == "fwgrad_bwgrad":
                        kwargs["check_fwd_over_rev"] = True
                        kwargs["check_rev_over_rev"] = False
                        kwargs["check_batched_grad"] = False
                        kwargs["check_undefined_grad"] = False

                    self.assertTrue(gradgradcheck(fn, gradcheck_args, **kwargs))
            else:
                self.assertTrue(False, msg="Unknown check requested!")

    def _grad_test_helper(self, device, dtype, op, variant, *, check_forward_ad=False, check_backward_ad=True,
                          check_batched_grad=None, check_batched_forward_grad=False):
        # [FUNCTION DOC] HELPER: Run the gradcheck variant of the gradient check with the given
        # forward and backward AD and batched grad options.
        return self._check_helper(device, dtype, op, variant, 'gradcheck', check_forward_ad=check_forward_ad,
                                  check_backward_ad=check_backward_ad, check_batched_grad=check_batched_grad,
                                  check_batched_forward_grad=check_batched_forward_grad)

    def _skip_helper(self, op, device, dtype):
        # [FUNCTION DOC] HELPER: Skip the test if the operator does not support backward for the
        # dtype or supports neither autograd nor forward AD.
        if dtype not in op.supported_backward_dtypes(torch.device(device).type):
            self.skipTest("Skipped! Op doesn't support autograd for this dtype.")
        if not op.supports_autograd and not op.supports_forward_ad:
            self.skipTest("Skipped! autograd not supported.")




# Base TestCase for NT tests; used to define common helpers, etc.
class NestedTensorTestCase(TestCase):
    # [CLASS DOC] Base TestCase for nested tensor tests with common helpers for comparing
    # nested tensors, including jagged ones, and for saving and restoring the nested tensor
    # global registries.
    def assertEqualIgnoringNestedInts(self, a, b):
        # [FUNCTION DOC] Assert two structures are equal after unbinding jagged nested tensors
        # into lists of tensors, so nested integer symbols are not compared.
        # unbinding NJTs allows us to compare them as essentially equal without
        # caring about exact nested int comparison
        def _unbind_njts(x):
            # [FUNCTION DOC] HELPER: Unbind a jagged nested tensor, return other
            # values unchanged.
            if isinstance(x, torch.Tensor) and x.is_nested and x.layout == torch.jagged:
                return x.unbind()
            else:
                return x

        self.assertEqual(pytree.tree_map(_unbind_njts, a), pytree.tree_map(_unbind_njts, b))

    def assertEqualNoncontigAware(self, a, b):
        # [FUNCTION DOC] Assert equality of two structures by unbinding jagged nested tensors
        # and also comparing the shapes of nested tensors, since plain equality ignores lengths.
        # assertEqual() doesn't take into account lengths, so hack around this
        # by comparing unbound components and shapes
        self.assertEqualIgnoringNestedInts(a, b)

        def _get_njt_shapes(x):
            # [FUNCTION DOC] HELPER: Return the shape of a nested tensor, None for any
            # other value.
            return (
                x.shape
                if isinstance(x, torch.Tensor) and x.is_nested
                else None
            )

        a_shapes = pytree.tree_map(_get_njt_shapes, a)
        b_shapes = pytree.tree_map(_get_njt_shapes, b)
        self.assertEqual(a_shapes, b_shapes)

    @contextlib.contextmanager
    def branch_nested_state(self):
        # [FUNCTION DOC] Context manager that remembers the nested tensor symint registry and
        # tensor id counter, and restores them after the body, to branch the nested state.
        """Context manager to branch and restore the nested tensor state."""
        nested_tensor_module = torch.nested._internal.nested_tensor
        original_tensor_symint_registry = nested_tensor_module._tensor_symint_registry.copy()
        original_tensor_id_counter = nested_tensor_module._tensor_id_counter
        try:
            yield
        finally:
            nested_tensor_module._tensor_id_counter = original_tensor_id_counter
            nested_tensor_module._tensor_symint_registry = original_tensor_symint_registry


def munge_exc(e, *, suppress_suffix=True, suppress_prefix=True, file=None, skip=0, strip_carets=True, strip_stack_attribution=True):
    # [FUNCTION DOC] Normalize the text of an exception so it can be compared against inline
    # expected output. Removes stack frames outside the given file (default: the caller's file,
    # with skip extra frames), replaces line numbers, local paths and the torch install prefix,
    # rewrites a bytecode name for compatibility, and optionally strips trailing logging hints,
    # a prefix line, source attribution blocks and caret lines. Returns the normalized string.
    # KNOWN ISSUE: the file path is used as a regular expression pattern without escaping.
    from torch._dynamo.trace_rules import _as_posix_path

    if file is None:
        file = inspect.stack()[1 + skip].filename  # skip one frame

    file = _as_posix_path(file)
    s = _as_posix_path(str(e))

    # Remove everything that looks like stack frames in NOT this file
    def repl_frame(m):
        # [FUNCTION DOC] HELPER: Replacement for a regex match of a stack frame: keep it only if
        # it belongs to the given file and is not the top level module code; otherwise drop it.
        if m.group(2) != file:
            return ""
        # Don't accept top-level, even for this script, these will wobble
        # depending on how the testing script was invoked
        if m.group(3) == "<module>":
            return ""

        return m.group(0)

    s = re.sub(
        r'( *)File "([^"]+)", line \d+, in (.+)\n(\1  .+\n( +[~^]* *\n)?)*',
        repl_frame,
        s,
    )
    s = re.sub(r"line \d+", "line N", s)
    s = re.sub(r".py:\d+", ".py:N", s)
    s = re.sub(r'https:/([a-zA-Z0-9_.-]+)', r'https://\1', s)
    s = re.sub(file, _as_posix_path(os.path.basename(file)), s)
    s = re.sub(_as_posix_path(os.path.join(os.path.dirname(torch.__file__), "")), "", s)
    # 3.10 CALL_FUNCTION bytecode compatibility for dynamo graph break messages
    s = re.sub(
        r"attempting to trace CALL_FUNCTION:.*$",
        "attempting to trace CALL: a function call, e.g. f(x, y):",
        s,
        flags=re.MULTILINE,
    )
    if suppress_suffix:
        s = re.sub(r"\n*Set TORCH_LOGS.+", "", s, flags=re.DOTALL)
        s = re.sub(r"\n*You can suppress this exception.+", "", s, flags=re.DOTALL)
        s = re.sub(r"\n*Set TORCHDYNAMO_VERBOSE=1.+", "", s, flags=re.DOTALL)
    if suppress_prefix:
        s = re.sub(r"Cannot export model.+\n\n", "", s)
    s = re.sub(r" +$", "", s, flags=re.MULTILINE)
    if strip_stack_attribution:
        # Strip the contents of "Stack variable source attribution" blocks but
        # keep the header, since the entries depend on whether specific bytecodes
        # have position info which varies across Python point releases.
        s = re.sub(
            r"(\nStack variable source attribution:)\n(?:.*\n)*?\n",
            r"\1\n\n",
            s,
        )
    if strip_carets:
        # Remove caret/tilde indicator lines (e.g. "    ~~~^^^^") since their
        # presence and alignment vary across Python versions.
        s = re.sub(r"\n[ ~^]*[~^][ ~^]*(?=\n|\Z)", "", s)
    return s


@contextmanager
def check_leaked_tensors(limit=1, matched_type=torch.Tensor):
    # [FUNCTION DOC] Context manager yielding a list that, after the body, is extended with
    # tensors (or the matched type) found in the garbage collector's uncollectable objects
    # (debug save-all mode is enabled during the body). Warns if any were found and tries to
    # draw back references with objgraph, limited to limit objects. The garbage collector debug
    # flags are reset to zero at the end.
    """Wrap around operations you want to ensure are not leaking tensor memory.

    This code intentionally ignores other reference cycles, which can be benign and which we have plenty
    of in pytorch code.  It focuses on any reference cycles that directly or indirectly result holding a Tensor alive,
    since this is likely a more serious leak than typical python refcycles.

    limit specifies how many tensors to dump debug graphs for (default=1)
    """
    def match_obj(obj):
        # [FUNCTION DOC] HELPER: Return True if the object is an instance of the matched type.
        return isinstance(obj, matched_type)

    try:
        gc.collect()
        gc.set_debug(gc.DEBUG_SAVEALL)
        garbage_objs = []  # type: ignore[var-annotated]

        # run the user code, after cleaning any existing refcycles, and then check for new ones
        # also allow usercode to check the garbage objs (e.g. for assertion) after exiting ctxmgr
        yield garbage_objs

        gc.collect()
        garbage_objs.extend(filter(match_obj, gc.garbage))
        num_garbage_objs = len(garbage_objs)
        if num_garbage_objs > 0:
            warnings.warn(
                f"{num_garbage_objs} tensors were found in the garbage. Did you introduce a reference cycle?", stacklevel=2
            )
            try:
                import objgraph  # type: ignore[import-not-found,import-untyped]
                warnings.warn(
                    f"Dumping first {limit} objgraphs of leaked {matched_type}s rendered to png", stacklevel=2
                )
                for g in garbage_objs[:limit]:
                    objgraph.show_backrefs([g], max_depth=10)
            except ImportError:
                warnings.warn("`pip install objgraph` to enable memory leak debugging", stacklevel=2)

    finally:
        gc.set_debug(0)


def remove_cpp_extensions_build_root():
    # [FUNCTION DOC] Delete the default build root directory for extensions, if it exists (with
    # a shell command on Windows to avoid permission errors). Errors are ignored elsewhere.
    """
    Removes the default root folder under which extensions are built.
    """
    default_build_root = cpp_extension.get_default_build_root()
    if os.path.exists(default_build_root):
        if IS_WINDOWS:
            # rmtree returns permission error: [WinError 5] Access is denied
            # on Windows, this is a workaround
            subprocess.run(["rm", "-rf", default_build_root], stdout=subprocess.PIPE)
        else:
            shutil.rmtree(default_build_root, ignore_errors=True)


def install_cpp_extension(extension_root):
    # [FUNCTION DOC] Build and install a C++ extension package from a directory into a local
    # install folder with pip (removing old build and install folders), run through the shell
    # helper, then add the resulting packages directory to the module search path. Raises
    # RuntimeError if the build fails or no packages directory is found.
    # Wipe the build / install dirs if they exist
    build_dir = os.path.join(extension_root, "build")
    install_dir = os.path.join(extension_root, "install")
    for d in (build_dir, install_dir):
        if os.path.exists(d):
            shutil.rmtree(d)

    # Build the extension
    cmd = [sys.executable, "-m", "pip", "install", extension_root, "-v", "--no-build-isolation", "--root", install_dir]
    return_code = shell(cmd, cwd=extension_root, env=os.environ)
    if return_code != 0:
        raise RuntimeError(f"build failed for cpp extension at {extension_root}")

    mod_install_dir = None
    # install directory is the one that is named site-packages
    for root, directories, _ in os.walk(install_dir):
        for directory in directories:
            if "-packages" in directory:
                mod_install_dir = os.path.join(root, directory)

    if mod_install_dir is None:
        raise RuntimeError(f"installation failed for cpp extension at {extension_root}")

    if mod_install_dir not in sys.path:
        sys.path.insert(0, mod_install_dir)


# Decorator to provide a helper to load inline extensions to a temp directory
def scoped_load_inline(func):
    # [FUNCTION DOC] Decorator that supplies a load_inline keyword argument to the test
    # function. The supplied function builds inline extensions in a temporary directory
    # (directly on Windows). Returns the wrapped function.

    @wraps(func)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Call the function with the generated load_inline helper added.
        def load_inline(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Load an inline extension in a fresh temporary build
            # directory. Raises AssertionError if the caller passes a build directory.
            if IS_WINDOWS:
                # TODO(xmfan): even using TemporaryDirectoryName will result in permission error
                return cpp_extension.load_inline(*args, **kwargs)

            if "build_directory" in kwargs:
                raise AssertionError("build_directory should not be specified when using scoped_load_inline")
            with TemporaryDirectoryName() as temp_dir_name:
                if kwargs.get("verbose", False):
                    print(f'Using temporary extension directory {temp_dir_name}...', file=sys.stderr)
                kwargs["build_directory"] = temp_dir_name
                return cpp_extension.load_inline(*args, **kwargs)

        return func(*args, load_inline=load_inline, **kwargs)
    return wrapper

# Single source of truth for which globals count as "fp32 precision state".
# Add a new flag here and both recover_orig_fp32_precision and the TestCase
# leak detector pick it up automatically.
def _fp32_precision_flag_specs():
    # [FUNCTION DOC] HELPER: Return the single table of the six fp32 precision settings (CUDA
    # matmul, cudnn conv and rnn, mkldnn matmul, conv and rnn), each as a label, a getter
    # function and a setter function. It drives snapshots, restoration and leak checks.
    return (
        ("torch.backends.cuda.matmul.fp32_precision",
            lambda: torch.backends.cuda.matmul.fp32_precision,
            lambda v: setattr(torch.backends.cuda.matmul, "fp32_precision", v)),
        ("torch.backends.cudnn.conv.fp32_precision",
            lambda: torch.backends.cudnn.conv.fp32_precision,  # type: ignore[attr-defined]
            lambda v: setattr(torch.backends.cudnn.conv, "fp32_precision", v)),
        ("torch.backends.cudnn.rnn.fp32_precision",
            lambda: torch.backends.cudnn.rnn.fp32_precision,  # type: ignore[attr-defined]
            lambda v: setattr(torch.backends.cudnn.rnn, "fp32_precision", v)),
        ("torch.backends.mkldnn.matmul.fp32_precision",
            lambda: torch.backends.mkldnn.matmul.fp32_precision,  # type: ignore[attr-defined]
            lambda v: setattr(torch.backends.mkldnn.matmul, "fp32_precision", v)),
        ("torch.backends.mkldnn.conv.fp32_precision",
            lambda: torch.backends.mkldnn.conv.fp32_precision,  # type: ignore[attr-defined]
            lambda v: setattr(torch.backends.mkldnn.conv, "fp32_precision", v)),
        ("torch.backends.mkldnn.rnn.fp32_precision",
            lambda: torch.backends.mkldnn.rnn.fp32_precision,  # type: ignore[attr-defined]
            lambda v: setattr(torch.backends.mkldnn.rnn, "fp32_precision", v)),
    )


def _snapshot_fp32_precision():
    # [FUNCTION DOC] HELPER: Return a tuple with the current value of every fp32 precision flag.
    return tuple(get() for _, get, _ in _fp32_precision_flag_specs())


def _restore_fp32_precision(snapshot):
    # [FUNCTION DOC] HELPER: Set all fp32 precision flags back to the values in a snapshot.
    # Raises TypeError if the snapshot is not a tuple of the expected length.
    specs = _fp32_precision_flag_specs()
    if not isinstance(snapshot, tuple) or len(snapshot) != len(specs):
        # Catches accidental shadowing of self._prev_fp32_precision by a
        # subclass (e.g. assigning a scalar); without this check zip() would
        # silently iterate over the wrong sequence.
        raise TypeError(
            f"fp32 precision snapshot must be a {len(specs)}-tuple, "
            f"got {type(snapshot).__name__} of length "
            f"{len(snapshot) if hasattr(snapshot, '__len__') else 'unknown'}"
        )
    for (_, _, set_), value in zip(specs, snapshot):
        set_(value)


def recover_orig_fp32_precision(fn):
    # [FUNCTION DOC] Decorate a function so that the fp32 precision flags are saved before each
    # call and restored afterwards, even if the function fails.
    @contextlib.contextmanager
    def recover():
        # [FUNCTION DOC] HELPER: Context manager that snapshots the fp32 precision flags and
        # restores them on exit.
        snap = _snapshot_fp32_precision()
        try:
            yield
        finally:
            _restore_fp32_precision(snap)

    return recover()(fn)

def skipIfPythonVersionMismatch(predicate):
    # [FUNCTION DOC] Return a lazy skip decorator that skips when the predicate, called with the
    # major, minor and micro Python version numbers, returns false.
    vi = sys.version_info
    return lazy_skip_if(
        lambda: not predicate(vi.major, vi.minor, vi.micro), "Python version mismatch"
    )

# Decorator to patch multiple test class members for the duration of the subtest
def patch_test_members(updates: dict[str, Any]):
    # [FUNCTION DOC] Return a decorator that temporarily sets the given attributes on the test
    # instance for the duration of the test and restores the original values afterwards.
    def decorator(test_func):
        # [FUNCTION DOC] HELPER: Wrap the test function with the member patching logic.
        @wraps(test_func)
        def wrapper(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Save the original attribute values, set the new ones, run
            # the test and return its result, and restore the originals in all cases.
            # Store the original values of the specified members
            original_values = {member: getattr(self, member) for member in updates}

            # Update the members before running the subtest
            for member, value in updates.items():
                setattr(self, member, value)

            # Run the test function, allowing subtests to run
            try:
                return test_func(self, *args, **kwargs)
            finally:
                # Restore the original values of the specified members after the subtest finishes
                for member, original_value in original_values.items():
                    setattr(self, member, original_value)

        return wrapper
    return decorator

def get_gcc_major_version():
    # [FUNCTION DOC] Return the major version of the installed gcc as an int, or None if gcc
    # cannot be run or its output cannot be parsed.
    """
    Return GCC major version as int, or None if GCC is not available.
    """
    try:
        out = subprocess.check_output(
            ["gcc", "-dumpfullversion", "-dumpversion"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        return int(out.split(".")[0])
    except Exception:
        return None


def run_concurrently(worker_func, num_threads=None, args=(), kwargs=None):
    # [FUNCTION DOC] Run a worker function (or one function per thread, if a list is given) in
    # multiple threads that wait on a barrier so they start together, to expose thread safety
    # bugs. Returns the list of results per thread. If any worker raised, the last recorded
    # exception is re-raised after all threads have joined. Extra args and kwargs are passed on.
    # KNOWN ISSUE: when a single callable is given and num_threads is omitted, taking the length
    # of the callable fails with a TypeError.
    # Adapted from CPython test suite. Runs worker_func in multiple threads
    # concurrently to help expose thread-safety issues. Works best in
    # combination with ThreadSanitizer (TSan).
    from collections.abc import Iterable

    if kwargs is None:
        kwargs = {}
    if num_threads is None:
        num_threads = len(worker_func)
    if not isinstance(worker_func, Iterable):
        worker_func = [worker_func] * num_threads

    barrier = threading.Barrier(num_threads)

    results = [None] * num_threads
    exc_value = None

    def wrapper_func(idx, func, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Wait on the barrier, run the function, store its result at the
        # thread's index; on any exception record it in the shared variable instead of raising.
        # Wait for all threads to reach this point before proceeding.
        try:
            barrier.wait()
            res = func(*args, **kwargs)
            results[idx] = res
        except Exception as e:
            nonlocal exc_value
            exc_value = e

    workers = [
        threading.Thread(target=wrapper_func, args=(i, func, *args),
                         kwargs=kwargs, daemon=True)
        for i, func in enumerate(worker_func)
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    # If a worker thread raises an exception, re-raise it.
    if exc_value is not None:
        raise exc_value

    return results
