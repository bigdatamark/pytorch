# mypy: ignore-errors

# [MODULE DOC] Shared infrastructure for PyTorch distributed tests. Provides test-skip exit codes
# and decorators (GPU count, world size, backend availability, ROCm, Windows), multi-process test
# base classes (MultiProcessTestCase, DistributedTestBase, MultiProcContinuousTest,
# C10dTorchCommsTestBase), a multi-threaded test runner using the threaded process group, subtest
# and TCP-store helpers, sparse reduce test generators and dynamo distributed harnesses.
# Import-time side effects: creates a module logger at INFO level, probes torchcomms for built
# backends and sets the TORCHCOMM_HAS_* globals when torchcomms is available, and computes default
# timeouts and world size (touching the CUDA device count on ROCm).

import faulthandler
import functools
import inspect
import itertools
import logging
import multiprocessing
import operator
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
import unittest
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from functools import partial, reduce, wraps
from io import StringIO
from typing import Any, NamedTuple
from unittest.mock import patch

import torch
import torch._dynamo.test_case
import torch.cuda.nccl
import torch.distributed as c10d
import torch.nn as nn
from torch._C._autograd import DeviceType
from torch._C._distributed_c10d import _SymmetricMemory
from torch._logging._internal import trace_log
from torch.distributed.distributed_c10d import _TORCHCOMM_AVAILABLE
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import (
    FILE_SCHEMA,
    find_free_port,
    IS_SANDCASTLE,
    LazyVal,
    retry_on_connect_failures,
    skip_but_pass_in_sandcastle,
    skip_but_pass_in_sandcastle_if,
    TEST_CUDA,
    TEST_HPU,
    TEST_WITH_ROCM,
    TEST_WITH_TSAN,
    TEST_XPU,
    TestCase,
)
from torch.testing._internal.distributed.multi_threaded_pg import (
    _install_threaded_pg,
    _uninstall_threaded_pg,
    ProcessLocalGroup,
)


TORCHCOMM_HAS_GLOO = False
TORCHCOMM_HAS_XCCL = False
TORCHCOMM_HAS_NCCL = False
TORCHCOMM_HAS_RCCL = False
TORCHCOMM_HAS_NCCLX = False
TORCHCOMM_HAS_RCCLX = False
if _TORCHCOMM_AVAILABLE:
    import torchcomms

    for _backend, _flag in [
        ("gloo", "TORCHCOMM_HAS_GLOO"),
        ("xccl", "TORCHCOMM_HAS_XCCL"),
        ("nccl", "TORCHCOMM_HAS_NCCL"),
        ("rcclx", "TORCHCOMM_HAS_RCCLX"),
        ("ncclx", "TORCHCOMM_HAS_NCCLX"),
    ]:
        if torchcomms.is_backend_built(_backend):
            globals()[_flag] = True


def setup_torchcomms_pg(
    backend: str,
    rank: int,
    world_size: int,
    device: torch.device,
    store_path: str,
    group_name: str,
) -> c10d.ProcessGroup:
    # [FUNCTION DOC] Builds a ProcessGroup whose NCCL backend is a torchcomms communicator
    # of the given backend name. Creates a FileStore at store_path, sets the TORCHCOMM_RANK
    # and TORCHCOMM_SIZE environment variables, creates the torchcomms comm with prefixed
    # stores, registers it on a new ProcessGroup, sets it as default backend and registers
    # the group by name in the c10d world. Returns the group. Raises on duplicate group
    # names or store problems.
    """Build a bare ProcessGroup whose backend is a torchcomms _BackendWrapper.

    Creates a torchcomms comm of the requested ``backend`` (e.g. ``nccl``,
    ``ncclx``) and wraps it in ``_BackendWrapper``, then registers it as the
    NCCL backend on a fresh ``ProcessGroup`` and publishes the group into
    ``c10d._world`` so name-based lookups (e.g. ``symm_mem.rendezvous(...,
    group=group_name)``) resolve to it.

    Callers are responsible for choosing a ``group_name`` (and matching
    ``store_path``) that is unique within the process — duplicates trip
    ``_register_pg_in_world`` and the underlying FileStore.
    """
    from torch.distributed.distributed_c10d import (
        _BackendWrapper,
        _register_pg_in_world,
    )

    file_store = c10d.FileStore(store_path, world_size)

    os.environ["TORCHCOMM_RANK"] = str(rank)
    os.environ["TORCHCOMM_SIZE"] = str(world_size)

    tc_comm = torchcomms.new_comm(
        backend,
        device,
        name=f"{group_name}_comm",
        store=c10d.PrefixStore(f"{group_name}_tc/", file_store),
        hints={"persistent_store": "true"},
    )

    pg = c10d.ProcessGroup(
        c10d.PrefixStore(f"{group_name}_pg/", file_store),
        tc_comm.get_rank(),
        tc_comm.get_size(),
    )
    pg._register_backend(
        tc_comm.get_device(),
        c10d.ProcessGroup.BackendType.NCCL,
        _BackendWrapper(tc_comm),
    )
    # _register_backend only updates the per-device map; backendType_ stays
    # UNDEFINED. getDefaultBackend()-dependent setters (including
    # use_pg_for_symm_mem_rendezvous) fail without this.
    pg._set_default_backend(c10d.ProcessGroup.BackendType.NCCL)

    pg._set_group_name(group_name)
    _register_pg_in_world(
        pg,
        backend_name=backend,
        store=file_store,
        group_name=group_name,
        backend_config=backend,
        rank_mapping={r: r for r in range(world_size)},
    )

    return pg


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ACCELERATOR_DIST_BACKENDS = ["nccl", "xccl", "hccl"]
DDP_RANK_DEVICES = ["cuda", "xpu"]
HAS_ACCELERATOR = TEST_CUDA or TEST_HPU or TEST_XPU


class TestSkip(NamedTuple):
    # [CLASS DOC] Named tuple pairing a process exit code with the human-readable skip message.
    # Instances are the values of TEST_SKIPS and let subprocess exit codes be mapped back to
    # skip reasons.
    exit_code: int
    message: str


TEST_SKIPS = {
    "backend_unavailable": TestSkip(
        72, "Skipped because distributed backend is not available."
    ),
    "small_worldsize": TestSkip(73, "Skipped due to small world size."),
    "odd_worldsize": TestSkip(87, "Skipped due to odd world size."),
    "no_cuda": TestSkip(74, "CUDA is not available."),
    "multi-gpu-1": TestSkip(75, "Need at least 1 CUDA device"),
    "multi-gpu-2": TestSkip(77, "Need at least 2 CUDA devices"),
    "multi-gpu-3": TestSkip(80, "Need at least 3 CUDA devices"),
    "multi-gpu-4": TestSkip(81, "Need at least 4 CUDA devices"),
    "multi-gpu-5": TestSkip(82, "Need at least 5 CUDA devices"),
    "multi-gpu-6": TestSkip(83, "Need at least 6 CUDA devices"),
    "multi-gpu-7": TestSkip(84, "Need at least 7 CUDA devices"),
    "multi-gpu-8": TestSkip(85, "Need at least 8 CUDA devices"),
    "nccl": TestSkip(76, "c10d not compiled with NCCL support"),
    "skipIfRocm": TestSkip(78, "Test skipped for ROCm"),
    "no_peer_access": TestSkip(79, "Test skipped because no GPU peer access"),
    "generic": TestSkip(
        86, "Test skipped at subprocess level, look at subprocess log for skip reason"
    ),
    "importerror": TestSkip(88, "Test skipped due to missing import"),
    "no_accelerator": TestSkip(89, "accelerator is not available."),
}


@dataclass
class DistTestCases:
    # [CLASS DOC] Holder of class-level tables describing backend capabilities for distributed
    # tests: skip_collective (collective name to set of backends lacking it) and backend_feature
    # (feature name to set of backends supporting it, with hpu/xpu entries added when those
    # devices exist). No instances state; used by reading the class attributes.
    # Backends that do not support a specific collective
    skip_collective = {}
    skip_collective["allgather_coalesced"] = {"nccl", "mpi", "ucc", "xccl"}
    skip_collective["reduce"] = set()
    skip_collective["sendrecv anysource"] = {"nccl", "ucc", "xccl"}
    skip_collective["cpu barrier"] = {"nccl", "ucc", "xccl"}

    # Sets showing that something is implemented
    backend_feature = {}
    backend_feature["gpu"] = {"nccl", "gloo", "ucc", "xccl"}
    backend_feature["cuda"] = {"nccl", "gloo", "ucc"}
    backend_feature["ddp"] = {"nccl", "gloo", "ucc", "xccl"}
    backend_feature["subgroup"] = {"nccl", "gloo", "ucc", "xccl"}
    backend_feature["plugin"] = set()
    if TEST_HPU:
        backend_feature["hpu"] = {"hccl"}
    if TEST_XPU:
        backend_feature["xpu"] = {"xccl"}


def requires_ddp_rank(device):
    # [FUNCTION DOC] Returns True when the device string is one of the device types (cuda, xpu)
    # that require a per-rank device index in DDP tests. Pure.
    return device in DDP_RANK_DEVICES


def skip_if_no_gpu(func):
    # [FUNCTION DOC] Decorator for multi-process tests: exits the process with the no_cuda skip
    # code when no CUDA, HPU or XPU device exists, and with the multi-gpu skip code for the world
    # size (from the WORLD_SIZE environment variable) when any available accelerator has fewer
    # devices than ranks. Otherwise calls the test.
    """Skips if the world size exceeds the number of GPUs, ensuring that if the
    test is run, each rank has its own GPU via ``torch.cuda.device(rank)``."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapped test body implementing the skip checks above; exits via
        # sys.exit with a skip exit code or returns the result of the test.
        # KNOWN ISSUE: the multi-gpu skip key only exists for world sizes 1 through 8, so larger
        # world sizes raise a KeyError.
        if not (TEST_CUDA or TEST_HPU or TEST_XPU):
            sys.exit(TEST_SKIPS["no_cuda"].exit_code)
        world_size = int(os.environ["WORLD_SIZE"])
        if TEST_CUDA and torch.cuda.device_count() < world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{world_size}"].exit_code)
        if TEST_HPU and torch.hpu.device_count() < world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{world_size}"].exit_code)
        if TEST_XPU and torch.xpu.device_count() < world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{world_size}"].exit_code)

        return func(*args, **kwargs)

    return wrapper


# TODO (kwen2501): what is the purpose of this decorator?  Tests with this
# decorator were always skipped. So they may be outdated already.
# Oct 2024: bumping the small-world criteria to < 8, as we are increasing the
# number of GPUs in CI from 2 to 4, and we need to continue skipping those tests
# to keep CI green. But this is just a temporary solution. We should clean up
# those tests somehow.
def skip_if_small_worldsize(func):
    # [FUNCTION DOC] Decorator that exits the process with the small_worldsize skip code when the
    # BACKEND environment variable is not mpi and WORLD_SIZE is below 8; otherwise runs the test.
    # The comment above notes it effectively skips most tests and may be outdated.
    @wraps(func)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapped test body: reads BACKEND and WORLD_SIZE from the
        # environment, exits with a skip code or runs the test. Raises KeyError if the variables
        # are unset.
        if (os.environ["BACKEND"] != "mpi") and int(os.environ["WORLD_SIZE"]) < 8:
            sys.exit(TEST_SKIPS["small_worldsize"].exit_code)

        return func(*args, **kwargs)

    return wrapper


def skip_if_odd_worldsize(func):
    # [FUNCTION DOC] Decorator that exits the process with the odd_worldsize skip code when the
    # BACKEND environment variable is not mpi and WORLD_SIZE is odd; otherwise runs the test.
    @wraps(func)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapped test body: reads BACKEND and WORLD_SIZE from the
        # environment, exits with a skip code or runs the test. Raises KeyError if the variables
        # are unset.
        if (os.environ["BACKEND"] != "mpi") and int(os.environ["WORLD_SIZE"]) % 2 == 1:
            sys.exit(TEST_SKIPS["odd_worldsize"].exit_code)

        return func(*args, **kwargs)

    return wrapper


def require_n_gpus_for_nccl_backend(n, backend):
    # [FUNCTION DOC] Decorator factory: for the nccl backend, skips (exits with the multi-gpu-n
    # skip code) when fewer than n CUDA devices exist; for other backends never skips. Args: n
    # minimum GPU count, backend name.
    def decorator(func):
        # [FUNCTION DOC] HELPER: Applies the GPU-count skip logic to a test function and returns
        # the wrapper.
        @wraps(func)
        def wrapper(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapped test body: exits with the skip code or returns the
            # test result.
            if backend == "nccl" and torch.cuda.device_count() < n:
                sys.exit(TEST_SKIPS[f"multi-gpu-{n}"].exit_code)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def import_transformers_or_skip():
    # [FUNCTION DOC] Decorator factory that skips a test (exit with the importerror skip code)
    # when the transformers package cannot be imported. The test runs inside the try block, so an
    # ImportError raised by the test itself also causes a skip.
    def decorator(func):
        # [FUNCTION DOC] HELPER: Wraps a test function with the import-or-skip behavior.
        @wraps(func)
        def wrapper(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapped test body: tries to import transformers classes, then
            # runs the test; exits with a skip code on ImportError.
            try:
                from transformers import AutoModelForMaskedLM, BertConfig  # noqa: F401

                return func(*args, **kwargs)
            except ImportError:
                sys.exit(TEST_SKIPS["importerror"].exit_code)

        return wrapper

    return decorator


def at_least_x_gpu(x):
    # [FUNCTION DOC] Returns True if any of CUDA, HPU or XPU is present with at least x devices,
    # else False. Arg x is the required device count.
    if TEST_CUDA and torch.cuda.device_count() >= x:
        return True
    if TEST_HPU and torch.hpu.device_count() >= x:
        return True
    if TEST_XPU and torch.xpu.device_count() >= x:
        return True
    return False


def _maybe_handle_skip_if_lt_x_gpu(args, msg) -> bool:
    # [FUNCTION DOC] HELPER: Calls the test instance's _handle_test_skip hook with the message if
    # args has a first element providing one, and returns True; returns False when there is no
    # such hook.
    # KNOWN ISSUE: the first element of args is read before the length check, so an empty args
    # raises an IndexError instead of returning False.
    _handle_test_skip = getattr(args[0], "_handle_test_skip", None)
    if len(args) == 0 or _handle_test_skip is None:
        return False
    _handle_test_skip(msg)
    return True


def skip_if_lt_x_gpu(x, *, allow_cpu=False):
    # [FUNCTION DOC] Decorator factory that skips a test when fewer than x accelerators (CUDA, HPU
    # or XPU) are available, unless allow_cpu is set and no accelerator exists. A skip is reported
    # through the test's _handle_test_skip hook when present, else by exiting with the multi-gpu-x
    # skip code. Otherwise the test runs and its result is returned (a skipped test returns None).
    """Skip if fewer than x accelerators available.

    Args:
        x: Minimum number of accelerators required.
        allow_cpu: If True, run the test on CPU-only machines (no accelerators).
    """

    def decorator(func):
        # [FUNCTION DOC] HELPER: Applies the accelerator-count skip logic to a test function and
        # returns the wrapper.
        @wraps(func)
        def wrapper(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapped test body: runs the test if enough devices exist,
            # else skips.
            if torch.cuda.is_available() and torch.cuda.device_count() >= x:
                return func(*args, **kwargs)
            if TEST_HPU and torch.hpu.device_count() >= x:
                return func(*args, **kwargs)
            if TEST_XPU and torch.xpu.device_count() >= x:
                return func(*args, **kwargs)
            if allow_cpu and not (torch.cuda.is_available() or TEST_HPU or TEST_XPU):
                return func(*args, **kwargs)
            test_skip = TEST_SKIPS[f"multi-gpu-{x}"]
            if not _maybe_handle_skip_if_lt_x_gpu(args, test_skip.message):
                sys.exit(test_skip.exit_code)

        return wrapper

    return decorator


def requires_world_size(n: int):
    # [FUNCTION DOC] Decorator factory that records the requested world size on the test function
    # (an attribute read by get_required_world_size) and skips the test via unittest.skipUnless
    # when fewer than n CUDA devices are available. Arg n is the number of ranks.
    # KNOWN ISSUE: the docstring usage example names a decorator spelled differently from
    # this function.
    """
    Decorator to request a specific world size for a test. The test harness can
    read this attribute to set the number of ranks to spawn. If there are fewer
    than `n` CUDA devices available, the test should be skipped by the harness.

    Usage:
        @require_world_size(3)
        def test_something(self):
            ...
    """

    def decorator(func):
        # [FUNCTION DOC] HELPER: Annotates the test function with the required world size, counts
        # CUDA devices at decoration time and returns the function wrapped with a skip condition.
        func._required_world_size = n
        available = torch.cuda.device_count()
        return unittest.skipUnless(
            available >= n, f"requires {n} GPUs, found {available}"
        )(func)

    return decorator


def get_required_world_size(obj: Any, default: int) -> int:
    # [FUNCTION DOC] Returns the world size requested for the currently running test method of obj
    # through requires_world_size, using the current test name helper when present or the unittest
    # method name. Returns default if the attribute is missing or anything fails (all exceptions
    # are swallowed).
    """
    Returns the requested world size for the currently running unittest method on `obj`
    if annotated via `@require_world_size(n)`, else returns `default`.
    """
    try:
        # Try MultiProcessTestCase helper first, then unittest fallback
        test_name = (
            obj._current_test_name()  # type: ignore[attr-defined]
            if hasattr(obj, "_current_test_name") and callable(obj._current_test_name)
            else obj._testMethodName
        )
        fn = getattr(obj, test_name)
        value = fn._required_world_size
        return int(value)
    except Exception:
        return default


# This decorator helps avoiding initializing cuda while testing other backends
def nccl_skip_if_lt_x_gpu(backend, x):
    # [FUNCTION DOC] Decorator factory that only for the nccl backend skips (exit with the
    # multi-gpu-x skip code or the test instance hook) when fewer than x CUDA devices exist,
    # avoiding CUDA initialization for other backends. Other backends always run the test.
    def decorator(func):
        # [FUNCTION DOC] HELPER: Applies the nccl-only GPU-count check to a test function and
        # returns the wrapper.
        @wraps(func)
        def wrapper(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapped test body: runs the test or skips as described above.
            if backend != "nccl":
                return func(*args, **kwargs)
            if torch.cuda.is_available() and torch.cuda.device_count() >= x:
                return func(*args, **kwargs)
            test_skip = TEST_SKIPS[f"multi-gpu-{x}"]
            if not _maybe_handle_skip_if_lt_x_gpu(args, test_skip.message):
                sys.exit(test_skip.exit_code)

        return wrapper

    return decorator


def verify_ddp_error_logged(model_DDP, err_substr):
    # [FUNCTION DOC] Asserts that a DDP model logged the expected error: reads the DDP logging
    # data and requires the iteration, has_error and error keys, then requires the error substring
    # (with any C++ stack trace suffix removed) to appear in the logged error. Raises
    # AssertionError on failure.
    # Verify error was logged in ddp_logging_data.
    ddp_logging_data = model_DDP._get_ddp_logging_data()
    if "iteration" not in ddp_logging_data:
        raise AssertionError("Expected 'iteration' in ddp_logging_data")
    if "has_error" not in ddp_logging_data:
        raise AssertionError("Expected 'has_error' in ddp_logging_data")
    if "error" not in ddp_logging_data:
        raise AssertionError("Expected 'error' in ddp_logging_data")
    logging_err = ddp_logging_data["error"]
    # Remove C++ stacktrace if needed.
    actual = (
        err_substr
        if err_substr.find("\nException raised from ") == -1
        else err_substr.split("\nException raised from ")[0]
    )
    if actual not in logging_err:
        raise AssertionError(
            f"Did not find expected {actual} in ddp logging data error: {logging_err}"
        )


def with_nccl_blocking_wait(func):
    # [FUNCTION DOC] Decorator that runs the test with TORCH_NCCL_BLOCKING_WAIT set to 1 and
    # TORCH_NCCL_ASYNC_ERROR_HANDLING unset, restoring both environment variables to their
    # previous values afterwards (even on failure). Mutates the process environment during
    # the test.
    """
    Convenience decorator to set/unset TORCH_NCCL_BLOCKING_WAIT flag. Note that use of
    this decorator will override the setting of TORCH_NCCL_ASYNC_ERROR_HANDLING for
    the particular test. After the test, both TORCH_NCCL_BLOCKING_WAIT and
    TORCH_NCCL_ASYNC_ERROR_HANDLING will be restored to their original values.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapped test body: saves and modifies the two environment
        # variables, runs the test and restores them in a finally block; returns the test result.
        # Save and unset TORCH_NCCL_ASYNC_ERROR_HANDLING
        try:
            cached_nccl_async_error_handling: str | None = os.environ[
                "TORCH_NCCL_ASYNC_ERROR_HANDLING"
            ]
            del os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"]
        except KeyError:
            # TORCH_NCCL_ASYNC_ERROR_HANDLING was unset
            cached_nccl_async_error_handling = None

        # Save val of TORCH_NCCL_BLOCKING_WAIT and set it.
        try:
            cached_nccl_blocking_wait: str | None = os.environ[
                "TORCH_NCCL_BLOCKING_WAIT"
            ]
        except KeyError:
            cached_nccl_blocking_wait = None
        finally:
            os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"

        try:
            ret = func(*args, **kwargs)
            return ret
        finally:
            # restore old values.
            if cached_nccl_async_error_handling is not None:
                os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = (
                    cached_nccl_async_error_handling
                )

            if cached_nccl_blocking_wait is not None:
                os.environ["TORCH_NCCL_BLOCKING_WAIT"] = cached_nccl_blocking_wait

    return wrapper


def with_dist_debug_levels(levels):
    # [FUNCTION DOC] Decorator factory that runs the test once for each TORCH_DISTRIBUTED_DEBUG
    # level in levels. For each level it sets the environment variable, reloads the c10d debug
    # level, runs the test and calls a barrier. Returns the result of the last run.
    # KNOWN ISSUE: the original level is restored only if it was set and the restore happens
    # inside the loop; an empty levels list raises UnboundLocalError.
    """
    Runs a test for each distributed debug level specified in levels.
    """

    def decorator(func):
        # [FUNCTION DOC] HELPER: Wraps the test function with the per-level loop.
        @wraps(func)
        def wrapper(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapped test body implementing the loop over debug levels;
            # mutates the environment and calls a collective barrier.
            old_level = os.environ.get("TORCH_DISTRIBUTED_DEBUG", None)
            for level in levels:
                os.environ["TORCH_DISTRIBUTED_DEBUG"] = level
                c10d.set_debug_level_from_env()
                ret = func(*args, **kwargs)
                c10d.barrier()
                if old_level is not None:
                    os.environ["TORCH_DISTRIBUTED_DEBUG"] = old_level
            # Only returns test return for last test, but since these are
            # unittests the return value is not really used and earlier tests
            # would've raised had they failed.
            return ret

        return wrapper

    return decorator


def requires_gloo():
    # [FUNCTION DOC] Returns a skip decorator (via the sandcastle-aware skip helper) that skips
    # the test when c10d was built without the Gloo backend. Evaluated at decoration time; no
    # other side effects.
    return skip_but_pass_in_sandcastle_if(
        not c10d.is_gloo_available(),
        "c10d was not compiled with the Gloo backend",
    )


def requires_nccl_version(version, msg):
    # [FUNCTION DOC] Returns a skip decorator for tests needing a minimum NCCL version. When CUDA
    # is not available, returns an identity decorator. If NCCL is not compiled in, returns a skip
    # decorator; otherwise skips when the installed NCCL version tuple is below version, with msg
    # included in the reason. Args: version tuple, msg reason text.
    if not TEST_CUDA:
        return lambda f: f
    if not c10d.is_nccl_available():
        return skip_but_pass_in_sandcastle(
            "c10d was not compiled with the NCCL backend",
        )
    else:
        return skip_but_pass_in_sandcastle_if(
            torch.cuda.nccl.version() < version,
            f"Requires NCCL version greater than or equal to: {version}, found: {torch.cuda.nccl.version()}, reason: {msg}",
        )


def requires_nccl_shrink():
    # [FUNCTION DOC] Returns a skip decorator requiring NCCL 2.27 or newer, as needed for group
    # shrink support. Delegates to requires_nccl_version.
    """
    Require NCCL shrink support (NCCL available and version >= 2.27).
    """
    return requires_nccl_version((2, 27), "Need NCCL 2.27+ for shrink_group")


def requires_nccl():
    # [FUNCTION DOC] Returns a skip decorator (via the sandcastle-aware skip helper) that skips
    # the test when c10d was built without the NCCL backend. Evaluated at decoration time; no
    # other side effects.
    return skip_but_pass_in_sandcastle_if(
        not c10d.is_nccl_available(),
        "c10d was not compiled with the NCCL backend",
    )


def requires_xccl():
    # [FUNCTION DOC] Returns a skip decorator (via the sandcastle-aware skip helper) that skips
    # the test when c10d was built without the XCCL backend. Evaluated at decoration time; no
    # other side effects.
    return skip_but_pass_in_sandcastle_if(
        not c10d.is_xccl_available(),
        "c10d was not compiled with the XCCL backend",
    )


def requires_ucc():
    # [FUNCTION DOC] Returns a skip decorator (via the sandcastle-aware skip helper) that skips
    # the test when c10d was built without the UCC backend. Evaluated at decoration time; no other
    # side effects.
    return skip_but_pass_in_sandcastle_if(
        not c10d.is_ucc_available(),
        "c10d was not compiled with the UCC backend",
    )


def requires_mpi():
    # [FUNCTION DOC] Returns a skip decorator (via the sandcastle-aware skip helper) that skips
    # the test when c10d was built without the MPI backend. Evaluated at decoration time; no other
    # side effects.
    return skip_but_pass_in_sandcastle_if(
        not c10d.is_mpi_available(),
        "c10d was not compiled with the MPI backend",
    )


def requires_accelerator_dist_backend(backends=None):
    # [FUNCTION DOC] Returns a skip decorator that skips unless at least one of the given
    # accelerator backends (default all of nccl, xccl, hccl) is available; hccl counts as
    # available when HPU testing is enabled. Unknown backend names count as unavailable.
    """
    Decorator to skip tests if no accelerator communication backend (NCCL, XCCL, HCCL) is available.

    Args:
        backends (Optional[List[str]]): Specific accelerator backends to check (e.g., ["nccl", "xccl", "hccl"]).
                                       If None, checks all supported accelerator backends (NCCL, XCCL, HCCL).

    Returns:
        callable: A decorator that skips the test if no specified accelerator backend is available.
    """
    if backends is None:
        backends = ACCELERATOR_DIST_BACKENDS

    backend_available = any(
        {
            "nccl": c10d.is_nccl_available,
            "xccl": c10d.is_xccl_available,
            "hccl": lambda: TEST_HPU,
        }.get(backend, lambda: False)()
        for backend in backends
    )

    return skip_but_pass_in_sandcastle_if(
        not backend_available,
        f"No accelerator communication backend available among {backends}",
    )


def requires_multicast_support():
    # [FUNCTION DOC] Returns a skip decorator that skips the test unless CUDA is available and
    # device 0 reports multicast support in symmetric memory.
    has_multicast_support = (
        torch.cuda.is_available()
        and _SymmetricMemory.has_multicast_support(DeviceType.CUDA, 0)
    )
    return skip_but_pass_in_sandcastle_if(
        not has_multicast_support,
        "multicast support is not available",
    )


def evaluate_platform_supports_symm_mem():
    # [FUNCTION DOC] Returns True when the platform can run symmetric memory tests: CUDA must be
    # present, and on ROCm the first device's architecture name must contain gfx942 or gfx950.
    # Returns False otherwise.
    if TEST_CUDA:
        if TEST_WITH_ROCM:
            arch_list = ["gfx942", "gfx950"]
            for arch in arch_list:
                if arch in torch.cuda.get_device_properties(0).gcnArchName:
                    return True
            return False
        else:
            return True
    else:
        return False


PLATFORM_SUPPORTS_SYMM_MEM: bool = LazyVal(
    lambda: evaluate_platform_supports_symm_mem()
)


def skip_if_rocm_multiprocess(func):
    # [FUNCTION DOC] Decorator that skips the test with the ROCm skip message when running on
    # ROCm. Returns the wrapped function.
    """Skips a test for ROCm multiprocess UTs"""
    return unittest.skipIf(TEST_WITH_ROCM, TEST_SKIPS["skipIfRocm"].message)(func)


def skip_if_rocm_arch_multiprocess(arch: tuple[str, ...]):
    # [FUNCTION DOC] Decorator factory that skips the test when running on ROCm and the first
    # device's architecture (name before any colon) is in the given tuple arch. Reads device
    # properties at decoration time.
    """Skips a test for given ROCm archs - multiprocess UTs"""

    def decorator(func):
        # [FUNCTION DOC] HELPER: Computes the skip reason at decoration time and returns the
        # function wrapped with a skip-if condition.
        reason = None
        if TEST_WITH_ROCM:
            prop = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
            if prop in arch:
                reason = f"skip_if_rocm_arch_multiprocess: test skipped on {arch}"

        return unittest.skipIf(reason is not None, reason)(func)

    return decorator


def skip_if_rocm_ver_lessthan_multiprocess(version=None):
    # [FUNCTION DOC] Decorator factory that skips the test on ROCm when the ROCm version (parsed
    # from the HIP version string, ignoring a git suffix) is lower than version, or when version
    # is None. Never skips off ROCm.
    # KNOWN ISSUE: the check that the parsed version tuple is None can never be true, and a None
    # version always skips.
    """Skips a test for ROCm based on ROCm ver - multiprocess UTs"""

    def decorator(func):
        # [FUNCTION DOC] HELPER: Computes the skip reason at decoration time and returns the
        # function wrapped with a skip-if condition.
        reason = None
        if TEST_WITH_ROCM:
            rocm_version = str(torch.version.hip)
            rocm_version = rocm_version.split("-", maxsplit=1)[0]  # ignore git sha
            rocm_version_tuple = tuple(int(x) for x in rocm_version.split("."))
            if (
                rocm_version_tuple is None
                or version is None
                or rocm_version_tuple < tuple(version)
            ):
                reason = f"skip_if_rocm_ver_lessthan_multiprocess: ROCm {rocm_version_tuple} is available but {version} required"

        return unittest.skipIf(reason is not None, reason)(func)

    return decorator


def skip_if_rocm_ver_atleast_multiprocess(version=None):
    # [FUNCTION DOC] Decorator factory that skips the test on ROCm when the ROCm version is at
    # least the given version tuple (known failure on newer releases). Never skips off ROCm or
    # when version is None.
    """Skips a test for ROCm if the version is at least the given tuple - multiprocess UTs"""

    def decorator(func):
        # [FUNCTION DOC] HELPER: Computes the skip reason at decoration time and returns the
        # function wrapped with a skip-if condition.
        reason = None
        if TEST_WITH_ROCM:
            rocm_version = str(torch.version.hip)
            rocm_version = rocm_version.split("-", maxsplit=1)[0]  # ignore git sha
            rocm_version_tuple = tuple(int(x) for x in rocm_version.split("."))
            if version is not None and rocm_version_tuple >= tuple(version):
                reason = f"skip_if_rocm_ver_atleast_multiprocess: known failure on ROCm {rocm_version_tuple} (>= {version})"

        return unittest.skipIf(reason is not None, reason)(func)

    return decorator


def skip_if_win32():
    # [FUNCTION DOC] Returns a skip decorator that skips the test on Windows (win32).
    return skip_but_pass_in_sandcastle_if(
        sys.platform == "win32",
        "This unit test case is not supported on Windows platform",
    )


def sm_is_or_higher_than(device: torch.device, major: int, minor: int) -> bool:
    # [FUNCTION DOC] Returns whether the device's compute capability is at least (major, minor).
    # Non-CUDA devices return True and ROCm builds return False; otherwise compares the CUDA
    # capability tuple.
    # KNOWN ISSUE: the docstring says it errors out for non-CUDA devices, but the code
    # returns True.
    """
    Returns True if the device's compute capability is (major, minor) or higher.
    Error out if the device is not a CUDA device.
    Returns False if device is a RoCM device.
    Returns True if device is a non-CUDA device.
    """
    if device.type != "cuda":
        return True

    if torch.version.hip is not None:
        # ROCm devices may have different compute capability codes
        return False

    return torch.cuda.get_device_capability(device) >= (major, minor)


@retry_on_connect_failures
def create_tcp_store(
    addr="localhost",
    world_size=1,
    is_master=True,
    timeout=timedelta(minutes=5),
    wait_for_workers=True,
    jit_class=False,
    use_libuv=True,
):
    # [FUNCTION DOC] Creates a TCP store on a free port; wrapped with a retry-on-connect-failure
    # decorator so a port collision is retried. Args: address, world size, is_master, timeout,
    # wait_for_workers, jit_class (use the TorchScript class with the timeout converted to
    # milliseconds), use_libuv. Returns the store object.
    """
    Creates a TCP store. Retries if the chosen port is already in use.
    """
    port = find_free_port()
    if jit_class:
        timeout_millisecond = int(timeout / timedelta(milliseconds=1))
        return torch.classes.dist_c10d.TCPStore(
            addr, port, world_size, is_master, timeout_millisecond
        )
    else:
        return c10d.TCPStore(
            addr,
            port,
            world_size,
            is_master,
            wait_for_workers=wait_for_workers,
            use_libuv=use_libuv,
        )


if TEST_WITH_TSAN:
    # TSAN runs much slower.
    TIMEOUT_DEFAULT = 500
else:
    TIMEOUT_DEFAULT = int(os.getenv("DISTRIBUTED_TESTS_DEFAULT_TIMEOUT", "300"))
TIMEOUT_OVERRIDE = {
    "test_ddp_uneven_inputs": 400,
    "test_DistributedDataParallel": 500,
}


# https://github.com/pytorch/pytorch/issues/75665
if TEST_WITH_ROCM:
    TIMEOUT_OVERRIDE["test_join_kwargs"] = 200


def create_device(interface=None, lazy_init: bool = False):
    # [FUNCTION DOC] Creates a Gloo device. Uses the loopback hostname 127.0.0.1 on Windows or
    # when no interface is given, otherwise the named network interface; lazy_init is passed
    # through. Returns the device.
    if sys.platform == "win32" or interface is None:
        return c10d.ProcessGroupGloo.create_device(
            hostname="127.0.0.1", lazy_init=lazy_init
        )
    else:
        return c10d.ProcessGroupGloo.create_device(
            interface=interface, lazy_init=lazy_init
        )


def get_timeout(test_id) -> int:
    # [FUNCTION DOC] Returns the timeout in seconds for a test id: an override from
    # TIMEOUT_OVERRIDE keyed by the last dotted component, else TIMEOUT_DEFAULT.
    return TIMEOUT_OVERRIDE.get(test_id.split(".")[-1], TIMEOUT_DEFAULT)


@contextmanager
def captured_output():
    # [FUNCTION DOC] Context manager that temporarily replaces stdout and stderr with in-memory
    # buffers, yields the two buffers, and restores the original streams afterward even on error.
    # Mutates process-global stream state.
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def simple_sparse_reduce_tests(rank: int, world_size: int, num_inputs: int = 1):
    # [FUNCTION DOC] Generates basic sparse reduction test cases (sum only). For each of six
    # generator variants (sparse dimensions 1 to 3, dense dimensions 1 to 3) returns a pair: the
    # list of per-input sparse tensors for this rank and the list of expected summed tensors
    # across all ranks. Args: rank, world size, number of inputs.
    """
    Generate a number of basic test cases for sparse reduction.
    These cover tensors with a varying number of sparse dimensions and a varying
    number of dense dimensions. The only reduction operation we support is sum.
    """

    def generate(rank: int, world_size: int, sparse_dims: int = 1, dense_dims: int = 0):
        # [FUNCTION DOC] HELPER: Builds one sparse COO tensor for a rank: first sparse dimension
        # has indices 0 through rank, other sparse dimensions are zero so all ranks overlap,
        # values are ones.
        # First sparse dimension is [0..rank].
        # Subsequent dimensions are always 0, so we know there is
        # a non-empty intersection between any two sparse tensors.
        indices = torch.reshape(torch.arange(rank + 1), (1, rank + 1))
        shape = [world_size] + [2 for _ in range(dense_dims)]
        for _ in range(sparse_dims - 1):
            indices = torch.cat((indices, torch.zeros(1, rank + 1)))
            shape.append(world_size)
        values = torch.ones([rank + 1] + [2 for _ in range(dense_dims)])
        return torch.sparse_coo_tensor(indices, values, shape)

    def compute_sum(fn, world_size: int):
        # [FUNCTION DOC] HELPER: Sums the tensors produced by the generator function for every
        # rank of the world and returns the total.
        return reduce(
            operator.add, [fn(rank, world_size) for rank in range(world_size)]
        )

    return [
        (
            [
                fn(num_inputs * rank + i, num_inputs * world_size)
                for i in range(num_inputs)
            ],
            [compute_sum(fn, num_inputs * world_size) for i in range(num_inputs)],
        )
        for fn in [
            partial(generate, sparse_dims=1),
            partial(generate, sparse_dims=2),
            partial(generate, sparse_dims=3),
            partial(generate, dense_dims=1),
            partial(generate, dense_dims=2),
            partial(generate, dense_dims=3),
        ]
    ]


# HELPER FOR MULTIGPU TESTS
def init_multigpu_helper(world_size: int, backend: str):
    # [FUNCTION DOC] Maps each rank to a list of GPU indices for multi-GPU tests, using CUDA, HPU
    # or XPU device counts.
    # KNOWN ISSUE: when world_size exceeds the GPU count the per-process count becomes zero
    # (integer division), producing empty lists.
    """Multigpu tests are designed to simulate the multi nodes with multi
    GPUs on each node. Nccl backend requires equal #GPUs in each process.
    On a single node, all visible GPUs are evenly
    divided to subsets, each process only uses a subset.
    """
    nGPUs = torch.cuda.device_count()
    if TEST_HPU:
        nGPUs = torch.hpu.device_count()
    if TEST_XPU:
        nGPUs = torch.xpu.device_count()
    visible_devices = range(nGPUs)

    # If rank is less than or equal to number of available GPU's
    # then each rank can be mapped to corresponding GPU.
    nGPUs_per_process = 1
    if world_size > nGPUs:
        nGPUs_per_process = nGPUs // world_size
    rank_to_GPU = {
        i: list(visible_devices[i * nGPUs_per_process : (i + 1) * nGPUs_per_process])
        for i in range(world_size)
    }
    return rank_to_GPU


tmp_dir: tempfile.TemporaryDirectory | None = None


def initialize_temp_directories(init_method: str | None = None) -> None:
    # [FUNCTION DOC] Creates a temporary directory tree (barrier, test_dir, init_dir) stored in
    # the module global tmp_dir and exports TEMP_DIR and INIT_METHOD environment variables.
    # INIT_METHOD is init_method if given, else a file URL to a shared init file in the temp dir.
    # Mutates globals and the environment.
    global tmp_dir
    tmp_dir = tempfile.TemporaryDirectory()
    os.environ["TEMP_DIR"] = tmp_dir.name
    os.mkdir(os.path.join(tmp_dir.name, "barrier"))
    os.mkdir(os.path.join(tmp_dir.name, "test_dir"))
    init_dir_path = os.path.join(tmp_dir.name, "init_dir")
    os.mkdir(init_dir_path)
    # Set init method if specified.
    if init_method is not None:
        os.environ["INIT_METHOD"] = init_method
    else:
        os.environ["INIT_METHOD"] = FILE_SCHEMA + os.path.join(
            init_dir_path, "shared_init_file"
        )


def cleanup_temp_dir() -> None:
    # [FUNCTION DOC] Cleans up the module global temporary directory if one was created;
    # no-op otherwise.
    if tmp_dir is not None:
        tmp_dir.cleanup()


def retrieve_result_from_completion_queue(
    process: torch.multiprocessing.Process,
    completion_queue: torch.multiprocessing.Queue,
    timeout: int | None = None,
) -> Any:
    # [FUNCTION DOC] Waits for a result on the completion queue of a worker process. Polls with a
    # bounded queue timeout, returning a RuntimeError instance (not raised) if the process died
    # without a result or the overall timeout elapsed; returns the queued result otherwise. Args:
    # process, queue, optional total timeout in seconds.
    """Get result from the completion_queue associated with process.

    When the process finished without putting a result or the timeout expired an exception instance will be returned"""
    queue_timeout = 120 if timeout is None else max(10, min(120, timeout // 4))
    start_time = time.time()
    # Periodically check the process for liveness
    while True:
        try:
            return completion_queue.get(timeout=queue_timeout)
        except queue.Empty:
            # If the process is no longer alive we cannot get a result from the queue unless it is there right now.
            # This can happen if the timeout occurred just before the process put its result and terminated.
            # So do a last check for emptiness before considering it as a failure.
            if not process.is_alive() and completion_queue.empty():
                return RuntimeError(f"Exited with {process.exitcode}")
        if timeout is not None:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                return RuntimeError(f"Process timed out after {elapsed}s")


# Most tests operate with this worldsize
if TEST_WITH_ROCM:
    DEFAULT_WORLD_SIZE = min(4, max(2, torch.cuda.device_count()))
else:
    DEFAULT_WORLD_SIZE = 4

# [How does MultiProcessTestCase work?]
# Each MultiProcessTestCase instance uses 1 + `world_size()` processes, by
# default `world_size()` returns 4. Let's take `test_rpc_spawn.py` as an
# example which inherits from this class. Its `Setup()` methods calls into
# `MultiProcessTestCase._spawn_processes()` which spawns `world_size()`
# subprocesses. During the spawn, the main process passes the test name to
# subprocesses, and the name is acquired from self.id(). The subprocesses
# then use the provided test function name to retrieve the function attribute
# from the test instance and run it. The main process simply waits for all
# subprocesses to join.


class MultiProcessTestCase(TestCase):
    # [CLASS DOC] Base class for tests that run in subprocesses. The main process (rank equal to
    # MAIN_PROCESS_RANK) spawns world_size processes, each of which re-creates the test case and
    # runs the same test method with its rank, then the main process joins them and checks exit
    # codes. State: rank, file_name (rendezvous file), processes, pid_to_pipe,
    # special_return_code_checks and skip_return_code_checks. Key methods: _spawn_processes, _run,
    # run_test, _join_processes, _check_return_codes. TEST_ERROR_EXIT_CODE marks a process that
    # failed with an exception.
    MAIN_PROCESS_RANK = -1
    # This exit code is used to indicate that the test code had an error and
    # exited abnormally. There are certain tests that might use sys.exit() to
    # simulate failures and in those cases, we can't have an exit code of 0,
    # but we still want to ensure we didn't run into any other errors.
    TEST_ERROR_EXIT_CODE = 10

    # do not early terminate for distributed tests.
    def _should_stop_test_suite(self) -> bool:
        # [FUNCTION DOC] HELPER: Returns False so distributed tests never stop the suite early.
        return False

    # Many test cases init a process group but do not destroy it.  This property
    # determines whether this base test class should call
    # `destroy_process_group` on behalf of the test. Its value is customizable
    # by derived TestCase's but it is a pan-TestCase value (cannot be customized
    # for each test).
    @property
    def destroy_pg_upon_exit(self) -> bool:
        # [FUNCTION DOC] Property: whether the base class destroys the default process group after
        # the test body in each subprocess. Defaults to True; subclasses may override.
        return True

    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Property: number of ranks to spawn, defaulting to DEFAULT_WORLD_SIZE.
        # Subclasses may override.
        return DEFAULT_WORLD_SIZE

    def join_or_run(self, fn):
        # [FUNCTION DOC] Returns a bound replacement for a test method: in the main process it
        # joins the spawned subprocesses, in a worker it runs the original method. Args: the
        # original test function.
        @wraps(fn)
        def wrapper(self):
            # [FUNCTION DOC] HELPER: Dispatches on whether this is the main process or a
            # worker rank.
            if self.rank == self.MAIN_PROCESS_RANK:
                self._join_processes(fn)
            else:
                fn()

        return types.MethodType(wrapper, self)

    # The main process spawns N subprocesses that run the test.
    # Constructor patches current instance test method to
    # assume the role of the main process and join its subprocesses,
    # or run the underlying test function.
    def __init__(
        self, method_name: str = "runTest", methodName: str = "runTest"
    ) -> None:
        # [FUNCTION DOC] Constructs the test case and replaces the named test method with a
        # version that joins subprocesses in the main process or runs the real body in workers.
        # Accepts both method_name and methodName for BC with testslide. Raises ValueError if an
        # explicitly named method does not exist.
        # methodName is the correct naming in unittest and testslide uses keyword arguments.
        # So we need to use both to 1) not break BC and, 2) support testslide.
        if methodName != "runTest":
            method_name = methodName
        super().__init__(method_name)
        try:
            fn = getattr(self, method_name)
            setattr(self, method_name, self.join_or_run(fn))
        except AttributeError as e:
            if methodName != "runTest":
                # we allow instantiation with no explicit method name
                # but not an *incorrect* or missing method name
                raise ValueError(
                    f"no such test method in {self.__class__}: {methodName}"
                ) from e

    def setUp(self) -> None:
        # [FUNCTION DOC] Per-test setup: initializes bookkeeping (return-code check overrides,
        # process list, rank set to the main process rank), creates a temporary rendezvous file
        # name and an empty pid-to-pipe map. Mutates instance state and creates a file.
        super().setUp()

        # Used for tests that are expected to return a non-0 exit code, such as
        # SIGABRT thrown by watchdog.
        self.special_return_code_checks: dict = {}

        # Used for tests that may return any exit code, which makes it hard to
        # check. This is rare, use with caution.
        self.skip_return_code_checks: list = []

        self.processes = []  # type: ignore[var-annotated]
        self.rank = self.MAIN_PROCESS_RANK
        with tempfile.NamedTemporaryFile(delete=False) as f:
            self.file_name = f.name
        # pid to pipe consisting of error message from process.
        self.pid_to_pipe = {}  # type: ignore[var-annotated]

    def tearDown(self) -> None:
        # [FUNCTION DOC] Per-test teardown: terminates any spawned processes and clears the
        # process list to avoid leaking file descriptors.
        super().tearDown()
        for p in self.processes:
            p.terminate()
        # Each Process instance holds a few open file descriptors. The unittest
        # runner creates a new TestCase instance for each test method and keeps
        # it alive until the end of the entire suite. We must thus reset the
        # processes to prevent an effective file descriptor leak.
        self.processes = []

    def _current_test_name(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the last dotted component of the test id, i.e. the test
        # method name.
        # self.id() == e.g. '__main__.TestDistributed.TestAdditive.test_get_rank'
        return self.id().split(".")[-1]

    def _start_processes(self, proc) -> None:
        # [FUNCTION DOC] HELPER: Starts world_size processes using the given process factory,
        # each running _run with its rank, the test name, the rendezvous file and a pipe end,
        # plus a fake_pg keyword. Records processes and parent pipes. Side effects: starts
        # processes and logs.
        self.processes = []
        for rank in range(int(self.world_size)):
            parent_conn, child_conn = torch.multiprocessing.Pipe()
            process = proc(
                target=self.__class__._run,
                name="process " + str(rank),
                args=(
                    rank,
                    self._current_test_name(),
                    self.file_name,
                    child_conn,
                ),
                kwargs={
                    "fake_pg": getattr(self, "fake_pg", False),
                },
            )
            process.start()
            logger.info("Started process %s with pid %s", rank, process.pid)
            self.pid_to_pipe[process.pid] = parent_conn
            self.processes.append(process)

    def _spawn_processes(self) -> None:
        # [FUNCTION DOC] HELPER: Sets the multiprocessing start method to spawn (ignoring if
        # already set) and starts the test processes with a spawn context.
        try:
            torch.multiprocessing.set_start_method("spawn")
        except RuntimeError:
            pass

        proc = torch.multiprocessing.get_context("spawn").Process
        self._start_processes(proc)

    class Event(Enum):
        # [CLASS DOC] HELPER: Enumeration of events the main process can send to workers;
        # currently only a request for a traceback dump.
        GET_TRACEBACK = 1

    @staticmethod
    def _event_listener(parent_pipe, signal_pipe, rank: int):
        # [FUNCTION DOC] HELPER: Body of the background thread in each worker: waits on the parent
        # pipe and a signal pipe; on a traceback request dumps all thread tracebacks to a
        # temporary file and sends the text back; exits when the parent pipe is closed or the
        # signal pipe fires. Args: parent pipe, signal pipe, rank.
        logger.debug("Starting event listener thread for rank %s", rank)
        while True:
            ready_pipes = multiprocessing.connection.wait([parent_pipe, signal_pipe])

            if parent_pipe in ready_pipes:
                if parent_pipe.closed:
                    logger.debug(
                        "Pipe closed for process %s, stopping event listener thread",
                        rank,
                    )
                    return

                event = parent_pipe.recv()
                logger.info("Received event %s on process %s", event, rank)

                if event == MultiProcessTestCase.Event.GET_TRACEBACK:
                    # Return traceback to the parent process.
                    with tempfile.NamedTemporaryFile(mode="r+") as tmp_file:
                        faulthandler.dump_traceback(tmp_file)
                        # Flush buffers and seek to read from the beginning
                        tmp_file.flush()
                        tmp_file.seek(0)
                        parent_pipe.send(tmp_file.read())

                        logger.info("Process %s sent traceback", rank)

            if signal_pipe in ready_pipes:
                return

    @classmethod
    def _run(
        cls, rank: int, test_name: str, file_name: str, parent_pipe, **kwargs
    ) -> None:
        # [FUNCTION DOC] HELPER: Entry point of each worker: builds a test instance for the
        # test name, sets rank and rendezvous file, then runs the test. Extra keyword arguments
        # are ignored.
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name
        self.run_test(test_name, parent_pipe)

    def run_test(self, test_name: str, parent_pipe) -> None:
        # [FUNCTION DOC] Runs the test body in a worker process. Starts the event listener thread,
        # enables C++ stack trace printing on fatal signals (non-Windows, non-macOS), sets
        # TORCH_SHOW_CPP_STACKTRACES and the RNG seed, then calls the test method. A SkipTest
        # exits with the generic skip code; other exceptions are logged, sent to the parent over
        # the pipe and exit with TEST_ERROR_EXIT_CODE. Always stops the listener and closes the
        # pipe; finally destroys the default process group if destroy_pg_upon_exit.
        # Start event listener thread.
        signal_recv_pipe, signal_send_pipe = torch.multiprocessing.Pipe(duplex=False)
        event_listener_thread = threading.Thread(
            target=MultiProcessTestCase._event_listener,
            args=(parent_pipe, signal_recv_pipe, self.rank),
            daemon=True,
        )
        event_listener_thread.start()
        if sys.platform != "win32" and sys.platform != "darwin":
            # Register signal handler to dump stack traces on FATALs.
            # Windows and MacOS do not support the signal handlers.
            torch._C._set_print_stack_traces_on_fatal_signal(True)
        # Show full C++ stacktraces when a Python error originating from C++ is raised.
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
        common_utils.set_rng_seed()

        # self.id() == e.g. '__main__.TestDistributed.test_get_rank'
        # We're retrieving a corresponding test and executing it.
        try:
            getattr(self, test_name)()
        except unittest.SkipTest as se:
            logger.info(
                "Process %s skipping test %s for following reason: %s",
                self.rank,
                test_name,
                se,
            )
            sys.exit(TEST_SKIPS["generic"].exit_code)
        except Exception:
            logger.error(
                "Caught exception: \n%s exiting process %s with exit code: %s",
                traceback.format_exc(),
                self.rank,
                MultiProcessTestCase.TEST_ERROR_EXIT_CODE,
            )
            # Send error to parent process.
            parent_pipe.send(traceback.format_exc())
            sys.exit(MultiProcessTestCase.TEST_ERROR_EXIT_CODE)
        finally:
            if signal_send_pipe is not None:
                signal_send_pipe.send(None)

            if event_listener_thread is None:
                raise AssertionError("Expected event_listener_thread to not be None")
            event_listener_thread.join()
            # Close pipe after done with test.
            parent_pipe.close()

        if self.destroy_pg_upon_exit:
            try:
                # Some tests do destroy the pgs, and destroy can't be called twice.
                # This avoids spewing warnings about improperly shutting down.
                c10d.destroy_process_group()
            except (AssertionError, ValueError):
                pass

    def _get_timedout_process_traceback(self) -> None:
        # [FUNCTION DOC] HELPER: Asks each still-running process for its traceback through its
        # pipe, waits up to five seconds for each reply, and logs the tracebacks or failure to
        # retrieve them. Handles connection errors.
        # KNOWN ISSUE: a local variable named traceback shadows the imported traceback module
        # inside this function.
        pipes = []
        for i, process in enumerate(self.processes):
            if process.exitcode is None:
                pipe = self.pid_to_pipe[process.pid]
                try:
                    pipe.send(MultiProcessTestCase.Event.GET_TRACEBACK)
                    pipes.append((i, pipe))
                except ConnectionError:
                    logger.exception(
                        "Encountered error while trying to get traceback for process %s",
                        i,
                    )

        # Wait for results.
        for rank, pipe in pipes:
            try:
                # Wait for traceback
                if pipe.poll(5):
                    if pipe.closed:
                        logger.info(
                            "Pipe closed for process %s, cannot retrieve traceback",
                            rank,
                        )
                        continue

                    traceback = pipe.recv()
                    logger.error(
                        "Process %s timed out with traceback: \n\n%s", rank, traceback
                    )
                else:
                    logger.error(
                        "Could not retrieve traceback for timed out process: %s", rank
                    )
            except ConnectionError:
                logger.exception(
                    "Encountered error while trying to get traceback for process %s",
                    rank,
                )

    def _join_processes(self, fn) -> None:
        # [FUNCTION DOC] HELPER: Main-process loop that polls the spawned processes until all
        # exit, one exits with the error code (then others are terminated) or the test timeout
        # passes (tracebacks are requested and processes terminated). Then checks return codes and
        # finally closes all pipes. Args: the test function, used for return-code overrides.
        timeout = get_timeout(self.id())
        start_time = time.time()
        subprocess_error = False
        try:
            while True:
                # check to see if any subprocess exited with an error early.
                for i, p in enumerate(self.processes):
                    # This is the exit code processes exit with if they
                    # encountered an exception.
                    if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE:
                        print(
                            f"Process {i} terminated with exit code {p.exitcode}, terminating remaining processes."
                        )
                        active_children = torch.multiprocessing.active_children()
                        for ac in active_children:
                            ac.terminate()
                        subprocess_error = True
                        break
                if subprocess_error:
                    break
                # All processes have joined cleanly if they all a valid exitcode
                if all(p.exitcode is not None for p in self.processes):
                    break
                # Check if we should time out the test. If so, we terminate each process.
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    self._get_timedout_process_traceback()
                    print(
                        f"Timing out after {timeout} seconds and killing subprocesses."
                    )
                    for p in self.processes:
                        p.terminate()
                    break
                # Sleep to avoid excessive busy polling.
                time.sleep(0.1)

            elapsed_time = time.time() - start_time
            self._check_return_codes(fn, elapsed_time)
        finally:
            # Close all pipes
            for pipe in self.pid_to_pipe.values():
                pipe.close()

    def _check_return_codes(self, fn, elapsed_time) -> None:
        # [FUNCTION DOC] Verifies exit codes of the spawned processes. Returns if none were
        # spawned. Raises RuntimeError with the child tracebacks if any exited with the error
        # code, or if any process did not finish. Skips for functions in skip_return_code_checks.
        # Maps a skip exit code of the first process to SkipTest (or just logs on sandcastle).
        # Otherwise asserts the first process exit code equals the expected code (0 or the value
        # in special_return_code_checks).
        """
        Checks that the return codes of all spawned processes match, and skips
        tests if they returned a return code indicating a skipping condition.
        """
        # If no processes are spawned, there is nothing to check.
        if not self.processes:
            logger.warning(
                "Note: no subprocesses were spawned, test was likely skipped."
            )
            return

        first_process = self.processes[0]
        # first, we check if there are errors in actual processes
        # (via TEST_ERROR_EXIT CODE), and raise an exception for those.
        # the reason we do this is to attempt to raise a more helpful error
        # message than "Process x terminated/timed out"
        # TODO: we should pipe the exception of the failed subprocess here.
        # Currently, the actual exception is displayed as a logging output.
        errored_processes = [
            (i, p)
            for i, p in enumerate(self.processes)
            if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE
        ]
        if errored_processes:
            error = ""
            for i, process in errored_processes:
                # Get error from pipe.
                error_message = self.pid_to_pipe[process.pid].recv()
                error += (
                    f"Process {i} exited with error code {MultiProcessTestCase.TEST_ERROR_EXIT_CODE} "
                    f"and exception:\n{error_message}\n"
                )

            raise RuntimeError(error)
        # If no process exited uncleanly, we check for timeouts, and then ensure
        # each process exited cleanly.
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError(
                    f"Process {i} terminated or timed out after {elapsed_time} seconds"
                )

        # Skip the test return code check
        if fn in self.skip_return_code_checks:
            return

        for skip in TEST_SKIPS.values():
            if first_process.exitcode == skip.exit_code:
                if IS_SANDCASTLE:
                    # Don't use unittest.skip to skip the test on sandcastle
                    # since it creates tasks for skipped tests assuming there
                    # is some follow-up needed. Instead just "pass" the test
                    # with an appropriate message.
                    logger.info(
                        "Skipping %s on sandcastle for the following reason: %s",
                        self.id(),
                        skip.message,
                    )
                    return
                else:
                    raise unittest.SkipTest(skip.message)

        # In most cases, we expect test to return exit code 0, standing for success.
        expected_return_code = 0
        # In some negative tests, we expect test to return non-zero exit code,
        # such as watchdog throwing SIGABRT.
        if fn in self.special_return_code_checks:
            expected_return_code = self.special_return_code_checks[fn]

        self.assertEqual(
            first_process.exitcode,
            expected_return_code,
            msg=f"Expected exit code {expected_return_code} but got {first_process.exitcode} for pid: {first_process.pid}",
        )

    @property
    def is_master(self) -> bool:
        # [FUNCTION DOC] Property: True if this instance is rank 0.
        return self.rank == 0


# Utility base class for distributed Multi Process Test cases
# This abstracts the PG creation and deletion, the backends are selected based
# on device type. The tests functions can be instantiated per device type using
# common_device_type.instantiate_device_type_tests
# other backends can add entry in backend() function
class DistributedTestBase(MultiProcessTestCase):
    # [CLASS DOC] Base class for device-generic distributed tests built on MultiProcessTestCase.
    # Spawns the processes in setUp, picks the backend from the device type (nccl for cuda, hccl
    # for hpu, xccl for xpu, gloo otherwise), and offers helpers to create the process group and
    # map ranks to devices.
    def setUp(self):
        # [FUNCTION DOC] Sets the WORLD_SIZE environment variable and spawns the test processes
        # after the base setup. Mutates the environment.
        super().setUp()
        os.environ["WORLD_SIZE"] = str(self.world_size)
        self._spawn_processes()

    def tearDown(self):
        # [FUNCTION DOC] Destroys the default process group (ignoring assertion errors) and
        # removes the rendezvous file (ignoring OS errors).
        # KNOWN ISSUE: the base class teardown is not called, so spawned processes are not
        # terminated here.
        try:
            torch.distributed.destroy_process_group()
        except AssertionError:
            pass
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def backend(self, device) -> str:
        # [FUNCTION DOC] Returns the backend name for the device string: nccl for cuda, hccl for
        # hpu, xccl for xpu, else gloo.
        if "cuda" in device:
            return "nccl"
        elif "hpu" in device:  # intel gaudi
            return "hccl"
        elif "xpu" in device:
            return "xccl"
        else:
            return "gloo"

    def create_pg(self, device, world_size=None):
        # [FUNCTION DOC] Initializes the default process group for the device using a FileStore at
        # the rendezvous file and the backend for the device. For nccl and xccl, sets the current
        # accelerator device and default device to the rank's device, raising RuntimeError if no
        # accelerator is found. Returns the default group. Args: device string, optional world
        # size (default self.world_size).
        # KNOWN ISSUE: the store is sized by the visible device count rather than the world size.
        if world_size is None:
            world_size = self.world_size
        num_visible_devices = torch.get_device_module(device).device_count()
        store = torch.distributed.FileStore(self.file_name, num_visible_devices)
        torch.distributed.init_process_group(
            backend=self.backend(device),
            world_size=world_size,
            rank=self.rank,
            store=store,
        )
        if "nccl" in self.backend(device) or "xccl" in self.backend(device):
            accelerator = torch.accelerator.current_accelerator()
            if accelerator:
                device_type = accelerator.type
                device = torch.device(f"{device_type}:{self.rank}")
                torch.set_default_device(device)
                torch.accelerator.set_device_index(device)
            else:
                raise RuntimeError(
                    f"Expected to find an accelerator when initializing process group"
                    f" with {self.backend(device)} backend, but got None"
                )
        return torch.distributed.distributed_c10d._get_default_group()

    def rank_to_device(self, device):
        # [FUNCTION DOC] Returns a mapping from each rank to a one-element list holding its device
        # index, wrapping around the number of visible devices of that device type.
        num_visible_devices = torch.get_device_module(device).device_count()
        return {i: [i % num_visible_devices] for i in range(self.world_size)}


def run_subtests(
    cls_inst,
    subtest_config: dict[str, list[Any]],
    test_fn: Callable,
    *test_args,
    **test_kwargs: Any,
):
    # [FUNCTION DOC] Runs test_fn as a subtest for every combination of the values in
    # subtest_config (mapping of keyword name to list of values), passing the chosen values as
    # keyword arguments. Resets dynamo before and after each combination and runs a barrier after
    # each. Args: test case instance, config, test function, extra args.
    """
    Runs a test function given by ``test_fn`` as a subtest according to the
    configurations specified by ``subtest_config``. This amortizes the
    costly setup overhead (including process spawn and initializing the
    process group) over the subtests.

    Args:
        subtest_config (Dict[str, List[Any]]): A mapping from subtest
            keyword argument name to a list of its possible values.
        test_fn (Callable): A callable that runs the actual test.
        test_args: Positional arguments to pass to ``test_fn``.
        test_kwargs: Keyword arguments to pass to ``test_fn``.
    """
    # Convert the config mapping to a list to have a fixed order
    subtest_config_items: list[tuple[str, list[Any]]] = list(subtest_config.items())
    subtest_config_keys: list[str] = [item[0] for item in subtest_config_items]
    subtest_config_values: list[list[Any]] = [item[1] for item in subtest_config_items]
    for values in itertools.product(*subtest_config_values):
        # Map keyword to chosen value
        subtest_kwargs = dict(zip(subtest_config_keys, values, strict=True))
        with cls_inst.subTest(**subtest_kwargs):
            torch._dynamo.reset()
            test_fn(*test_args, **test_kwargs, **subtest_kwargs)
            torch._dynamo.reset()
        c10d.barrier()


@functools.cache
def has_efa() -> bool:
    # [FUNCTION DOC] Cached check whether the machine has EFA by running the fi_info command;
    # returns True when it exits with code 0 and False if it fails or is not installed.
    """
    If shell command `fi_info -p efa -t FI_EP_RDM` returns exit code 0 then we assume that the machine has
    Libfabric EFA interfaces and EFA software components installed,
    see https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html.
    """

    try:
        return (
            subprocess.run(
                ["fi_info", "-p", "efa", "-t", "FI_EP_RDM"], check=False
            ).returncode
            == 0
        )
    except FileNotFoundError:
        pass
    return False


def tp_transports():
    # [FUNCTION DOC] Returns the tensorpipe transports to use, shm and uv when EFA is present (to
    # avoid InfiniBand), else None.
    """
    If the machine has Libfabric EFA interfaces and EFA software components installed it may cause
    'RuntimeError: In operator() at tensorpipe/common/ibv.h:172 "": Operation not supported' if tensorpipe
    uses InfiniBand transport, so we exclude it from tensorpipe transports,
    see https://github.com/pytorch/pytorch/issues/73885 and https://github.com/pytorch/pytorch/issues/65022
    """
    return ["shm", "uv"] if has_efa() else None


def spawn_threads_and_init_comms(
    func=None, timeout=TIMEOUT_DEFAULT, world_size=DEFAULT_WORLD_SIZE
):
    # [FUNCTION DOC] Decorator for test methods that run the method in world_size threads, each
    # with its own rank using the threaded process group. When called without a function, returns
    # a partial decorator with the given timeout and world size. Enables thread isolation mode
    # during the run and joins the threads, raising errors collected from the ranks. Args:
    # function, timeout, world size.
    # KNOWN ISSUE: the timeout argument is never used.
    """
    Wrapper to use with a test method
    """
    if func is None:
        return partial(
            spawn_threads_and_init_comms, timeout=timeout, world_size=world_size
        )

    def _run_test_method_with_multi_threads(world_size, callback):
        # [FUNCTION DOC] HELPER: Installs the threaded world, creates a shared hash store and
        # starts one thread per rank running worker with the callback. Returns the thread list.
        world = _install_threaded_pg()
        global_store = c10d.HashStore()

        def world_is_valid():
            # [FUNCTION DOC] HELPER: Returns True if the threaded world installed by this run is
            # still the active one.
            return world == c10d.distributed_c10d._world

        def worker(rank, world_pg, store):
            # [FUNCTION DOC] HELPER: Per-rank thread body: initializes the threaded process
            # group, runs the callback, records any exception in the shared queue and signals
            # other threads to terminate, and finally destroys the process group if the world is
            # still valid.
            c10d.init_process_group(
                backend="threaded", rank=rank, world_size=world_size, store=store
            )
            try:
                callback()
            except BaseException as ex:
                # Exceptions are handled in MultiThreadedTestCase
                MultiThreadedTestCase.exception_queue.put((rank, sys.exc_info()))
                ProcessLocalGroup.exception_handle(
                    ex
                )  # trigger _terminate event and awaken worker threads
            finally:
                if world_is_valid():
                    c10d.destroy_process_group()

        threads = []
        for rank in range(world_size):
            t = threading.Thread(target=worker, args=(rank, world, global_store))
            t.start()
            threads.append(t)

        return threads

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Replacement test method: turns thread isolation mode on, starts
        # the per-rank threads and joins them, always turning isolation off afterwards.
        # TODO: get test name from kwargs
        torch._C._distributed_c10d._set_thread_isolation_mode(True)
        try:
            threads = _run_test_method_with_multi_threads(
                world_size, lambda: func(self, *args, **kwargs)
            )
            # join and error handling
            MultiThreadedTestCase._join_threads(threads, func)
        finally:
            torch._C._distributed_c10d._set_thread_isolation_mode(False)

    return wrapper


class MultiThreadedTestCase(TestCase):
    # [CLASS DOC] Test runner that executes each test in world_size threads using the in-process
    # threaded process group. Subclasses must call _spawn_threads from setUp, and use
    # perThreadSetUp and perThreadTearDown for per-thread fixtures. State: rank (main thread rank
    # is MAIN_THREAD_RANK), threads, class-level exception_queue and global_store. The main thread
    # joins the threads and reports failures.
    """
    Test runner that runs all tests with the in-proc process group using
    multiple threads with the threaded process group.

    Each test spawns world_size threads and run the test method in each thread.

    Difference from regular MultiProcess test runner:
    Must explicitly defines SetUp and call self._spawn_threads() to run the tests.
    Cannot use setUp / tearDown (must use perThreadSetup / perThreadShutdown)
        to set up / tear down each thread when running each test.
    No global state possible
        How bad of a limitation is this?
    """

    exception_queue = queue.Queue()

    MAIN_THREAD_RANK = -1

    def join_or_run(self, fn):
        # [FUNCTION DOC] Returns a bound replacement for a test method: the main thread joins the
        # spawned threads while worker threads run the original test body.
        @wraps(fn)
        def wrapper(self):
            # [FUNCTION DOC] HELPER: Dispatches on whether this is the main thread or a
            # worker rank.
            if self.rank == self.MAIN_THREAD_RANK:
                self._join_threads(self.threads, fn)
            else:
                fn()

        return types.MethodType(wrapper, self)

    def __init__(
        self, method_name: str = "runTest", methodName: str = "runTest"
    ) -> None:
        # [FUNCTION DOC] Constructs the test case and replaces the named test method with a
        # version that joins threads or runs the real body. Accepts both method_name and
        # methodName. Raises ValueError if an explicitly named method does not exist.
        # methodName is the correct naming in unittest and testslide uses keyword arguments.
        # So we need to use both to 1) not break BC and, 2) support testslide.
        if methodName != "runTest":
            method_name = methodName
        super().__init__(method_name)
        try:
            fn = getattr(self, method_name)
            setattr(self, method_name, self.join_or_run(fn))
        except AttributeError as e:
            if methodName != "runTest":
                # we allow instantiation with no explicit method name
                # but not an *incorrect* or missing method name
                raise ValueError(
                    f"no such test method in {self.__class__}: {methodName}"
                ) from e

    def perThreadSetUp(self):
        # [FUNCTION DOC] Hook run in each worker thread before the test; does nothing by default.
        # super().setUp()  # TestCase.setUp() calls torch.manual_seed()
        pass

    def perThreadTearDown(self):
        # [FUNCTION DOC] Hook run in each worker thread after the test; does nothing by default.
        pass

    def setUp(self) -> None:
        # [FUNCTION DOC] Main-thread setup: base setup, sets rank to the main thread rank, clears
        # the thread list and enables C++ stack trace printing via the environment.
        """
        setUp only set up things in the main thread, if you want to configure things
        in the spawned threads, use perThreadSetUp
        """
        super().setUp()
        self.rank = self.MAIN_THREAD_RANK
        self.threads = []
        # Show full C++ stacktraces when a Python error originating from C++ is raised.
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

    def tearDown(self):
        # [FUNCTION DOC] Main-thread teardown: base teardown and clears the thread list.
        """
        tearDown only set up things in the main thread, if you want to configure things
        in the spawned threads, use perThreadTearDown
        """
        super().tearDown()
        self.threads = []

    def _spawn_threads(self):
        # [FUNCTION DOC] Enables thread isolation mode, installs the threaded world, creates a
        # class-level hash store and starts world_size threads each running _run. Raises
        # RuntimeError if the world is invalid. Intended to be called from setUp.
        """
        class method to spawn threads and run test, use this method in the SetUp of your TestCase
        """
        torch._C._distributed_c10d._set_thread_isolation_mode(True)
        test_name = self._current_test_name
        # for each test case, we need to create thread local world, and a global store
        world = _install_threaded_pg()
        self.__class__.global_store = c10d.HashStore()

        def world_is_valid():
            # [FUNCTION DOC] HELPER: Returns True if the threaded world installed here is still
            # the active one.
            return world == c10d.distributed_c10d._world

        if not world_is_valid():
            raise RuntimeError("Invalid world")

        for rank in range(self.world_size):
            t = threading.Thread(
                target=self.__class__._run, args=(test_name, rank, self.world_size)
            )
            t.start()
            self.threads.append(t)

    @classmethod
    def _run(cls, test_name, rank, world_size, **kwargs):
        # [FUNCTION DOC] HELPER: Thread entry point: builds a test instance, sets its rank, copies
        # the precision and tolerance settings into the thread-local state, then runs the test
        # with the threaded group.
        self = cls(test_name)
        self.rank = rank

        # precision/rel_tol is a thread-local setting since it may be overridden per test, need to make
        # every thread have the same value. This would be relevant when we use op db tests, where it
        # needs those states to be set i.e. using instantiate_device_type_tests()
        # TODO: figure out a better way to do this
        if hasattr(self, "_tls"):
            self._tls = threading.local()
            self._tls.precision = TestCase._precision
            self._tls.rel_tol = TestCase._rel_tol

        self.run_test_with_threaded_pg(test_name, rank, world_size)

    def run_test_with_threaded_pg(self, test_name, rank, world_size):
        # [FUNCTION DOC] Initializes the threaded process group for this rank, runs the per-thread
        # setup, then the test method. Any exception is queued with its rank and other threads are
        # signaled to terminate. Always destroys the process group and runs per-thread teardown.
        """
        Run the current test associated with `test_name` using the threaded process group.
        """
        c10d.init_process_group(
            backend="threaded",
            rank=rank,
            world_size=world_size,
            store=self.__class__.global_store,
        )
        self.perThreadSetUp()

        try:
            getattr(self, test_name)()
        except BaseException as ex:
            self.exception_queue.put((rank, sys.exc_info()))
            ProcessLocalGroup.exception_handle(
                ex
            )  # trigger _terminate event and awaken worker threads
        finally:
            c10d.destroy_process_group()
            self.perThreadTearDown()

    @classmethod
    def _join_threads(cls, threads, fn):
        # [FUNCTION DOC] HELPER: Joins all rank threads, queuing a TimeoutError for any still
        # alive after the default timeout, resets the process-local group, collects failures from
        # the exception queue and always uninstalls the threaded world and turns isolation mode
        # off. Then checks results.
        # KNOWN ISSUE: if joining raises before failed_ranks is assigned, the final check hits an
        # unbound variable.
        timeout = TIMEOUT_DEFAULT
        try:
            for idx, thread in enumerate(threads):
                thread.join(max(0, timeout))
                if thread.is_alive():
                    MultiThreadedTestCase.exception_queue.put(
                        (
                            idx,
                            (
                                TimeoutError,
                                TimeoutError(
                                    f"Rank failed to join in under {timeout} seconds"
                                ),
                                None,
                            ),
                        )
                    )
            ProcessLocalGroup.reset()
            failed_ranks = []
            while not cls.exception_queue.empty():
                failure = cls.exception_queue.get()
                failed_ranks.append(failure)
        finally:
            _uninstall_threaded_pg()
            torch._C._distributed_c10d._set_thread_isolation_mode(False)

        cls._check_return_codes(failed_ranks, timeout, fn)

    @classmethod
    def _check_return_codes(cls, failed_ranks, timeout, fn):
        # [FUNCTION DOC] HELPER: Examines failures from the threads: logs skips, raises
        # RuntimeError on timeouts or exceptions with the tracebacks, and for a skip exit code
        # raises SkipTest (or logs on sandcastle). Returns None on no failures.
        # Print based on exceptions raised from threads
        #   SkipTest: print info for each thread
        #   TimeoutError: raise RuntimeError for any timed out thread
        #   Normal Exception: print error for each thread that raises exception
        #   and raise a RuntimeError
        error_msg = ""
        skip_code = -1
        for rank, exc_info in failed_ranks:
            exc = exc_info[1]
            if isinstance(exc, unittest.SkipTest):
                logger.info(
                    "Thread %s skipping test %s for following reason: %s",
                    rank,
                    fn,
                    exc,
                )
                if skip_code < 0:
                    skip_code = TEST_SKIPS["generic"].exit_code
            elif isinstance(exc, TimeoutError):
                msg = f"Thread {rank} terminated or timed out after {timeout} seconds\n"
                logger.error(msg)
                raise RuntimeError(msg)
            elif isinstance(exc, Exception):
                msg = "".join(traceback.format_exception(*exc_info))
                logger.error("Caught exception: \n%s exiting thread %s", msg, rank)
                error_msg += f"Thread {rank} exited with exception:\n{msg}\n"
            elif isinstance(exc, SystemExit):
                if type(exc.code) is int and skip_code < 0:
                    skip_code = exc.code

        # check exceptions
        if len(error_msg) > 0:
            raise RuntimeError(error_msg)
        # check skip
        if skip_code > 0:
            for skip in TEST_SKIPS.values():
                if skip_code == skip.exit_code:
                    if IS_SANDCASTLE:
                        # "pass" the test with an appropriate message.
                        logger.info(
                            "Skipping %s on sandcastle for the following reason: %s",
                            fn,
                            skip.message,
                        )
                        return
                    else:
                        raise unittest.SkipTest(skip.message)

    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Property: number of threads (ranks), defaulting to DEFAULT_WORLD_SIZE.
        return DEFAULT_WORLD_SIZE

    @property
    def _current_test_name(self) -> str:
        # [FUNCTION DOC] HELPER: Property: the last dotted component of the test id, i.e. the test
        # method name.
        # self.id() == e.g. '__main__.TestDistributed.TestAdditive.test_get_rank'
        return self.id().split(".")[-1]

    def assertEqualOnRank(self, x, y, msg=None, *, rank=0):
        # [FUNCTION DOC] Asserts equality of x and y only on the given rank (default 0), since
        # threads share one CPU random number generator and results are only reliable there.
        """
        The reason why we have this util function instead of
        self.assertEqual is all threads are sharing one CPU RNG
        so the assertion result is only reliable on rank 0
        """
        if self.rank == rank:
            self.assertEqual(x, y, msg)

    def assertNotEqualOnRank(self, x, y, msg=None, *, rank=0):
        # [FUNCTION DOC] Asserts inequality of x and y only on the given rank (default 0).
        # KNOWN ISSUE: the msg argument is accepted but not passed on.
        if self.rank == rank:
            self.assertNotEqual(x, y)


class SaveForwardInputsModule(nn.Module):
    # [CLASS DOC] HELPER: Small test module wrapping a linear layer that records its input in a
    # shared mapping keyed by the module, optionally casting the input to the weight dtype. State:
    # l, forward_inputs, cast_forward_inputs.
    def __init__(
        self,
        forward_inputs: dict[nn.Module, torch.Tensor],
        cast_forward_inputs: bool,
    ) -> None:
        # [FUNCTION DOC] Stores the shared forward_inputs mapping and cast flag, and creates a 100
        # by 100 linear layer.
        super().__init__()
        self.l = nn.Linear(100, 100)
        self.forward_inputs = forward_inputs
        self.cast_forward_inputs = cast_forward_inputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Records the input under this module in the shared mapping, then applies
        # the linear layer, casting the input to the weight dtype if the cast flag is set. Returns
        # the output tensor.
        self.forward_inputs[self] = x
        return self.l(x.to(self.l.weight.dtype) if self.cast_forward_inputs else x)


class SaveForwardInputsModel(nn.Module):
    # [CLASS DOC] HELPER: Test model with two SaveForwardInputsModule children applied in
    # sequence that records its own input and those of the children in one shared mapping. State:
    # c1, c2, forward_inputs.
    def __init__(
        self,
        forward_inputs: dict[nn.Module, torch.Tensor],
        cast_forward_inputs: bool,
    ) -> None:
        # [FUNCTION DOC] Creates two child modules sharing the same forward_inputs mapping and
        # stores the mapping.
        super().__init__()
        self.c1 = SaveForwardInputsModule(forward_inputs, cast_forward_inputs)
        self.c2 = SaveForwardInputsModule(forward_inputs, cast_forward_inputs)
        self.forward_inputs = forward_inputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Records the input under this model in the shared mapping and returns the
        # output of the second child applied to the first child's output.
        self.forward_inputs[self] = x
        return self.c2(self.c1(x))


@contextmanager
def _dynamo_dist_per_rank_init(
    rank, world_size, backend=None, init_pg=True, fake_pg=False
):
    # [FUNCTION DOC] HELPER: Context manager for per-rank dynamo distributed setup. Sets the
    # accelerator device to the rank (unless fake_pg), sets MASTER_ADDR and MASTER_PORT
    # environment variables, initializes a process group (a fake one with a fake store when
    # fake_pg, else the given or default backend for the device type) when init_pg is true, and
    # resets dynamo and its counters. On exit resets dynamo and counters again and destroys the
    # process group if it was initialized. Mutates the environment.
    # To avoid multiple inheritance from _dynamo.test_case.TestCase and MultiProcessTestCase,
    # Just manually implement the most important part of the dynamo behavior to reset/clear.
    if not fake_pg:
        torch.accelerator.set_device_index(rank)

    device_type = (
        acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
    )
    if backend is None:
        backend = c10d.get_default_backend_for_device(device_type)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6789"
    if init_pg:
        if fake_pg:
            store = torch.testing._internal.distributed.fake_pg.FakeStore()
            c10d.init_process_group(
                backend="fake",
                world_size=world_size,
                rank=rank,
                store=store,
            )
        else:
            c10d.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    try:
        yield
    finally:
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        if init_pg:
            c10d.destroy_process_group()


class DynamoDistributedSingleProcTestCase(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Single-process dynamo test base class that initializes a world-size-1 process
    # group once for the whole class, with MASTER_ADDR and MASTER_PORT patched in the environment.
    # Class state: rank, device, device_ids.
    """
    Test harness for single-process dynamo distributed tests,
    initializes dist process group.

    Prefer this for simple tests, as it's easier to debug.
    """

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class fixture: patches environment variables, sets rank 0, derives the
        # device string from the current accelerator, and initializes a process group with the
        # default backend for that device and world size 1.
        # KNOWN ISSUE: the test for device ids checks whether the device type is contained in the
        # device string, which is always true, so device_ids is always None.
        super().setUpClass()
        # _exit_stack is set up in TestCase
        cls._exit_stack.enter_context(
            patch.dict(
                os.environ,
                {
                    "MASTER_ADDR": "localhost",
                    "MASTER_PORT": "12355",
                },
            )
        )
        cls.rank = 0
        device = torch.accelerator.current_accelerator().type
        cls.device = f"{device}:{cls.rank}"
        cls.device_ids = None if device in cls.device else [cls.rank]
        c10d.init_process_group(
            c10d.get_default_backend_for_device(device), rank=cls.rank, world_size=1
        )

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class fixture: destroys the process group and runs the base teardown.
        c10d.destroy_process_group()
        super().tearDownClass()


class DynamoDistributedMultiProcTestCase(DistributedTestBase):
    # [CLASS DOC] Dynamo distributed test base class for tests that need multiple accelerators;
    # spawns one process per accelerator through DistributedTestBase. Slow, so tests should be few
    # and decorated with a GPU count skip.
    """
    Use this for tests that actually run on multiple GPUs.

    Decorate tests with @skip_if_lt_x_gpu(ngpu)

    Note: MultiProcTestCase spawns processes per test and is slow.
    Prefer MultiThreadedTestCase for most tests. Perhaps use this one
    sparingly for integration tests.
    """

    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Property: number of ranks equal to the accelerator device count.
        return torch.accelerator.device_count()

    @classmethod
    def _run(
        cls, rank: int, test_name: str, file_name: str, parent_pipe, **kwargs
    ) -> None:
        # [FUNCTION DOC] HELPER: Worker entry point: adds a null handler to the trace log, then
        # builds a test instance, sets rank and rendezvous file, and runs the test (same as the
        # base implementation).
        trace_log.addHandler(logging.NullHandler())

        # The rest is copypasta from MultiProcessTestCase._run
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name
        self.run_test(test_name, parent_pipe)


class MultiProcContinuousTest(TestCase):
    # [CLASS DOC] Test base class that spawns world_size worker processes once per class and keeps
    # them alive across tests. For each test the main process sends the test id to every worker
    # over a task queue, workers run it and report back on a completion queue, and the main
    # process re-raises failures or skips. Class state: world_size, rank, rdvz_file, timeout,
    # poison_pill (set after a failure so later tests skip), processes, task_queues and
    # completion_queues, and a lazy spawn flag. Subclasses may override backend_str, device_type
    # and opts.
    # Class variables:
    MAIN_PROCESS_RANK = -1
    # number of test processes
    world_size: int = -2  # unset state
    # rank of the current process
    rank: int = -2  # unset state
    # Rendezvous file
    rdvz_file: str | None = None
    # timeout configured per class
    timeout: timedelta = timedelta(seconds=120)
    # Poison pill for rest of tests if one of them fails
    poison_pill: bool = False
    # Flag for lazy process spawning (to support instantiate_device_type_tests)
    _processes_spawned: bool = False

    @classmethod
    def backend_str(cls) -> str | None:
        # [FUNCTION DOC] Class method returning the backend name for the process group; None by
        # default so it is decided from tensors. Override in subclasses.
        """
        ProcessGroup backend str.
        To be customized by sub test classes, e.g. "nccl".
        Otherwise we return None -- lazily decided by tensor.
        """
        return None

    # Please override if you intend to test on specific device type
    @classmethod
    def device_type(cls) -> str:
        # [FUNCTION DOC] Class method returning the device type string: the current accelerator
        # type, else cpu. Override to test a specific device.
        curr_device = torch.accelerator.current_accelerator()
        if curr_device is None:
            return "cpu"
        return curr_device.type

    @classmethod
    def opts(cls, high_priority_stream=False):
        # [FUNCTION DOC] Class method returning process group options; None by default. Override
        # to supply backend options.
        """
        ProcessGroup init options.
        To be customized by sub test classes, e.g. ProcessGroupNCCLOpTest
        Here we return None.
        """
        return None

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        # [FUNCTION DOC] HELPER: Worker-side initialization of the default process group: requires
        # an rendezvous file (raises AssertionError otherwise), sets LOCAL_RANK, creates a
        # FileStore, initializes the process group with the backend, options and timeout of the
        # class and saves the default group on the class.
        if rdvz_file is None:
            raise AssertionError("Expected rdvz_file to not be None")
        # rank should be local_rank for tests running on <= 8 gpus which is how all these tests are designed
        # and we expect LOCAL_RANK set by torchrun. Setting it lets init_device_mesh set the device without
        # issuing a warning
        os.environ["LOCAL_RANK"] = str(rank)
        store = c10d.FileStore(rdvz_file, world_size)
        # create nccl processgroup with opts
        c10d.init_process_group(
            backend=cls.backend_str(),
            world_size=world_size,
            rank=rank,
            store=store,
            pg_options=cls.opts(),
            timeout=cls.timeout,
        )
        cls.pg = c10d.distributed_c10d._get_default_group()

    @classmethod
    def _run_test_given_id(cls, test_id: str, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Worker-side execution of one test by id: creates an instance from
        # the test name, sets its rank and world size, seeds the random number generators and
        # calls the test method with the keyword arguments.
        # self.id() == e.g. '__main__.TestDistributed.TestAdditive.test_get_rank'
        test_name = test_id.rsplit(".", maxsplit=1)[-1]
        # Get the test function from the test class
        self = cls(test_name)
        self.rank = cls.rank
        self.world_size = cls.world_size
        test_fn = getattr(self, test_name)

        # Ensure all the ranks use the same seed.
        common_utils.set_rng_seed()

        # Run the test function
        test_fn(**kwargs)

    @classmethod
    def _worker_loop(cls, rank, world_size, rdvz_file, task_queue, completion_queue):
        # [FUNCTION DOC] HELPER: Main loop of a worker process: validates rank, stores rank and
        # world size on the class, initializes the process group (an initialization exit with a
        # skip code is remembered and makes every test a skip), then repeatedly reads test ids
        # from the task queue until a stop sentinel. Reports each result, a skip exception or a
        # wrapped error with traceback on the completion queue. Destroys the process group at the
        # end only if no test raised.
        raised_exception = False
        # Sub tests are going to access these values, check first
        if not (0 <= rank < world_size):
            raise AssertionError(
                f"Expected 0 <= rank < world_size, got rank={rank}, world_size={world_size}"
            )
        # set class variables for the test class
        cls.rank = rank
        cls.world_size = world_size

        # Initialize the process group
        init_skip_reason = None
        try:
            cls._init_pg(rank, world_size, rdvz_file)
        except SystemExit as ex:
            exit_code = getattr(ex, "code", None)
            skip_entry = next(
                (v for v in TEST_SKIPS.values() if v.exit_code == exit_code),
                None,
            )
            if skip_entry:
                init_skip_reason = skip_entry.message
            else:
                raise

        # End of bootstrap
        logger.debug("Setup complete")

        # Loop forever, waiting for a test name to run
        while True:
            test_id = task_queue.get()
            logger.debug(f"Got test {test_id}")  # noqa: G004
            # None means exit
            if test_id is None:
                break

            # If init failed with a skip, respond with SkipTest for all tests
            if init_skip_reason is not None:
                completion_queue.put(unittest.SkipTest(init_skip_reason))
                continue

            # Run the test
            try:
                cls._run_test_given_id(test_id)
                completion_queue.put(test_id)
            except BaseException as ex:
                if isinstance(ex, unittest.SkipTest):
                    completion_queue.put(ex)
                    continue
                if isinstance(ex, SystemExit):
                    # Get exit code from the process
                    exit_code = getattr(ex, "code", None)

                    # Look up exit code in TEST_SKIPS to see if it is a valid skip
                    skip_entry = next(
                        (v for v in TEST_SKIPS.values() if v.exit_code == exit_code),
                        None,
                    )

                    # If we found an entry, we want to skip the test and the object back to the main process
                    if skip_entry:
                        completion_queue.put(unittest.SkipTest(skip_entry.message))
                        # Skip exception handling below, move to main thread for processing the skip
                        continue

                raised_exception = True
                # Send the exception and stack trace back to the dispatcher
                exc_info = sys.exc_info()
                tb_str = "".join(traceback.format_exception(*exc_info))
                # Create a new exception with the original exception and traceback
                enhanced_ex = RuntimeError(f"Exception in worker process:\n{tb_str}")
                enhanced_ex.__cause__ = ex
                completion_queue.put(enhanced_ex)

        # Termination
        logger.debug("Terminating ...")
        # Calling destroy_process_group when workers have exceptions
        # while others are doing collectives will cause a deadlock since
        # it waits for enqueued collectives to finish.
        # Only call this on a clean exit path
        if not raised_exception:
            c10d.destroy_process_group()

    @classmethod
    def _spawn_processes(cls, world_size) -> None:
        # [FUNCTION DOC] HELPER: Class method: creates the rendezvous file, sets the
        # multiprocessing start method to spawn, and starts one daemon worker process per rank
        # with its own task queue and completion queue. Stores them on the class.
        cls.processes = []
        cls.task_queues = []
        cls.completion_queues = []
        # Need a rendezvous file for `init_process_group` purpose.
        with tempfile.NamedTemporaryFile(delete=False) as f:
            cls.rdvz_file = f.name

        # CUDA multiprocessing requires spawn instead of fork, to make sure
        # child processes have their own memory space.
        try:
            torch.multiprocessing.set_start_method("spawn")
        except RuntimeError:
            # The start method has already been set
            pass

        for rank in range(int(world_size)):
            task_queue = torch.multiprocessing.Queue()
            completion_queue = torch.multiprocessing.Queue()
            process = torch.multiprocessing.Process(
                target=cls._worker_loop,
                name="process " + str(rank),
                daemon=True,  # so that child processes will exit if parent decides to terminate
                args=(rank, world_size, cls.rdvz_file, task_queue, completion_queue),
            )
            process.start()
            cls.processes.append(process)
            cls.task_queues.append(task_queue)
            cls.completion_queues.append(completion_queue)
            logger.debug("Started process %s with pid %s", rank, process.pid)

    @classmethod
    def _get_world_size(cls, device_type: str) -> int:
        # [FUNCTION DOC] HELPER: Class method returning the world size, evaluating a property
        # defined on the class using a temporary instance created without running the constructor.
        # If the world size is unset, uses the device count of the device type and raises SkipTest
        # if there are none.
        """
        Get world_size, handling both class variable and property definitions.
        Properties are instance-level and need special handling in class methods.
        """
        # Check if world_size is defined as a property (instance-level)
        world_size_attr = inspect.getattr_static(cls, "world_size", None)
        if isinstance(world_size_attr, property):
            # Create a temporary instance to evaluate the property
            # We use object.__new__ to avoid calling __init__ which may have side effects
            temp_instance = object.__new__(cls)
            world_size = world_size_attr.fget(temp_instance)
        else:
            world_size = cls.world_size

        # If world_size is not set (== -2), use device count
        if world_size == -2:
            world_size = torch.get_device_module(device_type).device_count()
            if world_size == 0:
                raise unittest.SkipTest(f"No {device_type} devices available")

        return world_size

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class fixture that only runs the base setup; process spawning is deferred
        # to the first test to support device-type test instantiation.
        """
        Class-scope test fixture. Run once for entire test class, before any test starts.
        Note: Process spawning is deferred to setUp to support instantiate_device_type_tests,
        which calls setUpClass during class creation before any tests run.
        """
        super().setUpClass()

    @classmethod
    def _ensure_processes_spawned(cls):
        # [FUNCTION DOC] HELPER: Class method: spawns the worker processes lazily once per class
        # (checking the class own dict so subclasses do not reuse a finished base class). Resolves
        # the device type, sets the world size, raises SkipTest if the requested backend is
        # unavailable, spawns the workers and marks them spawned.
        """
        Lazily spawn worker processes on first test run.
        This supports instantiate_device_type_tests which calls setUpClass during
        class creation (before any tests run), when spawning would be premature.
        """
        # Check the class's own dict: a subclass must not inherit the flag from
        # a base test class that already ran and tore down its workers, else it
        # would dispatch tests to the base class's dead worker processes.
        if cls.__dict__.get("_processes_spawned", False):
            return

        # Handle method, property, and string attribute for device_type
        # (instantiate_device_type_tests sets device_type as a string attribute,
        # making this compatible as a drop-in replacement for MultiProcessTestCase)
        device_type_attr = cls.__dict__.get("device_type", cls.device_type)
        if isinstance(device_type_attr, classmethod):
            device_type = device_type_attr.__func__(cls)
        elif isinstance(device_type_attr, property):
            # Note: fget expects an instance but we pass cls since no instance
            # exists yet. This works because DTensorTestMixin.device_type only
            # accesses class-level attributes (world_size, module constants).
            device_type = device_type_attr.fget(cls)
        elif callable(device_type_attr):
            device_type = device_type_attr()
        else:
            device_type = device_type_attr

        # Get world_size (handles both class variable and property)
        cls.world_size = cls._get_world_size(device_type)

        # Check if the specified backend is available before spawning processes
        backend = cls.backend_str() if callable(cls.backend_str) else cls.backend_str
        if backend is not None:
            backend_checks = {
                "nccl": c10d.is_nccl_available,
                "gloo": c10d.is_gloo_available,
                "mpi": c10d.is_mpi_available,
                "xccl": c10d.is_xccl_available,
            }
            check_fn = backend_checks.get(backend)
            if check_fn is not None and not check_fn():
                raise unittest.SkipTest(f"Backend '{backend}' is not available")

        logger.info(
            f"Testing class {cls.__name__} on {cls.world_size} {device_type}"  # noqa: G004
        )

        cls._spawn_processes(cls.world_size)
        cls._processes_spawned = True

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class fixture: if workers were spawned, sends a stop sentinel to each
        # worker, joins them and removes the rendezvous file, then runs the base teardown. Does
        # nothing special if no workers were spawned.
        """
        Class-scope test fixture. Run once for entire test class, after all tests finish.
        Tear down the process group if spawned.
        """
        # If processes were never spawned (e.g., all tests were skipped), nothing to tear down.
        # Check the class's own dict to avoid tearing down an already-finished
        # base class's workers (see _ensure_processes_spawned).
        if not cls.__dict__.get("_processes_spawned", False):
            super().tearDownClass()
            return

        logger.debug(f"Joining {cls.world_size} workers")  # noqa: G004
        # Enqueue "None" to all workers to tell them to exit
        for task_queue in cls.task_queues:
            task_queue.put(None)

        # Wait for all workers to exit
        for process in cls.processes:
            process.join()

        # Clear up the rendezvous file
        try:
            os.remove(cls.rdvz_file)
        except OSError:
            pass

        logger.info(f"Class {cls.__name__} finished")  # noqa: G004
        super().tearDownClass()

    def setUp(self) -> None:
        # [FUNCTION DOC] Per-test setup: ensures workers are spawned, sets this instance as the
        # dispatcher, raises SkipTest if a previous test poisoned the class, and sends the test id
        # to all workers.
        """
        Test fixture. Run before each test.
        """
        super().setUp()

        # Ensure processes are spawned (lazy initialization for instantiate_device_type_tests)
        self.__class__._ensure_processes_spawned()

        # I am the dispatcher
        self.rank = self.MAIN_PROCESS_RANK

        # If this test class hits an exception in one test, skip the rest of tests
        if self.__class__.poison_pill:
            raise unittest.SkipTest(f"Previous test failed, skipping {self.id()}")

        # Enqueue "current test" to all workers
        for i, task_queue in enumerate(self.task_queues):
            logger.debug(f"Sending Rank {i}: {self.id()}")  # noqa: G004
            task_queue.put(self.id())

    def _worker_run_main_wait(self, fn):
        # [FUNCTION DOC] HELPER: Returns a bound replacement for a test method. In the dispatcher
        # it waits for each worker result, draining all queues before raising the first deferred
        # skip or error (an error also sets poison_pill) and checks the result equals the test id;
        # in a worker it just runs the test.
        @wraps(fn)
        def wrapper(self):
            # [FUNCTION DOC] HELPER: Dispatches on whether this instance is the main dispatcher or
            # a worker rank.
            if self.rank == self.MAIN_PROCESS_RANK:
                logger.debug(f"Waiting for workers to finish {self.id()}")  # noqa: G004
                # Drain all completion queues before raising any exception,
                # so stale results don't desync subsequent tests.
                deferred_exception = None
                for i, (p, completion_queue) in enumerate(
                    zip(self.processes, self.completion_queues)
                ):
                    rv = retrieve_result_from_completion_queue(
                        p, completion_queue, timeout=get_timeout(self.id())
                    )
                    if deferred_exception is not None:
                        # Already captured an exception; just drain
                        continue
                    if isinstance(rv, unittest.SkipTest):
                        deferred_exception = rv
                        continue
                    if isinstance(rv, BaseException):
                        logger.warning(
                            f"Detected failure from Rank {i} in: {self.id()}, "  # noqa: G004
                            f"skipping rest of tests in Test class: {self.__class__.__name__}"
                        )
                        self.__class__.poison_pill = True
                        deferred_exception = rv
                        continue

                    # Success
                    if rv != self.id():
                        raise AssertionError(
                            f"Expected rv == self.id(), got {rv} != {self.id()}"
                        )
                    logger.debug(
                        f"Main proc detected rank {i} finished {self.id()}"  # noqa: G004
                    )

                if deferred_exception is not None:
                    raise deferred_exception
            else:
                # Worker just runs the test
                fn()

        return types.MethodType(wrapper, self)

    # The main process spawns N subprocesses that run the test.
    # Constructor patches current instance test method to
    # assume the role of the main process and join its subprocesses,
    # or run the underlying test function.
    def __init__(
        self, method_name: str = "runTest", methodName: str = "runTest"
    ) -> None:
        # [FUNCTION DOC] Constructs the test case and replaces the named test method with the
        # dispatcher or worker wrapper. Accepts method_name and methodName. Raises ValueError if
        # an explicitly named method does not exist.
        # methodName is the correct naming in unittest and testslide uses keyword arguments.
        # So we need to use both to 1) not break BC and, 2) support testslide.
        if methodName != "runTest":
            method_name = methodName
        super().__init__(method_name)
        try:
            fn = getattr(self, method_name)
            setattr(self, method_name, self._worker_run_main_wait(fn))
        except AttributeError as e:
            if methodName != "runTest":
                # we allow instantiation with no explicit method name
                # but not an *incorrect* or missing method name
                raise ValueError(
                    f"no such test method in {self.__class__}: {methodName}"
                ) from e


class C10dTorchCommsTestBase(MultiProcContinuousTest):
    # [CLASS DOC] Base class for continuous multi-process tests using torchcomms; picks the
    # backend by device type, skips when the torchcomms backend is not built, and configures
    # torchcomms environment variables on each worker. Class state: world_size.
    world_size: int = DEFAULT_WORLD_SIZE

    @staticmethod
    def backend(device) -> str:
        # [FUNCTION DOC] Static method returning the backend name for a device string: nccl for
        # cuda, hccl for hpu, xccl for xpu, else gloo.
        if "cuda" in device:
            return "nccl"
        elif "hpu" in device:
            return "hccl"
        elif "xpu" in device:
            return "xccl"
        else:
            return "gloo"

    @classmethod
    def backend_str(cls) -> str:
        # [FUNCTION DOC] Class method returning the backend name for the class device type, which
        # may be a string or a callable.
        device_type = cls.device_type
        if callable(device_type):
            device_type = device_type()
        return cls.backend(device_type)

    def _skip_if_backend_unavailable(self, device: str) -> None:
        # [FUNCTION DOC] HELPER: Skips the test (self.skipTest) if the torchcomms backend for the
        # device was not built, using the TORCHCOMM_HAS_* globals.
        backend_flags = {
            "gloo": TORCHCOMM_HAS_GLOO,
            "xccl": TORCHCOMM_HAS_XCCL,
            "nccl": TORCHCOMM_HAS_NCCL,
            "rccl": TORCHCOMM_HAS_RCCL,
            "ncclx": TORCHCOMM_HAS_NCCLX,
            "rcclx": TORCHCOMM_HAS_RCCLX,
        }
        backend_name = self.backend(device)
        if backend_name in backend_flags and not backend_flags[backend_name]:
            self.skipTest(f"torchcomms {backend_name} backend is not available")

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        # [FUNCTION DOC] HELPER: Worker-side initialization: enables the torchcomms config
        # flag, sets MASTER_ADDR, MASTER_PORT, TORCHCOMM_RANK, TORCHCOMM_SIZE and
        # TORCHCOMM_STORE_PATH environment variables, calls the base initialization, and for
        # nccl or xccl sets the default accelerator device to the rank, raising RuntimeError if
        # no accelerator is present.
        torch.distributed.config.use_torchcomms = True
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["TORCHCOMM_RANK"] = str(rank)
        os.environ["TORCHCOMM_SIZE"] = str(world_size)
        os.environ["TORCHCOMM_STORE_PATH"] = rdvz_file
        super()._init_pg(rank, world_size, rdvz_file)
        # Set up accelerator device if using nccl/xccl backend
        backend = cls.backend_str()
        if "nccl" in backend or "xccl" in backend:
            accelerator = torch.accelerator.current_accelerator()
            if accelerator:
                device = torch.device(f"{accelerator.type}:{rank}")
                torch.set_default_device(device)
                torch.accelerator.set_device_index(device)
            else:
                raise RuntimeError(
                    f"Expected to find an accelerator when initializing process group with {backend} backend, but got None"
                )

    def setUp(self) -> None:
        # [FUNCTION DOC] Per-test setup: resolves the device type, skips if the torchcomms backend
        # is unavailable, then runs the base setup.
        device_type = self.__class__.device_type
        logger.debug("Setting up test: %s on device type: %s", self.id(), device_type)
        if callable(device_type):
            device_type = device_type()
        self._skip_if_backend_unavailable(str(device_type))
        super().setUp()

    def rank_to_device(self, device):
        # [FUNCTION DOC] Returns a mapping from each rank to a one-element list of its device
        # index, wrapping around the visible device count.
        num_visible_devices = torch.get_device_module(device).device_count()
        return {i: [i % num_visible_devices] for i in range(self.world_size)}
