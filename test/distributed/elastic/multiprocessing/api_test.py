#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]
# [MODULE DOC] Tests for torch.distributed.elastic.multiprocessing (api): RunProcsResult and
# Std helpers, and start_processes launching workers either as python functions (spawn or
# forkserver, multiprocessing context) or as binary subprocesses. Covers return values,
# exit codes, signal and exception failures with error files, stdout/stderr redirects, tee,
# duplicate-log filters and line prefixes, argument validation, process waiting/closing
# (including bounded close for unkillable children) and no-zombie behavior when the parent is
# signaled. Defines many worker functions (echo0 to echo_large, dummy_compute, wait_fn) and
# helpers (bin, start_processes_zombie_test). Test classes are defined conditionally: most
# are skipped under dev-debug ASAN, Windows and macOS, and the "not CI" and fork-server
# groups also under CI. Runs via run_tests when executed as a script.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import asyncio
import ctypes
import multiprocessing
import os
import shutil
import signal
import sys
import tempfile
import time
import unittest
from collections.abc import Callable
from itertools import product
from unittest import mock

import torch
import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing import ProcessFailure, start_processes
from torch.distributed.elastic.multiprocessing.api import (
    _validate_full_rank,
    _wrap,
    DefaultLogsSpecs,
    MultiprocessContext,
    RunProcsResult,
    SignalException,
    Std,
    to_map,
)
from torch.distributed.elastic.multiprocessing.errors import ErrorHandler
from torch.testing._internal.common_utils import (
    IS_CI,
    IS_LINUX,
    IS_MACOS,
    IS_WINDOWS,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    skip_if_pytest,
    TEST_WITH_ASAN,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_WITH_ROCM,
    TEST_WITH_TSAN,
    TestCase,
)


class RunProcResultsTest(TestCase):
    # [CLASS DOC] Tests for RunProcsResult and ProcessFailure helpers. Each test has a temp
    # directory (test_dir) removed in tearDown.
    def setUp(self):
        # [FUNCTION DOC] Creates the per-test temp directory.
        super().setUp()
        self.test_dir = tempfile.mkdtemp(prefix=f"{self.__class__.__name__}_")

    def tearDown(self):
        # [FUNCTION DOC] Calls the parent tearDown and deletes the temp directory.
        super().tearDown()
        shutil.rmtree(self.test_dir)

    def test_is_failed(self):
        # [FUNCTION DOC] A result with only return values is not failed; a result holding a
        # failure entry is failed.
        pr_success = RunProcsResult(return_values={0: "a", 1: "b"})
        self.assertFalse(pr_success.is_failed())

        fail0 = ProcessFailure(
            local_rank=0, pid=998, exitcode=1, error_file="ignored.json"
        )
        pr_fail = RunProcsResult(failures={0: fail0})
        self.assertTrue(pr_fail.is_failed())

    def test_get_failures(self):
        # [FUNCTION DOC] Records two exceptions to error files, builds three ProcessFailure
        # objects (the third with a nonexistent file) and checks their timestamps are
        # non-decreasing.
        # KNOWN ISSUE: both exceptions are recorded to the first file, so the second error
        # file is never written, and no failure collection API is actually exercised.
        error_file0 = os.path.join(self.test_dir, "error0.json")
        error_file1 = os.path.join(self.test_dir, "error1.json")
        eh = ErrorHandler()
        with mock.patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": error_file0}):
            eh.record_exception(RuntimeError("error 0"))

        with mock.patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": error_file0}):
            eh.record_exception(RuntimeError("error 1"))

        fail0 = ProcessFailure(
            local_rank=0, pid=997, exitcode=1, error_file=error_file0
        )
        fail1 = ProcessFailure(
            local_rank=1, pid=998, exitcode=3, error_file=error_file1
        )
        fail2 = ProcessFailure(
            local_rank=2, pid=999, exitcode=15, error_file="no_exist.json"
        )

        self.assertLessEqual(fail0.timestamp, fail1.timestamp)
        self.assertLessEqual(fail1.timestamp, fail2.timestamp)


class StdTest(TestCase):
    # [CLASS DOC] Tests for parsing Std stream selections with Std.from_str.
    def test_from_value(self):
        # [FUNCTION DOC] The strings 0, 1, 2 and 3 parse to NONE, OUT, ERR and ALL.
        self.assertEqual(Std.NONE, Std.from_str("0"))
        self.assertEqual(Std.OUT, Std.from_str("1"))
        self.assertEqual(Std.ERR, Std.from_str("2"))
        self.assertEqual(Std.ALL, Std.from_str("3"))

    def test_from_value_map(self):
        # [FUNCTION DOC] Comma-separated rank:value pairs parse into a mapping from rank to Std.
        self.assertEqual({0: Std.OUT}, Std.from_str("0:1"))
        self.assertEqual({0: Std.OUT, 1: Std.OUT}, Std.from_str("0:1,1:1"))

    def test_from_str_bad_input(self):
        # [FUNCTION DOC] Malformed inputs (trailing comma, out-of-range value, missing colon
        # in a list) must make Std.from_str raise ValueError.
        bad_inputs = ["0:1,", "11", "0:1,1", "1,0:1"]
        for bad in bad_inputs:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    Std.from_str(bad)


def echo0(msg: str) -> None:
    # [FUNCTION DOC] HELPER: Worker entrypoint that prints msg to stdout and returns nothing.
    """
    void function
    """
    print(msg)


def echo1(msg: str, exitcode: int = 0) -> str:
    # [FUNCTION DOC] HELPER: Worker entrypoint reading the RANK environment variable. A
    # nonzero exitcode prints an exit notice to stderr and terminates the process with that
    # code. Otherwise, for each comma-separated message it prints a stdout and a stderr line
    # naming the message and rank, and returns the message joined with the rank by an
    # underscore.
    """
    returns ``msg`` or exits with the given exitcode (if nonzero)
    """

    rank = int(os.environ["RANK"])
    if exitcode != 0:
        print(f"exit {exitcode} from {rank}", file=sys.stderr)
        sys.exit(exitcode)
    else:
        for m in msg.split(","):
            print(f"{m} stdout from {rank}")
            print(f"{m} stderr from {rank}", file=sys.stderr)
        return f"{msg}_{rank}"


def echo2(msg: str, fail: bool = False) -> str:
    # [FUNCTION DOC] HELPER: Worker entrypoint returning msg, or raising RuntimeError with
    # msg as its text when fail is true.
    """
    returns ``msg`` or raises a RuntimeError if ``fail`` is set
    """
    if fail:
        raise RuntimeError(msg)
    return msg


def echo_large(size: int) -> dict[int, str]:
    # [FUNCTION DOC] HELPER: Worker entrypoint returning a mapping of size entries from each
    # index to a "test<index>" string, used to exceed the pipe buffer when sent back.
    """
    returns a large output ({0: test0", 1: "test1", ..., (size-1):f"test{size-1}"})
    """
    out = {}
    for idx in range(size):
        out[idx] = f"test{idx}"
    return out


def echo3(msg: str, fail: bool = False) -> str:
    # [FUNCTION DOC] HELPER: Worker entrypoint returning msg, or crashing the process with
    # SIGSEGV (by reading address zero through ctypes) when fail is true. Currently unused.
    """
    returns ``msg`` or induces a SIGSEGV if ``fail`` is set
    """
    if fail:
        ctypes.string_at(0)
    return msg


def dummy_compute() -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Worker entrypoint returning a random 100 by 100 tensor.
    """
    returns a predefined size random Tensor
    """
    return torch.rand(100, 100)


def redirects_oss_test() -> list[Std]:
    # [FUNCTION DOC] HELPER: Returns the redirect settings used in OSS test runs, a list
    # holding only Std.NONE.
    return [
        Std.NONE,
    ]


def redirects_all() -> list[Std]:
    # [FUNCTION DOC] HELPER: Returns every redirect setting: NONE, OUT, ERR and ALL.
    return [
        Std.NONE,
        Std.OUT,
        Std.ERR,
        Std.ALL,
    ]


def bin(name: str):
    # [FUNCTION DOC] HELPER: Returns the absolute path of a script in the bin directory next
    # to this test file. (Shadows the builtin of the same name inside this module.)
    dir = os.path.dirname(__file__)
    return os.path.join(dir, "bin", name)


def wait_fn(wait_time: int = 300) -> None:
    # [FUNCTION DOC] HELPER: Worker entrypoint that sleeps for wait_time seconds (default
    # 300) and then prints a completion message; tests kill it earlier.
    time.sleep(wait_time)
    print("Finished waiting")


def start_processes_zombie_test(
    idx: int,
    entrypoint: str | Callable,
    mp_queue: mp.Queue,
    log_dir: str,
    nproc: int = 2,
) -> None:
    # [FUNCTION DOC] HELPER: Runs inside a spawned parent process. Starts nproc workers
    # running entrypoint with empty args and envs, puts its own pid and then each worker pid
    # on mp_queue, and waits up to 300 seconds for them. If a SignalException arrives (the
    # test signals this parent), it closes the workers with that signal. The idx parameter is
    # unused (shadowed by the loop variable).
    """
    Starts processes
    """

    args = {}
    envs = {}
    for idx in range(nproc):
        args[idx] = ()
        envs[idx] = {}

    pc = start_processes(
        name="zombie_test",
        entrypoint=entrypoint,
        args=args,
        envs=envs,
        logs_specs=DefaultLogsSpecs(log_dir=log_dir),
    )
    my_pid = os.getpid()
    mp_queue.put(my_pid)
    for child_pid in pc.pids().values():
        mp_queue.put(child_pid)

    try:
        pc.wait(period=1, timeout=300)
    except SignalException as e:
        pc.close(e.sigval)


class _StartProcessesTest(TestCase):
    # [CLASS DOC] HELPER: Base test case for start_processes tests. State: a temp directory
    # (test_dir) removed in tearDown and the list of multiprocessing start methods to
    # exercise (_start_methods, default spawn only). Provides log directory creation,
    # file-content assertions, pid-gone assertions and the zombie-process workflow.
    def setUp(self):
        # [FUNCTION DOC] Creates the temp directory and sets the start methods to spawn.
        super().setUp()
        self.test_dir = tempfile.mkdtemp(prefix=f"{self.__class__.__name__}_")
        self._start_methods = ["spawn"]

    def tearDown(self):
        # [FUNCTION DOC] Calls the parent tearDown and deletes the temp directory.
        super().tearDown()
        shutil.rmtree(self.test_dir)

    def log_dir(self):
        # [FUNCTION DOC] HELPER: Returns a new unique directory under the test temp directory.
        return tempfile.mkdtemp(dir=self.test_dir)

    def assert_in_file(self, expected: list[str], filename: str) -> None:
        # [FUNCTION DOC] HELPER: Reads the file's lines and asserts each expected string, with
        # trailing whitespace trimmed and a newline added, is one of the file's lines.
        expected = [f"{line.rstrip()}\n" for line in expected]
        with open(filename) as fp:
            actual = fp.readlines()
            for line in expected:
                self.assertIn(line, actual)

    def assert_not_in_file(self, lines: list[str], filename: str) -> None:
        # [FUNCTION DOC] HELPER: Reads the file's lines and asserts none of the given strings
        # (trailing whitespace trimmed, newline added) is one of them.
        lines = [f"{line.rstrip()}\n" for line in lines]
        with open(filename) as fp:
            actual = fp.readlines()
            for line in lines:
                self.assertNotIn(line, actual)

    def assert_pids_noexist(self, pids: dict[int, int]):
        # [FUNCTION DOC] HELPER: For each pid in the rank-to-pid mapping, sends signal zero
        # and asserts an OSError results, i.e. the process no longer exists.
        for local_rank, pid in pids.items():
            with self.assertRaises(
                OSError, msg=f"local_rank: {local_rank} pid: {pid} should not exist"
            ):
                os.kill(pid, 0)

    def _test_zombie_workflow(
        self, entrypoint: str | Callable, signal_to_send: signal.Signals
    ) -> None:
        # [FUNCTION DOC] HELPER: Spawns a parent process (start_processes_zombie_test) that
        # starts two workers on the entrypoint, collects the parent and worker pids from a
        # queue (120 second timeout each), sends SIGTERM to the parent, waits 5 seconds and
        # asserts every worker pid is gone.
        # KNOWN ISSUE: the signal_to_send argument is ignored; SIGTERM is always sent.
        mp_queue = mp.get_context("spawn").Queue()
        child_nproc = 2
        mp.spawn(
            start_processes_zombie_test,
            nprocs=1,
            args=(entrypoint, mp_queue, self.log_dir(), child_nproc),
            join=False,
        )
        total_processes = child_nproc + 1
        pids = []
        for _ in range(total_processes):
            pids.append(mp_queue.get(timeout=120))
        parent_pid = pids[0]
        child_pids = pids[1:]

        os.kill(parent_pid, signal.SIGTERM)
        # Wait to give time for signal handlers to finish work
        time.sleep(5)
        for child_pid in child_pids:
            # Killing parent should kill all children, we expect that each call to
            # os.kill would raise OSError
            with self.assertRaises(OSError):
                os.kill(child_pid, 0)


# tests incompatible with tsan or asan
if not (TEST_WITH_DEV_DBG_ASAN or IS_WINDOWS or IS_MACOS):

    class StartProcessesAsFuncTest(_StartProcessesTest):
        # [CLASS DOC] start_processes tests with python function entrypoints: argument
        # validation, waiting and closing, return values and raised exceptions.
        def test_to_map(self):
            # [FUNCTION DOC] Checks to_map expands a single Std to every local rank, fills
            # missing ranks of a partial mapping with NONE, and keeps a full mapping unchanged.
            local_world_size = 2
            self.assertEqual(
                {0: Std.OUT, 1: Std.OUT}, to_map(Std.OUT, local_world_size)
            )
            self.assertEqual(
                {0: Std.NONE, 1: Std.OUT}, to_map({1: Std.OUT}, local_world_size)
            )
            self.assertEqual(
                {0: Std.ERR, 1: Std.OUT},
                to_map({0: Std.ERR, 1: Std.OUT}, local_world_size),
            )

        def test_invalid_log_dir(self):
            # [FUNCTION DOC] Passing the path of a regular file as the log directory makes
            # start_processes raise NotADirectoryError; any started context is closed.
            with tempfile.NamedTemporaryFile(dir=self.test_dir) as not_a_dir:
                cases = {
                    not_a_dir.name: NotADirectoryError,
                }

                for log_dir, expected_error in cases.items():
                    with self.subTest(log_dir=log_dir, expected_error=expected_error):
                        with self.assertRaises(expected_error):
                            pc = None
                            try:
                                pc = start_processes(
                                    name="echo",
                                    entrypoint=echo1,
                                    args={0: ("hello",)},
                                    envs={0: {"RANK": "0"}},
                                    logs_specs=DefaultLogsSpecs(log_dir=log_dir),
                                )
                            finally:
                                if pc:
                                    pc.close()

        def test_args_env_len_mismatch(self):
            # [FUNCTION DOC] When the args and envs mappings differ in size (either one
            # larger), start_processes must raise RuntimeError.
            cases = [
                # 1 x args; 2 x envs
                {
                    "args": {0: ("hello",)},
                    "envs": {0: {"RANK": "0"}, 1: {"RANK": "1"}},
                },
                # 2 x args; 1 x envs
                {
                    "args": {0: ("hello",), 1: ("world",)},
                    "envs": {0: {"RANK": "0"}},
                },
            ]

            for kwds in cases:
                args = kwds["args"]
                envs = kwds["envs"]
                with self.subTest(args=args, envs=envs):
                    with self.assertRaises(RuntimeError):
                        start_processes(
                            name="echo",
                            entrypoint=echo1,
                            args=args,
                            envs=envs,
                            logs_specs=DefaultLogsSpecs(log_dir=self.log_dir()),
                        )

        def test_pcontext_wait(self):
            # [FUNCTION DOC] Starts a one second sleep worker. A wait with a very short
            # timeout returns None (still running); a later wait without timeout returns a
            # result, after which every log tail is stopped.
            pc = start_processes(
                name="sleep",
                entrypoint=time.sleep,
                args={0: (1,)},
                envs={0: {}},
                logs_specs=DefaultLogsSpecs(log_dir=self.log_dir()),
                start_method="spawn",
            )

            self.assertIsNone(pc.wait(timeout=0.1, period=0.01))
            self.assertIsNotNone(pc.wait(period=0.1))
            for tail_log in pc._tail_logs:
                self.assertTrue(tail_log.stopped())

        def test_pcontext_wait_on_a_child_thread(self):
            # [FUNCTION DOC] Runs the wait test inside a worker thread of an event loop, to
            # check start and wait work when not called from the main thread.
            asyncio.run(asyncio.to_thread(self.test_pcontext_wait))

        def test_multiprocess_context_close(self):
            # [FUNCTION DOC] Starts a sleep worker, closes the context immediately, and checks
            # the worker pids no longer exist and all log tails are stopped.
            pc = start_processes(
                name="sleep",
                entrypoint=time.sleep,
                args={0: (1,)},
                envs={0: {}},
                logs_specs=DefaultLogsSpecs(log_dir=self.log_dir()),
                start_method="spawn",
            )

            pids = pc.pids()
            pc.close()
            self.assert_pids_noexist(pids)
            for tail_log in pc._tail_logs:
                self.assertTrue(tail_log.stopped())

        def test_function_with_tensor(self):
            # [FUNCTION DOC] A worker returning a tensor delivers it as the return value with
            # shape 100 by 100, and the worker process is gone afterwards.
            for start_method in self._start_methods:
                pc = start_processes(
                    name="dummy_compute",
                    entrypoint=dummy_compute,
                    args={0: ()},
                    envs={0: {}},
                    logs_specs=DefaultLogsSpecs(log_dir=self.log_dir()),
                    start_method=start_method,
                )

                results = pc.wait()
                self.assert_pids_noexist(pc.pids())
                for return_value in results.return_values.values():
                    self.assertIsInstance(return_value, torch.Tensor)
                    self.assertEqual((100, 100), return_value.shape)

        def test_void_function(self):
            # [FUNCTION DOC] Workers whose function returns nothing yield a return value of
            # None for each rank.
            for start_method in self._start_methods:
                with self.subTest(start_method=start_method):
                    pc = start_processes(
                        name="echo",
                        entrypoint=echo0,
                        args={0: ("hello",), 1: ("world",)},
                        envs={0: {}, 1: {}},
                        logs_specs=DefaultLogsSpecs(log_dir=self.log_dir()),
                        start_method=start_method,
                    )

                    results = pc.wait(period=0.1)
                    self.assertEqual({0: None, 1: None}, results.return_values)

        @skip_but_pass_in_sandcastle_if(
            TEST_WITH_DEV_DBG_ASAN, "tests incompatible with asan"
        )
        def test_function_large_ret_val(self):
            # [FUNCTION DOC] Four workers each return a 200000-entry mapping (about 10 MB);
            # checks wait does not deadlock on the full pipe and every result has all entries.
            # python multiprocessing.queue module uses pipes and actually PipedQueues
            # This means that if a single object is greater than a pipe size
            # the writer process will block until reader process will start
            # reading the pipe.
            # This test makes a worker fn to return huge output, around ~10 MB

            size = 200000
            for start_method in self._start_methods:
                with self.subTest(start_method=start_method):
                    pc = start_processes(
                        logs_specs=DefaultLogsSpecs(log_dir=self.log_dir()),
                        name="echo",
                        entrypoint=echo_large,
                        args={0: (size,), 1: (size,), 2: (size,), 3: (size,)},
                        envs={0: {}, 1: {}, 2: {}, 3: {}},
                        start_method=start_method,
                    )

                    results = pc.wait(period=0.1)
                    for i in range(pc.nprocs):
                        self.assertEqual(size, len(results.return_values[i]))

        @unittest.skipIf(
            TEST_WITH_ROCM,
            "Skipped on ROCm due to hang in MultiprocessContext.wait after Kineto bump (PR #177101, 1fd9c49); investigating",
        )
        def test_function_raise(self):
            # [FUNCTION DOC] Rank 0 raises an exception and rank 1 succeeds. Checks exactly one
            # failure is reported with exit code 1, no signal, the right pid, an error file
            # under the run-id log directory ending in attempt_0/0/error.json whose recorded
            # timestamp matches the failure's, no return values, and stopped log tails.
            """
            run 2x copies of echo2, raise an exception on the first
            """
            RAISE = True

            for start_method in self._start_methods:
                with self.subTest(start_method=start_method):
                    log_dir = self.log_dir()
                    pc = start_processes(
                        name="echo",
                        entrypoint=echo2,
                        args={0: ("hello", RAISE), 1: ("world",)},
                        envs={
                            0: {"TORCHELASTIC_RUN_ID": "run_id"},
                            1: {"TORCHELASTIC_RUN_ID": "run_id"},
                        },
                        logs_specs=DefaultLogsSpecs(log_dir=log_dir),
                        start_method=start_method,
                    )

                    results = pc.wait(period=0.1)

                    self.assert_pids_noexist(pc.pids())
                    self.assertEqual(1, len(results.failures))
                    self.assertFalse(results.return_values)

                    failure = results.failures[0]
                    error_file = failure.error_file
                    error_file_data = failure.error_file_data

                    self.assertEqual(1, failure.exitcode)
                    self.assertEqual("<N/A>", failure.signal_name())
                    self.assertEqual(pc.pids()[0], failure.pid)
                    self.assertTrue(
                        error_file.startswith(os.path.join(log_dir, "run_id_"))
                    )
                    self.assertTrue(error_file.endswith("attempt_0/0/error.json"))
                    self.assertEqual(
                        int(error_file_data["message"]["extraInfo"]["timestamp"]),
                        int(failure.timestamp),
                    )
                    for tail_log in pc._tail_logs:
                        self.assertTrue(tail_log.stopped())

        def test_wait_for_all_child_procs_to_exit(self):
            # [FUNCTION DOC] With the context marked done and the process pool mocked so join
            # returns true, false, false, true in turn, one poll must call join four times,
            # i.e. it keeps joining until every child has exited.
            """
            Tests that MultiprocessingContext actually waits for
            the child process to exit (not just that the entrypoint fn has
            finished running).
            """

            mpc = MultiprocessContext(
                name="echo",
                entrypoint=echo0,
                args={},
                envs={},
                start_method="spawn",
                logs_specs=DefaultLogsSpecs(log_dir=self.log_dir()),
            )

            with (
                mock.patch.object(mpc, "_is_done", return_value=True),
                mock.patch.object(mpc, "_pc"),
                mock.patch.object(
                    mpc._pc, "join", side_effect=[True, False, False, True]
                ) as mock_join,
            ):
                mpc._poll()
                self.assertEqual(4, mock_join.call_count)

        def test_multiprocessing_context_poll_raises_exception(self):
            # [FUNCTION DOC] With a mocked pool whose join raises ProcessRaisedException for a
            # process that has exit code -1, poll (close mocked) returns one failure whose
            # message is "Signal 1 (SIGHUP) received by PID 123".
            mp_context = MultiprocessContext(
                name="test_mp",
                entrypoint=echo0,
                args={0: (0, 1)},
                envs={0: {}},
                logs_specs=DefaultLogsSpecs(
                    log_dir=self.log_dir(), redirects=Std.ALL, tee=Std.ALL
                ),
                start_method="spawn",
            )
            mp_context._pc = mock.Mock()
            # Using mock since we cannot just set exitcode on process
            mock_process = mock.Mock()
            mock_process.exitcode = -1
            mp_context._pc.processes = [mock_process]
            e = mp.ProcessRaisedException(msg="test msg", error_index=0, error_pid=123)
            mp_context._pc.join.side_effect = e
            with mock.patch.object(mp_context, "close"):
                run_result = mp_context._poll()
                self.assertEqual(1, len(run_result.failures))
                failure = run_result.failures[0]
                self.assertEqual(
                    "Signal 1 (SIGHUP) received by PID 123", failure.message
                )

    class StartProcessesAsBinaryTest(_StartProcessesTest):
        # [CLASS DOC] start_processes tests with script entrypoints run as subprocesses:
        # closing, exit codes, raised errors, missing entrypoint and rank validation.
        ########################################
        # start_processes as binary tests
        ########################################

        def test_subprocess_context_close(self):
            # [FUNCTION DOC] Starts the long-sleeping zombie_test script, closes the context
            # and checks its pid no longer exists.
            pc = start_processes(
                name="sleep",
                entrypoint=bin("zombie_test.py"),
                args={0: (1,)},
                envs={0: {}},
                logs_specs=DefaultLogsSpecs(log_dir=self.log_dir()),
            )

            pids = pc.pids()
            pc.close()
            self.assert_pids_noexist(pids)

        def test_binary_exit(self):
            # [FUNCTION DOC] Runs echo4 with rank 0 exiting 138 and rank 1 sleeping. Expects two
            # failures: rank 0 with exit 138, no signal, stderr captured, and rank 1 killed by
            # SIGTERM (exit -15) with the signal message, both with a "<NONE>" error message,
            # and stopped log tails.
            FAIL = 138
            pc = start_processes(
                name="echo",
                entrypoint=bin("echo4.py"),
                args={0: ("--exitcode", FAIL, "foo"), 1: ("--exitcode", 0, "bar")},
                envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                logs_specs=DefaultLogsSpecs(
                    log_dir=self.log_dir(),
                    redirects={0: Std.ALL},
                ),
            )

            results = pc.wait(period=0.1)
            self.assertTrue(results.is_failed())
            self.assertEqual(2, len(results.failures))

            failure = results.failures[0]
            self.assertEqual(138, failure.exitcode)
            self.assertEqual("<N/A>", failure.signal_name())
            self.assertEqual("<NONE>", failure.error_file_data["message"])
            self.assert_in_file([f"exit {FAIL} from 0"], results.stderrs[0])
            self.assert_in_file([], results.stdouts[0])
            self.assertFalse(results.stderrs[1])
            self.assertFalse(results.stdouts[1])
            for tail_log in pc._tail_logs:
                self.assertTrue(tail_log.stopped())

            failure = results.failures[1]
            self.assertEqual(-15, failure.exitcode)
            self.assertEqual("SIGTERM", failure.signal_name())
            self.assertEqual("<NONE>", failure.error_file_data["message"])
            # Assert that the failure message contains expected substrings
            self.assertIn("Signal 15 (SIGTERM) received by PID", failure.message)

        def test_binary_raises(self):
            # [FUNCTION DOC] Runs echo2 with rank 0 raising an exception; expects one failure
            # with exit code 1, no signal name, a "<NONE>" error message (no error file), and
            # no remaining processes.
            pc = start_processes(
                name="echo",
                entrypoint=bin("echo2.py"),
                args={0: ("--raises", "true", "foo"), 1: ("bar",)},
                envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                logs_specs=DefaultLogsSpecs(log_dir=self.log_dir()),
            )

            results = pc.wait(period=0.1)

            self.assert_pids_noexist(pc.pids())
            self.assertTrue(results.is_failed())
            self.assertEqual(1, len(results.failures))

            failure = results.failures[0]
            self.assertEqual(1, failure.exitcode)
            self.assertEqual("<NONE>", failure.error_file_data["message"])
            self.assertEqual("<N/A>", failure.signal_name())

        def test_binary_incorrect_entrypoint(self):
            # [FUNCTION DOC] A nonexistent script as entrypoint makes start_processes raise
            # FileNotFoundError.
            # KNOWN ISSUE: the args for rank 0 are a plain string, not a one-element tuple.
            with self.assertRaises(FileNotFoundError):
                start_processes(
                    name="echo",
                    entrypoint="does_not_exist.py",
                    args={0: ("foo"), 1: ("bar",)},
                    envs={0: {}, 1: {}},
                    logs_specs=DefaultLogsSpecs(log_dir=self.log_dir()),
                )

        def test_validate_full_rank(self):
            # [FUNCTION DOC] _validate_full_rank on an empty mapping with an expected size of
            # 10 must raise RuntimeError.
            with self.assertRaises(RuntimeError):
                _validate_full_rank({}, 10, "")


# tests incompatible with tsan or asan, the redirect functionality does not work on macos or windows
if not (TEST_WITH_DEV_DBG_ASAN or IS_WINDOWS or IS_MACOS):

    class StartProcessesListAsFuncTest(_StartProcessesTest):
        # [CLASS DOC] start_processes tests with function entrypoints and redirect settings
        # limited to those the OSS runs support.
        def test_function(self):
            # [FUNCTION DOC] For each start method and OSS redirect setting, runs echo1 on two
            # ranks; checks return values are "hello_<rank>", no processes remain, and the
            # stdout/stderr files are empty or contain the expected lines depending on the
            # redirect setting.
            for start_method, redirs in product(
                self._start_methods, redirects_oss_test()
            ):
                with self.subTest(start_method=start_method, redirs=redirs):
                    pc = start_processes(
                        name="echo",
                        entrypoint=echo1,
                        args={0: ("hello",), 1: ("hello",)},
                        envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                        logs_specs=DefaultLogsSpecs(
                            log_dir=self.log_dir(),
                            redirects=redirs,
                        ),
                        start_method=start_method,
                    )

                    results = pc.wait(period=0.1)
                    nprocs = pc.nprocs

                    self.assert_pids_noexist(pc.pids())
                    self.assertEqual(
                        {i: f"hello_{i}" for i in range(nprocs)}, results.return_values
                    )

                    for i in range(nprocs):
                        if redirs & Std.OUT != Std.OUT:
                            self.assertFalse(results.stdouts[i])
                        if redirs & Std.ERR != Std.ERR:
                            self.assertFalse(results.stderrs[i])
                        if redirs & Std.OUT == Std.OUT:
                            self.assert_in_file(
                                [f"hello stdout from {i}"], results.stdouts[i]
                            )
                        if redirs & Std.ERR == Std.ERR:
                            self.assert_in_file(
                                [f"hello stderr from {i}"], results.stderrs[i]
                            )

    class StartProcessesListAsBinaryTest(_StartProcessesTest):
        # [CLASS DOC] start_processes tests with script entrypoints and redirect/tee/filter
        # options, including log line prefixes.
        ########################################
        # start_processes as binary tests
        ########################################
        def test_binary(self):
            # [FUNCTION DOC] Runs the echo1 script on two ranks with line prefixes; checks no
            # failure, two (empty) return values, no remaining processes, and redirected
            # output files per the redirect setting.
            for redirs in redirects_oss_test():
                with self.subTest(redirs=redirs):
                    pc = start_processes(
                        name="echo",
                        entrypoint=bin("echo1.py"),
                        args={0: ("hello",), 1: ("hello",)},
                        envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                        logs_specs=DefaultLogsSpecs(
                            log_dir=self.log_dir(),
                            redirects=redirs,
                        ),
                        log_line_prefixes={0: "[rank0]:", 1: "[rank1]:"},
                    )

                    results = pc.wait(period=0.1)

                    self.assert_pids_noexist(pc.pids())
                    # currently binaries return {rank: None}
                    self.assertEqual(2, len(results.return_values))
                    self.assertFalse(results.is_failed())

                    nprocs = pc.nprocs
                    for i in range(nprocs):
                        if redirs & Std.OUT != Std.OUT:
                            self.assertFalse(results.stdouts[i])
                        if redirs & Std.ERR != Std.ERR:
                            self.assertFalse(results.stderrs[i])
                        if redirs & Std.OUT == Std.OUT:
                            self.assert_in_file(
                                [f"hello stdout from {i}"], results.stdouts[i]
                            )
                        if redirs & Std.ERR == Std.ERR:
                            self.assert_in_file(
                                [f"hello stderr from {i}"], results.stderrs[i]
                            )

        @unittest.skipIf(
            IS_LINUX or TEST_WITH_ROCM,
            "https://github.com/pytorch/pytorch/issues/163230",
        )
        def test_binary_redirect_and_tee(self):
            # [FUNCTION DOC] Runs echo1 on two ranks with rank 0 redirecting stderr and teeing
            # stdout, rank 1 teeing stderr. Checks success, the expected lines in the
            # stdout/stderr files, empty stdout for rank 1, and stopped log tails. Skipped on
            # Linux and ROCm (linked issue), so it effectively never runs on Linux.
            pc = start_processes(
                name="trainer",
                entrypoint=bin("echo1.py"),
                args={0: ("hello",), 1: ("world",)},
                envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                logs_specs=DefaultLogsSpecs(
                    log_dir=self.log_dir(),
                    redirects={0: Std.ERR, 1: Std.NONE},
                    tee={0: Std.OUT, 1: Std.ERR},
                ),
                log_line_prefixes={0: "[rank0]:", 1: "[rank1]:"},
                start_method="spawn",
            )

            result = pc.wait()

            self.assertFalse(result.is_failed())
            self.assert_in_file(["hello stdout from 0"], pc.stdouts[0])
            self.assert_in_file(["hello stderr from 0"], pc.stderrs[0])
            self.assert_in_file(["world stderr from 1"], pc.stderrs[1])
            self.assertFalse(pc.stdouts[1])
            for tail_log in pc._tail_logs:
                self.assertTrue(tail_log.stopped())

        def test_binary_duplicate_log_filters(self):
            # [FUNCTION DOC] Runs echo1 with duplicate stdout filter "helloA" and stderr
            # filters "worldA" and "B". Checks the filtered stdout file has only prefixed
            # helloA lines (not helloB) and the filtered stderr file has the prefixed worldA
            # and worldB lines, and the log tails stop.
            envs = {0: {"RANK": "0"}, 1: {"RANK": "1"}}
            logs_specs = DefaultLogsSpecs(
                log_dir=self.log_dir(),
                redirects={0: Std.ERR, 1: Std.NONE},
                tee={0: Std.OUT, 1: Std.ERR},
            )
            logs_dest = logs_specs.reify(envs)
            pc = start_processes(
                name="trainer",
                entrypoint=bin("echo1.py"),
                args={0: ("helloA,helloB",), 1: ("worldA,worldB",)},
                envs=envs,
                logs_specs=logs_specs,
                log_line_prefixes={0: "[rank0]:", 1: "[rank1]:"},
                duplicate_stdout_filters=["helloA"],
                duplicate_stderr_filters=["worldA", "B"],
                start_method="spawn",
            )

            result = pc.wait()

            self.assertFalse(result.is_failed())
            self.assert_in_file(
                ["[rank0]:helloA stdout from 0"], logs_dest.filtered_stdout
            )
            self.assert_not_in_file(
                ["[rank0]:helloB stdout from 0"], logs_dest.filtered_stdout
            )
            self.assert_in_file(
                ["[rank1]:worldA stderr from 1"], logs_dest.filtered_stderr
            )
            self.assert_in_file(
                ["[rank1]:worldB stderr from 1"], logs_dest.filtered_stderr
            )
            for tail_log in pc._tail_logs:
                self.assertTrue(tail_log.stopped())


# tests incompatible with tsan or asan, the redirect functionality does not work on macos or windows
if not (TEST_WITH_DEV_DBG_ASAN or IS_WINDOWS or IS_MACOS or IS_CI):

    class StartProcessesNotCIAsFuncTest(_StartProcessesTest):
        # [CLASS DOC] start_processes function-entrypoint tests excluded from CI: wrapper
        # behavior, redirect and tee, duplicate filters, all redirect settings, exit codes and
        # no-zombie checks.
        @skip_if_pytest
        def test_wrap_bad(self):
            # [FUNCTION DOC] Calls the worker wrapper directly for all four combinations of
            # stdout/stderr redirect file or none; checks the function result is put on the
            # queue, redirected files contain the expected lines, and the finished-reading
            # event is awaited once.
            none = ""
            stdout_log = os.path.join(self.test_dir, "stdout.log")
            stderr_log = os.path.join(self.test_dir, "stderr.log")
            redirs = [
                (none, none),
                (none, stderr_log),
                (stdout_log, none),
                (stdout_log, stderr_log),
            ]

            for stdout_redir, stderr_redir in redirs:
                queue = multiprocessing.SimpleQueue()
                worker_finished_event_mock = mock.Mock()
                _wrap(
                    local_rank=0,
                    fn=echo1,
                    args={0: ("hello",)},
                    envs={0: {"RANK": "0"}},
                    stdout_redirects={0: stdout_redir},
                    stderr_redirects={0: stderr_redir},
                    ret_vals={0: queue},
                    queue_finished_reading_event=worker_finished_event_mock,
                    numa_options=None,
                )
                self.assertEqual("hello_0", queue.get())
                if stdout_redir:
                    self.assert_in_file(["hello stdout from 0"], stdout_log)
                if stderr_redir:
                    self.assert_in_file(["hello stderr from 0"], stderr_log)
                worker_finished_event_mock.wait.assert_called_once()

        def test_function_redirect_and_tee(self):
            # [FUNCTION DOC] Runs echo1 on two ranks with rank 0 redirecting stderr and teeing
            # stdout, and rank 1 teeing stderr; checks success, expected lines in the output
            # files, empty stdout for rank 1 and stopped log tails.
            for start_method in self._start_methods:
                with self.subTest(start_method=start_method):
                    pc = start_processes(
                        name="trainer",
                        entrypoint=echo1,
                        args={0: ("hello",), 1: ("world",)},
                        envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                        logs_specs=DefaultLogsSpecs(
                            log_dir=self.log_dir(),
                            redirects={0: Std.ERR, 1: Std.NONE},
                            tee={0: Std.OUT, 1: Std.ERR},
                        ),
                        start_method="spawn",
                    )

                    result = pc.wait()

                    self.assertFalse(result.is_failed())
                    self.assert_in_file(["hello stdout from 0"], pc.stdouts[0])
                    self.assert_in_file(["hello stderr from 0"], pc.stderrs[0])
                    self.assert_in_file(["world stderr from 1"], pc.stderrs[1])
                    self.assertFalse(pc.stdouts[1])
                    for tail_log in pc._tail_logs:
                        self.assertTrue(tail_log.stopped())

        def test_function_duplicate_log_filters(self):
            # [FUNCTION DOC] Same as the binary filter test but with a function entrypoint and
            # the default trainer-name line prefixes: filtered files keep only helloA on stdout
            # and worldA and worldB on stderr.
            for start_method in self._start_methods:
                with self.subTest(start_method=start_method):
                    envs = {0: {"RANK": "0"}, 1: {"RANK": "1"}}
                    logs_specs = DefaultLogsSpecs(
                        log_dir=self.log_dir(),
                        redirects={0: Std.ERR, 1: Std.NONE},
                        tee={0: Std.OUT, 1: Std.ERR},
                    )
                    logs_dest = logs_specs.reify(envs)
                    pc = start_processes(
                        name="trainer",
                        entrypoint=echo1,
                        args={0: ("helloA,helloB",), 1: ("worldA,worldB",)},
                        envs=envs,
                        logs_specs=logs_specs,
                        duplicate_stdout_filters=["helloA"],
                        duplicate_stderr_filters=["worldA", "B"],
                        start_method="spawn",
                    )

                    result = pc.wait()

                    self.assertFalse(result.is_failed())
                    self.assert_in_file(
                        ["[trainer0]:helloA stdout from 0"], logs_dest.filtered_stdout
                    )
                    self.assert_not_in_file(
                        ["[trainer0]:helloB stdout from 0"], logs_dest.filtered_stdout
                    )
                    self.assert_in_file(
                        ["[trainer1]:worldA stderr from 1"], logs_dest.filtered_stderr
                    )
                    self.assert_in_file(
                        ["[trainer1]:worldB stderr from 1"], logs_dest.filtered_stderr
                    )
                    for tail_log in pc._tail_logs:
                        self.assertTrue(tail_log.stopped())

        def test_function(self):
            # [FUNCTION DOC] Like the OSS variant but iterating over all four redirect
            # settings (none, out, err, all) for each start method.
            for start_method, redirs in product(self._start_methods, redirects_all()):
                with self.subTest(start_method=start_method, redirs=redirs):
                    pc = start_processes(
                        name="echo",
                        entrypoint=echo1,
                        args={0: ("hello",), 1: ("hello",)},
                        envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                        start_method=start_method,
                        logs_specs=DefaultLogsSpecs(
                            log_dir=self.log_dir(),
                            redirects=redirs,
                        ),
                    )

                    results = pc.wait(period=0.1)
                    nprocs = pc.nprocs

                    self.assert_pids_noexist(pc.pids())
                    self.assertEqual(
                        {i: f"hello_{i}" for i in range(nprocs)}, results.return_values
                    )

                    for i in range(nprocs):
                        if redirs & Std.OUT != Std.OUT:
                            self.assertFalse(results.stdouts[i])
                        if redirs & Std.ERR != Std.ERR:
                            self.assertFalse(results.stderrs[i])
                        if redirs & Std.OUT == Std.OUT:
                            self.assert_in_file(
                                [f"hello stdout from {i}"], results.stdouts[i]
                            )
                        if redirs & Std.ERR == Std.ERR:
                            self.assert_in_file(
                                [f"hello stderr from {i}"], results.stderrs[i]
                            )

        def test_function_exit(self):
            # [FUNCTION DOC] Rank 0 exits with code 138 from python and rank 1 succeeds.
            # Expects one failure with that exit code, no signal, correct pid, no error file
            # ("<N/A>"), the generic traceback hint message, a timestamp not in the future,
            # stderr captured for rank 0, and stopped log tails.
            """
            run 2x copies of echo1 fail (exit) the first
            functions that exit from python do not generate an error file
            (even if they are decorated with @record)
            """

            FAIL = 138
            for start_method in self._start_methods:
                with self.subTest(start_method=start_method):
                    pc = start_processes(
                        name="echo",
                        entrypoint=echo1,
                        args={0: ("hello", FAIL), 1: ("hello",)},
                        envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                        logs_specs=DefaultLogsSpecs(
                            log_dir=self.log_dir(),
                            redirects={0: Std.ERR},
                        ),
                        start_method=start_method,
                    )

                    results = pc.wait(period=0.1)

                    self.assert_pids_noexist(pc.pids())
                    self.assertTrue(results.is_failed())
                    self.assertEqual(1, len(results.failures))
                    self.assertFalse(results.return_values)

                    failure = results.failures[0]
                    error_file = failure.error_file

                    self.assertEqual(FAIL, failure.exitcode)
                    self.assertEqual("<N/A>", failure.signal_name())
                    self.assertEqual(pc.pids()[0], failure.pid)
                    self.assertEqual("<N/A>", error_file)
                    self.assertEqual(
                        "To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html",
                        failure.message,
                    )
                    self.assertLessEqual(failure.timestamp, int(time.time()))

                    self.assert_in_file([f"exit {FAIL} from 0"], results.stderrs[0])
                    self.assertFalse(results.stdouts[0])
                    self.assertFalse(results.stderrs[1])
                    self.assertFalse(results.stdouts[1])
                    for tail_log in pc._tail_logs:
                        self.assertTrue(tail_log.stopped())

        def test_no_zombie_process_function(self):
            # [FUNCTION DOC] For each of four signal types, runs the zombie workflow with a
            # long-waiting function entrypoint to check workers die with their parent.
            signals = [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT]
            for s in signals:
                self._test_zombie_workflow(wait_fn, s)

    class StartProcessesNotCIAsBinaryTest(_StartProcessesTest):
        # [CLASS DOC] start_processes binary-entrypoint tests excluded from CI: segfault
        # signal reporting and no-zombie checks.
        def test_binary_signal(self):
            # [FUNCTION DOC] Runs echo3 with rank 0 segfaulting; expects one failure whose
            # signal name is SIGSEGV (or "<N/A>" under ASAN or TSAN, which exit with code 1)
            # and a "<NONE>" error message, and no remaining processes.
            pc = start_processes(
                name="echo",
                entrypoint=bin("echo3.py"),
                args={0: ("--segfault", "true", "foo"), 1: ("bar",)},
                envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                logs_specs=DefaultLogsSpecs(
                    log_dir=self.log_dir(),
                ),
            )

            results = pc.wait(period=0.1)

            self.assert_pids_noexist(pc.pids())
            self.assertTrue(results.is_failed())
            self.assertEqual(1, len(results.failures))

            failure = results.failures[0]
            self.assertNotEqual(signal.SIGSEGV, failure.exitcode)
            if TEST_WITH_ASAN or TEST_WITH_TSAN:
                # ASAN/TSAN exit code is 1.
                self.assertEqual("<N/A>", failure.signal_name())
            else:
                self.assertEqual("SIGSEGV", failure.signal_name())
            self.assertEqual("<NONE>", failure.error_file_data["message"])

        def test_no_zombie_process_binary(self):
            # [FUNCTION DOC] For each of four signal types, runs the zombie workflow with the
            # long-sleeping zombie_test script to check workers die with their parent.
            signals = [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT]
            for s in signals:
                self._test_zombie_workflow(bin("zombie_test.py"), s)

    class ForkServerTest(
        StartProcessesAsFuncTest,
        StartProcessesListAsFuncTest,
        StartProcessesNotCIAsFuncTest,
    ):
        # [CLASS DOC] Re-runs all function-entrypoint start_processes tests using the
        # forkserver start method with parallel start enabled through an environment variable
        # that is saved in setUp and restored in tearDown.
        def setUp(self):
            # [FUNCTION DOC] Switches the start methods to forkserver, remembers the current
            # parallel-start environment variable and sets it to "1".
            super().setUp()
            self._start_methods = ["forkserver"]
            self.orig_paralell_env_val = os.environ.get(mp.ENV_VAR_PARALLEL_START)
            os.environ[mp.ENV_VAR_PARALLEL_START] = "1"

        def tearDown(self):
            # [FUNCTION DOC] Restores the parallel-start environment variable to its saved
            # value, deleting it if it was unset, after the parent tearDown.
            super().tearDown()
            if self.orig_paralell_env_val is None:
                del os.environ[mp.ENV_VAR_PARALLEL_START]
            else:
                os.environ[mp.ENV_VAR_PARALLEL_START] = self.orig_paralell_env_val


class BoundedCloseTest(TestCase):
    # [CLASS DOC] Tests that closing a MultiprocessContext or SubprocessContext finishes in
    # bounded time and never makes an unbounded join/wait call on mocked unkillable children.
    """Verify that _close in MultiprocessContext / SubprocessContext returns
    within a bounded time when child processes refuse to die (simulating a
    Linux D-state worker, which SIGKILL cannot reap).
    """

    def _make_multiprocess_context(self) -> MultiprocessContext:
        # [FUNCTION DOC] HELPER: Builds an unstarted MultiprocessContext for a sleep
        # entrypoint with a log directory that is registered for cleanup after the test.
        log_dir = tempfile.mkdtemp(prefix="BoundedCloseTest")
        self.addCleanup(shutil.rmtree, log_dir, ignore_errors=True)
        return MultiprocessContext(
            name="test",
            entrypoint=time.sleep,
            args={0: (0,)},
            envs={0: {}},
            start_method="spawn",
            logs_specs=DefaultLogsSpecs(log_dir=log_dir),
        )

    def test_multiprocess_context_close_bounded_when_unkillable(self):
        # [FUNCTION DOC] Gives the context a mocked process that stays alive and records the
        # timeouts passed to join, then calls _close with SIGTERM and timeout 1 (kill
        # mocked). Asserts join was called and every call used a positive, non-None timeout.
        ctx = self._make_multiprocess_context()
        unkillable = mock.Mock()
        unkillable.pid = 99999
        unkillable.is_alive.return_value = True
        # Simulate a process that never exits: every bounded join honors
        # the timeout the caller passes. We assert below that _close still
        # returns promptly because it now passes a bounded timeout.
        join_calls: list[float | None] = []

        def _join(t: float | None = None) -> None:
            # [FUNCTION DOC] HELPER: Mock join that only records the timeout it was given.
            join_calls.append(t)

        unkillable.join.side_effect = _join
        ctx._pc = mock.Mock()
        ctx._pc.processes = [unkillable]
        with mock.patch("os.kill"):
            ctx._close(death_sig=signal.SIGTERM, timeout=1)
        # _close must pass a bounded timeout to every join() call;
        # without the fix the final join would be unbounded (timeout=None).
        self.assertTrue(join_calls, "expected proc.join() to be called")
        for t in join_calls:
            self.assertIsNotNone(t, "proc.join() must be called with a bounded timeout")
            if t is None:  # for type narrowing
                raise AssertionError("expected t to be not None")
            self.assertGreater(t, 0)

    def test_subprocess_context_close_bounded_when_unkillable(self):
        # [FUNCTION DOC] Builds a SubprocessContext with a mocked handler whose process never
        # exits and whose bounded waits raise TimeoutExpired, then calls _close with SIGTERM
        # and timeout 1. Asserts wait was called and every call used a positive, non-None
        # timeout.
        log_dir = tempfile.mkdtemp(prefix="BoundedCloseTest")
        self.addCleanup(shutil.rmtree, log_dir, ignore_errors=True)
        from torch.distributed.elastic.multiprocessing.api import SubprocessContext

        ctx = SubprocessContext(
            name="test",
            entrypoint="/bin/true",
            args={0: ()},
            envs={0: {}},
            logs_specs=DefaultLogsSpecs(log_dir=log_dir),
        )
        handler = mock.Mock()
        handler.proc = mock.Mock()
        handler.proc.pid = 99999
        handler.proc.poll.return_value = None  # always "alive"

        import subprocess as _subprocess

        # Every bounded wait() raises TimeoutExpired immediately; the
        # unbounded form (timeout=None) is what we want to make sure
        # _close never invokes.
        wait_calls: list[float | None] = []

        def _wait(t: float | None = None) -> None:
            # [FUNCTION DOC] HELPER: Mock wait that records the timeout; with a timeout it
            # raises TimeoutExpired, with none it returns nothing.
            wait_calls.append(t)
            if t is None:
                return None
            raise _subprocess.TimeoutExpired(cmd="test", timeout=t)

        handler.proc.wait.side_effect = _wait
        ctx.subprocess_handlers = {0: handler}
        ctx._close(death_sig=signal.SIGTERM, timeout=1)
        # _close must pass a bounded timeout to every wait() call;
        # without the fix the final wait would be unbounded (timeout=None).
        self.assertTrue(wait_calls, "expected proc.wait() to be called")
        for t in wait_calls:
            self.assertIsNotNone(t, "proc.wait() must be called with a bounded timeout")
            if t is None:  # for type narrowing
                raise AssertionError("expected t to be not None")
            self.assertGreater(t, 0)


if __name__ == "__main__":
    run_tests()
