# Owner(s): ["module: multiprocessing"]

# [MODULE DOC] Tests for torch.multiprocessing process-spawning helpers (start_processes and
# spawn): successful runs, exception propagation, termination by signal or exit code, nested
# children dying with their parent, and picklability of the process error classes. Defines
# module-level worker functions (they must be importable by spawned children), a shared mixin
# test class run under spawn, fork and forkserver, and a perf test class. Importing the module
# has no side effect beyond the class body of Expensive, which sleeps while it is defined.
import os
import pickle
import random
import signal
import sys
import time
import unittest

import torch.multiprocessing as mp

from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    TestCase,
    parametrize,
    instantiate_parametrized_tests
)

def _test_success_func(i):
    # [FUNCTION DOC] HELPER: Worker that does nothing and returns; used for successful runs.
    pass


def _test_success_single_arg_func(i, arg):
    # [FUNCTION DOC] HELPER: Worker taking its process index and an optional queue; if a queue
    # is supplied (truthy) it puts its own index on it, otherwise does nothing.
    if arg:
        arg.put(i)


def _test_exception_single_func(i, arg):
    # [FUNCTION DOC] HELPER: Worker that raises a ValueError mentioning its index when the index
    # equals the given target; all other workers sleep one second and return normally.
    if i == arg:
        raise ValueError(f"legitimate exception from process {i:d}")
    time.sleep(1.0)


def _test_exception_all_func(i):
    # [FUNCTION DOC] HELPER: Worker that sleeps a random time under a tenth of a second, then
    # raises a ValueError mentioning its own index; every process fails.
    time.sleep(random.random() / 10)
    raise ValueError(f"legitimate exception from process {i:d}")


def _test_terminate_signal_func(i):
    # [FUNCTION DOC] HELPER: Worker where process zero kills itself with an abort signal; the
    # others (and zero, if it survived) sleep one second.
    if i == 0:
        os.kill(os.getpid(), signal.SIGABRT)
    time.sleep(1.0)


def _test_terminate_exit_func(i, arg):
    # [FUNCTION DOC] HELPER: Worker where process zero exits via system exit with the given code;
    # all others sleep four seconds, long enough to be terminated by the parent.
    if i == 0:
        sys.exit(arg)
    time.sleep(4.0)


def _test_success_first_then_exception_func(i, arg):
    # [FUNCTION DOC] HELPER: Worker where process zero returns immediately and every other
    # process sleeps briefly then raises a ValueError; the extra argument is unused.
    if i == 0:
        return
    time.sleep(0.1)
    raise ValueError("legitimate exception")


def _test_nested_child_body(i, ready_queue, nested_child_sleep):
    # [FUNCTION DOC] HELPER: Nested child worker: signals readiness by putting a None marker on
    # the ready queue, then sleeps for the given number of seconds.
    ready_queue.put(None)
    time.sleep(nested_child_sleep)


def _test_infinite_task(i):
    # [FUNCTION DOC] HELPER: Worker that sleeps forever in a loop; only external signals stop it.
    while True:
        time.sleep(1)


def _test_process_exit(idx):
    # [FUNCTION DOC] HELPER: Worker that exits immediately through system exit with code 12.
    sys.exit(12)


def _test_nested(i, pids_queue, nested_child_sleep, start_method):
    # [FUNCTION DOC] HELPER: Worker that starts two non-daemon child processes (non-joining) in
    # the same start method, publishes their pids on the given queue, waits until both report
    # ready, then sends itself a termination signal. The children are expected to die with it
    # (parent death signal registration, per the comments in the code).
    context = mp.get_context(start_method)
    nested_child_ready_queue = context.Queue()
    nprocs = 2
    mp_context = mp.start_processes(
        fn=_test_nested_child_body,
        args=(nested_child_ready_queue, nested_child_sleep),
        nprocs=nprocs,
        join=False,
        daemon=False,
        start_method=start_method,
    )
    pids_queue.put(mp_context.pids())

    # Wait for both children to have started, to ensure that they
    # have called prctl(2) to register a parent death signal.
    for _ in range(nprocs):
        nested_child_ready_queue.get()

    # Kill self. This should take down the child processes as well.
    os.kill(os.getpid(), signal.SIGTERM)

@instantiate_parametrized_tests
class _TestMultiProcessing:
    # [CLASS DOC] Mixin holding the shared multiprocessing tests, decorated so parametrized tests
    # are expanded. It is not itself a test case: concrete subclasses combine it with TestCase
    # and set the start_method class attribute (None here) to spawn, fork or forkserver.
    start_method = None

    def test_success(self):
        # [FUNCTION DOC] Starts two processes running the no-op worker with the configured start
        # method and expects start_processes to return without raising.
        mp.start_processes(_test_success_func, nprocs=2, start_method=self.start_method)

    def test_success_non_blocking(self):
        # [FUNCTION DOC] Starts two no-op processes without joining, then calls join with no
        # timeout three times; asserts the final call reports completion as true.
        mp_context = mp.start_processes(_test_success_func, nprocs=2, join=False, start_method=self.start_method)

        # After all processes (nproc=2) have joined it must return True
        mp_context.join(timeout=None)
        mp_context.join(timeout=None)
        self.assertTrue(mp_context.join(timeout=None))

    def test_first_argument_index(self):
        # [FUNCTION DOC] Starts two processes that each put their index on a shared simple queue
        # and asserts the collected indices, sorted, are zero and one (workers get index first).
        context = mp.get_context(self.start_method)
        queue = context.SimpleQueue()
        mp.start_processes(_test_success_single_arg_func, args=(queue,), nprocs=2, start_method=self.start_method)
        self.assertEqual([0, 1], sorted([queue.get(), queue.get()]))

    def test_exception_single(self):
        # [FUNCTION DOC] For each process index in turn, makes only that process raise a
        # ValueError and asserts start_processes raises an exception whose message ends with the
        # ValueError text naming that process.
        nprocs = 2
        for i in range(nprocs):
            with self.assertRaisesRegex(
                Exception,
                f"\nValueError: legitimate exception from process {i:d}$",
            ):
                mp.start_processes(_test_exception_single_func, args=(i,), nprocs=nprocs, start_method=self.start_method)

    def test_exception_all(self):
        # [FUNCTION DOC] Starts two processes that all raise; asserts the raised exception message
        # ends with a ValueError naming process zero or one (whichever failed first).
        with self.assertRaisesRegex(
            Exception,
            "\nValueError: legitimate exception from process (0|1)$",
        ):
            mp.start_processes(_test_exception_all_func, nprocs=2, start_method=self.start_method)

    def test_terminate_signal(self):
        # [FUNCTION DOC] Asserts that a process killed by an abort signal produces an error saying
        # process zero terminated with that signal name (either alias); on Windows the expected
        # message is instead an exit code of 22, since signals do not map to negative codes.
        # SIGABRT is aliased with SIGIOT
        message = "process 0 terminated with signal (SIGABRT|SIGIOT)"

        # Termination through with signal is expressed as a negative exit code
        # in multiprocessing, so we know it was a signal that caused the exit.
        # This doesn't appear to exist on Windows, where the exit code is always
        # positive, and therefore results in a different exception message.
        # Exit code 22 means "ERROR_BAD_COMMAND".
        if IS_WINDOWS:
            message = "process 0 terminated with exit code 22"

        with self.assertRaisesRegex(Exception, message):
            mp.start_processes(_test_terminate_signal_func, nprocs=2, start_method=self.start_method)

    @parametrize("grace_period", [None, 20])
    def test_terminate_exit(self, grace_period):
        # [FUNCTION DOC] Parametrized by grace period (none or 20). Process zero exits with code
        # 123 while process one sleeps. Joins repeatedly, expecting an exception reporting the exit
        # code and capturing warning logs. With no grace period, the first log must say process
        # one is terminated via signal; otherwise no warnings are logged. Finally asserts no
        # started process is still alive.
        exitcode = 123
        ctx = mp.start_processes(_test_terminate_exit_func, args=(exitcode,), nprocs=2, start_method=self.start_method, join=False)
        pid1 = ctx.processes[1].pid
        with self.assertRaisesRegex(
            Exception,
            f"process 0 terminated with exit code {exitcode:d}",
        ), self.assertLogs(level='WARNING') as logs:
            while not ctx.join(grace_period=grace_period):
                pass
        if grace_period is None:
            # pid1 is killed by signal.
            expected_log = f"Terminating process {pid1:d} via signal"
            self.assertIn(expected_log, logs.records[0].getMessage())
        else:
            # pid1 exits on its own.
            self.assertFalse(logs.records)

        # Check that no processes are left.
        for p in ctx.processes:
            self.assertFalse(p.is_alive())

    def test_success_first_then_exception(self):
        # [FUNCTION DOC] Asserts that when process zero succeeds early and others later raise, the
        # overall call still raises an exception mentioning the ValueError text.
        exitcode = 123
        with self.assertRaisesRegex(
            Exception,
            "ValueError: legitimate exception",
        ):
            mp.start_processes(_test_success_first_then_exception_func, args=(exitcode,), nprocs=2, start_method=self.start_method)

    @unittest.skipIf(
        sys.platform != "linux",
        "Only runs on Linux; requires prctl(2)",
    )
    def _test_nested(self):
        # [FUNCTION DOC] HELPER: Disabled (name has no test prefix, so it is not collected) and
        # Linux-only check of nested children. Starts one non-joining process running the nested
        # worker, reads the child pids from a queue, and polls until each pid no longer exists,
        # failing if they are still alive after half the nested child sleep time.
        # KNOWN ISSUE: the leading underscore in the method name means the test never runs.
        context = mp.get_context(self.start_method)
        pids_queue = context.Queue()
        nested_child_sleep = 20.0
        mp_context = mp.start_processes(  # noqa: F841
            fn=_test_nested,
            args=(pids_queue, nested_child_sleep, self.start_method),
            nprocs=1,
            join=False,
            daemon=False,
            start_method=self.start_method,
        )

        # Wait for nested children to terminate in time
        pids = pids_queue.get()
        start = time.time()
        while len(pids) > 0:
            for pid in pids:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pids.remove(pid)  # noqa: B909
                    break

            # This assert fails if any nested child process is still
            # alive after (nested_child_sleep / 2) seconds. By
            # extension, this test times out with an assertion error
            # after (nested_child_sleep / 2) seconds.
            self.assertLess(time.time() - start, nested_child_sleep / 2)
            time.sleep(0.1)

class SpawnTest(TestCase, _TestMultiProcessing):
    # [CLASS DOC] Runs the shared multiprocessing tests with the spawn start method and adds
    # tests using the public spawn function directly.
    start_method = 'spawn'

    def test_exception_raises(self):
        # [FUNCTION DOC] Asserts that spawn raises ProcessRaisedException when the single child
        # process raises an error.
        with self.assertRaises(mp.ProcessRaisedException):
            mp.spawn(_test_success_first_then_exception_func, args=(), nprocs=1)

    def test_signal_raises(self):
        # [FUNCTION DOC] Starts one never-ending child without joining, sends termination to each
        # pid from the context, and asserts joining raises ProcessExitedException.
        context = mp.spawn(_test_infinite_task, args=(), nprocs=1, join=False)
        for pid in context.pids():
            os.kill(pid, signal.SIGTERM)
        with self.assertRaises(mp.ProcessExitedException):
            context.join()

    def _test_process_exited(self):
        # [FUNCTION DOC] HELPER: Disabled (not collected due to its name). Spawns a child that
        # exits with code 12 and expects ProcessExitedException.
        # KNOWN ISSUE: the exit code assertion is inside the with block after the raising call,
        # so it never runs, and it reads an attribute from the context manager object rather
        # than its captured exception.
        with self.assertRaises(mp.ProcessExitedException) as e:
            mp.spawn(_test_process_exit, args=(), nprocs=1)
            self.assertEqual(12, e.exit_code)


@unittest.skipIf(
    IS_WINDOWS,
    "Fork is only available on Unix",
)
class ForkTest(TestCase, _TestMultiProcessing):
    # [CLASS DOC] Runs the shared multiprocessing tests with the fork start method; skipped on
    # Windows where fork is unavailable.
    start_method = 'fork'


@unittest.skipIf(
    IS_WINDOWS,
    "Fork is only available on Unix",
)
class ParallelForkServerShouldWorkTest(TestCase, _TestMultiProcessing):
    # [CLASS DOC] Runs the shared tests with the parallel start environment variable set to one
    # for the duration of each test, using the start_method inherited from the mixin (none). The
    # original value of the variable is saved in a class attribute for restoration. Skipped on
    # Windows.
    orig_paralell_env_val = None

    def setUp(self):
        # [FUNCTION DOC] Saves the current value of the parallel start environment variable, then
        # sets it to one in the process environment.
        super().setUp()
        self.orig_paralell_env_val = os.environ.get(mp.ENV_VAR_PARALLEL_START)
        os.environ[mp.ENV_VAR_PARALLEL_START] = "1"

    def tearDown(self):
        # [FUNCTION DOC] Runs base teardown, then restores the parallel start environment variable
        # to its saved value, deleting it if it was originally unset.
        super().tearDown()
        if self.orig_paralell_env_val is None:
            del os.environ[mp.ENV_VAR_PARALLEL_START]
        else:
            os.environ[mp.ENV_VAR_PARALLEL_START] = self.orig_paralell_env_val


@unittest.skipIf(
    IS_WINDOWS,
    "Fork is only available on Unix",
)
class ParallelForkServerPerfTest(TestCase):
    # [CLASS DOC] Performance test class for parallel forkserver startup. Skipped on Windows.

    @unittest.skipIf(
        sys.version_info >= (3, 13, 8),
        "Python 3.13.8+ changed forkserver module caching behavior",
        # https://docs.python.org/3.13/whatsnew/changelog.html
        # gh-126631
    )
    def test_forkserver_perf(self):
        # [FUNCTION DOC] Measures wall time to start four processes with the forkserver method
        # running a bound method of the slow-to-construct helper object, first with the parallel
        # start environment variable at zero (elapsed must be at least processes times the sleep
        # time) then at one (elapsed must be less than that). Restores the variable afterwards,
        # but not if an assertion fails. Skipped on Python 3.13.8 and later.

        start_method = 'forkserver'
        expensive = Expensive()
        nprocs = 4
        orig_paralell_env_val = os.environ.get(mp.ENV_VAR_PARALLEL_START)

        # test the non parallel case
        os.environ[mp.ENV_VAR_PARALLEL_START] = "0"
        start = time.perf_counter()
        mp.start_processes(expensive.my_call, nprocs=nprocs, start_method=start_method)
        elapsed = time.perf_counter() - start
        # the elapsed time should be at least {nprocs}x the sleep time
        self.assertGreaterEqual(elapsed, Expensive.SLEEP_SECS * nprocs)

        # test the parallel case
        os.environ[mp.ENV_VAR_PARALLEL_START] = "1"
        start = time.perf_counter()
        mp.start_processes(expensive.my_call, nprocs=nprocs, start_method=start_method)
        elapsed = time.perf_counter() - start
        # the elapsed time should be less than {nprocs}x the sleep time
        self.assertLess(elapsed, Expensive.SLEEP_SECS * nprocs)

        if orig_paralell_env_val is None:
            del os.environ[mp.ENV_VAR_PARALLEL_START]
        else:
            os.environ[mp.ENV_VAR_PARALLEL_START] = orig_paralell_env_val


class Expensive:
    # [CLASS DOC] HELPER: Helper whose class body sleeps for the class constant of five seconds
    # when the module is imported, simulating costly import-time startup overhead. Instances hold
    # a large string to make them expensive to pickle.
    SLEEP_SECS = 5
    # Simulate startup overhead such as large imports
    time.sleep(SLEEP_SECS)

    def __init__(self):
        # [FUNCTION DOC] HELPER: Allocates a one-million-character string attribute.
        self.config: str = "*" * 1000000

    def my_call(self, *args):
        # [FUNCTION DOC] HELPER: Does nothing; accepts any positional arguments (including the
        # process index) and is used as the worker target.
        pass


class ErrorTest(TestCase):
    # [CLASS DOC] Test case for the process error classes.
    def test_errors_pickleable(self):
        # [FUNCTION DOC] Builds a ProcessRaisedException and a ProcessExitedException and asserts
        # each survives a round trip through pickle serialization without raising.
        for error in (
            mp.ProcessRaisedException("Oh no!", 1, 1),
            mp.ProcessExitedException("Oh no!", 1, 1, 1),
        ):
            pickle.loads(pickle.dumps(error))


if __name__ == '__main__':
    run_tests()
