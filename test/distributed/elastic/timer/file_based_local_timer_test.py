# [MODULE DOC] Tests for the file-based local timer in torch.distributed.elastic.timer:
# FileTimerClient sends acquire/release requests through a named pipe file and FileTimerServer
# expires scopes by killing the owning worker process. The whole test body is defined only on
# platforms that are not Windows, macOS or ARM64. Importing has no side effects beyond defining
# classes; running spawns processes and creates pipe files under the temp directory.
# Owner(s): ["oncall: r2p"]

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import multiprocessing as mp
import os
import signal
import time
import unittest
import unittest.mock as mock
import uuid

import torch.distributed.elastic.timer as timer
from torch.testing._internal.common_utils import (
    IS_ARM64,
    IS_MACOS,
    IS_WINDOWS,
    run_tests,
    TEST_WITH_TSAN,
    TestCase,
)


# timer is not supported on these platforms
if not (IS_WINDOWS or IS_MACOS or IS_ARM64):
    # func2 should time out
    def func2(n, file_path):
        # [FUNCTION DOC] HELPER: Module-level worker target. If a file path is given, configures a
        # FileTimerClient for it as the process-wide client. Then, while n is positive, enters a
        # 0.1 second timer scope, recurses with n decreased by one (and no path), and sleeps 0.2
        # seconds, so the innermost surviving timer expires and the process gets killed.
        if file_path is not None:
            timer.configure(timer.FileTimerClient(file_path))
        if n > 0:
            with timer.expires(after=0.1):
                func2(n - 1, None)
                time.sleep(0.2)

    class FileTimerTest(TestCase):
        # [CLASS DOC] End-to-end tests of timer.expires with a FileTimerClient talking to a real
        # FileTimerServer. setUp starts a server on a unique pipe path under the temp directory
        # with a 0.01 second watchdog interval; tearDown stops it. Several tests spawn child
        # processes and check exit codes or the server's request count and pending timer table.
        def setUp(self):
            # [FUNCTION DOC] Creates a unique pipe file path from the pid and a uuid, builds a
            # FileTimerServer on it with name test and max interval 0.01 seconds, and starts it.
            # Stores both on self.
            super().setUp()
            self.max_interval = 0.01
            self.file_path = f"/tmp/test_file_path_{os.getpid()}_{uuid.uuid4()}"
            self.server = timer.FileTimerServer(
                self.file_path, "test", self.max_interval
            )
            self.server.start()

        def tearDown(self):
            # [FUNCTION DOC] Stops the server started in setUp after the base class teardown.
            super().tearDown()
            self.server.stop()

        def test_exception_propagation(self):
            # [FUNCTION DOC] Checks that an exception raised inside a timer scope propagates to the
            # caller as the same RuntimeError.
            with self.assertRaises(RuntimeError, msg="foobar"):
                with timer.expires(after=1):
                    raise RuntimeError("foobar")

        def test_no_client(self):
            # [FUNCTION DOC] Configures no client (None) and checks that entering a timer scope
            # raises RuntimeError.
            # no timer client configured; exception expected
            timer.configure(None)
            with self.assertRaises(RuntimeError):
                with timer.expires(after=1):
                    pass

        def test_client_interaction(self):
            # [FUNCTION DOC] Passes an explicit client to the scope without configuring a global
            # one; wraps acquire and release with spies and checks each is called once, with the
            # scope name (and any expiration for acquire).
            # no timer client configured but one passed in explicitly
            # no exception expected
            timer_client = timer.FileTimerClient(self.file_path)
            timer_client.acquire = mock.MagicMock(wraps=timer_client.acquire)
            timer_client.release = mock.MagicMock(wraps=timer_client.release)
            with timer.expires(after=1, scope="test", client=timer_client):
                pass

            timer_client.acquire.assert_called_once_with("test", mock.ANY)
            timer_client.release.assert_called_once_with("test")

        def test_happy_path(self):
            # [FUNCTION DOC] Configures a FileTimerClient and runs a short sleep well inside a 0.5
            # second scope; passes if nothing is raised or killed.
            timer.configure(timer.FileTimerClient(self.file_path))
            with timer.expires(after=0.5):
                time.sleep(0.1)

        def test_get_timer_recursive(self):
            # [FUNCTION DOC] Checks that nested scopes with the default scope name re-acquire one
            # timer so only the last takes effect. Recursion that finishes in time must not
            # trigger; a child process running func2 must be killed by SIGKILL (exit code is
            # minus SIGKILL).
            """
            If a function acquires a countdown timer with default scope,
            then recursive calls to the function should re-acquire the
            timer rather than creating a new one. That is only the last
            recursive call's timer will take effect.
            """
            timer.configure(timer.FileTimerClient(self.file_path))

            # func should not time out
            def func(n):
                # [FUNCTION DOC] HELPER: Recursive worker that nests a 0.1 second timer scope per
                # level and sleeps 0.05 seconds after each inner call; sized so it never expires.
                if n > 0:
                    with timer.expires(after=0.1):
                        func(n - 1)
                        time.sleep(0.05)

            func(4)

            p = mp.Process(target=func2, args=(2, self.file_path))
            p.start()
            p.join()
            self.assertEqual(-signal.SIGKILL, p.exitcode)

        def test_multiple_clients_interaction(self):
            # [FUNCTION DOC] Starts ten processes that each nest ten long (100 second) timer
            # scopes with their own client. After joining, runs the server once and checks the
            # server handled exactly two requests (acquire and release) per scope, i.e. 2 *
            # clients * requests.
            # func should not time out
            def func(n, file_path):
                # [FUNCTION DOC] HELPER: Child-process target. Configures a FileTimerClient when a
                # path is given, and nests n 100 second scopes recursively with a short sleep each;
                # never expires.
                if file_path is not None:
                    timer.configure(timer.FileTimerClient(file_path))
                if n > 0:
                    with timer.expires(after=100):
                        func(n - 1, None)
                        time.sleep(0.01)

            num_clients = 10
            num_requests_per_client = 10
            processes = []
            for _ in range(num_clients):
                p = mp.Process(
                    target=func, args=(num_requests_per_client, self.file_path)
                )
                processes.append(p)
                p.start()
            for p in processes:
                p.join()

            self.server.run_once()  # Allows the server to process all requests
            self.assertEqual(
                2 * num_clients * num_requests_per_client, self.server._request_count
            )

        @mock.patch("torch.distributed.elastic.timer.FileTimerServer._reap_worker")
        def test_exit_before_release(self, mock_reap):
            # [FUNCTION DOC] Patches the server's worker-reaping method. A child acquires a timer
            # expiring in 2 seconds and exits after 1 second without releasing. After waiting, the
            # server runs once and must not reap (process no longer exists) and must have cleared
            # all timers.
            def func1(file_path):
                # [FUNCTION DOC] HELPER: Child-process target. Creates and configures a
                # FileTimerClient, acquires scope test_scope expiring two seconds from now, and
                # sleeps one second without releasing.
                client = timer.FileTimerClient(file_path)
                timer.configure(client)
                expire = time.time() + 2
                client.acquire("test_scope", expire)
                time.sleep(1)

            p = mp.Process(target=func1, args=(self.file_path,))
            p.start()
            p.join()

            time.sleep(2)
            self.server.run_once()  # Allows the server to process all requests
            mock_reap.assert_not_called()
            self.assertEqual(0, len(self.server._timers))

        @mock.patch("torch.distributed.elastic.timer.FileTimerServer._reap_worker")
        @mock.patch(
            "torch.distributed.elastic.timer.FileTimerServer.is_process_running"
        )
        def test_exit_before_release_reap(self, mock_pid_exists, mock_reap):
            # [FUNCTION DOC] Same as the exit-before-release scenario but with the process-running
            # check patched to report true. After the server runs once, the reap method must have
            # been called and the timer table must be empty.
            def func1(file_path):
                # [FUNCTION DOC] HELPER: Child-process target. Creates and configures a
                # FileTimerClient, acquires scope test_scope expiring two seconds from now, and
                # sleeps one second without releasing.
                client = timer.FileTimerClient(file_path)
                timer.configure(client)
                expire = time.time() + 2
                client.acquire("test_scope", expire)
                time.sleep(1)

            mock_pid_exists.return_value = True
            p = mp.Process(target=func1, args=(self.file_path,))
            p.start()
            p.join()

            time.sleep(2)
            self.server.run_once()  # Allows the server to process all requests
            mock_reap.assert_called()
            self.assertEqual(0, len(self.server._timers))

        @staticmethod
        def _run(file_path, timeout, duration):
            # [FUNCTION DOC] HELPER: Static child-process target. Configures a FileTimerClient for
            # the given path and sleeps for the given duration inside a scope of the given timeout.
            client = timer.FileTimerClient(file_path)
            timer.configure(client)
            with timer.expires(after=timeout):
                time.sleep(duration)

        @unittest.skipIf(TEST_WITH_TSAN, "test is tsan incompatible")
        def test_timer(self):
            # [FUNCTION DOC] Runs a child whose sleep (1 second) exceeds its timer (0.1 second) and
            # checks it exits killed by SIGKILL. Skipped under thread sanitizer.
            timeout = 0.1
            duration = 1
            p = mp.Process(target=self._run, args=(self.file_path, timeout, duration))
            p.start()
            p.join()
            self.assertEqual(-signal.SIGKILL, p.exitcode)

    def _request_on_interval(file_path, n, interval, sem):
        # [FUNCTION DOC] HELPER: Intended to send n acquire requests, one per interval seconds, via
        # a FileTimerClient after releasing a semaphore.
        # KNOWN ISSUE: Unused in this file, and its docstring mentions an mp queue although it
        # takes a file path.
        """
        enqueues ``n`` timer requests into ``mp_queue`` one element per
        interval seconds. Releases the given semaphore once before going to work.
        """
        client = timer.FileTimerClient(file_path)
        sem.release()
        for _ in range(n):
            client.acquire("test_scope", 0)
            time.sleep(interval)

    class FileTimerClientTest(TestCase):
        # [CLASS DOC] Tests for FileTimerClient failure behavior when no server is listening.
        def test_send_request_without_server(self):
            # [FUNCTION DOC] Configures a client for a nonexistent pipe file and checks that
            # entering a timer scope raises BrokenPipeError.
            client = timer.FileTimerClient("test_file")
            timer.configure(client)
            with self.assertRaises(BrokenPipeError):
                with timer.expires(after=0.1):
                    time.sleep(0.1)

    class FileTimerServerTest(TestCase):
        # [CLASS DOC] Unit tests for FileTimerServer's watchdog handling of acquire, release and
        # expired requests, driving it by writing FileTimerRequest objects through a client.
        # os.kill and some server hooks are mocked so no real process is signalled. Uses fake
        # negative pids.
        def setUp(self):
            # [FUNCTION DOC] Builds (but does not start) a FileTimerServer on a unique pipe path
            # with a 0.01 second interval and stores the path and server.
            super().setUp()
            self.file_path = f"/tmp/test_file_path_{os.getpid()}_{uuid.uuid4()}"
            self.max_interval = 0.01
            self.server = timer.FileTimerServer(
                self.file_path, "test", self.max_interval
            )

        def tearDown(self):
            # [FUNCTION DOC] Stops the server after base class teardown.
            super().tearDown()
            self.server.stop()

        def test_watchdog_call_count(self):
            # [FUNCTION DOC] Spies on the server's watchdog, starts it, sends one valid timer,
            # waits 0.1 seconds and stops. Checks the watchdog ran wait divided by interval, give
            # or take one, times.
            """
            checks that the watchdog function ran wait/interval +- 1 times
            """
            self.server._run_watchdog = mock.MagicMock(wraps=self.server._run_watchdog)
            self.server.start()

            test_pid = -3
            client = timer.FileTimerClient(self.file_path)
            client._send_request(self._valid_timer(pid=test_pid, scope="test0"))

            wait = 0.1
            time.sleep(wait)
            self.server.stop()
            watchdog_call_count = self.server._run_watchdog.call_count
            self.assertGreaterEqual(
                watchdog_call_count, int(wait / self.max_interval) - 1
            )
            self.assertLessEqual(watchdog_call_count, int(wait / self.max_interval) + 1)

        def test_watchdog_empty_queue(self):
            # [FUNCTION DOC] Starts the server with nothing written to the pipe and checks that it
            # starts without error.
            """
            checks that the watchdog can run on an empty pipe
            """
            self.server.start()

        def _expired_timer(self, pid, scope):
            # [FUNCTION DOC] HELPER: Builds a FileTimerRequest for the given pid and scope that
            # expired 60 seconds ago, with SIGKILL as the signal.
            expired = time.time() - 60
            return timer.FileTimerRequest(
                worker_pid=pid,
                scope_id=scope,
                expiration_time=expired,
                signal=signal.SIGKILL,
            )

        def _valid_timer(self, pid, scope):
            # [FUNCTION DOC] HELPER: Builds a FileTimerRequest for the given pid and scope that
            # expires 60 seconds from now, with SIGKILL as the signal.
            valid = time.time() + 60
            return timer.FileTimerRequest(
                worker_pid=pid,
                scope_id=scope,
                expiration_time=valid,
                signal=signal.SIGKILL,
            )

        def _release_timer(self, pid, scope):
            # [FUNCTION DOC] HELPER: Builds a FileTimerRequest for the given pid and scope with
            # expiration time -1, which the server treats as a release.
            return timer.FileTimerRequest(
                worker_pid=pid, scope_id=scope, expiration_time=-1
            )

        @mock.patch("os.kill")
        @mock.patch(
            "torch.distributed.elastic.timer.file_based_local_timer.log_debug_info_for_expired_timers"
        )
        def test_expired_timers(self, mock_debug_info, mock_os_kill):
            # [FUNCTION DOC] With os.kill and the debug-info logger mocked, sends one expired and
            # one valid timer for the same pid. After one server pass, all timers of that pid must
            # be cleared, os.kill called once with the pid and SIGKILL, and debug info logged.
            """
            tests that a single expired timer on a process should terminate
            the process and clean up all pending timers that was owned by the process
            """
            self.server.start()

            test_pid = -3
            client = timer.FileTimerClient(self.file_path)
            client._send_request(self._expired_timer(pid=test_pid, scope="test1"))
            client._send_request(self._valid_timer(pid=test_pid, scope="test2"))

            self.server.run_once()  # Allows the server to process all requests
            self.assertEqual(0, len(self.server._timers))
            mock_os_kill.assert_called_once_with(test_pid, signal.SIGKILL)
            mock_debug_info.assert_called()

        @mock.patch("os.kill")
        def test_send_request_release(self, mock_os_kill):
            # [FUNCTION DOC] Sends an acquire then a release for the same scope, plus a release
            # for an unknown scope (a no-op). Checks the timer table is empty and os.kill was
            # never called.
            """
            tests that:
            1. a timer can be acquired then released (should not terminate process)
            2. a timer can be vacuously released (e.g. no-op)
            """
            self.server.start()

            client = timer.FileTimerClient(self.file_path)
            test_pid = -3
            client._send_request(self._valid_timer(pid=test_pid, scope="test1"))
            client._send_request(self._release_timer(pid=test_pid, scope="test1"))
            client._send_request(self._release_timer(pid=test_pid, scope="test2"))

            self.assertEqual(0, len(self.server._timers))
            mock_os_kill.assert_not_called()

        @mock.patch(
            "torch.distributed.elastic.timer.FileTimerServer.is_process_running"
        )
        @mock.patch("os.kill")
        def test_valid_timers(self, mock_os_kill, mock_pid_exists):
            # [FUNCTION DOC] With the process-running check mocked true and os.kill mocked, sends
            # four valid timers (two pids, two scopes each). After one server pass, all four keys
            # must be in the timer table and os.kill must not be called.
            """
            tests that valid timers are processed correctly and the process is left alone
            """
            self.server.start()
            mock_pid_exists.return_value = True

            client = timer.FileTimerClient(self.file_path)
            client._send_request(self._valid_timer(pid=-3, scope="test1"))
            client._send_request(self._valid_timer(pid=-3, scope="test2"))
            client._send_request(self._valid_timer(pid=-2, scope="test1"))
            client._send_request(self._valid_timer(pid=-2, scope="test2"))

            self.server.run_once()  # Allows the server to process all requests
            self.assertEqual(4, len(self.server._timers))
            self.assertTrue((-3, "test1") in self.server._timers)
            self.assertTrue((-3, "test2") in self.server._timers)
            self.assertTrue((-2, "test1") in self.server._timers)
            self.assertTrue((-2, "test2") in self.server._timers)
            mock_os_kill.assert_not_called()


if __name__ == "__main__":
    run_tests()
