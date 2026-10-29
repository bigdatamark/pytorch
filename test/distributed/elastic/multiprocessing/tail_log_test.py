#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]
# [MODULE DOC] Tests for torch.distributed.elastic.multiprocessing.tail_log.TailLog, which
# tails several per-rank log files into one destination stream with a per-rank prefix. Covers
# many concurrent writers into a string buffer and a file, custom line prefixes, a custom
# line filter, no log files, log files that never appear, and error handling when the tail
# function fails (a directory given as log file). Uses a thread pool to simulate writers and
# unittest.TestCase with no run_tests entry point.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import io
import os
import shutil
import sys
import tempfile
import time
import unittest
from concurrent.futures import wait
from concurrent.futures._base import ALL_COMPLETED
from concurrent.futures.thread import ThreadPoolExecutor
from unittest import mock

from torch.distributed.elastic.multiprocessing.tail_log import TailLog


def write(max: int, sleep: float, file: str):
    # [FUNCTION DOC] HELPER: Creates (truncating) the file and writes the integers from 0 up
    # to max, one per line, flushing after each and sleeping the given seconds between
    # lines. Simulates a worker producing a log.
    with open(file, "w") as fp:
        for i in range(max):
            print(i, file=fp, flush=True)
            time.sleep(sleep)


class TailLogTest(unittest.TestCase):
    # [CLASS DOC] Tests for TailLog. Per-test state: a temp directory (test_dir) for log
    # files and a thread pool (threadpool) that runs the simulated writers.
    def setUp(self):
        # [FUNCTION DOC] Creates the temp directory and the writer thread pool.
        # KNOWN ISSUE: the thread pool is never shut down in tearDown.
        super().setUp()
        self.test_dir = tempfile.mkdtemp(prefix=f"{self.__class__.__name__}_")
        self.threadpool = ThreadPoolExecutor()

    def tearDown(self):
        # [FUNCTION DOC] Recursively deletes the temp directory created in setUp.
        shutil.rmtree(self.test_dir)

    def test_tail(self):
        # [FUNCTION DOC] Starts TailLog over 32 not-yet-existing log files into a string
        # buffer, then runs 32 writers of 1000 lines each at different speeds. After stopping
        # the tail, checks each rank's header (name plus rank) received every number exactly
        # as a set, and that stopped() is false before stop and true after.
        """
        writer() writes 0 - max (on number on each line) to a log file.
        Run nprocs such writers and tail the log files into an IOString
        and validate that all lines are accounted for.
        """
        nprocs = 32
        max = 1000
        interval_sec = 0.0001

        log_files = {
            local_rank: os.path.join(self.test_dir, f"{local_rank}_stdout.log")
            for local_rank in range(nprocs)
        }

        dst = io.StringIO()
        tail = TailLog(
            name="writer", log_files=log_files, dst=dst, interval_sec=interval_sec
        ).start()
        # sleep here is intentional to ensure that the log tail
        # can gracefully handle and wait for non-existent log files
        time.sleep(interval_sec * 10)

        futs = []
        for local_rank, file in log_files.items():
            f = self.threadpool.submit(
                write, max=max, sleep=interval_sec * local_rank, file=file
            )
            futs.append(f)

        wait(futs, return_when=ALL_COMPLETED)
        self.assertFalse(tail.stopped())
        tail.stop()

        dst.seek(0)
        actual: dict[int, set[int]] = {}

        for line in dst.readlines():
            header, num = line.split(":")
            nums = actual.setdefault(header, set())
            nums.add(int(num))

        self.assertEqual(nprocs, len(actual))
        self.assertEqual(
            {f"[writer{i}]": set(range(max)) for i in range(nprocs)}, actual
        )
        self.assertTrue(tail.stopped())

    def test_tail_write_to_dst_file(self):
        # [FUNCTION DOC] Same scenario as test_tail but the destination is an open text file
        # with line buffering; after the tail stops the file is re-read and every rank's
        # numbers are verified.
        """
        writer() writes 0 - max (on number on each line) to a log file.
        Run nprocs such writers and tail the log files into a temp file
        and validate that all lines are accounted for.
        """
        nprocs = 32
        max = 1000
        interval_sec = 0.0001

        log_files = {
            local_rank: os.path.join(self.test_dir, f"{local_rank}_stdout.log")
            for local_rank in range(nprocs)
        }

        dst = os.path.join(self.test_dir, "tailed_stdout.log")
        with open(dst, "w", encoding="utf8", buffering=1) as dst_file:
            tail = TailLog(
                name="writer",
                log_files=log_files,
                dst=dst_file,
                interval_sec=interval_sec,
            ).start()
            # sleep here is intentional to ensure that the log tail
            # can gracefully handle and wait for non-existent log files
            time.sleep(interval_sec * 10)

            futs = []
            for local_rank, file in log_files.items():
                f = self.threadpool.submit(
                    write, max=max, sleep=interval_sec * local_rank, file=file
                )
                futs.append(f)

            wait(futs, return_when=ALL_COMPLETED)
            self.assertFalse(tail.stopped())
            tail.stop()

        actual: dict[int, set[int]] = {}
        with open(dst, encoding="utf8") as read_dst_file:
            for line in read_dst_file:
                header, num = line.split(":")
                nums = actual.setdefault(header, set())
                nums.add(int(num))

        self.assertEqual(nprocs, len(actual))
        self.assertEqual(
            {f"[writer{i}]": set(range(max)) for i in range(nprocs)}, actual
        )
        self.assertTrue(tail.stopped())

    def test_tail_with_custom_prefix(self):
        # [FUNCTION DOC] Tails 3 logs of 10 lines with a log_line_prefixes mapping and checks
        # the headers seen in the output are exactly the three custom prefixes.
        """
        writer() writes 0 - max (on number on each line) to a log file.
        Run nprocs such writers and tail the log files into an IOString
        and validate that all lines are accounted for.
        """
        nprocs = 3
        max = 10
        interval_sec = 0.0001

        log_files = {
            local_rank: os.path.join(self.test_dir, f"{local_rank}_stdout.log")
            for local_rank in range(nprocs)
        }

        dst = io.StringIO()
        log_line_prefixes = {n: f"[worker{n}][{n}]:" for n in range(nprocs)}
        tail = TailLog(
            "writer",
            log_files,
            dst,
            interval_sec=interval_sec,
            log_line_prefixes=log_line_prefixes,
        ).start()
        # sleep here is intentional to ensure that the log tail
        # can gracefully handle and wait for non-existent log files
        time.sleep(interval_sec * 10)
        futs = []
        for local_rank, file in log_files.items():
            f = self.threadpool.submit(
                write, max=max, sleep=interval_sec * local_rank, file=file
            )
            futs.append(f)
        wait(futs, return_when=ALL_COMPLETED)
        self.assertFalse(tail.stopped())
        tail.stop()
        dst.seek(0)

        headers: set[str] = set()
        for line in dst.readlines():
            header, _ = line.split(":")
            headers.add(header)
        self.assertEqual(nprocs, len(headers))
        for i in range(nprocs):
            self.assertIn(f"[worker{i}][{i}]", headers)
        self.assertTrue(tail.stopped())

    def test_tail_with_custom_filter(self):
        # [FUNCTION DOC] Tails 3 logs of 20 lines with a line filter that keeps only lines
        # containing the digit 2, and checks each rank output only numbers 2 and 12.
        """
        writer() writes 0 - max (on number on each line) to a log file.
        Run nprocs such writers and tail the log files into an IOString
        and validate that all lines are accounted for.
        """
        nprocs = 3
        max = 20
        interval_sec = 0.0001

        log_files = {
            local_rank: os.path.join(self.test_dir, f"{local_rank}_stdout.log")
            for local_rank in range(nprocs)
        }

        dst = io.StringIO()
        tail = TailLog(
            "writer",
            log_files,
            dst,
            interval_sec=interval_sec,
            log_line_filter=lambda line: "2" in line,  # only print lines containing '2'
        ).start()
        # sleep here is intentional to ensure that the log tail
        # can gracefully handle and wait for non-existent log files
        time.sleep(interval_sec * 10)
        futs = []
        for local_rank, file in log_files.items():
            f = self.threadpool.submit(
                write, max=max, sleep=interval_sec * local_rank, file=file
            )
            futs.append(f)
        wait(futs, return_when=ALL_COMPLETED)
        self.assertFalse(tail.stopped())
        tail.stop()
        dst.seek(0)

        actual: dict[int, set[int]] = {}
        for line in dst.readlines():
            header, num = line.split(":")
            nums = actual.setdefault(header, set())
            nums.add(int(num))
        self.assertEqual(nprocs, len(actual))
        self.assertEqual({f"[writer{i}]": {2, 12} for i in range(nprocs)}, actual)
        self.assertTrue(tail.stopped())

    def test_tail_no_files(self):
        # [FUNCTION DOC] With an empty log file mapping, TailLog starts, reports not stopped,
        # and after stop reports stopped; nothing else happens.
        """
        Ensures that the log tail can gracefully handle no log files
        in which case it does nothing.
        """
        tail = TailLog("writer", log_files={}, dst=sys.stdout).start()
        self.assertFalse(tail.stopped())
        tail.stop()
        self.assertTrue(tail.stopped())

    def test_tail_logfile_never_generates(self):
        # [FUNCTION DOC] For a log file that never appears, stop must still finish and mark
        # the tail stopped and shut down its internal thread pool (checks the private
        # shutdown flag).
        """
        Ensures that we properly shutdown the threadpool
        even when the logfile never generates.
        """

        tail = TailLog("writer", log_files={0: "foobar.log"}, dst=sys.stdout).start()
        tail.stop()
        self.assertTrue(tail.stopped())
        self.assertTrue(tail._threadpool._shutdown)

    @mock.patch("torch.distributed.elastic.multiprocessing.tail_log.logger")
    def test_tail_logfile_error_in_tail_fn(self, mock_logger):
        # [FUNCTION DOC] Passes a directory as the log file so the tail function fails while
        # opening it; after stop, the patched module logger must have recorded exactly one
        # exception.
        """
        Ensures that when there is an error in the tail_fn (the one that runs in the
        threadpool), it is dealt with and raised properly.
        """

        # try giving tail log a directory (should fail with an IsADirectoryError
        tail = TailLog("writer", log_files={0: self.test_dir}, dst=sys.stdout).start()
        tail.stop()

        mock_logger.exception.assert_called_once()
