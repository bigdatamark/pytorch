#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]
# [MODULE DOC] Tests for the stdout/stderr redirect context managers in
# torch.distributed.elastic.multiprocessing.redirects. They verify that output written from
# python, from C (libc printf/fprintf) and from child shell commands goes to the log file
# only while the redirect is active, that an invalid stream name raises ValueError, and that
# very large outputs are not lost. Import-time side effect: loads libc.so.6 via ctypes
# (Linux only) and looks up the C stderr stream. When run as a script it raises a
# RuntimeError because the test is not enabled in test discovery.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import ctypes
import os
import shutil
import sys
import tempfile
import unittest

from torch.distributed.elastic.multiprocessing.redirects import (
    redirect,
    redirect_stderr,
    redirect_stdout,
)


libc = ctypes.CDLL("libc.so.6")
c_stderr = ctypes.c_void_p.in_dll(libc, "stderr")


class RedirectsTest(unittest.TestCase):
    # [CLASS DOC] Tests for redirect, redirect_stdout and redirect_stderr. Each test has its
    # own temp directory (test_dir) holding the log files, removed in tearDown.
    def setUp(self):
        # [FUNCTION DOC] Creates a temp directory named after the class for log files.
        super().setUp()
        self.test_dir = tempfile.mkdtemp(prefix=f"{self.__class__.__name__}_")

    def tearDown(self):
        # [FUNCTION DOC] Recursively deletes the temp directory created in setUp.
        shutil.rmtree(self.test_dir)

    def test_redirect_invalid_std(self):
        # [FUNCTION DOC] Calling redirect with an unknown stream name must raise ValueError.
        with self.assertRaises(ValueError):
            with redirect("stdfoo", os.path.join(self.test_dir, "stdfoo.log")):
                pass

    def test_redirect_stdout(self):
        # [FUNCTION DOC] Prints from python, C and a shell command before, inside and after
        # redirect_stdout. Only the three lines produced inside the context must be in the
        # log file (compared as a set since ordering across writers is not guaranteed);
        # output outside goes to the console.
        stdout_log = os.path.join(self.test_dir, "stdout.log")

        # printing to stdout before redirect should go to console not stdout.log
        print("foo first from python")
        libc.printf(b"foo first from c\n")
        os.system("echo foo first from cmd")

        with redirect_stdout(stdout_log):
            print("foo from python")
            libc.printf(b"foo from c\n")
            os.system("echo foo from cmd")

        # make sure stdout is restored
        print("foo again from python")
        libc.printf(b"foo again from c\n")
        os.system("echo foo again from cmd")

        with open(stdout_log) as f:
            # since we print from python, c, cmd -> the stream is not ordered
            # do a set comparison
            lines = set(f.readlines())
            self.assertEqual(
                {"foo from python\n", "foo from c\n", "foo from cmd\n"}, lines
            )

    def test_redirect_stderr(self):
        # [FUNCTION DOC] Same as the stdout test but for stderr: python, C and shell writes
        # inside redirect_stderr land in the log file; those before and after do not.
        # KNOWN ISSUE: the "first" and "again" python prints go to stdout, not stderr, so
        # they do not exercise the stderr restore path.
        stderr_log = os.path.join(self.test_dir, "stderr.log")

        print("bar first from python")
        libc.fprintf(c_stderr, b"bar first from c\n")
        os.system("echo bar first from cmd 1>&2")

        with redirect_stderr(stderr_log):
            print("bar from python", file=sys.stderr)
            libc.fprintf(c_stderr, b"bar from c\n")
            os.system("echo bar from cmd 1>&2")

        print("bar again from python")
        libc.fprintf(c_stderr, b"bar again from c\n")
        os.system("echo bar again from cmd 1>&2")

        with open(stderr_log) as f:
            lines = set(f.readlines())
            self.assertEqual(
                {"bar from python\n", "bar from c\n", "bar from cmd\n"}, lines
            )

    def test_redirect_both(self):
        # [FUNCTION DOC] Nests redirect_stdout and redirect_stderr with separate log files and
        # checks each file receives only its stream's python and C lines from the redirected
        # section.
        stdout_log = os.path.join(self.test_dir, "stdout.log")
        stderr_log = os.path.join(self.test_dir, "stderr.log")

        print("first stdout from python")
        libc.printf(b"first stdout from c\n")

        print("first stderr from python", file=sys.stderr)
        libc.fprintf(c_stderr, b"first stderr from c\n")

        with redirect_stdout(stdout_log), redirect_stderr(stderr_log):
            print("redir stdout from python")
            print("redir stderr from python", file=sys.stderr)
            libc.printf(b"redir stdout from c\n")
            libc.fprintf(c_stderr, b"redir stderr from c\n")

        print("again stdout from python")
        libc.fprintf(c_stderr, b"again stderr from c\n")

        with open(stdout_log) as f:
            lines = set(f.readlines())
            self.assertEqual(
                {"redir stdout from python\n", "redir stdout from c\n"}, lines
            )

        with open(stderr_log) as f:
            lines = set(f.readlines())
            self.assertEqual(
                {"redir stderr from python\n", "redir stderr from c\n"}, lines
            )

    def _redirect_large_buffer(self, print_fn, num_lines=500_000):
        # [FUNCTION DOC] HELPER: Redirects stdout to a log file and calls print_fn with each
        # integer below num_lines (default 500000), then parses the integer after the colon
        # on each logged line and asserts the set equals the full range (no loss).
        stdout_log = os.path.join(self.test_dir, "stdout.log")

        with redirect_stdout(stdout_log):
            for i in range(num_lines):
                print_fn(i)

        with open(stdout_log) as fp:
            actual = {int(line.split(":")[1]) for line in fp}
            expected = set(range(num_lines))
            self.assertSetEqual(expected, actual)

    def test_redirect_large_buffer_py(self):
        # [FUNCTION DOC] Runs the large-buffer check using python print.
        def py_print(i):
            # [FUNCTION DOC] HELPER: Prints a line "py:<i>" through python stdout.
            print(f"py:{i}")

        self._redirect_large_buffer(py_print)

    def test_redirect_large_buffer_c(self):
        # [FUNCTION DOC] Runs the large-buffer check using libc printf.
        def c_print(i):
            # [FUNCTION DOC] HELPER: Prints a line "c:<i>" through C stdout via libc printf.
            libc.printf(bytes(f"c:{i}\n", "utf-8"))

        self._redirect_large_buffer(c_print)


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
