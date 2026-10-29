#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]
# [MODULE DOC] Unit tests for torch.distributed.elastic.multiprocessing.errors: ProcessFailure
# parsing of error files (old/new/MAST formats, signals, missing file), ChildFailedError
# message rendering and root-cause selection, and the record decorator (writes an error file
# named by TORCHELASTIC_ERROR_FILE when the wrapped function raises, ignores SystemExit and
# a missing env var, and re-raises ChildFailedError after copying the child error file).
# Defines @record fixture functions and a SentinelError. Uses temp directories and patched
# environment variables. Uses unittest.TestCase, with no run_tests entry point.

import json
import os
import shutil
import signal
import tempfile
import unittest
from unittest import mock

from torch.distributed.elastic.multiprocessing.errors import (
    ChildFailedError,
    ProcessFailure,
    record,
)
from torch.distributed.elastic.multiprocessing.errors.error_handler import ErrorHandler


class SentinelError(Exception):
    # [CLASS DOC] HELPER: Marker exception so tests can verify the original error is
    # propagated unchanged through the record decorator.
    # exists so that we can validate that
    # the correct error is raised and propagated
    pass


@record
def raise_exception_fn():
    # [FUNCTION DOC] HELPER: Record-decorated fixture that always raises SentinelError
    # with message foobar.
    raise SentinelError("foobar")


@record
def raise_system_exit_exception_fn(exit_code: int = 1):
    # [FUNCTION DOC] HELPER: Record-decorated fixture that raises a SystemExit whose code is
    # exit_code (default 1). Exit code 0 is swallowed by the decorator in the tests below.
    exp = SystemExit()
    exp.code = exit_code
    raise exp


@record
def good_fn():
    # [FUNCTION DOC] HELPER: Record-decorated fixture that succeeds, printing a greeting.
    print("hello world")


@record
def raise_child_failure_error_fn(name, child_error_file=""):
    # [FUNCTION DOC] HELPER: Record-decorated fixture simulating a failed child. If
    # child_error_file is non-empty, writes a SentinelError record into it (by pointing the
    # error-file env var there temporarily). Then builds a ProcessFailure for rank 0
    # (pid 997, exitcode 1) and raises ChildFailedError with the given name.
    if child_error_file:
        with mock.patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": child_error_file}):
            ErrorHandler().record_exception(SentinelError("foobar"))
    pf = ProcessFailure(local_rank=0, pid=997, exitcode=1, error_file=child_error_file)
    raise ChildFailedError(name, {0: pf})


def read_resource_file(resource_file: str) -> str:
    # [FUNCTION DOC] HELPER: Returns the full text of a file located next to this test
    # module, given its relative name; raises if the file does not exist. Unused here.
    with open(os.path.join(os.path.dirname(__file__), resource_file)) as fp:
        return "".join(fp.readlines())


class ApiTest(unittest.TestCase):
    # [CLASS DOC] Tests for ProcessFailure, ChildFailedError and the record decorator.
    # Each test gets a fresh temp directory (test_dir) and an error file path inside it
    # (test_error_file), removed in tearDown.
    def setUp(self):
        # [FUNCTION DOC] Creates a temp directory named after the class and sets the
        # test_error_file path (error.json) inside it; the file is not created.
        super().setUp()
        self.test_dir = tempfile.mkdtemp(prefix=self.__class__.__name__)
        self.test_error_file = os.path.join(self.test_dir, "error.json")

    def tearDown(self):
        # [FUNCTION DOC] Recursively deletes the temp directory created in setUp.
        shutil.rmtree(self.test_dir)

    def test_failure_incorrect_reply_file(self):
        # [FUNCTION DOC] Writes an error file whose JSON lacks the expected message key
        # and checks that constructing a ProcessFailure from it raises an exception.
        content = {"unknown_key": "unknown_value"}
        with open(self.test_error_file, "w") as fp:
            json.dump(content, fp)
        with self.assertRaises(Exception):
            ProcessFailure(
                local_rank=0, pid=997, exitcode=1, error_file=self.test_error_file
            )

    def failure_with_error_file(self, exception):
        # [FUNCTION DOC] HELPER: Records the given exception into test_error_file via
        # ErrorHandler (env var patched for the call) and returns a ProcessFailure for
        # rank 0, pid 997, exitcode 1 that reads that file.
        with mock.patch.dict(
            os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}
        ):
            ErrorHandler().record_exception(exception)
        return ProcessFailure(
            local_rank=0, pid=997, exitcode=1, error_file=self.test_error_file
        )

    def failure_without_error_file(self, exitcode):
        # [FUNCTION DOC] HELPER: Returns a ProcessFailure (rank 0, pid 997) with the given
        # exitcode and a nonexistent error file name, so no file contents are available.
        return ProcessFailure(
            local_rank=0, pid=997, exitcode=exitcode, error_file="ignored.json"
        )

    def test_process_failure_new_format(self):
        # [FUNCTION DOC] Error file with a plain string message and integer timestamp:
        # checks ProcessFailure exposes message and timestamp (10) as given.
        error_data = {"message": "test error message", "timestamp": 10}
        with open(self.test_error_file, "w") as fp:
            json.dump(error_data, fp)
        pf = ProcessFailure(
            local_rank=0, pid=997, exitcode=1, error_file=self.test_error_file
        )
        self.assertEqual("test error message", pf.message)
        self.assertEqual(10, pf.timestamp)

    def test_process_mast_error_format(self):
        # [FUNCTION DOC] Error file whose timestamp is the string "10": checks ProcessFailure
        # converts it to the integer 10 and keeps the message.
        error_data = {"message": "test error message", "timestamp": "10"}
        with open(self.test_error_file, "w") as fp:
            json.dump(error_data, fp)
        pf = ProcessFailure(
            local_rank=0, pid=997, exitcode=1, error_file=self.test_error_file
        )
        self.assertEqual("test error message", pf.message)
        self.assertEqual(10, pf.timestamp)

    def test_process_failure(self):
        # [FUNCTION DOC] Records a real exception to an error file and checks the resulting
        # ProcessFailure fields: rank, pid, exitcode, error file path, a timestamp equal to
        # the recorded one, a non-empty message, and signal_name of "<N/A>" for exitcode 1.
        pf = self.failure_with_error_file(exception=SentinelError("foobar"))
        self.assertEqual(0, pf.local_rank)
        self.assertEqual(997, pf.pid)
        self.assertEqual(1, pf.exitcode)
        self.assertEqual(self.test_error_file, pf.error_file)
        self.assertEqual(
            pf.error_file_data["message"]["extraInfo"]["timestamp"], str(pf.timestamp)
        )
        self.assertTrue(pf.message)  # check not None and not "" (empty string)
        self.assertEqual("<N/A>", pf.signal_name())

    def test_process_failure_signal(self):
        # [FUNCTION DOC] A negative exitcode of SIGSEGV with no error file yields signal_name
        # "SIGSEGV" and a message "Signal <num> (SIGSEGV) received by PID <pid>".
        pf = self.failure_without_error_file(exitcode=-signal.SIGSEGV)
        self.assertEqual("SIGSEGV", pf.signal_name())
        self.assertEqual(
            f"Signal {signal.SIGSEGV} (SIGSEGV) received by PID {pf.pid}", pf.message
        )

    def test_format_msg_enriches_root_signal_failure(self):
        # [FUNCTION DOC] Patches get_error_handler with a mock whose enrichment hook appends
        # a device-fault marker to the message. Builds a SIGABRT ProcessFailure and renders a
        # ChildFailedError; checks the marker appears in the rendered text but not in the
        # stored failure message, and the hook was called once with the message and the
        # error file path.
        # The root signal failure's rendered message is passed through the
        # handler's enrichment seam at format time (no-op base; a build-swapped
        # handler may append device-fault context). The seam works on the local
        # rendered string, so the failure's stored message is not mutated.
        with open(self.test_error_file, "w") as fp:
            json.dump({"message": "Fatal signal received", "timestamp": 10}, fp)
        handler = mock.MagicMock()
        handler.maybe_enrich_signal_failure_message.side_effect = (
            lambda message, error_file: message + "\n[device fault context]"
        )
        with mock.patch(
            "torch.distributed.elastic.multiprocessing.errors.get_error_handler",
            return_value=handler,
        ):
            pf = ProcessFailure(
                local_rank=0,
                pid=997,
                exitcode=-signal.SIGABRT,
                error_file=self.test_error_file,
            )
            rendered = str(ChildFailedError("trainer", {0: pf}))
        self.assertIn("[device fault context]", rendered)
        self.assertNotIn("[device fault context]", pf.message)
        handler.maybe_enrich_signal_failure_message.assert_called_once_with(
            mock.ANY, self.test_error_file
        )

    def test_process_failure_no_error_file(self):
        # [FUNCTION DOC] Exitcode 138 without an error file: signal_name and error_file are
        # "<N/A>" and the message is the generic hint pointing to the elastic errors docs.
        pf = self.failure_without_error_file(exitcode=138)
        self.assertEqual("<N/A>", pf.signal_name())
        self.assertEqual("<N/A>", pf.error_file)
        self.assertEqual(
            "To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html",
            pf.message,
        )

    def test_child_failed_error(self):
        # [FUNCTION DOC] Builds a ChildFailedError from three failures (two with error files,
        # one without), asserts get_first_failure returns the rank 0 failure as root cause,
        # and prints the error for manual inspection (the string literal in the body shows
        # an example of the output).
        pf0 = self.failure_with_error_file(exception=SentinelError("rank 0"))
        pf1 = self.failure_with_error_file(exception=SentinelError("rank 1"))
        pf2 = self.failure_without_error_file(exitcode=138)
        ex = ChildFailedError("trainer.par", {0: pf0, 1: pf1, 2: pf2})
        self.assertEqual(pf0, ex.get_first_failure()[1])
        # print is intentional and should prints something like this:
        """
        *********************************************
              trainer.par FAILED
        =============================================
        Root Cause:
        [0]:
          time: 2020-11-25_21:22:31
          rank: 0 (local_rank: 0)
          exitcode: 1 (pid: 997)
          error_file: /tmp/ApiTesttbb37ier/error.json
          traceback: "SentinelError: rank 0"
        =============================================
        Other Failures:
        [1]:
          time: 2020-11-25_21:22:31
          rank: 1 (local_rank: 0)
          exitcode: 1 (pid: 997)
          error_file: /tmp/ApiTesttbb37ier/error.json
          msg: "SentinelError: rank 1"
        [2]:
          time: 2020-11-25_21:22:31
          rank: 2 (local_rank: 0)
          exitcode: 138 (pid: 997)
          error_file: <N/A>
          traceback: To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
        *********************************************
        """
        print(ex)

    def test_record(self):
        # [FUNCTION DOC] With the error-file env var set, a raising record-decorated function
        # must still raise SentinelError, and the file must contain a message, a python
        # callstack and a timestamp under the nested message and extraInfo keys.
        with mock.patch.dict(
            os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}
        ):
            with self.assertRaises(SentinelError):
                raise_exception_fn()

        with open(self.test_error_file) as fp:
            err = json.load(fp)
            self.assertIsNotNone(err["message"]["message"])
            self.assertIsNotNone(err["message"]["extraInfo"]["py_callstack"])
            self.assertIsNotNone(err["message"]["extraInfo"]["timestamp"])

    def test_record_system_exit(self):
        # [FUNCTION DOC] A record-decorated function raising SystemExit with code 0 returns
        # without raising and writes no error file.
        # KNOWN ISSUE: patch.dict with an empty mapping does not clear the env var, and
        # test_error_file is never configured in the environment, so the file check is weak.
        with mock.patch.dict(os.environ, {}):
            raise_system_exit_exception_fn(exit_code=0)

        # no error file should have been generated
        self.assertFalse(os.path.isfile(self.test_error_file))

    def test_record_system_exit_erronr(self):
        # [FUNCTION DOC] A record-decorated function raising SystemExit with nonzero code (1)
        # still raises SystemExit; no error file is expected. The name has a typo (erronr).
        with mock.patch.dict(os.environ, {}):
            with self.assertRaises(SystemExit):
                raise_system_exit_exception_fn()

        # no error file should have been generated
        self.assertFalse(os.path.isfile(self.test_error_file))

    def test_record_no_error_file(self):
        # [FUNCTION DOC] Without an error-file env var, a raising record-decorated function
        # still propagates SentinelError and no error file appears at test_error_file.
        with mock.patch.dict(os.environ, {}):
            with self.assertRaises(SentinelError):
                raise_exception_fn()

        # no error file should have been generated
        self.assertFalse(os.path.isfile(self.test_error_file))

    def test_record_good_fn(self):
        # [FUNCTION DOC] A record-decorated function that succeeds must not create the error
        # file even when the env var points at it.
        with mock.patch.dict(
            os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}
        ):
            good_fn()
            # function did not error; no error file should be produced
            self.assertFalse(os.path.isfile(self.test_error_file))

    def test_record_child_failure(self):
        # [FUNCTION DOC] A child failure with a worker error file raises ChildFailedError out
        # of the record-decorated function; the test then loads the worker file and the
        # parent error file for comparison (with the exit code set as errorCode).
        # KNOWN ISSUE: the comparison uses assertTrue with two arguments, so the second is
        # only a message and the check always passes when the first string is non-empty.
        trainer_log_dir = os.path.join(self.test_dir, "trainer", "0")
        os.makedirs(trainer_log_dir)
        trainer_error_file = os.path.join(trainer_log_dir, "error.json")

        with mock.patch.dict(
            os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}
        ):
            with self.assertRaises(ChildFailedError) as cm:
                raise_child_failure_error_fn("trainer", trainer_error_file)
            pf = cm.exception.get_first_failure()[1]
            # compare worker error file with reply file and overridden error code
            with open(pf.error_file) as f:
                expect = json.load(f)
            expect["message"]["errorCode"] = pf.exitcode
            with open(self.test_error_file) as f:
                actual = json.load(f)
            self.assertTrue(
                json.dumps(expect, sort_keys=True),
                json.dumps(actual, sort_keys=True),
            )

    def test_record_child_failure_no_child_error_file(self):
        # [FUNCTION DOC] A ChildFailedError with no child error file is re-raised by record
        # and is not itself recorded: the parent error file must not exist afterwards.
        with mock.patch.dict(
            os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}
        ):
            with self.assertRaises(ChildFailedError):
                raise_child_failure_error_fn("trainer")

            # @record should only copy child error file when ChildFailedError
            # is raised - it should NOT record ChildFailedError itself
            # it SHOULD re-raise ChildFailedError for any upstream system
            # to handle it.
            self.assertFalse(os.path.isfile(self.test_error_file))

    def test_child_failed_error_signal_name_in_message(self):
        # [FUNCTION DOC] For a SIGSEGV failure, the rendered ChildFailedError string must
        # contain "(SIGSEGV)" and the negative exit code in an exitcode line.
        pf = self.failure_without_error_file(exitcode=-signal.SIGSEGV)
        ex = ChildFailedError("trainer.par", {0: pf})
        error_msg = str(ex)
        self.assertIn("(SIGSEGV)", error_msg)
        self.assertIn(f"exitcode  : {-signal.SIGSEGV}", error_msg)
