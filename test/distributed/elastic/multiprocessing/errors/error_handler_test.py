#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]
# [MODULE DOC] Unit tests for ErrorHandler and get_error_handler in the elastic
# multiprocessing errors package: faulthandler setup (errors tolerated), recording an
# exception to the file named by TORCHELASTIC_ERROR_FILE (no failure when unset), and
# copying a source error file to the configured destination (overwriting existing content).
# Uses temp directories, patched environment and mocks. Uses unittest.TestCase with no
# run_tests entry point.

import filecmp
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from torch.distributed.elastic.multiprocessing.errors.error_handler import ErrorHandler
from torch.distributed.elastic.multiprocessing.errors.handlers import get_error_handler


def raise_exception_fn():
    # [FUNCTION DOC] HELPER: Always raises RuntimeError with message foobar.
    raise RuntimeError("foobar")


class GetErrorHandlerTest(unittest.TestCase):
    # [CLASS DOC] Tests the get_error_handler factory.
    def test_get_error_handler(self):
        # [FUNCTION DOC] Checks get_error_handler returns an ErrorHandler instance.
        self.assertTrue(isinstance(get_error_handler(), ErrorHandler))


class ErrorHandlerTest(unittest.TestCase):
    # [CLASS DOC] Tests ErrorHandler initialize, record_exception and dump_error_file.
    # Each test gets a temp directory (test_dir) and an error file path (test_error_file),
    # removed in tearDown.
    def setUp(self):
        # [FUNCTION DOC] Creates a temp directory named after the class and sets the
        # test_error_file path inside it (file not created).
        super().setUp()
        self.test_dir = tempfile.mkdtemp(prefix=self.__class__.__name__)
        self.test_error_file = os.path.join(self.test_dir, "error.json")

    def tearDown(self):
        # [FUNCTION DOC] Recursively deletes the temp directory created in setUp.
        shutil.rmtree(self.test_dir)

    @patch("faulthandler.enable")
    def test_initialize(self, fh_enable_mock):
        # [FUNCTION DOC] Checks ErrorHandler.initialize enables faulthandler exactly once
        # (faulthandler.enable is patched).
        ErrorHandler().initialize()
        fh_enable_mock.assert_called_once()

    @patch("faulthandler.enable", side_effect=RuntimeError)
    def test_initialize_error(self, fh_enable_mock):
        # [FUNCTION DOC] When faulthandler.enable raises RuntimeError, initialize must swallow
        # it (no exception escapes) after having attempted the call once.
        # makes sure that initialize handles errors gracefully
        ErrorHandler().initialize()
        fh_enable_mock.assert_called_once()

    def test_record_exception(self):
        # [FUNCTION DOC] With the error-file env var set, records a caught RuntimeError and
        # checks the JSON file has a non-null message, python callstack and timestamp under
        # the nested message and extraInfo keys.
        with patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}):
            eh = ErrorHandler()
            eh.initialize()

            try:
                raise_exception_fn()
            except Exception as e:
                eh.record_exception(e)

            with open(self.test_error_file) as fp:
                err = json.load(fp)
                # error file content example:
                # {
                #   "message": {
                #     "message": "RuntimeError: foobar",
                #     "extraInfo": {
                #       "py_callstack": "Traceback (most recent call last):\n  <... OMITTED ...>",
                #       "timestamp": "1605774851"
                #     }
                #   }
            self.assertIsNotNone(err["message"]["message"])
            self.assertIsNotNone(err["message"]["extraInfo"]["py_callstack"])
            self.assertIsNotNone(err["message"]["extraInfo"]["timestamp"])

    def test_record_exception_no_error_file(self):
        # [FUNCTION DOC] Records a caught exception with no error-file path configured and
        # checks nothing is raised.
        # KNOWN ISSUE: patch.dict with an empty mapping does not remove the env var if it
        # is already set in the outer environment.
        # make sure record does not fail when no error file is specified in env vars
        with patch.dict(os.environ, {}):
            eh = ErrorHandler()
            eh.initialize()
            try:
                raise_exception_fn()
            except Exception as e:
                eh.record_exception(e)

    def test_dump_error_file(self):
        # [FUNCTION DOC] Records an exception into a source file, then with the env var
        # pointing at a different destination checks dump_error_file produces an identical
        # file. Also calls it with no destination configured, which should only log.
        src_error_file = os.path.join(self.test_dir, "src_error.json")
        eh = ErrorHandler()
        with patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": src_error_file}):
            eh.record_exception(RuntimeError("foobar"))

        with patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}):
            eh.dump_error_file(src_error_file)
            self.assertTrue(filecmp.cmp(src_error_file, self.test_error_file))

        with patch.dict(os.environ, {}):
            eh.dump_error_file(src_error_file)
            # just validate that dump_error_file works when
            # my error file is not set
            # should just log an error with src_error_file pretty printed

    def test_dump_error_file_overwrite_existing(self):
        # [FUNCTION DOC] Records different exceptions into destination and source files, then
        # dumps source onto destination and checks the destination now matches the source
        # (existing content is overwritten).
        dst_error_file = os.path.join(self.test_dir, "dst_error.json")
        src_error_file = os.path.join(self.test_dir, "src_error.json")
        eh = ErrorHandler()
        with patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": dst_error_file}):
            eh.record_exception(RuntimeError("foo"))

        with patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": src_error_file}):
            eh.record_exception(RuntimeError("bar"))

        with patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": dst_error_file}):
            eh.dump_error_file(src_error_file)
            self.assertTrue(filecmp.cmp(src_error_file, dst_error_file))
