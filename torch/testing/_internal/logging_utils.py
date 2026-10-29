# mypy: ignore-errors

# [MODULE DOC] Test helpers for the torch logging system (TORCH_LOGS and set_logs).
# Provides context managers that save/replace log state and apply settings through the
# environment variable or the API, decorators that run a test under both routes, a base
# test case that gathers emitted log records, and helpers capturing artifact logs into
# string streams. No import-time side effects beyond imports.

import torch._dynamo.test_case
import unittest.mock
import os
import contextlib
import torch._logging
import torch._logging._internal
from contextlib import AbstractContextManager
from collections.abc import Callable
from torch._dynamo.utils import LazyString
from torch._inductor import config as inductor_config
import logging
import io

@contextlib.contextmanager
def preserve_log_state():
    # [FUNCTION DOC] HELPER: context manager that saves the global log state, installs a fresh
    # empty state for the body, and on exit (even on error) restores the saved state and
    # re-initializes the loggers. Mutates torch logging global state.
    prev_state = torch._logging._internal._get_log_state()
    torch._logging._internal._set_log_state(torch._logging._internal.LogState())
    try:
        yield
    finally:
        torch._logging._internal._set_log_state(prev_state)
        torch._logging._internal._init_logs()

def log_settings(settings):
    # [FUNCTION DOC] Applies a TORCH_LOGS style settings string: enters a preserved log state
    # and a patched environment variable, then re-initializes logs immediately. Returns the
    # exit stack (already entered); the caller must use it as a context manager or close it
    # to restore the environment and log state.
    exit_stack = contextlib.ExitStack()
    settings_patch = unittest.mock.patch.dict(os.environ, {"TORCH_LOGS": settings})
    exit_stack.enter_context(preserve_log_state())
    exit_stack.enter_context(settings_patch)
    torch._logging._internal._init_logs()
    return exit_stack

def log_api(**kwargs):
    # [FUNCTION DOC] Like log_settings but configures logging through the set_logs API with the
    # given keyword arguments. Returns the entered exit stack that restores the log state when
    # closed.
    exit_stack = contextlib.ExitStack()
    exit_stack.enter_context(preserve_log_state())
    torch._logging.set_logs(**kwargs)
    return exit_stack


def kwargs_to_settings(**kwargs):
    # [FUNCTION DOC] Converts set_logs style keyword arguments into a comma separated settings
    # string. A boolean value adds the bare name; an integer level (debug, info or error
    # numbers) adds the name with a plus, empty or minus prefix; a mapping under the key
    # modules adds one entry per module name and level. Returns the joined string.
    # Raises a value error for any other value type or unsupported level.
    # KNOWN ISSUE: a false boolean still adds the bare name, enabling the setting.
    INT_TO_VERBOSITY = {10: "+", 20: "", 40: "-"}

    settings = []

    def append_setting(name, level):
        # [FUNCTION DOC] HELPER: appends the verbosity prefix plus name to the settings list;
        # raises a value error if name is not a string or level is not a supported integer.
        if isinstance(name, str) and isinstance(level, int) and level in INT_TO_VERBOSITY:
            settings.append(INT_TO_VERBOSITY[level] + name)
            return
        else:
            raise ValueError("Invalid value for setting")

    for name, val in kwargs.items():
        if isinstance(val, bool):
            settings.append(name)
        elif isinstance(val, int):
            append_setting(name, val)
        elif isinstance(val, dict) and name == "modules":
            for module_qname, level in val.items():
                append_setting(module_qname, level)
        else:
            raise ValueError("Invalid value for setting")

    return ",".join(settings)


# Note on testing strategy:
# This class does two things:
# 1. Runs two versions of a test:
#    1a. patches the env var log settings to some specific value
#    1b. calls torch._logging.set_logs(..)
# 2. patches the emit method of each setup handler to gather records
# that are emitted to each console stream
# 3. passes a ref to the gathered records to each test case for checking
#
# The goal of this testing in general is to ensure that given some settings env var
# that the logs are setup correctly and capturing the correct records.
def make_logging_test(**kwargs):
    # [FUNCTION DOC] Decorator factory: the given keyword arguments are log settings. The
    # decorated test body receives the list of gathered records and is run twice, first with
    # the settings applied through the environment variable (or no settings if none given),
    # then through the set_logs API, resetting dynamo before each run. The inductor graph cache
    # is disabled while running. Returns a decorator producing the wrapped test function.
    def wrapper(fn):
        # [FUNCTION DOC] HELPER: wraps fn into a test method taking only self.
        @inductor_config.patch({"fx_graph_cache": False})
        def test_fn(self):
            # [FUNCTION DOC] HELPER: runs fn with the records list under both setting routes.

            torch._dynamo.reset()
            records = []
            # run with env var
            if len(kwargs) == 0:
                with self._handler_watcher(records):
                    fn(self, records)
            else:
                with log_settings(kwargs_to_settings(**kwargs)), self._handler_watcher(records):
                    fn(self, records)

            # run with API
            torch._dynamo.reset()
            records.clear()
            with log_api(**kwargs), self._handler_watcher(records):
                fn(self, records)


        return test_fn

    return wrapper

def make_settings_test(settings):
    # [FUNCTION DOC] Decorator factory: like make_logging_test but takes a ready settings string
    # and runs the test only once, via the environment variable route. The decorated body gets
    # the list of gathered records.
    def wrapper(fn):
        # [FUNCTION DOC] HELPER: wraps fn into a test method taking only self.
        def test_fn(self):
            # [FUNCTION DOC] HELPER: resets dynamo and runs fn with the records list.
            torch._dynamo.reset()
            records = []
            # run with env var
            with log_settings(settings), self._handler_watcher(records):
                fn(self, records)

        return test_fn

    return wrapper

class LoggingTestCase(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Base dynamo test case for logging tests. Class setup patches the environment
    # and dynamo config (log testing flag, suppress_errors on, verbose off) via the shared exit
    # stack; teardown closes it and resets logging state. Offers record lookup assertions and a
    # handler watcher that collects emitted records.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Runs the base setup, then enters patches for the log testing environment
        # variable, suppress_errors (true) and verbose (false) on the class exit stack.
        super().setUpClass()
        cls._exit_stack.enter_context(
            unittest.mock.patch.dict(os.environ, {"___LOG_TESTING": ""})
        )
        cls._exit_stack.enter_context(
            unittest.mock.patch("torch._dynamo.config.suppress_errors", True)
        )
        cls._exit_stack.enter_context(
            unittest.mock.patch("torch._dynamo.config.verbose", False)
        )

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Closes the class exit stack, clears the global log state and
        # re-initializes the loggers.
        # KNOWN ISSUE: does not call the base class teardown.
        cls._exit_stack.close()
        torch._logging._internal.log_state.clear()
        torch._logging._init_logs()

    def hasRecord(self, records, m):
        # [FUNCTION DOC] Returns true if any record's formatted message contains the text m.
        return any(m in r.getMessage() for r in records)

    def getRecord(self, records, m):
        # [FUNCTION DOC] Returns the single record whose formatted message contains m. Fails the
        # test if more than one record matches or none does.
        record = None
        for r in records:
            # NB: not r.msg because it looks like 3.11 changed how they
            # structure log records
            if m in r.getMessage():
                self.assertIsNone(
                    record,
                    msg=LazyString(
                        lambda: f"multiple matching records: {record} and {r} among {records}"
                    ),
                )
                record = r
        if record is None:
            self.fail(f"did not find record with {m} among {records}")
        return record

    # This patches the emit method of each handler to gather records
    # as they are emitted
    def _handler_watcher(self, record_list):
        # [FUNCTION DOC] HELPER: for every registered torch logger, asserts it has one or two
        # torch-owned handlers and patches each handler's emit to also append the record (once
        # per record identity) to record_list. Returns an exit stack holding the patches; the
        # caller closes it to undo them. Fails the test on a bad handler count.
        exit_stack = contextlib.ExitStack()
        seen_record_ids = set()

        def emit_post_hook(record):
            # [FUNCTION DOC] HELPER: appends the record to the list if not already seen.
            nonlocal record_list
            record_id = id(record)
            if record_id not in seen_record_ids:
                seen_record_ids.add(record_id)
                record_list.append(record)

        # Registered logs are the only ones with handlers, so patch those. Count and
        # patch only torch-owned handlers: when running under pytest (or any framework
        # that attaches its own capture handlers to these loggers), those foreign
        # handlers must not count against torch's own handler budget.
        for log_qname in torch._logging._internal.log_registry.get_log_qnames():
            logger = logging.getLogger(log_qname)
            torch_handlers = [
                h
                for h in logger.handlers
                if torch._logging._internal._is_torch_handler(h)
            ]
            num_handlers = len(torch_handlers)
            self.assertLessEqual(
                num_handlers,
                2,
                "All pt2 loggers should only have at most two Torch handlers (debug artifacts and messages above debug level).",
            )

            self.assertGreater(num_handlers, 0, "All pt2 loggers should have more than zero Torch handlers")

            for handler in torch_handlers:
                old_emit = handler.emit

                def make_emit(old_emit):
                    # [FUNCTION DOC] HELPER: builds a replacement emit bound to old_emit.
                    def new_emit(record):
                        # [FUNCTION DOC] HELPER: calls the original emit then the post hook.
                        old_emit(record)
                        emit_post_hook(record)

                    return new_emit

                exit_stack.enter_context(
                    unittest.mock.patch.object(handler, "emit", make_emit(old_emit))
                )

        return exit_stack


def logs_to_string(module, log_option):
    # [FUNCTION DOC] Returns a pair: an in-memory text stream and a factory. Calling the factory
    # enables the given log option via settings and attaches a stream handler to the artifact
    # logger of the module, returning an exit stack to use as a context manager that undoes both.
    # The stream receives the artifact log output produced meanwhile.
    """Example:
    logs_to_string("torch._inductor.compile_fx", "post_grad_graphs")
    returns the output of TORCH_LOGS="post_grad_graphs" from the
    torch._inductor.compile_fx module.
    """
    log_stream = io.StringIO()
    handler = logging.StreamHandler(stream=log_stream)

    @contextlib.contextmanager
    def tmp_redirect_logs():
        # [FUNCTION DOC] HELPER: attaches the stream handler to the artifact logger for the body
        # and removes it afterwards.
        # KNOWN ISSUE: the logger lookup is inside the try block, so a failure there makes the
        # cleanup reference an unbound variable.
        try:
            logger = torch._logging.getArtifactLogger(module, log_option)
            logger.addHandler(handler)
            yield
        finally:
            logger.removeHandler(handler)

    def ctx_manager():
        # [FUNCTION DOC] HELPER: enables the log option and the redirect; returns the exit stack.
        exit_stack = log_settings(log_option)
        exit_stack.enter_context(tmp_redirect_logs())
        return exit_stack

    return log_stream, ctx_manager


def multiple_logs_to_string(module: str, *log_options: str) -> tuple[list[io.StringIO], Callable[[], AbstractContextManager[None]]]:
    # [FUNCTION DOC] Multi-option version of logs_to_string: returns a list of text streams, one
    # per log option in order, and a factory whose context enables all options together and
    # routes each option's artifact logger output to its own stream.
    """Example:
    multiple_logs_to_string("torch._inductor.compile_fx", "pre_grad_graphs", "post_grad_graphs")
    returns the output of TORCH_LOGS="pre_graph_graphs, post_grad_graphs" from the
    torch._inductor.compile_fx module.
    """
    log_streams = [io.StringIO() for _ in range(len(log_options))]
    handlers = [logging.StreamHandler(stream=log_stream) for log_stream in log_streams]

    @contextlib.contextmanager
    def tmp_redirect_logs():
        # [FUNCTION DOC] HELPER: attaches each stream handler to its option's artifact logger for
        # the body and removes them all afterwards.
        loggers = [torch._logging.getArtifactLogger(module, option) for option in log_options]
        try:
            for logger, handler in zip(loggers, handlers, strict=True):
                logger.addHandler(handler)
            yield
        finally:
            for logger, handler in zip(loggers, handlers, strict=True):
                logger.removeHandler(handler)

    def ctx_manager() -> AbstractContextManager[None]:
        # [FUNCTION DOC] HELPER: enables all log options (joined) plus the redirect; returns
        # the exit stack.
        exit_stack = log_settings(", ".join(log_options))
        exit_stack.enter_context(tmp_redirect_logs())
        return exit_stack  # type: ignore[return-value]

    return log_streams, ctx_manager
