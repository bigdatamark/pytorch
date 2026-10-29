# [MODULE DOC] Pytest configuration plugin for the PyTorch test suite (conftest). Adds
# command line and ini options (stepcurrent: --sc, --scs, --rs; --use-main-module;
# --hw-classification; sharding options; a rerun-aware junit xml writer under
# --junit-xml-reruns). Implements hooks and plugin classes: LogXMLReruns and
# _NodeReporterReruns (junit xml with rerun and escaped skip reasons),
# HardwareClassificationPytestPlugin (filters tests), StepcurrentPlugin (resume from the
# last running test), a terminal summary for reruns, per-test duration in verbose output,
# filtering to disabled tests when rerunning them, and automatic multigpu markers. Much of
# the xml code is copied and adapted from _pytest.junitxml. Import-time side effects:
# creates the stash key xml_key and falls back to a no-op parse_cmd_line_args if torch
# lacks it.
import copy
import functools
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from types import MethodType
from typing import Any, TYPE_CHECKING, TypeGuard

import pytest
from _pytest.config import Config, filename_arg
from _pytest.config.argparsing import Parser
from _pytest.junitxml import _NodeReporter, bin_xml_escape, LogXML
from _pytest.python import Module
from _pytest.reports import TestReport
from _pytest.stash import StashKey
from _pytest.terminal import _get_raw_skip_reason

from pytest_shard_custom import pytest_addoptions as shard_addoptions, PytestShardPlugin


try:
    from torch.testing._internal.common_utils import parse_cmd_line_args
except ImportError:
    # Temporary workaround needed until parse_cmd_line_args makes it into a nightlye because
    # main / PR's tests are sometimes run against the previous day's nightly which won't
    # have this function.
    def parse_cmd_line_args():
        # [FUNCTION DOC] HELPER: No-op fallback used when torch.testing does not provide
        # parse_cmd_line_args (older nightlies).
        pass


if TYPE_CHECKING:
    from _pytest._code.code import ReprFileLocation

    from torch.testing._internal.common_distributed import (
        MultiProcContinuousTest,
        MultiProcessTestCase,
    )

# a lot of this file is copied from _pytest.junitxml and modified to get rerun info

xml_key = StashKey["LogXMLReruns"]()
STEPCURRENT_CACHE_DIR = "cache/stepcurrent"


def pytest_addoption(parser: Parser) -> None:
    # [FUNCTION DOC] Pytest hook registering options: stepcurrent flags (--scs skip, --sc,
    # --rs run single), --use-main-module, --junit-xml-reruns and --junit-prefix-reruns with
    # matching ini entries for suite name, logging, passing-test logs, duration and xml
    # family, the --hw-classification filter (upper-cased names), and the shard options from
    # the shard plugin. Mutates the parser; returns nothing.
    group = parser.getgroup("general")
    group.addoption("--scs", action="store", default=None, dest="stepcurrent_skip")
    group.addoption("--sc", action="store", default=None, dest="stepcurrent")
    group.addoption("--rs", action="store", default=None, dest="run_single")

    parser.addoption("--use-main-module", action="store_true")
    group = parser.getgroup("terminal reporting")
    group.addoption(
        "--junit-xml-reruns",
        action="store",
        dest="xmlpath_reruns",
        metavar="path",
        type=functools.partial(filename_arg, optname="--junit-xml-reruns"),
        default=None,
        help="create junit-xml style report file at given path.",
    )
    group.addoption(
        "--junit-prefix-reruns",
        action="store",
        metavar="str",
        default=None,
        help="prepend prefix to classnames in junit-xml output",
    )
    parser.addini(
        "junit_suite_name_reruns", "Test suite name for JUnit report", default="pytest"
    )
    parser.addini(
        "junit_logging_reruns",
        "Write captured log messages to JUnit report: "
        "one of no|log|system-out|system-err|out-err|all",
        default="no",
    )
    parser.addini(
        "junit_log_passing_tests_reruns",
        "Capture log information for passing tests to JUnit report: ",
        type="bool",
        default=True,
    )
    parser.addini(
        "junit_duration_report_reruns",
        "Duration time to report: one of total|call",
        default="total",
    )
    parser.addini(
        "junit_family_reruns",
        "Emit XML for schema: one of legacy|xunit1|xunit2",
        default="xunit2",
    )
    parser.addoption(
        "--hw-classification",
        nargs="+",
        default=None,
        metavar="SCOPE",
        dest="hw_classification",
        type=str.upper,
        help="filter tests by hardware classification categories (e.g., GENERIC ACCELERATOR CPU CUDA MPS XPU)",
    )
    shard_addoptions(parser)


class HardwareClassificationPytestPlugin:
    # [CLASS DOC] Pytest plugin that keeps only tests whose class matches the requested
    # hardware classifications. State: hw_classification, a set of enum values, or None for no
    # filtering. Key method: pytest_collection_modifyitems.
    """Pytest plugin to filter collected tests by hw_classification."""

    def __init__(self, hw_classification):
        # [FUNCTION DOC] Stores the resolved hardware classification set (see
        # _resolve_hw_classification).
        self.hw_classification = self._resolve_hw_classification(hw_classification)

    @staticmethod
    def _resolve_hw_classification(hw_classification):
        # [FUNCTION DOC] HELPER: Maps the given classification names to the HardwareClassification
        # enum and returns them as a set, or None if no names were given. Raises KeyError for an
        # unknown name.
        if hw_classification is None:
            return None
        from torch.testing._internal.common_utils import HardwareClassification

        return {HardwareClassification[name] for name in hw_classification}

    def pytest_collection_modifyitems(self, items):
        # [FUNCTION DOC] Pytest hook that, when a classification is set, replaces the collected
        # items in place with those accepted by the shared torch filter helper (class looked up
        # from each item). Does nothing when no classification was requested.
        if self.hw_classification is None:
            return
        import torch.testing._internal.common_utils as _cu

        filtered = []
        _cu.filter_by_hw_classification(
            items,
            self.hw_classification,
            get_class=lambda item: getattr(item, "cls", None),
            on_match=filtered.append,
        )
        items[:] = filtered


def pytest_configure(config: Config) -> None:
    # [FUNCTION DOC] Pytest hook run at startup. Parses the torch command line args; registers
    # the rerun-aware xml logger (not on xdist workers) when a path was given; maps the skip
    # and run-single options onto the stepcurrent option and registers StepcurrentPlugin when
    # any is set; registers the shard plugin when num_shards is set, and the hardware
    # classification plugin when requested.
    parse_cmd_line_args()

    xmlpath = config.option.xmlpath_reruns
    # Prevent opening xmllog on worker nodes (xdist).
    if xmlpath and not hasattr(config, "workerinput"):
        junit_family = config.getini("junit_family_reruns")
        config.stash[xml_key] = LogXMLReruns(
            xmlpath,
            config.option.junitprefix,
            config.getini("junit_suite_name_reruns"),
            config.getini("junit_logging_reruns"),
            config.getini("junit_duration_report_reruns"),
            junit_family,
            config.getini("junit_log_passing_tests_reruns"),
        )
        config.pluginmanager.register(config.stash[xml_key])
    if config.getoption("stepcurrent_skip"):
        config.option.stepcurrent = config.getoption("stepcurrent_skip")
    if config.getoption("run_single"):
        config.option.stepcurrent = config.getoption("run_single")
    if config.getoption("stepcurrent"):
        config.pluginmanager.register(StepcurrentPlugin(config), "stepcurrentplugin")
    if config.getoption("num_shards"):
        config.pluginmanager.register(PytestShardPlugin(config), "pytestshardplugin")
    if config.getoption("hw_classification"):
        config.pluginmanager.register(
            HardwareClassificationPytestPlugin(config.getoption("hw_classification")),
            "hw_classification_plugin",
        )


def pytest_unconfigure(config: Config) -> None:
    # [FUNCTION DOC] Pytest hook run at exit; removes and unregisters the rerun xml logger if
    # one was registered.
    xml = config.stash.get(xml_key, None)
    if xml:
        del config.stash[xml_key]
        config.pluginmanager.unregister(xml)


class _NodeReporterReruns(_NodeReporter):
    # [CLASS DOC] Junit xml node reporter that differs from the base in how content is
    # written: content is escaped and written as a tag only when non-empty, and skip reasons
    # are escaped. Created by LogXMLReruns.node_reporter.
    def _prepare_content(self, content: str, header: str) -> str:
        # [FUNCTION DOC] HELPER: Returns the content unchanged (no header prefix or wrapping).
        return content

    def _write_content(self, report: TestReport, content: str, jheader: str) -> None:
        # [FUNCTION DOC] HELPER: Appends an xml element named after the header with the escaped
        # content as text to this node; does nothing for empty content.
        if content == "":
            return
        tag = ET.Element(jheader)
        tag.text = bin_xml_escape(content)
        self.append(tag)

    def append_skipped(self, report: TestReport) -> None:
        # [FUNCTION DOC] Records a skipped test in the xml node. For expected failures delegates
        # to the base class. Otherwise unpacks the file, line and reason from the report
        # (AssertionError if the location is not a tuple), strips the skipped prefix and adds an
        # escaped skipped element with the details, then writes captured output.
        # Referenced from the below
        # https://github.com/pytest-dev/pytest/blob/2178ee86d7c1ee93748cfb46540a6e40b4761f2d/src/_pytest/junitxml.py#L236C6-L236C6
        # Modified to escape characters not supported by xml in the skip reason.  Everything else should be the same.
        if hasattr(report, "wasxfail"):
            # Super here instead of the actual code so we can reduce possible divergence
            super().append_skipped(report)
        else:
            if not isinstance(report.longrepr, tuple):
                raise AssertionError(
                    f"Expected report.longrepr to be tuple, got {type(report.longrepr)}"
                )
            filename, lineno, skipreason = report.longrepr
            skipreason = skipreason.removeprefix("Skipped: ")
            details = f"{filename}:{lineno}: {skipreason}"

            skipped = ET.Element(
                "skipped", type="pytest.skip", message=bin_xml_escape(skipreason)
            )
            skipped.text = bin_xml_escape(details)
            self.append(skipped)
            self.write_captured_output(report)


class LogXMLReruns(LogXML):
    # [CLASS DOC] Junit xml log writer that also records reruns. Extends the pytest xml logger
    # and uses _NodeReporterReruns for per-test nodes. Key methods: append_rerun,
    # pytest_runtest_logreport and node_reporter.
    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Passes all arguments to the base class constructor unchanged.
        super().__init__(*args, **kwargs)

    def append_rerun(self, reporter: _NodeReporter, report: TestReport) -> None:
        # [FUNCTION DOC] Adds a rerun entry for a report to the reporter: a skipped note for
        # unexpectedly passing xfail tests, otherwise a rerun element with the escaped crash
        # message (or the long representation text) and full representation. Raises AssertionError
        # if longrepr is None.
        if hasattr(report, "wasxfail"):
            reporter._add_simple("skipped", "xfail-marked test passes unexpectedly")
        else:
            if report.longrepr is None:
                raise AssertionError("Expected report.longrepr to not be None")
            reprcrash: ReprFileLocation | None = getattr(
                report.longrepr, "reprcrash", None
            )
            if reprcrash is not None:
                message = reprcrash.message
            else:
                message = str(report.longrepr)
            message = bin_xml_escape(message)
            reporter._add_simple("rerun", message, str(report.longrepr))

    def pytest_runtest_logreport(self, report: TestReport) -> None:
        # [FUNCTION DOC] Pytest hook handling a test report: runs the base handling, records
        # reruns for reports with the rerun outcome, and rewrites the skip reason of skipped
        # reports to include the node id (mutating the report location tuple).
        super().pytest_runtest_logreport(report)
        if report.outcome == "rerun":
            reporter = self._opentestcase(report)
            self.append_rerun(reporter, report)
        if report.outcome == "skipped":
            if isinstance(report.longrepr, tuple):
                fspath, lineno, reason = report.longrepr
                reason = f"{report.nodeid}: {_get_raw_skip_reason(report)}"
                report.longrepr = (fspath, lineno, reason)

    def node_reporter(self, report: TestReport | str) -> _NodeReporterReruns:
        # [FUNCTION DOC] Returns the node reporter for a report or node id, creating and
        # registering it in the ordered list on first use. The cache key is the node id plus the
        # xdist worker node.
        # KNOWN ISSUE: per the existing TODO the key handling breaks with the each distribution
        # mode.
        nodeid: str | TestReport = getattr(report, "nodeid", report)
        # Local hack to handle xdist report order.
        workernode = getattr(report, "node", None)

        key = nodeid, workernode

        if key in self.node_reporters:
            # TODO: breaks for --dist=each
            return self.node_reporters[key]

        reporter = _NodeReporterReruns(nodeid, self)

        self.node_reporters[key] = reporter
        self.node_reporters_ordered.append(reporter)

        return reporter


# imitating summary_failures in pytest's terminal.py
# both hookwrapper and tryfirst to make sure this runs before pytest's
@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # [FUNCTION DOC] Pytest hook wrapper (runs first) that prints a RERUNS section before the
    # normal terminal summary when traceback style is not disabled: crash lines for line
    # style, else full failure headlines and summaries for each rerun report. Writes to the
    # terminal; then yields.
    # prints stack traces for reruns
    if terminalreporter.config.option.tbstyle != "no":
        reports = terminalreporter.getreports("rerun")
        if reports:
            terminalreporter.write_sep("=", "RERUNS")
            if terminalreporter.config.option.tbstyle == "line":
                for rep in reports:
                    line = terminalreporter._getcrashline(rep)
                    terminalreporter.write_line(line)
            else:
                for rep in reports:
                    msg = terminalreporter._getfailureheadline(rep)
                    terminalreporter.write_sep("_", msg, red=True, bold=True)
                    terminalreporter._outrep_summary(rep)
                    terminalreporter._handle_teardown_sections(rep.nodeid)
    yield


@pytest.hookimpl(tryfirst=True)
def pytest_pycollect_makemodule(module_path, parent) -> Module:
    # [FUNCTION DOC] Pytest hook that, when --use-main-module is set, builds the module
    # collector with its imported object replaced by the main module and returns it. Returns
    # None otherwise so pytest falls back to the default.
    if parent.config.getoption("--use-main-module"):
        mod = Module.from_parent(parent, path=module_path)
        mod._getobj = MethodType(lambda x: sys.modules["__main__"], mod)
        return mod


@pytest.hookimpl(hookwrapper=True)
def pytest_report_teststatus(report, config):
    # [FUNCTION DOC] Pytest hook wrapper that appends the test duration in seconds to the
    # verbose status text of each test report (any phase). Non-test-report objects are left
    # alone.
    # Add the test time to the verbose output, unfortunately I don't think this
    # includes setup or teardown
    pluggy_result = yield
    if not isinstance(report, pytest.TestReport):
        return
    outcome, letter, verbose = pluggy_result.get_result()
    if verbose:
        pluggy_result.force_result(
            (outcome, letter, f"{verbose} [{report.duration:.4f}s]")
        )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items: list[Any]) -> None:
    # [FUNCTION DOC] Pytest hook (runs last). Only active when the rerun-disabled-tests
    # environment variable is 1 and a disabled tests file exists: reads the json list of
    # disabled test names, parses name and class from each, and replaces the collected items
    # in place with copies of just the disabled ones. Returns nothing.
    """
    This hook is used when rerunning disabled tests to get rid of all skipped tests
    instead of running and skipping them N times. This avoids flooding the console
    and XML outputs with junk. So we want this to run last when collecting tests.
    """
    rerun_disabled_tests = os.getenv("PYTORCH_TEST_RERUN_DISABLED_TESTS", "0") == "1"
    if not rerun_disabled_tests:
        return

    disabled_regex = re.compile(r"(?P<test_name>.+)\s+\([^\.]+\.(?P<test_class>.+)\)")
    disabled_tests = defaultdict(set)

    # This environment has already been set by run_test before it calls pytest
    disabled_tests_file = os.getenv("DISABLED_TESTS_FILE", "")
    if not disabled_tests_file or not os.path.exists(disabled_tests_file):
        return

    with open(disabled_tests_file) as fp:
        for disabled_test in json.load(fp):
            m = disabled_regex.match(disabled_test)
            if m:
                test_name = m["test_name"]
                test_class = m["test_class"]
                disabled_tests[test_class].add(test_name)

    # When rerunning disabled test, ignore all test cases that are not disabled
    filtered_items = []

    for item in items:
        test_name = item.name
        test_class = item.parent.name

        if (
            test_class not in disabled_tests
            or test_name not in disabled_tests[test_class]
        ):
            continue

        cpy = copy.copy(item)
        cpy._initrequest()

        filtered_items.append(cpy)

    items.clear()
    # NB: Need to edit items directly here to have the list reflected back to pytest
    items.extend(filtered_items)


def _spawns_multiple_processes(
    cls: type | None,
) -> TypeGuard["type[MultiProcessTestCase | MultiProcContinuousTest] | None"]:
    # [FUNCTION DOC] HELPER: Returns whether a test class spawns several processes (so it
    # needs many GPUs): true when the class is None (ambiguity), false when torch.distributed
    # is not available, else whether it subclasses MultiProcessTestCase or
    # MultiProcContinuousTest.
    # KNOWN ISSUE: annotated as a type guard although it returns a plain answer.
    """
    A distributed test needs multiple GPUs iff its class spawns multiple
    processes. This is encoded by the base class: MultiProcessTestCase and
    MultiProcContinuousTest fork `world_size` subprocesses, one per GPU. Plain
    TestCase and MultiThreadedTestCase run in a single process (one GPU).

    Returns True (needs multiple GPUs) on any ambiguity, so we never route a
    process-spawning test to a single-GPU runner.
    """
    if cls is None:
        return True
    import torch.distributed as dist

    if not dist.is_available():
        # Distributed wasn't built: no test can spawn processes, so nothing is
        # multigpu and the single-GPU config runs everything.
        return False
    from torch.testing._internal.common_distributed import (
        MultiProcContinuousTest,
        MultiProcessTestCase,
    )

    return issubclass(cls, (MultiProcessTestCase, MultiProcContinuousTest))


def pytest_itemcollected(item: Any) -> None:
    # [FUNCTION DOC] Pytest hook run per collected item: adds the multigpu marker when the
    # item class spawns multiple processes (or is unknown), so CI can select by marker.
    """
    Auto-apply the `multigpu` marker based on the resolved test class. Runs
    per-item during collection, before pytest applies `-m` deselection, so the
    distributed CI configs can partition a file into multigpu and `not
    multigpu` halves without touching the test source.
    """
    if _spawns_multiple_processes(getattr(item, "cls", None)):
        item.add_marker("multigpu")


class StepcurrentPlugin:
    # [CLASS DOC] Pytest plugin that saves the currently running test id in the pytest cache
    # and, on the next run, skips everything before it (or only runs that test). State:
    # config, the cache, the lastrun cache key and value, initial value, skip and run_single
    # flags, a status string and the cache key recording that a failing xml was made. Hooks
    # report a status line and update the cache.
    # Modified fromo _pytest/stepwise.py in order to save the currently running
    # test instead of the last failed test
    def __init__(self, config: Config) -> None:
        # [FUNCTION DOC] Reads the stepcurrent name from options, computes cache keys, loads the
        # last run test id (None if missing) and initial value, stores skip and run_single flags,
        # and resets the made-failing-xml cache flag to false. Raises AssertionError if the cache
        # plugin is disabled.
        self.config = config
        self.report_status = ""
        if config.cache is None:
            raise AssertionError("Expected config.cache to not be None")
        self.cache: pytest.Cache = config.cache
        directory = f"{STEPCURRENT_CACHE_DIR}/{config.getoption('stepcurrent')}"
        self.lastrun_location = f"{directory}/lastrun"
        self.lastrun: str | None = self.cache.get(self.lastrun_location, None)
        self.initial_val = self.lastrun
        self.skip: bool = config.getoption("stepcurrent_skip")
        self.run_single: bool = config.getoption("run_single")

        self.made_failing_xml_location = f"{directory}/made_failing_xml"
        self.cache.set(self.made_failing_xml_location, False)

    def pytest_collection_modifyitems(self, config: Config, items: list[Any]) -> None:
        # [FUNCTION DOC] Pytest hook that finds the item matching the last run id and deselects
        # all items before it (also that item when the skip flag is set; in run-single mode all
        # later items as well) via the deselected hook, mutating the items list. Sets a status
        # message; does nothing when no previous run exists or the id is not found.
        if not self.lastrun:
            self.report_status = "Cannot find last run test, not skipping"
            return

        # check all item nodes until we find a match on last run
        failed_index = None
        for index, item in enumerate(items):
            if item.nodeid == self.lastrun:
                failed_index = index
                if self.skip:
                    failed_index += 1
                break

        # If the previously failed test was not found among the test items,
        # do not skip any tests.
        if failed_index is None:
            self.report_status = "previously run test not found, not skipping."
        else:
            self.report_status = f"skipping {failed_index} already run items."
            deselected = items[:failed_index]
            del items[:failed_index]
            if self.run_single:
                self.report_status += f" Running only {items[0].nodeid}"
                deselected += items[1:]
                del items[1:]
            config.hook.pytest_deselected(items=deselected)

    def pytest_report_collectionfinish(self) -> str | None:
        # [FUNCTION DOC] Pytest hook returning the status line to print after collection when
        # verbosity is not negative and a status exists, else None.
        if self.config.getoption("verbose") >= 0 and self.report_status:
            return f"stepcurrent: {self.report_status}"
        return None

    def pytest_runtest_protocol(self, item, nextitem) -> None:
        # [FUNCTION DOC] Pytest hook that records the id of the item about to run as the last run,
        # both on the plugin and in the cache. Returns None so pytest continues its normal
        # protocol.
        self.lastrun = item.nodeid
        self.cache.set(self.lastrun_location, self.lastrun)

    def pytest_sessionfinish(self, session, exitstatus):
        # [FUNCTION DOC] Pytest hook at session end: on success restores the cache entry to the
        # value read at startup; on failure sets the made-failing-xml flag in the cache.
        if exitstatus == 0:
            self.cache.set(self.lastrun_location, self.initial_val)
        if exitstatus != 0:
            self.cache.set(self.made_failing_xml_location, True)
