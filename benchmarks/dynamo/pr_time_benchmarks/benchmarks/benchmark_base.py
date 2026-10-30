# [MODULE DOC] Base class and shared measurement logic for the PR-time dynamo benchmarks.
# BenchmarkBase measures either the raw instruction count of the work step or the compile-time
# instruction count recorded by dynamo, takes the minimum over several iterations, and writes
# results as CSV rows and as JSON records for the OSS benchmark database. Import-time side
# effect: when the CI environment variable equals true, imports the scribe logger and creates a
# logger for compile-time benchmark entries (used to log counts to an internal store).
import csv
import gc
import json
import os
from abc import ABC, abstractmethod
from typing_extensions import Self

from sympy.core.cache import clear_cache as clear_sympy_cache

import torch._C._instruction_counter as i_counter
import torch._dynamo.config as config
from torch._dynamo.utils import CompileTimeInstructionCounter


log_to_scuba = os.getenv("CI", "false") == "true"
if log_to_scuba:
    from fbscribelogger import make_scribe_logger

    scribe_log_torch_benchmark_compile_time = make_scribe_logger(
        "TorchBenchmarkCompileTime",
        """
        struct TorchBenchmarkCompileTimeLogEntry {

        # The commit SHA that triggered the workflow, e.g., 02a6b1d30f338206a71d0b75bfa09d85fac0028a. Derived from GITHUB_SHA.
        4: optional string commit_sha;

        # The unit timestamp in second for the Scuba Time Column override
        6: optional i64 time;
        7: optional i64 instruction_count; # Instruction count of compilation step
        8: optional string name; # Benchmark name

        # Commit date (not author date) of the commit in commit_sha as timestamp, e.g., 1724208105.  Increasing if merge bot is used, though not monotonic; duplicates occur when stack is landed.
        16: optional i64 commit_date;

        # A unique number for each workflow run within a repository, e.g., 19471190684. Derived from GITHUB_RUN_ID.
        17: optional string github_run_id;

        # A unique number for each attempt of a particular workflow run in a repository, e.g., 1. Derived from GITHUB_RUN_ATTEMPT.
        18: optional string github_run_attempt;

        # Indicates if branch protections or rulesets are configured for the ref that triggered the workflow run. Derived from GITHUB_REF_PROTECTED.
        20: optional bool github_ref_protected;

        # The fully-formed ref of the branch or tag that triggered the workflow run, e.g., refs/pull/133891/merge or refs/heads/main. Derived from GITHUB_REF.
        21: optional string github_ref;

        # The weight of the record according to current sampling rate
        25: optional i64 weight;

        # The name of the current job. Derived from JOB_NAME, e.g., linux-jammy-py3.8-gcc11 / test (default, 3, 4, linux.2xlarge).
        26: optional string github_job;

        # The GitHub user who triggered the job.  Derived from GITHUB_TRIGGERING_ACTOR.
        27: optional string github_triggering_actor;

        # A unique number for each run of a particular workflow in a repository, e.g., 238742. Derived from GITHUB_RUN_NUMBER.
        28: optional string github_run_number_str;
        }
        """,
    )


class BenchmarkBase(ABC):
    # [CLASS DOC] Abstract base for a single benchmark. Subclasses provide the name, optional
    # description, per-iteration setup and the measured work step, and may provide one-time
    # setup. Configuration (category, device, backend, mode, dynamic) is stored at
    # construction. Chainable enable methods choose the metric; collect_all runs the
    # measurement and fills the results list of tuples with name, metric name and value, and
    # append_results persists them. Class attributes hold the metric switches (both off by
    # default) and the iteration count (5).
    # Measure total number of instruction spent in _work.
    # Garbage collection is NOT disabled during _work().
    _enable_instruction_count = False

    # Measure total number of instruction spent in convert_frame.compile_inner
    # Garbage collection is disabled during _work() to avoid noise.
    _enable_compile_time_instruction_count = False

    # number of iterations used to run when collecting instruction_count or compile_time_instruction_count.
    _num_iterations = 5

    def __init__(
        self,
        category: str,
        device: str,
        backend: str = "",
        mode: str = "",
        dynamic=None,
    ) -> None:
        # [FUNCTION DOC] Stores the category, device, backend, mode (training or inference
        # label, empty by default) and dynamic setting. These are used for naming and for the
        # dashboard filters in the JSON output.
        # These individual attributes are used to support different filters on the
        # dashboard later
        self._category = category
        self._device = device
        self._backend = backend
        self._mode = mode  # Training or inference
        self._dynamic = dynamic

    def with_iterations(self, value: int) -> Self:
        # [FUNCTION DOC] Sets how many iterations are measured and returns self for chaining.
        self._num_iterations = value
        return self

    def enable_instruction_count(self) -> Self:
        # [FUNCTION DOC] Turns on measuring raw instruction count of the work step and returns
        # self for chaining.
        self._enable_instruction_count = True
        return self

    def enable_compile_time_instruction_count(self) -> Self:
        # [FUNCTION DOC] Turns on measuring the compile-time instruction count and returns self
        # for chaining.
        self._enable_compile_time_instruction_count = True
        return self

    def name(self) -> str:
        # [FUNCTION DOC] Returns the unique benchmark name; the base implementation returns an
        # empty string and subclasses override it.
        return ""

    def backend(self) -> str:
        # [FUNCTION DOC] Returns the backend label given at construction.
        return self._backend

    def mode(self) -> str:
        # [FUNCTION DOC] Returns the mode label given at construction.
        return self._mode

    def category(self) -> str:
        # [FUNCTION DOC] Returns the category given at construction.
        return self._category

    def device(self) -> str:
        # [FUNCTION DOC] Returns the device given at construction.
        return self._device

    def is_dynamic(self) -> bool | None:
        # [FUNCTION DOC] Returns the dynamic-shapes setting given at construction (a bool
        # or None).
        return self._dynamic

    def description(self) -> str:
        # [FUNCTION DOC] Returns a description; the base implementation returns an empty string.
        return ""

    @abstractmethod
    def _prepare(self) -> None:
        # [FUNCTION DOC] Abstract per-iteration setup, run before every measured iteration; the
        # base body does nothing.
        pass

    @abstractmethod
    def _work(self) -> None:
        # [FUNCTION DOC] Abstract measured work step, run once per iteration; the base body
        # does nothing.
        pass

    def _prepare_once(self) -> None:  # noqa: B027
        # [FUNCTION DOC] Optional one-time setup run at the start of collect_all before any
        # measurement; the base implementation does nothing.
        pass

    @staticmethod
    def _clear_sympy_interning_cache() -> None:
        # [FUNCTION DOC] HELPER: Clears sympy's global caches (static method). They act as an
        # interning table, so clearing them before each iteration makes the measurement start
        # from a deterministic cold state.
        # Sympy's cacheit LRU caches double as an interning table: evictions
        # mean structurally equal expressions stop being identity-equal, which
        # permanently degrades ShapeEnv cache lookups (deep __eq__ instead of
        # pointer comparison). Which entries got evicted depends on what
        # happened to run earlier in the process (e.g. whether inductor was
        # imported before or during the first measured compile), so clear the
        # caches before each iteration to measure a deterministic cold state.
        clear_sympy_cache()

    def _count_instructions(self) -> int:
        # [FUNCTION DOC] Measures raw instruction count of the work step. For each configured
        # iteration it runs the per-iteration setup, clears sympy caches, starts the instruction
        # counter, runs the work and stops the counter. Prints each count and the benchmark
        # name, and returns the minimum over the iterations. Garbage collection stays enabled.
        print(f"collecting instruction count for {self.name()}")
        results = []
        for i in range(self._num_iterations):
            self._prepare()
            self._clear_sympy_interning_cache()
            id = i_counter.start()
            self._work()
            count = i_counter.end(id)
            print(f"instruction count for iteration {i} is {count}")
            results.append(count)
        return min(results)

    def _count_compile_time_instructions(self) -> int:
        # [FUNCTION DOC] Measures the compile-time instruction count. Disables garbage
        # collection, turns on the dynamo option that records compile-time counts, then for each
        # iteration runs setup, clears sympy caches, collects garbage, clears the counter, runs
        # the work and reads the counter. Raises RuntimeError if a count is zero. Returns the
        # minimum count, turns the recording option off on success and always re-enables garbage
        # collection at the end.
        # KNOWN ISSUE: if the work or the zero check raises, the recording config option is left
        # enabled because only the success path resets it.
        gc.disable()

        try:
            print(f"collecting compile time instruction count for {self.name()}")
            config.record_compile_time_instruction_count = True

            results = []
            for i in range(self._num_iterations):
                self._prepare()
                self._clear_sympy_interning_cache()
                gc.collect()
                # CompileTimeInstructionCounter.record is only called on convert_frame._compile_inner
                # hence this will only count instruction count spent in compile_inner.
                CompileTimeInstructionCounter.clear()
                self._work()
                count = CompileTimeInstructionCounter.value()
                if count == 0:
                    raise RuntimeError(
                        "compile time instruction count is 0, please check your benchmarks"
                    )
                print(f"compile time instruction count for iteration {i} is {count}")
                results.append(count)

            config.record_compile_time_instruction_count = False
            return min(results)
        finally:
            gc.enable()

    def _write_to_json(self, output_dir: str) -> None:
        # [FUNCTION DOC] Writes the collected results to a JSON file named after the benchmark
        # in the given directory. Each result with a metric name and a value that is not None
        # becomes a record holding benchmark info (name pr_time_benchmarks, mode, dynamic flag,
        # device, description), model info (name, category as type, backend) and the metric
        # with a one-element value list. Overwrites any existing file. Requires results to have
        # been collected.
        """
        Write the result into JSON format, so that it can be uploaded to the benchmark database
        to be displayed on OSS dashboard. The JSON format is defined at
        https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
        """
        records = []
        for entry in self.results:
            metric_name = entry[1]
            value = entry[2]

            if not metric_name or value is None:
                continue

            records.append(
                {
                    "benchmark": {
                        "name": "pr_time_benchmarks",
                        "mode": self.mode(),
                        "extra_info": {
                            "is_dynamic": self.is_dynamic(),
                            "device": self.device(),
                            "description": self.description(),
                        },
                    },
                    "model": {
                        "name": self.name(),
                        "type": self.category(),
                        "backend": self.backend(),
                    },
                    "metric": {
                        "name": metric_name,
                        "benchmark_values": [value],
                    },
                }
            )

        with open(os.path.join(output_dir, f"{self.name()}.json"), "w") as f:
            json.dump(records, f)

    def append_results(self, path: str) -> None:
        # [FUNCTION DOC] Appends the collected results as CSV rows (name, metric, value) to the
        # file at the path, creating it if needed, then writes the JSON record file into the
        # directory of that path. Requires results to have been collected.
        with open(path, "a", newline="") as csvfile:
            # Create a writer object
            writer = csv.writer(csvfile)
            # Write the data to the CSV file
            for entry in self.results:
                writer.writerow(entry)

        # TODO (huydhn) This requires the path to write to, so it needs to be in the same place
        # as the CSV writer for now
        self._write_to_json(os.path.dirname(os.path.abspath(path)))

    def print(self) -> None:
        # [FUNCTION DOC] Prints each collected result as one comma-separated line of name,
        # metric and value.
        for entry in self.results:
            print(f"{entry[0]},{entry[1]},{entry[2]}")

    def collect_all(self) -> Self:
        # [FUNCTION DOC] Runs the benchmark: calls the one-time setup, resets the results list,
        # and raises RuntimeError when both metrics are enabled. If raw instruction counting is
        # enabled it records the count under the instruction_count metric. If compile-time
        # counting is enabled it measures with C++ symbolic shape guards disabled and records
        # the compile_time_instruction_count metric. When running in CI each count is also sent
        # to the scribe logger. Returns self.
        # KNOWN ISSUE: if neither metric is enabled, results stays empty with no warning.
        self._prepare_once()
        self.results = []
        if (
            self._enable_instruction_count
            and self._enable_compile_time_instruction_count
        ):
            raise RuntimeError(
                "not supported until we update the logger, both logs to the same field now"
            )

        if self._enable_instruction_count:
            r = self._count_instructions()
            self.results.append((self.name(), "instruction_count", r))
            if log_to_scuba:
                scribe_log_torch_benchmark_compile_time(
                    name=self.name(),
                    instruction_count=r,
                )
        if self._enable_compile_time_instruction_count:
            # enable_cpp_symbolic_shape_guards has impact on these benchmarks
            # Keep using False value for consistency.
            with config.patch("enable_cpp_symbolic_shape_guards", False):
                r = self._count_compile_time_instructions()

            self.results.append(
                (
                    self.name(),
                    "compile_time_instruction_count",
                    r,
                )
            )
            if log_to_scuba:
                # TODO add a new field compile_time_instruction_count to the logger.
                scribe_log_torch_benchmark_compile_time(
                    name=self.name(),
                    instruction_count=r,
                )
        return self
