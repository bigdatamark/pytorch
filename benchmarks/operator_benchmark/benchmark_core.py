# [MODULE DOC] Core of the operator micro-benchmark framework. Defines the TestConfig record, the
# global registry of benchmark test definitions (BENCHMARK_TESTER) and the list of skipped
# operator names, helpers that expand a registered config list into concrete test cases, and
# BenchmarkRunner, which filters, times, prints and exports results (CSV, JSON, dashboard JSON).
# Import-time side effects: imports torch and the C++ extension utilities and creates the empty
# registry.
import ast
import copy
import csv
import functools
import json
import os
import platform
import timeit
from collections import namedtuple
from dataclasses import asdict, dataclass
from typing import Any

import benchmark_utils

import numpy as np

import torch

# needs to be imported after torch
import torch.utils.cpp_extension as cpp_extension  # noqa: F401
from torch.utils.benchmark import Timer


"""Performance microbenchmarks.

This module contains core functionalities for performance microbenchmark tests.
"""

"""
This is used to store configs of tests
An example input is:
TestConfig(test_name='add_M8_N2_K1', input_config='M: 8, N: 2, K: 1',
    tag='long', run_backward=False)
"""
TestConfig = namedtuple("TestConfig", "test_name input_config tag run_backward")


BENCHMARK_TESTER = []

SKIP_OP_LISTS = ["weight_norm_sparsifier_step"]


def _register_test(*test_metainfo):
    # [FUNCTION DOC] HELPER: Registers one benchmark definition by appending the
    # given tuple of arguments (configs, benchmark op class, test case factory,
    # backward flag, and optionally an op name and function mapping) to the global
    # BENCHMARK_TESTER list. Mutates global state; returns nothing.
    """save the metainfo needed to create a test. Currently test_metainfo
    takes two different inputs:
    1) This input when adds single op to the benchmark
     _register_test(configs, pt_bench_op, create_pytorch_op_test_case,
                      run_backward=True)
    2) This input when adds a list of ops to the benchmark
    _register_test(configs, pt_bench_op, create_pytorch_op_test_case,
                      run_backward=False,
                      op_name_function=op)
    """
    BENCHMARK_TESTER.append(test_metainfo)


def _create_test(
    bench_op_obj, orig_test_attrs, tags, OperatorTestCase, run_backward, bwd_input
):
    # [FUNCTION DOC] HELPER: Builds one test case from an initialized benchmark op. Deep-copies
    # the attribute mapping, converts all values to strings, and derives the human readable
    # input_config string (attributes joined without braces or quotes). If bwd_input is truthy it
    # is added to the attributes under a backward key so the test name reflects which input is
    # being differentiated. The name comes from the op's test_name method. Returns whatever the
    # OperatorTestCase factory returns for the op and a TestConfig (name, input config, tags,
    # run_backward); for the PyTorch factory that is a pair of a unique name string and a test
    # case object. The caller's mapping is not modified.
    """Create tests with the benchmark backend.
    Args:
        bench_op_obj: an object which instantiated from a subclass of
            TorchBenchmarkBase which includes tensor
            creation and operator execution.
        orig_test_attrs: a dictionary includes test configs.
        tags: a attribute in test config to filter inputs
        OperatorTestCase: a named tuple to save the metadata of an test
        run_backward: a bool parameter indicating backward path
    """
    test_attrs = copy.deepcopy(orig_test_attrs)
    test_attrs = {k: str(v) for k, v in test_attrs.items()}
    ascii_test_attrs = ast.literal_eval(json.dumps(test_attrs))
    input_config = str(ascii_test_attrs)[1:-1].replace("'", "")
    if bwd_input:
        # When auto_set is used, the test name needs to include input.
        test_attrs.update({"bwd": bwd_input})
    test_name = bench_op_obj.test_name(**test_attrs)
    test_config = TestConfig(test_name, input_config, tags, run_backward)
    return OperatorTestCase(bench_op_obj, test_config)


def _build_test(
    configs, bench_op, OperatorTestCase, run_backward, op_name_function=None
):
    # [FUNCTION DOC] HELPER: Generator that expands a list of configs into test cases. Each config
    # is a list of single-entry mappings; a tags entry is extracted, entries naming cuda, mps or
    # xpu devices cause the whole config to be skipped when that backend is unavailable, the rest
    # are merged into the init attributes. Raises ValueError if tags are missing and
    # AssertionError if the op factory returns None. For each kept config it creates the op,
    # optionally sets the module name and op function from op_name_function, records the backward
    # flag, calls init, extracts inputs, and disables requires_grad on module parameters when no
    # backward is run. It yields one test case, and then, when auto_set tensors exist, yields one
    # more per such input using a deep-copied op re-initialized with an incremented pass counter.
    """Generate PyTorch/Caffe2 tests of operators with different inputs.
    Args:
        configs: a dictionary that has the input shapes
        bench_op: a subclass of TorchBenchmarkBase which includes tensor
            creation and operator execution
        OperatorTestCase: a named tuple to save the metadata of an test
        run_backward: a bool parameter indicating backward path
        op_name_function: a dictionary includes operator name and function
    """
    for config in configs:
        test_attrs = {}
        tags = None
        keep_config = True
        for attr in config:
            # tags is only used in our benchmark backend to filter tests and
            # it will be removed from config which is then passed to the init function
            # an example of config and atrr is:
            # config: [{'M': 16}, {'N': 16}, {'K': 64}, {'tags': 'short'}]
            # attr: {'tags': 'short'}
            if "tags" in attr:
                tags = attr["tags"]
                continue

            # if 'cuda' is specified in input shape but the testing machines doesn't
            # support, we will skip this input
            if "cuda" in attr.values():
                if not torch.cuda.is_available():
                    keep_config = False
                    break

            if "mps" in attr.values():
                if not torch.backends.mps.is_available():
                    keep_config = False
                    break

            # same for 'xpu': skip configs that target XPU on machines without XPU support
            if "xpu" in attr.values():
                if not torch.xpu.is_available():
                    keep_config = False
                    break

            test_attrs.update(attr)

        if not keep_config:
            continue

        if tags is None:
            raise ValueError("Missing tags in configs")

        op = bench_op()
        if op is None:
            raise AssertionError("Can't create test: bench_op() returned None")
        # op_name_function is a dictionary which has op_name and op_function.
        # an example of op_name_function is:
        # {'op_name' : 'abs', 'op_function' : torch.abs}
        # op_function is concatenated with the input dict then passed to the init function
        # op_name is passed to the set_module_name function
        init_dict = copy.deepcopy(test_attrs)
        if op_name_function is not None:
            op_name = op_name_function["op_name"]
            init_dict.update({"op_func": op_name_function["op_func"]})
            op.set_module_name(op_name)

        op._set_backward_test(run_backward)
        op.init(**init_dict)
        op.extract_inputs_tuple()

        if not run_backward:
            for attr in vars(op).values():
                if isinstance(attr, torch.nn.Module):
                    for param in attr.parameters():
                        param.requires_grad = False

        input_name = None

        # _num_inputs_require_grads is used to track the number of tensors
        # which use auto_set().
        if op._num_inputs_require_grads > 0:
            input_name = "all"
        yield _create_test(
            op, test_attrs, tags, OperatorTestCase, run_backward, input_name
        )

        # This for loop is only used when auto_set is used.
        # _pass_count counts how many times init has been called.
        # _auto_set_counter is reset after init is called.
        for i in range(op._num_inputs_require_grads):
            op._pass_count += 1
            op._auto_set_counter = 0

            # TODO(mingzhe09088): remove this deepcopy when we encounter
            # performance issue.
            new_op = copy.deepcopy(op)
            new_op.init(**init_dict)
            # Input name index will start from input1
            input_name = i + 1
            yield _create_test(
                new_op, test_attrs, tags, OperatorTestCase, run_backward, input_name
            )


class BenchmarkRunner:
    # [CLASS DOC] Drives the benchmark run: stores command line arguments and timing
    # parameters (starting iteration count 100 unless overridden, growth multiplier 2,
    # minimum 1 second per measurement, iteration cap one million), filters registered
    # tests, measures forward or backward execution, prints results and writes CSV/JSON
    # outputs. Eager, JIT and compile modes are selected by the arguments and are mutually
    # exclusive for JIT and compile. Main entry point is run.
    """BenchmarkRunner is responsible for benchmarking all the registered
    benchmark test groups.

    Attributes:
        tag_filter (str): control the benchmarks which matches the tag.
        operator (str): only run benchmark test cases that contains
    this filter string in the test case's id.
        test_name (str): only run benchmark test cases that matches this filter,
        this is a case-sensitive substring match and it happens in
        the _keep_test method.
    """

    def __init__(self, args):
        # [FUNCTION DOC] Initializes the runner from the parsed arguments: copies the settings
        # (jit, compile, number of runs, csv output, operator range via benchmark_utils), sets the
        # default warmup to 100 when the argument is -1, and uses the explicit iteration count
        # when it is set and not -1. When a specific test name is selected the tag filter is
        # cleared. Raises ValueError if both JIT and compile are requested.
        # TODO: consider time-bound constraints as well.
        self.args = args
        self.iters = 100
        self.has_explicit_iteration_count = False
        self.multiplier = 2
        self.predefined_minimum_secs = 1
        self.max_iters = 1e6
        self.use_jit = args.use_jit
        self.use_compile = args.use_compile
        if self.use_jit and self.use_compile:
            raise ValueError(
                "use_jit and use_compile are mutually exclusive, please specify one."
            )
        self.num_runs = args.num_runs
        self.print_per_iter = False
        self.output_csv = args.output_csv
        self.operator_range = benchmark_utils.get_operator_range(args.operator_range)
        # 100 is the default warmup iterations
        if self.args.warmup_iterations == -1:
            self.args.warmup_iterations = 100
        if self.args.iterations and self.args.iterations != -1:
            self.has_explicit_iteration_count = True
            self.iters = self.args.iterations
        # when a specific test is selected by a user, we don't need
        # to match the tag anymore
        if self.args.test_name is not None:
            self.args.tag_filter = None

    def _print_header(self):
        # [FUNCTION DOC] HELPER: Prints the banner for the run (separator lines, title
        # and the tag filter). In list-tests mode it prints a list heading; in
        # list-operators mode it prints a heading, creates the printed_ops_list set used
        # for de-duplication, and prints the requested operators if given.
        DASH_LINE = "-" * 40
        print(
            f"# {DASH_LINE}\n"
            "# PyTorch/Caffe2 Operator Micro-benchmarks\n"
            f"# {DASH_LINE}\n"
            f"# Tag : {self.args.tag_filter}\n"
        )
        if self.args.list_tests:
            print("# List of tests:")
        elif self.args.list_ops:
            print("# List of Operators to run:")
            self.printed_ops_list = set()
            if self.args.operators:
                print(f"# {self.args.operators}")

    def _print_perf_result(self, results, test_case):
        # [FUNCTION DOC] HELPER: Prints the result of one test case to stdout. With the
        # aibench report option it currently returns immediately and prints nothing.
        # KNOWN ISSUE: the aibench branch has an early return, so the code after it
        # is unreachable, and that dead code also indexes a string literal instead
        # of the results list (a bug if it were ever reached).
        # Otherwise prints the mode (JIT, Compile or Eager), test name and input config,
        # then either one line per run (when multiple runs) or the single execution time,
        # the peak memory and, when available, the memory bandwidth.
        if self.args.report_aibench:
            # Output for AIBench
            # Print out per iteration execution time instead of avg time
            return
            test_name = "_".join([test_case.framework, test_case.test_config.test_name])
            for run in range(self.num_runs):
                print(
                    f"{test_case.framework}Observer "
                    + json.dumps(
                        {
                            "type": test_name,
                            "metric": "latency",
                            "unit": "us",
                            "value": str(results["reported_run_time_us"[run]]),
                        }
                    )
                )
        else:
            print(
                f"# Mode: {'JIT' if self.use_jit else 'Compile' if self.use_compile else 'Eager'}"
            )
            print(
                f"# Name: {test_case.test_config.test_name}\n# Input: {test_case.test_config.input_config}"
            )

            mode = "Backward" if test_case.test_config.run_backward else "Forward"
            if self.num_runs > 1:
                for run in range(self.num_runs):
                    print(
                        f"Run: {run}, {mode} Execution Time (us) : {results['reported_run_time_us'][run]:.3f}"
                    )
                print()
            else:
                print(
                    f"{mode} Execution Time (us) : {results['reported_run_time_us'][0]:.3f}"
                )
                print(f"Peak Memory (KB) : {results['peak_memory']}")
                # Calculate and print memory bandwidth if operator provides memory traffic
                if results.get("memory_bandwidth_gb_s") is not None:
                    print(
                        f"Memory Bandwidth (GB/s) : {results['memory_bandwidth_gb_s']:.2f}"
                    )
                print()

    def _perf_result_to_dict(self, results, test_case):
        # [FUNCTION DOC] HELPER: Counterpart of the print routine that returns a dictionary
        # instead of printing: test name, input config, runtime mode, Forward or Backward,
        # latency of the first run rounded to 3 digits and its unit, peak memory and unit,
        # memory bandwidth and unit, plus one entry per input configuration key parsed out of
        # the config string. Returns an empty mapping with the aibench option. Raises
        # AssertionError if brackets in the config string are mismatched.
        """This function is the parallel of _print_perf_result, which instead of
        writing information to terminal, returns a dictionary.
        """
        if self.args.report_aibench:
            return {}

        out = {
            "test_name": test_case.test_config.test_name,
            "input_config": test_case.test_config.input_config,
            "runtime": (
                "JIT" if self.use_jit else "Compile" if self.use_compile else "Eager"
            ),
            "run": "Backward" if test_case.test_config.run_backward else "Forward",
            "latency": round(results["reported_run_time_us"][0], 3),
            "latency unit": "us",
            "peak memory": results["peak_memory"],
            "memory unit": "KB",
            "memory bandwidth": results.get("memory_bandwidth_gb_s"),
            "memory bandwidth unit": "GB/s",
        }

        # parsing test_case.test_config.input_config, adding it as entries to the 'out' dictionary
        # input: 'M: 1, N: 1, K: 1, device: cpu'
        # output: {'M':'1', 'N':'1', 'K':'1', 'device': 'cpu'}
        # splitting the string on unnested commas
        def split(s):
            # [FUNCTION DOC] HELPER: Splits a string on commas that are not nested inside
            # brackets, braces or parentheses. Returns the list of pieces; raises
            # AssertionError on a closing bracket that does not match the innermost open one.
            open_to_close = {"{": "}", "(": ")", "[": "]"}
            break_idxs = [-1]
            curr_brackets = []
            for i, c in enumerate(s):
                if c in open_to_close:
                    curr_brackets.append(c)
                elif c in open_to_close.values():
                    if not curr_brackets or open_to_close[curr_brackets[-1]] != c:
                        raise AssertionError(
                            f"ERROR: not able to parse the string! Mismatched bracket '{c}'"
                        )
                    curr_brackets.pop()
                elif c == "," and (not curr_brackets):
                    break_idxs.append(i)
            break_idxs.append(len(s))
            out = []
            for i in range(len(break_idxs) - 1):
                start, end = break_idxs[i], break_idxs[i + 1]
                out.append(s[start + 1 : end])
            return out

        key_vals = split(
            test_case.test_config.input_config
        )  # 'M: [(32, 16), (64, 32)], ZPB: 2' -> ['M: [(32, 16), (64, 32)]', 'ZPB: 2']
        key_vals = [
            (key.strip(), value.strip())
            for key, value in map(lambda str: str.split(":"), key_vals)  # noqa: C417
        ]  # ['M: (32, 16)', 'ZPB: 2'] -> [('M', '(32, 16)'), ('ZPB', '2')]
        out.update(key_vals)

        return out

    def _predict_num_iter_needed(self, i):
        # [FUNCTION DOC] HELPER: Returns the next iteration count to try, the
        # given count multiplied by the runner's growth multiplier.
        return i * self.multiplier

    def _iteration_result_is_significant(
        self, iters, run_time_sec, curr_test_total_time, has_explicit_iteration_count
    ):
        # [FUNCTION DOC] HELPER: Decides whether a measurement can be reported. True when
        # the iteration count exceeds the cap, or the last run took longer than the
        # predefined minimum, or an explicit iteration count was given, and in addition the
        # accumulated test time exceeds the user's minimum time per test.
        """This function decides whether the measured time can be reported based on the
        following conditions: 1) the number of iterations is larger than the max_iters.
        2) the execution time is larger than the predefined minimum_time
        3) the execution time is larger than user defined minimum_time
        """
        return (
            iters > self.max_iters
            or run_time_sec > self.predefined_minimum_secs
            or has_explicit_iteration_count
        ) and curr_test_total_time > self.args.min_time_per_test

    def _launch_forward(self, test_case, iters, print_per_iter):
        # [FUNCTION DOC] HELPER: Measures the forward pass in seconds for the given iteration
        # count. Selects the eager, JIT or compile forward function of the test case. GPU
        # synchronization is enabled when the test name mentions cuda or xpu. Without GPU sync it
        # times one call with timeit; with GPU sync it uses an adaptive autorange Timer and
        # returns the median per-call time multiplied by the iteration count.
        """Use Python's timeit module to measure execution time (unit: second)."""
        test_name = test_case.test_config.test_name
        gpu_sync = ("cuda" in test_name) or ("xpu" in test_name)
        func = test_case.run_forward
        if self.use_jit:
            func = test_case.run_jit_forward
        if self.use_compile:
            func = test_case.run_compile_forward

        if not gpu_sync:
            forward_time = timeit.timeit(
                functools.partial(func, iters, print_per_iter, gpu_sync), number=1
            )
            return forward_time
        timer = Timer(
            stmt="func(iters, print_per_iter, gpu_sync)",
            globals={
                "func": func,
                "iters": iters,
                "print_per_iter": print_per_iter,
                "gpu_sync": gpu_sync,
            },
        )
        result = timer.adaptive_autorange(min_run_time=0.0001)
        return result.median * iters

    def _launch_backward(self, test_case, iters, print_per_iter=False):
        # [FUNCTION DOC] HELPER: Measures the backward pass in seconds. First runs the forward
        # once and has the test case reduce its output to a mean, then times the test case
        # backward run with an adaptive autorange Timer and returns the median multiplied by the
        # iteration count. GPU sync is used when the test name mentions cuda or xpu.
        """This function runs forward path of an op to get an output. Then the backward path is executed
        and the execution time is reported
        """
        test_name = test_case.test_config.test_name
        gpu_sync = ("cuda" in test_name) or ("xpu" in test_name)
        test_case.run_forward(num_runs=1, print_per_iter=False, gpu_sync=gpu_sync)
        test_case._output_mean()

        timer = Timer(
            stmt="test_case.run_backward(iters, print_per_iter, gpu_sync)",
            globals={
                "test_case": test_case,
                "iters": iters,
                "print_per_iter": print_per_iter,
                "gpu_sync": gpu_sync,
            },
        )
        result = timer.adaptive_autorange(min_run_time=0.0001)
        return result.median * iters

    def _measure_metrics(self, launch_test, test_case, iters, print_per_iter):
        # [FUNCTION DOC] HELPER: Repeatedly calls the given launch function, growing the iteration
        # count until the result is significant. Each pass resets peak memory stats and
        # synchronizes on the input tensor's device module when the first input is a tensor
        # (memory is not tracked for CPU or non-tensor inputs), records peak allocated memory and
        # appends the per-iteration time in microseconds. With the aibench option prints an
        # observer JSON line per pass. Returns the median per-iteration time in microseconds and
        # the peak memory in KB.
        """
        This function execute the operator for <iters> iterations then look at the time.
        If it's not significant, the number of iterations will be increased before rerun.
        The execution stops when the time becomes significant.
        """
        curr_test_total_time = 0
        time_trace = []
        peak_memory = 0
        input_values = test_case.op_bench.inputs.values()
        device, device_module = None, None
        if input_values and isinstance(next(iter(input_values)), torch.Tensor):
            # The device and device module information are crucial for memory metric calculation,
            # In case of ops where inputs are integers (not tensor), memory metrics need not be calculated.
            sample_input = next(iter(input_values))
            device = sample_input.device
            device_module = torch.get_device_module(device.type)
        # TODO: add support for cpu memory measurement
        while True:
            if device_module is not None:
                if hasattr(device_module, "reset_peak_memory_stats"):
                    device_module.reset_peak_memory_stats(device)
            run_time_sec = launch_test(test_case, iters, print_per_iter)
            if device_module is not None:
                device_module.synchronize()
            # Memory measurement process
            if device_module is not None:
                if hasattr(device_module, "max_memory_allocated"):
                    peak_memory = device_module.max_memory_allocated(device)
            curr_test_total_time += run_time_sec
            # Analyze time after each run to decide if the result is stable
            results_are_significant = self._iteration_result_is_significant(
                iters,
                run_time_sec,
                curr_test_total_time,
                self.has_explicit_iteration_count,
            )

            report_run_time = 1e6 * run_time_sec / iters
            time_trace.append(report_run_time)
            # Print out the time spent in each epoch in ms
            if self.args.report_aibench:
                mode = (
                    "JIT"
                    if self.use_jit
                    else "Compile"
                    if self.use_compile
                    else "Eager"
                )
                test_name = "_".join(
                    [test_case.framework, test_case.test_config.test_name, mode]
                )
                print(
                    "PyTorchObserver "
                    + json.dumps(
                        {
                            "type": test_name,
                            "metric": "latency",
                            "unit": "ms",
                            "value": str(report_run_time / 1e3),
                        },
                    )
                )
            if results_are_significant:
                break

            # Re-estimate the hopefully-sufficient
            # iteration count, and run the benchmark again...
            iters = self._predict_num_iter_needed(iters)
        reported_run_time_us = np.percentile(np.array(time_trace), 50)
        return reported_run_time_us, peak_memory / 1024

    def _check_keep(self, test_flag, cmd_flag):
        # [FUNCTION DOC] HELPER: Filter predicate that is true when no
        # command line value is given or the test value equals it.
        return cmd_flag is None or test_flag == cmd_flag

    def _check_operator_first_char(self, test_flag, cmd_flag):
        # [FUNCTION DOC] HELPER: Filter predicate that is true when no command
        # line value is given or the lower-cased first character of the operator
        # name is contained in the given range of characters.
        return cmd_flag is None or test_flag[:1].lower() in cmd_flag

    def _check_keep_list(self, test_flag, cmd_flag_list):
        # [FUNCTION DOC] HELPER: Filter predicate that is true when no list
        # is given or the test value equals any entry of the list.
        return cmd_flag_list is None or any(
            test_flag == cmd_flag for cmd_flag in cmd_flag_list
        )

    def _check_skip(self, test_module, cmd_flag):
        # [FUNCTION DOC] HELPER: Filter predicate that is true when no
        # skip list is given or the module name is not in it.
        return cmd_flag is None or (test_module not in cmd_flag)

    def _keep_test(self, test_case):
        # [FUNCTION DOC] HELPER: Returns whether a test case passes all command line filters:
        # exact test name, operator list (parsed with benchmark_utils), not in the skip list,
        # first letter within the operator range, tag (unless the tag filter is all), forward-only
        # handling, and device (only when the input config mentions a device). The code comment
        # mentions substring matching although most filters here are exact comparisons.
        # TODO: consider regex matching for test filtering.
        # Currently, this is a sub-string matching.
        op_test_config = test_case.test_config

        operators = (
            benchmark_utils.process_arg_list(self.args.operators)
            if self.args.operators
            else None
        )

        # Filter framework, operator, test_name, tag, forward_only
        return (
            self._check_keep(op_test_config.test_name, self.args.test_name)
            and self._check_keep_list(test_case.op_bench.module_name(), operators)
            and self._check_skip(test_case.op_bench.module_name(), SKIP_OP_LISTS)
            and self._check_operator_first_char(
                test_case.op_bench.module_name(), self.operator_range
            )
            and (
                self.args.tag_filter == "all"
                or self._check_keep(op_test_config.tag, self.args.tag_filter)
            )
            and (
                not self.args.forward_only
                or op_test_config.run_backward != self.args.forward_only
            )
            and (
                self.args.device == "None"
                or "device" not in test_case.test_config.input_config
                or self.args.device in op_test_config.test_name
            )
        )

    def _print_test_case_info(self, test_case):
        # [FUNCTION DOC] HELPER: Handles the list modes. With list-tests prints the test
        # name and returns True. With list-operators prints each operator module name once
        # (only when no operators filter is set) using the printed_ops_list set, and returns
        # True. Otherwise returns False so the test is actually run.
        # Print out the test name and skip the real execution
        if self.args.list_tests:
            print(f"# {test_case.test_config.test_name}")
            return True
        elif self.args.list_ops:
            if self.args.operators is None:
                op_name = test_case.op_bench.module_name()

                if op_name not in self.printed_ops_list:
                    print(f"# {op_name}")
                    self.printed_ops_list.add(op_name)
            return True

        return False

    def _output_csv(self, filename, headers, row):
        # [FUNCTION DOC] HELPER: Appends a row to a CSV file, creating it if needed. If the file
        # exists its rows are read, and the header row is replaced by the new headers when those
        # are longer (an earlier failed run may have left it incomplete), otherwise the existing
        # header is kept. Floats are formatted with six decimals, every row is padded with the
        # string zero up to the header width, and the whole file is rewritten with newline line
        # endings.
        if os.path.exists(filename):
            with open(filename) as fd:
                lines = list(csv.reader(fd)) or [[]]
                if headers and len(headers) > len(lines[0]):
                    # if prior results failed the header might not be filled in yet
                    lines[0] = headers
                else:
                    headers = lines[0]
        else:
            lines = [headers]
        lines.append([(f"{x:.6f}" if isinstance(x, float) else x) for x in row])
        with open(filename, "w") as fd:
            writer = csv.writer(fd, lineterminator="\n")
            for line in lines:
                writer.writerow(list(line) + ["0"] * (len(headers) - len(line)))

    def _output_json(
        self,
        perf_list,
        output_file,
        benchmark_name="PyTorch operator benchmark",
    ):
        # [FUNCTION DOC] HELPER: Writes results in the benchmark-database JSON format. Does
        # nothing for an empty list. For every result entry it builds three records (latency, peak
        # memory and memory bandwidth), each with benchmark info (name, mode inference or training
        # from Forward or Backward, dtype taken from the part after the dot, device, device
        # architecture, compile flag, operator name taken from the test name prefix), model info
        # and metric info, using local dataclasses. The list of records is written to the output
        # file with indentation. Raises an error if the dtype string has no dot.
        """
        Write the result into JSON format, so that it can be uploaded to the benchmark database
        to be displayed on OSS dashboard. The JSON format is defined at
        https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
        """
        if not perf_list:
            return

        # Prepare headers and records for JSON output
        records = []
        for perf_item in perf_list:
            # Extract data from perf_item
            test_name = perf_item.get("test_name", "unknown")
            input_config = perf_item.get("input_config", "")
            run_type = perf_item.get("run")
            latency = perf_item.get("latency", 0)
            peak_memory = perf_item.get("peak memory", 0)
            memory_bandwidth = perf_item.get("memory bandwidth", 0)
            device = perf_item.get("device", "unknown")
            dtype = perf_item.get("dtype", "torch.float").split(".")[1]
            runtime = perf_item.get("runtime", None)

            # Extract mode based on run_type
            mode = None
            if run_type == "Forward":
                mode = "inference"
            elif run_type == "Backward":
                mode = "training"

            # Extract use_compile from it
            if runtime == "Compile":
                use_compile = True
            elif runtime == "Eager":
                use_compile = False
            else:
                use_compile = None

            device_arch = (
                torch.cuda.get_device_name(0)
                if device == "cuda"
                else platform.processor()
                if device == "cpu"
                else "unknown"
            )

            # Extract operator name from test_name
            operator_name = test_name.split("_")[0]

            # Create the record
            @dataclass
            class BenchmarkInfo:
                # [CLASS DOC] HELPER: Local record for the benchmark part of a database entry:
                # name, mode (may be absent), dtype and a free-form extra info mapping.
                name: str
                mode: str | None
                dtype: str
                extra_info: dict[str, Any]

            @dataclass
            class ModelInfo:
                # [CLASS DOC] HELPER: Local record for the model part of a database
                # entry: name, type, list of origins and an extra info mapping.
                name: str
                type: str
                origins: list[str]
                extra_info: dict[str, Any]

            @dataclass
            class MetricInfo:
                # [CLASS DOC] HELPER: Local record for a metric: name, unit,
                # list of measured values and an optional target value.
                name: str
                unit: str
                benchmark_values: list[float]
                target_value: float | None

            @dataclass
            class BenchmarkRecord:
                # [CLASS DOC] HELPER: Local record combining the benchmark, model and metric parts
                # of one database entry; converted to a plain mapping when written out.
                benchmark: BenchmarkInfo
                model: ModelInfo
                metric: MetricInfo

            # Add record for latency
            record_latency = BenchmarkRecord(
                benchmark=BenchmarkInfo(
                    name=benchmark_name,
                    mode=mode,
                    dtype=dtype,
                    extra_info={
                        "input_config": input_config,
                        "device": device,
                        "arch": device_arch,
                        "use_compile": use_compile,
                        "operator_name": operator_name,
                    },
                ),
                model=ModelInfo(
                    name=test_name,
                    type="micro-benchmark",
                    origins=["pytorch"],
                    extra_info={"operator_name": operator_name},
                ),
                metric=MetricInfo(
                    name="latency",
                    unit="us",
                    benchmark_values=[latency],
                    target_value=None,
                ),
            )
            records.append(asdict(record_latency))

            # Add record for peak memory
            record_memory = copy.deepcopy(record_latency)
            record_memory.metric = MetricInfo(
                name="peak memory",
                unit="KB",
                benchmark_values=[peak_memory],
                target_value=None,
            )
            records.append(asdict(record_memory))

            # Add record for memory bandwidth
            record_memory_bandwidth = copy.deepcopy(record_latency)
            record_memory_bandwidth.metric = MetricInfo(
                name="memory bandwidth",
                unit="GB/s",
                benchmark_values=[memory_bandwidth],
                target_value=None,
            )
            records.append(asdict(record_memory_bandwidth))

        # Write all records to the output file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    def run(self):
        # [FUNCTION DOC] Main loop of the runner. Prints the header, then for every registered
        # definition builds test cases and, for each one that is not just being listed and
        # passes the filters, seeds numpy from the test id hash (32-bit), prints the benchmark
        # name, runs a warmup with the forward or backward launcher, then measures the
        # configured number of runs. It computes memory bandwidth from the op's reported memory
        # traffic when available, prints the result, appends a row to the CSV output, and
        # collects dictionaries when JSON output is requested. At the end writes the dashboard
        # JSON and/or plain JSON files. Returns nothing.
        # KNOWN ISSUE: the seed comes from the hash of a string test id, and string hashes
        # are randomized per process unless the hash seed environment variable is fixed, so
        # the input data is not reproducible across runs despite the comment.
        self._print_header()
        output_csv_filename = self.args.output_csv
        headers = [
            "Benchmarking Framework",
            "Benchmarking Module Name",
            "Case Name",
            "tag",
            "run_backward",
            "Execution Time",
            "Peak Memory (KB)",
            "Memory Bandwidth (GB/s)",
        ]

        if self.args.output_json or self.args.output_json_for_dashboard:
            perf_list = []

        for test_metainfo in BENCHMARK_TESTER:
            for test in _build_test(*test_metainfo):
                full_test_id, test_case = test
                op_test_config = test_case.test_config

                if self._print_test_case_info(test_case):
                    continue

                if not self._keep_test(test_case):
                    continue

                # To reduce variance, fix a numpy randseed to the test case,
                # so that the randomly generated input tensors remain the
                # same for each test case.
                # The random seed is limited to 32-bit because of numpy
                # requirement.
                np.random.seed(seed=hash(full_test_id) & ((1 << 32) - 1))

                print(
                    f"# Benchmarking {test_case.framework}: {test_case.op_bench.module_name()}"
                )

                if op_test_config.run_backward:
                    launch_func = self._launch_backward
                else:
                    launch_func = self._launch_forward

                # Warmup
                launch_func(
                    test_case, self.args.warmup_iterations, print_per_iter=False
                )
                # Actual Execution
                results = [
                    self._measure_metrics(
                        launch_func, test_case, self.iters, self.print_per_iter
                    )
                    for _ in range(self.num_runs)
                ]
                result_dict = dict()
                result_dict["reported_run_time_us"] = [r[0] for r in results]
                result_dict["peak_memory"] = results[0][1]

                # Calculate memory bandwidth if operator provides memory traffic
                memory_traffic_bytes = test_case.op_bench.get_memory_traffic_bytes()
                if memory_traffic_bytes is not None:
                    execution_time_s = result_dict["reported_run_time_us"][0] / 1e6
                    result_dict["memory_bandwidth_gb_s"] = (
                        memory_traffic_bytes / execution_time_s / 1e9
                    )
                else:
                    result_dict["memory_bandwidth_gb_s"] = None

                self._print_perf_result(results=result_dict, test_case=test_case)

                # output results to csv
                self._output_csv(
                    output_csv_filename,
                    headers,
                    [
                        test_case.framework,
                        test_case.op_bench.module_name(),
                        (
                            test_case.test_config.test_name + "_BACKWARD"
                            if test_case.test_config.run_backward is True
                            else test_case.test_config.test_name
                        ),
                        test_case.test_config.tag,
                        test_case.test_config.run_backward,
                        result_dict["reported_run_time_us"][0],
                        result_dict["peak_memory"],
                        result_dict["memory_bandwidth_gb_s"],
                    ],
                )
                if self.args.output_json or self.args.output_json_for_dashboard:
                    perf_list.append(self._perf_result_to_dict(result_dict, test_case))

        if self.args.output_json_for_dashboard:
            self._output_json(
                perf_list, self.args.output_json_for_dashboard, self.args.benchmark_name
            )

        if self.args.output_json:
            with open(self.args.output_json, "w") as f:
                json.dump(perf_list, f)
