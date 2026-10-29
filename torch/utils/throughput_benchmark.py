# mypy: allow-untyped-defs
# [MODULE DOC] Python wrapper around the C++ throughput benchmark. Provides ThroughputBenchmark
# (drive a module with multiple calling threads and stored inputs), ExecutionStats (derived
# latency and throughput numbers) and format_time. Importing it loads torch._C only.

import torch._C


def format_time(time_us=None, time_ms=None, time_s=None) -> str:
    # [FUNCTION DOC] Format a duration given in exactly one unit (microseconds, milliseconds or
    # seconds) as a string with three decimals and the largest fitting suffix (s, ms or us).
    # Raises AssertionError unless exactly one argument is provided.
    """Define time formatting."""
    if sum([time_us is not None, time_ms is not None, time_s is not None]) != 1:
        raise AssertionError("Expected only one of time_us, time_ms, time_s is given.")

    US_IN_SECOND = 1e6
    US_IN_MS = 1e3

    if time_us is None:
        if time_ms is not None:
            time_us = time_ms * US_IN_MS
        elif time_s is not None:
            time_us = time_s * US_IN_SECOND
        else:
            raise AssertionError("Shouldn't reach here :)")

    if time_us >= US_IN_SECOND:
        return f'{time_us / US_IN_SECOND:.3f}s'
    if time_us >= US_IN_MS:
        return f'{time_us / US_IN_MS:.3f}ms'
    return f'{time_us:.3f}us'


class ExecutionStats:
    # [CLASS DOC] Read-only view over the C++ benchmark result plus the config that produced it.
    # Holds the native stats object and the benchmark config; exposes average latency, iteration
    # count and derived total time and iterations per second. Printing gives a readable report.
    def __init__(self, c_stats, benchmark_config) -> None:
        # [FUNCTION DOC] Store the native stats object and the benchmark config.
        self._c_stats = c_stats
        self.benchmark_config = benchmark_config

    @property
    def latency_avg_ms(self):
        # [FUNCTION DOC] Average latency per input example in milliseconds, from native stats.
        return self._c_stats.latency_avg_ms

    @property
    def num_iters(self):
        # [FUNCTION DOC] Number of iterations actually executed across all threads.
        return self._c_stats.num_iters

    @property
    def iters_per_second(self):
        # [FUNCTION DOC] Iterations divided by the derived total time, over all calling threads.
        """Return total number of iterations per second across all calling threads."""
        return self.num_iters / self.total_time_seconds

    @property
    def total_time_seconds(self):
        # [FUNCTION DOC] Estimated wall time in seconds: iterations times average latency,
        # divided by the number of calling threads (threads run concurrently).
        return self.num_iters * (
            self.latency_avg_ms / 1000.0) / self.benchmark_config.num_calling_threads

    def __str__(self) -> str:
        # [FUNCTION DOC] Multi-line report: average latency, iteration count, iterations per
        # second and total time, with times formatted by format_time.
        return '\n'.join([
            "Average latency per example: " + format_time(time_ms=self.latency_avg_ms),
            f"Total number of iterations: {self.num_iters}",
            f"Total number of iterations per second (across all threads): {self.iters_per_second:.2f}",
            "Total time: " + format_time(time_s=self.total_time_seconds)
        ])


class ThroughputBenchmark:
    # [CLASS DOC] Wrapper around the native benchmark object (kept in a private attribute).
    # Inputs are added once and stored natively; benchmark runs them from several threads
    # and returns ExecutionStats. Accepts eager modules and scripted modules.
    """
    This class is a wrapper around a c++ component throughput_benchmark::ThroughputBenchmark.

    This wrapper on the throughput_benchmark::ThroughputBenchmark component is responsible
    for executing a PyTorch module (nn.Module or ScriptModule) under an inference
    server like load. It can emulate multiple calling threads to a single module
    provided. In the future we plan to enhance this component to support inter and
    intra-op parallelism as well as multiple models running in a single process.

    Please note that even though nn.Module is supported, it might incur an overhead
    from the need to hold GIL every time we execute Python code or pass around
    inputs as Python objects. As soon as you have a ScriptModule version of your
    model for inference deployment it is better to switch to using it in this
    benchmark.

    Example::

        >>> # xdoctest: +SKIP("undefined vars")
        >>> from torch.utils import ThroughputBenchmark
        >>> bench = ThroughputBenchmark(my_module)
        >>> # Pre-populate benchmark's data set with the inputs
        >>> for input in inputs:
        ...     # Both args and kwargs work, same as any PyTorch Module / ScriptModule
        ...     bench.add_input(input[0], x2=input[1])
        >>> # Inputs supplied above are randomly used during the execution
        >>> stats = bench.benchmark(
        ...     num_calling_threads=4,
        ...     num_warmup_iters = 100,
        ...     num_iters = 1000,
        ... )
        >>> print("Avg latency (ms): {}".format(stats.latency_avg_ms))
        >>> print("Number of iterations: {}".format(stats.num_iters))
    """

    def __init__(self, module) -> None:
        # [FUNCTION DOC] Create the native benchmark for the module; a ScriptModule is passed
        # via its underlying C++ module, any other module is passed as is.
        if isinstance(module, torch.jit.ScriptModule):
            self._benchmark = torch._C.ThroughputBenchmark(module._c)
        else:
            self._benchmark = torch._C.ThroughputBenchmark(module)

    def run_once(self, *args, **kwargs):
        # [FUNCTION DOC] Run the module once on a stored input and return its output.
        # Arguments are forwarded to the native run_once (docstring says input id; the native
        # call decides what the arguments mean). Useful as a sanity check.
        """
        Given input id (input_idx) run benchmark once and return prediction.

        This is useful for testing that benchmark actually runs the module you
        want it to run. input_idx here is an index into inputs array populated
        by calling add_input() method.
        """
        return self._benchmark.run_once(*args, **kwargs)

    def add_input(self, *args, **kwargs) -> None:
        # [FUNCTION DOC] Store one set of module arguments in the native benchmark. During the
        # run, threads pick randomly among all stored inputs. Returns nothing.
        """
        Store a single input to a module into the benchmark memory and keep it there.

        During the benchmark execution every thread is going to pick up a
        random input from all the inputs ever supplied to the benchmark via
        this function.
        """
        self._benchmark.add_input(*args, **kwargs)

    def benchmark(
            self,
            num_calling_threads=1,
            num_warmup_iters=10,
            num_iters=100,
            profiler_output_path=""):
        # [FUNCTION DOC] Build a native config from the arguments (threads, warmup iterations
        # per thread, total iterations, optional profiler trace path), run the benchmark and
        # return ExecutionStats. A non-empty path enables the autograd profiler during the
        # main run and writes a trace file there (file I/O side effect).
        # KNOWN ISSUE: the docstring omits num_calling_threads from the documented arguments.
        """
        Run a benchmark on the module.

        Args:
            num_warmup_iters (int): Warmup iters are used to make sure we run a module
                a few times before actually measuring things. This way we avoid cold
                caches and any other similar problems. This is the number of warmup
                iterations for each of the threads separately

            num_iters (int): Number of iterations the benchmark should run with.
                This number is separate from the warmup iterations. Also the number is
                shared across all the threads. Once the num_iters iterations across all
                the threads is reached, we will stop execution. Though total number of
                iterations might be slightly larger. Which is reported as
                stats.num_iters where stats is the result of this function

            profiler_output_path (str): Location to save Autograd Profiler trace.
                If not empty, Autograd Profiler will be enabled for the main benchmark
                execution (but not the warmup phase). The full trace will be saved
                into the file path provided by this argument


        This function returns BenchmarkExecutionStats object which is defined via pybind11.
        It currently has two fields:
            - num_iters - number of actual iterations the benchmark have made
            - avg_latency_ms - average time it took to infer on one input example in milliseconds
        """
        config = torch._C.BenchmarkConfig()
        config.num_calling_threads = num_calling_threads
        config.num_warmup_iters = num_warmup_iters
        config.num_iters = num_iters
        config.profiler_output_path = profiler_output_path
        c_stats = self._benchmark.benchmark(config)
        return ExecutionStats(c_stats, config)
