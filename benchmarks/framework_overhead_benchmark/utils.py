# [MODULE DOC] Shared constants, config records and timing helpers for the framework overhead
# benchmark: the inner loop count, the BenchmarkConfig and ModuleConfig records, unit conversion
# helpers and two ways to measure the latency per operation.
import time
from collections import namedtuple

from torch.utils import ThroughputBenchmark


NUM_LOOP_ITERS = 1000
BenchmarkConfig = namedtuple("BenchmarkConfig", "num_warmup_iters num_iters")
ModuleConfig = namedtuple("ModuleConfig", "pt_fn c2_op num_params graph_mode")


def ms_to_us(time_ms):
    # [FUNCTION DOC] HELPER: Convert milliseconds to microseconds (multiply by one thousand).
    return time_ms * 1e3


def secs_to_us(time_s):
    # [FUNCTION DOC] HELPER: Convert seconds to microseconds (multiply by one million).
    return time_s * 1e6


def secs_to_ms(time_s):
    # [FUNCTION DOC] HELPER: Convert seconds to milliseconds (multiply by one thousand).
    return time_s * 1e3


def benchmark_using_throughput_benchmark(config, module):
    # [FUNCTION DOC] Measure with the ThroughputBenchmark utility: wrap the inner module, add the
    # input tensors, run one thread with the configured warmup and measured iteration counts, and
    # return the average latency in milliseconds divided by the inner loop count (per addition).
    # Prints a message.
    print("Benchmarking via ThroughputBenchmark")
    bench = ThroughputBenchmark(module.module)
    bench.add_input(*module.tensor_inputs)
    stats = bench.benchmark(1, config.num_warmup_iters, config.num_iters)
    return stats.latency_avg_ms / NUM_LOOP_ITERS


def benchmark_module(config, module, use_throughput_benchmark=False):
    # [FUNCTION DOC] Measure latency per operation in milliseconds. If use_throughput_benchmark
    # is true, delegate to the ThroughputBenchmark path. Otherwise run warmup iterations, time
    # the measured iterations with wall clock time, and return elapsed milliseconds divided by
    # iteration count and by the inner loop count. Prints a status line.
    if use_throughput_benchmark:
        return benchmark_using_throughput_benchmark(config, module)
    module.forward(config.num_warmup_iters)
    print(f"Running module for {config.num_iters} iterations")
    start = time.time()
    module.forward(config.num_iters)
    end = time.time()
    time_elapsed_s = end - start
    return secs_to_ms(time_elapsed_s) / config.num_iters / NUM_LOOP_ITERS
