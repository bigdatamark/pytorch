# [MODULE DOC] Entry point of the framework overhead benchmark. Parses flags (op, eager mode,
# throughput benchmark, debug, save, warmup and measurement iteration counts), builds the
# benchmark configuration, runs the add op benchmark in graph mode (traced) or eager mode and
# prints the latency per iteration in microseconds. Only forward passes of the add op are
# supported.
import argparse

from pt_wrapper_module import WrapperModule
from SimpleAddModule import add_tensors_loop, SimpleAddModule

from utils import benchmark_module, BenchmarkConfig, ModuleConfig, ms_to_us


""" Framework overhead benchmark script.
Benchmark framework overhead.
Currently supported ops: add.
As of now runs only forward pass.
Supports both graph mode and eager mode. In graph mode the module is traced via JIT tracing.
Debug option prints the traced graph is graph_mode is enabled.
Graph can be saved via save option. Saved in the directory where benchmark is run.
Example build/run:
To run PT benchmark:
buck run @mode/opt <path-to-framework_overhead_benchmark>:framework_overhead_benchmark --
 --add-op --graph-mode --eager-mode (Runs both graph mode and eager mode)
buck run @mode/opt <path-to-framework_overhead_benchmark>:framework_overhead_benchmark --
 --add-op --graph-mode (Runs only graph mode)
"""

SUPPORTED_OPS = {"add_op"}


def parse_op_args(op):
    # [FUNCTION DOC] HELPER: Split the op string on commas into a list that is then discarded; no
    # value is returned and nothing else happens (unused placeholder).
    op_list = op.split(",")  # noqa: F841


def print_results(result):
    # [FUNCTION DOC] Print a banner, then for every key and latency in the result mapping print
    # one line with the key and the value converted from milliseconds to microseconds, then a
    # banner.
    print("===================================")
    for key, value in result.items():
        print(f"{key}, latency per iter (us):{ms_to_us(value)}")
    print("===================================")


def benchmark_simple_fn(args, config, module_config, module_type, result):
    # [FUNCTION DOC] Benchmark one module type. Wraps module_type in a WrapperModule using the
    # module config and the debug and save flags, forms a result key from the function name, the
    # operand count and the graph mode flag, times it with benchmark_module (optionally with the
    # throughput benchmark) and stores the per iteration latency in milliseconds in the result
    # mapping under that key. Prints a status line. Returns None.
    """Benchmarks a PyTorch traceable function specified in the config.
    Instantiates a wrapper object that wraps the object of module_type and runs the forward
    method using benchmark_module.
    Args:
        config:         contains number of warmup and benchmark iterations.
        module_config:  module_config which contains op, number of parameters that op takes
                    and whether graph mode is enabled or not.
        module_type:    Type of the module to be wrapped. e.g. SimpleAddModule for add op.
        result:         dictionary instance to be populated with the benchmark result (latency per iter).
    """
    print(f"Benchmarking {module_type.__name__}")
    f_name = (
        module_config.pt_fn.__name__ + ":Num Operands=" + str(module_config.num_params)
    )
    graph_mode_str = "Graph mode" + ":" + str(module_config.graph_mode)
    result_key = ",".join((f_name, graph_mode_str))
    module = WrapperModule(module_type, module_config, args.debug, args.save)
    latency_per_iter_ms = benchmark_module(
        config, module, args.use_throughput_benchmark
    )
    result[result_key] = latency_per_iter_ms


def main():
    # [FUNCTION DOC] Command line entry. Parses arguments; if the op is not in the supported set
    # it prints a message and returns without running. Otherwise builds a benchmark config from
    # the iteration counts, graph mode on unless eager mode is requested, and for the add op runs
    # the benchmark with two operands, then prints the results.
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", default="add_op", dest="op", type=str)
    parser.add_argument(
        "--use-throughput-benchmark",
        "--use_throughput_benchmark",
        default=False,
        dest="use_throughput_benchmark",
        action="store_true",
    )
    parser.add_argument("--debug", default=False, dest="debug", action="store_true")
    parser.add_argument("--save", default=False, dest="save", action="store_true")
    parser.add_argument(
        "--eager-mode",
        "--eager_mode",
        default=False,
        dest="eager_mode",
        action="store_true",
    )
    parser.add_argument(
        "--num-warmup-iters", "--num_warmup_iters", type=int, default=100
    )
    parser.add_argument("--num-iters", "--num_iters", type=int, default=1000)
    args = parser.parse_args()

    if args.op not in SUPPORTED_OPS:
        print(f"Op {args.op} is not supported: Supported ops are:{SUPPORTED_OPS}")
        return
    num_warmup_iters = args.num_warmup_iters
    num_iters = args.num_iters
    config = BenchmarkConfig(num_warmup_iters, num_iters)
    graph_mode = True
    if args.eager_mode:
        graph_mode = False
    result = {}
    if args.op == "add_op":
        num_params = 2
        module_config = ModuleConfig(add_tensors_loop, None, num_params, graph_mode)
        benchmark_simple_fn(args, config, module_config, SimpleAddModule, result)
    print_results(result)


if __name__ == "__main__":
    main()
