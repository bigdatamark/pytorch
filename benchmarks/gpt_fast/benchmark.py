# [MODULE DOC] Micro-benchmark suite for the gpt-fast style workloads (a MLP with layer norm,
# layer norm alone, gather-gemv and gemv) compiled with torch.compile. Each benchmark is
# registered in the shared experiment registry from common and returns a list of Experiment
# records holding a target and a measured value. The main function runs all or one registered
# experiment and appends each result to a CSV file and to a JSON lines file next to it, and a
# command line entry point exposes the output path and a single-experiment filter. Import-time
# side effects: registers four experiments in the global registry.
import argparse
import csv
import dataclasses
import json
import os

from common import all_experiments, Experiment, register_experiment
from generate import get_arch_name

import torch
import torch.nn as nn
from torch._inductor.runtime.benchmarking import benchmarker
from torch.utils.flop_counter import FlopCounterMode


WARMUP_ITER = 5

A100_40G_BF16_TFLOPS = 312


class SimpleMLP(nn.Module):
    # [CLASS DOC] Small nn.Module used by the MLP benchmark: a module list of a linear layer
    # (input to hidden), a layer norm over the hidden size, a second linear layer (hidden to
    # output) and a layer norm over the output size, all created in the given dtype. The forward
    # pass applies the layers in order.
    def __init__(self, input_dim, hidden_dim, output_dim, dtype):
        # [FUNCTION DOC] HELPER: Build the four layers (two linear, two layer norm) in the given
        # dtype and store them in a module list named layers.
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim, hidden_dim, dtype=dtype),
                nn.LayerNorm(hidden_dim, dtype=dtype),
                nn.Linear(hidden_dim, output_dim, dtype=dtype),
                nn.LayerNorm(output_dim, dtype=dtype),
            ]
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Apply every layer in sequence to the input and return
        # the result.
        for layer in self.layers:
            x = layer(x)
        return x


@register_experiment(name="mlp_layer_norm_gelu")
def run_mlp_layer_norm_gelu(device: str = "cuda"):
    # [FUNCTION DOC] Benchmark a MLP-with-layer-norm module, registered as the experiment named
    # for the mlp layer norm gelu. For each hidden input size in a fixed list of four sizes it
    # builds the module in bfloat16 on the device, counts flops of one eager call on a one
    # dimensional random bfloat16 input, compiles it without dynamic shapes, warms up, times it
    # with the inductor benchmarker, and accumulates a utilization estimate against the A100
    # 40GB bfloat16 peak. The estimate is averaged over sizes and returned as a one-element list
    # of Experiment records (metric flops_utilization, target given as a string, actual
    # formatted with two decimals). Needs a device; compiles with side effects on
    # compile caches.
    # KNOWN ISSUE: the benchmarker result is multiplied by one thousand and then multiplied by
    # flops without dividing by time, so the value is not a true utilization ratio, and the
    # input dtype is always bfloat16 regardless of the module dtype.
    dtype_flops_utilization_map = {
        torch.bfloat16: "0.8",
    }
    input_shapes = [1024, 4096, 8192, 16384]
    intermediate_size = 14336
    results = []
    for dtype, expected_flops_utilization in dtype_flops_utilization_map.items():
        flops_utilization = 0
        for D in input_shapes:
            mod = SimpleMLP(
                input_dim=D, hidden_dim=intermediate_size, output_dim=D, dtype=dtype
            ).to(device)

            x = torch.randn(D, device=device, dtype=torch.bfloat16)

            with FlopCounterMode(display=False) as mode:
                mod(x)

            flops = mode.get_total_flops()

            compiled_mod = torch.compile(mod, dynamic=False)

            for _ in range(WARMUP_ITER):
                compiled_mod(x)

            us_per_iter = benchmarker.benchmark(compiled_mod, (x,), {}) * 1000
            flops_utilization += us_per_iter * flops / 1e9 / A100_40G_BF16_TFLOPS

        flops_utilization = flops_utilization / len(input_shapes)
        dtype_str = str(dtype).replace("torch.", "")
        results.append(
            Experiment(
                "mlp_layer_norm_gelu",
                "flops_utilization",
                expected_flops_utilization,
                f"{flops_utilization:.02f}",
                dtype_str,
                device,
                get_arch_name(),
            )
        )
    return results


@register_experiment(name="layer_norm")
def run_layer_norm(device: str = "cuda"):
    # [FUNCTION DOC] Benchmark layer norm, registered under the name layer_norm. For each of
    # four feature sizes it builds a layer norm module on the device in bfloat16, creates a
    # random input with 4096 rows, compiles without dynamic shapes, warms up, times it, and
    # converts the time to effective memory bandwidth in GB per second assuming one read and one
    # write of the whole input. Bandwidth is averaged over sizes and returned as a one-element
    # list with an Experiment record (metric memory bandwidth in GB/s, target as a string,
    # actual formatted with two decimals).
    dtype_memory_bandwidth_map = {
        torch.bfloat16: "950",
    }
    input_shapes = [1024, 4096, 8192, 16384]
    BS = 4096
    results = []
    for dtype, expected_memory_bandwidth in dtype_memory_bandwidth_map.items():
        memory_bandwidth = 0
        for D in input_shapes:
            mod = nn.LayerNorm(D).to(device=device, dtype=dtype)

            x = torch.randn(BS, D, device=device, dtype=dtype)

            compiled_mod = torch.compile(mod, dynamic=False)

            for _ in range(WARMUP_ITER):
                compiled_mod(x)

            us_per_iter = benchmarker.benchmark(compiled_mod, (x,), {}) * 1000
            memory_bandwidth += (1e6 / us_per_iter) * 2 * BS * D * dtype.itemsize / 1e9

        memory_bandwidth = memory_bandwidth / len(input_shapes)
        dtype_str = str(dtype).replace("torch.", "")
        results.append(
            Experiment(
                "layer_norm",
                "memory_bandwidth(GB/s)",
                expected_memory_bandwidth,
                f"{memory_bandwidth:.02f}",
                dtype_str,
                device,
                get_arch_name(),
            )
        )
    return results


@register_experiment(name="gather_gemv")
@torch._inductor.config.patch(coordinate_descent_tuning=True)
def run_gather_gemv(device: str = "cuda"):
    # [FUNCTION DOC] Benchmark a mixture-of-experts style gather followed by matrix-vector
    # product, registered as gather_gemv and run with coordinate descent tuning enabled in
    # inductor via a config patch decorator. For int8 and bfloat16 expert weights and four
    # sizes, a random weight stack with 8 experts is built, two experts (indices three and five)
    # are selected, cast to the bfloat16 vector dtype, and multiplied with a vector. The
    # function is compiled without dynamic shapes, warmed up and timed; bandwidth counts two
    # weight matrices of the dtype size. Returns one Experiment per dtype with the averaged
    # bandwidth in GB/s.
    E = 8
    dtype_memory_bandwidth_map = {
        torch.int8: "990",
        torch.bfloat16: "1060",
    }
    input_shapes = [1024, 4096, 8192, 16384]
    results = []
    for dtype, expected_memory_bandwidth in dtype_memory_bandwidth_map.items():
        memory_bandwidth = 0
        for D in input_shapes:

            def gather_gemv(W, score_idxs, x):
                # [FUNCTION DOC] HELPER: Index the stacked expert weights with the index tensor,
                # cast the selected experts to the dtype of x and return their matrix product
                # with x.
                return W[score_idxs].to(x.dtype) @ x

            W = torch.randn(E, D, D, device=device).to(dtype=dtype)
            x = torch.randn(D, device=device, dtype=torch.bfloat16)
            score_idxs = torch.tensor([3, 5], device=device)

            compiled_fn = torch.compile(gather_gemv, dynamic=False)

            for _ in range(WARMUP_ITER):
                compiled_fn(W, score_idxs, x)

            us_per_iter = (
                benchmarker.benchmark(
                    compiled_fn,
                    (
                        W,
                        score_idxs,
                        x,
                    ),
                    {},
                )
                * 1000
            )
            memory_bandwidth += (1e6 / us_per_iter) * 2 * D * D * dtype.itemsize / 1e9

        memory_bandwidth = memory_bandwidth / len(input_shapes)
        dtype_str = str(dtype).replace("torch.", "")
        results.append(
            Experiment(
                "gather_gemv",
                "memory_bandwidth(GB/s)",
                expected_memory_bandwidth,
                f"{memory_bandwidth:.02f}",
                dtype_str,
                device,
                get_arch_name(),
            )
        )
    return results


@register_experiment(name="gemv")
@torch._inductor.config.patch(coordinate_descent_tuning=True)
def run_gemv(device: str = "cuda"):
    # [FUNCTION DOC] Benchmark a dense matrix-vector product, registered as gemv and run with
    # coordinate descent tuning enabled in inductor via a config patch decorator. For int8 and
    # bfloat16 weights and four sizes it builds a square random weight and a bfloat16 vector,
    # compiles the cast-and-multiply function without dynamic shapes, warms up, times it, and
    # converts to bandwidth in GB per second counting one pass over the weight. Returns one
    # Experiment per dtype with the bandwidth averaged over the sizes.
    dtype_memory_bandwidth_map = {
        torch.int8: "870",
        torch.bfloat16: "990",
    }
    input_shapes = [1024, 4096, 8192, 16384]
    results = []
    for dtype, expected_memory_bandwidth in dtype_memory_bandwidth_map.items():
        memory_bandwidth = 0
        for D in input_shapes:

            def gemv(W, x):
                # [FUNCTION DOC] HELPER: Cast the weight to the dtype of x and return the
                # matrix-vector product with x.
                return W.to(x.dtype) @ x

            W = torch.randn(D, D, device=device).to(dtype=dtype)
            x = torch.randn(D, device=device, dtype=torch.bfloat16)

            compiled_fn = torch.compile(gemv, dynamic=False)

            for _ in range(WARMUP_ITER):
                compiled_fn(W, x)

            us_per_iter = (
                benchmarker.benchmark(
                    compiled_fn,
                    (
                        W,
                        x,
                    ),
                    {},
                )
                * 1000
            )
            memory_bandwidth += (1e6 / us_per_iter) * D * D * dtype.itemsize / 1e9

        memory_bandwidth = memory_bandwidth / len(input_shapes)
        dtype_str = str(dtype).replace("torch.", "")
        results.append(
            Experiment(
                "gemv",
                "memory_bandwidth(GB/s)",
                expected_memory_bandwidth,
                f"{memory_bandwidth:.02f}",
                dtype_str,
                device,
                get_arch_name(),
            )
        )
    return results


def output_csv(output_file, headers, row):
    # [FUNCTION DOC] Append one result row to a CSV file, creating it if needed. If the file
    # exists, its rows are read; when the supplied headers are longer than the existing first
    # row they replace it (earlier runs may have failed before filling the header), otherwise
    # the existing header is reused for padding. If the file does not exist, the headers become
    # the first row. Floats are written with six decimals. The file is rewritten completely,
    # padding short rows with zero strings. When the path differs from the default name, the
    # parent directory is created.
    # KNOWN ISSUE: a custom path without a directory part makes the directory creation call fail
    # with an error because the directory name is empty.
    if os.path.exists(output_file):
        with open(output_file) as fd:
            lines = list(csv.reader(fd)) or [[]]
            if headers and len(headers) > len(lines[0]):
                # if prior results failed the header might not be filled in yet
                lines[0] = headers
            else:
                headers = lines[0]
    else:
        lines = [headers]

    if output_file != DEFAULT_OUTPUT_FILE:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    lines.append([(f"{x:.6f}" if isinstance(x, float) else x) for x in row])
    with open(output_file, "w") as fd:
        writer = csv.writer(fd, lineterminator="\n")
        for line in lines:
            writer.writerow(list(line) + ["0"] * (len(headers) - len(line)))


def output_json(output_file, headers, row):
    # [FUNCTION DOC] Append one result as a JSON line to a file whose name is the CSV path with
    # its extension replaced by json. The row is mapped to the headers by position and arranged
    # into the schema of the OSS benchmark database: benchmark name, mode inference, dtype,
    # device and arch; the model name and type (OSS model if is_model is truthy, else
    # micro-benchmark); and the metric with its actual and target values. Opens the file in
    # append mode and prints one serialized record.
    """
    Write the result into JSON format, so that it can be uploaded to the benchmark database
    to be displayed on OSS dashboard. The JSON format is defined at
    https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
    """
    mapping_headers = {headers[i]: v for i, v in enumerate(row)}
    record = {
        "benchmark": {
            "name": "PyTorch gpt-fast benchmark",
            "mode": "inference",
            "dtype": mapping_headers["dtype"],
            "extra_info": {
                "device": mapping_headers["device"],
                "arch": mapping_headers["arch"],
            },
        },
        "model": {
            "name": mapping_headers["name"],
            "type": "OSS model" if mapping_headers["is_model"] else "micro-benchmark",
            "origins": ["pytorch"],
        },
        "metric": {
            "name": mapping_headers["metric"],
            "benchmark_values": [mapping_headers["actual"]],
            "target_value": mapping_headers["target"],
        },
    }

    with open(f"{os.path.splitext(output_file)[0]}.json", "a") as f:
        print(json.dumps(record), file=f)


DEFAULT_OUTPUT_FILE = "gpt_fast_benchmark.csv"


def main(output_file=DEFAULT_OUTPUT_FILE, only_model=None):
    # [FUNCTION DOC] Run registered experiments and record the results. With no filter all
    # registered functions run; with a name only that one runs. For each function the device is
    # cuda if available (cpu if the availability check raises AssertionError), the cudagraph
    # step marker is advanced, the function is called with the device, and each returned
    # Experiment is converted to a plain tuple. Afterwards every row is written with the CSV and
    # the JSON writers. Returns nothing.
    # KNOWN ISSUE: for an unknown experiment name a message is printed but execution continues
    # and then fails with a lookup error instead of returning.
    results = []

    if not only_model:
        experiments = all_experiments.values()
    else:
        if only_model not in all_experiments:
            print(
                f"Unknown model: {only_model}, all available models: {all_experiments.keys()}"
            )
        # only run the specified model
        experiments = [all_experiments[only_model]]
    for func in experiments:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except AssertionError:
            # This happens when torch is compiled with CUDA turning off completely
            device = "cpu"

        torch.compiler.cudagraph_mark_step_begin()
        lst = func(device)
        for x in lst:
            results.append(dataclasses.astuple(x))

    headers = [field.name for field in dataclasses.fields(Experiment)]

    for row in results:
        output_csv(output_file, headers, row)
        # Also write the output in JSON format so that it can be ingested into the OSS benchmark database
        output_json(output_file, headers, row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help="Set the output CSV file to save the benchmark results",
    )
    parser.add_argument(
        "--only",
        help="Specify a model or micro-benchmark name to run exclusively",
    )
    args = parser.parse_args()

    main(output_file=args.output, only_model=args.only)
