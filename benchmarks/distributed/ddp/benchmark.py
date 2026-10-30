#!/usr/bin/env python3
#
# Measure distributed training iteration time.
#
# This program performs a sweep over a) a number of model architectures, and
# b) an increasing number of processes. This produces a 1-GPU baseline,
# an 8-GPU baseline (if applicable), as well as measurements for however
# many processes can participate in training.
#
# [MODULE DOC] Script measuring distributed data parallel iteration time across torchvision
# models. Provides helpers for gathering objects across ranks, benchmark classes, a sweep
# over process group sizes, and a main entry point that parses arguments, initializes a
# gloo metadata group, and optionally writes a JSON report. Needs 8 GPUs per machine.

import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
import time

import numpy as np
import torchvision

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim


def allgather_object(obj):
    # [FUNCTION DOC] HELPER: Gathers one picklable object from every rank in the default
    # process group and returns the list ordered by rank. Collective call.
    out = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(out, obj)
    return out


def allgather_run(cmd):
    # [FUNCTION DOC] HELPER: Runs a shell command (split into arguments, output captured)
    # on this process, then gathers the decoded stdout from all ranks. Raises
    # AssertionError (including stderr text) if the command exits nonzero.
    proc = subprocess.run(shlex.split(cmd), capture_output=True)
    if proc.returncode != 0:
        raise AssertionError(
            f"Command '{cmd}' failed with return code {proc.returncode}: {proc.stderr.decode('utf-8')}"
        )
    return allgather_object(proc.stdout.decode("utf-8"))


def allequal(iterator):
    # [FUNCTION DOC] HELPER: Returns True if all items of the iterable compare equal to the
    # first one; an empty iterable counts as equal.
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def benchmark_process_group(pg, benchmark, use_ddp_for_single_rank=True):
    # [FUNCTION DOC] Runs one benchmark on the given process group and returns the timing
    # list (seconds per iteration) for this rank. Seeds CPU and CUDA RNGs with the group
    # rank, builds model and a single input/target pair from the benchmark object, wraps the
    # model in DistributedDataParallel unless the group has one rank and the flag disables
    # it, trains for 5 warmup plus 10 measured SGD iterations, and discards warmup times.
    torch.manual_seed(pg.rank())
    torch.cuda.manual_seed(pg.rank())

    model = benchmark.create_model()
    data = [(benchmark.generate_inputs(), benchmark.generate_target())]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), 0.001, momentum=0.9, weight_decay=1e-4)
    if use_ddp_for_single_rank or pg.size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            process_group=pg,
            bucket_cap_mb=benchmark.bucket_size,
        )

    measurements = []
    warmup_iterations = 5
    measured_iterations = 10
    for inputs, target in data * (warmup_iterations + measured_iterations):
        start = time.time()
        output = model(*inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        measurements.append(time.time() - start)

    # Throw away measurements for warmup iterations
    return measurements[warmup_iterations:]


def run_benchmark(benchmark, ranks, opts):
    # [FUNCTION DOC] Creates a process group for the given ranks, runs the benchmark on
    # ranks that belong to it (extra options are passed as keyword arguments), destroys the
    # group, and syncs all ranks. Returns the concatenated measurements from all ranks.
    group = dist.new_group(ranks=ranks, backend=benchmark.distributed_backend)
    measurements = []
    if dist.get_rank() in set(ranks):
        if not opts:
            opts = {}
        measurements = benchmark_process_group(group, benchmark, **opts)
    dist.destroy_process_group(group)
    dist.barrier()

    # Aggregate measurements for better estimation of percentiles
    return list(itertools.chain(*allgather_object(measurements)))


def sweep(benchmark):
    # [FUNCTION DOC] Runs the full sweep for one benchmark: a per-rank warmup, a no-DDP
    # single GPU run, DDP runs on 1, 2 and 4 GPUs, and one run per full 8-GPU machine,
    # executed in increasing group size. Rank 0 prints a table of percentiles. Returns a
    # list of mappings with the ranks and aggregated measurements, excluding warmup runs.
    # KNOWN ISSUE: the percentile print formats a float throughput with an integer format
    # code, which raises ValueError at runtime on rank 0.
    # Synthesize the set of benchmarks to run.
    # This list contain tuples for ("string prefix", [rank...]).
    benchmarks = []

    def append_benchmark(prefix, ranks, opts=None):
        # [FUNCTION DOC] HELPER: Registers a benchmark in the enclosing list as a tuple of
        # label (prefixed with the GPU count), ranks and options.
        prefix = f"{len(ranks):4} GPUs -- {prefix}"
        benchmarks.append((prefix, ranks, opts))

    def local_print(msg):
        # [FUNCTION DOC] HELPER: Prints the message without newline and flushes, on rank 0 only.
        if dist.get_rank() == 0:
            print(msg, end="", flush=True)

    def print_header():
        # [FUNCTION DOC] HELPER: Prints the column header for the percentile table (rank 0).
        local_print("\n")
        local_print(" " * 22)
        for _ in [50, 75, 90, 95]:
            local_print(f"{'sec/iter':14s}{'ex/sec':10s}")
        local_print("\n")

    def print_measurements(prefix, nelem, measurements):
        # [FUNCTION DOC] HELPER: Prints, on rank 0, the 50th, 75th, 90th and 95th percentile
        # of the sorted measurements with the label, and examples per second computed from
        # the element count.
        measurements = sorted(measurements)
        local_print(f"{prefix:8s}:")
        for p in [50, 75, 90, 95]:
            v = np.percentile(measurements, p)
            local_print(f"  p{p:02d}:  {v:1.3f}s  {nelem / v:6d}/s")
        local_print("\n")

    # Every process runs once by themselves to warm up (CUDA init, etc).
    append_benchmark("  warmup", [dist.get_rank()], {"use_ddp_for_single_rank": False})

    # Single machine baselines
    append_benchmark("  no ddp", range(1), {"use_ddp_for_single_rank": False})
    append_benchmark("   1M/1G", range(1))
    append_benchmark("   1M/2G", range(2))
    append_benchmark("   1M/4G", range(4))

    # Multi-machine benchmarks
    for i in range(1, (dist.get_world_size() // 8) + 1):
        append_benchmark(f"   {i:d}M/8G", range(i * 8))

    # Run benchmarks in order of increasing number of GPUs
    print_header()
    results = []
    for prefix, ranks, opts in sorted(benchmarks, key=lambda tup: len(tup[1])):
        # Turn range into materialized list.
        ranks = list(ranks)
        measurements = run_benchmark(benchmark, ranks, opts)
        if "warmup" not in prefix:
            print_measurements(prefix, benchmark.batch_size, measurements)
            results.append({"ranks": ranks, "measurements": measurements})

    return results


class Benchmark:
    # [CLASS DOC] Abstract base for a DDP benchmark. Stores device, batch size (fixed at 32),
    # distributed backend name and DDP bucket size. Subclasses supply model, inputs, target.
    def __init__(self, device, distributed_backend, bucket_size):
        # [FUNCTION DOC] Stores the device, backend name and bucket size; sets batch size 32.
        self.device = device
        self.batch_size = 32
        self.distributed_backend = distributed_backend
        self.bucket_size = bucket_size

    def __str__(self):
        # [FUNCTION DOC] Abstract; subclasses return a description string. Always raises
        # NotImplementedError here.
        raise NotImplementedError

    def create_model(self):
        # [FUNCTION DOC] Abstract; subclasses return the model. Always raises NotImplementedError.
        raise NotImplementedError

    def generate_inputs(self):
        # [FUNCTION DOC] Abstract; subclasses return the input list. Always raises
        # NotImplementedError here.
        raise NotImplementedError

    def generate_target(self):
        # [FUNCTION DOC] Abstract; subclasses return the target. Always raises NotImplementedError.
        raise NotImplementedError


class TorchvisionBenchmark(Benchmark):
    # [CLASS DOC] Benchmark for a torchvision model looked up by name, with random image
    # input of shape batch x 3 x 224 x 224 and constant class label 1 as target.
    def __init__(self, device, distributed_backend, bucket_size, model):
        # [FUNCTION DOC] Initializes the base class and stores the torchvision model name.
        super().__init__(
            device,
            distributed_backend,
            bucket_size,
        )
        self.model = model

    def __str__(self):
        # [FUNCTION DOC] Returns a description of the model name and batch size.
        return f"{self.model} with batch size {self.batch_size}"

    def create_model(self):
        # [FUNCTION DOC] Builds the named torchvision model (random weights) on the device.
        return torchvision.models.__dict__[self.model]().to(self.device)

    def generate_inputs(self):
        # [FUNCTION DOC] Returns a one element list holding a random image batch on the device.
        return [torch.rand([self.batch_size, 3, 224, 224], device=self.device)]

    def generate_target(self):
        # [FUNCTION DOC] Returns a long tensor of batch size entries all equal to class 1.
        return torch.tensor([1] * self.batch_size, dtype=torch.long, device=self.device)


def main():
    # [FUNCTION DOC] Entry point. Parses arguments (rank, world size, backend, bucket size,
    # master address and port, optional model, optional JSON output path), requires exactly
    # 8 GPUs per node (AssertionError otherwise), initializes a gloo group for metadata,
    # exits with status 1 if nvidia-smi topology differs across machines, prints a header on
    # rank 0, sets the CUDA device by rank modulo 8, sweeps the chosen model or four default
    # resnet/resnext models, and on rank 0 writes the JSON report if requested.
    parser = argparse.ArgumentParser(description="PyTorch distributed benchmark suite")
    parser.add_argument("--rank", type=int, default=os.environ["RANK"])
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--distributed-backend", type=str, default="nccl")
    parser.add_argument("--bucket-size", type=int, default=25)
    parser.add_argument("--master-addr", type=str, required=True)
    parser.add_argument("--master-port", type=str, required=True)
    parser.add_argument("--model", type=str)
    parser.add_argument(
        "--json", type=str, metavar="PATH", help="Write file with benchmark results"
    )
    args = parser.parse_args()

    num_gpus_per_node = torch.cuda.device_count()
    if num_gpus_per_node != 8:
        raise AssertionError(
            f"Expected 8 GPUs per machine, but found {num_gpus_per_node}"
        )

    # The global process group used only for communicating benchmark
    # metadata, like measurements. Not for benchmarking itself.
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=args.rank,
        world_size=args.world_size,
    )

    output = allgather_run("nvidia-smi topo -m")
    if not allequal(output):
        print('Output of "nvidia-smi topo -m" differs between machines')
        sys.exit(1)

    if args.rank == 0:
        print("-----------------------------------")
        print("PyTorch distributed benchmark suite")
        print("-----------------------------------")
        print()
        print(f"* PyTorch version: {torch.__version__}")
        print(f"* CUDA version: {torch.version.cuda}")
        print(f"* Distributed backend: {args.distributed_backend}")
        print(f"* Maximum bucket size: {args.bucket_size}MB")
        print()
        print("--- nvidia-smi topo -m ---")
        print()
        print(output[0])
        print("--------------------------")
        print()

    torch.cuda.set_device(dist.get_rank() % 8)
    device = torch.device(f"cuda:{dist.get_rank() % 8:d}")

    benchmarks = []
    if args.model:
        benchmarks.append(
            TorchvisionBenchmark(
                device=device,
                distributed_backend=args.distributed_backend,
                bucket_size=args.bucket_size,
                model=args.model,
            )
        )
    else:
        for model in ["resnet50", "resnet101", "resnext50_32x4d", "resnext101_32x8d"]:
            benchmarks.append(
                TorchvisionBenchmark(
                    device=device,
                    distributed_backend=args.distributed_backend,
                    bucket_size=args.bucket_size,
                    model=model,
                )
            )

    benchmark_results = []
    for benchmark in benchmarks:
        if args.rank == 0:
            print(f"\nBenchmark: {str(benchmark)}")
        result = sweep(benchmark)
        benchmark_results.append(
            {
                "model": benchmark.model,
                "batch_size": benchmark.batch_size,
                "result": result,
            }
        )

    # Write file with benchmark results if applicable
    if args.rank == 0 and args.json:
        report = {
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "distributed_backend": args.distributed_backend,
            "bucket_size": args.bucket_size,
            "benchmark_results": benchmark_results,
        }
        with open(args.json, "w") as f:
            json.dump(report, f)


if __name__ == "__main__":
    main()
