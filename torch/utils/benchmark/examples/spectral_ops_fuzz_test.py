# mypy: allow-untyped-defs
# [MODULE DOC]
# Command line microbenchmark for the torch.fft transforms. Defines a table of benchmarks
# (BENCHMARKS, with lookup BENCHMARK_MAP and names), runs each on the chosen devices with
# fuzzed inputs from the spectral fuzzer, optionally writes a CSV of the timings, and prints a
# Compare table. The script body runs only under the main guard; importing has no side effects
# beyond defining names.
"""Microbenchmarks for the torch.fft module"""
from argparse import ArgumentParser
from collections import namedtuple
from collections.abc import Iterable

import torch
import torch.fft
from torch.utils import benchmark
from torch.utils.benchmark.op_fuzzers.spectral import SpectralOpFuzzer


def _dim_options(ndim):
    # [FUNCTION DOC] HELPER: Returns the list of transform dimension choices to benchmark for a
    # tensor of 1 to 3 dimensions (none meaning all dimensions, single axes, and axis pairs).
    # Raises a value error for any other dimension count.
    if ndim == 1:
        return [None]
    elif ndim == 2:
        return [0, 1, None]
    elif ndim == 3:
        return [0, 1, 2, (0, 1), (0, 2), None]
    raise ValueError(f"Expected ndim in range 1-3, got {ndim}")


def run_benchmark(name: str, function: object, dtype: torch.dtype, seed: int, device: str, samples: int,
                  probability_regular: float):
    # [FUNCTION DOC] Benchmarks one transform function over the given number of fuzzed inputs
    # of the given dtype. For each sample and each dim choice it times the call (several thread
    # counts on cpu, one on cuda) with a one second minimum run, attaches metadata (name,
    # device, dim, shape and the fuzzed tensor properties) and collects the measurements.
    # Returns the list of measurements.
    cuda = device == 'cuda'
    spectral_fuzzer = SpectralOpFuzzer(seed=seed, dtype=dtype, cuda=cuda,
                                       probability_regular=probability_regular)
    results = []
    for tensors, tensor_params, params in spectral_fuzzer.take(samples):
        shape = [params['k0'], params['k1'], params['k2']][:params['ndim']]
        str_shape = ' x '.join([f"{s:<4}" for s in shape])
        sub_label = f"{str_shape} {'' if tensor_params['x']['is_contiguous'] else '(discontiguous)'}"
        for dim in _dim_options(params['ndim']):
            for nthreads in (1, 4, 16) if not cuda else (1,):
                measurement = benchmark.Timer(
                    stmt='func(x, dim=dim)',
                    globals={'func': function, 'x': tensors['x'], 'dim': dim},
                    label=f"{name}_{device}",
                    sub_label=sub_label,
                    description=f"dim={dim}",
                    num_threads=nthreads,
                ).blocked_autorange(min_run_time=1)
                measurement.metadata = {
                    'name': name,
                    'device': device,
                    'dim': dim,
                    'shape': shape,
                }
                measurement.metadata.update(tensor_params['x'])
                results.append(measurement)
    return results


Benchmark = namedtuple('Benchmark', ['name', 'function', 'dtype'])
BENCHMARKS = [
    Benchmark('fft_real', torch.fft.fftn, torch.float32),
    Benchmark('fft_complex', torch.fft.fftn, torch.complex64),
    Benchmark('ifft', torch.fft.ifftn, torch.complex64),
    Benchmark('rfft', torch.fft.rfftn, torch.float32),
    Benchmark('irfft', torch.fft.irfftn, torch.complex64),
]
BENCHMARK_MAP = {b.name: b for b in BENCHMARKS}
BENCHMARK_NAMES = [b.name for b in BENCHMARKS]
DEVICE_NAMES = ['cpu', 'cuda']

def _output_csv(file, results) -> None:
    # [FUNCTION DOC] HELPER: Writes a header and then one comma separated line per measurement
    # (benchmark, device, threads, element count, shape, contiguity, dim, mean, median and
    # interquartile range in microseconds) to the open file.
    # KNOWN ISSUE: the shape string is computed only in the branch where dim is not iterable,
    # so an iterable dim reuses a stale value or hits an unbound name on the first row.
    file.write('benchmark,device,num_threads,numel,shape,contiguous,dim,mean (us),median (us),iqr (us)\n')
    for measurement in results:
        metadata = measurement.metadata
        device, dim, shape, name, numel, contiguous = (
            metadata['device'], metadata['dim'], metadata['shape'],
            metadata['name'], metadata['numel'], metadata['is_contiguous'])

        if isinstance(dim, Iterable):
            dim_str = '-'.join(str(d) for d in dim)
        else:
            dim_str = str(dim)
            shape_str = 'x'.join(str(s) for s in shape)

        print(name, device, measurement.task_spec.num_threads, numel, shape_str, contiguous, dim_str,  # type: ignore[possibly-undefined]
              measurement.mean * 1e6, measurement.median * 1e6, measurement.iqr * 1e6,
              sep=',', file=file)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=str, choices=DEVICE_NAMES, nargs='+', default=DEVICE_NAMES)
    parser.add_argument('--bench', type=str, choices=BENCHMARK_NAMES, nargs='+', default=BENCHMARK_NAMES)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--probability-regular', '--probability_regular', type=float, default=1.0)
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()

    num_benchmarks = len(args.device) * len(args.bench)
    i = 0
    results = []
    for device in args.device:
        for bench in (BENCHMARK_MAP[b] for b in args.bench):
            results += run_benchmark(
                name=bench.name, function=bench.function, dtype=bench.dtype,
                seed=args.seed, device=device, samples=args.samples,
                probability_regular=args.probability_regular)
            i += 1
            print(f'Completed {bench.name} benchmark on {device} ({i} of {num_benchmarks})')

    if args.output is not None:
        with open(args.output, 'w') as f:
            _output_csv(f, results)

    compare = benchmark.Compare(results)
    compare.trim_significant_figures()
    compare.colorize()
    compare.print()
