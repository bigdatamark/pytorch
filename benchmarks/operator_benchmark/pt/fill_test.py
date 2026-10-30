# [MODULE DOC] Operator microbenchmark for the in-place fill_ method. Short configs use cpu and
# cuda with int32; long configs span many dtypes (bool through double) on every device type
# returned by get_all_device_types, which is evaluated at import time. Registers tests at import.
# Running as a script starts the op_bench runner.
import operator_benchmark as op_bench

import torch
from torch.testing._internal.common_device_type import get_all_device_types


"""Microbenchmark for Fill_ operator."""

fill_short_configs = op_bench.config_list(
    attr_names=["N"],
    attrs=[
        [1],
        [1024],
        [2048],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "dtype": [torch.int32],
    },
    tags=["short"],
)

fill_long_configs = op_bench.cross_product_configs(
    N=[10, 1000],
    device=get_all_device_types(),
    dtype=[
        torch.bool,
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.half,
        torch.float,
        torch.double,
    ],
    tags=["long"],
)


class Fill_Benchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for filling a zero vector of length N with a constant, in place.
    def init(self, N, device, dtype):
        # [FUNCTION DOC] Create a zero vector of length N on the device, cast to dtype, store it in
        # the inputs mapping and set the module name to fill_. Returns nothing.
        self.inputs = {"input_one": torch.zeros(N, device=device).type(dtype)}
        self.set_module_name("fill_")

    def forward(self, input_one):
        # [FUNCTION DOC] Fill input_one in place with the value ten and return it. The tensor is
        # mutated, so repeated calls reuse the same storage.
        return input_one.fill_(10)


op_bench.generate_pt_test(fill_short_configs + fill_long_configs, Fill_Benchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
