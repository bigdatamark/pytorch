# [MODULE DOC] Operator benchmark for as_strided over a random matrix with several
# sizes, strides and storage offsets on cpu and cuda. Defines short and long config
# lists and the benchmark class; registers forward tests for both at import time
# and runs the benchmark runner when executed as a script.
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for as_strided operator"""


# Configs for PT as_strided operator
as_strided_configs_short = op_bench.config_list(
    attr_names=["M", "N", "size", "stride", "storage_offset"],
    attrs=[
        [8, 8, (2, 2), (1, 1), 0],
        [256, 256, (32, 32), (1, 1), 0],
        [512, 512, (64, 64), (2, 2), 1],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)

as_strided_configs_long = op_bench.cross_product_configs(
    M=[512],
    N=[1024],
    size=[(16, 16), (128, 128)],
    stride=[(1, 1)],
    storage_offset=[0, 1],
    device=["cpu", "cuda"],
    tags=["long"],
)


class As_stridedBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for the as_strided view operator. The inputs mapping holds a random
    # matrix and the size, stride and storage offset arguments; module name is as_strided.
    def init(self, M, N, size, stride, storage_offset, device):
        # [FUNCTION DOC] Creates a random matrix of the given rows and columns on
        # the device, stores it with the size, stride and storage offset in the
        # inputs mapping, and sets the module name to as_strided.
        self.inputs = {
            "input_one": torch.rand(M, N, device=device),
            "size": size,
            "stride": stride,
            "storage_offset": storage_offset,
        }
        self.set_module_name("as_strided")

    def forward(
        self, input_one, size: list[int], stride: list[int], storage_offset: int
    ):
        # [FUNCTION DOC] Returns a strided view of the input with the given size, stride and
        # storage offset. The annotated argument types allow the module to be scripted by
        # TorchScript.
        return torch.as_strided(input_one, size, stride, storage_offset)


op_bench.generate_pt_test(
    as_strided_configs_short + as_strided_configs_long, As_stridedBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
