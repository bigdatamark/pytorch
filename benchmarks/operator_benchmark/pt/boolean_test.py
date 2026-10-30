# [MODULE DOC] Operator microbenchmarks for the boolean reductions all and any on random bool
# tensors, on CPU and CUDA. Registers tests at import time with long and short configs for all, and
# a separate config set for any. Running as a script starts the op_bench runner.
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for boolean operators. Supports both Caffe2/PyTorch."""

# Configs for PT all operator
all_long_configs = op_bench.cross_product_configs(
    M=[8, 128], N=[32, 64], K=[256, 512], device=["cpu", "cuda"], tags=["long"]
)


all_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)


class AllBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for the all reduction over a random boolean MxNxK tensor.
    def init(self, M, N, K, device):
        # [FUNCTION DOC] Create a random boolean tensor of shape MxNxK on the device, store it in
        # the inputs mapping and set the reported module name to all. Returns nothing.
        self.inputs = {
            "input_one": torch.randint(0, 2, (M, N, K), device=device, dtype=torch.bool)
        }
        self.set_module_name("all")

    def forward(self, input_one):
        # [FUNCTION DOC] Return the all reduction of input_one, a single boolean scalar tensor.
        return torch.all(input_one)


# The generated test names based on all_short_configs will be in the following pattern:
# all_M8_N16_K32_devicecpu
# all_M8_N16_K32_devicecpu_bwdall
# all_M8_N16_K32_devicecpu_bwd1
# all_M8_N16_K32_devicecpu_bwd2
# ...
# Those names can be used to filter tests.

op_bench.generate_pt_test(all_long_configs + all_short_configs, AllBenchmark)

"""Mircobenchmark for any operator."""


class AnyBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for the any reduction over a random boolean MxN tensor.
    def init(self, M, N, device):
        # [FUNCTION DOC] Create a random boolean tensor of shape MxN on the device, store it in the
        # inputs mapping and set the reported module name to any. Returns nothing.
        self.inputs = {
            "input_one": torch.randint(0, 2, (M, N), device=device, dtype=torch.bool)
        }
        self.set_module_name("any")

    def forward(self, input_one):
        # [FUNCTION DOC] Return the any reduction of input_one, a single boolean scalar tensor.
        return torch.any(input_one)


any_configs = op_bench.cross_product_configs(
    M=[8, 256],
    N=[256, 16],
    device=["cpu", "cuda"],
    tags=["any"],
)

op_bench.generate_pt_test(any_configs, AnyBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
