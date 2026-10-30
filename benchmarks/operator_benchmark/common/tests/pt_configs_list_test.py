# [MODULE DOC] Example benchmark of element-wise add over explicit size rows
# (three rows) crossed with devices cpu and cuda and dtypes float32 and float64,
# under the short tag. Registers a forward-only test at import time and runs the
# benchmark runner main function when executed as a script.
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for element-wise Add operator. Supports both Caffe2/PyTorch."""

add_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [8, 16, 32],
        [16, 16, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "dtype": [torch.float, torch.float64],
    },
    tags=["short"],
)


class AddBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Add benchmark with module name add. The first input requires gradients
    # and the second does not, with device and dtype taken from the config.
    def init(self, M, N, K, device, dtype):
        # [FUNCTION DOC] Creates two random three dimensional tensors of the
        # given sizes on the given device with the given dtype (the first with
        # requires_grad enabled) and sets the module name to add.
        self.input_one = torch.rand(
            M, N, K, device=device, dtype=dtype, requires_grad=True
        )
        self.input_two = torch.rand(M, N, K, device=device, dtype=dtype)
        self.set_module_name("add")

    def forward(self):
        # [FUNCTION DOC] Returns the elementwise sum of the two stored inputs.
        return torch.add(self.input_one, self.input_two)


op_bench.generate_pt_test(add_short_configs, AddBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
