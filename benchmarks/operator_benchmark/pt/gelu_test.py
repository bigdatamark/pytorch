# [MODULE DOC] Operator microbenchmark for the functional gelu activation on random NCHW CPU
# tensors, using a single long config set. Registers the test at import time. Running as a script
# starts the op_bench runner.
import operator_benchmark as op_bench

import torch


"""
Microbenchmarks for the gelu operators.
"""

gelu_configs_long = op_bench.cross_product_configs(
    N=[1, 4], C=[3], H=[16, 256], W=[16, 256], device=["cpu"], tags=["long"]
)


class GeluBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for gelu applied to a random NxCxHxW tensor.
    def init(self, N, C, H, W, device):
        # [FUNCTION DOC] Create a random NxCxHxW tensor on the device and store it in the inputs
        # mapping. Returns nothing. No module name is set.
        self.inputs = {"input": torch.rand(N, C, H, W, device=device)}

    def forward(self, input):
        # [FUNCTION DOC] Return the elementwise gelu of input.
        return torch.nn.functional.gelu(input)


op_bench.generate_pt_test(gelu_configs_long, GeluBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
