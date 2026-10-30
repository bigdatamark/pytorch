# [MODULE DOC] Operator microbenchmark for torch.diag on CPU and CUDA, covering both a 2-D input
# (extract a diagonal) and a 1-D input (build a matrix), with an optional preallocated out tensor.
# Registers short-config tests at import time. Running as a script starts the runner.
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for diag operator"""


# Configs for PT diag operator
diag_configs_short = op_bench.config_list(
    attr_names=["dim", "M", "N", "diagonal", "out"],
    attrs=[
        [1, 64, 64, 0, True],
        [2, 128, 128, -10, False],
        [1, 256, 256, 20, True],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)


class DiagBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for diag with selectable input rank, diagonal offset and out usage.
    def init(self, dim, M, N, diagonal, out, device):
        # [FUNCTION DOC] Build the input: a random MxN matrix if dim is 2, otherwise a random
        # vector of length M. Stores it with the diagonal offset, the out flag and an empty out
        # tensor on the device in the inputs mapping, and sets the module name to diag.
        # Returns nothing.
        self.inputs = {
            "input": torch.rand(M, N, device=device)
            if dim == 2
            else torch.rand(M, device=device),
            "diagonal": diagonal,
            "out": out,
            "out_tensor": torch.tensor(
                (),
                device=device,
            ),
        }
        self.set_module_name("diag")

    def forward(self, input, diagonal: int, out: bool, out_tensor):
        # [FUNCTION DOC] Compute diag of input with the given diagonal offset. If out is true,
        # write into the provided out tensor, otherwise allocate a new result. Returns the result.
        if out:
            return torch.diag(input, diagonal=diagonal, out=out_tensor)
        else:
            return torch.diag(input, diagonal=diagonal)


op_bench.generate_pt_test(diag_configs_short, DiagBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
