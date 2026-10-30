# [MODULE DOC] Microbenchmark for the stack operator with the out argument. Defines config sets
# for static runtime shapes, short and long shapes, and shapes with more than four dimensions
# (which take a separate CUDA code path), each crossed with device and stack dim. Registers all
# of them at import time; running as a script starts the benchmark runner.
import random

import operator_benchmark as op_bench

import torch


"""Microbenchmarks for Stack operator"""

# Configs for PT stack operator
stack_configs_static_runtime = op_bench.config_list(
    attr_names=["sizes", "N"],
    attrs=[
        [(20, 40), 5],
        [(1, 40), 5],
    ],
    cross_product_configs={"device": ["cpu", "cuda"], "dim": list(range(3))},
    tags=["static_runtime"],
)

stack_configs_short = op_bench.config_list(
    attr_names=["sizes", "N"],
    attrs=[
        [(1, 1, 1), 2],  # noqa: E241
        [(512, 512, 2), 2],  # noqa: E241
        [(128, 1024, 2), 2],  # noqa: E241
    ],
    cross_product_configs={"device": ["cpu", "cuda"], "dim": list(range(4))},
    tags=["short"],
)

stack_configs_long = op_bench.config_list(
    attr_names=["sizes", "N"],
    attrs=[
        [(2**10, 2**10, 2), 2],  # noqa: E241
        [(2**10 + 1, 2**10 - 1, 2), 2],  # noqa: E226,E241
        [(2**10, 2**10, 2), 2],  # noqa: E241
    ],
    cross_product_configs={"device": ["cpu", "cuda"], "dim": list(range(4))},
    tags=["long"],
)

# There is a different codepath on CUDA for >4 dimensions
stack_configs_multidim = op_bench.config_list(
    attr_names=["sizes", "N"],
    attrs=[
        [(2**6, 2**5, 2**2, 2**4, 2**5), 2],  # noqa: E241
        [(2**4, 2**5, 2**2, 2**4, 2**5), 8],  # noqa: E241
        [
            (2**3 + 1, 2**5 - 1, 2**2 + 1, 2**4 - 1, 2**5 + 1),
            17,
        ],  # noqa: E226,E241
    ],
    cross_product_configs={"device": ["cpu", "cuda"], "dim": list(range(6))},
    tags=["multidim"],
)


class StackBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark of stacking several random tensors into a preallocated output
    # tensor.
    def init(self, sizes, N, dim, device):
        # [FUNCTION DOC] Args: sizes, N (number of tensors), dim, device. Seeds the python
        # random module with 42. If sizes is a list and N is minus one, sizes is taken as the
        # list of shapes; otherwise N shapes are generated, each by copying the entries of
        # sizes, calling any entry that is callable. Creates a random tensor per shape and a
        # random result tensor with the first shape to serve as the out argument. Stores
        # result, inputs and dim in self.inputs and sets the module name.
        random.seed(42)
        inputs = []
        gen_sizes = []
        if type(sizes) is list and N == -1:
            gen_sizes = sizes
        else:
            for i in range(N):
                gen_sizes.append(
                    [
                        old_size() if callable(old_size) else old_size
                        for old_size in sizes
                    ]
                )

        for s in gen_sizes:
            inputs.append(torch.rand(s, device=device))
        result = torch.rand(gen_sizes[0], device=device)
        self.inputs = {"result": result, "inputs": inputs, "dim": dim}
        self.set_module_name("stack")

    def forward(self, result: torch.Tensor, inputs: list[torch.Tensor], dim: int):
        # [FUNCTION DOC] Args: result (output tensor), inputs (list of tensors), dim. Stacks
        # the inputs along dim, writing into result, and returns it.
        return torch.stack(inputs, dim=dim, out=result)


op_bench.generate_pt_test(
    stack_configs_static_runtime
    + stack_configs_short
    + stack_configs_long
    + stack_configs_multidim,
    StackBenchmark,
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
