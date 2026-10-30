# [MODULE DOC] Operator microbenchmark for the in-place index_add_ method on a random MxNxK tensor
# along dimension 0, 1 or 2 (float dtype). Short configs run on CPU, long configs on CPU and CUDA.
# Registers tests at import time. Running as a script starts the op_bench runner.
import numpy

import operator_benchmark as op_bench

import torch


"""Microbenchmarks for index_add_ operator."""


configs_short = op_bench.config_list(
    attr_names=["M", "N", "K", "dim"],
    attrs=[[8, 32, 1, 0], [256, 512, 1, 1], [512, 512, 1, 2]],
    cross_product_configs={"device": ["cpu"], "dtype": [torch.float]},
    tags=["short"],
)


configs_long = op_bench.cross_product_configs(
    M=[1, 128, 1024],
    N=[2, 256, 512],
    K=[1, 2, 8],
    dim=[0, 1, 2],
    device=["cpu", "cuda"],
    dtype=[torch.float],
    tags=["long"],
)


class IndexAddBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for index_add_ with a random subset of indices along the chosen
    # dimension. State in the inputs mapping: tensor, dim, index and source.
    def init(self, M, N, K, dim, dtype, device):
        # [FUNCTION DOC] Create a random MxNxK tensor, choose a random count (at least one) of
        # distinct indices along dim using the unseeded numpy generator, and build a source tensor
        # whose size along dim equals that count. Stores tensor, dim, index and source in the
        # inputs mapping and sets the module name to index_add_. The index count is therefore not
        # reproducible between runs.
        # creating the original tensor
        tensor = torch.rand(M, N, K, dtype=dtype, device=device)

        # creating index
        index_max_len = tensor.shape[dim]
        index_len = numpy.random.randint(1, index_max_len + 1)
        index = torch.tensor(
            numpy.random.choice(index_max_len, index_len, replace=False), device=device
        )

        src_dims = [M, N, K]
        src_dims[dim] = index_len
        source = torch.rand(*src_dims, dtype=dtype, device=device)

        self.inputs = {
            "tensor": tensor,
            "dim": dim,
            "index": index,
            "source": source,
        }
        self.set_module_name("index_add_")

    def forward(self, tensor, dim, index, source):
        # [FUNCTION DOC] Add source into tensor in place along dim at the given indices and return
        # tensor. The tensor is mutated by every call.
        return tensor.index_add_(dim, index, source)


op_bench.generate_pt_test(configs_short + configs_long, IndexAddBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
