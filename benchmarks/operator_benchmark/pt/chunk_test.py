# [MODULE DOC] Operator microbenchmark for torch.chunk on random 2-D tensors, on CPU and CUDA, with
# short and long config sets. Registers tests at import time. Running as a script starts the
# op_bench runner.
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for Chunk operator"""


# Configs for PT Chunk operator
chunk_short_configs = op_bench.config_list(
    attr_names=["M", "N", "chunks"],
    attrs=[
        [8, 8, 2],
        [256, 512, 2],
        [512, 512, 2],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)

chunks_long_configs = op_bench.cross_product_configs(
    M=[128, 1024], N=[128, 1024], chunks=[2, 4], device=["cpu", "cuda"], tags=["long"]
)


class ChunkBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for splitting a random MxN tensor into a number of chunks.
    def init(self, M, N, chunks, device):
        # [FUNCTION DOC] Create a random MxN tensor on the device, store it with the chunk count in
        # the inputs mapping and set the module name to chunk. Returns nothing.
        self.inputs = {"input_one": torch.rand(M, N, device=device), "chunks": chunks}
        self.set_module_name("chunk")

    def forward(self, input_one, chunks: int):
        # [FUNCTION DOC] Split input_one into the requested number of chunks and return them.
        return torch.chunk(input_one, chunks)


op_bench.generate_pt_test(chunk_short_configs + chunks_long_configs, ChunkBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
