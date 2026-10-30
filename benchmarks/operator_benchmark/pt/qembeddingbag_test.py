# [MODULE DOC] Microbenchmark for the quantized EmbeddingBag module (torch.ao.nn.quantized).
# Registers one benchmark class over the short embedding-bag configs from pt.configs at import
# time; running as a script starts the benchmark runner.
import numpy
from pt import configs

import operator_benchmark as op_bench

import torch
import torch.ao.nn.quantized as nnq


"""
Microbenchmarks for qEmbeddingBag operators.
"""


class QEmbeddingBagBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark of a quantized EmbeddingBag module lookup with random indices and
    # offsets.
    def init(
        self,
        embeddingbags,
        dim,
        mode,
        input_size,
        offset,
        sparse,
        include_last_offset,
        device,
    ):
        # [FUNCTION DOC] Builds the benchmark. Args: embeddingbags (table rows), dim, mode,
        # input_size, offset, sparse (accepted but unused), include_last_offset, device.
        # Creates the quantized module on the device, seeds numpy with a fixed value, and draws
        # random indices in [0, embeddingbags) as a long tensor. The offset tensor is the given
        # start offset followed by the index count, concatenated. Stores indices and offsets in
        # self.inputs and sets the module name.
        self.embedding = nnq.EmbeddingBag(
            num_embeddings=embeddingbags,
            embedding_dim=dim,
            mode=mode,
            include_last_offset=include_last_offset,
        ).to(device=device)
        numpy.random.seed((1 << 32) - 1)
        self.input = torch.tensor(
            numpy.random.randint(0, embeddingbags, input_size), device=device
        ).long()
        offset = torch.LongTensor([offset], device=device)
        self.offset = torch.cat(
            (offset, torch.tensor([self.input.size(0)], dtype=torch.long)), 0
        )
        self.inputs = {"input": self.input, "offset": self.offset}
        self.set_module_name("qEmbeddingBag")

    def forward(self, input, offset):
        # [FUNCTION DOC] Runs the embedding bag module on the indices and offsets and returns
        # the output.
        return self.embedding(input, offset)


op_bench.generate_pt_test(configs.embeddingbag_short_configs, QEmbeddingBagBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
