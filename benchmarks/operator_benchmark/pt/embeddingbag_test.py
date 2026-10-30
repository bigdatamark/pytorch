# [MODULE DOC] Operator microbenchmarks for torch.nn.EmbeddingBag and torch.nn.Embedding using
# shared short configs (CPU only). Inputs are random indices from a numpy generator seeded with a
# fixed value. Forward and gradient tests are registered at import time. Running as a script starts
# the op_bench runner.
import numpy
from pt import configs

import operator_benchmark as op_bench

import torch


"""Embedding and EmbeddingBag Operator Benchmark"""


class EmbeddingBagBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for an EmbeddingBag module. State: the embedding module, plus an index
    # tensor and an offsets tensor in the inputs mapping.
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
        # [FUNCTION DOC] Build an EmbeddingBag module with the given table size, dim, mode, sparse
        # and include_last_offset options on the device. Seeds numpy, draws input_size random
        # indices below the table size, and builds an offsets tensor from the given first offset
        # followed by the number of indices. Stores both in the inputs mapping and sets the module
        # name to embeddingbag. The final offsets concatenation mixes a device tensor with a CPU
        # one, so only CPU configs work.
        self.embedding = torch.nn.EmbeddingBag(
            num_embeddings=embeddingbags,
            embedding_dim=dim,
            mode=mode,
            include_last_offset=include_last_offset,
            sparse=sparse,
        ).to(device=device)
        numpy.random.seed((1 << 32) - 1)
        offsets = torch.LongTensor([offset], device=device)
        input = torch.tensor(
            numpy.random.randint(0, embeddingbags, input_size), device=device
        ).long()
        self.inputs = {
            "input": input,
            "offset": torch.cat(
                (offsets, torch.tensor([input.size(0)], dtype=torch.long)), 0
            ),
        }
        self.set_module_name("embeddingbag")

    def forward(self, input, offset):
        # [FUNCTION DOC] Look up and reduce the bags of input indices using offset; return
        # the result.
        return self.embedding(input, offset)


op_bench.generate_pt_test(configs.embeddingbag_short_configs, EmbeddingBagBenchmark)
op_bench.generate_pt_gradient_test(
    configs.embeddingbag_short_configs, EmbeddingBagBenchmark
)


class EmbeddingBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for a plain Embedding module lookup. State: the embedding module and an
    # index tensor in the inputs mapping.
    def init(self, num_embeddings, embedding_dim, input_size, device):
        # [FUNCTION DOC] Build an Embedding module of the given size and dim on the device, seed
        # numpy, draw input_size random indices below the table size, store them in the inputs
        # mapping and set the module name to embedding. Returns nothing.
        self.embedding = torch.nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        ).to(device=device)
        numpy.random.seed((1 << 32) - 1)
        input = torch.tensor(
            numpy.random.randint(0, num_embeddings, input_size), device=device
        ).long()
        self.inputs = {"input": input}
        self.set_module_name("embedding")

    def forward(self, input):
        # [FUNCTION DOC] Return the embedding vectors for the input indices.
        return self.embedding(input)


op_bench.generate_pt_test(configs.embedding_short_configs, EmbeddingBenchmark)
op_bench.generate_pt_gradient_test(configs.embedding_short_configs, EmbeddingBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
