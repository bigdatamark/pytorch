# [MODULE DOC] Operator microbenchmarks for quantization-aware-training Embedding and EmbeddingBag
# modules from torch.ao.nn.qat, using the default embedding QAT qconfig and the shared embedding
# configs from the pt configs module. EmbeddingBag configs with sparse set are filtered out because
# QAT does not support them. Forward and gradient tests are registered at import time. Running as a
# script starts the op_bench runner.
import numpy
from pt import configs

import operator_benchmark as op_bench

import torch
import torch.ao.nn.qat as nnqat
from torch.ao.quantization import default_embedding_qat_qconfig


"""
Microbenchmarks for QAT Embedding + EmbeddingBag operators.
"""


class QATEmbeddingBagBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for a QAT EmbeddingBag module. State: the embedding module, plus an
    # index tensor and an offsets tensor in the inputs mapping.
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
        # [FUNCTION DOC] Build a QAT EmbeddingBag with the given table size, dim, mode, sparse and
        # include_last_offset options and the default embedding QAT qconfig on the device. Seeds
        # numpy, draws random indices below the table size, and builds an offsets tensor from the
        # first offset followed by the number of indices. Stores both in the inputs mapping and
        # sets the module name to qatEmbeddingBag. The offsets concatenation mixes a device tensor
        # with a CPU one, so only CPU configs work.
        qconfig = default_embedding_qat_qconfig
        self.embedding = nnqat.EmbeddingBag(
            num_embeddings=embeddingbags,
            embedding_dim=dim,
            mode=mode,
            include_last_offset=include_last_offset,
            sparse=sparse,
            device=device,
            qconfig=qconfig,
        )
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
        self.set_module_name("qatEmbeddingBag")

    def forward(self, input, offset):
        # [FUNCTION DOC] Look up and reduce the bags of input indices using offset; return
        # the result.
        return self.embedding(input, offset)


# Currently, EmbeddingBag QAT does not support sparse embeddings.
embeddingbag_short_dense_configs = [
    config
    for config in configs.embeddingbag_short_configs
    if {"sparse": True} not in config
]

op_bench.generate_pt_test(embeddingbag_short_dense_configs, QATEmbeddingBagBenchmark)
op_bench.generate_pt_gradient_test(
    embeddingbag_short_dense_configs, QATEmbeddingBagBenchmark
)


class QATEmbeddingBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for a QAT Embedding module. State: the embedding module and an index
    # tensor, also kept as an attribute named input.
    def init(self, num_embeddings, embedding_dim, input_size, device):
        # [FUNCTION DOC] Build a QAT Embedding of the given size and dim with the default embedding
        # QAT qconfig on the device (the qconfig is also assigned again afterwards). Seeds numpy,
        # draws input_size random indices below the table size, stores them in the inputs mapping
        # and sets the module name to qatEmbedding.
        qconfig = default_embedding_qat_qconfig
        self.embedding = nnqat.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            qconfig=qconfig,
            device=device,
        )
        self.embedding.qconfig = default_embedding_qat_qconfig
        numpy.random.seed((1 << 32) - 1)
        self.input = torch.tensor(
            numpy.random.randint(0, num_embeddings, input_size), device=device
        ).long()
        self.inputs = {"input": self.input}
        self.set_module_name("qatEmbedding")

    def forward(self, input):
        # [FUNCTION DOC] Return the embedding vectors for the input indices.
        return self.embedding(input)


op_bench.generate_pt_test(configs.embedding_short_configs, QATEmbeddingBenchmark)
op_bench.generate_pt_gradient_test(
    configs.embedding_short_configs, QATEmbeddingBenchmark
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
