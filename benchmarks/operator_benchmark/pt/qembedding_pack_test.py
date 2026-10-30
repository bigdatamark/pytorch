# [MODULE DOC] Benchmarks for quantized embedding-bag weight packing and unpacking ops (byte,
# 4-bit and 2-bit variants) in torch.ops.quantized. Defines config sets (short, long and
# three-dimensional), two op lists (prepack, unpack) and four benchmark classes, then registers
# the pairings at import time. Running as a script starts the benchmark runner.
import operator_benchmark as op_bench

import torch


embeddingbag_conversion_short_configs = op_bench.cross_product_configs(
    num_embeddings=(80,), embedding_dim=(128, 256, 512), tags=("short",)
)

embeddingbag_conversion_long_configs = op_bench.cross_product_configs(
    num_embeddings=(100, 120, 1000),
    embedding_dim=(16, 64, 128, 256, 512, 1024, 2048),
    tags=("long",),
)

embeddingbag_conversion_three_dim_configs = op_bench.cross_product_configs(
    num_embeddings=(80,),
    embedding_dim=(128, 256, 512),
    batch_size=(10,),
    tags=("short",),
)

conversion_ops = op_bench.op_list(
    attrs=(
        ("qembeddingbag_byte_prepack", torch.ops.quantized.embedding_bag_byte_prepack),
        ("qembeddingbag_4bit_prepack", torch.ops.quantized.embedding_bag_4bit_prepack),
        ("qembeddingbag_2bit_prepack", torch.ops.quantized.embedding_bag_2bit_prepack),
    ),
    attr_names=("op_name", "op_func"),
)

unpack_ops = op_bench.op_list(
    attrs=(
        ("qembeddingbag_byte_unpack", torch.ops.quantized.embedding_bag_byte_unpack),
        ("qembeddingbag_4bit_unpack", torch.ops.quantized.embedding_bag_4bit_unpack),
        ("qembeddingbag_2bit_unpack", torch.ops.quantized.embedding_bag_2bit_unpack),
    ),
    attr_names=("op_name", "op_func"),
)


class EmbeddingBagFloatToFusedBase(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark of a prepack op converting a 2D float weight into the fused
    # quantized format.
    def init(self, num_embeddings, embedding_dim, op_func):
        # [FUNCTION DOC] Args: num_embeddings, embedding_dim, op_func. Stores a random float
        # weight of shape num_embeddings x embedding_dim (values shifted by 1) in self.inputs
        # and keeps op_func.
        self.inputs = {
            "weight": torch.rand(num_embeddings, embedding_dim, dtype=torch.float) + 1
        }
        self.op_func = op_func

    def forward(self, weight):
        # [FUNCTION DOC] Runs the prepack op on the weight and returns the packed tensor.
        return self.op_func(weight)


class EmbeddingBagThreeDimFloatToFusedBase(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark of a prepack op on a 3D float weight with a leading batch
    # dimension.
    def init(self, num_embeddings, embedding_dim, batch_size, op_func):
        # [FUNCTION DOC] Args: num_embeddings, embedding_dim, batch_size, op_func. Stores a
        # random float weight of shape batch_size x num_embeddings x embedding_dim (values
        # shifted by 1) and keeps op_func.
        self.inputs = {
            "weight": torch.rand(
                batch_size, num_embeddings, embedding_dim, dtype=torch.float
            )
            + 1
        }
        self.op_func = op_func

    def forward(self, weight):
        # [FUNCTION DOC] Runs the prepack op on the 3D weight and returns the packed tensor.
        return self.op_func(weight)


class EmbeddingBagFusedToFloatBase(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark of an unpack op converting a packed 2D uint8 tensor back to float.
    def init(self, num_embeddings, embedding_dim, op_func):
        # [FUNCTION DOC] Args: num_embeddings, embedding_dim, op_func. Makes a random float
        # tensor with embedding_dim + 8 columns (extra bytes for per-row scale and bias) and
        # casts it to uint8 as the packed input. This is not a real packed buffer, only the
        # right shape and dtype.
        weight = torch.randn(num_embeddings, embedding_dim + 8, dtype=torch.float)
        self.inputs = {"packed_weight": weight.to(torch.uint8)}
        self.op_func = op_func

    def forward(self, packed_weight):
        # [FUNCTION DOC] Runs the unpack op on the packed weight and returns the result.
        return self.op_func(packed_weight)


class EmbeddingBagThreeDimFusedToFloatBase(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark of an unpack op on a packed 3D uint8 tensor with a batch dimension.
    def init(self, num_embeddings, embedding_dim, batch_size, op_func):
        # [FUNCTION DOC] Args: num_embeddings, embedding_dim, batch_size, op_func. Builds a
        # random float tensor of shape batch_size x num_embeddings x (embedding_dim + 8) and
        # casts it to uint8 as the packed input (shape only, not a real packed buffer). Keeps
        # op_func.
        weight = torch.randn(
            batch_size, num_embeddings, embedding_dim + 8, dtype=torch.float
        )
        self.inputs = {"packed_weight": weight.to(torch.uint8)}
        self.op_func = op_func

    def forward(self, packed_weight):
        # [FUNCTION DOC] Runs the unpack op on the 3D packed weight and returns the result.
        return self.op_func(packed_weight)


op_bench.generate_pt_tests_from_op_list(
    conversion_ops,
    embeddingbag_conversion_short_configs + embeddingbag_conversion_long_configs,
    EmbeddingBagFloatToFusedBase,
)
op_bench.generate_pt_tests_from_op_list(
    unpack_ops,
    embeddingbag_conversion_short_configs + embeddingbag_conversion_long_configs,
    EmbeddingBagFusedToFloatBase,
)
op_bench.generate_pt_tests_from_op_list(
    conversion_ops,
    embeddingbag_conversion_three_dim_configs,
    EmbeddingBagThreeDimFloatToFusedBase,
)
op_bench.generate_pt_tests_from_op_list(
    unpack_ops,
    embeddingbag_conversion_three_dim_configs,
    EmbeddingBagThreeDimFusedToFloatBase,
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
