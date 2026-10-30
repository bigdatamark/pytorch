# [MODULE DOC] Operator benchmark for comparison ops (eq, ne, lt, gt, le, ge) on quantized
# tensors. Builds a cross product of size, quantized dtype, contiguity, scalar-vs-tensor second
# operand and out-variant flags, then registers one benchmark per op via
# generate_pt_tests_from_op_list. Registration happens at import time; running the file as a
# script starts the benchmark runner.
import operator_benchmark as op_bench

import torch


qcomparators_configs = op_bench.cross_product_configs(
    N=(8, 64),
    dtype=(torch.quint8, torch.qint8, torch.qint32),
    contig=(False, True),
    other_scalar=(False, True),
    out_variant=(False, True),
    tags=("short",),
)

qcomparators_ops = op_bench.op_list(
    attrs=(
        ("eq", torch.eq),
        ("ne", torch.ne),
        ("lt", torch.lt),
        ("gt", torch.gt),
        ("le", torch.le),
        ("ge", torch.ge),
    ),
    attr_names=("op_name", "op_func"),
)


class QComparatorBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark that applies one quantized comparison op to a square quantized
    # tensor. The op callable is injected by the op list; init stores it and the inputs,
    # forward runs it.
    def init(self, N, dtype, contig, other_scalar, out_variant, op_func):
        # [FUNCTION DOC] Builds the benchmark inputs. Args: N (side of the square tensor),
        # dtype (quantized dtype), contig (whether the first operand stays contiguous),
        # other_scalar (compare against an integer scalar instead of a tensor), out_variant
        # (write into a preallocated bool tensor), op_func (the comparison callable). Creates a
        # random float tensor in roughly [-128, 128), quantizes it with scale 1.0 and zero
        # point 0, and clones it as the second operand. When not contiguous, the first operand
        # is permuted with reversed dimensions. Stores the op in self.qop and all inputs in the
        # self.inputs mapping. Returns nothing.
        # TODO: Consider more diverse shapes
        f_input = (torch.rand(N, N) - 0.5) * 256
        scale = 1.0
        zero_point = 0

        q_input_a = torch.quantize_per_tensor(
            f_input, scale=scale, zero_point=zero_point, dtype=dtype
        )
        q_input_b = q_input_a.clone()

        if not contig:
            permute_dims = list(range(f_input.ndim))[::-1]
            q_input_a = q_input_a.permute(permute_dims)

        self.qop = op_func
        self.inputs = {
            "q_input_a": q_input_a,
            "q_input_b": q_input_b,
            "out_variant": out_variant,
            "other_scalar": other_scalar,
        }

    def forward(self, q_input_a, q_input_b, out_variant: bool, other_scalar: bool):
        # [FUNCTION DOC] Runs the comparison. Args are the two quantized tensors plus the
        # out_variant and other_scalar flags. Picks one of four calls: first operand vs the
        # scalar 42 or vs the second tensor, each with or without a zero-dim bool out tensor.
        # Returns the comparison result.
        if out_variant:
            if other_scalar:
                return self.qop(q_input_a, 42, out=torch.tensor(True, dtype=torch.bool))
            else:
                return self.qop(
                    q_input_a, q_input_b, out=torch.tensor(True, dtype=torch.bool)
                )
        else:
            if other_scalar:
                return self.qop(q_input_a, 42)
            else:
                return self.qop(q_input_a, q_input_b)


op_bench.generate_pt_tests_from_op_list(
    qcomparators_ops, qcomparators_configs, QComparatorBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
