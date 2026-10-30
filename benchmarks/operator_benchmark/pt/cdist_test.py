# [MODULE DOC] Operator microbenchmark for torch.cdist (pairwise p-norm distances) over a cross
# product of batch, row counts, feature size and norm order p, on cpu, cuda and mps. Registers a
# forward test and a gradient test at import time. Running as a script starts the op_bench runner.
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for torch.cdist (pairwise distances)."""


cdist_configs = op_bench.cross_product_configs(
    B=[1, 8],
    P=[64, 512],
    R=[64, 512],
    M=[16, 64],
    p=[1.0, 2.0, 3.0, 4.0],
    device=["cpu", "cuda", "mps"],
    tags=["long"],
)


class CdistBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for cdist between two random batched point sets. State in the inputs
    # mapping: x1, x2 and the norm order p.
    def init(self, device, B, P, R, M, p):
        # [FUNCTION DOC] Create random x1 of shape BxPxM and x2 of shape BxRxM on the device, with
        # requires_grad set by the harness auto_set flag, and store them with p in the inputs
        # mapping. Sets the module name to cdist. Returns nothing.
        self.inputs = {
            "x1": torch.rand(B, P, M, device=device, requires_grad=self.auto_set()),
            "x2": torch.rand(B, R, M, device=device, requires_grad=self.auto_set()),
            "p": p,
        }
        self.set_module_name("cdist")

    def forward(self, x1, x2, p):
        # [FUNCTION DOC] Return the batched pairwise distance matrix (shape BxPxR) between x1 and
        # x2 using norm order p.
        return torch.cdist(x1, x2, p=p)


op_bench.generate_pt_test(
    cdist_configs,
    CdistBenchmark,
)

op_bench.generate_pt_gradient_test(
    cdist_configs,
    CdistBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
