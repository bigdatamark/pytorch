# [MODULE DOC] Example benchmark of add with a single size (8 by 8 by 8) on cuda and cpu under
# the short tag, registered both as forward and as backward (gradient) tests. Registers at
# import time and runs the benchmark runner main function when executed as a script.
import operator_benchmark as op_bench

import torch


add_configs = op_bench.cross_product_configs(
    M=[8], N=[8], K=[8], device=["cuda", "cpu"], tags=["short"]
)


class AddBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Add benchmark with module name add whose two inputs
    # both require gradients, on the device given by the config.
    def init(self, M, N, K, device):
        # [FUNCTION DOC] Creates two random three dimensional tensors on the given
        # device, both with gradients enabled, and sets the module name to add.
        self.input_one = torch.rand(M, N, K, device=device, requires_grad=True)
        self.input_two = torch.rand(M, N, K, device=device, requires_grad=True)
        self.set_module_name("add")

    def forward(self):
        # [FUNCTION DOC] Returns the elementwise sum of the two stored inputs.
        return torch.add(self.input_one, self.input_two)


op_bench.generate_pt_test(add_configs, AddBenchmark)
op_bench.generate_pt_gradient_test(add_configs, AddBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
