# [MODULE DOC] Example benchmark for the add operator with automatic requires_grad
# assignment: cross product of sizes (8 or 1 by 8 or 2 by 8 or 4) under the short tag,
# registered both as a forward test and as a backward (gradient) test. Registers at import
# time and runs the benchmark runner main function when executed as a script.
import operator_benchmark as op_bench

import torch


add_configs = op_bench.cross_product_configs(
    M=[8, 1], N=[8, 2], K=[8, 4], tags=["short"]
)


# This benchmark uses the auto_set to automatically set requires_grad
# for both inputs. The test name can also be used for filtering.
class AddBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Add benchmark whose two inputs get their requires_grad
    # flag from the auto_set mechanism so that each input can be
    # differentiated in turn in backward tests. Module name is add.
    def init(self, M, N, K):
        # [FUNCTION DOC] Creates two random three dimensional inputs of the given sizes, each with
        # the requires_grad value returned by auto_set, and sets the module name to add.
        self.input_one = torch.rand(M, N, K, requires_grad=self.auto_set())
        self.input_two = torch.rand(M, N, K, requires_grad=self.auto_set())
        self.set_module_name("add")

    def forward(self):
        # [FUNCTION DOC] Returns the elementwise sum of the two stored inputs.
        return torch.add(self.input_one, self.input_two)


op_bench.generate_pt_test(add_configs, AddBenchmark)
op_bench.generate_pt_gradient_test(add_configs, AddBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
