# [MODULE DOC] Operator microbenchmark for single optimizer steps (AdamW, Adam, SGD, RMSprop,
# Adagrad) on one large CUDA float32 parameter with a random gradient, for three sizes. Tests are
# registered from the op list at import time. Includes a memory traffic estimate. Running as a
# script starts the op_bench runner.
import operator_benchmark as op_bench

import torch
import torch.optim as optim


"""Microbenchmarks for optimizer operators."""


optimizer_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["adamw", optim.AdamW],
        ["adam", optim.Adam],
        ["sgd", optim.SGD],
        ["rmsprop", optim.RMSprop],
        ["adagrad", optim.Adagrad],
    ],
)

optimizer_configs_long = op_bench.cross_product_configs(
    shape=[(100000,), (1000000,), (10000000,)],
    device=["cuda"],
    tags=["long"],
)


class OptimizerBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for one optimizer step on a single parameter tensor. State: the
    # optimizer class in op_func, the parameter with a preset gradient, and the optimizer instance.
    def init(self, op_func, device, shape):
        # [FUNCTION DOC] Create a random float32 parameter requiring grad of the given shape on the
        # device and assign it a random gradient. Builds the optimizer with learning rate 0.001
        # over that parameter, adding momentum 0.9 for SGD only. Stores the parameter as a
        # dummy input.
        self.op_func = op_func
        self.param = torch.randn(
            shape, device=device, requires_grad=True, dtype=torch.float32
        )
        self.param.grad = torch.randn(shape, device=device)

        kwargs = {"momentum": 0.9} if op_func == optim.SGD else {}
        self.optimizer = op_func([self.param], lr=0.001, **kwargs)

        self.inputs = {"dummy": self.param}

    def forward(self, dummy):
        # [FUNCTION DOC] Run one optimizer step, which updates the parameter in place, and return
        # the parameter. The dummy argument is unused.
        self.optimizer.step()
        return self.param

    def get_memory_traffic_bytes(self):
        # [FUNCTION DOC] Estimate bytes moved by one step as parameter element count times element
        # size times a fixed multiplier: seven for Adam and AdamW (parameter, gradient and two
        # moment buffers read; parameter and moments written), five for every other optimizer.
        # Memory traffic calculation for bandwidth
        total_elements = self.param.numel()
        bytes_per_element = self.param.element_size()
        # SGD w/ momentum: read(param, grad, momentum) + write(param, momentum) = 5x
        # Adam/AdamW: read(param, grad, exp_avg, exp_avg_sq) + write(param, exp_avg, exp_avg_sq) = 7x
        # Adagrad/RMSprop: read(param, grad, state) + write(param, state) = 5x
        if self.op_func in (optim.Adam, optim.AdamW):
            memory_multiplier = 7
        else:
            memory_multiplier = 5
        return total_elements * bytes_per_element * memory_multiplier


op_bench.generate_pt_tests_from_op_list(
    optimizer_list, optimizer_configs_long, OptimizerBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
