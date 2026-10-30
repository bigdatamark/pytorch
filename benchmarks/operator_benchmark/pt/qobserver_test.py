# [MODULE DOC] Benchmarks for quantization observers from torch.ao.quantization.observer.
# Builds short and long config sets as dictionaries and op_bench config lists for per-tensor,
# per-channel and histogram observers (device cpu and cuda, except histogram which is cpu
# only), three op lists and two benchmark classes, then registers them at import time. Running
# as a script starts the benchmark runner.
import operator_benchmark as op_bench

import torch
import torch.ao.quantization.observer as obs


qobserver_short_configs_dict = {
    "attr_names": ("C", "M", "N", "dtype", "device"),
    "attrs": (
        (3, 512, 512, torch.quint8, "cpu"),
        (3, 512, 512, torch.quint8, "cuda"),
    ),
    "tags": ("short",),
}

q_hist_observer_short_configs_dict = {
    "attr_names": ("C", "M", "N", "dtype", "device"),
    "attrs": ((3, 512, 512, torch.quint8, "cpu"),),
    "tags": ("short",),
}

qobserver_long_configs_dict = {
    "C": (32, 64),
    "M": (256, 1024),
    "N": (256, 1024),
    "device": ("cpu", "cuda"),
    "dtype": (torch.quint8,),  # dtype doesn't change the timing, keep the same
    "tags": ("long",),
}

q_hist_observer_long_configs_dict = {
    "C": (1, 3, 8),
    "M": (256, 1024),
    "N": (256, 1024),
    "device": ("cpu",),
    "dtype": (torch.quint8,),  # dtype doesn't change the timing, keep the same
    "tags": ("long",),
}


qobserver_per_tensor_configs_short = op_bench.config_list(
    cross_product_configs={
        "qscheme": (torch.per_tensor_affine, torch.per_tensor_symmetric)
    },
    **qobserver_short_configs_dict,
)

qobserver_per_tensor_configs_long = op_bench.cross_product_configs(
    qscheme=(torch.per_tensor_affine, torch.per_tensor_symmetric),
    **qobserver_long_configs_dict,
)

qobserver_per_channel_configs_short = op_bench.config_list(
    cross_product_configs={
        "qscheme": (torch.per_channel_affine, torch.per_channel_symmetric)
    },
    **qobserver_short_configs_dict,
)

qobserver_per_channel_configs_long = op_bench.cross_product_configs(
    qscheme=(torch.per_channel_affine, torch.per_channel_symmetric),
    **qobserver_long_configs_dict,
)

q_hist_observer_per_tensor_configs_short = op_bench.config_list(
    cross_product_configs={
        "qscheme": (torch.per_tensor_affine, torch.per_tensor_symmetric)
    },
    **q_hist_observer_short_configs_dict,
)

q_hist_observer_per_tensor_configs_long = op_bench.cross_product_configs(
    qscheme=(torch.per_tensor_affine, torch.per_tensor_symmetric),
    **q_hist_observer_long_configs_dict,
)


qobserver_per_tensor_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["MinMaxObserver", obs.MinMaxObserver],
        ["MovingAverageMinMaxObserver", obs.MovingAverageMinMaxObserver],
    ],
)

qobserver_per_channel_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["PerChannelMinMaxObserver", obs.PerChannelMinMaxObserver],
        [
            "MovingAveragePerChannelMinMaxObserver",
            obs.MovingAveragePerChannelMinMaxObserver,
        ],
    ],
)

q_hist_observer_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["HistogramObserver", obs.HistogramObserver],
        ["HistogramObserverCalculateQparams", obs.HistogramObserver],
    ],
)


class QObserverBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark that feeds a random float tensor to an observer and then asks for
    # the quantization parameters, so the timing covers both observation and calculate_qparams.
    def init(self, C, M, N, dtype, qscheme, op_func, device):
        # [FUNCTION DOC] Args: C, M, N (input shape), dtype, qscheme, op_func (observer class),
        # device. Stores a random float tensor of shape C x M x N on the device in self.inputs
        # and builds the observer with the dtype and qscheme, moved to the device.
        self.inputs = {"f_input": torch.rand(C, M, N, device=device)}
        self.op_func = op_func(dtype=dtype, qscheme=qscheme).to(device)

    def forward(self, f_input):
        # [FUNCTION DOC] Runs the observer on the input (updating its statistics) and returns
        # the result of calculate_qparams, the scale and zero point.
        self.op_func(f_input)
        return self.op_func.calculate_qparams()


class QObserverBenchmarkCalculateQparams(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark that times only calculate_qparams. The observer is fed the data
    # once during setup so the timed call works on already collected statistics.
    def init(self, C, M, N, dtype, qscheme, op_func, device):
        # [FUNCTION DOC] Args: C, M, N, dtype, qscheme, op_func, device. Creates a random float
        # tensor of shape C x M x N and an observer on the device, runs the observer once on
        # the tensor, and leaves the inputs mapping empty so forward takes no arguments.
        self.f_input = torch.rand(C, M, N, device=device)
        self.q_observer = op_func(dtype=dtype, qscheme=qscheme).to(device)
        self.q_observer(self.f_input)
        self.inputs = {}

    def forward(self):
        # [FUNCTION DOC] Returns the quantization parameters computed by the already populated
        # observer.
        return self.q_observer.calculate_qparams()


op_bench.generate_pt_tests_from_op_list(
    qobserver_per_tensor_list,
    qobserver_per_tensor_configs_short + qobserver_per_tensor_configs_long,
    QObserverBenchmark,
)

op_bench.generate_pt_tests_from_op_list(
    qobserver_per_channel_list,
    qobserver_per_channel_configs_short + qobserver_per_channel_configs_long,
    QObserverBenchmark,
)

op_bench.generate_pt_tests_from_op_list(
    q_hist_observer_list,
    q_hist_observer_per_tensor_configs_short + q_hist_observer_per_tensor_configs_long,
    QObserverBenchmarkCalculateQparams,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
