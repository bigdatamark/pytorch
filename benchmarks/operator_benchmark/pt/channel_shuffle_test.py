# [MODULE DOC] Operator microbenchmark for channel_shuffle on random 4-D image batches, in
# contiguous and channels_last memory formats. Provides a long cross-product config set and a short
# explicit one. Registers tests at import time. Running as a script starts the runner.
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for channel_shuffle operator."""


# Configs for PT channel_shuffle operator
channel_shuffle_long_configs = op_bench.cross_product_configs(
    batch_size=[4, 8],
    channels_per_group=[32, 64],
    height=[32, 64],
    width=[32, 64],
    groups=[4, 8],
    channel_last=[True, False],
    tags=["long"],
)


channel_shuffle_short_configs = op_bench.config_list(
    attr_names=["batch_size", "channels_per_group", "height", "width", "groups"],
    attrs=[
        [2, 16, 16, 16, 2],
        [2, 32, 32, 32, 2],
        [4, 32, 32, 32, 4],
        [4, 64, 64, 64, 4],
        [8, 64, 64, 64, 8],
        [16, 64, 64, 64, 16],
    ],
    cross_product_configs={
        "channel_last": [True, False],
    },
    tags=["short"],
)


class ChannelSHuffleBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for channel_shuffle on a random NCHW tensor. The class name has a
    # capitalisation typo (ChannelSHuffle) that is harmless. State: input_data and groups.
    def init(self, batch_size, channels_per_group, height, width, groups, channel_last):
        # [FUNCTION DOC] Create a random CPU tensor with channels equal to channels_per_group times
        # groups and the given batch, height and width. If channel_last is true, convert it to the
        # channels_last memory format. Stores it with groups in the inputs mapping and sets the
        # module name to channel_shuffle. There is no device argument, so tensors are always
        # on CPU.
        channels = channels_per_group * groups
        data_shape = (batch_size, channels, height, width)
        input_data = torch.rand(data_shape)
        if channel_last:
            input_data = input_data.contiguous(memory_format=torch.channels_last)
        self.inputs = {"input_data": input_data, "groups": groups}
        self.set_module_name("channel_shuffle")

    def forward(self, input_data, groups: int):
        # [FUNCTION DOC] Return input_data with its channels shuffled across the given number
        # of groups.
        return torch.channel_shuffle(input_data, groups)


op_bench.generate_pt_test(
    channel_shuffle_short_configs + channel_shuffle_long_configs,
    ChannelSHuffleBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
