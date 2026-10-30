# [MODULE DOC] Operator microbenchmarks for convolution modules: Conv1d, ConvTranspose1d, Conv2d,
# ConvTranspose2d, pointwise Conv2d, Conv3d and ConvTranspose3d. Config lists come from the shared
# pt configs module. Each benchmark builds a module and an input, and reports a memory traffic
# estimate (input, weight and output elements times element size; bias is not counted). Forward
# tests are registered for all, gradient tests for non-CPU configs, all at import time. Running as
# a script starts the op_bench runner.
from pt import configs

import operator_benchmark as op_bench

import torch
import torch.nn as nn


"""
Microbenchmarks for Conv1d and ConvTranspose1d operators.
"""


class Conv1dBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for a Conv1d module. State: the input tensor in the inputs mapping and
    # the conv1d module.
    def init(self, IC, OC, kernel, stride, N, L, device):
        # [FUNCTION DOC] Create a random input of shape N x IC x L (requires_grad from auto_set)
        # and a Conv1d module with the given channels, kernel and stride on the device. Sets the
        # module name.
        self.inputs = {
            "input": torch.rand(N, IC, L, device=device, requires_grad=self.auto_set())
        }
        self.conv1d = nn.Conv1d(IC, OC, kernel, stride=stride).to(device=device)
        self.set_module_name("Conv1d")

    def forward(self, input):
        # [FUNCTION DOC] Apply the convolution module to input and return the output.
        return self.conv1d(input)

    def get_memory_traffic_bytes(self):
        # [FUNCTION DOC] Estimate memory traffic in bytes as input plus weight plus output element
        # counts times the element size. The output size is found by running the module once
        # without gradients.
        """Calculate memory traffic for Conv1d: read(input + weight) + write(output)"""
        input_tensor = self.inputs["input"]
        # Run forward to get output shape
        with torch.no_grad():
            output = self.conv1d(input_tensor)

        bytes_per_element = input_tensor.element_size()
        # Input: N × IC × L
        input_elements = input_tensor.numel()
        # Weight: OC × IC × kernel
        weight_elements = self.conv1d.weight.numel()
        # Output: N × OC × L_out
        output_elements = output.numel()

        total_elements = input_elements + weight_elements + output_elements
        return total_elements * bytes_per_element


class ConvTranspose1dBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for a ConvTranspose1d module. State: the input tensor in the inputs
    # mapping and the convtranspose1d module.
    def init(self, IC, OC, kernel, stride, N, L, device):
        # [FUNCTION DOC] Create a random input of shape N x IC x L and a ConvTranspose1d module
        # with the given channels, kernel and stride on the device. Sets the module name.
        self.inputs = {"input": torch.rand(N, IC, L, device=device)}
        self.convtranspose1d = nn.ConvTranspose1d(IC, OC, kernel, stride=stride).to(
            device=device
        )
        self.set_module_name("ConvTranspose1d")

    def forward(self, input):
        # [FUNCTION DOC] Apply the transposed convolution module to input and return the output.
        return self.convtranspose1d(input)

    def get_memory_traffic_bytes(self):
        # [FUNCTION DOC] Estimate memory traffic in bytes as input plus weight plus output element
        # counts times the element size. The output size is found by running the module once
        # without gradients.
        """Calculate memory traffic for ConvTranspose1d: read(input + weight) + write(output)"""
        input_tensor = self.inputs["input"]
        # Run forward to get output shape
        with torch.no_grad():
            output = self.convtranspose1d(input_tensor)

        bytes_per_element = input_tensor.element_size()
        # Input: N × IC × L
        input_elements = input_tensor.numel()
        # Weight: IC × OC × kernel
        weight_elements = self.convtranspose1d.weight.numel()
        # Output: N × OC × L_out
        output_elements = output.numel()

        total_elements = input_elements + weight_elements + output_elements
        return total_elements * bytes_per_element


op_bench.generate_pt_test(
    configs.conv_1d_configs_short + configs.conv_1d_configs_long, Conv1dBenchmark
)
op_bench.generate_pt_gradient_test(
    configs.remove_cpu(configs.conv_1d_configs_short + configs.conv_1d_configs_long),
    Conv1dBenchmark,
)

op_bench.generate_pt_test(
    configs.convtranspose_1d_configs_short
    + configs.conv_1d_configs_short
    + configs.conv_1d_configs_long,
    ConvTranspose1dBenchmark,
)


"""
Microbenchmarks for Conv2d, ConvTranspose2d, and Conv2dPointwise operators.
"""


class Conv2dBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for a Conv2d module with groups and padding. State: input tensor in the
    # inputs mapping and the conv2d module.
    def init(self, IC, OC, kernel, stride, N, H, W, G, pad, device):
        # [FUNCTION DOC] Create a random input of shape N x IC x H x W and a Conv2d module with the
        # given channels, kernel, stride, groups G and padding on the device. Sets the module name.
        self.inputs = {"input": torch.rand(N, IC, H, W, device=device)}
        self.conv2d = nn.Conv2d(
            IC, OC, kernel, stride=stride, groups=G, padding=pad
        ).to(device=device)
        self.set_module_name("Conv2d")

    def forward(self, input):
        # [FUNCTION DOC] Apply the convolution module to input and return the output.
        return self.conv2d(input)

    def get_memory_traffic_bytes(self):
        # [FUNCTION DOC] Estimate memory traffic in bytes as input plus weight plus output element
        # counts times the element size. The output size is found by running the module once
        # without gradients.
        """Calculate memory traffic for Conv2d: read(input + weight) + write(output)"""
        input_tensor = self.inputs["input"]
        # Run forward to get output shape
        with torch.no_grad():
            output = self.conv2d(input_tensor)

        bytes_per_element = input_tensor.element_size()
        # Input: N × IC × H × W
        input_elements = input_tensor.numel()
        # Weight: OC × (IC/G) × kernel × kernel
        weight_elements = self.conv2d.weight.numel()
        # Output: N × OC × H_out × W_out
        output_elements = output.numel()

        total_elements = input_elements + weight_elements + output_elements
        return total_elements * bytes_per_element


class ConvTranspose2dBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for a ConvTranspose2d module with groups and padding. State: input
    # tensor in the inputs mapping and the convtranspose2d module.
    def init(self, IC, OC, kernel, stride, N, H, W, G, pad, device):
        # [FUNCTION DOC] Create a random input of shape N x IC x H x W and a ConvTranspose2d module
        # with the given channels, kernel, stride, groups G and padding on the device. Sets the
        # module name.
        self.inputs = {"input": torch.rand(N, IC, H, W, device=device)}
        self.convtranspose2d = nn.ConvTranspose2d(
            IC, OC, kernel, stride=stride, groups=G, padding=pad
        ).to(device=device)
        self.set_module_name("ConvTranspose2d")

    def forward(self, input):
        # [FUNCTION DOC] Apply the transposed convolution module to input and return the output.
        return self.convtranspose2d(input)

    def get_memory_traffic_bytes(self):
        # [FUNCTION DOC] Estimate memory traffic in bytes as input plus weight plus output element
        # counts times the element size. The output size is found by running the module once
        # without gradients.
        """Calculate memory traffic for ConvTranspose2d: read(input + weight) + write(output)"""
        input_tensor = self.inputs["input"]
        # Run forward to get output shape
        with torch.no_grad():
            output = self.convtranspose2d(input_tensor)

        bytes_per_element = input_tensor.element_size()
        # Input: N × IC × H × W
        input_elements = input_tensor.numel()
        # Weight: IC × (OC/G) × kernel × kernel
        weight_elements = self.convtranspose2d.weight.numel()
        # Output: N × OC × H_out × W_out
        output_elements = output.numel()

        total_elements = input_elements + weight_elements + output_elements
        return total_elements * bytes_per_element


class Conv2dPointwiseBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for pointwise convolution: a Conv2d module whose kernel size is fixed
    # to one, with groups and padding. State: input tensor in the inputs mapping and the
    # conv2d module.
    def init(self, IC, OC, stride, N, H, W, G, pad, device):
        # [FUNCTION DOC] Create a random input of shape N x IC x H x W and a Conv2d module with
        # kernel size one, the given stride, groups G and padding on the device. Sets the
        # module name.
        self.inputs = {"input": torch.rand(N, IC, H, W, device=device)}
        # Use 1 as kernel for pointwise convolution
        self.conv2d = nn.Conv2d(IC, OC, 1, stride=stride, groups=G, padding=pad).to(
            device=device
        )
        self.set_module_name("Conv2dPointwise")

    def forward(self, input):
        # [FUNCTION DOC] Apply the pointwise convolution module to input and return the output.
        return self.conv2d(input)

    def get_memory_traffic_bytes(self):
        # [FUNCTION DOC] Estimate memory traffic in bytes as input plus weight plus output element
        # counts times the element size. The output size is found by running the module once
        # without gradients.
        """Calculate memory traffic for Conv2dPointwise: read(input + weight) + write(output)"""
        input_tensor = self.inputs["input"]
        # Run forward to get output shape
        with torch.no_grad():
            output = self.conv2d(input_tensor)

        bytes_per_element = input_tensor.element_size()
        # Input: N × IC × H × W
        input_elements = input_tensor.numel()
        # Weight: OC × (IC/G) × 1 × 1
        weight_elements = self.conv2d.weight.numel()
        # Output: N × OC × H_out × W_out
        output_elements = output.numel()

        total_elements = input_elements + weight_elements + output_elements
        return total_elements * bytes_per_element


op_bench.generate_pt_test(
    configs.conv_2d_configs_short + configs.conv_2d_configs_long, Conv2dBenchmark
)
op_bench.generate_pt_test(
    configs.conv_2d_configs_short + configs.conv_2d_configs_long,
    ConvTranspose2dBenchmark,
)
op_bench.generate_pt_test(
    configs.conv_2d_pw_configs_short + configs.conv_2d_pw_configs_long,
    Conv2dPointwiseBenchmark,
)
op_bench.generate_pt_gradient_test(
    configs.remove_cpu(configs.conv_2d_configs_short + configs.conv_2d_configs_long),
    Conv2dBenchmark,
)
op_bench.generate_pt_gradient_test(
    configs.remove_cpu(configs.conv_2d_configs_short + configs.conv_2d_configs_long),
    ConvTranspose2dBenchmark,
)
op_bench.generate_pt_gradient_test(
    configs.remove_cpu(
        configs.conv_2d_pw_configs_short + configs.conv_2d_pw_configs_long
    ),
    Conv2dPointwiseBenchmark,
)


"""
Microbenchmarks for Conv3d and ConvTranspose3d operators.
"""


class Conv3dBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for a Conv3d module with a configurable dtype (float32 by default).
    # State: input tensor in the inputs mapping and the conv3d module.
    def init(self, IC, OC, kernel, stride, N, D, H, W, device, dtype=torch.float32):
        # [FUNCTION DOC] Create a random input of shape N x IC x D x H x W with the dtype and a
        # Conv3d module with the given channels, kernel and stride, moved to the device and dtype.
        # Sets the module name.
        self.inputs = {"input": torch.rand(N, IC, D, H, W, device=device, dtype=dtype)}
        self.conv3d = nn.Conv3d(IC, OC, kernel, stride=stride).to(
            device=device, dtype=dtype
        )
        self.set_module_name("Conv3d")

    def forward(self, input):
        # [FUNCTION DOC] Apply the convolution module to input and return the output.
        return self.conv3d(input)

    def get_memory_traffic_bytes(self):
        # [FUNCTION DOC] Estimate memory traffic in bytes as input plus weight plus output element
        # counts times the element size. The output size is found by running the module once
        # without gradients.
        """Calculate memory traffic for Conv3d: read(input + weight) + write(output)"""
        input_tensor = self.inputs["input"]
        # Run forward to get output shape
        with torch.no_grad():
            output = self.conv3d(input_tensor)

        bytes_per_element = input_tensor.element_size()
        # Input: N × IC × D × H × W
        input_elements = input_tensor.numel()
        # Weight: OC × IC × kernel × kernel × kernel
        weight_elements = self.conv3d.weight.numel()
        # Output: N × OC × D_out × H_out × W_out
        output_elements = output.numel()

        total_elements = input_elements + weight_elements + output_elements
        return total_elements * bytes_per_element


class ConvTranspose3dBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for a ConvTranspose3d module with a configurable dtype (float32 by
    # default). State: input tensor in the inputs mapping and the convtranspose3d module.
    def init(self, IC, OC, kernel, stride, N, D, H, W, device, dtype=torch.float32):
        # [FUNCTION DOC] Create a random input of shape N x IC x D x H x W with the dtype and a
        # ConvTranspose3d module with the given channels, kernel and stride, moved to the device
        # and dtype. Sets the module name.
        self.inputs = {"input": torch.rand(N, IC, D, H, W, device=device, dtype=dtype)}
        self.convtranspose3d = nn.ConvTranspose3d(IC, OC, kernel, stride=stride).to(
            device=device, dtype=dtype
        )
        self.set_module_name("ConvTranspose3d")

    def forward(self, input):
        # [FUNCTION DOC] Apply the transposed convolution module to input and return the output.
        return self.convtranspose3d(input)

    def get_memory_traffic_bytes(self):
        # [FUNCTION DOC] Estimate memory traffic in bytes as input plus weight plus output element
        # counts times the element size. The output size is found by running the module once
        # without gradients.
        """Calculate memory traffic for ConvTranspose3d: read(input + weight) + write(output)"""
        input_tensor = self.inputs["input"]
        # Run forward to get output shape
        with torch.no_grad():
            output = self.convtranspose3d(input_tensor)

        bytes_per_element = input_tensor.element_size()
        # Input: N × IC × D × H × W
        input_elements = input_tensor.numel()
        # Weight: IC × OC × kernel × kernel × kernel
        weight_elements = self.convtranspose3d.weight.numel()
        # Output: N × OC × D_out × H_out × W_out
        output_elements = output.numel()

        total_elements = input_elements + weight_elements + output_elements
        return total_elements * bytes_per_element


op_bench.generate_pt_test(configs.conv_3d_configs_short, Conv3dBenchmark)
op_bench.generate_pt_test(configs.conv_3d_configs_short, ConvTranspose3dBenchmark)
op_bench.generate_pt_gradient_test(
    configs.remove_cpu(configs.conv_3d_configs_long), Conv3dBenchmark
)
op_bench.generate_pt_gradient_test(
    configs.remove_rocm_slow_3d_transpose(
        configs.remove_cpu(configs.conv_3d_configs_long)
    ),
    ConvTranspose3dBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
