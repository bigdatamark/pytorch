# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Test that a model already containing a nested FSDP unit can be wrapped in a
# new top-level FSDP again (e.g. for inference after training) and give the same output.
# Skips the whole file (exit code 0) if torch.distributed is unavailable or under dev-asan.
# Tests are instantiated per device type for cuda, hpu and xpu via instantiate_device_type_tests.
import sys

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear, Module, Sequential
from torch.optim import SGD
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTestContinuous, get_devtype
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN


device_type = torch.device(get_devtype())

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class InnerModel(Module):
    # [CLASS DOC] HELPER: Module holding a Sequential with a single FSDP-wrapped linear
    # layer (5 to 5), so it already contains a nested FSDP unit.
    def __init__(self, device):
        # [FUNCTION DOC] HELPER: Builds the Sequential with one FSDP-wrapped linear. The device
        # argument is unused; the FSDP device_id comes from the module level device_type.
        super().__init__()
        self.layers = Sequential(FSDP(Linear(5, 5), device_id=device_type.type))

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the output of the wrapped Sequential on x.
        return self.layers(x)


class TestMultipleWrapping(FSDPTestContinuous):
    # [CLASS DOC] Distributed test case for re-wrapping a model with nested FSDP in a new FSDP.
    @skip_if_lt_x_gpu(2)
    def test_multiple_wrapping(self, device):
        # [FUNCTION DOC] Requires at least 2 GPUs. Wraps InnerModel in FSDP and trains it for 3
        # SGD steps, then runs one forward to get a reference output. Wraps the same inner
        # model in a second FSDP and asserts its output on the same input equals the first.
        """
        This test simulates wrapping the module after training to run inference.
        This is required in cases where later in a session, the model is wrapped again in FSDP but
        contains nested FSDP wrappers within the module.
        """
        inner_model = InnerModel(device)
        model = FSDP(inner_model).to(device_type.type)
        optim = SGD(model.parameters(), lr=0.1)
        for _ in range(3):
            input = torch.rand((1, 5), dtype=torch.float).to(device_type.type)
            input.requires_grad = True
            output = model(input)
            output.sum().backward()
            optim.step()
            optim.zero_grad()
        input = torch.rand((1, 5), dtype=torch.float).to(device_type.type)
        output = model(input)
        # second time to rewrap the inner model
        # rewrapped_model = FSDP(inner_model, device_id=device)
        rewrapped_model = FSDP(inner_model).to(device_type.type)
        rewrapped_output = rewrapped_model(input)
        self.assertEqual(output, rewrapped_output)


devices = ("cuda", "hpu", "xpu")
instantiate_device_type_tests(
    TestMultipleWrapping, globals(), only_for=devices, allow_xpu=True
)
if __name__ == "__main__":
    run_tests()
