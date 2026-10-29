# Owner(s): ["oncall: distributed"]
import sys

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear, Module
from torch.optim import SGD
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTestContinuous
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    subtest,
    TEST_WITH_DEV_DBG_ASAN,
)


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestInput(FSDPTestContinuous):
    # [CLASS DOC] Single-rank multi-process test case covering FSDP handling of list and
    # dict forward inputs: training works with them, and container identity is preserved.
    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Property giving the number of ranks to spawn; always 1.
        return 1

    @skip_if_lt_x_gpu(1)
    @parametrize("input_cls", [subtest(dict, name="dict"), subtest(list, name="list")])
    def test_input_type(self, device, input_cls):
        # [FUNCTION DOC] Test: wraps a small linear model in FSDP and trains it for 5 SGD
        # steps where each input is either a one-element list or a dict keyed "in" holding a
        # random tensor that requires grad. Passes if forward, backward and step succeed.
        """Test FSDP with input being a list or a dict, only single GPU."""

        class Model(Module):
            # [CLASS DOC] HELPER: Module with one 4x4 linear layer whose forward accepts
            # either a list (uses the first element) or a dict (uses the "in" entry).
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the single 4x4 linear layer.
                super().__init__()
                self.layer = Linear(4, 4)

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Extracts the tensor from the list or dict input and
                # applies the layer; raises AssertionError if input is neither a list nor dict.
                if isinstance(input, list):
                    input = input[0]
                else:
                    if not isinstance(input, dict):
                        raise AssertionError(
                            f"Expected dict, got {type(input)}: {input}"
                        )
                    input = input["in"]
                return self.layer(input)

        fsdp_kwargs = {
            "device_id": device,
        }
        model = FSDP(Model().to(device), **fsdp_kwargs)
        optim = SGD(model.parameters(), lr=0.1)
        for _ in range(5):
            in_data = torch.rand(64, 4).to(device)
            in_data.requires_grad = True
            if input_cls is list:
                in_data = [in_data]
            else:
                self.assertTrue(input_cls is dict)
                in_data = {"in": in_data}
            out = model(in_data)
            out.sum().backward()
            optim.step()
            optim.zero_grad()

    @skip_if_lt_x_gpu(1)
    @parametrize("input_cls", [subtest(dict, name="dict"), subtest(list, name="list")])
    def test_input_identity_preserved(self, device, input_cls):
        # [FUNCTION DOC] Test: runs one FSDP forward with a list or dict input on a model
        # that mutates its input (appends a tensor to the list, or adds an "added" key to the
        # dict). Asserts the caller's own container shows the mutation, i.e. FSDP did not
        # copy the container.
        """
        Test that FSDP preserves object identity for container inputs.
        """

        class MutatingModel(Module):
            # [CLASS DOC] HELPER: Module with one 4x4 linear layer whose forward mutates its
            # list or dict input in place by adding a random tensor before using entry 0 or "in".
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the single 4x4 linear layer.
                super().__init__()
                self.layer = Linear(4, 4)

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Appends (list) or inserts under "added" (dict) a
                # random 4x4 tensor on the input's device, then applies the layer to the first
                # element or the "in" entry. Side effect: mutates the caller's container.
                if isinstance(input, list):
                    input.append(torch.randn(4, 4, device=input[0].device))
                    x = input[0]
                else:
                    assert isinstance(input, dict), input  # noqa: S101
                    input["added"] = torch.randn(4, 4, device=input["in"].device)
                    x = input["in"]
                return self.layer(x)

        fsdp_kwargs = {
            "device_id": device,
        }
        model = FSDP(MutatingModel().to(device), **fsdp_kwargs)

        in_data = torch.rand(64, 4, device=device)
        if input_cls is list:
            in_data = [in_data]
            initial_len = len(in_data)
            model(in_data)
            # The list should have been mutated (appended to)
            self.assertEqual(len(in_data), initial_len + 1)
        else:
            self.assertTrue(input_cls is dict)
            in_data = {"in": in_data}
            model(in_data)
            # The dict should have been mutated (new key added)
            self.assertIn("added", in_data)


devices = ("cuda", "hpu", "xpu")
instantiate_device_type_tests(TestInput, globals(), only_for=devices, allow_xpu=True)
if __name__ == "__main__":
    run_tests()
