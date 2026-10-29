# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Test checking that FSDP training matches DistributedDataParallel when an
# FSDP-wrapped submodule is called twice within a single forward pass.
# Skips the whole file (exit code 0) if torch.distributed is unavailable or under dev-asan.
# Tests are instantiated per device type for cpu, hpu and xpu via instantiate_device_type_tests.
import sys

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear, Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTestContinuous,
    get_devtype,
    get_full_params,
)
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


class Model(Module):
    # [CLASS DOC] HELPER: Small two-layer model. The inner linear layer (optionally wrapped in
    # its own FSDP unit) is applied twice to the same input in forward, and the two results
    # are summed before the outer linear layer.
    def __init__(self, wrap_fsdp):
        # [FUNCTION DOC] HELPER: Seeds the global RNG with zero for deterministic init, builds
        # the inner linear (4 to 4), wraps it in FSDP if wrap_fsdp is true, and builds the
        # outer linear (4 to 5).
        super().__init__()
        # keep everything deterministic for model initialization
        torch.manual_seed(0)
        self.inner = Linear(4, 4)
        if wrap_fsdp:
            self.inner = FSDP(self.inner)
        self.outer = Linear(4, 5)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs the inner layer twice on the same input x, adds the two
        # outputs, and returns the outer layer applied to the sum.
        # Forward twice.
        i = self.inner(x)
        j = self.inner(x)
        return self.outer(i + j)


class TestMultiForward(FSDPTestContinuous):
    # [CLASS DOC] Distributed test case comparing DDP and FSDP parameters after a short
    # training run with a model that calls one submodule twice per forward.
    def _dist_train(self, wrap_fsdp):
        # [FUNCTION DOC] HELPER: Trains Model for 3 SGD steps (lr 0.1) on a fixed random batch
        # of 64 inputs, on the test device. If wrap_fsdp is true the model and its inner layer
        # are FSDP wrapped and the full unsharded parameters are returned via get_full_params;
        # otherwise the model is wrapped in DistributedDataParallel and its parameter list
        # is returned. Side effect: resets the global RNG seed to zero.
        # keep everything deterministic for input data
        torch.manual_seed(0)
        model = Model(wrap_fsdp).to(device_type.type)
        if wrap_fsdp:
            model = FSDP(model, device_id=device_type.type)
        else:
            model = DistributedDataParallel(model, device_ids=[device_type.type])
        optim = SGD(model.parameters(), lr=0.1)
        in_data = torch.rand(64, 4).to(device_type.type)
        in_data.requires_grad = True
        for _ in range(3):
            out = model(in_data)
            out.sum().backward()
            optim.step()
            optim.zero_grad()
        if wrap_fsdp:
            return get_full_params(model)
        return list(model.parameters())

    @skip_if_lt_x_gpu(2)
    def test_multi_forward(self):
        # [FUNCTION DOC] Requires at least 2 GPUs. Trains once with DDP and once with FSDP
        # and asserts the resulting parameters are equal, showing that FSDP handles a wrapped
        # submodule being run twice in one forward.
        # DDP
        ddp_state = self._dist_train(wrap_fsdp=False)
        # FSDP
        fsdp_state = self._dist_train(wrap_fsdp=True)
        self.assertEqual(ddp_state, fsdp_state)


devices = ("cpu", "hpu", "xpu")
instantiate_device_type_tests(
    TestMultiForward, globals(), only_for=devices, allow_xpu=True
)
if __name__ == "__main__":
    run_tests()
