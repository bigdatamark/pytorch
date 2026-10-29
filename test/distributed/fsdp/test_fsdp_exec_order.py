# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Multi-GPU tests of FSDP's forward execution-order validation (enabled by the
# detailed distributed debug level). Uses a model with two alternative computation paths to
# check that differing all-gather order across ranks raises an error in the first iteration,
# warns in later iterations, and does not warn when only toggling train/eval mode.
# Import-time side effects: computes the device type; exits with status 0 if distributed is
# unavailable or under dev-asan; instantiates the device-generic test class for cuda/hpu/xpu.
# Running the file calls run_tests().

import sys
import warnings
from contextlib import nullcontext

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTestContinuous, get_devtype
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)


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


class Model(torch.nn.Module):
    # [CLASS DOC] HELPER: Small model with a shared first layer and two mutually exclusive
    # second stages (layer1 versus the three-layer layer2) selected by the use_alt_path flag.
    # Only the active path's parameters require grad (layer2 starts frozen). Both stages have
    # 36 flattened elements so mismatched collectives could silently pair up without checks.
    """
    Model that supports two computation paths: `layer0` -> `layer1` and
    `layer0` -> `layer2`. Notably, both `layer1` and `layer2` have 36 elements
    when flattened, which means that their corresponding all-gathers and
    reduce-scatters may be silently matched if we do not perform any checks.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds layer0 (5 to 6), layer1 (6 to 6, no bias) and layer2
        # (6 to 3 to 6 with ReLU, no bias), sets the normal path as active and freezes all
        # layer2 parameters.
        super().__init__()
        self.layer0 = torch.nn.Linear(5, 6)
        self.layer1 = torch.nn.Linear(6, 6, bias=False)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(6, 3, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 6, bias=False),
        )
        self.relu = torch.nn.ReLU()
        self.use_alt_path = False
        for param in self.layer2.parameters():
            param.requires_grad = False

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies layer0 and ReLU, then either layer2 (alternate path)
        # or layer1 (normal path) followed by ReLU, depending on use_alt_path. Returns result.
        # `layer0` -> `layer1` (normal)
        # `layer0` -> `layer2` (alternate)
        z = self.relu(self.layer0(x))
        z = (
            self.relu(self.layer2(z))
            if self.use_alt_path
            else self.relu(self.layer1(z))
        )
        return z

    def get_input(self, device):
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random 8 by 5 batch
        # moved to the given device.
        return (torch.randn((8, 5)).to(device),)

    def get_loss(self, input, output):
        # [FUNCTION DOC] HELPER: Loss is the sum of the output; input is unused.
        return output.sum()

    def run_backward(self, loss):
        # [FUNCTION DOC] HELPER: Runs backward on the loss.
        loss.backward()

    def flip_path(self):
        # [FUNCTION DOC] HELPER: Switches the active path: freezes the currently active
        # stage's parameters, unfreezes the other stage's, and toggles use_alt_path. Mutates
        # requires_grad flags.
        params_to_freeze = (
            self.layer2.parameters() if self.use_alt_path else self.layer1.parameters()
        )
        params_to_unfreeze = (
            self.layer1.parameters() if self.use_alt_path else self.layer2.parameters()
        )
        for param in params_to_freeze:
            param.requires_grad = False
        for param in params_to_unfreeze:
            param.requires_grad = True
        self.use_alt_path = not self.use_alt_path

    @staticmethod
    def wrap(sharding_strategy: ShardingStrategy, device):
        # [FUNCTION DOC] HELPER: Static factory. Creates a Model, wraps layer1 and layer2 each
        # in their own FSDP instance and then the whole model in a root FSDP, all with the
        # given sharding strategy and device_id, and returns the root moved to the device.
        model = Model()
        model.layer1 = FSDP(
            model.layer1, sharding_strategy=sharding_strategy, device_id=device
        )
        model.layer2 = FSDP(
            model.layer2, sharding_strategy=sharding_strategy, device_id=device
        )
        fsdp_model = FSDP(model, sharding_strategy=sharding_strategy, device_id=device)
        return fsdp_model.to(device)


class TestFSDPExecOrder(FSDPTestContinuous):
    # [CLASS DOC] Multi-process test case for forward execution-order validation, using
    # FULL_SHARD and SHARD_GRAD_OP; each test enables the detailed distributed debug level.
    @skip_if_lt_x_gpu(2)
    @parametrize(
        "sharding_strategy",
        [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP],
    )
    def test_invalid_first_iter_order(
        self,
        device,
        sharding_strategy: ShardingStrategy,
    ):
        # [FUNCTION DOC] Rank 0 keeps the normal path while all other ranks flip to the
        # alternate path before the first forward; asserts the first forward raises
        # RuntimeError whose message starts with "Forward order differs across ranks".
        # Side effect: sets the global distributed debug level.
        """Tests that FSDP errors if the all-gather order differs across ranks
        in the first iteration."""
        # Rank 0 runs the forward pass in one order and all other ranks run in
        # different order
        dist.set_debug_level(dist.DebugLevel.DETAIL)
        fsdp_model = Model.wrap(sharding_strategy, device_type)
        if self.rank != 0:
            fsdp_model.flip_path()
        inp = fsdp_model.module.get_input(device_type)
        # Match the error message with the following prefix
        error_regex = "^(Forward order differs across ranks)"
        with self.assertRaisesRegex(RuntimeError, error_regex):
            fsdp_model(*inp)

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "sharding_strategy",
        [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP],
    )
    @parametrize("iters_before_path_change", [1, 3])
    def test_invalid_later_iter_order(
        self,
        device,
        sharding_strategy: ShardingStrategy,
        iters_before_path_change: int,
    ):
        # [FUNCTION DOC] Trains for 1 or 3 iterations with the same path on all ranks, then
        # flips the path on every rank except rank 0. Non-zero ranks must emit a UserWarning
        # (forward order differs from the first iteration) on the next forward; rank 0 expects
        # none. One further iteration then runs without expecting issues.
        """Tests that FSDP warns the user if the all-gather order changes after
        the first iteration."""
        dist.set_debug_level(dist.DebugLevel.DETAIL)
        # On the first iteration, all ranks run the same order, and on the next
        # iteration, all but rank 0 run in a different order
        fsdp_model = Model.wrap(sharding_strategy, device_type)
        for _ in range(iters_before_path_change):
            inp = fsdp_model.module.get_input(device_type)
            output = fsdp_model(*inp)
            loss = fsdp_model.module.get_loss(inp, output).to(device_type)
            fsdp_model.module.run_backward(loss)
        # Match the warning message with the following prefix
        regex = (
            "^(Forward order differs from that of the first iteration "
            f"on rank {self.rank}. Collectives are unchecked and may give "
            "incorrect results or hang)"
        )
        context = (
            self.assertWarnsRegex(
                expected_warning=UserWarning,
                expected_regex=regex,
            )
            if self.rank != 0
            else nullcontext()
        )
        if self.rank != 0:
            fsdp_model.flip_path()
        inp = fsdp_model.module.get_input(device_type)
        # Expect a warning for the forward pass all-gather
        with context:  # warning for forward pass all-gather
            output = fsdp_model(*inp)
        loss = fsdp_model.module.get_loss(inp, output).to(device_type)
        fsdp_model.module.run_backward(loss)
        # Run an additional iteration to check that there are no more warnings
        inp = fsdp_model.module.get_input(device_type)
        output = fsdp_model(*inp)
        loss = fsdp_model.module.get_loss(inp, output).to(device_type)
        fsdp_model.module.run_backward(loss)

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "sharding_strategy",
        [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP],
    )
    def test_train_eval(self, device, sharding_strategy: ShardingStrategy):
        # [FUNCTION DOC] Alternates train() and eval() for two epochs of three iterations each
        # (backward only in train mode), recording all warnings, and raises AssertionError if
        # any recorded warning starts with "Forward order differs", i.e. order validation must
        # not trigger when switching modes or skipping backward in eval mode.
        dist.set_debug_level(dist.DebugLevel.DETAIL)
        fsdp_model = Model.wrap(sharding_strategy, device_type)
        NUM_ITERS = 3
        NUM_EPOCHS = 2
        with warnings.catch_warnings(record=True) as w:  # records warnings to `w`
            for _ in range(NUM_EPOCHS):
                fsdp_model.train()
                for _ in range(NUM_ITERS):
                    inp = fsdp_model.module.get_input(device_type)
                    output = fsdp_model(*inp)
                    loss = fsdp_model.module.get_loss(inp, output).to(device_type)
                    fsdp_model.module.run_backward(loss)
                fsdp_model.eval()
                for _ in range(NUM_ITERS):
                    inp = fsdp_model.module.get_input(device_type)
                    output = fsdp_model(*inp)
                    fsdp_model.module.get_loss(inp, output).to(device_type)
        # Check that the order validation warning was not issued (errors do not
        # need to be checked since they will be directly reported)
        warning_prefix = "Forward order differs"
        for warning in w:
            if str(warning.message).startswith(warning_prefix):
                raise AssertionError(
                    f"Warning was incorrectly issued: {warning.message}"
                )
        # If we still validate the forward execution order in eval mode, then
        # an `AssertionError` will be raised above for both sharding strategies


devices = ("cuda", "hpu", "xpu")
instantiate_device_type_tests(
    TestFSDPExecOrder, globals(), only_for=devices, allow_xpu=True
)
if __name__ == "__main__":
    run_tests()
