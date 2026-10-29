# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Multi-GPU test that FSDP trains a model with a frozen "trunk" and a trainable
# "head" to the same final parameters as DDP. The freezing is done either by clearing the
# trunk's requires_grad flags or by setting trunk gradients to None each step; the trunk may
# be wrapped in FSDP before or after freezing, may run with autograd disabled, and may be one
# block or a nested sequence of blocks. Defines two helper models and a freezing-method enum.
# Import-time side effects: exits with status 0 if distributed is unavailable or under
# dev-asan; computes the accelerator device type (cpu fallback); parametrized tests are
# instantiated at module level. Running the file calls run_tests().

import contextlib
import sys
from enum import Enum

import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_full_params
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
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

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class Model(nn.Module):
    # [CLASS DOC] HELPER: Image classifier with a "trunk" (3x3 conv to 64 channels, ReLU, global
    # average pool, flatten) and a "head" (linear 64 to 10). Records in autograd_ctx whether the
    # trunk should run under no_grad. Can wrap trunk and head in separate FSDP instances.
    def __init__(
        self,
        with_fsdp,
        freeze_after_wrap_fsdp,
        disable_autograd,
        fsdp_kwargs,
    ):
        # [FUNCTION DOC] HELPER: Builds trunk and head; if with_fsdp and freeze_after_wrap_fsdp
        # are both true, wraps them in FSDP immediately using fsdp_kwargs. Stores a context
        # factory (no_grad when disable_autograd, else a null context) used in forward.
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Linear(64, 10)
        if with_fsdp and freeze_after_wrap_fsdp:
            self.fsdp_wrap(fsdp_kwargs)
        self.autograd_ctx = (
            torch.no_grad if disable_autograd else contextlib.nullcontext
        )

    def fsdp_wrap(self, fsdp_kwargs):
        # [FUNCTION DOC] HELPER: Replaces trunk and head with FSDP-wrapped versions built with
        # fsdp_kwargs. Mutates the module.
        self.trunk = FSDP(self.trunk, **fsdp_kwargs)
        self.head = FSDP(self.head, **fsdp_kwargs)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs the trunk inside the stored autograd context (possibly
        # no_grad), then the head outside it, and returns the head output.
        with self.autograd_ctx():
            x = self.trunk(x)
        return self.head(x)


class NestedTrunkModel(nn.Module):
    # [CLASS DOC] HELPER: Same role as Model but the trunk is a Sequential of two conv+ReLU
    # blocks (3 to 64 and 64 to 64 channels) and the head is pool, flatten and a linear layer.
    # When wrapped, each trunk block gets its own FSDP instance inside a trunk FSDP.
    def __init__(
        self,
        with_fsdp,
        freeze_after_wrap_fsdp,
        disable_autograd,
        fsdp_kwargs,
    ):
        # [FUNCTION DOC] HELPER: Builds the two-block trunk and the head; wraps with FSDP right
        # away only if with_fsdp and freeze_after_wrap_fsdp are true; stores the autograd
        # context factory (no_grad when disable_autograd) used by forward.
        super().__init__()
        self.trunk = nn.Sequential(
            self._create_block(3, 64, with_fsdp, freeze_after_wrap_fsdp),
            self._create_block(64, 64, with_fsdp, freeze_after_wrap_fsdp),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10),
        )
        if with_fsdp and freeze_after_wrap_fsdp:
            self.fsdp_wrap(fsdp_kwargs)
        self.autograd_ctx = (
            torch.no_grad if disable_autograd else contextlib.nullcontext
        )

    def fsdp_wrap(self, fsdp_kwargs):
        # [FUNCTION DOC] HELPER: Wraps each trunk block in its own FSDP (replacing it in the
        # Sequential), then wraps the trunk and the head in FSDP, all with fsdp_kwargs.
        for name, child in self.trunk.named_children():
            wrapped_child = FSDP(child, **fsdp_kwargs)
            setattr(self.trunk, name, wrapped_child)
        self.trunk = FSDP(self.trunk, **fsdp_kwargs)
        self.head = FSDP(self.head, **fsdp_kwargs)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs the trunk inside the stored autograd context (possibly
        # no_grad), then the head, and returns the head output.
        with self.autograd_ctx():
            x = self.trunk(x)
        return self.head(x)

    def _create_block(
        self, in_channels, out_channels, with_fsdp, freeze_after_wrap_fsdp
    ):
        # [FUNCTION DOC] HELPER: Returns a Sequential of a 3x3 convolution and an in-place
        # ReLU. The with_fsdp and freeze_after_wrap_fsdp arguments are accepted but unused.
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        return block


class FreezingMethod(str, Enum):
    # [CLASS DOC] HELPER: String enum of how the trunk is frozen: GradToNone resets trunk
    # gradients to None after each backward; RequiresGrad turns off requires_grad on trunk
    # parameters before training.
    GradToNone = "grad_to_none"
    RequiresGrad = "requires_grad"


class TestFreezingWeights(FSDPTest):
    # [CLASS DOC] Multi-process test case comparing FSDP and DDP training results for models
    # with a frozen trunk; _dist_train performs one training run and returns final parameters.
    def _create_model(
        self,
        with_fsdp,
        with_nested_trunk,
        freeze_after_wrap_fsdp,
        disable_autograd,
        fsdp_kwargs,
    ):
        # [FUNCTION DOC] HELPER: Factory returning a NestedTrunkModel if with_nested_trunk is
        # true, else a Model, constructed with the given flags and FSDP keyword arguments.
        if with_nested_trunk:
            model = NestedTrunkModel(
                with_fsdp, freeze_after_wrap_fsdp, disable_autograd, fsdp_kwargs
            )
        else:
            model = Model(
                with_fsdp, freeze_after_wrap_fsdp, disable_autograd, fsdp_kwargs
            )
        return model

    def _dist_train(
        self,
        with_nested_trunk,
        freezing_method,
        freeze_after_wrap_fsdp,
        with_fsdp,
        disable_autograd,
        forward_prefetch,
    ):
        # [FUNCTION DOC] HELPER: Trains the chosen model for three SGD steps (lr 0.1, momentum
        # 0.9, cross-entropy on a fixed seeded batch of two images) under DDP or FSDP, after
        # optionally freezing the trunk via requires_grad; with the grad-to-none method it
        # clears trunk gradients each step. FSDP wrapping happens before or after construction
        # per freeze_after_wrap_fsdp and forward_prefetch is passed to FSDP. Returns the full
        # (gathered) parameters for FSDP or the parameter list for DDP.
        torch.manual_seed(0)
        batch = torch.randn(size=(2, 3, 224, 224)).to(device_type)

        fsdp_kwargs = {
            "device_id": self.rank,
            "forward_prefetch": forward_prefetch,
        }

        ddp_kwargs = {
            "device_ids": [self.rank],
            "find_unused_parameters": bool(disable_autograd),
        }

        model = self._create_model(
            with_fsdp,
            with_nested_trunk,
            freeze_after_wrap_fsdp,
            disable_autograd,
            fsdp_kwargs,
        )
        model = model.to(device_type)

        # freezing the trunk using requires_grad.
        if freezing_method == FreezingMethod.RequiresGrad:
            for param in model.trunk.parameters():
                param.requires_grad = False

        if with_fsdp:
            if not freeze_after_wrap_fsdp:
                model.fsdp_wrap(fsdp_kwargs)
            model = FSDP(model, **fsdp_kwargs)
        else:
            model = DistributedDataParallel(model, **ddp_kwargs)

        target = torch.tensor([0, 1], dtype=torch.long).to(device_type)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        for _ in range(3):
            out = model(batch)
            fake_loss = criterion(out, target)
            optimizer.zero_grad()
            fake_loss.backward()
            if freezing_method == FreezingMethod.GradToNone:
                for param in model.module.trunk.parameters():
                    param.grad = None
            optimizer.step()

        if with_fsdp:
            return get_full_params(model)

        return list(model.parameters())

    @skip_if_lt_x_gpu(2)
    @parametrize("with_nested_trunk", [True, False])
    @parametrize(
        "freezing_method", [FreezingMethod.RequiresGrad, FreezingMethod.GradToNone]
    )
    @parametrize("freeze_after_wrap_fsdp", [True, False])
    @parametrize("disable_autograd", [True, False])
    @parametrize("forward_prefetch", [True, False])
    def test_freezing_weights(
        self,
        with_nested_trunk,
        freezing_method,
        freeze_after_wrap_fsdp,
        disable_autograd,
        forward_prefetch,
    ):
        # [FUNCTION DOC] Parametrized over nested trunk, freezing method, wrap-before-freeze,
        # disabled autograd and forward prefetch. Trains with DDP and with FSDP, asserts the
        # final parameters are equal (including device), and for the requires_grad freezing
        # method also that each parameter's requires_grad flag matches between the two.
        # DDP
        ddp_state = self._dist_train(
            with_nested_trunk,
            freezing_method,
            freeze_after_wrap_fsdp,
            with_fsdp=False,
            disable_autograd=disable_autograd,
            forward_prefetch=False,  # does not apply to DDP
        )

        # FSDP
        fsdp_state = self._dist_train(
            with_nested_trunk,
            freezing_method,
            freeze_after_wrap_fsdp,
            with_fsdp=True,
            disable_autograd=disable_autograd,
            forward_prefetch=forward_prefetch,
        )

        self.assertEqual(
            ddp_state,
            fsdp_state,
            exact_device=True,
            msg="FullyShardedDataParallel states didn't match PyTorch DDP states",
        )

        if freezing_method == FreezingMethod.RequiresGrad:
            for ddp_param, fsdp_param in zip(ddp_state, fsdp_state):
                self.assertEqual(ddp_param.requires_grad, fsdp_param.requires_grad)


instantiate_parametrized_tests(TestFreezingWeights)

if __name__ == "__main__":
    run_tests()
