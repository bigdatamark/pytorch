# [MODULE DOC] Tests for FullyShardedDataParallel.apply: that applying an initializer function to
# an FSDP-wrapped model changes parameter values in place across nested wrapping, and that calling
# it inside summon_full_params raises. Import-time side effects: exits the interpreter with status
# 0 when distributed is unavailable or under dev-debug ASAN; computes device_type; instantiates
# device-specific test classes for cuda, hpu and xpu.
# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    DEVICEInitMode,
    FSDPInitMode,
    FSDPTestContinuous,
    get_devtype,
    NestedWrappedModule,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

device_type = torch.device(get_devtype())


class TestApply(FSDPTestContinuous):
    # [CLASS DOC] Multi-process tests (FSDPTestContinuous) of FSDP.apply. CUDA memory leak checking
    # is disabled because FSDP v1 reference cycles delay freeing of model tensors.
    # FSDP v1 has reference cycles that prevent GC from freeing model tensors,
    # causing false positives in the CUDA memory leak checker.
    _do_cuda_memory_leak_check = False

    @property
    def world_size(self):
        # [FUNCTION DOC] Property giving the rank count: 2, unless an accelerator is present with
        # fewer than 2 devices, in which case that device count.
        if torch.accelerator.is_available():
            gpu_cnt = torch.accelerator.device_count()
            if gpu_cnt < 2:
                return gpu_cnt
        return 2

    @torch.no_grad()
    def _init_linear_weights(self, m):
        # [FUNCTION DOC] HELPER: Function for module apply that, under no-grad, fills the
        # weight and bias of linear layers (exact type only) with 1.0 in place; other modules
        # are left alone.
        if type(m) is nn.Linear:
            m.weight.fill_(1.0)
            m.bias.fill_(1.0)

    def check_weights(self, fsdp, expected_tensor_fn, check):
        # [FUNCTION DOC] HELPER: Summons full parameters recursively, and for every parameter of
        # every exact linear module compares it with the tensor from the expected function using
        # the supplied check callable, with a message showing both values.
        with FSDP.summon_full_params(fsdp, recurse=True):
            linear_modules = [
                module for module in fsdp.modules() if type(module) is nn.Linear
            ]
            for module in linear_modules:
                for param in module.parameters():
                    expected = expected_tensor_fn(param)
                    check(param, expected, f"Got {param} but expected {expected}")

    def _check_apply(self, fsdp):
        # [FUNCTION DOC] HELPER: Checks linear parameters are not all 1.0 beforehand, calls apply
        # with the initializer on the FSDP model, then checks all linear parameters equal 1.0.
        # Assert linear weights are not all 1.0
        self.check_weights(
            fsdp, lambda param: torch.empty_like(param).fill_(1.0), self.assertNotEqual
        )

        fsdp.apply(self._init_linear_weights)

        # Ensure all weights are 1.0
        self.check_weights(
            fsdp, lambda param: torch.empty_like(param).fill_(1.0), self.assertEqual
        )

    @skip_if_lt_x_gpu(2)
    def test_nested_module_apply(self):
        # [FUNCTION DOC] Builds a nested wrapped module on the test device with recursive FSDP
        # wrapping and checks apply modifies linear parameters. Requires at least 2 GPUs.
        """Tests that ``apply()`` modifies parameter values in-place on a
        non-FSDP-root nested FSDP-wrapped model."""
        fsdp_kwargs = {"device_id": device_type.type}
        nested_wrapped_module = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_AFTER,
            fsdp_kwargs=fsdp_kwargs,
        )
        self._check_apply(nested_wrapped_module)

    @skip_if_lt_x_gpu(2)
    def test_transformer_module_apply(self):
        # [FUNCTION DOC] Same apply check on a recursively wrapped transformer with shared
        # parameters. Requires at least 2 GPUs.
        """Tests that ``apply()`` modifies parameter values in-place on an
        FSDP-wrapped transformer model with shared parameters."""
        fsdp_kwargs = {"device_id": device_type.type}
        transformer = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_AFTER,
            fsdp_kwargs=fsdp_kwargs,
        )
        self._check_apply(transformer)

    @skip_if_lt_x_gpu(2)
    def test_apply_in_summon_raises_error(self):
        # [FUNCTION DOC] Checks that calling apply while inside summon_full_params raises
        # ValueError mentioning the expected states. Requires at least 2 GPUs.
        """Tests that calling ``apply()`` on an FSDP instance inside the
        ``summon_full_params()`` context raises an error."""
        fsdp_kwargs = {"device_id": device_type.type}
        transformer = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_AFTER,
            fsdp_kwargs=fsdp_kwargs,
        )
        with transformer.summon_full_params(transformer):
            with self.assertRaisesRegex(ValueError, "expected to be in states"):
                transformer.apply(self._init_linear_weights)


devices = ("cuda", "hpu", "xpu")
instantiate_device_type_tests(TestApply, globals(), only_for=devices, allow_xpu=True)
if __name__ == "__main__":
    run_tests()
