# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Tests for the composable replicate() API
# (torch.distributed._composable.replicate), which wraps a module with DDP-style gradient
# synchronization. Contains a thread-based test of state_dict parity with the local module, a
# multi-process gloo test class comparing replicated training against local large-batch training
# (and testing device_id, ignored_modules and argument moving), and a subclass checking
# interplay with fully_shard. Import-time side effects: computes the module-level device_type
# from the current accelerator (cpu if none) and its device_module. Most multi-process tests are
# skipped on Linux through issue-linked skip decorators.

import unittest
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed._composable.replicate import replicate
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    MultiThreadedTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    IS_LINUX,
    run_tests,
    TEST_WITH_ROCM,
    TEST_XPU,
)


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
device_module = torch.get_device_module(device_type)


class Net(nn.Module):
    # [CLASS DOC] HELPER: Tiny model of three chained 2x2 Linear layers fc1, fc2, fc3.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the three 2x2 Linear layers.
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, 2)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies fc1, fc2, fc3 in order and returns the result.
        return self.fc3(self.fc2(self.fc1(x)))


class ReplicateStateDictTest(MultiThreadedTestCase):
    # [CLASS DOC] Multi-threaded (4 ranks, set by world_size) tests that replicate() leaves
    # state_dict keys and values identical to those of an unwrapped copy of the model.
    @property
    def world_size(self):
        # [FUNCTION DOC] Returns the fixed world size, 4, for the threaded test case.
        return 4

    def setUp(self):
        # [FUNCTION DOC] Runs the base setUp and then spawns the per-rank threads for the test.
        super().setUp()
        self._spawn_threads()

    def _check_state_dict_parity(self, sd_1, sd_2):
        # [FUNCTION DOC] HELPER: Asserts two state dicts have the same keys in the same order
        # and equal values, comparing by position. Zip silently truncates, so differing lengths
        # are not detected.
        for k1, k2 in zip(sd_1.keys(), sd_2.keys()):
            self.assertEqual(k1, k2)

        for v1, v2 in zip(sd_1.values(), sd_2.values()):
            self.assertEqual(v1, v2)

    def test_replicate_single_module_save_load(self):
        # [FUNCTION DOC] Applies replicate to a whole deep copy of Net and checks its state_dict
        # matches that of the original.
        """
        Tests that replicate() on a single module state_dict
        matches local module state_dict.
        """
        model = Net()
        replicate_model = replicate(deepcopy(model))
        local_sd = model.state_dict()
        ddp_sd = replicate_model.state_dict()
        self._check_state_dict_parity(local_sd, ddp_sd)

    def test_replicate_non_root_multiple_save_load(self):
        # [FUNCTION DOC] Applies replicate separately to fc1, fc2 and fc3 of a deep copy of Net
        # and checks its state_dict matches the original model's.
        """
        Tests the replicate() on multiple submodules matches
        local module state_dict.
        """
        model = Net()
        replicate_model = deepcopy(model)
        replicate(replicate_model.fc1)
        replicate(replicate_model.fc2)
        replicate(replicate_model.fc3)

        local_sd = model.state_dict()
        ddp_sd = replicate_model.state_dict()
        self._check_state_dict_parity(local_sd, ddp_sd)


class ReplicateTest(MultiProcContinuousTest):
    # [CLASS DOC] Multi-process (world size 2, gloo backend, cpu device type) continuous test class
    # for replicate(). Checks training parity with local training, device_id handling, moving args
    # to the device, ignored modules and constructor kwargs. Most tests are skipped on Linux.
    world_size = 2

    @classmethod
    def backend_str(cls) -> str:
        # [FUNCTION DOC] Returns the process-group backend name, "gloo".
        return "gloo"

    @classmethod
    def device_type(cls) -> str:
        # [FUNCTION DOC] Returns the device type for the continuous test, "cpu".
        return "cpu"

    def _compare_module(self, mod, replicate_mod):
        # [FUNCTION DOC] HELPER: Trains a local model on the global batch and a replicated model
        # on each rank's slice (one sample per rank) for two iterations with plain SGD (lr 1)
        # steps, asserting the parameter counts and values match within tolerance each
        # iteration. The global input is reshuffled with a seeded permutation after each
        # iteration. Side effects: zeroes grads.
        local_batch_size = 1
        global_batch_size = self.world_size * local_batch_size
        input = torch.randn(global_batch_size, 2)
        target = torch.randn(global_batch_size, 2)

        def step_model(model, input, target):
            # [FUNCTION DOC] HELPER: One training step: sets train mode, computes mse loss
            # against the target (moved to the output device), backpropagates, subtracts each
            # grad from its parameter under no_grad, and clears the grads.
            model.train()
            output = model(input)
            loss = F.mse_loss(output, target.to(output.device))
            loss.backward()
            for param in model.parameters():
                with torch.no_grad():
                    param -= param.grad
                param.grad = None

        for iteration in range(2):
            step_model(mod, input, target)
            step_model(
                replicate_mod,
                input[
                    self.rank * local_batch_size : (self.rank + 1) * local_batch_size
                ],
                target[
                    self.rank * local_batch_size : (self.rank + 1) * local_batch_size
                ],
            )

            self.assertEqual(
                len(list(mod.parameters())),
                len(list(replicate_mod.parameters())),
            )
            for i, j in zip(mod.parameters(), replicate_mod.parameters()):
                self.assertEqual(i, j, rtol=1.3e-06, atol=5e-5)

            # Shuffle the input so that DDP input is different
            torch.manual_seed(iteration)
            input = input[torch.randperm(global_batch_size)]

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/180205")
    def test_replicate_single_module(self):
        # [FUNCTION DOC] Checks that replicate on a single deep-copied module trains
        # equivalently to the local model.
        model = Net()
        replicate_model = replicate(deepcopy(model))
        self._compare_module(model, replicate_model)

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_ROCM, "https://github.com/pytorch/pytorch/issues/179948"
    )
    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(TEST_XPU, "XPU does not support gloo backend")
    def test_replicate_move_args_kwargs_to_device(self):
        # [FUNCTION DOC] Checks replicate with a device_id moves CPU positional and keyword
        # tensor inputs to the device so forward and backward run without error; needs 2
        # accelerators.
        class MyNet(nn.Module):
            # [CLASS DOC] HELPER: Module whose forward optionally multiplies input by a
            # keyword-only matrix before one Linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates one 2x2 Linear layer a.
                super().__init__()
                self.a = nn.Linear(2, 2)

            def forward(self, inp, *, kwarg=None):
                # [FUNCTION DOC] HELPER: If kwarg is given, matrix-multiplies the input by it;
                # then applies a.
                if kwarg is not None:
                    inp = inp @ kwarg
                return self.a(inp)

        torch.accelerator.set_device_index(self.rank)
        model = MyNet().to(device_type)
        replicate(model, device_id=torch.accelerator.current_device_index())
        # CPU input ensures replicate can move arg and kwargs to device.
        a, b = torch.randn(2, 2), torch.randn(2, 2)
        model(a, kwarg=b).sum().backward()

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/179854")
    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(TEST_XPU, "XPU does not support gloo backend")
    def test_replicate_ignore_module(self):
        # [FUNCTION DOC] Checks that with fc1 passed in ignored_modules, fc1 gradients differ
        # across ranks (not synchronized) while fc2 and fc3 gradients are all-gathered equal.
        # Needs 2 accelerators; seeds make inputs differ per rank.
        torch.accelerator.set_device_index(self.rank)
        # Seed ensures diff input and thus different local grads across ranks.
        torch.manual_seed(self.rank)
        device_module.manual_seed(self.rank)
        model = Net().to(device_type)
        replicate(model, ignored_modules=[model.fc1])
        # CPU input ensures that replicate can move input to GPU as DDP does.
        inp = torch.randn(5, 2, device=device_type) * (self.rank + 1)
        out = model(inp) * 10
        out.sum().backward()
        # FC1 grads should not be synchronized, FC2 and 3 should be.
        fc1_grad = model.fc1.weight.grad
        tensor_list = [torch.zeros_like(fc1_grad) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, fc1_grad)
        grad, rest = tensor_list[0], tensor_list[1:]
        for g in rest:
            self.assertNotEqual(grad, g)

        for dp_grad in [model.fc2.weight.grad, model.fc3.weight.grad]:
            tensor_list = [
                torch.zeros_like(dp_grad) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(tensor_list, dp_grad)
            grad, rest = tensor_list[0], tensor_list[1:]
            for g in rest:
                self.assertEqual(grad, g)

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_ROCM, "https://github.com/pytorch/pytorch/issues/180127"
    )
    def test_replicate_multi_module(self):
        # [FUNCTION DOC] Checks replicate applied separately to each of fc1, fc2, fc3 trains
        # equivalently to the local model.
        model = Net()
        replicate_model = deepcopy(model)
        replicate(replicate_model.fc1)
        replicate(replicate_model.fc2)
        replicate(replicate_model.fc3)
        self._compare_module(model, replicate_model)

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/180265")
    def test_replicate_with_kwargs(self):
        # [FUNCTION DOC] Checks replicate accepts DDP kwargs (bucket_cap_mb,
        # gradient_as_bucket_view) and still trains equivalently to the local model.
        model = Net()
        replicate_model = replicate(
            deepcopy(model), bucket_cap_mb=1, gradient_as_bucket_view=True
        )
        self._compare_module(model, replicate_model)

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/179746")
    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(TEST_XPU, "XPU does not support gloo backend")
    def test_replicate_device_id(self):
        # [FUNCTION DOC] Checks the device_ids of the underlying DDP object, created at first
        # forward: None for a cpu device_id, and [0] for a device given as torch.device or as an
        # int. Needs 2 accelerators.
        model = Net()
        model_cuda = deepcopy(model).to(device_type)
        model_cuda2 = deepcopy(model_cuda)
        replicate(model, device_id=torch.device("cpu"))
        # DDP instance is attached in first pre forward
        model(torch.randn(2, 2))
        replicate_ddp_weakref = replicate.state(model)._ddp_weakref()
        # Should be None for CPU training
        self.assertEqual(None, replicate_ddp_weakref.device_ids)

        replicate(
            model_cuda, device_id=torch.device(torch.accelerator.current_device_index())
        )
        # DDP instance is attached in first pre forward
        model_cuda(torch.randn(2, 2))
        replicate_ddp_weakref = replicate.state(model_cuda)._ddp_weakref()
        self.assertEqual([0], replicate_ddp_weakref.device_ids)
        # Pass in int as device_id
        replicate(model_cuda2, device_id=int(torch.accelerator.current_device_index()))
        # DDP instance is attached in first pre forward
        model_cuda2(torch.randn(2, 2))
        replicate_ddp_weakref = replicate.state(model_cuda2)._ddp_weakref()
        self.assertEqual([0], replicate_ddp_weakref.device_ids)

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/176155")
    def test_replicate_wrong_device_id_type(self):
        # [FUNCTION DOC] Checks replicate raises RuntimeError mentioning device_id must be int
        # or torch.device when given a list.
        model = Net()
        with self.assertRaisesRegex(
            RuntimeError, "Expected device_id to be int or torch.device"
        ):
            replicate(model, device_id=[torch.device("cpu")])


class ReplicateFullyShardInit(ReplicateTest):
    # [CLASS DOC] Subclass of ReplicateTest (so it also re-runs those tests) adding a test that
    # replicate can be applied on top of fully_shard.
    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/179810")
    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(TEST_XPU, "XPU does not support gloo backend")
    def test_replicate_fully_shard_init(self):
        # [FUNCTION DOC] Applies fully_shard to each linear and to the container, then replicate
        # to the whole ToyModel, asserting weights are DTensors before and after a forward that
        # triggers lazy init. Needs 2 accelerators.
        class ToyModel(nn.Module):
            # [CLASS DOC] HELPER: Model of a three-Linear Sequential followed by a projection,
            # no bias.
            def __init__(self, dim: int):
                # [FUNCTION DOC] HELPER: Builds bias-free dim x dim Linear layers: three in
                # linears plus proj.
                super().__init__()
                self.linears = nn.Sequential(
                    nn.Linear(dim, dim, bias=False),
                    nn.Linear(dim, dim, bias=False),
                    nn.Linear(dim, dim, bias=False),
                )
                self.proj = nn.Linear(dim, dim, bias=False)

            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Applies linears and then proj to x and returns the
                # result.
                y = self.linears(x)
                y = self.proj(y)
                return y

        torch.accelerator.set_device_index(self.rank)
        dim = 3
        bz = 2
        model = ToyModel(dim).to(device_type)
        for linear in model.linears:
            fully_shard(linear)
        fully_shard(model.linears)
        replicate(model, device_id=torch.accelerator.current_device_index())
        for linear in model.linears:
            self.assertTrue(isinstance(linear.weight, DTensor))
        inp = torch.rand(bz, dim)
        # trigger lazy init
        model(inp).sum()
        for linear in model.linears:
            self.assertTrue(isinstance(linear.weight, DTensor))


if __name__ == "__main__":
    run_tests()
