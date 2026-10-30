# Owner(s): ["oncall: jit"]

# [MODULE DOC] JIT tests for replicating scripted, traced and eager-wrapped modules across GPUs
# with torch.nn.parallel replicate. All tests need at least two CUDA devices and are skipped
# otherwise. Importing adds the test directory to sys.path; running the file directly is refused
# in favor of test/test_jit.py.
import os
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.parallel as dp


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase, RUN_CUDA_MULTI_GPU


class TestDataParallel(JitTestCase):
    # [CLASS DOC] Test case holding small module fixtures (eager wrappers whose forward is
    # ignored by the compiler, and script modules) plus tests that replicate them onto GPUs 0
    # and 1 and check device placement, outputs and tensor sharing.
    class Mpy(torch.nn.Module):
        # [CLASS DOC] HELPER: Eager module wrapping a sequential stack of linear, batch norm,
        # ReLU and linear layers (2 features); forward is marked ignored by the JIT.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Builds the inner sequential stack and stores it as
            # attribute m.
            super(TestDataParallel.Mpy, self).__init__()
            self.m = nn.Sequential(
                nn.Linear(2, 2), nn.BatchNorm1d(2), nn.ReLU(), nn.Linear(2, 2)
            )

        @torch.jit.ignore
        def forward(self, input):
            # [FUNCTION DOC] HELPER: Runs the input through the inner sequential stack; ignored
            # by the compiler so it stays Python.
            return self.m(input)

    class Mpy1(torch.nn.Module):
        # [CLASS DOC] HELPER: Eager module that wraps one given block; forward is marked ignored
        # by the JIT.
        def __init__(self, block):
            # [FUNCTION DOC] HELPER: Stores the given block as attribute m.
            super(TestDataParallel.Mpy1, self).__init__()
            self.m = block

        @torch.jit.ignore
        def forward(self, input):
            # [FUNCTION DOC] HELPER: Calls the wrapped block forward method on the input and
            # returns the result.
            return self.m.forward(input)

    class Mpy2(torch.nn.Module):
        # [CLASS DOC] HELPER: Eager module chaining two given blocks, with forward ignored by
        # the JIT.
        def __init__(self, block1, block2):
            # [FUNCTION DOC] HELPER: Stores the two given blocks as attributes m1 and m2 (the
            # same object may be passed twice to share it).
            super(TestDataParallel.Mpy2, self).__init__()
            self.m1 = block1
            self.m2 = block2

        @torch.jit.ignore
        def forward(self, input):
            # [FUNCTION DOC] HELPER: Applies the first block forward method to the input, then
            # calls the second block on that result and returns it.
            x = self.m1.forward(input)
            return self.m2(x)

    class Msm(torch.jit.ScriptModule):
        # [CLASS DOC] HELPER: ScriptModule holding a sequential stack of linear, batch norm,
        # ReLU and linear layers (2 features), registered as a constant attribute; forward is a
        # script method.
        __constants__ = ["m"]

        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Builds the sequential stack and stores it as attribute m.
            super(TestDataParallel.Msm, self).__init__()
            self.m = nn.Sequential(
                nn.Linear(2, 2), nn.BatchNorm1d(2), nn.ReLU(), nn.Linear(2, 2)
            )

        @torch.jit.script_method
        def forward(self, input):
            # [FUNCTION DOC] HELPER: Script method that runs the input through the
            # sequential stack.
            return self.m(input)

    class Msm1(torch.jit.ScriptModule):
        # [CLASS DOC] HELPER: ScriptModule that wraps one given submodule and applies it in a
        # script method forward.
        def __init__(self, block):
            # [FUNCTION DOC] HELPER: Stores the given block as attribute block.
            super(TestDataParallel.Msm1, self).__init__()
            self.block = block

        @torch.jit.script_method
        def forward(self, input):
            # [FUNCTION DOC] HELPER: Script method that calls the wrapped block on the input and
            # returns the result.
            x = self.block(input)
            return x

    def check_replicas(self, module, replicas, input_shape=(2, 2)):
        # [FUNCTION DOC] HELPER: Verifies replicas against the original module. Evaluates the
        # module on a random CUDA input of the given shape (default 2 by 2), then for each
        # replica at index i asserts every parameter and buffer lives on device i and that
        # running the replica on the input moved to device i gives the same values as the
        # original output. Raises assertion failure on mismatch.
        input = torch.randn(input_shape).cuda()
        expected_output = module(input).data
        for i, replica in enumerate(replicas):
            for p in replica.parameters():
                self.assertEqual(p.get_device(), i)
            for b in replica.buffers():
                self.assertEqual(b.get_device(), i)
            replica_input = input.cuda(i)
            self.assertEqual(replica(replica_input).data, expected_output)

    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "multi-GPU not supported")
    def test_python_submodule_script(self):
        # [FUNCTION DOC] Replicates an eager wrapper around a script module onto devices 0 and 1
        # and checks the replicas with check_replicas. Skipped without multiple GPUs.
        module = self.Mpy1(self.Msm()).cuda()
        replicas = dp.replicate(module, {0, 1})
        self.check_replicas(module, replicas)

    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "multi-GPU not supported")
    def test_shared_module(self):
        # [FUNCTION DOC] Builds a script module shared by two eager wrappers inside a chain,
        # replicates the chain onto devices 0 and 1, and checks the replicas with
        # check_replicas. Skipped without multiple GPUs.
        s = self.Msm()
        p1 = self.Mpy1(s)
        module = self.Mpy2(p1, s).cuda()
        replicas = dp.replicate(module, {0, 1})
        self.check_replicas(module, replicas)

    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "multi-GPU not supported")
    def test_traced_module(self):
        # [FUNCTION DOC] Traces an eager wrapper around the eager fixture module using a ones
        # example input, moves it to CUDA, replicates it onto devices 0 and 1, and checks the
        # replicas with check_replicas. Skipped without multiple GPUs.
        module = torch.jit.trace(self.Mpy1(self.Mpy()), torch.ones(2, 2)).cuda()
        replicas = dp.replicate(module, {0, 1})
        self.check_replicas(module, replicas)

    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "multi-GPU not supported")
    def test_tensor_sharing(self):
        # [FUNCTION DOC] Replicates a script module wrapper to devices 0 and 1, then checks
        # memory sharing: parameters and buffers of the replica on device 0 share storage with
        # the original, while those of the replica on device 1 do not. Skipped without
        # multiple GPUs.
        module = self.Msm1(self.Msm()).cuda()
        replica = dp.replicate(module, {0, 1})

        def assert_share_data(t1, t2):
            # [FUNCTION DOC] HELPER: Returns true if both tensors are on the same device and
            # their storages have the same data pointer (identity of memory only, not values).
            # Only checks that they point to the same memory on the same device.
            return (
                t1.device == t2.device
                and t1.storage().data_ptr() == t2.storage().data_ptr()
            )

        for p1, p2 in zip(module.parameters(), replica[0].parameters()):
            self.assertTrue(assert_share_data(p1, p2))

        for p1, p2 in zip(module.buffers(), replica[0].buffers()):
            self.assertTrue(assert_share_data(p1, p2))

        for p1, p2 in zip(module.parameters(), replica[1].parameters()):
            self.assertFalse(assert_share_data(p1, p2))

        for p1, p2 in zip(module.buffers(), replica[1].buffers()):
            self.assertFalse(assert_share_data(p1, p2))

    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "multi-GPU not supported")
    def test_tensor_sharing_with_forward(self):
        # [FUNCTION DOC] Checks replica semantics across an in-place parameter update. Runs
        # forward and backward on the original, subtracts the gradient from each parameter
        # through the raw data view under no-grad, runs forward again, and asserts the same-GPU
        # replica (a shallow copy sharing params) now gives the updated output, while the
        # other-GPU replica (a deep copy) still gives the pre-update output. Skipped without
        # multiple GPUs.
        module = self.Msm1(self.Msm()).cuda()
        replica = dp.replicate(module, {0, 1})
        x = torch.ones(2, 2, requires_grad=True).cuda()
        first_forward = module(x)
        first_forward.sum().backward()
        with torch.no_grad():
            for p in module.parameters():
                # Use .data here to avoid version counter bump.
                # The graph created by the following forward will be wrong but
                # we never backward through them so it's fine
                p.data -= 1.0 * p.grad
        second_forward = module(x)

        # replica which is on the same GPU has a shallow copy of the original
        # params and buffers
        r0_forward = replica[0](x)
        self.assertEqual(second_forward, r0_forward)

        # replica which is on a different GPU has a deep copy of the original
        # params and buffers
        x1 = torch.ones(2, 2, requires_grad=True).cuda(device=1)
        r1_forward = replica[1](x1)
        self.assertEqual(first_forward, r1_forward)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
