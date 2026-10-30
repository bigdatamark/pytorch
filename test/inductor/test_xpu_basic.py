# Owner(s): ["module: inductor"]
# [MODULE DOC] Temporary smoke tests for the first Intel GPU inductor integration: elementwise
# add, sub, mul and div on random float tensors, run through the shared inductor model checker on
# the xpu device. Import-time side effects: requires the filelock package (import fails otherwise)
# and appends the test directory to the module search path. The main block runs the tests only
# when XPU and Triton are available. The file carries a TODO to remove it.
import importlib
import os
import sys

import torch


importlib.import_module("filelock")

pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model_gpu,
    TestCase,
)


# TODO: Remove this file.
# This is a temporary test case to test the base functionality of first Intel GPU Inductor integration.
# We are working on reuse and pass the test cases in test/inductor/*  step by step.
# Will remove this file when pass full test in test/inductor/*.


class XpuBasicTests(TestCase):
    # [CLASS DOC] Test case using the shared GPU model-check helper as its common runner with the
    # device set to xpu. Each test defines a small binary function and compares compiled and
    # eager.
    common = check_model_gpu
    device = "xpu"

    def test_add(self):
        # [FUNCTION DOC] Checks addition of two random 4-D float tensors via the common runner.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns the sum of the two arguments.
            return a + b

        self.common(fn, (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)))

    def test_sub(self):
        # [FUNCTION DOC] Checks subtraction of two random 4-D float tensors via the common runner.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns the difference of the two arguments.
            return a - b

        self.common(fn, (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)))

    def test_mul(self):
        # [FUNCTION DOC] Checks multiplication of two random 4-D float tensors via the common
        # runner.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns the product of the two arguments.
            return a * b

        self.common(fn, (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)))

    def test_div(self):
        # [FUNCTION DOC] Checks division of two random 4-D float tensors via the common runner.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns the quotient of the two arguments.
            return a / b

        self.common(fn, (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests
    from torch.testing._internal.inductor_utils import HAS_XPU_AND_TRITON

    if HAS_XPU_AND_TRITON:
        run_tests(needs="filelock")
