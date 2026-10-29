# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests that Python assertions on tensor values inside compiled code raise errors
# across eager, aot_eager and inductor backends when false and not when true, and that inductor
# lowers the assertion into a single fused GPU kernel containing a Triton device assert. Defines
# the module global device_type as the current accelerator type, or cpu, at import time.

import torch
import torch._inductor.config
from torch._inductor import metrics
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.triton_utils import requires_gpu_and_triton


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


@instantiate_parametrized_tests
class TestTorchDeviceAssertTrigger(TestCase):
    # [CLASS DOC] Parametrized tests of assertion behavior under compilation with
    # different backends.
    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_assert_should_throw(self, backend):
        # [FUNCTION DOC] Parametrized by backend. Compiles two CPU functions whose assertion on
        # an all-positive reduction of a tensor with a negative entry is false (one stores the
        # reduction first, one asserts inline). Asserts calling each compiled function raises a
        # runtime error whose message contains the assertion message. Dynamo is reset before
        # each compile.
        def func():
            # [FUNCTION DOC] HELPER: Builds a CPU tensor with a negative value and asserts all
            # entries are positive, via an intermediate result; raises the assertion with the
            # message should throw.
            a = torch.tensor([1.0, -2.0], device="cpu")
            result = torch.all(a > 0)
            assert result, "should throw"  # noqa: S101

        def func_inline():
            # [FUNCTION DOC] HELPER: Same as above but the assertion is made directly on
            # the reduction.
            a = torch.tensor([1.0, -2.0], device="cpu")
            assert torch.all(a > 0), "should throw"  # noqa: S101

        with self.assertRaisesRegex(RuntimeError, "should throw"):
            torch._dynamo.reset()
            f_c = torch.compile(func, backend=backend)
            f_c()

        with self.assertRaisesRegex(RuntimeError, "should throw"):
            torch._dynamo.reset()
            f_c = torch.compile(func_inline, backend=backend)
            f_c()

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_assert_should_not_throw(self, backend):
        # [FUNCTION DOC] Parametrized by backend. Compiles and runs two CPU functions whose
        # positivity assertion holds (one stores the reduction first, one asserts inline);
        # passes if neither raises.
        def func():
            # [FUNCTION DOC] HELPER: Builds an all-positive CPU tensor and asserts positivity
            # through an intermediate result; never fails.
            a = torch.tensor([1.0, 2.0], device="cpu")
            result = torch.all(a > 0)
            assert result, "should throw"  # noqa: S101

        def func_inline():
            # [FUNCTION DOC] HELPER: Builds an all-positive CPU tensor and asserts positivity
            # directly; never fails.
            a = torch.tensor([1.0, 2.0], device="cpu")
            assert torch.all(a > 0), "should throw"  # noqa: S101

        torch._dynamo.reset()
        f_c = torch.compile(func, backend=backend)
        f_c()

        torch._dynamo.reset()
        f_c = torch.compile(func_inline, backend=backend)
        f_c()

    @requires_gpu_and_triton
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_assert_fusion(self):
        # [FUNCTION DOC] Requires GPU and triton, with caches disabled and inductor metrics
        # logging on. Compiles a function that creates a tensor on the accelerator and asserts
        # it is all positive; resets metrics and asserts the generated kernel count is zero
        # before running and one after, then resets logging. Shows the tensor creation,
        # reduction and assertion fuse to one kernel.
        torch._logging.set_logs(inductor_metrics=True)

        def func():
            # [FUNCTION DOC] HELPER: Creates a positive tensor on the accelerator and asserts
            # all entries are positive.
            a = torch.tensor([1.0, 2.0], device=device_type)
            result = torch.all(a > 0)
            assert result, "should throw"  # noqa: S101

        torch._dynamo.reset()
        f_c = torch.compile(func, backend="inductor")
        metrics.reset()
        self.assertEqual(metrics.generated_kernel_count, 0)
        f_c()
        self.assertEqual(metrics.generated_kernel_count, 1)
        torch._logging.set_logs()

    @requires_gpu_and_triton
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_run_assert_triton(self):
        # [FUNCTION DOC] Requires GPU and triton, with caches disabled. Compiles with inductor
        # a function that asserts a positive accelerator tensor, checks it runs without
        # raising, then gets the generated code and asserts the triton device assert appears
        # exactly once.
        @torch.compile(backend="inductor")
        def fn():
            # [FUNCTION DOC] HELPER: Creates a positive tensor on the accelerator and asserts
            # all entries are positive; compiled with the inductor backend.
            a = torch.tensor([1.0, 2.0], device=device_type)
            result = torch.all(a > 0)
            assert result, "should throw"  # noqa: S101

        def should_not_throw(fn):
            # [FUNCTION DOC] HELPER: Calls the given function and returns True if it completes,
            # False if it raises any exception.
            try:
                fn()
                return True
            except Exception:
                return False

        self.assertEqual(should_not_throw(fn), True)

        _, code = run_and_get_code(fn)
        self.assertEqual(code[0].count("tl.device_assert"), 1)


if __name__ == "__main__":
    run_tests()
