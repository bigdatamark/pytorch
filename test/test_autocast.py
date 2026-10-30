# Owner(s): ["module: unknown"]
# [MODULE DOC] Autocast tests: CPU op-list tests built on the shared TestAutocast base, a device-
# generic check that the autocast weight cache is global across forward and backward, MPS dtype
# checks, API validation tests (device strings, non-callable use), and no_grad / inference_mode
# interaction tests. Device-generic classes are instantiated at import time for all device types,
# MPS included for the first one.


import torch
from torch.testing._internal.autocast_test_lists import (
    AutocastCPUTestLists,
    TestAutocast,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyAccelerator,
    onlyMPS,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)
from torch.utils._python_dispatch import TorchDispatchMode


class TestAutocastCPU(TestAutocast):
    # [CLASS DOC] CPU autocast tests. Each test iterates an op list from AutocastCPUTestLists
    # (created in setUp as autocast_lists) and compares autocast output with manual casting.
    hw_classification = HardwareClassification.CPU

    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs base setUp and creates the CPU autocast op lists fixture.
        super().setUp()
        self.autocast_lists = AutocastCPUTestLists(torch.device("cpu"))

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Deletes the op lists fixture, then runs base tearDown.
        del self.autocast_lists
        super().tearDown()

    @skipIfTorchDynamo()
    def test_autocast_torch_expect_builtin_promote(self):
        # [FUNCTION DOC] For each promote-policy torch function entry, runs the out-of-place
        # autocast check twice: default bfloat16 amp with the first args and float16 amp with the
        # second args, both expecting float32 compute and the listed output type. Skipped under
        # Dynamo.
        for (
            op,
            args1,
            args2,
            out_type,
        ) in self.autocast_lists.torch_expect_builtin_promote:
            self._run_autocast_outofplace(
                op, args1, torch.float32, device="cpu", out_type=out_type
            )
            self._run_autocast_outofplace(
                op,
                args2,
                torch.float32,
                device="cpu",
                out_type=out_type,
                amp_dtype=torch.float16,
            )

    @skipIfTorchDynamo()
    def test_autocast_methods_expect_builtin_promote(self):
        # [FUNCTION DOC] Same as the torch-function promote test but for Tensor methods (module is
        # None), covering bfloat16 and float16 amp dtypes with the listed output type. Skipped
        # under Dynamo.
        for (
            op,
            args1,
            args2,
            out_type,
        ) in self.autocast_lists.methods_expect_builtin_promote:
            self._run_autocast_outofplace(
                op, args1, torch.float32, device="cpu", module=None, out_type=out_type
            )
            self._run_autocast_outofplace(
                op,
                args2,
                torch.float32,
                device="cpu",
                module=None,
                out_type=out_type,
                amp_dtype=torch.float16,
            )

    @skipIfTorchDynamo()
    def test_autocast_torch_16(self):
        # [FUNCTION DOC] For each op expected to run in lower precision, splits optional kwargs
        # and checks autocast casts to bfloat16 (default amp dtype) and to float16 (amp dtype
        # float16). Skipped under Dynamo.
        for op_with_args in self.autocast_lists.torch_16:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(
                op, args, torch.bfloat16, device="cpu", add_kwargs=maybe_kwargs
            )
            self._run_autocast_outofplace(
                op,
                args,
                torch.float16,
                device="cpu",
                add_kwargs=maybe_kwargs,
                amp_dtype=torch.float16,
            )

    @skipIfTorchDynamo()
    def test_autocast_nn_16(self):
        # [FUNCTION DOC] Same as the lower-precision torch test but ops are looked up in the
        # internal nn module namespace; checks bfloat16 and float16 amp dtypes. Skipped under
        # Dynamo.
        for op_with_args in self.autocast_lists.nn_16:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(
                op,
                args,
                torch.bfloat16,
                device="cpu",
                module=torch._C._nn,
                add_kwargs=maybe_kwargs,
            )
            self._run_autocast_outofplace(
                op,
                args,
                torch.float16,
                device="cpu",
                module=torch._C._nn,
                add_kwargs=maybe_kwargs,
                amp_dtype=torch.float16,
            )

    @skipIfTorchDynamo()
    def test_autocast_torch_fp32(self):
        # [FUNCTION DOC] For each op expected to run in float32, checks autocast with bfloat16 and
        # with float16 amp dtype both cast inputs to float32. Skipped under Dynamo.
        for op_with_args in self.autocast_lists.torch_fp32:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(
                op, args, torch.float32, device="cpu", add_kwargs=maybe_kwargs
            )
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                device="cpu",
                add_kwargs=maybe_kwargs,
                amp_dtype=torch.float16,
            )

    @skipIfTorchDynamo()
    def test_autocast_nn_fp32(self):
        # [FUNCTION DOC] Same as the float32 torch test but for ops in the internal nn namespace,
        # for both bfloat16 and float16 amp dtypes. Skipped under Dynamo.
        for op_with_args in self.autocast_lists.nn_fp32:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                device="cpu",
                module=torch._C._nn,
                add_kwargs=maybe_kwargs,
            )
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                device="cpu",
                module=torch._C._nn,
                add_kwargs=maybe_kwargs,
                amp_dtype=torch.float16,
            )

    @skipIfTorchDynamo()
    def test_autocast_torch_need_autocast_promote(self):
        # [FUNCTION DOC] For each op using the autocast promote policy, checks float32 results
        # with the first args under default amp and the second args under float16 amp. Skipped
        # under Dynamo.
        for op, args1, args2 in self.autocast_lists.torch_need_autocast_promote:
            self._run_autocast_outofplace(op, args1, torch.float32, device="cpu")
            self._run_autocast_outofplace(
                op, args2, torch.float32, device="cpu", amp_dtype=torch.float16
            )

    def test_autocast_rnn(self):
        # [FUNCTION DOC] Only when mkldnn and bfloat16 support are available: an LSTM converted to
        # bfloat16 must raise ValueError on float32 input without autocast (dtype mismatch
        # message), and must run without error inside CPU autocast. Otherwise does nothing.
        if (
            torch.backends.mkldnn.is_available()
            and torch.ops.mkldnn._is_mkldnn_bf16_supported()
        ):
            x = torch.randn(1, 2, 1)
            hx = torch.randn(2, 2, 1)
            cx = torch.randn(2, 2, 1)

            m = torch.nn.LSTM(1, 1, 2).to(torch.bfloat16)

            # Raise ValueError when autocast is not enabled
            with self.assertRaisesRegex(
                ValueError, r"RNN input dtype .* does not match weight dtype"
            ):
                m(x, (hx, cx))

            # Should be able to run the below case with autocast
            with torch.amp.autocast(device_type="cpu"):
                m(x, (hx, cx))

    def test_autocast_disabled_with_fp32_dtype(self):
        # [FUNCTION DOC] Smoke test that entering CPU autocast with float32 dtype and enabled
        # False does not fail when creating a tensor.
        with torch.autocast(device_type="cpu", dtype=torch.float32, enabled=False):
            _ = torch.ones(10)

    def test_generic_autocast(self):
        # [FUNCTION DOC] For each lower-precision torch op, runs it twice under the same CPU
        # autocast entry point and asserts the two outputs are equal. Both blocks use the same
        # API, so it only checks determinism.
        # KNOWN ISSUE: both blocks use the same autocast API, so no separate generic path is
        # actually compared.
        for op_with_args in self.autocast_lists.torch_16:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            with torch.amp.autocast(device_type="cpu"):
                generic_autocast_output = getattr(torch, op)(*args, **maybe_kwargs)
            with torch.amp.autocast(device_type="cpu"):
                cpu_autocast_output = getattr(torch, op)(*args, **maybe_kwargs)
            self.assertEqual(generic_autocast_output, cpu_autocast_output)

    def test_cpu_autocast_deprecated_warning(self):
        # [FUNCTION DOC] Asserts that using the deprecated CPU autocast context manager emits a
        # FutureWarning pointing to the unified API; the body just creates a tensor.
        with self.assertWarnsRegex(
            FutureWarning,
            r"`torch.cpu.amp.autocast\(args...\)` is deprecated. Please use `torch.amp.autocast\('cpu', args...\)` instead.",
        ):
            with torch.cpu.amp.autocast():
                _ = torch.ones(10)


class CustomLinear(torch.autograd.Function):
    # [CLASS DOC] HELPER: Autograd function computing a linear op in forward and, in backward,
    # computing both input and weight gradients with matmul under autocast on the input device.
    @staticmethod
    def forward(ctx, x, w_t):
        # [FUNCTION DOC] HELPER: Saves x and w_t for backward and returns the linear op of x with
        # w_t.
        ctx.save_for_backward(x, w_t)
        return torch.nn.functional.linear(x, w_t)

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] HELPER: Recomputes the input gradient and the weight gradient (via matmul
        # with transposes) inside an autocast region for the saved input's device type. Returns
        # both.
        x, w_t = ctx.saved_tensors
        with torch.autocast(device_type=x.device.type):
            dL_dX = torch.matmul(grad_output, w_t)
            dL_dW = torch.matmul(x.transpose(0, 1), grad_output).transpose(0, 1)
        return dL_dX, dL_dW


class WeightDTypeCastCounterMode(TorchDispatchMode):
    # [CLASS DOC] HELPER: Dispatch mode that counts casts of a given weight to float16 and, while
    # active, replaces the autocast cache clearing function with a no-op (restored on exit).
    # Attributes: weight (tracked tensor) and dtype_cast_counter (count of casts).
    def __init__(self, weight):
        # [FUNCTION DOC] HELPER: Stores the tracked weight and zeroes dtype_cast_counter.
        super().__init__()
        self.dtype_cast_counter = 0
        self.weight = weight

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] HELPER: Increments dtype_cast_counter when the call is the to-copy op on
        # the tracked weight with float16 target dtype, then runs the op normally and returns its
        # result.
        if (
            func is torch.ops.aten._to_copy.default
            and args[0] is self.weight
            and kwargs["dtype"] is torch.float16
        ):
            self.dtype_cast_counter += 1
        return func(*args, **kwargs)

    def __enter__(self):
        # [FUNCTION DOC] HELPER: Saves and disables the autocast cache clearing function, then
        # enters the base mode. Side effect: mutates the global torch namespace until exit.
        self.old_clear_cache = torch.clear_autocast_cache
        torch.clear_autocast_cache = lambda: None
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # [FUNCTION DOC] HELPER: Restores the saved cache clearing function, then exits the base
        # mode and returns its result.
        torch.clear_autocast_cache = self.old_clear_cache
        return super().__exit__(exc_type, exc_val, exc_tb)


class TestAutocastDevice(TestCase):
    # [CLASS DOC] Device-generic autocast tests, accelerator-only (some MPS-only), checking cast
    # cache behavior and dtype policy; instantiated per device type below.
    hw_classification = HardwareClassification.ACCELERATOR

    @onlyAccelerator
    def test_cast_cache_is_global(self, device):
        # [FUNCTION DOC] Runs forward under autocast and backward (which re-enters autocast) with
        # the cache clearing disabled, and asserts the weight was cast to float16 exactly once,
        # showing the cache is global across forward and backward.
        """
        Verifies that the autocast cache is global. This is done by
        mocking out cache clearing at the end of the forward pass,
        running forward+backward with an explicit call to autocast in the
        backward, and verifying that the weight only get cast to float16 once.
        """

        data = torch.randn(2, 3).to(device)
        weight = torch.nn.Parameter(torch.randn(4, 3).to(device))

        with WeightDTypeCastCounterMode(weight) as mode:
            with torch.autocast(device_type=device):
                output = CustomLinear.apply(data, weight)
                s = output.sum()
            s.backward()

        self.assertEqual(mode.dtype_cast_counter, 1)

    @onlyAccelerator
    def test_cache_disabled(self, device):
        # [FUNCTION DOC] Marks the weight as a cached tensor and enables cached tensors, then runs
        # the same forward and backward and asserts the weight was cast twice (no caching). The
        # finally clause always disables cached tensors again.
        data = torch.randn(2, 3).to(device)
        weight = torch.nn.Parameter(torch.randn(4, 3).to(device))

        try:
            torch._C._set_cached_tensors_enabled(True)
            torch._C._add_cached_tensor(weight)

            with WeightDTypeCastCounterMode(weight) as mode:
                with torch.autocast(device_type=device):
                    output = CustomLinear.apply(data, weight)
                    s = output.sum()
                s.backward()

            # we should not have cached the conversion of the weight
            self.assertEqual(mode.dtype_cast_counter, 2)

        finally:
            torch._C._set_cached_tensors_enabled(False)

    # index_put under AMP follows a cast policy called "promote",
    # https://github.com/pytorch/pytorch/blob/4fcd15a667df5b80e81db6563d8d3123a0cbd051/aten/src/ATen/autocast_mode.h#L205-L230
    # That means:
    #   (1) double precision is ignored,
    #   (2) if any argument is float, then all arguments are promoted to float,
    #   (3) if all arguments are of lower precision dtype, then all dtypes must be equal to the same amp autocast dtype.
    # Since AMP autocast dtype is thread-local, it is not preserved across thread boundaries during autograd execution,
    # and due to the multi-threaded nature of the autograd, the forward pass is being run in bfloat16, while the backward
    # pass defaults to float16. The dtype mismatch leads to the error in the policy, as the criteria (3) is not satisfied.
    # For more info see https://github.com/pytorch/pytorch/issues/132715.
    @onlyAccelerator
    def test_autocast_prioritize(self, device):
        # [FUNCTION DOC] Under bfloat16 autocast runs the index-put op on a grad-requiring
        # bfloat16 tensor, then backpropagates the mean. Regression check that the promote policy
        # does not fail on a dtype mismatch in backward; no explicit assertions, it must not
        # raise.
        dtype = torch.bfloat16

        with torch.autocast(device_type=device, enabled=True, dtype=dtype):
            t = torch.randn([3, 4, 5], dtype=dtype, device=device, requires_grad=True)
            index = torch.randint(
                low=0, high=3, size=[3, 4, 5], dtype=torch.int64, device=device
            )
            val = torch.randn(1, dtype=dtype, device=device)

            res = torch.index_put(t, [index], val)

            loss = res.mean()
            loss.backward()

    @onlyMPS
    def test_mps_autocast_error_message(self, device):
        # [FUNCTION DOC] Asserts a UserWarning about MPS autocast supporting only bfloat16 and
        # float16 is emitted when autocast with float32 is entered on the mps device type.
        with self.assertWarnsRegex(
            UserWarning,
            "MPS Autocast only supports dtypes of torch.bfloat16, torch.float16 currently.",
        ):
            with torch.autocast(device_type="mps", dtype=torch.float32):
                _ = torch.ones(10)

    @onlyMPS
    def test_mps_autocast_bfloat16_supported(self, device):
        # [FUNCTION DOC] Under bfloat16 autocast on mps, multiplies two float32 matrices and
        # asserts the result dtype is bfloat16.
        with torch.amp.autocast(device_type="mps", dtype=torch.bfloat16):
            x = torch.randn(2, 3, device=device)
            y = torch.randn(3, 3, device=device)
            result = torch.mm(x, y)
            self.assertEqual(result.dtype, torch.bfloat16)


class TestTorchAutocast(TestCase):
    # [CLASS DOC] Device-independent tests of the torch.autocast constructor and dtype queries.
    hw_classification = HardwareClassification.GENERIC

    def test_autocast_fast_dtype(self):
        # [FUNCTION DOC] Asserts the default autocast dtype is half for cuda and bfloat16 for cpu.
        gpu_fast_dtype = torch.get_autocast_dtype(device_type="cuda")
        cpu_fast_dtype = torch.get_autocast_dtype(device_type="cpu")
        self.assertEqual(gpu_fast_dtype, torch.half)
        self.assertEqual(cpu_fast_dtype, torch.bfloat16)

    def test_invalid_device(self):
        # [FUNCTION DOC] Asserts that entering autocast with an invalid device string raises
        # RuntimeError with the invalid-device message, and that querying availability for that
        # string raises the same error.
        dev = "not a real device"
        msg = f"Invalid device string: '{dev}'"
        with self.assertRaisesRegex(RuntimeError, msg):
            with torch.autocast(device_type=dev):
                _ = torch.tensor(1)
        with self.assertRaisesRegex(RuntimeError, msg):
            if not torch.amp.is_autocast_available(device_type=dev):
                raise AssertionError(f"autocast should be available for {dev}")

    def test_non_string_device(self):
        # [FUNCTION DOC] Asserts that passing a device object instead of a string as device_type
        # raises ValueError naming the received type.
        """Test that `autocast` throws a ValueError when provided a `torch.device` object for `device_type` instead of a string"""
        dev = torch.device("cpu")
        msg = f"Expected `device_type` of type `str`, got: `{type(dev)}`"
        with self.assertRaisesRegex(expected_exception=ValueError, expected_regex=msg):
            torch.autocast(device_type=dev)

    def test_autocast_called_with_non_callable(self):
        # [FUNCTION DOC] Asserts that applying an autocast object to a tensor instead of a
        # function raises TypeError saying a callable is required.
        """Test that autocast gives a clear error when misused as a function wrapper"""
        x = torch.randn(2, 3)
        msg = r"autocast\(\)\(func\) requires a callable, but got Tensor"
        with self.assertRaisesRegex(TypeError, msg):
            torch.autocast(device_type="cpu")(x)


class TestTorchAutocastDevice(TestCase):
    # [CLASS DOC] Device-generic regression tests for autocast combined with no_grad and
    # inference_mode (a caching issue). Instantiated per device type at module level; the test
    # methods take a device argument and have no device filter decorators.
    hw_classification = HardwareClassification.ACCELERATOR

    def test_autocast_nograd_caching_issue_158232(self, device):
        # [FUNCTION DOC] Within autocast, a no_grad forward must give an output without grad, and
        # a later normal forward must require grad and have a grad_fn. Backward then runs and the
        # parameters must have gradients.
        dtype = torch.get_autocast_dtype(device_type=device)
        model = torch.nn.Linear(2, 2).to(device)
        inp = torch.randn(8, 2, device=device)

        with torch.autocast(device, dtype=dtype, enabled=True):
            # First forward pass in no_grad context (e.g., shape inference)
            with torch.no_grad():
                out1 = model(inp)
                self.assertFalse(
                    out1.requires_grad, "Output in no_grad should not require grad"
                )

            # Second forward pass with gradients enabled (e.g., training)
            out2 = model(inp)
            self.assertTrue(
                out2.requires_grad,
                "Output should require gradients after exiting no_grad",
            )
            self.assertIsNotNone(
                out2.grad_fn, "Output should have grad_fn after exiting no_grad"
            )

            # Backward pass should work
            loss = out2.mean()
            loss.backward()

        # Verify gradients were computed
        self.assertIsNotNone(model.weight.grad)
        self.assertIsNotNone(model.bias.grad)

    def test_autocast_inference_mode_interaction(self, device):
        # [FUNCTION DOC] Checks inference_mode nested inside autocast (output has no grad and has
        # the autocast dtype, later grad forward and backward work) and autocast nested inside
        # inference_mode (output has no grad and the autocast dtype).
        dtype = torch.get_autocast_dtype(device_type=device)
        model = torch.nn.Linear(2, 2).to(device)
        inp = torch.randn(8, 2, device=device)

        # Test 1: inference_mode inside autocast
        with torch.autocast(device, dtype=dtype, enabled=True):
            torch.clear_autocast_cache()
            with torch.inference_mode():
                out1 = model(inp)
                self.assertFalse(out1.requires_grad)
                self.assertEqual(out1.dtype, dtype)

            # After exiting inference_mode, gradients should work
            out2 = model(inp)
            self.assertTrue(out2.requires_grad)
            out2.mean().backward()

        # Test 2: autocast inside inference_mode
        with torch.inference_mode():
            with torch.autocast(device, dtype=dtype, enabled=True):
                out = model(inp)
                self.assertFalse(out.requires_grad)
                self.assertEqual(out.dtype, dtype)

    def test_autocast_caching_still_works_with_gradients(self, device):
        # [FUNCTION DOC] Runs three grad-enabled forwards in one autocast region and asserts every
        # output requires grad and has a grad_fn; backward runs on all three, retaining the graph
        # for the first two.
        dtype = torch.get_autocast_dtype(device_type=device)
        model = torch.nn.Linear(2, 2).to(device)
        inp = torch.randn(8, 2, device=device)

        with torch.autocast(device, dtype=dtype, enabled=True):
            # Multiple forward passes with gradients enabled
            out1 = model(inp)
            out2 = model(inp)
            out3 = model(inp)

            # All should have gradients
            self.assertTrue(out1.requires_grad)
            self.assertTrue(out2.requires_grad)
            self.assertTrue(out3.requires_grad)

            # All should have grad_fn
            self.assertIsNotNone(out1.grad_fn)
            self.assertIsNotNone(out2.grad_fn)
            self.assertIsNotNone(out3.grad_fn)

            # Backward should work on all
            out1.mean().backward(retain_graph=True)
            out2.mean().backward(retain_graph=True)
            out3.mean().backward()

    def test_autocast_mixed_grad_contexts(self, device):
        # [FUNCTION DOC] Alternates no_grad and grad-enabled forwards in one autocast region,
        # asserting requires_grad is false for the no_grad outputs and true for the others, then
        # runs backward on the sum of the two grad-enabled output means.
        dtype = torch.get_autocast_dtype(device_type=device)
        model = torch.nn.Linear(2, 2).to(device)
        inp = torch.randn(8, 2, device=device)

        with torch.autocast(device, dtype=dtype, enabled=True):
            # Pass 1: no_grad
            with torch.no_grad():
                out1 = model(inp)
                self.assertFalse(out1.requires_grad)

            # Pass 2: gradients enabled
            out2 = model(inp)
            self.assertTrue(out2.requires_grad)

            # Pass 3: no_grad again
            with torch.no_grad():
                out3 = model(inp)
                self.assertFalse(out3.requires_grad)

            # Pass 4: gradients enabled again
            out4 = model(inp)
            self.assertTrue(out4.requires_grad)

            # Backward on gradient-enabled outputs
            (out2.mean() + out4.mean()).backward()


instantiate_device_type_tests(TestAutocastDevice, globals(), allow_mps=True)
instantiate_device_type_tests(TestTorchAutocastDevice, globals())

if __name__ == "__main__":
    run_tests()
