# Owner(s): ["module: tests"]
# [MODULE DOC] Self-tests of the torch.testing machinery. Covers assertEqual and callable-
# message behavior of TestCase, isclose semantics, CUDA device-assert early termination,
# OpInfo dtype helpers, device-type test instantiation, make_tensor,
# assert_close/assert_allclose, the TestCase assertion helpers, parametrization utilities,
# SampleInput/OpInfo plumbing, module infos, and decorators. Instantiates device-type test
# classes at import time.

import collections
import dataclasses
import doctest
import functools
import importlib
import inspect
import itertools
import math
import os
import re
import subprocess
import sys
import unittest.mock
from typing import Any
from collections.abc import Callable
from collections.abc import Iterator

import torch

from torch.testing import make_tensor
from torch.testing._internal.common_utils import (
    IS_FBCODE, IS_JETSON, IS_MACOS, IS_SANDCASTLE, IS_WINDOWS, TestCase, run_tests, slowTest,
    parametrize, reparametrize, subtest, instantiate_parametrized_tests, dtype_name,
    TEST_WITH_ROCM, decorateIf, skipIfXpu
)
from torch.testing._internal.common_device_type import \
    (PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY, PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY, dtypes,
     get_device_type_test_bases, instantiate_device_type_tests, onlyCPU, onlyCUDA, onlyNativeDeviceTypes,
     deviceCountAtLeast, ops, expectedFailureMeta, OpDTypes)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal import opinfo
from torch.testing._internal.common_dtype import all_types_and_complex_and, floating_types
from torch.testing._internal.common_modules import modules, module_db, ModuleInfo
from torch.testing._internal.opinfo.core import SampleInput, DecorateInfo, OpInfo
import operator
import string

# For testing TestCase methods and torch.testing functions
class TestTesting(TestCase):
    # [CLASS DOC] Device-generic tests for TestCase assertions and torch.testing helpers.
    # Stateless; it is instantiated per device type by instantiate_device_type_tests after the
    # class definition.
    # Ensure that assertEqual handles numpy arrays properly
    @dtypes(*all_types_and_complex_and(torch.bool, torch.half))
    def test_assertEqual_numpy(self, device, dtype):
        # [FUNCTION DOC] For many dtypes and several shapes (including empty and scalar),
        # builds a random tensor and checks assertEqual accepts numpy array vs tensor in
        # either order and numpy vs numpy, with zero tolerances.
        S = 10
        test_sizes = [
            (),
            (0,),
            (S,),
            (S, S),
            (0, S),
            (S, 0)]
        for test_size in test_sizes:
            a = make_tensor(test_size, dtype=dtype, device=device, low=-5, high=5)
            a_n = a.cpu().numpy()
            msg = f'size: {test_size}'
            self.assertEqual(a_n, a, rtol=0, atol=0, msg=msg)
            self.assertEqual(a, a_n, rtol=0, atol=0, msg=msg)
            self.assertEqual(a_n, a_n, rtol=0, atol=0, msg=msg)

    def test_assertEqual_longMessage(self):
        # [FUNCTION DOC] Checks that with longMessage enabled, a user msg is appended on a new
        # line after the default failure message, which is first captured with longMessage
        # disabled. Restores the original longMessage flag in a finally block.
        actual = "actual"
        expected = "expected"

        long_message = self.longMessage
        try:
            # Capture the default error message by forcing TestCase.longMessage = False
            self.longMessage = False
            try:
                self.assertEqual(actual, expected)
            except AssertionError as error:
                default_msg = str(error)
            else:
                raise AssertionError("AssertionError not raised")

            self.longMessage = True
            extra_msg = "sentinel"
            with self.assertRaisesRegex(AssertionError, re.escape(f"{default_msg}\n{extra_msg}")):
                self.assertEqual(actual, expected, msg=extra_msg)
        finally:
            self.longMessage = long_message

    def test_callable_msg(self):
        # [FUNCTION DOC] Checks that a callable msg is called only on failure, once, with the
        # standard message, and that its return value becomes the final text, for several
        # assert methods. A plain string msg is appended after the standard text.
        # A callable msg is invoked only on failure, across all assert* methods.
        invoked = []

        def lazy(standard_msg):
            # [FUNCTION DOC] HELPER: Message callback that records the standard message it
            # receives in the enclosing list and returns it with a sentinel line appended.
            invoked.append(standard_msg)
            return f"{standard_msg}\nsentinel"

        # Passing: callable not invoked.
        self.assertEqual(1, 1, msg=lazy)
        self.assertTrue(True, msg=lazy)
        self.assertIn(1, [1, 2], msg=lazy)
        self.assertGreater(2, 1, msg=lazy)
        self.assertIsNone(None, msg=lazy)
        self.assertEqual(invoked, [])

        # Failing: callable invoked once with the standard message.
        failing = [
            lambda: self.assertTrue(False, msg=lazy),
            lambda: self.assertIn(9, [1, 2], msg=lazy),
            lambda: self.assertGreater(1, 2, msg=lazy),
            lambda: self.assertEqual(1, 2, msg=lazy),
        ]
        for fail in failing:
            invoked.clear()
            with self.assertRaises(AssertionError) as cm:
                fail()
            self.assertEqual(len(invoked), 1)
            self.assertEqual(str(cm.exception), f"{invoked[0]}\nsentinel")

        # A plain string msg is unchanged.
        with self.assertRaisesRegex(AssertionError, re.escape("True is not false : plain")):
            self.assertFalse(True, msg="plain")

    def _isclose_helper(self, tests, device, dtype, equal_nan, atol=1e-08, rtol=1e-05):
        # [FUNCTION DOC] HELPER: For each (a, b, expected) test triple builds one-element
        # tensors of the dtype on the device, calls isclose with given equal_nan, atol, rtol,
        # and asserts the result equals the expected boolean.
        for test in tests:
            a = torch.tensor((test[0],), device=device, dtype=dtype)
            b = torch.tensor((test[1],), device=device, dtype=dtype)

            actual = torch.isclose(a, b, equal_nan=equal_nan, atol=atol, rtol=rtol)
            expected = test[2]
            self.assertEqual(actual.item(), expected)

    def test_isclose_bool(self, device):
        # [FUNCTION DOC] Checks isclose on bool tensors: equal values are close, differing
        # ones are not.
        tests = (
            (True, True, True),
            (False, False, True),
            (True, False, False),
            (False, True, False),
        )

        self._isclose_helper(tests, device, torch.bool, False)

    @dtypes(torch.uint8,
            torch.int8, torch.int16, torch.int32, torch.int64)
    def test_isclose_integer(self, device, dtype):
        # [FUNCTION DOC] Checks isclose on integer dtypes with default tolerances, then with
        # large atol and rtol, then with negative values where unsigned uint8 wraps so results
        # differ from signed types.
        tests = (
            (0, 0, True),
            (0, 1, False),
            (1, 0, False),
        )

        self._isclose_helper(tests, device, dtype, False)

        # atol and rtol tests
        tests = [
            (0, 1, True),
            (1, 0, False),
            (1, 3, True),
        ]

        self._isclose_helper(tests, device, dtype, False, atol=.5, rtol=.5)

        if dtype is torch.uint8:
            tests = [
                (-1, 1, False),
                (1, -1, False)
            ]
        else:
            tests = [
                (-1, 1, True),
                (1, -1, True)
            ]

        self._isclose_helper(tests, device, dtype, False, atol=1.5, rtol=.5)

    @onlyNativeDeviceTypes
    @dtypes(torch.float16, torch.float32, torch.float64)
    def test_isclose_float(self, device, dtype):
        # [FUNCTION DOC] Checks isclose on float16/32/64 for infinities, NaN, default and
        # loose tolerances (with an epsilon depending on half precision), and that NaNs match
        # only when equal_nan is true.
        tests = (
            (0, 0, True),
            (0, -1, False),
            (float('inf'), float('inf'), True),
            (-float('inf'), float('inf'), False),
            (float('inf'), float('nan'), False),
            (float('nan'), float('nan'), False),
            (0, float('nan'), False),
            (1, 1, True),
        )

        self._isclose_helper(tests, device, dtype, False)

        # atol and rtol tests
        eps = 1e-2 if dtype is torch.half else 1e-6
        tests = (
            (0, 1, True),
            (0, 1 + eps, False),
            (1, 0, False),
            (1, 3, True),
            (1 - eps, 3, False),
            (-.25, .5, True),
            (-.25 - eps, .5, False),
            (.25, -.5, True),
            (.25 + eps, -.5, False),
        )

        self._isclose_helper(tests, device, dtype, False, atol=.5, rtol=.5)

        # equal_nan = True tests
        tests = (
            (0, float('nan'), False),
            (float('inf'), float('nan'), False),
            (float('nan'), float('nan'), True),
        )

        self._isclose_helper(tests, device, dtype, True)

    @unittest.skipIf(IS_SANDCASTLE, "Skipping because doesn't work on sandcastle")
    @dtypes(torch.complex64, torch.complex128)
    def test_isclose_complex(self, device, dtype):
        # [FUNCTION DOC] Checks isclose on complex dtypes: infinities, NaN components, real-
        # only and imaginary-only tolerance cases, complex magnitude-based cases, and
        # equal_nan behavior.
        tests = (
            (complex(1, 1), complex(1, 1 + 1e-8), True),
            (complex(0, 1), complex(1, 1), False),
            (complex(1, 1), complex(1, 0), False),
            (complex(1, 1), complex(1, float('nan')), False),
            (complex(1, float('nan')), complex(1, float('nan')), False),
            (complex(1, 1), complex(1, float('inf')), False),
            (complex(float('inf'), 1), complex(1, float('inf')), False),
            (complex(-float('inf'), 1), complex(1, float('inf')), False),
            (complex(-float('inf'), 1), complex(float('inf'), 1), False),
            (complex(float('inf'), 1), complex(float('inf'), 1), True),
            (complex(float('inf'), 1), complex(float('inf'), 1 + 1e-4), False),
        )

        self._isclose_helper(tests, device, dtype, False)

        # atol and rtol tests

        # atol and rtol tests
        eps = 1e-6
        tests = (
            # Complex versions of float tests (real part)
            (complex(0, 0), complex(1, 0), True),
            (complex(0, 0), complex(1 + eps, 0), False),
            (complex(1, 0), complex(0, 0), False),
            (complex(1, 0), complex(3, 0), True),
            (complex(1 - eps, 0), complex(3, 0), False),
            (complex(-.25, 0), complex(.5, 0), True),
            (complex(-.25 - eps, 0), complex(.5, 0), False),
            (complex(.25, 0), complex(-.5, 0), True),
            (complex(.25 + eps, 0), complex(-.5, 0), False),
            # Complex versions of float tests (imaginary part)
            (complex(0, 0), complex(0, 1), True),
            (complex(0, 0), complex(0, 1 + eps), False),
            (complex(0, 1), complex(0, 0), False),
            (complex(0, 1), complex(0, 3), True),
            (complex(0, 1 - eps), complex(0, 3), False),
            (complex(0, -.25), complex(0, .5), True),
            (complex(0, -.25 - eps), complex(0, .5), False),
            (complex(0, .25), complex(0, -.5), True),
            (complex(0, .25 + eps), complex(0, -.5), False),
        )

        self._isclose_helper(tests, device, dtype, False, atol=.5, rtol=.5)

        # atol and rtol tests for isclose
        tests = (
            # Complex-specific tests
            (complex(1, -1), complex(-1, 1), False),
            (complex(1, -1), complex(2, -2), True),
            (complex(-math.sqrt(2), math.sqrt(2)),
             complex(-math.sqrt(.5), math.sqrt(.5)), True),
            (complex(-math.sqrt(2), math.sqrt(2)),
             complex(-math.sqrt(.501), math.sqrt(.499)), False),
            (complex(2, 4), complex(1., 8.8523607), True),
            (complex(2, 4), complex(1., 8.8523607 + eps), False),
            (complex(1, 99), complex(4, 100), True),
        )
        self._isclose_helper(tests, device, dtype, False, atol=.5, rtol=.5)

        # equal_nan = True tests
        tests = (
            (complex(1, 1), complex(1, float('nan')), False),
            (complex(1, 1), complex(float('nan'), 1), False),
            (complex(float('nan'), 1), complex(float('nan'), 1), True),
            (complex(float('nan'), 1), complex(1, float('nan')), True),
            (complex(float('nan'), float('nan')), complex(float('nan'), float('nan')), True),
        )
        self._isclose_helper(tests, device, dtype, True)

    # Tests that isclose with rtol or atol values less than zero throws a
    #   RuntimeError
    @dtypes(torch.bool, torch.uint8,
            torch.int8, torch.int16, torch.int32, torch.int64,
            torch.float16, torch.float32, torch.float64)
    def test_isclose_atol_rtol_greater_than_zero(self, device, dtype):
        # [FUNCTION DOC] Checks that isclose raises RuntimeError when atol or rtol (or both)
        # is negative.
        t = torch.tensor((1,), device=device, dtype=dtype)

        with self.assertRaises(RuntimeError):
            torch.isclose(t, t, atol=-1, rtol=1)
        with self.assertRaises(RuntimeError):
            torch.isclose(t, t, atol=1, rtol=-1)
        with self.assertRaises(RuntimeError):
            torch.isclose(t, t, atol=-1, rtol=-1)

    def test_isclose_equality_shortcut(self):
        # [FUNCTION DOC] Checks that two int64 tensors differing by one at 2**53 are not close
        # with zero tolerances, i.e. equal-value shortcut compares integers exactly rather
        # than as floats.
        # For values >= 2**53, integers differing by 1 can no longer differentiated by torch.float64 or lower precision
        # floating point dtypes. Thus, even with rtol == 0 and atol == 0, these tensors would be considered close if
        # they were not compared as integers.
        a = torch.tensor(2 ** 53, dtype=torch.int64)
        b = a + 1

        self.assertFalse(torch.isclose(a, b, rtol=0, atol=0))

    @dtypes(torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_isclose_nan_equality_shortcut(self, device, dtype):
        # [FUNCTION DOC] Checks that NaN values (real, or complex with NaN in different parts)
        # compare close under equal_nan with zero tolerances.
        if dtype.is_floating_point:
            a = b = torch.nan
        else:
            a = complex(torch.nan, 0)
            b = complex(0, torch.nan)

        expected = True
        tests = [(a, b, expected)]

        self._isclose_helper(tests, device, dtype, equal_nan=True, rtol=0, atol=0)

    @onlyNativeDeviceTypes
    @dtypes(torch.float16, torch.float32)
    def test_isclose_equal_nan_broadcast(self, device, dtype):
        # [FUNCTION DOC] Regression test: isclose with equal_nan handles broadcasting (one-
        # sided, mutual and same-shape cases) and returns the expected boolean tensors.
        # Regression test: isclose with equal_nan=True should handle broadcasting
        # See https://github.com/pytorch/pytorch/issues/174985
        nan = torch.nan

        # One-sided broadcast: different sizes
        a = torch.tensor([nan], device=device, dtype=dtype)
        b = torch.tensor([nan, nan], device=device, dtype=dtype)
        result = torch.isclose(a, b, equal_nan=True)
        self.assertEqual(result, torch.tensor([True, True], device=device))

        # One-sided broadcast: scalar-like vs vector
        a = torch.tensor([nan], device=device, dtype=dtype)
        b = torch.tensor([nan, 1.0, nan], device=device, dtype=dtype)
        result = torch.isclose(a, b, equal_nan=True)
        self.assertEqual(result, torch.tensor([True, False, True], device=device))

        # Mutual broadcast
        a = torch.tensor([[nan], [1.0]], device=device, dtype=dtype)  # [2, 1]
        b = torch.tensor([[nan, 1.0]], device=device, dtype=dtype)    # [1, 2]
        result = torch.isclose(a, b, equal_nan=True)
        expected = torch.tensor([[True, False], [False, True]], device=device)
        self.assertEqual(result, expected)

        # Same shape (fast path) still works
        a = torch.tensor([nan, 1.0], device=device, dtype=dtype)
        b = torch.tensor([nan, 1.0], device=device, dtype=dtype)
        result = torch.isclose(a, b, equal_nan=True)
        self.assertEqual(result, torch.tensor([True, True], device=device))

    # The following tests (test_cuda_assert_*) are added to ensure test suite terminates early
    # when CUDA assert was thrown. Because all subsequent test will fail if that happens.
    # These tests are slow because it spawn another process to run test suite.
    # See: https://github.com/pytorch/pytorch/issues/49019
    @onlyCUDA
    @slowTest
    def test_cuda_assert_should_stop_common_utils_test_suite(self, device):
        # [FUNCTION DOC] CUDA only, slow. Runs a child process with a small test file whose
        # first test causes an unrecoverable CUDA device assert, and checks stderr shows a
        # device-side assert (or ROCm launch failure) and, on CUDA, that exactly one error is
        # reported (early termination).
        # test to ensure common_utils.py override has early termination for CUDA.
        stderr = TestCase.runWithPytorchAPIUsageStderr("""\
#!/usr/bin/env python3

import torch
from torch.testing._internal.common_utils import (TestCase, run_tests, slowTest)

class TestThatContainsCUDAAssertFailure(TestCase):

    @slowTest
    def test_throw_unrecoverable_cuda_exception(self):
        x = torch.rand(10, device='cuda')
        # cause unrecoverable CUDA exception, recoverable on CPU
        y = x[torch.tensor([25])].cpu()

    @slowTest
    def test_trivial_passing_test_case_on_cpu_cuda(self):
        x1 = torch.tensor([0., 1.], device='cuda')
        x2 = torch.tensor([0., 1.], device='cpu')
        self.assertEqual(x1, x2)

if __name__ == '__main__':
    run_tests()
""")
        # CUDA says "device-side assert triggered"
        # ROCm says "unspecified launch failure" or HSA_STATUS_ERROR_EXCEPTION
        has_cuda_assert = 'CUDA error: device-side assert triggered' in stderr
        has_hip_assert = 'launch failure' in stderr or 'HSA_STATUS_ERROR_EXCEPTION' in stderr
        self.assertTrue(
            has_cuda_assert or has_hip_assert,
            lambda msg: f"{msg}\nExpected device assert error in stderr, got: {stderr}",
        )
        if torch.version.cuda:
            # should run only 1 test because it throws unrecoverable error.
            self.assertIn('errors=1', stderr)


    @onlyCUDA
    @slowTest
    def test_cuda_assert_should_stop_common_device_type_test_suite(self, device):
        # [FUNCTION DOC] Same as the previous test but the child uses device-type
        # instantiation; checks the device assert appears in stderr and the suite stops early
        # with exactly one error on CUDA.
        # test to ensure common_device_type.py override has early termination for CUDA.
        stderr = TestCase.runWithPytorchAPIUsageStderr("""\
#!/usr/bin/env python3

import torch
from torch.testing._internal.common_utils import (TestCase, run_tests, slowTest)
from torch.testing._internal.common_device_type import instantiate_device_type_tests

class TestThatContainsCUDAAssertFailure(TestCase):

    @slowTest
    def test_throw_unrecoverable_cuda_exception(self, device):
        x = torch.rand(10, device=device)
        # cause unrecoverable CUDA exception, recoverable on CPU
        y = x[torch.tensor([25])].cpu()

    @slowTest
    def test_trivial_passing_test_case_on_cpu_cuda(self, device):
        x1 = torch.tensor([0., 1.], device=device)
        x2 = torch.tensor([0., 1.], device='cpu')
        self.assertEqual(x1, x2)

instantiate_device_type_tests(
    TestThatContainsCUDAAssertFailure,
    globals(),
    only_for='cuda'
)

if __name__ == '__main__':
    run_tests()
""")
        # CUDA says "device-side assert triggered"
        # ROCm says "unspecified launch failure" or HSA_STATUS_ERROR_EXCEPTION
        has_cuda_assert = 'CUDA error: device-side assert triggered' in stderr
        has_hip_assert = 'launch failure' in stderr or 'HSA_STATUS_ERROR_EXCEPTION' in stderr
        self.assertTrue(
            has_cuda_assert or has_hip_assert,
            lambda msg: f"{msg}\nExpected device assert error in stderr, got: {stderr}",
        )
        if torch.version.cuda:
            # should run only 1 test because it throws unrecoverable error.
            self.assertIn('errors=1', stderr)


    @unittest.skip("https://github.com/pytorch/pytorch/issues/106308")
    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support device side asserts")
    @onlyCUDA
    @slowTest
    def test_cuda_assert_should_not_stop_common_distributed_test_suite(self, device):
        # [FUNCTION DOC] Skipped (issue link in decorator) and CUDA only. Runs a child process
        # using MultiProcessTestCase with a device assert and expects both tests to error,
        # since early termination is deliberately disabled for distributed test suites.
        # test to ensure common_distributed.py override should not early terminate CUDA.
        stderr = TestCase.runWithPytorchAPIUsageStderr("""\
#!/usr/bin/env python3

import torch
from torch.testing._internal.common_utils import (run_tests, slowTest)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import MultiProcessTestCase

class TestThatContainsCUDAAssertFailure(MultiProcessTestCase):

    @slowTest
    def test_throw_unrecoverable_cuda_exception(self, device):
        x = torch.rand(10, device=device)
        # cause unrecoverable CUDA exception, recoverable on CPU
        y = x[torch.tensor([25])].cpu()

    @slowTest
    def test_trivial_passing_test_case_on_cpu_cuda(self, device):
        x1 = torch.tensor([0., 1.], device=device)
        x2 = torch.tensor([0., 1.], device='cpu')
        self.assertEqual(x1, x2)

instantiate_device_type_tests(
    TestThatContainsCUDAAssertFailure,
    globals(),
    only_for='cuda'
)

if __name__ == '__main__':
    run_tests()
""")
        # we are currently disabling CUDA early termination for distributed tests.
        self.assertIn('errors=2', stderr)

    @expectedFailureMeta  # This is only supported for CPU and CUDA
    @onlyNativeDeviceTypes
    def test_get_supported_dtypes(self, device):
        # [FUNCTION DOC] Expected failure on non-native devices. For atan2, topk and xlogy
        # checks that dtypes found dynamically via get_supported_dtypes and the dispatch hint
        # match the statically declared dtypes for the current device type.
        # Test the `get_supported_dtypes` helper function.
        # We acquire the dtypes for few Ops dynamically and verify them against
        # the correct statically described values.
        ops_to_test = list(filter(lambda op: op.formatted_name in ['atan2', 'topk', 'xlogy'], op_db))

        for op in ops_to_test:
            dynamic_dtypes = opinfo.utils.get_supported_dtypes(op, op.sample_inputs_func, self.device_type)
            dynamic_dispatch = opinfo.utils.dtypes_dispatch_hint(dynamic_dtypes)
            if self.device_type == 'cpu':
                dtypes = op.dtypes
            else:  # device_type ='cuda'
                dtypes = op.dtypesIfCUDA

            self.assertTrue(set(dtypes) == set(dynamic_dtypes))
            self.assertTrue(set(dtypes) == set(dynamic_dispatch.dispatch_fn()))

    @onlyCPU
    @ops(
        [
            op
            for op in op_db
            if len(
                op.supported_dtypes("cpu").symmetric_difference(
                    op.supported_dtypes("cuda")
                )
            )
            > 0
        ][:1],
        dtypes=OpDTypes.none,
    )
    def test_supported_dtypes(self, device, op):
        # [FUNCTION DOC] CPU only, run for one OpInfo whose CPU and CUDA dtypes differ. Checks
        # supported_dtypes differs between cpu and cuda, and that cuda string, cuda:0 string,
        # and device objects with different indices give equal results.
        self.assertNotEqual(op.supported_dtypes("cpu"), op.supported_dtypes("cuda"))
        self.assertEqual(op.supported_dtypes("cuda"), op.supported_dtypes("cuda:0"))
        self.assertEqual(
            op.supported_dtypes(torch.device("cuda")),
            op.supported_dtypes(torch.device("cuda", index=1)),
        )

    def test_setup_and_teardown_run_for_device_specific_tests(self, device):
        # [FUNCTION DOC] Runs a child process that defines and instantiates a device-type test
        # class with setUpClass storing a value and tearDownClass raising it; checks stderr
        # contains the error text naming the device-specific class, proving class setup and
        # teardown run for device tests.
        # TODO: Move this (and other similar text blocks) to some fixtures/ subdir
        stderr = TestCase.runWithPytorchAPIUsageStderr(f"""\
#!/usr/bin/env python3

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase, run_tests

class TestFoo(TestCase):
    @classmethod
    def setUpClass(cls):
        # store something on the test class to query during teardown
        cls.stored_thing = "called with " + cls.__name__

    @classmethod
    def tearDownClass(cls):
        # throw here so we know teardown was run
        raise RuntimeError(cls.stored_thing)

    def test_bar(self, device):
        # make sure the test can access the stored thing
        print(self.stored_thing)

instantiate_device_type_tests(TestFoo, globals(), only_for='{self.device_type}')

if __name__ == '__main__':
    run_tests()
""")
        expected_device_class_name = f"TestFoo{self.device_type.upper()}"
        expected_error_text = f"RuntimeError: called with {expected_device_class_name}"
        self.assertIn(expected_error_text, stderr)


instantiate_device_type_tests(TestTesting, globals())


class TestFrameworkUtils(TestCase):
    # [CLASS DOC] Tests for testing-framework utilities such as device-type test filtering by
    # environment variables. Stateless.

    @unittest.skipIf(IS_WINDOWS, "Skipping because doesn't work for windows")
    @unittest.skipIf(IS_SANDCASTLE, "Skipping because doesn't work on sandcastle")
    def test_filtering_env_var(self):
        # [FUNCTION DOC] Runs a child process with a generated device-type test file under
        # several environments: no filter runs one test per device base, only-for cpu runs
        # one, except-for cpu runs one fewer than all, and setting both filters must not end
        # in OK. Checks the "Ran N test" line.
        # Test environment variable selected device type test generator.
        test_filter_file_template = """\
#!/usr/bin/env python3

import torch
from torch.testing._internal.common_utils import (TestCase, run_tests)
from torch.testing._internal.common_device_type import instantiate_device_type_tests

class TestEnvironmentVariable(TestCase):

    def test_trivial_passing_test(self, device):
        x1 = torch.tensor([0., 1.], device=device)
        x2 = torch.tensor([0., 1.], device='cpu')
        self.assertEqual(x1, x2)

instantiate_device_type_tests(
    TestEnvironmentVariable,
    globals(),
)

if __name__ == '__main__':
    run_tests()
"""
        test_bases_count = len(get_device_type_test_bases())
        # Test without setting env var should run everything.
        env = dict(os.environ)
        for k in ['CI', PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY, PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY]:
            if k in env:
                del env[k]
        _, stderr = TestCase.run_process_no_exception(test_filter_file_template, env=env)
        self.assertIn(f'Ran {test_bases_count} test', stderr.decode('ascii'))

        # Test with setting only_for should only run 1 test.
        env[PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY] = 'cpu'
        _, stderr = TestCase.run_process_no_exception(test_filter_file_template, env=env)
        self.assertIn('Ran 1 test', stderr.decode('ascii'))

        # Test with setting except_for should run 1 less device type from default.
        del env[PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY]
        env[PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY] = 'cpu'
        _, stderr = TestCase.run_process_no_exception(test_filter_file_template, env=env)
        self.assertIn(f'Ran {test_bases_count - 1} test', stderr.decode('ascii'))

        # Test with setting both should throw exception
        env[PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY] = 'cpu'
        _, stderr = TestCase.run_process_no_exception(test_filter_file_template, env=env)
        self.assertNotIn('OK', stderr.decode('ascii'))


class TestEnvironmentDefFlag(TestCase):
    # [CLASS DOC] Tests precedence between an explicit environment variable and an implied-by
    # function in TestEnvironment.def_flag. Tracks names of flags defined during a test in the
    # list _defined and the common_utils module in _cu, so they can be removed afterwards.
    """Verify env-var-vs-implication precedence in TestEnvironment.def_flag."""

    def setUp(self):
        # [FUNCTION DOC] Calls the parent setUp, stores the common_utils module in _cu and
        # initializes the empty list _defined of flag names created by the test.
        super().setUp()
        import torch.testing._internal.common_utils as _cu
        self._cu = _cu
        self._defined: list[str] = []

    def tearDown(self):
        # [FUNCTION DOC] Removes from the common_utils module every flag attribute defined
        # during the test. It does not call the parent tearDown.
        for name in self._defined:
            if hasattr(self._cu, name):
                delattr(self._cu, name)

    def _def_flag(self, name, **kwargs):
        # [FUNCTION DOC] HELPER: Records the flag name for later cleanup, defaults
        # include_in_repro to false and delegates to TestEnvironment.def_flag with the given
        # keyword arguments; returns its value.
        from torch.testing._internal.common_utils import TestEnvironment
        self._defined.append(name)
        kwargs.setdefault("include_in_repro", False)
        return TestEnvironment.def_flag(name, **kwargs)

    def test_explicit_zero_overrides_implication(self):
        # [FUNCTION DOC] Regression: an environment variable set to zero makes the flag false
        # even when the implication function returns true.
        # Regression: PYTORCH_TEST_WITH_ROCM=0 must override
        # implied_by_fn=lambda: torch.version.hip is not None.
        with unittest.mock.patch.dict(os.environ, {"FOO_DF_1": "0"}):
            self.assertFalse(self._def_flag(
                "FOO_DF_1", env_var="FOO_DF_1",
                implied_by_fn=lambda: True))

    def test_explicit_one_with_no_implication(self):
        # [FUNCTION DOC] An environment variable set to one makes the flag true even when the
        # implication function returns false.
        with unittest.mock.patch.dict(os.environ, {"FOO_DF_2": "1"}):
            self.assertTrue(self._def_flag(
                "FOO_DF_2", env_var="FOO_DF_2",
                implied_by_fn=lambda: False))

    def test_unset_with_implication_true(self):
        # [FUNCTION DOC] With the variable unset and the implication returning true, the flag
        # is true.
        env = {k: v for k, v in os.environ.items() if k != "FOO_DF_3"}
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(self._def_flag(
                "FOO_DF_3", env_var="FOO_DF_3",
                implied_by_fn=lambda: True))

    def test_unset_with_implication_false(self):
        # [FUNCTION DOC] With the variable unset and the implication returning false, the flag
        # is false.
        env = {k: v for k, v in os.environ.items() if k != "FOO_DF_4"}
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(self._def_flag(
                "FOO_DF_4", env_var="FOO_DF_4",
                implied_by_fn=lambda: False))

    def test_default_true_explicit_zero_overrides_implication(self):
        # [FUNCTION DOC] With default true, an environment variable of zero still forces the
        # flag false despite a true implication.
        with unittest.mock.patch.dict(os.environ, {"FOO_DF_5": "0"}):
            self.assertFalse(self._def_flag(
                "FOO_DF_5", env_var="FOO_DF_5",
                default=True, implied_by_fn=lambda: True))


def make_assert_close_inputs(actual: Any, expected: Any) -> list[tuple[Any, Any]]:
    # [FUNCTION DOC] HELPER: Returns a list of ten (actual, expected) input pairs for
    # assert_close: the raw pair, then the pair wrapped in tuples, lists, dicts, OrderedDicts
    # and mixed nestings of those containers.
    """Makes inputs for :func:`torch.testing.assert_close` functions based on two examples.

    Args:
        actual (Any): Actual input.
        expected (Any): Expected input.

    Returns:
        List[Tuple[Any, Any]]: Pair of example inputs, as well as the example inputs wrapped in sequences
        (:class:`tuple`, :class:`list`), and mappings (:class:`dict`, :class:`~collections.OrderedDict`).
    """
    return [
        (actual, expected),
        # tuple vs. tuple
        ((actual,), (expected,)),
        # list vs. list
        ([actual], [expected]),
        # tuple vs. list
        ((actual,), [expected]),
        # dict vs. dict
        ({"t": actual}, {"t": expected}),
        # OrderedDict vs. OrderedDict
        (collections.OrderedDict([("t", actual)]), collections.OrderedDict([("t", expected)])),
        # dict vs. OrderedDict
        ({"t": actual}, collections.OrderedDict([("t", expected)])),
        # list of tuples vs. tuple of lists
        ([(actual,)], ([expected],)),
        # list of dicts vs. tuple of OrderedDicts
        ([{"t": actual}], (collections.OrderedDict([("t", expected)]),)),
        # dict of lists vs. OrderedDict of tuples
        ({"t": [actual]}, collections.OrderedDict([("t", (expected,))])),
    ]


def assert_close_with_inputs(actual: Any, expected: Any) -> Iterator[Callable]:
    # [FUNCTION DOC] HELPER: Generator yielding, for each input pair from
    # make_assert_close_inputs, a partial of torch.testing.assert_close with that pair bound
    # as positional arguments, so callers can pass extra keyword options.
    """Yields :func:`torch.testing.assert_close` with predefined positional inputs based on two examples.

    .. note::

        Every test that does not test for a specific input should iterate over this to maximize the coverage.

    Args:
        actual (Any): Actual input.
        expected (Any): Expected input.

    Yields:
        Callable: :func:`torch.testing.assert_close` with predefined positional inputs.
    """
    for inputs in make_assert_close_inputs(actual, expected):
        yield functools.partial(torch.testing.assert_close, *inputs)


class TestAssertClose(TestCase):
    # [CLASS DOC] Tests for torch.testing.assert_close input validation and matching rules,
    # each run over all container wrappings via assert_close_with_inputs. Contains a nested
    # exception class.
    def test_mismatching_types_subclasses(self):
        # [FUNCTION DOC] Tensor vs Parameter subclass compares fine by default.
        actual = torch.rand(())
        expected = torch.nn.Parameter(actual)

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_mismatching_types_type_equality(self):
        # [FUNCTION DOC] With allow_subclasses false, tensor vs Parameter raises TypeError
        # naming the type.
        actual = torch.empty(())
        expected = torch.nn.Parameter(actual)

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(TypeError, str(type(expected))):
                fn(allow_subclasses=False)

    def test_mismatching_types(self):
        # [FUNCTION DOC] Tensor vs numpy array raises TypeError for both allow_subclasses
        # settings.
        actual = torch.empty(2)
        expected = actual.numpy()

        for fn, allow_subclasses in itertools.product(assert_close_with_inputs(actual, expected), (True, False)):
            with self.assertRaisesRegex(TypeError, str(type(expected))):
                fn(allow_subclasses=allow_subclasses)

    def test_unknown_type(self):
        # [FUNCTION DOC] Comparing two plain strings raises TypeError naming the unsupported
        # type.
        actual = "0"
        expected = "0"

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(TypeError, str(type(actual))):
                fn()

    def test_mismatching_shape(self):
        # [FUNCTION DOC] Scalar-shaped vs one-element tensors raise AssertionError mentioning
        # shape.
        actual = torch.empty(())
        expected = actual.clone().reshape((1,))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, "shape"):
                fn()

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), reason="MKLDNN is not available.")
    def test_unknown_layout(self):
        # [FUNCTION DOC] Skipped without MKLDNN. Comparing a dense tensor with an mkldnn
        # tensor raises ValueError mentioning layout.
        actual = torch.empty((2, 2))
        expected = actual.to_mkldnn()

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(ValueError, "layout"):
                fn()

    def test_meta(self):
        # [FUNCTION DOC] Two meta tensors of equal shape compare without error.
        actual = torch.empty((2, 2), device="meta")
        expected = torch.empty((2, 2), device="meta")

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_mismatching_layout(self):
        # [FUNCTION DOC] Every pair of strided, sparse COO and sparse CSR tensors raises
        # AssertionError mentioning layout.
        strided = torch.empty((2, 2))
        sparse_coo = strided.to_sparse()
        sparse_csr = strided.to_sparse_csr()

        for actual, expected in itertools.combinations((strided, sparse_coo, sparse_csr), 2):
            for fn in assert_close_with_inputs(actual, expected):
                with self.assertRaisesRegex(AssertionError, "layout"):
                    fn()

    def test_mismatching_layout_no_check(self):
        # [FUNCTION DOC] The same layout pairs pass when check_layout is false.
        strided = torch.randn((2, 2))
        sparse_coo = strided.to_sparse()
        sparse_csr = strided.to_sparse_csr()

        for actual, expected in itertools.combinations((strided, sparse_coo, sparse_csr), 2):
            for fn in assert_close_with_inputs(actual, expected):
                fn(check_layout=False)

    def test_mismatching_dtype(self):
        # [FUNCTION DOC] Float vs int tensors raise AssertionError mentioning dtype.
        actual = torch.empty((), dtype=torch.float)
        expected = actual.clone().to(torch.int)

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, "dtype"):
                fn()

    def test_mismatching_dtype_no_check(self):
        # [FUNCTION DOC] Float vs int tensors with equal values pass when check_dtype is
        # false.
        actual = torch.ones((), dtype=torch.float)
        expected = actual.clone().to(torch.int)

        for fn in assert_close_with_inputs(actual, expected):
            fn(check_dtype=False)

    def test_mismatching_stride(self):
        # [FUNCTION DOC] With check_stride true, tensors with reversed strides raise
        # AssertionError mentioning stride.
        actual = torch.empty((2, 2))
        expected = torch.as_strided(actual.clone().t().contiguous(), actual.shape, actual.stride()[::-1])

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, "stride"):
                fn(check_stride=True)

    def test_mismatching_stride_no_check(self):
        # [FUNCTION DOC] With default arguments, tensors with different strides but equal
        # values pass.
        actual = torch.rand((2, 2))
        expected = torch.as_strided(actual.clone().t().contiguous(), actual.shape, actual.stride()[::-1])
        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_only_rtol(self):
        # [FUNCTION DOC] Passing only rtol without atol raises ValueError.
        actual = torch.empty(())
        expected = actual.clone()

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaises(ValueError):
                fn(rtol=0.0)

    def test_only_atol(self):
        # [FUNCTION DOC] Passing only atol without rtol raises ValueError.
        actual = torch.empty(())
        expected = actual.clone()

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaises(ValueError):
                fn(atol=0.0)

    def test_mismatching_values(self):
        # [FUNCTION DOC] Different integer values raise AssertionError.
        actual = torch.tensor(1)
        expected = torch.tensor(2)

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaises(AssertionError):
                fn()

    def test_mismatching_values_rtol(self):
        # [FUNCTION DOC] A relative difference above the given rtol with zero atol raises
        # AssertionError.
        eps = 1e-3
        actual = torch.tensor(1.0)
        expected = torch.tensor(1.0 + eps)

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaises(AssertionError):
                fn(rtol=eps / 2, atol=0.0)

    def test_mismatching_values_atol(self):
        # [FUNCTION DOC] An absolute difference above the given atol with zero rtol raises
        # AssertionError.
        eps = 1e-3
        actual = torch.tensor(0.0)
        expected = torch.tensor(eps)

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaises(AssertionError):
                fn(rtol=0.0, atol=eps / 2)

    def test_matching(self):
        # [FUNCTION DOC] A tensor and its clone are close.
        actual = torch.tensor(1.0)
        expected = actual.clone()

        torch.testing.assert_close(actual, expected)

    def test_matching_rtol(self):
        # [FUNCTION DOC] A relative difference within the given rtol with zero atol passes.
        eps = 1e-3
        actual = torch.tensor(1.0)
        expected = torch.tensor(1.0 + eps)

        for fn in assert_close_with_inputs(actual, expected):
            fn(rtol=eps * 2, atol=0.0)

    def test_matching_atol(self):
        # [FUNCTION DOC] An absolute difference within the given atol with zero rtol passes.
        eps = 1e-3
        actual = torch.tensor(0.0)
        expected = torch.tensor(eps)

        for fn in assert_close_with_inputs(actual, expected):
            fn(rtol=0.0, atol=eps * 2)

    # TODO: the code that this test was designed for was removed in https://github.com/pytorch/pytorch/pull/56058
    #  We need to check if this test is still needed or if this behavior is now enabled by default.
    def test_matching_conjugate_bit(self):
        # [FUNCTION DOC] A conjugate-view complex tensor equals the explicitly conjugated
        # value.
        # KNOWN ISSUE: a TODO above says the code this test targeted was removed, so its
        # relevance is unclear.
        actual = torch.tensor(complex(1, 1)).conj()
        expected = torch.tensor(complex(1, -1))

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_matching_nan(self):
        # [FUNCTION DOC] NaN scalars, real or complex, raise AssertionError by default.
        nan = float("NaN")

        tests = (
            (nan, nan),
            (complex(nan, 0), complex(0, nan)),
            (complex(nan, nan), complex(nan, 0)),
            (complex(nan, nan), complex(nan, nan)),
        )

        for actual, expected in tests:
            for fn in assert_close_with_inputs(actual, expected):
                with self.assertRaises(AssertionError):
                    fn()

    def test_matching_nan_with_equal_nan(self):
        # [FUNCTION DOC] The same NaN pairs pass when equal_nan is true.
        nan = float("NaN")

        tests = (
            (nan, nan),
            (complex(nan, 0), complex(0, nan)),
            (complex(nan, nan), complex(nan, 0)),
            (complex(nan, nan), complex(nan, nan)),
        )

        for actual, expected in tests:
            for fn in assert_close_with_inputs(actual, expected):
                fn(equal_nan=True)

    def test_numpy(self):
        # [FUNCTION DOC] A numpy array and a copy of it are close.
        tensor = torch.rand(2, 2, dtype=torch.float32)
        actual = tensor.numpy()
        expected = actual.copy()

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_scalar(self):
        # [FUNCTION DOC] Int, float and complex versions of one random number compare in all
        # pairings, passing check_dtype only when both have the same Python type.
        number = torch.randint(10, size=()).item()
        for actual, expected in itertools.product((int(number), float(number), complex(number)), repeat=2):
            check_dtype = type(actual) is type(expected)

            for fn in assert_close_with_inputs(actual, expected):
                fn(check_dtype=check_dtype)

    def test_bool(self):
        # [FUNCTION DOC] Bool tensors that are equal pass.
        actual = torch.tensor([True, False])
        expected = actual.clone()

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_none(self):
        # [FUNCTION DOC] Two None values pass.
        actual = expected = None

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_none_mismatch(self):
        # [FUNCTION DOC] None against False, zero, NaN or a NaN tensor raises AssertionError.
        expected = None

        for actual in (False, 0, torch.nan, torch.tensor(torch.nan)):
            for fn in assert_close_with_inputs(actual, expected):
                with self.assertRaises(AssertionError):
                    fn()


    def test_docstring_examples(self):
        # [FUNCTION DOC] Runs the doctest examples in the assert_close docstring with torch in
        # the globals and raises AssertionError listing the reports if any fail.
        finder = doctest.DocTestFinder(verbose=False)
        runner = doctest.DocTestRunner(verbose=False, optionflags=doctest.NORMALIZE_WHITESPACE)
        globs = dict(torch=torch)
        doctests = finder.find(torch.testing.assert_close, globs=globs)[0]
        failures = []
        runner.run(doctests, out=lambda report: failures.append(report))
        if failures:
            raise AssertionError(f"Doctest found {len(failures)} failures:\n\n" + "\n".join(failures))

    def test_default_tolerance_selection_mismatching_dtypes(self):
        # [FUNCTION DOC] A bfloat16 and a float64 value pass with check_dtype false, showing
        # tolerances come from the less precise dtype rather than the promoted one.
        # If the default tolerances where selected based on the promoted dtype, i.e. float64,
        # these tensors wouldn't be considered close.
        actual = torch.tensor(0.99, dtype=torch.bfloat16)
        expected = torch.tensor(1.0, dtype=torch.float64)

        for fn in assert_close_with_inputs(actual, expected):
            fn(check_dtype=False)

    class UnexpectedException(Exception):
        # [CLASS DOC] HELPER: Custom exception used by tests that patch TensorLikePair so
        # assert_close sees an unexpected, non-builtin exception.
        """The only purpose of this exception is to test ``assert_close``'s handling of unexpected exceptions. Thus,
        the test should mock a component to raise this instead of the regular behavior. We avoid using a builtin
        exception here to avoid triggering possible handling of them.
        """

    @unittest.mock.patch("torch.testing._comparison.TensorLikePair.__init__", side_effect=UnexpectedException)
    def test_unexpected_error_originate(self, _):
        # [FUNCTION DOC] With the TensorLikePair constructor patched to raise the custom
        # exception, assert_close raises RuntimeError mentioning an unexpected exception. The
        # patched mock argument is unused.
        actual = torch.tensor(1.0)
        expected = actual.clone()

        with self.assertRaisesRegex(RuntimeError, "unexpected exception"):
            torch.testing.assert_close(actual, expected)

    @unittest.mock.patch("torch.testing._comparison.TensorLikePair.compare", side_effect=UnexpectedException)
    def test_unexpected_error_compare(self, _):
        # [FUNCTION DOC] Same as above but the patched method is the compare method.
        actual = torch.tensor(1.0)
        expected = actual.clone()

        with self.assertRaisesRegex(RuntimeError, "unexpected exception"):
            torch.testing.assert_close(actual, expected)




class TestAssertCloseMultiDevice(TestCase):
    # [CLASS DOC] Multi-device assert_close tests; instantiated only for cuda right after the
    # class.
    @deviceCountAtLeast(1)
    def test_mismatching_device(self, devices):
        # [FUNCTION DOC] For every ordered pair of distinct devices among cpu and the given
        # devices, tensors on different devices raise AssertionError mentioning device.
        for actual_device, expected_device in itertools.permutations(("cpu", *devices), 2):
            actual = torch.empty((), device=actual_device)
            expected = actual.clone().to(expected_device)
            for fn in assert_close_with_inputs(actual, expected):
                with self.assertRaisesRegex(AssertionError, "device"):
                    fn()

    @deviceCountAtLeast(1)
    def test_mismatching_device_no_check(self, devices):
        # [FUNCTION DOC] The same device pairs pass when check_device is false.
        for actual_device, expected_device in itertools.permutations(("cpu", *devices), 2):
            actual = torch.rand((), device=actual_device)
            expected = actual.clone().to(expected_device)
            for fn in assert_close_with_inputs(actual, expected):
                fn(check_device=False)


instantiate_device_type_tests(TestAssertCloseMultiDevice, globals(), only_for="cuda")


class TestAssertCloseErrorMessage(TestCase):
    # [CLASS DOC] Tests the wording of assert_close failure messages. Stateless.
    def test_identifier_tensor_likes(self):
        # [FUNCTION DOC] Integer tensor mismatch message mentions Tensor-likes.
        actual = torch.tensor([1, 2, 3, 4])
        expected = torch.tensor([1, 2, 5, 6])

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Tensor-likes")):
                fn()

    def test_identifier_scalars(self):
        # [FUNCTION DOC] Integer scalar mismatch message mentions Scalars.
        actual = 3
        expected = 5
        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Scalars")):
                fn()

    def test_not_equal(self):
        # [FUNCTION DOC] With zero tolerances the message says not equal.
        actual = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        expected = torch.tensor([1, 2, 5, 6], dtype=torch.float32)

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("not equal")):
                fn(rtol=0.0, atol=0.0)

    def test_not_close(self):
        # [FUNCTION DOC] With nonzero tolerances the message says not close.
        actual = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        expected = torch.tensor([1, 2, 5, 6], dtype=torch.float32)

        for fn, (rtol, atol) in itertools.product(
            assert_close_with_inputs(actual, expected), ((1.3e-6, 0.0), (0.0, 1e-5), (1.3e-6, 1e-5))
        ):
            with self.assertRaisesRegex(AssertionError, re.escape("not close")):
                fn(rtol=rtol, atol=atol)

    def test_mismatched_elements(self):
        # [FUNCTION DOC] The message reports two of four elements, 50 percent, mismatched.
        actual = torch.tensor([1, 2, 3, 4])
        expected = torch.tensor([1, 2, 5, 6])

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Mismatched elements: 2 / 4 (50.0%)")):
                fn()

    def test_abs_diff(self):
        # [FUNCTION DOC] The message gives the greatest absolute difference 2 and its index.
        actual = torch.tensor([[1, 2], [3, 4]])
        expected = torch.tensor([[1, 2], [5, 4]])

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Greatest absolute difference: 2 at index (1, 0)")):
                fn()

    def test_small_float_dtype(self):
        # [FUNCTION DOC] For several float8 dtypes and extra trailing singleton dimensions,
        # checks the first mismatched element index in the message is correct, and equal
        # tensors pass.
        for dtype in [
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
            torch.float8_e8m0fnu,
        ]:
            w_vector = torch.tensor([3.14, 1.0], dtype=dtype)
            x_vector = torch.tensor([1.0, 3.14], dtype=dtype)
            y_vector = torch.tensor([3.14, 3.14], dtype=dtype)
            z_vector = torch.tensor([1.0, 3.14], dtype=dtype)

            for additional_dims in range(4):
                new_shape = list(w_vector.shape) + ([1] * additional_dims)
                w_tensor = w_vector.reshape(new_shape)
                x_tensor = x_vector.reshape(new_shape)
                y_tensor = y_vector.reshape(new_shape)
                z_tensor = z_vector.reshape(new_shape)

                for fn in assert_close_with_inputs(x_tensor, y_tensor):
                    expected_shape = (0,) + (0,) * (additional_dims)
                    with self.assertRaisesRegex(
                        AssertionError, re.escape(f"The first mismatched element is at index {expected_shape}")
                    ):
                        fn()

                for fn in assert_close_with_inputs(w_tensor, y_tensor):
                    expected_shape = (1,) + (0,) * (additional_dims)
                    with self.assertRaisesRegex(
                        AssertionError, re.escape(f"The first mismatched element is at index {expected_shape}")
                    ):
                        fn()
                for fn in assert_close_with_inputs(x_tensor, z_tensor):
                    fn()

    def test_abs_diff_scalar(self):
        # [FUNCTION DOC] Scalar message gives absolute difference 2.
        actual = 3
        expected = 5

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Absolute difference: 2")):
                fn()

    def test_rel_diff(self):
        # [FUNCTION DOC] The message gives the greatest relative difference 0.5 and its index.
        actual = torch.tensor([[1, 2], [3, 4]])
        expected = torch.tensor([[1, 4], [3, 4]])

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Greatest relative difference: 0.5 at index (0, 1)")):
                fn()

    def test_rel_diff_scalar(self):
        # [FUNCTION DOC] Scalar message gives relative difference 0.5.
        actual = 2
        expected = 4

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Relative difference: 0.5")):
                fn()

    def test_zero_div_zero(self):
        # [FUNCTION DOC] Only checks AssertionError is raised for a case with matching zero
        # elements. The comment claims the message must not contain nan.
        # KNOWN ISSUE: the regex used matches any text, so the nan-absence claim is not really
        # enforced.
        actual = torch.tensor([1.0, 0.0])
        expected = torch.tensor([2.0, 0.0])

        for fn in assert_close_with_inputs(actual, expected):
            # Although it looks complicated, this regex just makes sure that the word 'nan' is not part of the error
            # message. That would happen if the 0 / 0 is used for the mismatch computation although it matches.
            with self.assertRaisesRegex(AssertionError, "((?!nan).)*"):
                fn()

    def test_rtol(self):
        # [FUNCTION DOC] The message shows the allowed rtol when atol is zero.
        rtol = 1e-3

        actual = torch.tensor((1, 2))
        expected = torch.tensor((2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape(f"(up to {rtol} allowed)")):
                fn(rtol=rtol, atol=0.0)

    def test_atol(self):
        # [FUNCTION DOC] The message shows the allowed atol when rtol is zero.
        atol = 1e-3

        actual = torch.tensor((1, 2))
        expected = torch.tensor((2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape(f"(up to {atol} allowed)")):
                fn(rtol=0.0, atol=atol)

    def test_msg_str(self):
        # [FUNCTION DOC] A string msg argument appears in the AssertionError.
        msg = "Custom error message!"

        actual = torch.tensor(1)
        expected = torch.tensor(2)

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, msg):
                fn(msg=msg)

    def test_msg_callable(self):
        # [FUNCTION DOC] A callable msg argument is used to build the AssertionError text.
        msg = "Custom error message"

        actual = torch.tensor(1)
        expected = torch.tensor(2)

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, msg):
                fn(msg=lambda _: msg)


class TestAssertCloseContainer(TestCase):
    # [CLASS DOC] Tests assert_close on sequences, mappings and dataclasses. Stateless.
    def test_sequence_mismatching_len(self):
        # [FUNCTION DOC] Sequences of different length raise AssertionError.
        actual = (torch.empty(()),)
        expected = ()

        with self.assertRaises(AssertionError):
            torch.testing.assert_close(actual, expected)

    def test_sequence_mismatching_values_msg(self):
        # [FUNCTION DOC] A mismatch in the second sequence item reports the item index in the
        # message.
        t1 = torch.tensor(1)
        t2 = torch.tensor(2)

        actual = (t1, t1)
        expected = (t1, t2)

        with self.assertRaisesRegex(AssertionError, re.escape("item [1]")):
            torch.testing.assert_close(actual, expected)

    def test_mapping_mismatching_keys(self):
        # [FUNCTION DOC] Mappings with different keys raise AssertionError.
        actual = {"a": torch.empty(())}
        expected = {}

        with self.assertRaises(AssertionError):
            torch.testing.assert_close(actual, expected)

    def test_mapping_mismatching_values_msg(self):
        # [FUNCTION DOC] A mismatch in mapping key b reports that key in the message.
        t1 = torch.tensor(1)
        t2 = torch.tensor(2)

        actual = {"a": t1, "b": t1}
        expected = {"a": t1, "b": t2}

        with self.assertRaisesRegex(AssertionError, re.escape("item ['b']")):
            torch.testing.assert_close(actual, expected)

    def test_dataclass_with_tensor_fields(self):
        # [FUNCTION DOC] Defines a local dataclass with a tensor field and checks assertEqual
        # and assert_close succeed for equal instances (shared and distinct tensors) by
        # comparing fields.
        # Python 3.13+ removed the same-object shortcut in dataclass __eq__, so
        # Foo(t, 1) == Foo(t, 1) raises when t is a multi-element tensor. assertEqual
        # / assert_close should still succeed by comparing fields directly.
        @dataclasses.dataclass
        class Foo:
            # [CLASS DOC] HELPER: Local dataclass holding a tensor and an int.
            t: torch.Tensor
            i: int

        t = torch.zeros(2)
        actual = Foo(t, 1)
        expected = Foo(t, 1)

        self.assertEqual(actual, expected)
        for fn in assert_close_with_inputs(actual, expected):
            fn()

        # Distinct equal tensor values should also compare equal.
        self.assertEqual(Foo(torch.zeros(2), 1), Foo(torch.zeros(2), 1))

    def test_dataclass_compare_false_fields_ignored(self):
        # [FUNCTION DOC] Checks fields marked not compared are ignored when comparing
        # dataclass instances.
        @dataclasses.dataclass
        class Foo:
            # [CLASS DOC] HELPER: Local dataclass with a tensor and an ignored integer field
            # excluded from comparison.
            t: torch.Tensor
            ignored: int = dataclasses.field(compare=False, default=0)

        self.assertEqual(Foo(torch.zeros(2), 1), Foo(torch.zeros(2), 2))

    def test_dataclass_mismatching_tensor_field_msg(self):
        # [FUNCTION DOC] Compares local dataclass instances whose two-element tensor fields
        # differ; checks the AssertionError message names the differing field as an item.
        # Multi-element tensors force field-by-field recursion.
        @dataclasses.dataclass
        class Foo:
            # [CLASS DOC] HELPER: Local dataclass with a tensor field and an int field.
            t: torch.Tensor
            i: int

        # Multi-element tensors force the field-recursion path (scalar tensors
        # can make dataclass __eq__ return a bool and fall through to ObjectPair).
        actual = Foo(torch.zeros(2), 0)
        expected = Foo(torch.ones(2), 0)

        with self.assertRaisesRegex(AssertionError, re.escape("item ['t']")):
            torch.testing.assert_close(actual, expected)

    def test_nested_dataclass_with_tensor_fields(self):
        # [FUNCTION DOC] Nested dataclasses (outer holds an inner with a tensor) compare equal
        # via assertEqual when built from the same tensor.
        @dataclasses.dataclass
        class Inner:
            # [CLASS DOC] HELPER: Local dataclass wrapping one tensor field.
            t: torch.Tensor

        @dataclasses.dataclass
        class Outer:
            # [CLASS DOC] HELPER: Local dataclass holding an Inner instance and an int.
            inner: Inner
            i: int

        t = torch.ones(3)
        self.assertEqual(Outer(Inner(t), 1), Outer(Inner(t), 1))

    def test_dataclass_custom_eq_respected(self):
        # [FUNCTION DOC] A dataclass with eq disabled and a custom equality based on tensor
        # shape and tag is compared with its own equality, not field by field: different
        # values but equal shapes pass, different shapes raise AssertionError.
        # eq=False dataclasses with custom __eq__ must not be forced through
        # field-by-field comparison (which would ignore their semantics).
        @dataclasses.dataclass(eq=False)
        class ByShape:
            # [CLASS DOC] HELPER: Dataclass without generated equality; defines equality
            # comparing tensor shape and tag string.
            t: torch.Tensor
            tag: str

            def __eq__(self, other: object) -> bool:
                # [FUNCTION DOC] HELPER: Returns NotImplemented for other types, otherwise
                # whether shapes and tags are equal.
                if not isinstance(other, ByShape):
                    return NotImplemented
                return self.t.shape == other.t.shape and self.tag == other.tag

        # Values differ but shape/tag match: custom __eq__ says equal.
        self.assertEqual(
            ByShape(torch.zeros(2), "a"),
            ByShape(torch.ones(2), "a"),
        )
        with self.assertRaises(AssertionError):
            self.assertEqual(
                ByShape(torch.zeros(2), "a"),
                ByShape(torch.zeros(3), "a"),
            )

    def test_dataclass_union_like_getattr_raises(self):
        # [FUNCTION DOC] Mimics a union-like exported-program type whose unset fields raise
        # AttributeError on access and whose equality uses only the active variant. Checks
        # assertEqual passes for equal instances and raises AssertionError for differing
        # active values.
        # Mimics torch._export.serde.union._Union: unset fields raise on access,
        # and equality is defined by an active variant only.
        @dataclasses.dataclass(eq=False, repr=False)
        class UnionLike:
            # [CLASS DOC] HELPER: Union-like dataclass with optional int and tensor variants
            # plus a private type tag; accessing an unset variant raises AttributeError.
            as_int: int | None = None
            as_tensor: torch.Tensor | None = None
            _type: str = dataclasses.field(default="", repr=False, compare=False)

            def __post_init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the type tag to the name of whichever variant
                # was provided (int checked first, then tensor); leaves it empty if neither.
                if self.as_int is not None:
                    self._type = "as_int"
                elif self.as_tensor is not None:
                    self._type = "as_tensor"

            def __getattribute__(self, name: str) -> object:
                # [FUNCTION DOC] HELPER: Attribute access hook raising AttributeError when an
                # unset variant field is read, i.e. its value is None and it is not the active
                # variant; otherwise returns the value.
                attr = super().__getattribute__(name)
                field_names = {"as_int", "as_tensor"}
                if attr is None and name in field_names and name != self._type:
                    raise AttributeError(f"Field {name} is not set.")
                return attr

            def __eq__(self, other: object) -> bool:
                # [FUNCTION DOC] HELPER: Equal only to another instance of the same class with
                # the same active variant and equal active value; returns False for other
                # types.
                if not isinstance(other, UnionLike):
                    return False
                return self._type == other._type and getattr(self, self._type) == getattr(
                    other, other._type
                )

            def __repr__(self) -> str:
                # [FUNCTION DOC] HELPER: Text showing the active variant name and its value.
                return f"UnionLike({self._type}={getattr(self, self._type)})"

        actual = UnionLike(as_int=1)
        expected = UnionLike(as_int=1)
        self.assertEqual(actual, expected)
        with self.assertRaises(AssertionError):
            self.assertEqual(UnionLike(as_int=1), UnionLike(as_int=2))


class TestAssertCloseSparseCOO(TestCase):
    # [CLASS DOC] Tests assert_close on sparse COO tensors. Stateless.
    def test_matching_coalesced(self):
        # [FUNCTION DOC] Equal coalesced COO tensors pass.
        indices = (
            (0, 1),
            (1, 0),
        )
        values = (1, 2)
        actual = torch.sparse_coo_tensor(indices, values, size=(2, 2)).coalesce()
        expected = actual.clone()

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_matching_uncoalesced(self):
        # [FUNCTION DOC] Equal uncoalesced COO tensors pass.
        indices = (
            (0, 1),
            (1, 0),
        )
        values = (1, 2)
        actual = torch.sparse_coo_tensor(indices, values, size=(2, 2))
        expected = actual.clone()

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_mismatching_sparse_dims(self):
        # [FUNCTION DOC] Different numbers of sparse dimensions raise AssertionError naming
        # that count.
        t = torch.randn(2, 3, 4)
        actual = t.to_sparse()
        expected = t.to_sparse(2)

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("number of sparse dimensions in sparse COO tensors")):
                fn()

    def test_mismatching_nnz(self):
        # [FUNCTION DOC] Different numbers of specified values raise AssertionError naming
        # that count.
        actual_indices = (
            (0, 1),
            (1, 0),
        )
        actual_values = (1, 2)
        actual = torch.sparse_coo_tensor(actual_indices, actual_values, size=(2, 2))

        expected_indices = (
            (0, 1, 1,),
            (1, 0, 0,),
        )
        expected_values = (1, 1, 1)
        expected = torch.sparse_coo_tensor(expected_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("number of specified values in sparse COO tensors")):
                fn()

    def test_mismatching_indices_msg(self):
        # [FUNCTION DOC] Differing index positions raise AssertionError mentioning Sparse COO
        # indices.
        actual_indices = (
            (0, 1),
            (1, 0),
        )
        actual_values = (1, 2)
        actual = torch.sparse_coo_tensor(actual_indices, actual_values, size=(2, 2))

        expected_indices = (
            (0, 1),
            (1, 1),
        )
        expected_values = (1, 2)
        expected = torch.sparse_coo_tensor(expected_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse COO indices")):
                fn()

    def test_mismatching_values_msg(self):
        # [FUNCTION DOC] Differing stored values raise AssertionError mentioning Sparse COO
        # values.
        actual_indices = (
            (0, 1),
            (1, 0),
        )
        actual_values = (1, 2)
        actual = torch.sparse_coo_tensor(actual_indices, actual_values, size=(2, 2))

        expected_indices = (
            (0, 1),
            (1, 0),
        )
        expected_values = (1, 3)
        expected = torch.sparse_coo_tensor(expected_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse COO values")):
                fn()


@unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Not all sandcastle jobs support CSR testing")
class TestAssertCloseSparseCSR(TestCase):
    # [CLASS DOC] Tests assert_close on sparse CSR tensors; skipped in fbcode and sandcastle.
    def test_matching(self):
        # [FUNCTION DOC] Equal CSR tensors pass.
        crow_indices = (0, 1, 2)
        col_indices = (1, 0)
        values = (1, 2)
        actual = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(2, 2))
        expected = actual.clone()

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_mismatching_crow_indices_msg(self):
        # [FUNCTION DOC] Differing compressed row index arrays raise AssertionError mentioning
        # crow_indices.
        actual_crow_indices = (0, 1, 2)
        actual_col_indices = (0, 1)
        actual_values = (1, 2)
        actual = torch.sparse_csr_tensor(actual_crow_indices, actual_col_indices, actual_values, size=(2, 2))

        expected_crow_indices = (0, 2, 2)
        expected_col_indices = actual_col_indices
        expected_values = actual_values
        expected = torch.sparse_csr_tensor(expected_crow_indices, expected_col_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse CSR crow_indices")):
                fn()

    def test_mismatching_col_indices_msg(self):
        # [FUNCTION DOC] Differing column index arrays raise AssertionError mentioning
        # col_indices.
        actual_crow_indices = (0, 1, 2)
        actual_col_indices = (1, 0)
        actual_values = (1, 2)
        actual = torch.sparse_csr_tensor(actual_crow_indices, actual_col_indices, actual_values, size=(2, 2))

        expected_crow_indices = actual_crow_indices
        expected_col_indices = (1, 1)
        expected_values = actual_values
        expected = torch.sparse_csr_tensor(expected_crow_indices, expected_col_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse CSR col_indices")):
                fn()

    def test_mismatching_values_msg(self):
        # [FUNCTION DOC] Differing values raise AssertionError mentioning Sparse CSR values.
        actual_crow_indices = (0, 1, 2)
        actual_col_indices = (1, 0)
        actual_values = (1, 2)
        actual = torch.sparse_csr_tensor(actual_crow_indices, actual_col_indices, actual_values, size=(2, 2))

        expected_crow_indices = actual_crow_indices
        expected_col_indices = actual_col_indices
        expected_values = (1, 3)
        expected = torch.sparse_csr_tensor(expected_crow_indices, expected_col_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse CSR values")):
                fn()


@unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Not all sandcastle jobs support CSC testing")
class TestAssertCloseSparseCSC(TestCase):
    # [CLASS DOC] Tests assert_close on sparse CSC tensors; skipped in fbcode and sandcastle.
    def test_matching(self):
        # [FUNCTION DOC] Equal CSC tensors pass.
        ccol_indices = (0, 1, 2)
        row_indices = (1, 0)
        values = (1, 2)
        actual = torch.sparse_csc_tensor(ccol_indices, row_indices, values, size=(2, 2))
        expected = actual.clone()

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_mismatching_ccol_indices_msg(self):
        # [FUNCTION DOC] Differing compressed column index arrays raise AssertionError
        # mentioning ccol_indices.
        actual_ccol_indices = (0, 1, 2)
        actual_row_indices = (0, 1)
        actual_values = (1, 2)
        actual = torch.sparse_csc_tensor(actual_ccol_indices, actual_row_indices, actual_values, size=(2, 2))

        expected_ccol_indices = (0, 2, 2)
        expected_row_indices = actual_row_indices
        expected_values = actual_values
        expected = torch.sparse_csc_tensor(expected_ccol_indices, expected_row_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse CSC ccol_indices")):
                fn()

    def test_mismatching_row_indices_msg(self):
        # [FUNCTION DOC] Differing row index arrays raise AssertionError mentioning
        # row_indices.
        actual_ccol_indices = (0, 1, 2)
        actual_row_indices = (1, 0)
        actual_values = (1, 2)
        actual = torch.sparse_csc_tensor(actual_ccol_indices, actual_row_indices, actual_values, size=(2, 2))

        expected_ccol_indices = actual_ccol_indices
        expected_row_indices = (1, 1)
        expected_values = actual_values
        expected = torch.sparse_csc_tensor(expected_ccol_indices, expected_row_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse CSC row_indices")):
                fn()

    def test_mismatching_values_msg(self):
        # [FUNCTION DOC] Differing values raise AssertionError mentioning Sparse CSC values.
        actual_ccol_indices = (0, 1, 2)
        actual_row_indices = (1, 0)
        actual_values = (1, 2)
        actual = torch.sparse_csc_tensor(actual_ccol_indices, actual_row_indices, actual_values, size=(2, 2))

        expected_ccol_indices = actual_ccol_indices
        expected_row_indices = actual_row_indices
        expected_values = (1, 3)
        expected = torch.sparse_csc_tensor(expected_ccol_indices, expected_row_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse CSC values")):
                fn()


@unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Not all sandcastle jobs support BSR testing")
class TestAssertCloseSparseBSR(TestCase):
    # [CLASS DOC] Tests assert_close on sparse BSR (blocked) tensors; skipped in fbcode and
    # sandcastle.
    def test_matching(self):
        # [FUNCTION DOC] Equal BSR tensors pass.
        crow_indices = (0, 1, 2)
        col_indices = (1, 0)
        values = ([[1]], [[2]])
        actual = torch.sparse_bsr_tensor(crow_indices, col_indices, values, size=(2, 2))
        expected = actual.clone()

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_mismatching_crow_indices_msg(self):
        # [FUNCTION DOC] Differing compressed row index arrays raise AssertionError mentioning
        # crow_indices.
        actual_crow_indices = (0, 1, 2)
        actual_col_indices = (0, 1)
        actual_values = ([[1]], [[2]])
        actual = torch.sparse_bsr_tensor(actual_crow_indices, actual_col_indices, actual_values, size=(2, 2))

        expected_crow_indices = (0, 2, 2)
        expected_col_indices = actual_col_indices
        expected_values = actual_values
        expected = torch.sparse_bsr_tensor(expected_crow_indices, expected_col_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse BSR crow_indices")):
                fn()

    def test_mismatching_col_indices_msg(self):
        # [FUNCTION DOC] Differing column index arrays raise AssertionError mentioning
        # col_indices.
        actual_crow_indices = (0, 1, 2)
        actual_col_indices = (1, 0)
        actual_values = ([[1]], [[2]])
        actual = torch.sparse_bsr_tensor(actual_crow_indices, actual_col_indices, actual_values, size=(2, 2))

        expected_crow_indices = actual_crow_indices
        expected_col_indices = (1, 1)
        expected_values = actual_values
        expected = torch.sparse_bsr_tensor(expected_crow_indices, expected_col_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse BSR col_indices")):
                fn()

    def test_mismatching_values_msg(self):
        # [FUNCTION DOC] Differing block values raise AssertionError mentioning Sparse BSR
        # values.
        actual_crow_indices = (0, 1, 2)
        actual_col_indices = (1, 0)
        actual_values = ([[1]], [[2]])
        actual = torch.sparse_bsr_tensor(actual_crow_indices, actual_col_indices, actual_values, size=(2, 2))

        expected_crow_indices = actual_crow_indices
        expected_col_indices = actual_col_indices
        expected_values = ([[1]], [[3]])
        expected = torch.sparse_bsr_tensor(expected_crow_indices, expected_col_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse BSR values")):
                fn()


@unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Not all sandcastle jobs support BSC testing")
class TestAssertCloseSparseBSC(TestCase):
    # [CLASS DOC] Tests assert_close on sparse BSC (blocked) tensors; skipped in fbcode and
    # sandcastle.
    def test_matching(self):
        # [FUNCTION DOC] Equal BSC tensors pass.
        ccol_indices = (0, 1, 2)
        row_indices = (1, 0)
        values = ([[1]], [[2]])
        actual = torch.sparse_bsc_tensor(ccol_indices, row_indices, values, size=(2, 2))
        expected = actual.clone()

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_mismatching_ccol_indices_msg(self):
        # [FUNCTION DOC] Differing compressed column index arrays raise AssertionError
        # mentioning ccol_indices.
        actual_ccol_indices = (0, 1, 2)
        actual_row_indices = (0, 1)
        actual_values = ([[1]], [[2]])
        actual = torch.sparse_bsc_tensor(actual_ccol_indices, actual_row_indices, actual_values, size=(2, 2))

        expected_ccol_indices = (0, 2, 2)
        expected_row_indices = actual_row_indices
        expected_values = actual_values
        expected = torch.sparse_bsc_tensor(expected_ccol_indices, expected_row_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse BSC ccol_indices")):
                fn()

    def test_mismatching_row_indices_msg(self):
        # [FUNCTION DOC] Differing row index arrays raise AssertionError mentioning
        # row_indices.
        actual_ccol_indices = (0, 1, 2)
        actual_row_indices = (1, 0)
        actual_values = ([[1]], [[2]])
        actual = torch.sparse_bsc_tensor(actual_ccol_indices, actual_row_indices, actual_values, size=(2, 2))

        expected_ccol_indices = actual_ccol_indices
        expected_row_indices = (1, 1)
        expected_values = actual_values
        expected = torch.sparse_bsc_tensor(expected_ccol_indices, expected_row_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse BSC row_indices")):
                fn()

    def test_mismatching_values_msg(self):
        # [FUNCTION DOC] Differing block values raise AssertionError mentioning Sparse BSC
        # values.
        actual_ccol_indices = (0, 1, 2)
        actual_row_indices = (1, 0)
        actual_values = ([[1]], [[2]])
        actual = torch.sparse_bsc_tensor(actual_ccol_indices, actual_row_indices, actual_values, size=(2, 2))

        expected_ccol_indices = actual_ccol_indices
        expected_row_indices = actual_row_indices
        expected_values = ([[1]], [[3]])
        expected = torch.sparse_bsc_tensor(expected_ccol_indices, expected_row_indices, expected_values, size=(2, 2))

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse BSC values")):
                fn()


class TestAssertCloseQuantized(TestCase):
    # [CLASS DOC] Tests assert_close on quantized tensors. Stateless.
    def test_mismatching_is_quantized(self):
        # [FUNCTION DOC] A float tensor vs a quantized tensor raises AssertionError mentioning
        # is_quantized.
        actual = torch.tensor(1.0)
        expected = torch.quantize_per_tensor(actual, scale=1.0, zero_point=0, dtype=torch.qint32)

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, "is_quantized"):
                fn()

    def test_mismatching_qscheme(self):
        # [FUNCTION DOC] Per-tensor vs per-channel quantized tensors raise AssertionError
        # mentioning qscheme.
        t = torch.tensor((1.0,))
        actual = torch.quantize_per_tensor(t, scale=1.0, zero_point=0, dtype=torch.qint32)
        expected = torch.quantize_per_channel(
            t,
            scales=torch.tensor((1.0,)),
            zero_points=torch.tensor((0,)),
            axis=0,
            dtype=torch.qint32,
        )

        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, "qscheme"):
                fn()

    def test_matching_per_tensor(self):
        # [FUNCTION DOC] A per-tensor quantized tensor and its clone pass.
        actual = torch.quantize_per_tensor(torch.tensor(1.0), scale=1.0, zero_point=0, dtype=torch.qint32)
        expected = actual.clone()

        for fn in assert_close_with_inputs(actual, expected):
            fn()

    def test_matching_per_channel(self):
        # [FUNCTION DOC] A per-channel quantized tensor and its clone pass.
        actual = torch.quantize_per_channel(
            torch.tensor((1.0,)),
            scales=torch.tensor((1.0,)),
            zero_points=torch.tensor((0,)),
            axis=0,
            dtype=torch.qint32,
        )
        expected = actual.clone()

        for fn in assert_close_with_inputs(actual, expected):
            fn()


class TestMakeTensor(TestCase):
    # [CLASS DOC] Tests for torch.testing.make_tensor. The class attribute supported_dtypes is
    # a dtypes decorator covering bool, integer, half, bfloat16, float and complex types used
    # by most tests.
    supported_dtypes = dtypes(
        torch.bool,
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        torch.float16, torch.bfloat16, torch.float32, torch.float64,
        torch.complex32, torch.complex64, torch.complex128,
    )

    @supported_dtypes
    @parametrize("shape", [(), (0,), (1,), (1, 1), (2,), (2, 3), (8, 16, 32)])
    @parametrize("splat_shape", [False, True])
    def test_smoke(self, dtype, device, shape, splat_shape):
        # [FUNCTION DOC] For many shapes (with shape passed as one argument or splatted),
        # checks the result is a tensor with the requested shape, dtype and device.
        t = torch.testing.make_tensor(*shape if splat_shape else shape, dtype=dtype, device=device)

        self.assertIsInstance(t, torch.Tensor)
        self.assertEqual(t.shape, shape)
        self.assertEqual(t.dtype, dtype)
        self.assertEqual(t.device, torch.device(device))

    @supported_dtypes
    @parametrize("requires_grad", [False, True])
    def test_requires_grad(self, dtype, device, requires_grad):
        # [FUNCTION DOC] requires_grad is honored for float and complex dtypes; for bool and
        # integer dtypes asking for it raises ValueError with an explanatory message.
        make_tensor = functools.partial(
            torch.testing.make_tensor,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

        if not requires_grad or dtype.is_floating_point or dtype.is_complex:
            t = make_tensor()
            self.assertEqual(t.requires_grad, requires_grad)
        else:
            with self.assertRaisesRegex(
                    ValueError, "`requires_grad=True` is not supported for boolean and integral dtypes"
            ):
                make_tensor()

    @supported_dtypes
    @parametrize("noncontiguous", [False, True])
    @parametrize("shape", [(), (0,), (1,), (1, 1), (2,), (2, 3), (8, 16, 32)])
    def test_noncontiguous(self, dtype, device, noncontiguous, shape):
        # [FUNCTION DOC] Checks the tensor is contiguous unless noncontiguous was requested
        # and it has at least two elements.
        numel = functools.reduce(operator.mul, shape, 1)

        t = torch.testing.make_tensor(shape, dtype=dtype, device=device, noncontiguous=noncontiguous)
        self.assertEqual(t.is_contiguous(), not noncontiguous or numel < 2)

    @supported_dtypes
    @parametrize(
        "memory_format_and_shape",
        [
            (None, (2, 3, 4)),
            (torch.contiguous_format, (2, 3, 4)),
            (torch.channels_last, (2, 3, 4, 5)),
            (torch.channels_last_3d, (2, 3, 4, 5, 6)),
            (torch.preserve_format, (2, 3, 4)),
        ],
    )
    def test_memory_format(self, dtype, device, memory_format_and_shape):
        # [FUNCTION DOC] For several memory formats with matching ranks (None means
        # contiguous), checks the result is contiguous in that format.
        memory_format, shape = memory_format_and_shape

        t = torch.testing.make_tensor(shape, dtype=dtype, device=device, memory_format=memory_format)

        self.assertTrue(
            t.is_contiguous(memory_format=torch.contiguous_format if memory_format is None else memory_format)
        )

    @supported_dtypes
    def test_noncontiguous_memory_format(self, dtype, device):
        # [FUNCTION DOC] Passing both noncontiguous and memory_format raises ValueError saying
        # they are mutually exclusive.
        with self.assertRaisesRegex(ValueError, "`noncontiguous` and `memory_format` are mutually exclusive"):
            torch.testing.make_tensor(
                (2, 3, 4, 5),
                dtype=dtype,
                device=device,
                noncontiguous=True,
                memory_format=torch.channels_last,
            )

    @supported_dtypes
    def test_exclude_zero(self, dtype, device):
        # [FUNCTION DOC] With exclude_zero set, a 10000-element tensor with range -1 to 2 has
        # no zeros.
        t = torch.testing.make_tensor(10_000, dtype=dtype, device=device, exclude_zero=True, low=-1, high=2)

        self.assertTrue((t != 0).all())

    @supported_dtypes
    def test_low_high_smoke(self, dtype, device):
        # [FUNCTION DOC] Checks values lie within [low, high) for explicit bounds 0 and 2
        # (viewing complex as real pairs first).
        low_inclusive, high_exclusive = 0, 2

        t = torch.testing.make_tensor(10_000, dtype=dtype, device=device, low=low_inclusive, high=high_exclusive)
        if dtype.is_complex:
            t = torch.view_as_real(t)

        self.assertTrue(((t >= low_inclusive) & (t < high_exclusive)).all())

    @supported_dtypes
    def test_low_high_default_smoke(self, dtype, device):
        # [FUNCTION DOC] Uses the per-dtype default ranges (bool 0-2, uint8 0-10, signed ints
        # -9-10, others -9-9) as explicit bounds and checks all values fall within them.
        low_inclusive, high_exclusive = {
            torch.bool: (0, 2),
            torch.uint8: (0, 10),
            **dict.fromkeys([torch.int8, torch.int16, torch.int32, torch.int64], (-9, 10)),
        }.get(dtype, (-9, 9))

        t = torch.testing.make_tensor(10_000, dtype=dtype, device=device, low=low_inclusive, high=high_exclusive)
        if dtype.is_complex:
            t = torch.view_as_real(t)

        self.assertTrue(((t >= low_inclusive) & (t < high_exclusive)).all())

    @parametrize("low_high", [(0, 0), (1, 0), (0, -1)])
    @parametrize("value_types", list(itertools.product([int, float], repeat=2)))
    @supported_dtypes
    def test_low_ge_high(self, dtype, device, low_high, value_types):
        # [FUNCTION DOC] For low not less than high, floating or complex dtypes with equal
        # bounds emit a FutureWarning and fill with that value; all other cases raise
        # ValueError about low and high.
        low, high = (value_type(value) for value, value_type in zip(low_high, value_types))

        if low == high and (dtype.is_floating_point or dtype.is_complex):
            with self.assertWarnsRegex(
                    FutureWarning,
                    "Passing `low==high` to `torch.testing.make_tensor` for floating or complex types is deprecated",
            ):
                t = torch.testing.make_tensor(10_000, dtype=dtype, device=device, low=low, high=high)
            self.assertEqual(t, torch.full_like(t, complex(low, low) if dtype.is_complex else low))
        else:
            with self.assertRaisesRegex(ValueError, "`low` must be less than `high`"):
                torch.testing.make_tensor(dtype=dtype, device=device, low=low, high=high)

    @supported_dtypes
    @parametrize("low_high", [(None, torch.nan), (torch.nan, None), (torch.nan, torch.nan)])
    def test_low_high_nan(self, dtype, device, low_high):
        # [FUNCTION DOC] A NaN low or high raises ValueError saying neither can be NaN.
        low, high = low_high

        with self.assertRaisesRegex(ValueError, "`low` and `high` cannot be NaN"):
            torch.testing.make_tensor(dtype=dtype, device=device, low=low, high=high)

    @supported_dtypes
    def test_low_high_outside_valid_range(self, dtype, device):
        # [FUNCTION DOC] Uses the dtype's integer min and max to build bounds outside the
        # valid range (both below and above) and checks make_tensor raises ValueError.
        make_tensor = functools.partial(torch.testing.make_tensor, dtype=dtype, device=device)

        def get_dtype_limits(dtype):
            # [FUNCTION DOC] HELPER: Returns the integer (min, max) of a dtype, using 0 and 1
            # for bool, finfo for float/complex and iinfo for integers; bounds are converted
            # to Python ints so they can exceed range.
            if dtype is torch.bool:
                return 0, 1

            info = (torch.finfo if dtype.is_floating_point or dtype.is_complex else torch.iinfo)(dtype)
            # We are using integer bounds here, because otherwise it would be impossible to pass `low` and `high`
            # outside their valid range. Python uses 64bit floating point numbers and thus trying to do something like
            # `torch.ffinfo(torch.float64)max * 2` will always result in `inf`. On the flipside, Pythons `int` is
            # unbounded.
            return int(info.min), int(info.max)

        lowest_inclusive, highest_inclusive = get_dtype_limits(dtype)

        with self.assertRaisesRegex(ValueError, ""):
            low, high = (-2, -1) if lowest_inclusive == 0 else (lowest_inclusive * 4, lowest_inclusive * 2)
            make_tensor(low=low, high=high)

        with self.assertRaisesRegex(ValueError, ""):
            make_tensor(low=highest_inclusive * 2, high=highest_inclusive * 4)

    @dtypes(torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_low_high_boolean_integral1(self, dtype, device):
        # [FUNCTION DOC] For bool and integer dtypes, bounds just inside (-1, 1) produce a
        # tensor of all zeros.
        shape = (10_000,)
        eps = 1e-4

        actual = torch.testing.make_tensor(shape, dtype=dtype, device=device, low=-(1 - eps), high=1 - eps)
        expected = torch.zeros(shape, dtype=dtype, device=device)

        torch.testing.assert_close(actual, expected)

    @dtypes(torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_low_high_boolean_integral2(self, dtype, device):
        # [FUNCTION DOC] For bool and integer dtypes, bounds one apart at the top of the
        # representable range (for int64 one below the max) produce a tensor filled with the
        # low bound.
        shape = (10_000,)
        if dtype is torch.bool:
            low = 1
        elif dtype is torch.int64:
            # Due to its internals, `make_tensor` is not able to sample `torch.iinfo(torch.int64).max`
            low = torch.iinfo(dtype).max - 1
        else:
            low = torch.iinfo(dtype).max
        high = low + 1

        actual = torch.testing.make_tensor(shape, dtype=dtype, device=device, low=low, high=high)
        expected = torch.full(shape, low, dtype=dtype, device=device)

        torch.testing.assert_close(actual, expected)


instantiate_device_type_tests(TestMakeTensor, globals())


def _get_test_names_for_test_class(test_cls):
    # [FUNCTION DOC] HELPER: Returns the sorted list of qualified names (class name dot
    # attribute) of all attributes in the class own namespace whose names start with test_.
    """ Convenience function to get all test names for a given test class. """
    test_names = [f'{test_cls.__name__}.{key}' for key in test_cls.__dict__
                  if key.startswith('test_')]
    return sorted(test_names)


def _get_test_funcs_for_test_class(test_cls):
    # [FUNCTION DOC] HELPER: Returns a list of (attribute, name) pairs for all attributes in
    # the class own namespace whose names start with test_.
    """ Convenience function to get all (test function, parametrized_name) pairs for a given test class. """
    test_funcs = [(getattr(test_cls, key), key) for key in test_cls.__dict__ if key.startswith('test_')]
    return test_funcs


class TestTestParametrization(TestCase):
    # [CLASS DOC] Tests names and decorator handling produced by parametrize and
    # instantiate_parametrized_tests, using throwaway TestCase classes defined inside each
    # test.
    def test_default_names(self):
        # [FUNCTION DOC] Checks default generated names: single param gets name_value, multi-
        # param gets each name_value joined.

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class with two parametrized no-op tests.
            @parametrize("x", range(5))
            def test_default_names(self, x):
                # [FUNCTION DOC] HELPER: No-op body; only its generated names are inspected.
                pass

            @parametrize("x,y", [(1, 2), (2, 3), (3, 4)])
            def test_two_things_default_names(self, x, y):
                # [FUNCTION DOC] HELPER: No-op body; only its generated names are inspected.
                pass

        instantiate_parametrized_tests(TestParametrized)

        expected_test_names = [
            'TestParametrized.test_default_names_x_0',
            'TestParametrized.test_default_names_x_1',
            'TestParametrized.test_default_names_x_2',
            'TestParametrized.test_default_names_x_3',
            'TestParametrized.test_default_names_x_4',
            'TestParametrized.test_two_things_default_names_x_1_y_2',
            'TestParametrized.test_two_things_default_names_x_2_y_3',
            'TestParametrized.test_two_things_default_names_x_3_y_4',
        ]
        test_names = _get_test_names_for_test_class(TestParametrized)
        self.assertEqual(expected_test_names, test_names)

    def test_name_fn(self):
        # [FUNCTION DOC] Checks name_fn customizes names for a single param, composes across
        # stacked parametrize decorators, and accepts multiple params for a multi-name
        # parametrize.

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class with three parametrized no-op tests using
            # name functions.
            @parametrize("bias", [False, True], name_fn=lambda b: 'bias' if b else 'no_bias')
            def test_custom_names(self, bias):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

            @parametrize("x", [1, 2], name_fn=str)
            @parametrize("y", [3, 4], name_fn=str)
            @parametrize("z", [5, 6], name_fn=str)
            def test_three_things_composition_custom_names(self, x, y, z):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

            @parametrize("x,y", [(1, 2), (1, 3), (1, 4)], name_fn=lambda x, y: f'{x}__{y}')
            def test_two_things_custom_names_alternate(self, x, y):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        instantiate_parametrized_tests(TestParametrized)

        expected_test_names = [
            'TestParametrized.test_custom_names_bias',
            'TestParametrized.test_custom_names_no_bias',
            'TestParametrized.test_three_things_composition_custom_names_1_3_5',
            'TestParametrized.test_three_things_composition_custom_names_1_3_6',
            'TestParametrized.test_three_things_composition_custom_names_1_4_5',
            'TestParametrized.test_three_things_composition_custom_names_1_4_6',
            'TestParametrized.test_three_things_composition_custom_names_2_3_5',
            'TestParametrized.test_three_things_composition_custom_names_2_3_6',
            'TestParametrized.test_three_things_composition_custom_names_2_4_5',
            'TestParametrized.test_three_things_composition_custom_names_2_4_6',
            'TestParametrized.test_two_things_custom_names_alternate_1__2',
            'TestParametrized.test_two_things_custom_names_alternate_1__3',
            'TestParametrized.test_two_things_custom_names_alternate_1__4',
        ]
        test_names = _get_test_names_for_test_class(TestParametrized)
        self.assertEqual(expected_test_names, test_names)

    def test_name_fn_with_dot_raises(self):
        # [FUNCTION DOC] A name function producing a dot in a test name makes instantiation
        # raise RuntimeError, because dots break unittest name loading.
        # Dots in test names break unittest.TestLoader.loadTestsFromName, so
        # instantiation should fail loudly rather than produce an unloadable test.
        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class whose parametrization generates a name
            # containing a dot.
            @parametrize("dtype", [torch.bfloat16], name_fn=str)
            def test_bad_name(self, dtype):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        with self.assertRaisesRegex(RuntimeError, 'contains a "." character'):
            instantiate_parametrized_tests(TestParametrized)

    def test_subtest_name_with_dot_raises(self):
        # [FUNCTION DOC] A subtest with a dotted name makes instantiation raise RuntimeError.
        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class with a subtest named with a dot.
            @parametrize("x", [subtest(1, name="a.b")])
            def test_bad_name(self, x):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        with self.assertRaisesRegex(RuntimeError, 'contains a "." character'):
            instantiate_parametrized_tests(TestParametrized)

    def test_reparametrize(self):
        # [FUNCTION DOC] Checks reparametrize can add an argument and rename tests, or drop
        # cases by yielding None, and verifies the resulting test names.

        def include_is_even_arg(test_name, param_kwargs):
            # [FUNCTION DOC] HELPER: Reparametrization callback; yields the test name suffixed
            # with even or odd and the kwargs extended with an is_even boolean.
            x = param_kwargs["x"]
            is_even = x % 2 == 0
            new_param_kwargs = dict(param_kwargs)
            new_param_kwargs["is_even"] = is_even
            is_even_suffix = "_even" if is_even else "_odd"
            new_test_name = f"{test_name}{is_even_suffix}"
            yield (new_test_name, new_param_kwargs)

        def exclude_odds(test_name, param_kwargs):
            # [FUNCTION DOC] HELPER: Reparametrization callback; yields the unchanged name and
            # kwargs for even values and None (dropping the case) for odd values.
            x = param_kwargs["x"]
            is_even = x % 2 == 0
            yield None if not is_even else (test_name, param_kwargs)

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class using reparametrize in two tests.
            @reparametrize(parametrize("x", range(5)), include_is_even_arg)
            def test_foo(self, x, is_even):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

            @reparametrize(parametrize("x", range(5)), exclude_odds)
            def test_bar(self, x):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        instantiate_parametrized_tests(TestParametrized)

        expected_test_names = [
            'TestParametrized.test_bar_x_0',
            'TestParametrized.test_bar_x_2',
            'TestParametrized.test_bar_x_4',
            'TestParametrized.test_foo_x_0_even',
            'TestParametrized.test_foo_x_1_odd',
            'TestParametrized.test_foo_x_2_even',
            'TestParametrized.test_foo_x_3_odd',
            'TestParametrized.test_foo_x_4_even',
        ]
        test_names = _get_test_names_for_test_class(TestParametrized)
        self.assertEqual(expected_test_names, test_names)

    def test_subtest_names(self):
        # [FUNCTION DOC] Checks subtest names replace default value-based names, for single
        # and multi-param cases, and verifies the resulting sorted test names.

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class with named subtests.
            @parametrize("bias", [subtest(True, name='bias'),
                                  subtest(False, name='no_bias')])
            def test_custom_names(self, bias):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

            @parametrize("x,y", [subtest((1, 2), name='double'),
                                 subtest((1, 3), name='triple'),
                                 subtest((1, 4), name='quadruple')])
            def test_two_things_custom_names(self, x, y):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        instantiate_parametrized_tests(TestParametrized)

        expected_test_names = [
            'TestParametrized.test_custom_names_bias',
            'TestParametrized.test_custom_names_no_bias',
            'TestParametrized.test_two_things_custom_names_double',
            'TestParametrized.test_two_things_custom_names_quadruple',
            'TestParametrized.test_two_things_custom_names_triple',
        ]
        test_names = _get_test_names_for_test_class(TestParametrized)
        self.assertEqual(expected_test_names, test_names)

    def test_apply_param_specific_decorators(self):
        # [FUNCTION DOC] Checks decorators attached to one subtest are applied only to that
        # generated test.
        # Test that decorators can be applied on a per-param basis.

        def test_dec(func):
            # [FUNCTION DOC] HELPER: Decorator that sets a marker attribute on the function
            # and returns it.
            func._decorator_applied = True
            return func

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class with one parametrize listing three subtests.
            @parametrize("x", [subtest(1, name='one'),
                               subtest(2, name='two', decorators=[test_dec]),
                               subtest(3, name='three')])
            def test_param(self, x):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        instantiate_parametrized_tests(TestParametrized)

        for test_func, name in _get_test_funcs_for_test_class(TestParametrized):
            self.assertEqual(hasattr(test_func, '_decorator_applied'), name == 'test_param_two')

    def test_compose_param_specific_decorators(self):
        # [FUNCTION DOC] Checks per-subtest decorators from two stacked parametrize decorators
        # compose: the marker is on tests where x is 2 or y is false.
        # Test that multiple per-param decorators compose correctly.

        def test_dec(func):
            # [FUNCTION DOC] HELPER: Decorator that sets a marker attribute on the function
            # and returns it.
            func._decorator_applied = True
            return func

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class with two stacked parametrize decorators.
            @parametrize("x", [subtest(1),
                               subtest(2, decorators=[test_dec]),
                               subtest(3)])
            @parametrize("y", [subtest(False, decorators=[test_dec]),
                               subtest(True)])
            def test_param(self, x, y):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        instantiate_parametrized_tests(TestParametrized)

        for test_func, name in _get_test_funcs_for_test_class(TestParametrized):
            # Decorator should be applied whenever either x == 2 or y == False.
            should_apply = ('x_2' in name) or ('y_False' in name)
            self.assertEqual(hasattr(test_func, '_decorator_applied'), should_apply)

    def test_modules_decorator_misuse_error(self):
        # [FUNCTION DOC] Using the modules decorator with instantiate_parametrized_tests
        # raises RuntimeError saying it is for device-specific contexts.
        # Test that @modules errors out when used with instantiate_parametrized_tests().

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class using the modules decorator.
            @modules(module_db)
            def test_modules(self, module_info):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        with self.assertRaisesRegex(RuntimeError, 'intended to be used in a device-specific context'):
            instantiate_parametrized_tests(TestParametrized)

    def test_ops_decorator_misuse_error(self):
        # [FUNCTION DOC] Using the ops decorator with instantiate_parametrized_tests raises
        # RuntimeError saying it is for device-specific contexts.
        # Test that @ops errors out when used with instantiate_parametrized_tests().

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class using the ops decorator.
            @ops(op_db)
            def test_ops(self, module_info):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        with self.assertRaisesRegex(RuntimeError, 'intended to be used in a device-specific context'):
            instantiate_parametrized_tests(TestParametrized)

    def test_multiple_handling_of_same_param_error(self):
        # [FUNCTION DOC] Two parametrize decorators for the same parameter name make
        # instantiation raise RuntimeError about multiple parametrization decorators.
        # Test that multiple decorators handling the same param errors out.

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class with duplicate parametrization of one
            # argument.
            @parametrize("x", range(3))
            @parametrize("x", range(5))
            def test_param(self, x):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        with self.assertRaisesRegex(RuntimeError, 'multiple parametrization decorators'):
            instantiate_parametrized_tests(TestParametrized)

    @parametrize("x", [1, subtest(2, decorators=[unittest.expectedFailure]), 3])
    def test_subtest_expected_failure(self, x):
        # [FUNCTION DOC] Parametrized test where the subtest for x equal 2 is marked expected
        # failure and raises RuntimeError; other values pass.
        if x == 2:
            raise RuntimeError('Boom')

    @parametrize("x", [subtest(1, decorators=[unittest.expectedFailure]), 2, 3])
    @parametrize("y", [4, 5, subtest(6, decorators=[unittest.expectedFailure])])
    def test_two_things_subtest_expected_failure(self, x, y):
        # [FUNCTION DOC] Two stacked parametrizations with expected-failure subtests (x equal
        # 1, y equal 6); it raises RuntimeError for those combinations only.
        if x == 1 or y == 6:
            raise RuntimeError('Boom')


class TestTestParametrizationDeviceType(TestCase):
    # [CLASS DOC] Device-generic tests for parametrization names and decorators together with
    # device and dtype instantiation, using throwaway classes defined inside each test.
    def test_unparametrized_names(self, device):
        # [FUNCTION DOC] Protects naming of device and dtype tests without parametrization:
        # the generated class for the device has names including device and each dtype.
        # This test exists to protect against regressions in device / dtype test naming
        # due to parametrization logic.

        device = self.device_type

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class with a device test and a device-and-dtype
            # test.
            def test_device_specific(self, device):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

            @dtypes(torch.float32, torch.float64)
            def test_device_dtype_specific(self, device, dtype):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        locals_dict = dict(locals())
        instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

        device_cls = locals_dict[f'TestParametrized{device.upper()}']
        expected_test_names = [name.format(device_cls.__name__, device) for name in (
            '{}.test_device_dtype_specific_{}_float32',
            '{}.test_device_dtype_specific_{}_float64',
            '{}.test_device_specific_{}')
        ]
        test_names = _get_test_names_for_test_class(device_cls)
        self.assertEqual(expected_test_names, test_names)

    def test_name_fn_with_dot_raises(self, device):
        # [FUNCTION DOC] With device instantiation, a name function producing a dot raises
        # RuntimeError.
        device = self.device_type

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class whose parametrization generates a dotted
            # name.
            @parametrize("dtype", [torch.bfloat16], name_fn=str)
            def test_bad_name(self, device, dtype):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        locals_dict = dict(locals())
        with self.assertRaisesRegex(RuntimeError, 'contains a "." character'):
            instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

    def test_empty_param_names(self, device):
        # [FUNCTION DOC] Parametrize with an empty name string and empty or non-empty values
        # still yields one test per function with only the device in its name.
        # If no param names are passed, ensure things still work without parametrization.
        device = self.device_type

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class with two tests parametrized with empty
            # names.
            @parametrize("", [])
            def test_foo(self, device):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

            @parametrize("", range(5))
            def test_bar(self, device):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        locals_dict = dict(locals())
        instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

        device_cls = locals_dict[f'TestParametrized{device.upper()}']
        expected_test_names = [name.format(device_cls.__name__, device) for name in (
            '{}.test_bar_{}',
            '{}.test_foo_{}')
        ]
        test_names = _get_test_names_for_test_class(device_cls)
        self.assertEqual(expected_test_names, test_names)

    def test_empty_param_list(self, device):
        # [FUNCTION DOC] Reusing an exhausted generator for a second parametrization yields no
        # values; checks instantiation raises ValueError saying an empty arg_values was
        # passed.
        # If no param values are passed, ensure a helpful error message is thrown.
        # In the wild, this could indicate reuse of an exhausted generator.
        device = self.device_type

        generator = (a for a in range(5))

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class whose two tests share one generator of
            # values.
            @parametrize("x", generator)
            def test_foo(self, device, x):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

            # Reuse generator from first test function.
            @parametrize("y", generator)
            def test_bar(self, device, y):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        with self.assertRaisesRegex(ValueError, 'An empty arg_values was passed'):
            locals_dict = dict(locals())
            instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

    def test_default_names(self, device):
        # [FUNCTION DOC] Checks default parametrized names include the device suffix for
        # single and multi-param cases.
        device = self.device_type

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class with two parametrized device tests.
            @parametrize("x", range(5))
            def test_default_names(self, device, x):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

            @parametrize("x,y", [(1, 2), (2, 3), (3, 4)])
            def test_two_things_default_names(self, device, x, y):
                # [FUNCTION DOC] HELPER: No-op body.
                pass


        locals_dict = dict(locals())
        instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

        device_cls = locals_dict[f'TestParametrized{device.upper()}']
        expected_test_names = [name.format(device_cls.__name__, device) for name in (
            '{}.test_default_names_x_0_{}',
            '{}.test_default_names_x_1_{}',
            '{}.test_default_names_x_2_{}',
            '{}.test_default_names_x_3_{}',
            '{}.test_default_names_x_4_{}',
            '{}.test_two_things_default_names_x_1_y_2_{}',
            '{}.test_two_things_default_names_x_2_y_3_{}',
            '{}.test_two_things_default_names_x_3_y_4_{}')
        ]
        test_names = _get_test_names_for_test_class(device_cls)
        self.assertEqual(expected_test_names, test_names)

    def test_default_name_non_primitive(self, device):
        # [FUNCTION DOC] Checks default names when values are floats, strings or arbitrary
        # objects: floats and strings are sanitized, objects fall back to index-based names.
        device = self.device_type

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class using non-primitive parameter values.
            @parametrize("x", [1, .5, "foo", object()])
            def test_default_names(self, device, x):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

            @parametrize("x,y", [(1, object()), (object(), .5), (object(), object())])
            def test_two_things_default_names(self, device, x, y):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        locals_dict = dict(locals())
        instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

        device_cls = locals_dict[f'TestParametrized{device.upper()}']
        expected_test_names = sorted(name.format(device_cls.__name__, device) for name in (
            '{}.test_default_names_x_1_{}',
            '{}.test_default_names_x_0_5_{}',
            '{}.test_default_names_x_foo_{}',
            '{}.test_default_names_x3_{}',
            '{}.test_two_things_default_names_x_1_y0_{}',
            '{}.test_two_things_default_names_x1_y_0_5_{}',
            '{}.test_two_things_default_names_x2_y2_{}')
        )
        test_names = _get_test_names_for_test_class(device_cls)
        self.assertEqual(expected_test_names, test_names)

    def test_name_fn(self, device):
        # [FUNCTION DOC] Checks name functions combine with the device suffix, for one, three
        # stacked and multi-name parametrizations.
        device = self.device_type

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class with three parametrized device tests with
            # name functions.
            @parametrize("bias", [False, True], name_fn=lambda b: 'bias' if b else 'no_bias')
            def test_custom_names(self, device, bias):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

            @parametrize("x", [1, 2], name_fn=str)
            @parametrize("y", [3, 4], name_fn=str)
            @parametrize("z", [5, 6], name_fn=str)
            def test_three_things_composition_custom_names(self, device, x, y, z):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

            @parametrize("x,y", [(1, 2), (1, 3), (1, 4)], name_fn=lambda x, y: f'{x}__{y}')
            def test_two_things_custom_names_alternate(self, device, x, y):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        locals_dict = dict(locals())
        instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

        device_cls = locals_dict[f'TestParametrized{device.upper()}']
        expected_test_names = [name.format(device_cls.__name__, device) for name in (
            '{}.test_custom_names_bias_{}',
            '{}.test_custom_names_no_bias_{}',
            '{}.test_three_things_composition_custom_names_1_3_5_{}',
            '{}.test_three_things_composition_custom_names_1_3_6_{}',
            '{}.test_three_things_composition_custom_names_1_4_5_{}',
            '{}.test_three_things_composition_custom_names_1_4_6_{}',
            '{}.test_three_things_composition_custom_names_2_3_5_{}',
            '{}.test_three_things_composition_custom_names_2_3_6_{}',
            '{}.test_three_things_composition_custom_names_2_4_5_{}',
            '{}.test_three_things_composition_custom_names_2_4_6_{}',
            '{}.test_two_things_custom_names_alternate_1__2_{}',
            '{}.test_two_things_custom_names_alternate_1__3_{}',
            '{}.test_two_things_custom_names_alternate_1__4_{}')
        ]
        test_names = _get_test_names_for_test_class(device_cls)
        self.assertEqual(expected_test_names, test_names)

    def test_subtest_names(self, device):
        # [FUNCTION DOC] Checks subtest names combine with the device suffix.
        device = self.device_type

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class with named subtests on device tests.
            @parametrize("bias", [subtest(True, name='bias'),
                                  subtest(False, name='no_bias')])
            def test_custom_names(self, device, bias):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

            @parametrize("x,y", [subtest((1, 2), name='double'),
                                 subtest((1, 3), name='triple'),
                                 subtest((1, 4), name='quadruple')])
            def test_two_things_custom_names(self, device, x, y):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        locals_dict = dict(locals())
        instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

        device_cls = locals_dict[f'TestParametrized{device.upper()}']
        expected_test_names = [name.format(device_cls.__name__, device) for name in (
            '{}.test_custom_names_bias_{}',
            '{}.test_custom_names_no_bias_{}',
            '{}.test_two_things_custom_names_double_{}',
            '{}.test_two_things_custom_names_quadruple_{}',
            '{}.test_two_things_custom_names_triple_{}')
        ]
        test_names = _get_test_names_for_test_class(device_cls)
        self.assertEqual(expected_test_names, test_names)

    def test_ops_composition_names(self, device):
        # [FUNCTION DOC] Combines the ops decorator with a parametrize over a flag and checks
        # generated names for every op in op_db and each supported dtype for the device, with
        # flag name parts.
        device = self.device_type

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class with one op test parametrized by a flag.
            @ops(op_db)
            @parametrize("flag", [False, True], lambda f: 'flag_enabled' if f else 'flag_disabled')
            def test_op_parametrized(self, device, dtype, op, flag):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        locals_dict = dict(locals())
        instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

        device_cls = locals_dict[f'TestParametrized{device.upper()}']
        expected_test_names = []
        for op in op_db:
            for dtype in op.supported_dtypes(torch.device(device).type):
                for flag_part in ('flag_disabled', 'flag_enabled'):
                    expected_name = f'{device_cls.__name__}.test_op_parametrized_{op.formatted_name}_{flag_part}_{device}_{dtype_name(dtype)}'
                    expected_test_names.append(expected_name)

        test_names = _get_test_names_for_test_class(device_cls)
        self.assertEqual(sorted(expected_test_names), sorted(test_names))

    def test_modules_composition_names(self, device):
        # [FUNCTION DOC] Combines the modules decorator with a flag parametrize and checks
        # generated names for each module info, dtype, and train or eval mode when the module
        # differs between them.
        device = self.device_type

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class with one module test parametrized by a flag.
            @modules(module_db)
            @parametrize("flag", [False, True], lambda f: 'flag_enabled' if f else 'flag_disabled')
            def test_module_parametrized(self, device, dtype, module_info, training, flag):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        locals_dict = dict(locals())
        instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

        device_cls = locals_dict[f'TestParametrized{device.upper()}']
        expected_test_names = []
        for module_info in module_db:
            for dtype in module_info.dtypes:
                for flag_part in ('flag_disabled', 'flag_enabled'):
                    expected_train_modes = (
                        ['train_mode', 'eval_mode'] if module_info.train_and_eval_differ else [''])
                    for training_part in expected_train_modes:
                        expected_name = '{}.test_module_parametrized_{}{}_{}_{}_{}'.format(
                            device_cls.__name__, module_info.formatted_name,
                            '_' + training_part if len(training_part) > 0 else '',
                            flag_part, device, dtype_name(dtype))
                        expected_test_names.append(expected_name)

        test_names = _get_test_names_for_test_class(device_cls)
        self.assertEqual(sorted(expected_test_names), sorted(test_names))

    def test_ops_decorator_applies_op_and_param_specific_decorators(self, device):
        # [FUNCTION DOC] Builds a test OpInfo with a decorator active for float64 on cpu when
        # x is 2, plus decorateIf and subtest decorators, and checks the marker lands only on
        # the expected generated tests.
        # Test that decorators can be applied on a per-op / per-param basis.

        # Create a test op, OpInfo entry, and decorator to apply.
        def test_op(x):
            # [FUNCTION DOC] HELPER: Negates its input; the operator wrapped by the test
            # OpInfo.
            return -x

        def test_dec(func):
            # [FUNCTION DOC] HELPER: Decorator that sets a marker attribute on the function
            # and returns it.
            func._decorator_applied = True
            return func

        test_op_info = OpInfo(
            'test_op',
            op=test_op,
            dtypes=floating_types(),
            sample_inputs_func=lambda _: [],
            decorators=[
                DecorateInfo(test_dec, 'TestParametrized', 'test_op_param',
                             device_type='cpu', dtypes=[torch.float64],
                             active_if=lambda p: p['x'] == 2)
            ])

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class with three op tests exercising decorator
            # application.
            @ops(op_db + [test_op_info])
            @parametrize("x", [2, 3])
            def test_op_param(self, device, dtype, op, x):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

            @ops(op_db + [test_op_info])
            @parametrize("y", [
                subtest(4),
                subtest(5, decorators=[test_dec])])
            def test_other(self, device, dtype, op, y):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

            @decorateIf(test_dec, lambda p: p['dtype'] == torch.int16)
            @ops(op_db)
            def test_three(self, device, dtype, op):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        device = self.device_type
        locals_dict = dict(locals())
        instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)
        device_cls = locals_dict[f'TestParametrized{device.upper()}']

        for test_func, name in _get_test_funcs_for_test_class(device_cls):
            should_apply = (name == 'test_op_param_test_op_x_2_cpu_float64' or
                            ('test_other' in name and 'y_5' in name) or
                            ('test_three' in name and name.endswith('_int16')))
            self.assertEqual(hasattr(test_func, '_decorator_applied'), should_apply)

    def test_modules_decorator_applies_module_and_param_specific_decorators(self, device):
        # [FUNCTION DOC] Module analog of the op decorator test: defines a test module and
        # ModuleInfo with a conditional decorator and checks the marker lands only on expected
        # generated tests.
        # Test that decorators can be applied on a per-module / per-param basis.

        # Create a test module, ModuleInfo entry, and decorator to apply.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Tiny module with one random parameter vector added to its
            # input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the parameter holding three random values.
                super().__init__()
                self.x = torch.nn.Parameter(torch.randn(3))

            def forward(self, y):
                # [FUNCTION DOC] HELPER: Returns the parameter plus the input.
                return self.x + y

        def test_dec(func):
            # [FUNCTION DOC] HELPER: Decorator that sets a marker attribute on the function
            # and returns it.
            func._decorator_applied = True
            return func

        test_module_info = ModuleInfo(
            TestModule,
            module_inputs_func=lambda _: [],
            decorators=[
                DecorateInfo(test_dec, 'TestParametrized', 'test_module_param',
                             device_type='cpu', dtypes=[torch.float64],
                             active_if=lambda p: p['x'] == 2)
            ])

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class with three module tests exercising per-
            # module and per-param decorators.
            @modules(module_db + [test_module_info])
            @parametrize("x", [2, 3])
            def test_module_param(self, device, dtype, module_info, training, x):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

            @modules(module_db + [test_module_info])
            @parametrize("y", [
                subtest(4),
                subtest(5, decorators=[test_dec])])
            def test_other(self, device, dtype, module_info, training, y):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

            @decorateIf(test_dec, lambda p: p['dtype'] == torch.float64)
            @modules(module_db)
            def test_three(self, device, dtype, module_info):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        device = self.device_type
        locals_dict = dict(locals())
        instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)
        device_cls = locals_dict[f'TestParametrized{device.upper()}']

        for test_func, name in _get_test_funcs_for_test_class(device_cls):
            should_apply = (name == 'test_module_param_TestModule_x_2_cpu_float64' or
                            ('test_other' in name and 'y_5' in name) or
                            ('test_three' in name and name.endswith('float64')))
            self.assertEqual(hasattr(test_func, '_decorator_applied'), should_apply)

    def test_param_specific_decoration(self, device):
        # [FUNCTION DOC] Checks decorateIf with a predicate on parametrized args applies the
        # marker only to the generated test with x equal 1 and y true.

        def test_dec(func):
            # [FUNCTION DOC] HELPER: Decorator that sets a marker attribute on the function
            # and returns it.
            func._decorator_applied = True
            return func

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class with a conditionally decorated two-parameter
            # test.
            @decorateIf(test_dec, lambda params: params["x"] == 1 and params["y"])
            @parametrize("x", range(5))
            @parametrize("y", [False, True])
            def test_param(self, x, y):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        device = self.device_type
        locals_dict = dict(locals())
        instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)
        device_cls = locals_dict[f'TestParametrized{device.upper()}']

        for test_func, name in _get_test_funcs_for_test_class(device_cls):
            should_apply = ('test_param_x_1_y_True' in name)
            self.assertEqual(hasattr(test_func, '_decorator_applied'), should_apply)

    def test_dtypes_composition_valid(self, device):
        # [FUNCTION DOC] Checks dtypes and parametrize compose when parametrize does not set
        # dtype: one test per combination of x value and dtype, with names including device
        # and dtype.
        # Test checks that @parametrize and @dtypes compose as expected when @parametrize
        # doesn't set dtype.

        device = self.device_type

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class combining dtypes with a parametrize over x.
            @dtypes(torch.float32, torch.float64)
            @parametrize("x", range(3))
            def test_parametrized(self, x, dtype):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        locals_dict = dict(locals())
        instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

        device_cls = locals_dict[f'TestParametrized{device.upper()}']
        expected_test_names = [name.format(device_cls.__name__, device) for name in (
            '{}.test_parametrized_x_0_{}_float32',
            '{}.test_parametrized_x_0_{}_float64',
            '{}.test_parametrized_x_1_{}_float32',
            '{}.test_parametrized_x_1_{}_float64',
            '{}.test_parametrized_x_2_{}_float32',
            '{}.test_parametrized_x_2_{}_float64')
        ]
        test_names = _get_test_names_for_test_class(device_cls)
        self.assertEqual(sorted(expected_test_names), sorted(test_names))

    def test_dtypes_composition_invalid(self, device):
        # [FUNCTION DOC] Checks dtypes cannot be combined with a parametrize over dtype, nor
        # with ops, both raising RuntimeError saying dtype is handled multiple times.
        # Test checks that @dtypes cannot be composed with parametrization decorators when they
        # also try to set dtype.

        device = self.device_type

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class combining dtypes with a dtype parametrize.
            @dtypes(torch.float32, torch.float64)
            @parametrize("dtype", [torch.int32, torch.int64])
            def test_parametrized(self, dtype):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        with self.assertRaisesRegex(RuntimeError, "handled multiple times"):
            locals_dict = dict(locals())
            instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

        # Verify proper error behavior with @ops + @dtypes, as both try to set dtype.

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class combining dtypes with ops.
            @dtypes(torch.float32, torch.float64)
            @ops(op_db)
            def test_parametrized(self, op, dtype):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        with self.assertRaisesRegex(RuntimeError, "handled multiple times"):
            locals_dict = dict(locals())
            instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

    def test_multiple_handling_of_same_param_error(self, device):
        # [FUNCTION DOC] Combining ops and modules (both set dtype) raises RuntimeError saying
        # a parameter is handled multiple times.
        # Test that multiple decorators handling the same param errors out.
        # Both @modules and @ops handle the dtype param.

        class TestParametrized(TestCase):
            # [CLASS DOC] HELPER: Throwaway class combining ops and modules decorators.
            @ops(op_db)
            @modules(module_db)
            def test_param(self, device, dtype, op, module_info, training):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        with self.assertRaisesRegex(RuntimeError, "handled multiple times"):
            locals_dict = dict(locals())
            instantiate_device_type_tests(TestParametrized, locals_dict, only_for=device)

    @parametrize("x", [1, subtest(2, decorators=[unittest.expectedFailure]), 3])
    def test_subtest_expected_failure(self, device, x):
        # [FUNCTION DOC] Device-parametrized test with subtest x equal 2 marked expected
        # failure; raises RuntimeError for that value only.
        if x == 2:
            raise RuntimeError('Boom')

    @parametrize("x", [subtest(1, decorators=[unittest.expectedFailure]), 2, 3])
    @parametrize("y", [4, 5, subtest(6, decorators=[unittest.expectedFailure])])
    def test_two_things_subtest_expected_failure(self, device, x, y):
        # [FUNCTION DOC] Two stacked parametrizations with expected-failure subtests (x equal
        # 1, y equal 6); raises RuntimeError for those values only.
        if x == 1 or y == 6:
            raise RuntimeError('Boom')


instantiate_parametrized_tests(TestTestParametrization)
instantiate_device_type_tests(TestTestParametrizationDeviceType, globals())


class TestImports(TestCase):
    # [CLASS DOC] Tests about importing torch: circular import freedom, laziness, no warnings
    # or global logging changes, and no sympy or triton import. Several tests start fresh
    # Python subprocesses.
    @classmethod
    def _check_python_output(cls, program) -> str:
        # [FUNCTION DOC] HELPER: Runs the given program text in a fresh Python interpreter
        # with all warnings shown, merging stderr into stdout, using the directory of this
        # file as working directory. Returns the decoded output; raises CalledProcessError on
        # a nonzero exit.
        return subprocess.check_output(
            [sys.executable, "-W", "always", "-c", program],
            stderr=subprocess.STDOUT,
            # On Windows, opening the subprocess with the default CWD makes `import torch`
            # fail, so just set CWD to this script's directory
            cwd=os.path.dirname(os.path.realpath(__file__)),).decode("utf-8")

    @skipIfXpu(msg="The test is flaky on XPU, see https://github.com/pytorch/pytorch/issues/110040")
    # The test is flaky on ROCm/XPU and has been open and close multiple times
    # https://github.com/pytorch/pytorch/issues/110040
    def test_circular_dependencies(self) -> None:
        # [FUNCTION DOC] Walks the torch package directory and imports every Python module
        # except a list of ignored prefixes (optional dependencies, distributed on some
        # platforms, executables named main), raising RuntimeError naming the module on any
        # import failure, and asserts each result is a module.
        """ Checks that all modules inside torch can be imported
        Prevents regression reported in https://github.com/pytorch/pytorch/issues/77441 """
        ignored_modules = ["torch.utils.tensorboard",  # deps on tensorboard
                           "torch.distributed.elastic.rendezvous",  # depps on etcd
                           "torch.backends._coreml",  # depends on pycoreml
                           "torch.contrib.",  # something weird
                           "torch.testing._internal.distributed.",  # just fails
                           "torch.ao.pruning._experimental.",  # depends on pytorch_lightning, not user-facing
                           "torch.onnx._internal",  # depends on onnx-script
                           "torch._inductor.runtime.triton_helpers",  # depends on triton
                           "torch._native.ops.bmm_outer_product.triton_kernels",  # depends on triton
                           "torch._native.ops.foreach_mm",  # depends on nvmath-python, cuda-python
                           "torch._native.ops.polar.nvmath_impl",  # depends on nvmath-python, cuda-python
                           "torch._native.ops.scatter_add",  # depends on cutlass
                           "torch._native.ops.sum.inner_tree_kernel",  # depends on cutlass
                           "torch._native.ops.topk",  # depends on cutlass
                           "torch._inductor.codegen.cuda",  # depends on cutlass
                           "torch._inductor.codegen.cutedsl",  # depends on cutlass
                           "torch.distributed.benchmarks",  # depends on RPC and DDP Optim
                           "torch.distributed.debug._frontend",  # depends on tabulate
                           "torch.distributed.examples",  # requires CUDA and torchvision
                           "torch.distributed.tensor.examples",  # example scripts
                           "torch.distributed._tools.sac_ilp",  # depends on pulp
                           "torch.csrc",  # files here are devtools, not part of torch
                           "torch.include",  # torch include files after install
                           "torch._inductor.kernel.vendored_templates.cutedsl",  # depends on cutlass
                           "torch._vendor.quack",  # depends on cutlass / cuda-python
                           "torch.profiler._cupti",  # depends on cupti-python
                           ]
        if IS_WINDOWS or IS_MACOS or IS_JETSON:
            # Distributed should be importable on Windows(except nn.api.), but not on Mac
            if IS_MACOS or IS_JETSON:
                ignored_modules.append("torch.distributed.")
            else:
                ignored_modules.append("torch.distributed.nn.api.")
                ignored_modules.append("torch.distributed.optim.")
                ignored_modules.append("torch.distributed.rpc.")
            ignored_modules.append("torch.testing._internal.dist_utils")
            # And these both end up with transitive dependencies on distributed
            ignored_modules.append("torch.nn.parallel._replicated_tensor_ddp_interop")
            ignored_modules.append("torch.testing._internal.common_fsdp")
            ignored_modules.append("torch.testing._internal.common_distributed")

        if sys.version_info < (3, 12):
            # depends on Python 3.12+ syntax
            ignored_modules.append("torch.testing._internal.py312_intrinsics")

        torch_dir = os.path.dirname(torch.__file__)
        for base, _, files in os.walk(torch_dir):
            prefix = os.path.relpath(base, os.path.dirname(torch_dir)).replace(os.path.sep, ".")
            for f in files:
                if not f.endswith(".py"):
                    continue
                mod_name = f"{prefix}.{f[:-3]}" if f != "__init__.py" else prefix
                # Do not attempt to import executable modules
                if f == "__main__.py":
                    continue
                if any(mod_name.startswith(x) for x in ignored_modules):
                    continue
                try:
                    mod = importlib.import_module(mod_name)
                except Exception as e:
                    raise RuntimeError(f"Failed to import {mod_name}: {e}") from e
                self.assertTrue(inspect.ismodule(mod))

    def test_lazy_imports_are_lazy(self) -> None:
        # [FUNCTION DOC] In a subprocess checks that none of the torch lazy modules is loaded
        # after importing torch; expects True printed.
        out = self._check_python_output("import sys;import torch;print(all(x not in sys.modules for x in torch._lazy_modules))")
        self.assertEqual(out.strip(), "True")

    def test_no_warning_on_import(self) -> None:
        # [FUNCTION DOC] Importing torch in a subprocess with all warnings enabled must print
        # nothing.
        out = self._check_python_output("import torch")
        self.assertEqual(out, "")

    def test_not_import_sympy(self) -> None:
        # [FUNCTION DOC] In a subprocess checks that sympy is not loaded by importing torch;
        # the failure message explains how to profile imports.
        out = self._check_python_output("import torch;import sys;print('sympy' not in sys.modules)")
        self.assertEqual(out.strip(), "True",
                         "PyTorch should not depend on SymPy at import time as importing SymPy is *very* slow.\n"
                         "See the beginning of the following blog post for how to profile and find which file is importing sympy:\n"
                         "https://dev-discuss.pytorch.org/t/delving-into-what-happens-when-you-import-torch/1589\n\n"
                         "If you hit this error, you may want to:\n"
                         "  - Refactor your code to avoid depending on sympy files you may not need to depend\n"
                         "  - Use TYPE_CHECKING if you are using sympy + strings if you are using sympy on type annotations\n"
                         "  - Import things that depend on SymPy locally")

    def test_not_import_triton(self) -> None:
        # [FUNCTION DOC] In a subprocess checks that triton is not loaded by importing torch.
        out = self._check_python_output("import torch;import sys;print('triton' not in sys.modules)")
        self.assertEqual(out.strip(), "True")

    @parametrize('path', ['torch', 'functorch'])
    def test_no_mutate_global_logging_on_import(self, path) -> None:
        # [FUNCTION DOC] For torch and functorch, runs a subprocess that imports it, attaches
        # a stream handler to the root logger at info level, logs a message, and checks that
        # exactly that message is printed, i.e. import did not configure global logging.
        # Calling logging.basicConfig, among other things, modifies the global
        # logging state. It is not OK to modify the global logging state on
        # `import torch` (or other submodules we own) because users do not expect it.
        expected = string.ascii_lowercase
        commands = [
            'import logging',
            f'import {path}',
            '_logger = logging.getLogger("torch_test_testing")',
            'logging.root.addHandler(logging.StreamHandler())',
            'logging.root.setLevel(logging.INFO)',
            f'_logger.info("{expected}")'
        ]
        out = self._check_python_output("; ".join(commands))
        self.assertEqual(out.strip(), expected)

class TestOpInfos(TestCase):
    # [CLASS DOC] Tests for SampleInput and related OpInfo behavior. Stateless.
    def test_sample_input(self) -> None:
        # [FUNCTION DOC] Checks SampleInput construction: natural form (input, positional
        # args, keywords) and explicit args and kwargs form give the same fields; mixing the
        # two forms, or giving metadata with extra args, raises AssertionError; metadata alone
        # with only the input is allowed.
        a, b, c, d, e = (object() for _ in range(5))

        # Construction with natural syntax
        s = SampleInput(a, b, c, d=d, e=e)
        if s.input is not a:
            raise AssertionError("s.input should be a")
        if s.args != (b, c):
            raise AssertionError(f"s.args should be (b, c), got {s.args}")
        if s.kwargs != dict(d=d, e=e):
            raise AssertionError(f"s.kwargs mismatch: got {s.kwargs}")

        # Construction with explicit args and kwargs
        s = SampleInput(a, args=(b,), kwargs=dict(c=c, d=d, e=e))
        if s.input is not a:
            raise AssertionError("s.input should be a")
        if s.args != (b,):
            raise AssertionError(f"s.args should be (b,), got {s.args}")
        if s.kwargs != dict(c=c, d=d, e=e):
            raise AssertionError(f"s.kwargs mismatch: got {s.kwargs}")

        # Construction with a mixed form will error
        with self.assertRaises(AssertionError):
            s = SampleInput(a, b, c, args=(d, e))

        with self.assertRaises(AssertionError):
            s = SampleInput(a, b, c, kwargs=dict(d=d, e=e))

        with self.assertRaises(AssertionError):
            s = SampleInput(a, args=(b, c), d=d, e=e)

        with self.assertRaises(AssertionError):
            s = SampleInput(a, b, c=c, kwargs=dict(d=d, e=e))

        # Mixing metadata into "natural" construction will error
        with self.assertRaises(AssertionError):
            s = SampleInput(a, b, name="foo")

        with self.assertRaises(AssertionError):
            s = SampleInput(a, b, output_process_fn_grad=lambda x: x)

        with self.assertRaises(AssertionError):
            s = SampleInput(a, b, broadcasts_input=True)

        # But when only input is given, metadata is allowed for backward
        # compatibility
        s = SampleInput(a, broadcasts_input=True)
        if s.input is not a:
            raise AssertionError("s.input should be a")
        if not s.broadcasts_input:
            raise AssertionError("s.broadcasts_input should be True")

    def test_sample_input_metadata(self) -> None:
        # [FUNCTION DOC] Checks default metadata (identity output function, no broadcast,
        # empty name) and that with_metadata mutates and returns the same object with new
        # values.
        a, b = (object() for _ in range(2))
        s1 = SampleInput(a, b=b)
        self.assertIs(s1.output_process_fn_grad(None), None)
        self.assertFalse(s1.broadcasts_input)
        self.assertEqual(s1.name, "")

        s2 = s1.with_metadata(
            output_process_fn_grad=lambda x: a,
            broadcasts_input=True,
            name="foo",
        )
        self.assertIs(s1, s2)
        self.assertIs(s2.output_process_fn_grad(None), a)
        self.assertTrue(s2.broadcasts_input)
        self.assertEqual(s2.name, "foo")


# Tests that validate the various sample generating functions on each OpInfo.
class TestOpInfoSampleFunctions(TestCase):
    # [CLASS DOC] Device-generic tests that OpInfo sample, reference and error input
    # generators return iterators rather than eager collections.

    @ops(op_db, dtypes=OpDTypes.any_one)
    def test_opinfo_sample_generators(self, device, dtype, op):
        # [FUNCTION DOC] For one dtype of every op, sample_inputs returns an Iterator.
        # Test op.sample_inputs doesn't generate multiple samples when called
        samples = op.sample_inputs(device, dtype)
        self.assertIsInstance(samples, Iterator)

    @ops([op for op in op_db if op.reference_inputs_func is not None], dtypes=OpDTypes.any_one)
    def test_opinfo_reference_generators(self, device, dtype, op):
        # [FUNCTION DOC] For ops that define reference inputs, reference_inputs returns an
        # Iterator.
        # Test op.reference_inputs doesn't generate multiple samples when called
        samples = op.reference_inputs(device, dtype)
        self.assertIsInstance(samples, Iterator)

    @ops([op for op in op_db if op.error_inputs_func is not None], dtypes=OpDTypes.none)
    def test_opinfo_error_generators(self, device, op):
        # [FUNCTION DOC] For ops that define error inputs, error_inputs returns an Iterator.
        # Test op.error_inputs doesn't generate multiple inputs when called
        samples = op.error_inputs(device)
        self.assertIsInstance(samples, Iterator)

# Tests test classification.
class TestHardwareClassifications(TestCase):
    # [CLASS DOC] Tests HardwareClassificationTestLoader from common_utils, which filters test
    # cases by a class-level hardware classification. The module is stored in the attribute
    # _cu by setUp.
    def setUp(self):
        # [FUNCTION DOC] Calls the parent setUp and stores the common_utils module in _cu.
        super().setUp()
        import torch.testing._internal.common_utils as _cu

        self._cu = _cu

    def _suite_test_names(self, suite) -> set[str]:
        # [FUNCTION DOC] HELPER: Returns the set of last name components of the ids of all
        # test cases found by recursively iterating the suite.
        return {test_case.id().split(".")[-1] for test_case in self._cu.HardwareClassificationTestLoader.iter_test_cases_recursively(suite)}

    def test_filter_suite(self):
        # [FUNCTION DOC] Builds a suite from generic, CUDA and unclassified test classes and
        # checks a loader for the generic classification keeps only the generic test.
        requirement = self._cu.HardwareClassification

        class GenericTest(TestCase):
            # [CLASS DOC] HELPER: Throwaway test class classified as generic.
            hw_classification = requirement.GENERIC

            def test_generic(self):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        class CudaTest(TestCase):
            # [CLASS DOC] HELPER: Throwaway test class classified as CUDA.
            hw_classification = requirement.CUDA

            def test_cuda(self):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        class MissingClassificationTest(TestCase):
            # [CLASS DOC] HELPER: Throwaway test class with no classification.
            def test_missing_classification(self):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        suite = unittest.TestSuite(
            [
                unittest.defaultTestLoader.loadTestsFromTestCase(GenericTest),
                unittest.defaultTestLoader.loadTestsFromTestCase(CudaTest),
                unittest.defaultTestLoader.loadTestsFromTestCase(MissingClassificationTest),
            ]
        )

        loader = self._cu.HardwareClassificationTestLoader({self._cu.HardwareClassification.GENERIC})
        filtered_suite = loader.get_filtered_suite(suite)

        self.assertEqual(
            self._suite_test_names(filtered_suite),
            {"test_generic"},
        )

    def test_filter_suite_uses_inherited_metadata(self):
        # [FUNCTION DOC] Checks classification set on a base class is inherited by a subclass
        # so the subclass test passes an accelerator filter.
        requirement = self._cu.HardwareClassification

        class AcceleratorBase(TestCase):
            # [CLASS DOC] HELPER: Throwaway base class classified as accelerator.
            hw_classification = requirement.ACCELERATOR

        class AcceleratorChild(AcceleratorBase):
            # [CLASS DOC] HELPER: Throwaway subclass inheriting the accelerator
            # classification.
            def test_inherited_classification(self):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(AcceleratorChild)
        loader = self._cu.HardwareClassificationTestLoader({self._cu.HardwareClassification.ACCELERATOR})
        filtered_suite = loader.get_filtered_suite(suite)

        self.assertEqual(
            self._suite_test_names(filtered_suite),
            {"test_inherited_classification"},
        )

    def test_hw_classification_test_loader(self):
        # [FUNCTION DOC] Builds a fake module with two generic classes and a CUDA class and
        # checks a generic loader keeps only generic tests for module, multiple names and
        # single name loading (the last yields no tests because the named test is CUDA).
        requirement = self._cu.HardwareClassification
        import types

        # Build a mock module with test classes
        class GenericA(TestCase):
            # [CLASS DOC] HELPER: Throwaway generic test class with two tests.
            hw_classification = requirement.GENERIC

            def test_a1(self):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

            def test_a2(self):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        class GenericB(TestCase):
            # [CLASS DOC] HELPER: Throwaway generic test class with one test.
            hw_classification = requirement.GENERIC

            def test_b1(self):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        class Cuda(TestCase):
            # [CLASS DOC] HELPER: Throwaway CUDA-classified test class with one test.
            hw_classification = requirement.CUDA

            def test_c1(self):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        mod = types.ModuleType("_test_hc_module")
        mod.GenericA = GenericA
        mod.GenericB = GenericB
        mod.Cuda = Cuda

        loader = self._cu.HardwareClassificationTestLoader({requirement.GENERIC})

        with self.subTest(method="loadTestsFromModule"):
            suite = loader.loadTestsFromModule(mod)
            self.assertEqual(
                self._suite_test_names(suite),
                {"test_a1", "test_a2", "test_b1"},
            )

        with self.subTest(method="loadTestsFromNames"):
            suite = loader.loadTestsFromNames(
                ["GenericA.test_a1", "Cuda.test_c1"],
                module=mod,
            )
            self.assertEqual(self._suite_test_names(suite), {"test_a1"})

        with self.subTest(method="loadTestsFromName"):
            suite = loader.loadTestsFromName(
                "Cuda.test_c1",
                module=mod,
            )
            self.assertEqual(self._suite_test_names(suite), set())

    def test_hw_classification_test_loader_no_filter(self):
        # [FUNCTION DOC] With no filter set given to the loader (none), loading a fake module
        # returns tests from both generic and CUDA classes.
        import types

        class GenericA(TestCase):
            # [CLASS DOC] HELPER: Throwaway generic test class with one test.
            hw_classification = self._cu.HardwareClassification.GENERIC

            def test_a(self):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        class Cuda(TestCase):
            # [CLASS DOC] HELPER: Throwaway CUDA-classified test class with one test.
            hw_classification = self._cu.HardwareClassification.CUDA

            def test_b(self):
                # [FUNCTION DOC] HELPER: No-op body.
                pass

        mod = types.ModuleType("_test_hc_module_none")
        mod.GenericA = GenericA
        mod.Cuda = Cuda

        loader = self._cu.HardwareClassificationTestLoader(None)
        suite = loader.loadTestsFromModule(mod)

        self.assertEqual(self._suite_test_names(suite), {"test_a", "test_b"})


instantiate_device_type_tests(TestOpInfoSampleFunctions, globals())
instantiate_parametrized_tests(TestImports)


if __name__ == '__main__':
    run_tests()
