# Owner(s): ["module: PrivateUse1"]

# [MODULE DOC] Tests for the PrivateUse1 device-type test infrastructure as used by the
# openreg backend: bypassing onlyCUDA/onlyOn restrictions, OpInfo op_overrides and
# op_allowlist, per-test and dtype-aware exclusions, and the distributed backend hook.
# At import time it temporarily sets PrivateUse1TestBase test configs and instantiates the
# device-generic test classes for the openreg device only; each class verifies in
# tearDownClass that the expected variants did or did not run.

import unittest
from collections import defaultdict
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCUDA,
    onlyOn,
    ops,
    precisionOverride,
    PrivateUse1TestBase,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.opinfo.core import DecorateInfo, OpInfo


class TestBypassDeviceRestrictions(TestCase):
    # [CLASS DOC] Checks that a backend opting in through bypass_device_restrictions can run
    # tests gated by onlyCUDA and onlyOn. State: executed_count, a class-level counter of
    # executed test bodies, verified to equal two in tearDownClass.
    """Verify that PrivateUse1 backends can run tests decorated with @onlyCUDA.

    PrivateUse1TestBase sets bypass_device_restrictions = False by default.
    Backends that are ready to run @onlyOn-gated tests must explicitly opt in
    by setting bypass_device_restrictions = True in their setUp or subclass.
    This allows out-of-tree backends to run accelerator tests that are
    currently gated behind @onlyCUDA while the long-term migration to
    device-generic tests is in progress.
    """

    executed_count = 0

    def setUp(self):
        # [FUNCTION DOC] Sets the instance flag that opts in to bypassing device restrictions
        # (shadowing the base class default of False), then runs the base setUp.
        # Explicitly opt-in: instance attribute shadows the False default on
        # PrivateUse1TestBase so @onlyOn-based decorators allow this test to run.
        self.bypass_device_restrictions = True
        super().setUp()

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Raises AssertionError unless exactly two test bodies executed (the
        # onlyCUDA and onlyOn tests), then calls the base tearDownClass.
        expected_runs = 2
        actual_runs = cls.executed_count
        if actual_runs != expected_runs:
            raise AssertionError(
                f"Bypass logic failed! "
                f"Expected {expected_runs} tests to run, "
                f"but only {actual_runs} executed."
            )
        super().tearDownClass()

    @onlyCUDA
    def test_bypass_only_cuda(self, device):
        # [FUNCTION DOC] Gated by onlyCUDA but must run on openreg; increments the executed
        # counter and checks the device type is openreg.
        type(self).executed_count += 1
        self.assertEqual(torch.device(device).type, "openreg")

    @onlyOn(["cuda"])
    def test_bypass_only_on(self, device):
        # [FUNCTION DOC] Gated by onlyOn for cuda but must run on openreg; increments the
        # executed counter and checks the device type is openreg.
        type(self).executed_count += 1
        self.assertEqual(torch.device(device).type, "openreg")


def _make_dummy_op(name, **kwargs):
    # [FUNCTION DOC] HELPER: Builds an OpInfo with the given name, an identity op, float32 and
    # float64 dtypes and a sample generator that yields no samples; extra keyword arguments
    # (for example decorators) are passed to OpInfo.
    return OpInfo(
        name,
        op=lambda x: x,
        dtypes={torch.float32, torch.float64},
        sample_inputs_func=lambda op, device, dtype, requires_grad, **kw: [],
        **kwargs,
    )


op_normal = _make_dummy_op("op_normal")
op_skip = _make_dummy_op("op_skip")
op_skip_f32 = _make_dummy_op("op_skip_f32")
op_xfail = _make_dummy_op("op_xfail")
# op_precision starts with a low precision override (1e-5) declared in the OpInfo itself.
# op_overrides will later register a higher override (1e-2) for float32 to verify
# that the op_overrides entry takes precedence (because it is appended last to
# op.decorators and therefore applied last, overwriting precision_overrides).
op_precision = _make_dummy_op(
    "op_precision",
    decorators=[DecorateInfo(precisionOverride({torch.float32: 1e-5}))],
)

# Dummy ops for testing combined op_allowlist + op_overrides
op_combined_supported = _make_dummy_op("op_combined_supported")
op_combined_skip = _make_dummy_op("op_combined_skip")
op_combined_unsupported = _make_dummy_op("op_combined_unsupported")


@contextmanager
def _temp_test_configs(obj, **configs):
    # [FUNCTION DOC] HELPER: Context manager that applies the given test configs to obj via
    # its set_test_configs method and restores the previous values (None if unset) on exit,
    # even if the body raises. Side effect: mutates obj's configuration temporarily.
    backup = {k: getattr(obj, k, None) for k in configs}
    obj.set_test_configs(**configs)
    try:
        yield
    finally:
        obj.set_test_configs(**backup)


class TestDeviceTypeOpenReg(TestCase):
    # [CLASS DOC] Exercises op_overrides (skip, expected failure, dtype-specific skip,
    # precision override) on dummy OpInfos. State: _executed, a class-level mapping from op
    # name to the number of test bodies that ran, checked in tearDownClass.
    # Track which test variants actually executed so we can assert on them in
    # tearDownClass and catch silent regressions (e.g. accidentally skipping
    # all op variants or executing too few/many).
    _executed: dict = defaultdict(int)

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Raises AssertionError unless op_normal and op_precision ran, op_skip
        # and op_xfail bodies never ran, and op_skip_f32 ran exactly once (float64 only); then
        # calls the base tearDownClass.
        # KNOWN ISSUE: the op_skip_f32 failure message lacks a space between its two parts.
        should_run = ("op_normal", "op_precision")
        should_not_run = ("op_skip", "op_xfail")
        for op_name in should_run:
            if op_name not in cls._executed or cls._executed[op_name] == 0:
                raise AssertionError(f"{op_name} tests should have run")
        for op_name in should_not_run:
            if op_name in cls._executed and cls._executed[op_name] != 0:
                raise AssertionError(f"{op_name} test body should never run")
        if cls._executed["op_skip_f32"] != 1:
            raise AssertionError(
                "op_skip_f32 should have run exactly once,"
                f"but ran {cls._executed['op_skip_f32']} times"
            )
        super().tearDownClass()

    @ops([op_normal, op_skip, op_skip_f32, op_xfail, op_precision])
    def test_op(self, device, dtype, op):
        # [FUNCTION DOC] Op-parametrized body. For op_skip and op_xfail it fails deliberately
        # (they must be skipped or expected-fail by overrides). For op_precision with float32
        # it checks the effective precision is the op_overrides value, not the OpInfo one.
        # Otherwise it increments the per-op execution counter.
        if op.name in ("op_skip", "op_xfail"):
            self.fail(f"{op.name} deliberately fails")
        if op.name == "op_precision" and dtype == torch.float32:
            self.assertEqual(
                self.precision,
                1e-2,
                msg=(
                    lambda msg: f"{msg}\nExpected op_overrides precisionOverride (1e-2) to win over "
                    f"OpInfo precisionOverride (1e-5), but got {self.precision}"
                ),
            )
        type(self)._executed[op.name] += 1

    @ops([op_normal])
    def test_op_narrow_ops(self, device, dtype, op):
        # [FUNCTION DOC] Runs with only op_normal exposed to confirm that extra op_overrides
        # keys naming ops absent from this list do not raise a KeyError. Body is empty.
        """Verify that having extra entries in op_overrides that are NOT present in
        the @ops list does not raise a KeyError (safety check in _apply_op_overrides).

        op_overrides includes op_skip and PrivateUse1TestBase.op_overrides also
        lists op_skip; here @ops only exposes op_normal.  The safety check
        introduced in _apply_op_overrides must silently ignore the missing keys.
        """
        # If we reach here without a KeyError the safety check worked.


class TestSkippedSpecificTestCases(TestCase):
    # [CLASS DOC] Verifies per-test exclusion: one test is excluded for openreg and one runs.
    # State: executed_count, a class-level counter that must equal one in tearDownClass.
    executed_count = 0

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Raises AssertionError unless exactly one test body executed, then
        # calls the base tearDownClass.
        expected_runs = 1
        if cls.executed_count != expected_runs:
            raise AssertionError(
                f"Skip logic failed! Expected {expected_runs} tests to run, "
                f"but {cls.executed_count} executed."
            )
        super().tearDownClass()

    def test_runs(self, device):
        # [FUNCTION DOC] Non-excluded test: increments the counter and checks the device type
        # is openreg.
        type(self).executed_count += 1
        self.assertEqual(torch.device(device).type, "openreg")

    def test_skipped(self, device):
        # [FUNCTION DOC] Excluded for openreg via test_exclusions, so it should never be
        # instantiated; if it runs it increments the counter and then fails deliberately.
        type(self).executed_count += 1
        self.assertEqual(torch.device(device).type, "openreg")
        self.fail("This test should not be instantiated for openreg")


class TestSkippedWholeTestClass(TestCase):
    # [CLASS DOC] Verifies whole-class exclusion: all members are excluded for openreg, so no
    # test should be instantiated.
    def test_skipped_class_member(self, device):
        # [FUNCTION DOC] Must never run for openreg; fails deliberately if it does.
        self.fail("This class should not be instantiated for openreg")


class TestDtypeAwareExclusions(TestCase):
    # [CLASS DOC] Verifies dtype-specific exclusions of dtype-parametrized tests. State:
    # executed_dtypes, a class-level mapping from test name to the list of dtypes (or dtype
    # tuples) that actually ran, compared against the expected mapping in tearDownClass.
    executed_dtypes: dict = defaultdict(list)

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Raises AssertionError unless the recorded dtypes equal the expected
        # mapping (float64 only for the single dtype test, one float64/int64 tuple for the
        # tuple test, nothing for the fully excluded test), then calls the base tearDownClass.
        expected = {
            "test_dtype_filter": [torch.float64],
            "test_dtype_tuple_filter": [(torch.float64, torch.int64)],
        }
        if cls.executed_dtypes != expected:
            raise AssertionError(
                f"Dtype-aware exclusions failed: expected {expected}, "
                f"got {cls.executed_dtypes}"
            )
        super().tearDownClass()

    @dtypes(torch.float32, torch.float64)
    def test_dtype_filter(self, device, dtype):
        # [FUNCTION DOC] Float32 is excluded for openreg, so only float64 runs; records the
        # dtype and checks the device type is openreg.
        self.assertEqual(torch.device(device).type, "openreg")
        type(self).executed_dtypes["test_dtype_filter"].append(dtype)

    @dtypes(torch.float32, torch.float64)
    def test_all_dtypes_excluded(self, device, dtype):
        # [FUNCTION DOC] Every dtype variant is excluded for openreg, so this must never run;
        # fails deliberately if it does.
        self.assertEqual(torch.device(device).type, "openreg")
        self.fail("This method should not have any dtype variants for openreg")

    @dtypes((torch.float64, torch.int32), (torch.float64, torch.int64))
    def test_dtype_tuple_filter(self, device, dtypes):
        # [FUNCTION DOC] Tuple-dtype variants where int32 is excluded, so only the
        # float64/int64 pair runs; records the tuple and checks the device type is openreg.
        self.assertEqual(torch.device(device).type, "openreg")
        type(self).executed_dtypes["test_dtype_tuple_filter"].append(dtypes)


class TestSupportedOpsWithOverrides(TestCase):
    # [CLASS DOC] Verifies op_allowlist filtering combined with op_overrides. State:
    # _executed_combined, a class-level mapping from op name to execution count; the
    # supported op must run, the overridden-skip op and the non-allowlisted op must not.
    """Verify that op_allowlist filtering works together with op_overrides.

    op_allowlist filters which ops generate variants.
    op_overrides adds decorators to those ops that pass the filter.
    """

    _executed_combined: dict = defaultdict(int)

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Raises AssertionError unless the supported op ran and neither the
        # skipped op nor the non-allowlisted op ran, then calls the base tearDownClass.
        # op_combined_supported should run (it's in op_allowlist, no skip decorator)
        if cls._executed_combined["op_combined_supported"] == 0:
            raise AssertionError("op_combined_supported should have run")
        # op_combined_skip should NOT run (in op_allowlist but has skip decorator)
        if cls._executed_combined["op_combined_skip"] != 0:
            raise AssertionError("op_combined_skip should be skipped by op_overrides")
        # op_combined_unsupported should NOT run (not in op_allowlist)
        if cls._executed_combined["op_combined_unsupported"] != 0:
            raise AssertionError(
                "op_combined_unsupported should not run (not in op_allowlist)"
            )
        super().tearDownClass()

    @ops([op_combined_supported, op_combined_skip, op_combined_unsupported])
    def test_combined_filter(self, device, dtype, op):
        # [FUNCTION DOC] Increments the execution counter for the op being tested.
        type(self)._executed_combined[op.name] += 1


OPENREG_OP_OVERRIDES = {
    "op_skip": [DecorateInfo(unittest.skip("skip op_skip"))],
    "op_skip_f32": [
        DecorateInfo(unittest.skip("skip op_skip"), dtypes=(torch.float32,))
    ],
    "op_xfail": [DecorateInfo(unittest.expectedFailure)],
    # This overrides the 1e-5 precision already declared on op_precision's OpInfo.
    "op_precision": [DecorateInfo(precisionOverride({torch.float32: 1e-2}))],
}
with _temp_test_configs(PrivateUse1TestBase, op_overrides=OPENREG_OP_OVERRIDES):
    instantiate_device_type_tests(
        TestDeviceTypeOpenReg, globals(), only_for=("openreg",)
    )

instantiate_device_type_tests(
    TestBypassDeviceRestrictions, globals(), only_for="openreg"
)

OPENREG_TEST_EXCLUSIONS = {
    "TestSkippedSpecificTestCases": ["test_skipped"],
    "TestSkippedWholeTestClass": "*",
}
with _temp_test_configs(PrivateUse1TestBase, test_exclusions=OPENREG_TEST_EXCLUSIONS):
    instantiate_device_type_tests(
        TestSkippedSpecificTestCases, globals(), only_for="openreg"
    )
    instantiate_device_type_tests(
        TestSkippedWholeTestClass, globals(), only_for="openreg"
    )

OPENREG_DTYPE_TEST_EXCLUSIONS = {
    "TestDtypeAwareExclusions": {
        "test_dtype_filter": {
            "dtypes": [torch.float32],
        },
        "test_all_dtypes_excluded": {
            "dtypes": [torch.float32, torch.float64],
        },
        "test_dtype_tuple_filter": {
            "dtypes": [torch.int32],
        },
    },
}
with _temp_test_configs(
    PrivateUse1TestBase, test_exclusions=OPENREG_DTYPE_TEST_EXCLUSIONS
):
    instantiate_device_type_tests(
        TestDtypeAwareExclusions, globals(), only_for="openreg"
    )

with _temp_test_configs(
    PrivateUse1TestBase,
    op_overrides={
        "op_combined_skip": [DecorateInfo(unittest.skip("skip via op_overrides"))]
    },
    op_allowlist=("op_combined_supported", "op_combined_skip"),
):
    instantiate_device_type_tests(
        TestSupportedOpsWithOverrides, globals(), only_for=("openreg",)
    )


@unittest.skipIf(not dist.is_available(), "Distributed not available, skipping tests")
class TestDistributedBackendHook(TestCase):
    # [CLASS DOC] Skipped when torch.distributed is unavailable. Checks the test base class
    # reports the distributed backend associated with the openreg device.
    def test_distributed_backend_for_openreg(self, device):
        # [FUNCTION DOC] Asserts the instantiated class's distributed_backend hook returns
        # "occl".
        self.assertEqual(type(self).distributed_backend(), "occl")


instantiate_device_type_tests(
    TestDistributedBackendHook, globals(), only_for=("openreg",)
)

if __name__ == "__main__":
    run_tests()
