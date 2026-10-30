# Owner(s): ["module: onnx"]
from __future__ import annotations
# [MODULE DOC] Shared helpers for the ONNX test suite: constants for batch and RNN sizes, an
# enum of model kinds, a family of decorators that skip or expected-fail tests depending on
# opset version, tracing vs scripting mode, dynamic shapes, model type, CI, or ONNX Runtime
# version, ready-made skip decorators for missing CUDA, Travis, bfloat16 CUDA and the QNNPACK
# quantization backend, small utilities for flattening tensors and seeding random generators,
# and the ExportTestCase base class. Import-time side effects: appends the parent test directory
# to the module search path (inserted before the last entry) and sets the default tensor dtype
# to float32. Decorators that read test instance attributes (opset_version, is_script,
# dynamic_shapes, model_type) expect to be applied to test methods of a class that defines them.

import functools
import os
import random
import sys
import unittest
from enum import auto, Enum

import numpy as np
import packaging.version
import pytest

import torch
from torch.autograd import function
from torch.testing._internal import common_utils


pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(-1, pytorch_test_dir)

torch.set_default_dtype(torch.float)

BATCH_SIZE = 2

RNN_BATCH_SIZE = 7
RNN_SEQUENCE_LENGTH = 11
RNN_INPUT_SIZE = 5
RNN_HIDDEN_SIZE = 3


class TorchModelType(Enum):
    # [CLASS DOC] Enumeration of the kinds of model a test exports: a plain torch.nn.Module or
    # an ExportedProgram from torch.export. Values are automatic and only used for identity
    # comparison by the xfail and skip decorators.
    TORCH_NN_MODULE = auto()
    TORCH_EXPORT_EXPORTEDPROGRAM = auto()


def _skipper(condition, reason):
    # [FUNCTION DOC] HELPER: Builds a decorator factory. Given a zero-argument condition
    # callable and a reason, it returns a decorator whose wrapped test evaluates the condition
    # at call time and raises a unittest skip with the reason when true, otherwise calls the
    # test.
    def decorator(f):
        # [FUNCTION DOC] HELPER: Inner decorator that applies the wrapping to the test function
        # and preserves its metadata.
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper replacing the decorated test; evaluates the
            # captured condition at call time, raises a unittest SkipTest with the reason if it
            # is true, otherwise returns the result of the original function.
            if condition():
                raise unittest.SkipTest(reason)
            return f(*args, **kwargs)

        return wrapper

    return decorator


skipIfNoCuda = _skipper(lambda: not torch.cuda.is_available(), "CUDA is not available")

skipIfTravis = _skipper(lambda: os.getenv("TRAVIS"), "Skip In Travis")

skipIfNoBFloat16Cuda = _skipper(
    lambda: not torch.cuda.is_bf16_supported(), "BFloat16 CUDA is not available"
)

skipIfQuantizationBackendQNNPack = _skipper(
    lambda: torch.backends.quantized.engine == "qnnpack",
    "Not compatible with QNNPack quantization backend",
)


# skips tests for all versions below min_opset_version.
# add this wrapper to prevent running the test for opset_versions
# smaller than `min_opset_version`.
def skipIfUnsupportedMinOpsetVersion(min_opset_version):
    # [FUNCTION DOC] Decorator factory: tests are skipped (SkipTest raised at call time) when
    # the test instance opset_version is lower than min_opset_version. The skip message shows
    # both versions.
    def skip_dec(func):
        # [FUNCTION DOC] HELPER: Inner decorator that applies the wrapping to the test function
        # and preserves its metadata.
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper replacing the decorated test; reads the instance
            # opset version, raises SkipTest if below the minimum, else runs the test.
            if self.opset_version < min_opset_version:
                raise unittest.SkipTest(
                    f"Unsupported opset_version: {self.opset_version} < {min_opset_version}"
                )
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


# skips tests for all versions above max_opset_version.
# add this wrapper to prevent running the test for opset_versions
# higher than `max_opset_version`.
def skipIfUnsupportedMaxOpsetVersion(max_opset_version):
    # [FUNCTION DOC] Decorator factory: tests are skipped (SkipTest raised at call time) when
    # the instance opset_version is greater than max_opset_version. The skip message shows both
    # versions.
    def skip_dec(func):
        # [FUNCTION DOC] HELPER: Inner decorator that applies the wrapping to the test function
        # and preserves its metadata.
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper replacing the decorated test; raises SkipTest if
            # the instance opset version exceeds the maximum, else runs the test.
            if self.opset_version > max_opset_version:
                raise unittest.SkipTest(
                    f"Unsupported opset_version: {self.opset_version} > {max_opset_version}"
                )
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


# skips tests for all opset versions.
def skipForAllOpsetVersions():
    # [FUNCTION DOC] Decorator factory with no arguments: the decorated test is skipped whenever
    # the instance opset_version is truthy (nonzero), i.e. effectively always for real opsets.
    def skip_dec(func):
        # [FUNCTION DOC] HELPER: Inner decorator that applies the wrapping to the test function
        # and preserves its metadata.
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper replacing the decorated test; raises SkipTest when
            # the instance opset version is truthy, else runs the test.
            if self.opset_version:
                raise unittest.SkipTest(
                    "Skip verify test for unsupported opset_version"
                )
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


def skipTraceTest(skip_before_opset_version: int | None = None, reason: str = ""):
    # [FUNCTION DOC] Decorator factory that skips a test in tracing (non-script) mode. With a
    # minimum opset given, skipping applies when the instance opset_version is lower than it;
    # with none, always. Args: skip_before_opset_version (optional int), reason (text added to
    # the skip message).
    """Skip tracing test for opset version less than skip_before_opset_version.

    Args:
        skip_before_opset_version: The opset version before which to skip tracing test.
            If None, tracing test is always skipped.
        reason: The reason for skipping tracing test.

    Returns:
        A decorator for skipping tracing test.
    """

    def skip_dec(func):
        # [FUNCTION DOC] HELPER: Inner decorator that applies the wrapping to the test function
        # and preserves its metadata.
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper replacing the decorated test; sets the instance
            # attribute skip_this_opset according to the opset threshold (side effect on the
            # test case), raises SkipTest if that flag is set and the instance is not in script
            # mode, otherwise runs the test.
            if skip_before_opset_version is not None:
                self.skip_this_opset = self.opset_version < skip_before_opset_version
            else:
                self.skip_this_opset = True
            if self.skip_this_opset and not self.is_script:
                raise unittest.SkipTest(f"Skip verify test for torch trace. {reason}")
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


def skipScriptTest(skip_before_opset_version: int | None = None, reason: str = ""):
    # [FUNCTION DOC] Decorator factory that skips a test in scripting mode. With a minimum opset
    # given, skipping applies when the instance opset_version is lower than it; with none,
    # always. Args: skip_before_opset_version (optional int), reason (text added to the skip
    # message).
    """Skip scripting test for opset version less than skip_before_opset_version.

    Args:
        skip_before_opset_version: The opset version before which to skip scripting test.
            If None, scripting test is always skipped.
        reason: The reason for skipping scripting test.

    Returns:
        A decorator for skipping scripting test.
    """

    def skip_dec(func):
        # [FUNCTION DOC] HELPER: Inner decorator that applies the wrapping to the test function
        # and preserves its metadata.
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper replacing the decorated test; sets the instance
            # attribute skip_this_opset according to the opset threshold (side effect on the
            # test case), raises SkipTest if that flag is set and the instance is in script
            # mode, otherwise runs the test.
            if skip_before_opset_version is not None:
                self.skip_this_opset = self.opset_version < skip_before_opset_version
            else:
                self.skip_this_opset = True
            if self.skip_this_opset and self.is_script:
                raise unittest.SkipTest(f"Skip verify test for TorchScript. {reason}")
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


# NOTE: This decorator is currently unused, but we may want to use it in the future when
# we have more tests that are not supported in released ORT.
def skip_min_ort_version(reason: str, version: str, dynamic_only: bool = False):
    # [FUNCTION DOC] Decorator factory (currently unused) that skips a test when the instance
    # ort_version is older than the given version string, comparing only release tuples. With
    # dynamic_only, tests run normally when the instance does not use dynamic shapes.
    # KNOWN ISSUE: the skip message states the same version for both the installed and required
    #   version because it formats the required version twice.
    def skip_dec(func):
        # [FUNCTION DOC] HELPER: Inner decorator that applies the wrapping to the test function
        # and preserves its metadata.
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper replacing the decorated test; compares the parsed
            # instance ORT version with the required one; if older, either runs the test
            # (dynamic-only case without dynamic shapes) or raises SkipTest; otherwise runs the
            # test.
            if (
                packaging.version.parse(self.ort_version).release
                < packaging.version.parse(version).release
            ):
                if dynamic_only and not self.dynamic_shapes:
                    return func(self, *args, **kwargs)

                raise unittest.SkipTest(
                    f"ONNX Runtime version: {version} is older than required version {version}. "
                    f"Reason: {reason}."
                )
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


def xfail_dynamic_fx_test(
    error_message: str,
    model_type: TorchModelType | None = None,
    reason: str | None = None,
):
    # [FUNCTION DOC] Decorator factory that turns a test into an expected failure when the
    # instance uses dynamic shapes and, if model_type is given, the instance model_type equals
    # it. error_message is the text that the failure must contain; reason is optional text for
    # the xfail report. Other cases run the test normally.
    """Xfail dynamic exporting test.

    Args:
        reason: The reason for xfailing dynamic exporting test.
        model_type (TorchModelType): The model type to xfail dynamic exporting test for.
            When None, model type is not used to xfail dynamic tests.

    Returns:
        A decorator for xfailing dynamic exporting test.
    """

    def skip_dec(func):
        # [FUNCTION DOC] HELPER: Inner decorator that applies the wrapping to the test function
        # and preserves its metadata.
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper replacing the decorated test; when dynamic shapes
            # are on and the model type matches (or no type was given) runs the test through the
            # xfail decorator with the message and reason and returns its result; otherwise runs
            # the test directly.
            if self.dynamic_shapes and (
                not model_type or self.model_type == model_type
            ):
                return xfail(error_message, reason)(func)(self, *args, **kwargs)
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


def skip_dynamic_fx_test(reason: str, model_type: TorchModelType = None):
    # [FUNCTION DOC] Decorator factory that skips a test (SkipTest) when the instance uses
    # dynamic shapes and, if model_type is given, the instance model_type equals it. The reason
    # is added to the skip message.
    """Skip dynamic exporting test.

    Args:
        reason: The reason for skipping dynamic exporting test.
        model_type (TorchModelType): The model type to skip dynamic exporting test for.
            When None, model type is not used to skip dynamic tests.

    Returns:
        A decorator for skipping dynamic exporting test.
    """

    def skip_dec(func):
        # [FUNCTION DOC] HELPER: Inner decorator that applies the wrapping to the test function
        # and preserves its metadata.
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper replacing the decorated test; raises SkipTest when
            # dynamic shapes are on and the model type matches or none was given, otherwise runs
            # the test.
            if self.dynamic_shapes and (
                not model_type or self.model_type == model_type
            ):
                raise unittest.SkipTest(
                    f"Skip verify dynamic shapes test for FX. {reason}"
                )
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


def skip_in_ci(reason: str):
    # [FUNCTION DOC] Decorator factory that skips a test (SkipTest) when the CI environment
    # variable is set to a non-empty value. The reason is added to the skip message.
    """Skip test in CI.

    Args:
        reason: The reason for skipping test in CI.

    Returns:
        A decorator for skipping test in CI.
    """

    def skip_dec(func):
        # [FUNCTION DOC] HELPER: Inner decorator that applies the wrapping to the test function
        # and preserves its metadata.
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper replacing the decorated test; raises SkipTest when
            # the CI environment variable is set, otherwise runs the test.
            if os.getenv("CI"):
                raise unittest.SkipTest(f"Skip test in CI. {reason}")
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


def xfail(error_message: str, reason: str | None = None):
    # [FUNCTION DOC] Decorator factory marking a test as expected to fail with a particular
    # error. If the test raises an exception whose text contains error_message, pytest xfail is
    # triggered (with reason, or a default message); if the text does not contain it, an
    # AssertionError describing the mismatch is raised; if the test does not raise, pytest fail
    # reports unexpected success.
    """Expect failure.

    Args:
        reason: The reason for expected failure.

    Returns:
        A decorator for expecting test failure.
    """

    def wrapper(func):
        # [FUNCTION DOC] HELPER: Inner decorator applying the expected-failure wrapping to the
        # test function.
        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Runs the test inside a try block catching ordinary
            # exceptions and applying the xfail rules described above; returns nothing and never
            # returns the test result.
            try:
                func(self, *args, **kwargs)
            except Exception as e:
                if error_message not in str(e):
                    raise AssertionError(
                        f"Expected error message: {error_message} NOT in {str(e)}"
                    ) from None
                pytest.xfail(reason if reason else f"Expected failure: {error_message}")
            else:
                pytest.fail("Unexpected success!")

        return inner

    return wrapper


# skips tests for opset_versions listed in unsupported_opset_versions.
# if the PyTorch test cannot be run for a specific version, add this wrapper
# (for example, an op was modified but the change is not supported in PyTorch)
def skipIfUnsupportedOpsetVersion(unsupported_opset_versions):
    # [FUNCTION DOC] Decorator factory: tests are skipped (SkipTest) when the instance
    # opset_version is a member of the given collection of unsupported opset versions.
    def skip_dec(func):
        # [FUNCTION DOC] HELPER: Inner decorator that applies the wrapping to the test function
        # and preserves its metadata.
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper replacing the decorated test; raises SkipTest when
            # the instance opset version is in the unsupported collection, else runs the test.
            if self.opset_version in unsupported_opset_versions:
                raise unittest.SkipTest(
                    "Skip verify test for unsupported opset_version"
                )
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


def skipShapeChecking(func):
    # [FUNCTION DOC] Decorator that, when the test is called, first sets the instance attribute
    # check_shape to false (persisting on the instance) and then runs the test, returning its
    # result.
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper replacing the decorated test; disables shape checking
        # on the instance, then runs the test.
        self.check_shape = False
        return func(self, *args, **kwargs)

    return wrapper


def skipDtypeChecking(func):
    # [FUNCTION DOC] Decorator that, when the test is called, first sets the instance attribute
    # check_dtype to false (persisting on the instance) and then runs the test, returning its
    # result.
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper replacing the decorated test; disables dtype checking
        # on the instance, then runs the test.
        self.check_dtype = False
        return func(self, *args, **kwargs)

    return wrapper


def xfail_if_model_type_is_exportedprogram(
    error_message: str, reason: str | None = None
):
    # [FUNCTION DOC] Decorator factory that makes a test an expected failure (via the xfail
    # decorator, with the given error message and optional reason) when the instance model_type
    # is the exported-program kind; other model types run normally.
    """xfail test with models using ExportedProgram as input.

    Args:
        error_message: The error message to raise when the test is xfailed.
        reason: The reason for xfail the ONNX export test.

    Returns:
        A decorator for xfail tests.
    """

    def xfail_dec(func):
        # [FUNCTION DOC] HELPER: Inner decorator that applies the wrapping to the test function
        # and preserves its metadata.
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper replacing the decorated test; routes the test
            # through xfail when the instance model type is exported-program, else runs it
            # directly.
            if self.model_type == TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM:
                return xfail(error_message, reason)(func)(self, *args, **kwargs)
            return func(self, *args, **kwargs)

        return wrapper

    return xfail_dec


def xfail_if_model_type_is_not_exportedprogram(
    error_message: str, reason: str | None = None
):
    # [FUNCTION DOC] Decorator factory that makes a test an expected failure (via the xfail
    # decorator, with the given error message and optional reason) when the instance model_type
    # is anything other than the exported-program kind; exported-program tests run normally.
    """xfail test without models using ExportedProgram as input.

    Args:
        reason: The reason for xfail the ONNX export test.

    Returns:
        A decorator for xfail tests.
    """

    def xfail_dec(func):
        # [FUNCTION DOC] HELPER: Inner decorator that applies the wrapping to the test function
        # and preserves its metadata.
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper replacing the decorated test; routes the test
            # through xfail when the instance model type is not exported-program, else runs it
            # directly.
            if self.model_type != TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM:
                return xfail(error_message, reason)(func)(self, *args, **kwargs)
            return func(self, *args, **kwargs)

        return wrapper

    return xfail_dec


def flatten(x):
    # [FUNCTION DOC] Flattens a nested structure of inputs or outputs using the autograd
    # function iteration helper, keeps only tensor leaves, and returns them as a tuple.
    return tuple(function._iter_filter(lambda o: isinstance(o, torch.Tensor))(x))


def set_rng_seed(seed):
    # [FUNCTION DOC] Seeds the global torch generator, the Python random module and the numpy
    # generator with the same seed. Global side effect; returns nothing.
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class ExportTestCase(common_utils.TestCase):
    # [CLASS DOC] Base class for tests of torch.onnx functionality. Extends the common test case
    # and makes each test deterministic by seeding in setUp.
    """Test case for ONNX export.

    Any test case that tests functionalities under torch.onnx should inherit from this class.
    """

    def setUp(self):
        # [FUNCTION DOC] Runs the parent setUp, seeds the Python, numpy and torch generators
        # with zero, then, if an accelerator is currently available, also seeds all devices of
        # that accelerator type with zero through the matching device module.
        super().setUp()
        # TODO(#88264): Flaky test failures after changing seed.
        set_rng_seed(0)
        device_type = getattr(
            torch.accelerator.current_accelerator(check_available=True), "type", None
        )
        if device_type:
            torch.get_device_module(device_type).manual_seed_all(0)
