# mypy: ignore-errors
# [MODULE DOC]
# Device-generic test framework: turns test template classes (tests that take a "device" and
# optionally "dtype"/"op" arguments) into concrete per-device-type test classes.
# Main contents: DeviceTypeTestBase and its per-backend subclasses (CPU, CUDA, MPS, XPU, HPU,
# Lazy, PrivateUse1); instantiate_device_type_tests, which replaces a template class in a given
# module scope with generated classes; the @dtypes/@ops/@precisionOverride/@toleranceOverride
# parametrization decorators; and many skip / only / expected-failure decorators keyed on device
# type, library availability, and CUDA/ROCm/cuDNN versions; plus largeTensorTest memory gating.
# Import-time side effects: probes torch for available backends to build the module-level list
# device_type_test_bases; if the TORCH_TEST_DEVICES environment variable is set, executes each
# listed Python file (separated by colons) and appends the TEST_CLASS it defines; computes
# flex-attention platform flags and the float8 type/max constants (queries CUDA/XPU/MPS state);
# creates a thread-local holding the default precision and relative tolerance.

import copy
import gc
import inspect
import logging
import os
import runpy
import sys
import threading
import unittest
from collections import namedtuple
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import Enum
from functools import partial, wraps
from typing import Any, ClassVar, TypeVar
from typing_extensions import ParamSpec

import torch
from torch._inductor.utils import GPU_TYPES
from torch._utils import _is_privateuse1_backend_available
from torch.testing._internal.common_cuda import (
    _get_torch_cuda_version,
    _get_torch_hipblaslt_version,
    _get_torch_rocm_version,
    TEST_CUSPARSE_GENERIC,
    TEST_HIPSPARSE_GENERIC,
)
from torch.testing._internal.common_dtype import get_all_dtypes
from torch.testing._internal.common_utils import (
    _TestParametrizer,
    clear_tracked_input,
    compose_parametrize_fns,
    dtype_name,
    get_tracked_input,
    IS_FBCODE,
    IS_MACOS,
    IS_REMOTE_GPU,
    IS_S390X,
    IS_SANDCASTLE,
    IS_WINDOWS,
    NATIVE_DEVICES,
    PRINT_REPRO_ON_FAILURE,
    skipCUDANonDefaultStreamIf,
    skipIfTorchDynamo,
    TEST_HPU,
    TEST_MKL,
    TEST_MPS,
    TEST_WITH_ASAN,
    TEST_WITH_MIOPEN_SUGGEST_NHWC,
    TEST_WITH_MTIA,
    TEST_WITH_ROCM,
    TEST_WITH_TORCHINDUCTOR,
    TEST_WITH_TSAN,
    TEST_WITH_UBSAN,
    TEST_XPU,
    TestCase,
    validate_test_name,
)


_T = TypeVar("_T")
_P = ParamSpec("_P")

try:
    import psutil  # type: ignore[import]

    HAS_PSUTIL = True
except ModuleNotFoundError:
    HAS_PSUTIL = False
    psutil = None

log = logging.getLogger(__name__)

# Note [Writing Test Templates]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This note was written shortly after the PyTorch 1.9 release.
# If you notice it's out-of-date or think it could be improved then please
# file an issue.
#
# PyTorch has its own framework for instantiating test templates. That is, for
#   taking test classes that look similar to unittest or pytest
#   compatible test classes and optionally doing the following:
#
#     - instantiating a version of the test class for each available device type
#         (often the CPU, CUDA, and META device types)
#     - further instantiating a version of each test that's always specialized
#         on the test class's device type, and optionally specialized further
#         on datatypes or operators
#
# This functionality is similar to pytest's parametrize functionality
#   (see https://docs.pytest.org/en/6.2.x/parametrize.html), but with considerable
#   additional logic that specializes the instantiated test classes for their
#   device types (see CPUTestBase and CUDATestBase below), supports a variety
#   of composable decorators that allow for test filtering and setting
#   tolerances, and allows tests parametrized by operators to instantiate
#   only the subset of device type x dtype that operator supports.
#
# This framework was built to make it easier to write tests that run on
#   multiple device types, multiple datatypes (dtypes), and for multiple
#   operators. It's also useful for controlling which tests are run. For example,
#   only tests that use a CUDA device can be run on platforms with CUDA.
#   Let's dive in with an example to get an idea for how it works:
#
# --------------------------------------------------------
# A template class (looks like a regular unittest TestCase)
# class TestClassFoo(TestCase):
#
#   # A template test that can be specialized with a device
#   # NOTE: this test case is not runnable by unittest or pytest because it
#   #   accepts an extra positional argument, "device", that they do not understand
#   def test_bar(self, device):
#     pass
#
# # Function that instantiates a template class and its tests
# instantiate_device_type_tests(TestCommon, globals())
# --------------------------------------------------------
#
# In the above code example we see a template class and a single test template
#   that can be instantiated with a device. The function
#   instantiate_device_type_tests(), called at file scope, instantiates
#   new test classes, one per available device type, and new tests in those
#   classes from these templates. It actually does this by removing
#   the class TestClassFoo and replacing it with classes like TestClassFooCPU
#   and TestClassFooCUDA, instantiated test classes that inherit from CPUTestBase
#   and CUDATestBase respectively. Additional device types, like XLA,
#   (see https://github.com/pytorch/xla) can further extend the set of
#   instantiated test classes to create classes like TestClassFooXLA.
#
# The test template, test_bar(), is also instantiated. In this case the template
#   is only specialized on a device, so (depending on the available device
#   types) it might become test_bar_cpu() in TestClassFooCPU and test_bar_cuda()
#   in TestClassFooCUDA. We can think of the instantiated test classes as
#   looking like this:
#
# --------------------------------------------------------
# # An instantiated test class for the CPU device type
# class TestClassFooCPU(CPUTestBase):
#
#   # An instantiated test that calls the template with the string representation
#   #   of a device from the test class's device type
#   def test_bar_cpu(self):
#     test_bar(self, 'cpu')
#
# # An instantiated test class for the CUDA device type
# class TestClassFooCUDA(CUDATestBase):
#
#   # An instantiated test that calls the template with the string representation
#   #   of a device from the test class's device type
#   def test_bar_cuda(self):
#     test_bar(self, 'cuda:0')
# --------------------------------------------------------
#
# These instantiated test classes ARE discoverable and runnable by both
#   unittest and pytest. One thing that may be confusing, however, is that
#   attempting to run "test_bar" will not work, despite it appearing in the
#   original template code. This is because "test_bar" is no longer discoverable
#   after instantiate_device_type_tests() runs, as the above snippet shows.
#   Instead "test_bar_cpu" and "test_bar_cuda" may be run directly, or both
#   can be run with the option "-k test_bar".
#
# Removing the template class and adding the instantiated classes requires
#   passing "globals()" to instantiate_device_type_tests(), because it
#   edits the file's Python objects.
#
# As mentioned, tests can be additionally parametrized on dtypes or
#   operators. Datatype parametrization uses the @dtypes decorator and
#   require a test template like this:
#
# --------------------------------------------------------
# # A template test that can be specialized with a device and a datatype (dtype)
# @dtypes(torch.float32, torch.int64)
# def test_car(self, device, dtype)
#   pass
# --------------------------------------------------------
#
# If the CPU and CUDA device types are available this test would be
#   instantiated as 4 tests that cover the cross-product of the two dtypes
#   and two device types:
#
#     - test_car_cpu_float32
#     - test_car_cpu_int64
#     - test_car_cuda_float32
#     - test_car_cuda_int64
#
# The dtype is passed as a torch.dtype object.
#
# Tests parametrized on operators (actually on OpInfos, more on that in a
#   moment...) use the @ops decorator and require a test template like this:
# --------------------------------------------------------
# # A template test that can be specialized with a device, dtype, and OpInfo
# @ops(op_db)
# def test_car(self, device, dtype, op)
#   pass
# --------------------------------------------------------
#
# See the documentation for the @ops decorator below for additional details
#   on how to use it and see the note [OpInfos] in
#   common_methods_invocations.py for more details on OpInfos.
#
# A test parametrized over the entire "op_db", which contains hundreds of
#   OpInfos, will likely have hundreds or thousands of instantiations. The
#   test will be instantiated on the cross-product of device types, operators,
#   and the dtypes the operator supports on that device type. The instantiated
#   tests will have names like:
#
#     - test_car_add_cpu_float32
#     - test_car_sub_cuda_int64
#
# The first instantiated test calls the original test_car() with the OpInfo
#   for torch.add as its "op" argument, the string 'cpu' for its "device" argument,
#   and the dtype torch.float32 for is "dtype" argument. The second instantiated
#   test calls the test_car() with the OpInfo for torch.sub, a CUDA device string
#   like 'cuda:0' or 'cuda:1' for its "device" argument, and the dtype
#   torch.int64 for its "dtype argument."
#
# In addition to parametrizing over device, dtype, and ops via OpInfos, the
#   @parametrize decorator is supported for arbitrary parametrizations:
# --------------------------------------------------------
# # A template test that can be specialized with a device, dtype, and value for x
# @parametrize("x", range(5))
# def test_car(self, device, dtype, x)
#   pass
# --------------------------------------------------------
#
# See the documentation for @parametrize in common_utils.py for additional details
#   on this. Note that the instantiate_device_type_tests() function will handle
#   such parametrizations; there is no need to additionally call
#   instantiate_parametrized_tests().
#
# Clever test filtering can be very useful when working with parametrized
#   tests. "-k test_car" would run every instantiated variant of the test_car()
#   test template, and "-k test_car_add" runs every variant instantiated with
#   torch.add.
#
# It is important to use the passed device and dtype as appropriate. Use
#   helper functions like make_tensor() that require explicitly specifying
#   the device and dtype so they're not forgotten.
#
# Test templates can use a variety of composable decorators to specify
#   additional options and requirements, some are listed here:
#
#     - @deviceCountAtLeast(<minimum number of devices to run test with>)
#         Passes a list of strings representing all available devices of
#         the test class's device type as the test template's "device" argument.
#         If there are fewer devices than the value passed to the decorator
#         the test is skipped.
#     - @dtypes(<list of tuples of dtypes>)
#         In addition to accepting multiple dtypes, the @dtypes decorator
#         can accept a sequence of tuple pairs of dtypes. The test template
#         will be called with each tuple for its "dtype" argument.
#     - @onlyNativeDeviceTypes
#         Skips the test if the device is not a native device type (currently CPU, CUDA, Meta)
#     - @onlyCPU
#         Skips the test if the device is not a CPU device
#     - @onlyCUDA
#         Skips the test if the device is not a CUDA device
#     - @onlyMPS
#         Skips the test if the device is not a MPS device
#     - @skipCPUIfNoLapack
#         Skips the test if the device is a CPU device and LAPACK is not installed
#     - @skipCPUIfNoMkl
#         Skips the test if the device is a CPU device and MKL is not installed
#     - @skipCUDAIfNoMagma
#         Skips the test if the device is a CUDA device and MAGMA is not installed
#     - @skipCUDAIfRocm
#         Skips the test if the device is a CUDA device and ROCm is being used


# Note [Adding a Device Type]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To add a device type:
#
#   (1) Create a new "TestBase" extending DeviceTypeTestBase.
#       See CPUTestBase and CUDATestBase below.
#   (2) Define the "device_type" attribute of the base to be the
#       appropriate string.
#   (3) Add logic to this file that appends your base class to
#       device_type_test_bases when your device type is available.
#   (4) (Optional) Write setUpClass/tearDownClass class methods that
#       instantiate dependencies (see MAGMA in CUDATestBase).
#   (5) (Optional) Override the "instantiate_test" method for total
#       control over how your class creates tests.
#
# setUpClass is called AFTER tests have been created and BEFORE and ONLY IF
# they are run. This makes it useful for initializing devices and dependencies.


def _dtype_test_suffix(dtypes):
    # [FUNCTION DOC] HELPER: Builds the name suffix for a test variant from its dtype(s).
    # A non-empty list or tuple gives an underscore plus the dtype names joined by underscores;
    # a single truthy dtype gives an underscore plus its name; anything else (empty, None)
    # gives an empty string. Pure function, no side effects.
    """Returns the test suffix for a dtype, sequence of dtypes, or None."""
    if isinstance(dtypes, (list, tuple)):
        if len(dtypes) == 0:
            return ""
        return "_" + "_".join(dtype_name(d) for d in dtypes)
    elif dtypes:
        return f"_{dtype_name(dtypes)}"
    else:
        return ""


def _update_param_kwargs(param_kwargs, name, value):
    # [FUNCTION DOC] HELPER: Sets one argument in the mapping of test keyword arguments.
    # First removes any existing entry under the singular name or the plural name (name plus
    # "s"). Then a list/tuple value is stored under the plural name, any other non-None value
    # under the singular name, and None leaves the mapping without the argument.
    # Mutates param_kwargs in place; returns nothing.
    """Adds a kwarg with the specified name and value to the param_kwargs dict."""
    # Make name plural (e.g. devices / dtypes) if the value is composite.
    plural_name = f"{name}s"

    # Clear out old entries of the arg if any.
    if name in param_kwargs:
        del param_kwargs[name]
    if plural_name in param_kwargs:
        del param_kwargs[plural_name]

    if isinstance(value, (list, tuple)):
        param_kwargs[plural_name] = value
    elif value is not None:
        param_kwargs[name] = value

    # Leave param_kwargs as-is when value is None.


class DeviceTypeTestBase(TestCase):
    # [CLASS DOC] Base class for the per-device-type test classes generated from templates.
    # Class attributes: device_type (string name), bypass_device_restrictions (lets only-on
    # decorators run anyway), op_overrides / op_allowlist / test_exclusions (optional test
    # configuration, all None by default), and a thread-local holder of precision/rel_tol.
    # Key methods: instantiate_test (expands one template test into concrete tests),
    # get_primary_device / get_all_devices (device strings), precision and rel_tol properties,
    # and helpers that apply op overrides, allowlists, exclusions and per-dtype tolerances.
    # Subclasses set device_type and usually override device discovery in setUpClass.
    device_type: str = "generic_device_type"

    # When True, @onlyOn-based decorators (@onlyCUDA, @onlyMPS, etc.) will not
    # skip tests for this device type. This is a pragmatic short-term solution to
    # allow PrivateUse1 backends to run tests that are currently gated behind
    # device-specific decorators. It is intended to be used together with the
    # skip mechanism (see https://github.com/pytorch/pytorch/issues/177253).
    # In the longer term, we are incrementally migrating accelerator tests to be
    # device-generic and removing @onlyCUDA on tests that should be device-generic.
    bypass_device_restrictions: bool = False

    # Decorators and skips to apply to tests that are parametrized by ops.
    # Keys are OpInfo.full_name (e.g. "op" or "linalg.norm"), NOT OpInfo.name.
    # Note that OpInfo.full_name and OpInfo.name have difference: OpInfo.full_name
    # includes the variant suffix (e.g., "div.floor_rounding") if it has, otherwise,
    # OpInfo.full_name identical to OpInfo.name.
    # These are intentionally placed on DeviceTypeTestBase (rather than solely on
    # PrivateUse1TestBase) so that in-tree backends can adopt the same mechanism
    # in the future.
    op_overrides = None  # type: Optional[dict[str, list[DecorateInfo]]]

    # An optional mechanism to limit which ops generate test variants.
    # When set, only ops whose full_name is in this collection will generate tests.
    # This is useful for less mature backends that only implement a few operators.
    # Keys are OpInfo.full_name (e.g. "add", "mul", "linalg.norm").
    # If None (default), all ops in the @ops decorator's op_list generate variants.
    op_allowlist = None  # type: Optional[Collection[str]]

    # An optional skip mechanism built upon instantiate_device_type_tests(),
    # designed to filter generated tests at different granularities.
    #
    # Supported exclusions:
    #   - Exclude an entire test class
    #   - Exclude specific test methods within a class
    #   - Exclude specific @dtypes/@dtypesIf-generated dtype variants of a
    #     test method
    #
    # Format:
    #
    #   test_exclusions = {
    #       # Exclude all generated variants in the class.
    #       "TestClassA": "*",
    #
    #       # Simple form: exclude all generated variants of specific methods.
    #       "TestClassB": ["test_a", "test_b"],
    #
    #       # Advanced form: supports both unconditional method exclusions
    #       # and fine-grained generated variant exclusions (e.g. by dtype).
    #       "TestClassC": {
    #           "test_a": "*",
    #           "test_b": {
    #               "dtypes": [torch.float32],
    #           },
    #       },
    #   }
    #
    # Note:
    #   Conditional filtering currently only supports per-method dtype-based
    #   filtering for variants generated from @dtypes and @dtypesIf decorators
    #   in the unified mapping form. Class-level dtype exclusions are not
    #   supported.
    #   A tuple/list dtype variant is excluded if any dtype in the variant is
    #   listed in the method's excluded dtypes.
    #   @ops-generated dtype variants and other parametrized arguments are
    #   ignored for now.
    test_exclusions: ClassVar[dict[str, Any] | None] = None

    # Flag to disable test suite early due to unrecoverable error such as CUDA error.
    _stop_test_suite = False

    # Precision is a thread-local setting since it may be overridden per test
    _tls = threading.local()
    _tls.precision = TestCase._precision
    _tls.rel_tol = TestCase._rel_tol

    @property
    def precision(self):
        # [FUNCTION DOC] Property: returns the absolute tolerance from the thread-local state.
        return self._tls.precision

    @precision.setter
    def precision(self, prec):
        # [FUNCTION DOC] Setter: stores the absolute tolerance in the thread-local state,
        # which is shared by all instances in the same thread.
        self._tls.precision = prec

    @property
    def rel_tol(self):
        # [FUNCTION DOC] Property: returns the relative tolerance from the thread-local state.
        return self._tls.rel_tol

    @rel_tol.setter
    def rel_tol(self, prec):
        # [FUNCTION DOC] Setter: stores the relative tolerance in the thread-local state.
        self._tls.rel_tol = prec

    @classmethod
    def _apply_op_overrides(cls, ops, test=None):
        # [FUNCTION DOC] HELPER: Attaches override decorators to the ops used by an @ops test.
        # Collects class-level overrides (mapping from op full name to decorator list) and, if
        # a test is given, the overrides stored on it. If there are none, returns immediately.
        # Otherwise deep-copies the op list, appends each decorator to the matching op (class
        # decorators first get their device_type set to this class's device type or the
        # registered privateuse1 backend name), skips names not in the list, and replaces the
        # op list on the ops object (mutated in place). Returns nothing.
        class_overrides = cls.op_overrides or {}
        test_overrides = {} if test is None else getattr(test, "_op_overrides", {})

        if not class_overrides and not test_overrides:
            return

        op_dict = {op.full_name: op for op in copy.deepcopy(ops.op_list)}

        for op_name, decorators in class_overrides.items():
            for decorator in decorators:
                if cls.device_type == "privateuse1":
                    decorator.device_type = torch._C._get_privateuse1_backend_name()
                else:
                    decorator.device_type = cls.device_type
                # op_name may not be in op_dict if @ops() has restricted the
                # OpInfo list to a smaller set than op_overrides covers.
                if op_name in op_dict:
                    op_dict[op_name].decorators += (decorator,)

        for op_name, decorators in test_overrides.items():
            for decorator in decorators:
                if op_name in op_dict:
                    op_dict[op_name].decorators += (decorator,)

        ops.op_list = list(op_dict.values())

    # Returns a string representing the device that single device tests should use.
    # Note: single device tests use this device exclusively.
    @classmethod
    def get_primary_device(cls):
        # [FUNCTION DOC] Classmethod: returns the device string single-device tests use. The
        # default is just the device type name; subclasses override to return an indexed device.
        return cls.device_type

    @classmethod
    def distributed_backend(cls) -> str:
        # [FUNCTION DOC] Classmethod: returns the default distributed backend name for this
        # device type, by asking torch.distributed (imported lazily inside the call).
        """
        Default distributed backend for this device type.
        """
        import torch.distributed as dist

        return dist.get_default_backend_for_device(cls.device_type)

    @classmethod
    def _get_test_exclusions(cls, test_class_name):
        # [FUNCTION DOC] HELPER: Looks up the exclusion rule configured for a template class
        # name in test_exclusions. Returns the rule, or None if no configuration or no entry.
        test_exclusions = getattr(cls, "test_exclusions", None)
        if test_exclusions is not None and test_class_name in test_exclusions:
            return test_exclusions[test_class_name]
        return None

    @classmethod
    def _should_exclude(cls, test_class_name, *, test_name=None, dtype_variant=None):
        # [FUNCTION DOC] HELPER: Decides whether a generated class, method, or dtype variant is
        # excluded by test_exclusions. With only a class name, true when the rule is the
        # wildcard. With a test name, true for the wildcard method rule or when the name is in a
        # list rule. With a dtype variant too, true when any dtype of the variant is in the
        # method's excluded "dtypes" list. Returns false when no rule exists.
        # Raises an error if a dtype variant is given without a test name.
        if test_name is None and dtype_variant is not None:
            raise AssertionError("dtype_variant requires test_name")

        exclusion_rule = cls._get_test_exclusions(test_class_name)
        if exclusion_rule is None:
            return False

        def _check_class():
            # [FUNCTION DOC] HELPER: True if the whole class is excluded (rule is the wildcard).
            # "TestClassA": "*" — exclude all generated variants in the class.
            return exclusion_rule == "*"

        def _check_method():
            # [FUNCTION DOC] HELPER: True if the method is excluded entirely: either the mapping
            # rule maps it to the wildcard, or a list rule contains its name.
            # "TestClassC": {"test_a": "*"} — exclude all generated variants of
            # a specific method.
            if isinstance(exclusion_rule, Mapping):
                return exclusion_rule.get(test_name) == "*"
            # "TestClassB": ["test_a", "test_b"] — simple form, exclude all
            # generated variants of specific methods.
            if isinstance(exclusion_rule, list):
                return test_name in exclusion_rule
            return False

        def _check_dtype():
            # [FUNCTION DOC] HELPER: True if the method's mapping rule lists "dtypes" and the
            # dtype variant (a single dtype, or a list/tuple where any member counts) is in it.
            # "TestClassC": {"test_b": {"dtypes": [torch.float32]}} — exclude
            # specific dtype-generated variants of a method.
            if not isinstance(exclusion_rule, Mapping):
                return False
            method_exclusion = exclusion_rule.get(test_name)
            if not isinstance(method_exclusion, Mapping):
                return False
            excluded_dtypes = set(method_exclusion.get("dtypes", ()))
            if not excluded_dtypes:
                return False
            if isinstance(dtype_variant, (list, tuple)):
                return any(
                    component_dtype in excluded_dtypes
                    for component_dtype in dtype_variant
                )
            return dtype_variant in excluded_dtypes

        if test_name is None:
            return _check_class()
        if dtype_variant is None:
            return _check_method()
        return _check_dtype()

    @classmethod
    def _apply_op_allowlist(cls, ops):
        # [FUNCTION DOC] HELPER: If op_allowlist is set, replaces the op list of the given ops
        # object with only the ops whose full name is in the allowlist (mutates ops). No-op when
        # the allowlist is None.
        """Filters ops.op_list to only include ops declared in op_allowlist.

        If op_allowlist is None (default), no filtering is applied.
        If op_allowlist is set, only ops whose full_name is in the collection
        will generate test variants.

        Args:
            ops: The ops decorator instance whose op_list will be filtered.
        """
        if cls.op_allowlist is None:
            return

        supported_set = set(cls.op_allowlist)
        ops.op_list = [op for op in ops.op_list if op.full_name in supported_set]

    @classmethod
    def _init_and_get_primary_device(cls):
        # [FUNCTION DOC] HELPER: Returns the primary device string; if that raises (device info
        # is only set by setUpClass for some backends), calls setUpClass once on the class and
        # tries again. Side effect: may run the class's setUpClass.
        try:
            return cls.get_primary_device()
        except Exception:
            # For CUDATestBase, XPUTestBase, XLATestBase, and possibly others, the primary device won't be available
            # until setUpClass() sets it. Call that manually here if needed.
            if hasattr(cls, "setUpClass"):
                cls.setUpClass()
            return cls.get_primary_device()

    # Returns a list of strings representing all available devices of this
    # device type. The primary device must be the first string in the list
    # and the list must contain no duplicates.
    # Note: UNSTABLE API. Will be replaced once PyTorch has a device generic
    #   mechanism of acquiring all available devices.
    @classmethod
    def get_all_devices(cls):
        # [FUNCTION DOC] Classmethod: returns the list of all devices of this type, primary
        # first, without duplicates. The default only contains the primary device.
        return [cls.get_primary_device()]

    # Returns the dtypes the test has requested.
    # Prefers device-specific dtype specifications over generic ones.
    @classmethod
    def _get_dtypes(cls, test):
        # [FUNCTION DOC] HELPER: Returns the dtypes requested by a test via the dtype decorators,
        # preferring the entry for this device type over the generic "all" entry; None if the
        # test has no dtypes attribute. Raises an error if device-specific dtypes were given
        # without the generic ones.
        if not hasattr(test, "dtypes"):
            return None

        default_dtypes = test.dtypes.get("all")
        msg = f"@dtypes is mandatory when using @dtypesIf however '{test.__name__}' didn't specify it"
        if default_dtypes is None:
            raise AssertionError(msg)

        return test.dtypes.get(cls.device_type, default_dtypes)

    def _get_precision_override(self, test, dtype):
        # [FUNCTION DOC] HELPER: Returns the absolute tolerance override the test declares for
        # the dtype, or the current precision if the test has no overrides or none for the dtype.
        if not hasattr(test, "precision_overrides"):
            return self.precision
        return test.precision_overrides.get(dtype, self.precision)

    def _get_tolerance_override(self, test, dtype):
        # [FUNCTION DOC] HELPER: Returns the (atol, rtol) pair the test declares for the dtype,
        # or the current precision and relative tolerance if there is no override for it.
        if not hasattr(test, "tolerance_overrides"):
            return self.precision, self.rel_tol
        return test.tolerance_overrides.get(dtype, tol(self.precision, self.rel_tol))

    def _apply_precision_override_for_test(self, test, param_kwargs):
        # [FUNCTION DOC] HELPER: If the test variant has a dtype (or dtypes) argument, sets the
        # instance precision and relative tolerance from the test's precision and tolerance
        # overrides (tolerance overrides win). Mutates the thread-local tolerance state.
        dtype = param_kwargs.get("dtype")
        dtype = param_kwargs.get("dtypes", dtype)
        if dtype:
            self.precision = self._get_precision_override(test, dtype)
            self.precision, self.rel_tol = self._get_tolerance_override(test, dtype)

    @classmethod
    def set_test_configs(
        cls,
        *,
        op_overrides=None,
        op_allowlist=None,
        test_exclusions=None,
    ):
        # [FUNCTION DOC] Classmethod: replaces all three configuration class attributes
        # (op_overrides, op_allowlist, test_exclusions) with the given values; omitted ones
        # become None, so calling it with no arguments resets the configuration.
        """
        Sets or resets the test configuration fields.

        WARNING: This method is designed to perform a FULL replacement of the
        current configuration. Calling this method without any arguments will
        act as a "reset", clearing all stored configurations back to their
        default `None` state.
        """
        cls.op_overrides = op_overrides
        cls.op_allowlist = op_allowlist
        cls.test_exclusions = test_exclusions

    # Creates device-specific tests.
    @classmethod
    def instantiate_test(cls, name, test, *, generic_cls=None):
        # [FUNCTION DOC] Classmethod: expands one template test into concrete test methods set
        # on this class. Uses the test's parametrize function (default: one variant), composed
        # with dtype parametrization if dtype decorators are present (dtype variants listed in
        # test_exclusions are dropped; returns early if none remain). Each variant gets a name of
        # the base name, optional parameter suffix, device type, then dtype suffix; the name is
        # validated and the variant is installed via a nested helper.
        # Raises an error if a generated name already exists on the class.
        def instantiate_test_helper(
            cls, name, *, test, param_kwargs=None, decorator_fn=lambda _: []
        ):
            # [FUNCTION DOC] HELPER: Builds one concrete test and sets it on the class. Adds a
            # device argument (or the list of all devices if the test requires several) when the
            # test signature asks for one, applies the decorators returned by decorator_fn, and
            # wraps the test so that it runs with per-dtype tolerance overrides, restores the
            # previous tolerances afterwards, and on a RuntimeError records whether the suite
            # should stop (suppressing the error for expected failures on fatal errors).
            # Raises an error if the name is already defined on the class.
            # Add the device param kwarg if the test needs device or devices.
            param_kwargs = {} if param_kwargs is None else param_kwargs
            test_sig_params = inspect.signature(test).parameters
            if "device" in test_sig_params or "devices" in test_sig_params:
                device_arg: str = cls._init_and_get_primary_device()
                if hasattr(test, "num_required_devices"):
                    device_arg = cls.get_all_devices()
                _update_param_kwargs(param_kwargs, "device", device_arg)

            # Apply decorators based on param kwargs.
            for decorator in decorator_fn(param_kwargs):
                test = decorator(test)

            # Constructs the test
            @wraps(test)
            def instantiated_test(self, param_kwargs=param_kwargs):
                # [FUNCTION DOC] HELPER: The generated test method. Saves precision settings,
                # applies overrides for the dtype, calls the template test with the stored
                # keyword arguments and returns its result. On RuntimeError it sets the
                # stop-suite flag from _should_stop_test_suite and re-raises (or returns quietly
                # for an expected failure when stopping). Always restores the saved settings.
                # KNOWN ISSUE: _should_stop_test_suite is not defined on the base class, HPU or
                # PrivateUse1 bases, so a RuntimeError there raises AttributeError instead.
                # Sets precision and runs test
                # Note: precision is reset after the test is run
                guard_precision = self.precision
                guard_rel_tol = self.rel_tol
                try:
                    self._apply_precision_override_for_test(test, param_kwargs)
                    result = test(self, **param_kwargs)
                except RuntimeError as rte:
                    # check if rte should stop entire test suite.
                    self._stop_test_suite = self._should_stop_test_suite()
                    # Check if test has been decorated with `@expectedFailure`
                    # Using `__unittest_expecting_failure__` attribute, see
                    # https://github.com/python/cpython/blob/ffa505b580464/Lib/unittest/case.py#L164
                    # In that case, make it fail with "unexpected success" by suppressing exception
                    if (
                        getattr(test, "__unittest_expecting_failure__", False)
                        and self._stop_test_suite
                    ):
                        import sys

                        print(
                            "Suppressing fatal exception to trigger unexpected success",
                            file=sys.stderr,
                        )
                        return
                    # raise the runtime error as is for the test suite to record.
                    raise rte
                finally:
                    self.precision = guard_precision
                    self.rel_tol = guard_rel_tol

                return result

            if hasattr(cls, name):
                raise AssertionError(f"Redefinition of test {name}")
            setattr(cls, name, instantiated_test)

        def default_parametrize_fn(test, generic_cls, device_cls):
            # [FUNCTION DOC] HELPER: Default parametrization; yields a single variant with the
            # test unchanged, empty suffix, no keyword arguments, and no extra decorators.
            # By default, no parametrization is needed.
            yield (test, "", {}, lambda _: [])

        # Parametrization decorators set the parametrize_fn attribute on the test.
        parametrize_fn = getattr(test, "parametrize_fn", default_parametrize_fn)

        # If one of the @dtypes* decorators is present, also parametrize over the dtypes set by it.
        dtypes = cls._get_dtypes(test)
        if dtypes is not None:
            if generic_cls is not None:
                dtypes = tuple(
                    dtype
                    for dtype in dtypes
                    if not cls._should_exclude(
                        generic_cls.__name__, test_name=name, dtype_variant=dtype
                    )
                )

            if not dtypes:
                return

            def dtype_parametrize_fn(test, generic_cls, device_cls, dtypes=dtypes):
                # [FUNCTION DOC] HELPER: Yields one variant per dtype, with an empty suffix (the
                # dtype name is appended later, after the device) and the dtype as a keyword.
                for dtype in dtypes:
                    param_kwargs: dict[str, Any] = {}
                    _update_param_kwargs(param_kwargs, "dtype", dtype)

                    # Note that an empty test suffix is set here so that the dtype can be appended
                    # later after the device.
                    yield (test, "", param_kwargs, lambda _: [])

            parametrize_fn = compose_parametrize_fns(
                dtype_parametrize_fn, parametrize_fn
            )

        # Instantiate the parametrized tests.
        for (
            test,  # noqa: B020
            test_suffix,
            param_kwargs,
            decorator_fn,
        ) in parametrize_fn(test, generic_cls, cls):
            test_suffix = "" if test_suffix == "" else "_" + test_suffix
            cls_device_type = (
                cls.device_type
                if cls.device_type != "privateuse1"
                else torch._C._get_privateuse1_backend_name()
            )
            device_suffix = "_" + cls_device_type

            # Note: device and dtype suffix placement
            # Special handling here to place dtype(s) after device according to test name convention.
            dtype_kwarg = None
            if "dtype" in param_kwargs or "dtypes" in param_kwargs:
                dtype_kwarg = (
                    param_kwargs["dtypes"]
                    if "dtypes" in param_kwargs
                    else param_kwargs["dtype"]
                )
            test_name = (
                f"{name}{test_suffix}{device_suffix}{_dtype_test_suffix(dtype_kwarg)}"
            )
            validate_test_name(test_name)

            instantiate_test_helper(
                cls=cls,
                name=test_name,
                test=test,
                param_kwargs=param_kwargs,
                decorator_fn=decorator_fn,
            )

    def run(self, result=None):
        # [FUNCTION DOC] Runs the test through the parent implementation, then asks the result
        # object to stop the whole run if the stop-suite flag was set by a fatal error.
        # KNOWN ISSUE: with the default result of None, stopping would call a method on None.
        super().run(result=result)
        # Early terminate test if _stop_test_suite is set.
        if self._stop_test_suite:
            result.stop()


class CPUTestBase(DeviceTypeTestBase):
    # [CLASS DOC] Test base for the CPU device type (device_type "cpu"); uses the default
    # primary device ("cpu") and never stops the suite on errors.
    device_type = "cpu"

    # No critical error should stop CPU test suite
    def _should_stop_test_suite(self):
        # [FUNCTION DOC] HELPER: Always returns false; no CPU error is fatal for the suite.
        return False


class CUDATestBase(DeviceTypeTestBase):
    # [CLASS DOC] Test base for CUDA (device_type "cuda"). Class state filled by setUpClass:
    # primary_device (current CUDA device string), no_magma, no_cudnn and cudnn_version.
    # Enables the CUDA memory-leak check and non-default-stream mode for its tests.
    device_type = "cuda"
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True
    primary_device: ClassVar[str]
    cudnn_version: ClassVar[Any]
    no_magma: ClassVar[bool]
    no_cudnn: ClassVar[bool]

    def has_cudnn(self):
        # [FUNCTION DOC] Returns true if cuDNN was found usable by setUpClass (inverse of the
        # no_cudnn flag). Only valid after setUpClass has run.
        return not self.no_cudnn

    @classmethod
    def get_primary_device(cls):
        # [FUNCTION DOC] Classmethod: returns the primary CUDA device string chosen in
        # setUpClass (an attribute that does not exist before it runs).
        return cls.primary_device

    @classmethod
    def get_all_devices(cls):
        # [FUNCTION DOC] Classmethod: returns the primary device string followed by the strings
        # of all other visible CUDA devices (by index), so the primary is first.
        primary_device_idx = int(cls.get_primary_device().split(":")[1])
        num_devices = torch.cuda.device_count()

        prim_device = cls.get_primary_device()
        cuda_str = "cuda:{0}"
        non_primary_devices = [
            cuda_str.format(idx)
            for idx in range(num_devices)
            if idx != primary_device_idx
        ]
        return [prim_device] + non_primary_devices

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Classmethod: initializes CUDA by allocating a small tensor, then
        # records on the class whether MAGMA is missing, whether cuDNN is unusable (and its
        # version if usable), and the current CUDA device as the primary device string.
        # Side effects: initializes the CUDA context and sets class attributes.
        # has_magma shows up after cuda is initialized
        t = torch.ones(1).cuda()
        cls.no_magma = not torch.cuda.has_magma

        # Determines if cuDNN is available and its version
        cls.no_cudnn = not torch.backends.cudnn.is_acceptable(t)
        cls.cudnn_version = None if cls.no_cudnn else torch.backends.cudnn.version()

        # Acquires the current device as the primary (test) device
        cls.primary_device = f"cuda:{torch.cuda.current_device()}"


# See Note [Lazy Tensor tests in device agnostic testing]
lazy_ts_backend_init = False


class LazyTestBase(DeviceTypeTestBase):
    # [CLASS DOC] Test base for the lazy tensor device type (device_type "lazy"); only added
    # to the generated classes when include_lazy is requested. Initializes the TorchScript
    # lazy backend once per process, tracked by the module-level flag lazy_ts_backend_init.
    device_type = "lazy"

    def _should_stop_test_suite(self):
        # [FUNCTION DOC] HELPER: Always returns false; no lazy error is fatal for the suite.
        return False

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Classmethod: imports the lazy modules and, the first time only,
        # connects the TorchScript backend to the lazy key (global flag set to true).
        import torch._lazy
        import torch._lazy.metrics
        import torch._lazy.ts_backend

        global lazy_ts_backend_init
        if not lazy_ts_backend_init:
            # Need to connect the TS backend to lazy key before running tests
            torch._lazy.ts_backend.init()
            lazy_ts_backend_init = True


class MPSTestBase(DeviceTypeTestBase):
    # [CLASS DOC] Test base for Apple MPS (device_type "mps"); single device, primary device
    # fixed to "mps:0" by setUpClass. Not enabled by default in generic instantiation.
    device_type = "mps"
    primary_device: ClassVar[str]

    @classmethod
    def get_primary_device(cls):
        # [FUNCTION DOC] Classmethod: returns the primary device string set by setUpClass.
        return cls.primary_device

    @classmethod
    def get_all_devices(cls):
        # [FUNCTION DOC] Classmethod: returns a list holding only the primary device, since one
        # MPS device is supported.
        # currently only one device is supported on MPS backend
        prim_device = cls.get_primary_device()
        return [prim_device]

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Classmethod: sets the primary device to the fixed string "mps:0".
        cls.primary_device = "mps:0"

    def _should_stop_test_suite(self):
        # [FUNCTION DOC] HELPER: Always returns false; no MPS error is fatal for the suite.
        return False


class XPUTestBase(DeviceTypeTestBase):
    # [CLASS DOC] Test base for Intel XPU (device_type "xpu"); primary device is the current
    # XPU device chosen in setUpClass. Only added to generated classes when allow_xpu is set.
    device_type = "xpu"
    primary_device: ClassVar[str]

    @classmethod
    def get_primary_device(cls):
        # [FUNCTION DOC] Classmethod: returns the primary device string set by setUpClass.
        return cls.primary_device

    @classmethod
    def get_all_devices(cls):
        # [FUNCTION DOC] Classmethod: returns the primary device string followed by all other
        # XPU devices by index.
        # KNOWN ISSUE: the comment below about MPS was copied from the MPS class and does not
        # match this code, which supports several devices.
        # currently only one device is supported on MPS backend
        primary_device_idx = int(cls.get_primary_device().split(":")[1])
        num_devices = torch.xpu.device_count()

        prim_device = cls.get_primary_device()
        xpu_str = "xpu:{0}"
        non_primary_devices = [
            xpu_str.format(idx)
            for idx in range(num_devices)
            if idx != primary_device_idx
        ]
        return [prim_device] + non_primary_devices

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Classmethod: sets the primary device to the current XPU device.
        cls.primary_device = f"xpu:{torch.xpu.current_device()}"

    def _should_stop_test_suite(self):
        # [FUNCTION DOC] HELPER: Always returns false; no XPU error is fatal for the suite.
        return False


class HPUTestBase(DeviceTypeTestBase):
    # [CLASS DOC] Test base for Habana HPU (device_type "hpu"); primary device fixed to
    # "hpu:0". Added to the generated classes when HPU support is detected.
    # KNOWN ISSUE: defines no _should_stop_test_suite, which generated tests call after a
    # RuntimeError.
    device_type = "hpu"
    primary_device: ClassVar[str]

    @classmethod
    def get_primary_device(cls):
        # [FUNCTION DOC] Classmethod: returns the primary device string set by setUpClass.
        return cls.primary_device

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Classmethod: sets the primary device to the fixed string "hpu:0".
        cls.primary_device = "hpu:0"


class PrivateUse1TestBase(DeviceTypeTestBase):
    # [CLASS DOC] Test base for an out-of-tree backend registered as PrivateUse1. device_type
    # starts as "privateuse1" and is overwritten in setUpClass with the registered backend
    # name; device_mod holds that backend's torch module. Restriction bypass defaults to false.
    # KNOWN ISSUE: defines no _should_stop_test_suite, which generated tests call after a
    # RuntimeError.
    primary_device: ClassVar[str]
    device_mod = None
    device_type = "privateuse1"
    bypass_device_restrictions = False

    @classmethod
    def get_primary_device(cls):
        # [FUNCTION DOC] Classmethod: returns the primary device string set by setUpClass.
        return cls.primary_device

    @classmethod
    def get_all_devices(cls):
        # [FUNCTION DOC] Classmethod: returns the primary device followed by the other devices
        # of the backend module (by index), named with the backend's device type.
        primary_device_idx = int(cls.get_primary_device().split(":")[1])
        num_devices = cls.device_mod.device_count()
        prim_device = cls.get_primary_device()
        device_str = f"{cls.device_type}:{{0}}"
        non_primary_devices = [
            device_str.format(idx)
            for idx in range(num_devices)
            if idx != primary_device_idx
        ]
        return [prim_device] + non_primary_devices

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Classmethod: sets device_type to the registered backend name, looks up
        # the backend's module on torch, and sets the primary device from its current device.
        # Raises an error if torch has no module for that backend name.
        cls.device_type = torch._C._get_privateuse1_backend_name()
        cls.device_mod = getattr(torch, cls.device_type, None)
        if cls.device_mod is None:
            raise AssertionError(
                f"torch has no module of `{cls.device_type}`, you should register "
                "a module by `torch._register_device_module`."
            )
        cls.primary_device = f"{cls.device_type}:{cls.device_mod.current_device()}"


# Adds available device-type-specific test base classes
def get_device_type_test_bases():
    # [FUNCTION DOC] Builds the default list of test base classes. In Sandcastle/FB builds:
    # CUDA when remote GPU is present and no sanitizer or MTIA is active, else CPU only.
    # Elsewhere: CPU, plus CUDA if available, plus PrivateUse1 if a backend is registered.
    # Returns the list; no side effects.
    # set type to List[Any] due to mypy list-of-union issue:
    # https://github.com/python/mypy/issues/3351
    test_bases: list[Any] = []

    if IS_SANDCASTLE or IS_FBCODE:
        if IS_REMOTE_GPU:
            # Skip if sanitizer is enabled or we're on MTIA machines
            if (
                not TEST_WITH_ASAN
                and not TEST_WITH_TSAN
                and not TEST_WITH_UBSAN
                and not TEST_WITH_MTIA
            ):
                test_bases.append(CUDATestBase)
        else:
            test_bases.append(CPUTestBase)
    else:
        test_bases.append(CPUTestBase)
        if torch.cuda.is_available():
            test_bases.append(CUDATestBase)

        if _is_privateuse1_backend_available():
            test_bases.append(PrivateUse1TestBase)
        # Disable MPS testing in generic device testing temporarily while we're
        # ramping up support.
        # elif torch.backends.mps.is_available():
        #   test_bases.append(MPSTestBase)

    return test_bases


device_type_test_bases = get_device_type_test_bases()


def filter_desired_device_types(device_type_test_bases, except_for=None, only_for=None):
    # [FUNCTION DOC] Filters test base classes by device type name. except_for removes bases
    # whose device type is listed; only_for keeps just the listed ones; each may be a string
    # or list. If a PrivateUse1 backend is registered, its real name in the filter lists is
    # mapped to "privateuse1" before comparing. Returns a new list.
    # Raises an error if a device appears in both lists.
    # KNOWN ISSUE: a plain string passed for a list-typed argument is not split, and the
    # membership test then becomes a substring test.
    # device type cannot appear in both except_for and only_for
    intersect = set(except_for if except_for else []) & set(
        only_for if only_for else []
    )
    if intersect:
        raise AssertionError(
            f"device ({intersect}) appeared in both except_for and only_for"
        )

    # Replace your privateuse1 backend name with 'privateuse1'
    # This handles the case where PrivateUse1TestBase.device_type has been
    # changed from "privateuse1" to the actual backend name (e.g., "openreg")
    # by setUpClass being called during previous instantiate_device_type_tests calls
    if _is_privateuse1_backend_available():
        privateuse1_backend_name = torch._C._get_privateuse1_backend_name()

        def func_replace(x: str) -> str:
            # [FUNCTION DOC] HELPER: Replaces the registered backend name in the string with
            # "privateuse1".
            return x.replace(privateuse1_backend_name, "privateuse1")

        except_for = (
            ([func_replace(x) for x in except_for] if except_for is not None else None)
            if not isinstance(except_for, str)
            else func_replace(except_for)
        )
        only_for = (
            ([func_replace(x) for x in only_for] if only_for is not None else None)
            if not isinstance(only_for, str)
            else func_replace(only_for)
        )
    else:

        def func_replace(x: str) -> str:
            # [FUNCTION DOC] HELPER: Identity; no PrivateUse1 backend, so nothing to rename.
            return x

    if except_for:
        device_type_test_bases = filter(
            lambda x: func_replace(x.device_type) not in except_for,
            device_type_test_bases,
        )
    if only_for:
        device_type_test_bases = filter(
            lambda x: func_replace(x.device_type) in only_for,
            device_type_test_bases,
        )

    return list(device_type_test_bases)


# Note [How to extend DeviceTypeTestBase to add new test device]
# The following logic optionally allows downstream projects like pytorch/xla to
# add more test devices.
# Instructions:
#  - Add a python file (e.g. pytorch/xla/test/pytorch_test_base.py) in downstream project.
#    - Inside the file, one should inherit from `DeviceTypeTestBase` class and define
#      a new DeviceTypeTest class (e.g. `XLATestBase`) with proper implementation of
#      `instantiate_test` method.
#    - DO NOT import common_device_type inside the file.
#      `runpy.run_path` with `globals()` already properly setup the context so that
#      `DeviceTypeTestBase` is already available.
#    - Set a top-level variable `TEST_CLASS` equal to your new class.
#      E.g. TEST_CLASS = XLATensorBase
#  - To run tests with new device type, set `TORCH_TEST_DEVICE` env variable to path
#    to this file. Multiple paths can be separated by `:`.
# See pytorch/xla/test/pytorch_test_base.py for a more detailed example.
_TORCH_TEST_DEVICES = os.environ.get("TORCH_TEST_DEVICES", None)
if _TORCH_TEST_DEVICES:
    for path in _TORCH_TEST_DEVICES.split(":"):
        # runpy (a stdlib module) lacks annotations
        mod = runpy.run_path(path, init_globals=globals())  # type: ignore[func-returns-value]
        device_type_test_bases.append(mod["TEST_CLASS"])


PYTORCH_CUDA_MEMCHECK = os.getenv("PYTORCH_CUDA_MEMCHECK", "0") == "1"

PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY = "PYTORCH_TESTING_DEVICE_ONLY_FOR"
PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY = "PYTORCH_TESTING_DEVICE_EXCEPT_FOR"
PYTORCH_TESTING_DEVICE_FOR_CUSTOM_KEY = "PYTORCH_TESTING_DEVICE_FOR_CUSTOM"


def get_desired_device_type_test_bases(
    except_for=None, only_for=None, include_lazy=False, allow_mps=False, allow_xpu=False
):
    # [FUNCTION DOC] Returns the test base classes to instantiate for a test file. Starts from
    # the module-level list, optionally adds MPS (allow_mps and available) and XPU (allow_xpu
    # and available), always adds HPU if available, filters by except_for/only_for, optionally
    # appends the lazy base (include_lazy; prints a notice instead in FB builds). Then adds
    # device types named in the PYTORCH_TESTING_DEVICE_FOR_CUSTOM environment variable and
    # applies the PYTORCH_TESTING_DEVICE_ONLY_FOR / _EXCEPT_FOR comma-separated variables.
    # Returns a list; the custom-device step goes through a set so order is not guaranteed.
    # allow callers to specifically opt tests into being tested on MPS, similar to `include_lazy`
    test_bases = device_type_test_bases.copy()
    if allow_mps and TEST_MPS and MPSTestBase not in test_bases:
        test_bases.append(MPSTestBase)
    if allow_xpu and TEST_XPU and XPUTestBase not in test_bases:
        test_bases.append(XPUTestBase)
    if TEST_HPU and HPUTestBase not in test_bases:
        test_bases.append(HPUTestBase)
    # Filter out the device types based on user inputs
    desired_device_type_test_bases = filter_desired_device_types(
        test_bases, except_for, only_for
    )
    if include_lazy:
        # Note [Lazy Tensor tests in device agnostic testing]
        # Right now, test_view_ops.py runs with LazyTensor.
        # We don't want to opt every device-agnostic test into using the lazy device,
        # because many of them will fail.
        # So instead, the only way to opt a specific device-agnostic test file into
        # lazy tensor testing is with include_lazy=True
        if IS_FBCODE:
            print(
                "TorchScript backend not yet supported in FBCODE/OVRSOURCE builds",
                file=sys.stderr,
            )
        else:
            desired_device_type_test_bases.append(LazyTestBase)

    def split_if_not_empty(x: str):
        # [FUNCTION DOC] HELPER: Splits a comma-separated string into a list; empty gives [].
        return x.split(",") if x else []

    # run some cuda testcases on other devices if available
    # Usage:
    # export PYTORCH_TESTING_DEVICE_FOR_CUSTOM=privateuse1
    env_custom_only_for = split_if_not_empty(
        os.getenv(PYTORCH_TESTING_DEVICE_FOR_CUSTOM_KEY, "")
    )
    if env_custom_only_for:
        desired_device_type_test_bases += filter(
            lambda x: x.device_type in env_custom_only_for, test_bases
        )
        desired_device_type_test_bases = list(set(desired_device_type_test_bases))

    # Filter out the device types based on environment variables if available
    # Usage:
    # export PYTORCH_TESTING_DEVICE_ONLY_FOR=cuda,cpu
    # export PYTORCH_TESTING_DEVICE_EXCEPT_FOR=xla
    env_only_for = split_if_not_empty(
        os.getenv(PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY, "")
    )
    env_except_for = split_if_not_empty(
        os.getenv(PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY, "")
    )

    return filter_desired_device_types(
        desired_device_type_test_bases, env_except_for, env_only_for
    )


# Adds 'instantiated' device-specific test cases to the given scope.
# The tests in these test cases are derived from the generic tests in
# generic_test_class. This function should be used instead of
# instantiate_parametrized_tests() if the test class contains
# device-specific tests (NB: this supports additional @parametrize usage).
#
# See note "Writing Test Templates"
# TODO: remove "allow_xpu" option after Intel GPU support all test case instantiate by this function.
def instantiate_device_type_tests(
    generic_test_class,
    scope,
    except_for=None,
    only_for=None,
    include_lazy=False,
    allow_mps=False,
    allow_xpu=False,
):
    # [FUNCTION DOC] Replaces a generic test template class in the given module scope with one
    # generated class per desired device type (named template name plus upper-case device
    # type). Each generated class derives from the device base and the template; template
    # test methods are expanded via the base's instantiate_test (skipping excluded ones),
    # other members are copied over, and setUpClass/tearDownClass call both the base's and the
    # template's versions. Generated classes get the template's module name and are added to
    # scope. Finally the generic test methods are deleted from the template.
    # Side effects: mutates scope (deletes the template name) and the template class.
    # Raises a KeyError if the template is not in scope.
    # Removes the generic test class from its enclosing scope so its tests
    # are not discoverable.
    del scope[generic_test_class.__name__]

    generic_members = set(generic_test_class.__dict__.keys())
    generic_tests = [x for x in generic_members if x.startswith("test")]

    # Creates device-specific test cases
    for base in get_desired_device_type_test_bases(
        except_for, only_for, include_lazy, allow_mps, allow_xpu
    ):
        # Skip the entire class
        if base._should_exclude(generic_test_class.__name__):
            continue

        class_name = generic_test_class.__name__ + base.device_type.upper()

        # type set to Any and suppressed due to unsupported runtime class:
        # https://github.com/python/mypy/wiki/Unsupported-Python-Features
        device_type_test_class: Any = type(class_name, (base, generic_test_class), {})

        # Arrange for setUpClass and tearDownClass methods defined both in the test template
        # class and in the generic base to be called. This allows device-parameterized test
        # classes to support setup and teardown.
        # NB: This should be done before instantiate_test() is called as that invokes setup.
        @classmethod
        def _setUpClass(cls):
            # [FUNCTION DOC] HELPER: Combined class setup; first runs the device base's
            # setUpClass (sets primary device), then the template class's setUpClass on cls.
            # This should always be called, whether or not the test class invokes
            # super().setUpClass(), to set the primary device.
            base.setUpClass()
            # We want to call the @classmethod defined in the generic base, but pass
            # it the device-specific class object (cls), hence the __func__ call.
            generic_test_class.setUpClass.__func__(cls)

        @classmethod
        def _tearDownClass(cls):
            # [FUNCTION DOC] HELPER: Combined class teardown; runs the template class's
            # tearDownClass on cls first, then the device base's tearDownClass.
            # We want to call the @classmethod defined in the generic base, but pass
            # it the device-specific class object (cls), hence the __func__ call.
            generic_test_class.tearDownClass.__func__(cls)
            base.tearDownClass()

        device_type_test_class.setUpClass = _setUpClass
        device_type_test_class.tearDownClass = _tearDownClass

        for name in generic_members:
            if name in generic_tests:  # Instantiates test member
                # Skip the specified methods.
                if base._should_exclude(generic_test_class.__name__, test_name=name):
                    continue
                test = getattr(generic_test_class, name)
                # XLA-compat shim (XLA's instantiate_test takes doesn't take generic_cls)
                sig = inspect.signature(device_type_test_class.instantiate_test)
                if len(sig.parameters) == 3:
                    # Instantiates the device-specific tests
                    device_type_test_class.instantiate_test(
                        name, copy.deepcopy(test), generic_cls=generic_test_class
                    )
                else:
                    device_type_test_class.instantiate_test(name, copy.deepcopy(test))
            # Ports non-test member. Setup / teardown have already been handled above
            elif name not in device_type_test_class.__dict__:
                nontest = getattr(generic_test_class, name)
                setattr(device_type_test_class, name, nontest)

        # Mimics defining the instantiated class in the caller's file
        # by setting its module to the given class's and adding
        # the module to the given scope.
        # This lets the instantiated class be discovered by unittest.
        device_type_test_class.__module__ = generic_test_class.__module__
        scope[class_name] = device_type_test_class

    # Delete the generic form of the test functions (e.g. TestFoo.test_bar()) so they're
    # not discoverable. This mutates the original class (TestFoo), which was removed from
    # scope above. At this point, device-specific tests (e.g. TestFooCUDA.test_bar_cuda)
    # have already been created and the generic forms are no longer needed.
    for name in generic_tests:
        delattr(generic_test_class, name)


# Category of dtypes to run an OpInfo-based test for
# Example use: @ops(dtype=OpDTypes.supported)
#
# There are 7 categories:
# - supported: Every dtype supported by the operator. Use for exhaustive
#              testing of all dtypes.
# - unsupported: Run tests on dtypes not supported by the operator. e.g. for
#                testing the operator raises an error and doesn't crash.
# - supported_backward: Every dtype supported by the operator's backward pass.
# - unsupported_backward: Run tests on dtypes not supported by the operator's backward pass.
# - any_one: Runs a test for one dtype the operator supports. Prioritizes dtypes the
#     operator supports in both forward and backward.
# - none: Useful for tests that are not dtype-specific. No dtype will be passed to the test
#         when this is selected.
# - any_common_cpu_cuda_one: Pick a dtype that supports both CPU and CUDA.
class OpDTypes(Enum):
    # [CLASS DOC] Enumeration of which dtype set an @ops test should run over: supported,
    # unsupported, supported_backward, unsupported_backward, any_one, none, and
    # any_common_cpu_cuda_one. Values are integers 0 to 6; supported is the default.
    supported = 0  # Test all supported dtypes (default)
    unsupported = 1  # Test only unsupported dtypes
    supported_backward = 2  # Test all supported backward dtypes
    unsupported_backward = 3  # Test only unsupported backward dtypes
    any_one = 4  # Test precisely one supported dtype
    none = 5  # Instantiate no dtype variants (no dtype kwarg needed)
    any_common_cpu_cuda_one = (
        6  # Test precisely one supported dtype that is common to both cuda and cpu
    )


# Arbitrary order
ANY_DTYPE_ORDER = (
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
    torch.float16,
    torch.bfloat16,
    torch.long,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)


def _serialize_sample(sample_input):
    # [FUNCTION DOC] HELPER: Returns a string for a sample input, using its summary method if
    # it has one, else its string conversion.
    # NB: For OpInfos, SampleInput.summary() prints in a cleaner way.
    if getattr(sample_input, "summary", None) is not None:
        return sample_input.summary()
    return str(sample_input)


SkipSpec = namedtuple(
    "SkipSpec",
    [
        "op_name",
        "variant_name",
        "device_type",
        "dtypes",
        "expected_failure",
    ],
)


def xfail(op_name, variant_name="", *, device_type=None, dtypes=None):
    # [FUNCTION DOC] Creates a SkipSpec marking the op (and optional variant) as an expected
    # failure, optionally limited to a device type and dtypes. Pure; returns the spec.
    return SkipSpec(op_name, variant_name, device_type, dtypes, True)


def skip(op_name, variant_name="", *, device_type=None, dtypes=None):
    # [FUNCTION DOC] Creates a SkipSpec marking the op (and optional variant) as skipped,
    # optionally limited to a device type and dtypes. Pure; returns the spec.
    return SkipSpec(op_name, variant_name, device_type, dtypes, False)


def skipOps(to_skip):
    # [FUNCTION DOC] Decorator factory taking a list of skip specs (SkipSpec tuples, or
    # objects with the same fields and an optional custom decorator). The returned decorator
    # records, on the test function, a mapping from op full name (op plus optional variant) to
    # a list of DecorateInfo entries (skip or expected-failure) scoped to the test's class and
    # method name taken from its qualified name. The ops machinery applies them later.
    def wrapped(fn):
        # [FUNCTION DOC] HELPER: Does the work of skipOps for one test function: merges the new
        # entries into any existing override mapping stored on the function, stores it back
        # and returns the same function. Imports DecorateInfo lazily.
        from torch.testing._internal.opinfo.core import DecorateInfo

        parts = fn.__qualname__.split(".")
        test_name = parts[-1]
        cls_name = parts[-2] if len(parts) >= 2 else ""
        overrides = getattr(fn, "_op_overrides", {})
        for skip_spec in to_skip:
            if hasattr(skip_spec, "op_name"):
                op_name = skip_spec.op_name
                variant_name = skip_spec.variant_name
                device_type = skip_spec.device_type
                dtypes = skip_spec.dtypes
                if hasattr(skip_spec, "decorator"):
                    decorator_callable = skip_spec.decorator
                else:
                    expected_failure = skip_spec.expected_failure
                    decorator_callable = (
                        unittest.expectedFailure
                        if expected_failure
                        else unittest.skip("Skipped!")
                    )
            else:
                op_name, variant_name, device_type, dtypes, expected_failure = skip_spec
                decorator_callable = (
                    unittest.expectedFailure
                    if expected_failure
                    else unittest.skip("Skipped!")
                )
            full_name = f"{op_name}.{variant_name}" if variant_name else op_name
            decorator = DecorateInfo(
                decorator_callable,
                cls_name,
                test_name,
                device_type=device_type,
                dtypes=dtypes,
            )
            overrides.setdefault(full_name, []).append(decorator)
        fn._op_overrides = overrides
        return fn

    return wrapped


# Decorator that defines the OpInfos a test template should be instantiated for.
#
# Example usage:
#
# @ops(unary_ufuncs)
# def test_numerics(self, device, dtype, op):
#   <test_code>
#
# This will instantiate variants of test_numerics for each given OpInfo,
# on each device the OpInfo's operator supports, and for every dtype supported by
# that operator. There are a few caveats to the dtype rule, explained below.
#
# The @ops decorator can accept two
# additional arguments, "dtypes" and "allowed_dtypes". If "dtypes" is specified
# then the test variants are instantiated for those dtypes, regardless of
# what the operator supports. If given "allowed_dtypes" then test variants
# are instantiated only for the intersection of allowed_dtypes and the dtypes
# they would otherwise be instantiated with. That is, allowed_dtypes composes
# with the options listed above and below.
#
# The "dtypes" argument can also accept additional values (see OpDTypes above):
#   OpDTypes.supported - the test is instantiated for all dtypes the operator
#     supports
#   OpDTypes.unsupported - the test is instantiated for all dtypes the operator
#     doesn't support
#   OpDTypes.supported_backward - the test is instantiated for all dtypes the
#     operator's gradient formula supports
#   OpDTypes.unsupported_backward - the test is instantiated for all dtypes the
#     operator's gradient formula doesn't support
#   OpDTypes.any_one - the test is instantiated for one dtype the
#     operator supports. The dtype supports forward and backward if possible.
#   OpDTypes.none - the test is instantiated without any dtype. The test signature
#     should not include a dtype kwarg in this case.
#   OpDTypes.any_common_cpu_cuda_one - the test is instantiated for a dtype
#     that supports both CPU and CUDA.
#
# These options allow tests to have considerable control over the dtypes
#   they're instantiated for.


class ops(_TestParametrizer):
    # [CLASS DOC] Decorator that parametrizes a test template over OpInfos and their dtypes.
    # State: op_list (list of OpInfos), opinfo_dtypes (a dtype sequence or an OpDTypes
    # choice), allowed_dtypes (optional set filter), skip_if_dynamo (wrap tests so they skip
    # under Dynamo). Used with instantiate_device_type_tests only; the variant names combine
    # op name, device and dtype.
    def __init__(
        self,
        op_list,
        *,
        dtypes: OpDTypes | Sequence[torch.dtype] = OpDTypes.supported,
        allowed_dtypes: Sequence[torch.dtype] | None = None,
        skip_if_dynamo=True,
    ):
        # [FUNCTION DOC] Stores the op list (copied into a list, so a generator is consumed
        # once), the dtype selection, the allowed dtypes (as a set, or None) and the flag.
        self.op_list = list(op_list)
        self.opinfo_dtypes = dtypes
        self.allowed_dtypes = (
            set(allowed_dtypes) if allowed_dtypes is not None else None
        )
        self.skip_if_dynamo = skip_if_dynamo

    def _parametrize_test(self, test, generic_cls, device_cls):
        # [FUNCTION DOC] Generator yielding (wrapped test, name, keyword args, decorator
        # function) for every op and chosen dtype on the given device class. First applies the
        # device class's op allowlist, then its op overrides (mutating this object's op list).
        # Dtypes per op depend on opinfo_dtypes, then are intersected with allowed_dtypes.
        # The test is wrapped to attach tracked-input info to failures, optionally skipped
        # under Dynamo; matching sample skip rules are copied onto the wrapper.
        # Raises an error if device_cls is None, on an unknown dtype choice, and (unless the
        # DSL restriction environment variable is set) if the op list was empty.
        # KNOWN ISSUE: the empty-result fallbacks assign an empty mapping where a set is
        # expected, which breaks the later set intersection when allowed_dtypes is given.
        """Parameterizes the given test function across each op and its associated dtypes."""
        if device_cls is None:
            raise RuntimeError(
                "The @ops decorator is only intended to be used in a device-specific "
                "context; use it with instantiate_device_type_tests() instead of "
                "instantiate_parametrized_tests()"
            )

        # Order matters: op_allowlist filters first, then op_overrides adds decorators
        # This ensures op_overrides only applies to ops that passed the op_allowlist filter
        device_cls._apply_op_allowlist(self)
        device_cls._apply_op_overrides(self, test)
        op = check_exhausted_iterator = object()
        for op in self.op_list:
            # Determine the set of dtypes to use.
            dtypes: set[torch.dtype] | set[None]
            if isinstance(self.opinfo_dtypes, Sequence):
                dtypes = set(self.opinfo_dtypes)
            elif self.opinfo_dtypes == OpDTypes.unsupported_backward:
                dtypes = set(get_all_dtypes()).difference(
                    op.supported_backward_dtypes(device_cls.device_type)
                )
            elif self.opinfo_dtypes == OpDTypes.supported_backward:
                dtypes = op.supported_backward_dtypes(device_cls.device_type)
            elif self.opinfo_dtypes == OpDTypes.unsupported:
                dtypes = set(get_all_dtypes()).difference(
                    op.supported_dtypes(device_cls.device_type)
                )
            elif self.opinfo_dtypes == OpDTypes.supported:
                dtypes = set(op.supported_dtypes(device_cls.device_type))
            elif self.opinfo_dtypes == OpDTypes.any_one:
                # Tries to pick a dtype that supports both forward or backward
                supported = set(op.supported_dtypes(device_cls.device_type))
                supported_backward = op.supported_backward_dtypes(
                    device_cls.device_type
                )
                supported_both = supported.intersection(supported_backward)
                dtype_set = supported_both if len(supported_both) > 0 else supported
                for dtype in ANY_DTYPE_ORDER:
                    if dtype in dtype_set:
                        dtypes = {dtype}
                        break
                else:
                    dtypes = {}
            elif self.opinfo_dtypes == OpDTypes.any_common_cpu_cuda_one:
                # Tries to pick a dtype that supports both CPU and CUDA
                supported = set(op.dtypes).intersection(op.dtypesIfCUDA)
                if supported:
                    dtypes = {
                        next(dtype for dtype in ANY_DTYPE_ORDER if dtype in supported)
                    }
                else:
                    dtypes = {}

            elif self.opinfo_dtypes == OpDTypes.none:
                dtypes = {None}
            else:
                raise RuntimeError(f"Unknown OpDType: {self.opinfo_dtypes}")

            if self.allowed_dtypes is not None:
                dtypes = dtypes.intersection(self.allowed_dtypes)

            # Construct the test name; device / dtype parts are handled outside.
            # See [Note: device and dtype suffix placement]
            test_name = op.formatted_name

            # Filter sample skips / xfails to only those that apply to the OpInfo.
            # These are defined on the test function via decorators.
            sample_skips_and_xfails = getattr(test, "sample_skips_and_xfails", None)
            if sample_skips_and_xfails is not None:
                sample_skips_and_xfails = [
                    rule
                    for rule in sample_skips_and_xfails
                    if rule.op_match_fn(device_cls.device_type, op)
                ]

            for dtype in dtypes:
                # Construct parameter kwargs to pass to the test.
                param_kwargs = {"op": op}
                _update_param_kwargs(param_kwargs, "dtype", dtype)

                # NOTE: test_wrapper exists because we don't want to apply
                #   op-specific decorators to the original test.
                #   Test-specific decorators are applied to the original test,
                #   however.
                try:

                    @wraps(test)
                    def test_wrapper(*args, **kwargs):
                        # [FUNCTION DOC] HELPER: Calls the original test. Skip signals pass
                        # through; other errors get a message naming the last tracked sample
                        # input when repro printing is enabled. Always clears tracked input.
                        try:
                            return test(*args, **kwargs)
                        except unittest.SkipTest as e:
                            raise e
                        except Exception as e:
                            tracked_input = get_tracked_input()
                            if PRINT_REPRO_ON_FAILURE and tracked_input is not None:
                                e_tracked = Exception(
                                    f"{str(e)}\n\nCaused by {tracked_input.type_desc} "
                                    f"at index {tracked_input.index}: "
                                    f"{_serialize_sample(tracked_input.val)}"
                                )
                                e_tracked._tracked_input = tracked_input  # type: ignore[attr]
                                raise e_tracked from e
                            raise e
                        finally:
                            clear_tracked_input()

                    if self.skip_if_dynamo and not TEST_WITH_TORCHINDUCTOR:
                        test_wrapper = skipIfTorchDynamo(
                            "Policy: we don't run OpInfo tests w/ Dynamo"
                        )(test_wrapper)

                    # Initialize info for the last input seen. This is useful for tracking
                    # down which inputs caused a test failure. Note that TrackedInputIter is
                    # responsible for managing this.
                    test.tracked_input = None

                    decorator_fn = partial(
                        op.get_decorators,
                        generic_cls.__name__,
                        test.__name__,
                        device_cls.device_type,
                        dtype,
                    )

                    if sample_skips_and_xfails is not None:
                        test_wrapper.sample_skips_and_xfails = sample_skips_and_xfails

                    yield (test_wrapper, test_name, param_kwargs, decorator_fn)
                except Exception as ex:
                    # Provides an error message for debugging before rethrowing the exception
                    log.info("Failed to instantiate %s for op %s", test_name, op.name)
                    raise ex
        if op is check_exhausted_iterator:
            # When OPINFO_RESTRICT_TO_DSL narrows op_db to a DSL subset, many
            # per-test op lists legitimately become empty -don't fail collection.
            if os.environ.get("OPINFO_RESTRICT_TO_DSL"):
                return
            raise ValueError(
                "An empty op_list was passed to @ops. "
                "Note that this may result from reuse of a generator."
            )


# Decorator that skips a test if the given condition is true.
# Notes:
#   (1) Skip conditions stack.
#   (2) Skip conditions can be bools or strings. If a string the
#       test base must have defined the corresponding attribute to be False
#       for the test to run. If you want to use a string argument you should
#       probably define a new decorator instead (see below).
#   (3) Prefer the existing decorators to defining the 'device_type' kwarg.
class skipIf:
    # [CLASS DOC] Decorator that skips a test when a condition holds. State: dep (a bool, or
    # the name of a test-class attribute that must be falsy for the test to run), reason (skip
    # message), and device_type (None for all devices, or a name / collection of names the
    # skip is limited to). Conditions stack when several are applied.
    def __init__(self, dep, reason, device_type=None):
        # [FUNCTION DOC] Stores the condition, the skip reason and the optional device filter.
        self.dep = dep
        self.reason = reason
        self.device_type = device_type

    def __call__(self, fn):
        # [FUNCTION DOC] Wraps the test method. At call time, if the test's device type matches
        # the filter (or no filter) and the condition holds (true bool, or a string naming an
        # attribute that is truthy or missing on the test instance), raises a skip with the
        # reason; otherwise runs the test and returns its result.
        @wraps(fn)
        def dep_fn(slf, *args, **kwargs):
            # [FUNCTION DOC] HELPER: The wrapper described by the enclosing call method.
            if (
                self.device_type is None
                or self.device_type == slf.device_type
                or (
                    isinstance(self.device_type, Iterable)
                    and slf.device_type in self.device_type
                )
            ):
                if (isinstance(self.dep, str) and getattr(slf, self.dep, True)) or (
                    isinstance(self.dep, bool) and self.dep
                ):
                    raise unittest.SkipTest(self.reason)

            return fn(slf, *args, **kwargs)

        return dep_fn


# Skips a test on CPU if the condition is true.
class skipCPUIf(skipIf):
    # [CLASS DOC] skipIf restricted to the "cpu" device type.
    def __init__(self, dep, reason):
        # [FUNCTION DOC] Initializes the parent with the device filter fixed to cpu.
        super().__init__(dep, reason, device_type="cpu")


# Skips a test on CUDA if the condition is true.
class skipCUDAIf(skipIf):
    # [CLASS DOC] skipIf restricted to the "cuda" device type.
    def __init__(self, dep, reason):
        # [FUNCTION DOC] Initializes the parent with the device filter fixed to cuda.
        super().__init__(dep, reason, device_type="cuda")


# Skips a test on XPU if the condition is true.
class skipXPUIf(skipIf):
    # [CLASS DOC] skipIf restricted to the "xpu" device type.
    def __init__(self, dep, reason):
        # [FUNCTION DOC] Initializes the parent with the device filter fixed to xpu.
        super().__init__(dep, reason, device_type="xpu")


# Skips a test on XPU or CUDA if the condition is true.
class skipGPUIf(skipIf):
    # [CLASS DOC] skipIf restricted to the GPU device types listed in GPU_TYPES (CUDA, XPU).
    def __init__(self, dep, reason):
        # [FUNCTION DOC] Initializes the parent with the device filter set to GPU_TYPES.
        super().__init__(dep, reason, device_type=GPU_TYPES)


# Skips a test on Lazy if the condition is true.
class skipLazyIf(skipIf):
    # [CLASS DOC] skipIf restricted to the "lazy" device type.
    def __init__(self, dep, reason):
        # [FUNCTION DOC] Initializes the parent with the device filter fixed to lazy.
        super().__init__(dep, reason, device_type="lazy")


# Skips a test on Meta if the condition is true.
class skipMetaIf(skipIf):
    # [CLASS DOC] skipIf restricted to the "meta" device type.
    def __init__(self, dep, reason):
        # [FUNCTION DOC] Initializes the parent with the device filter fixed to meta.
        super().__init__(dep, reason, device_type="meta")


# Skips a test on MPS if the condition is true.
class skipMPSIf(skipIf):
    # [CLASS DOC] skipIf restricted to the "mps" device type.
    def __init__(self, dep, reason):
        # [FUNCTION DOC] Initializes the parent with the device filter fixed to mps.
        super().__init__(dep, reason, device_type="mps")


class skipHPUIf(skipIf):
    # [CLASS DOC] skipIf restricted to the "hpu" device type.
    def __init__(self, dep, reason):
        # [FUNCTION DOC] Initializes the parent with the device filter fixed to hpu.
        super().__init__(dep, reason, device_type="hpu")


# Skips a test on XLA if the condition is true.
class skipXLAIf(skipIf):
    # [CLASS DOC] skipIf restricted to the "xla" device type.
    def __init__(self, dep, reason):
        # [FUNCTION DOC] Initializes the parent with the device filter fixed to xla.
        super().__init__(dep, reason, device_type="xla")


class skipPRIVATEUSE1If(skipIf):
    # [CLASS DOC] skipIf restricted to the registered PrivateUse1 backend's device type name.
    def __init__(self, dep, reason):
        # [FUNCTION DOC] Looks up the registered PrivateUse1 backend name at construction time
        # and initializes the parent with the device filter set to it.
        device_type = torch._C._get_privateuse1_backend_name()
        super().__init__(dep, reason, device_type=device_type)


def _has_sufficient_memory(device, size):
    # [FUNCTION DOC] HELPER: Reports whether the device has at least size bytes free.
    # For cuda/xpu/mtia: false if no accelerator or it is unavailable, true if the current
    # accelerator is a different type, otherwise runs garbage collection, empties the cache and
    # compares against free device memory (CUDA scaled by the per-process memory fraction).
    # For cpu/mps: uses psutil available memory (sizes inflated for sanitizers and s390x),
    # collecting garbage and clearing MPS caches first if short. Raises a skip for xla, unknown
    # device types, or when psutil is missing. Returns a bool.
    device_ = torch.device(device)
    device_type = device_.type
    if device_type in ["cuda", "xpu", "mtia"]:
        acc = torch.accelerator.current_accelerator()
        # Case 1: no accelerator found
        if not acc:
            return False
        # Case 2: accelerator found but not matching device type
        if acc.type != device_type:
            return True
        # Case 3: accelerator found and matching device type but not available
        if not torch.accelerator.is_available():
            return False
        # Case 4: accelerator found and matching device type and available
        gc.collect()
        torch.accelerator.empty_cache()

        if device_.index is None:
            device_ = torch.device(device_type, 0)

        if device_type == "cuda":
            return (
                torch.cuda.memory.mem_get_info(device_)[0]
                * torch.cuda.memory.get_per_process_memory_fraction(device_)
            ) >= size

        if device_type == "xpu":
            return torch.xpu.memory.mem_get_info(device_)[0] >= size

        if device_type == "mtia":
            # MTIA has no mem_get_info; the dram stats dict exposes free_bytes
            # (see torch/csrc bindings / mtia_hooks.cpp).
            return torch.mtia.memory_stats(device_)["dram"]["free_bytes"] >= size

    if device_type == "xla":
        raise unittest.SkipTest("TODO: Memory availability checks for XLA?")

    if device_type not in ["cpu", "mps"]:
        raise unittest.SkipTest("Unknown device type")

    # CPU
    if not HAS_PSUTIL:
        raise unittest.SkipTest("Need psutil to determine if memory is sufficient")

    # The sanitizers have significant memory overheads
    if TEST_WITH_ASAN or TEST_WITH_TSAN or TEST_WITH_UBSAN:
        effective_size = size * 10
    else:
        effective_size = size

    # don't try using all RAM on s390x, leave some for service processes
    if IS_S390X:
        effective_size = effective_size * 2

    if psutil.virtual_memory().available < effective_size:
        gc.collect()
        # Sync and cleanup MPS memory before checking available memory
        if device_type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()

    return psutil.virtual_memory().available >= effective_size


def _parse_size(size):
    # [FUNCTION DOC] HELPER: Converts a size given as a string ending in GB (any case) to bytes
    # (gibibytes times the leading integer); any non-string (number or callable) is returned
    # unchanged. Raises an error for other string suffixes.
    if isinstance(size, str):
        if not size.endswith(("GB", "gb")):
            raise AssertionError(f"only bytes or GB supported, got {size!r}")
        return 1024**3 * int(size[:-2])
    return size


def _test_device(self, device):
    # [FUNCTION DOC] HELPER: Chooses which device to check: the explicit device if given, else
    # the test instance's primary device if it has that method, else its device attribute.
    if device is not None:
        return device
    if hasattr(self, "get_primary_device"):
        return self.get_primary_device()
    return self.device


def largeTensorTest(size, device=None, inductor=TEST_WITH_TORCHINDUCTOR):
    # [FUNCTION DOC] Decorator factory that skips a test when the device lacks memory for the
    # test's total footprint. size is bytes, a "N GB" string, or a callable taking the test
    # arguments. device overrides the checked device. With inductor on, a GPU run with the C++
    # wrapper doubles the size. Raises a skip at call time when memory is insufficient.
    """Skip test if the device has insufficient memory to run the test

    size is the test's total memory footprint, and may be a number of bytes, a
    string of the form "N GB", or a callable. It is not a per-allocation limit;
    for that see `largeMPSBufferTest`.

    If the test is a device generic test, available memory on the primary device will be checked.
    It can also be overridden by the optional `device=` argument.
    In other tests, the `device=` argument needs to be specified.
    """
    size = _parse_size(size)

    def inner(fn):
        # [FUNCTION DOC] HELPER: Decorator returned by largeTensorTest; wraps the test function.
        @wraps(fn)
        def dep_fn(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Resolves the size and device, skips if memory is
            # insufficient, otherwise runs the test and returns its result.
            size_bytes: int = size(self, *args, **kwargs) if callable(size) else size
            _device = _test_device(self, device)

            # If this is running with GPU cpp_wrapper, the autotuning step will generate
            # an additional array of the same size as the input.
            if inductor and torch._inductor.config.cpp_wrapper and _device != "cpu":
                size_bytes *= 2
            if not _has_sufficient_memory(_device, size_bytes):
                raise unittest.SkipTest(f"Insufficient {_device} memory")

            return fn(self, *args, **kwargs)

        return dep_fn

    return inner


def largeMPSBufferTest(size, device=None):
    # [FUNCTION DOC] Decorator factory that skips an MPS test when its largest single
    # allocation (bytes, "N GB" string, or callable) would reach the MPS maximum buffer length.
    # Does nothing on other devices. Raises a skip at call time when too large.
    """Skip test if MPS cannot allocate a single buffer of the given size

    Unlike `largeTensorTest`, size is the largest *individual* allocation the
    test makes, checked against the device's maxBufferLength (a hard cap the
    MPS allocator enforces regardless of how much memory is free). Tests that
    need both a large footprint and a large single buffer should use both
    decorators. This is a no-op on non-MPS devices.
    """
    size = _parse_size(size)

    def inner(fn):
        # [FUNCTION DOC] HELPER: Decorator returned by largeMPSBufferTest; wraps the test.
        @wraps(fn)
        def dep_fn(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: On an MPS device, skips when the size is at or above the
            # maximum buffer length; then runs the test and returns its result.
            _device = _test_device(self, device)
            if torch.device(_device).type == "mps":
                size_bytes = size(self, *args, **kwargs) if callable(size) else size
                # The allocator's check is strict, so match it with >=.
                if size_bytes >= torch._C._mps_maxBufferLength():
                    raise unittest.SkipTest(
                        f"Needs MPS maxBufferLength > {size_bytes} bytes"
                    )

            return fn(self, *args, **kwargs)

        return dep_fn

    return inner


class expectedFailure:
    # [CLASS DOC] Decorator marking a test as expected to fail on a device type, optionally
    # only for one dtype. State: device_type (None matches any) and dtype (None matches any).
    # When matched, the test passes if it raises any exception and fails if it succeeds.
    def __init__(self, device_type, dtype=None):
        # [FUNCTION DOC] Stores the device type and dtype filters.
        self.device_type = device_type
        self.dtype = dtype

    def __call__(self, fn):
        # [FUNCTION DOC] Returns a wrapper around the test function (see the wrapper's notes).
        @wraps(fn)
        def efail_fn(slf, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Determines the test's device type (from the instance, or
            # its device string if it has no device type) and dtype (keyword argument or
            # attribute). If both match the filters, runs the test and swallows any exception,
            # failing the test if no exception occurred. Otherwise runs the test normally.
            if (
                not hasattr(slf, "device_type")
                and hasattr(slf, "device")
                and isinstance(slf.device, str)
            ):
                target_device_type = slf.device
            else:
                target_device_type = slf.device_type

            target_dtype = kwargs.get("dtype", getattr(slf, "dtype", None))
            device_matches = (
                self.device_type is None or self.device_type == target_device_type
            )
            dtype_matches = self.dtype is None or self.dtype == target_dtype

            if device_matches and dtype_matches:
                try:
                    fn(slf, *args, **kwargs)
                except Exception:
                    return
                else:
                    slf.fail("expected test to fail, but it passed")

            return fn(slf, *args, **kwargs)

        return efail_fn


class onlyOn:
    # [CLASS DOC] Decorator that runs a test only on the given device type(s). State:
    # device_type, a string or list; the check is a membership test, so for a plain string it
    # is a substring test.
    def __init__(self, device_type: str | list):
        # [FUNCTION DOC] Stores the allowed device type name or list.
        self.device_type = device_type

    def __call__(self, fn):
        # [FUNCTION DOC] Returns a wrapper around the test function (see the wrapper's notes).
        @wraps(fn)
        def only_fn(slf, *args, **kwargs):
            # [FUNCTION DOC] HELPER: If the device type is not allowed: runs the test anyway
            # when the class sets the bypass flag; in Sandcastle/FB prints a message to stderr
            # and returns without running; otherwise raises a skip. If allowed, runs the test.
            if slf.device_type not in self.device_type:
                if getattr(slf, "bypass_device_restrictions", False):
                    return fn(slf, *args, **kwargs)
                reason = f"Only runs on {self.device_type}"
                if IS_SANDCASTLE or IS_FBCODE:
                    print(
                        f"Skipping {fn.__name__} on sandcastle for following reason: {reason}",
                        file=sys.stderr,
                    )
                    return
                raise unittest.SkipTest(reason)

            return fn(slf, *args, **kwargs)

        return only_fn


# Decorator that provides all available devices of the device type to the test
# as a list of strings instead of providing a single device string.
# Skips the test if the number of available devices of the variant's device
# type is less than the 'num_required_devices' arg.
class deviceCountAtLeast:
    # [CLASS DOC] Decorator for multi-device tests. State: num_required_devices. Marks the
    # test so the framework passes the list of all devices of the type as its device argument.
    def __init__(self, num_required_devices):
        # [FUNCTION DOC] Stores the minimum number of devices needed.
        self.num_required_devices = num_required_devices

    def __call__(self, fn):
        # [FUNCTION DOC] Sets the num_required_devices attribute on the test function (mutating
        # it) and returns a wrapper that skips when fewer devices are given. Raises an error if
        # the attribute is already set (applied twice).
        if hasattr(fn, "num_required_devices"):
            raise AssertionError(f"deviceCountAtLeast redefinition for {fn.__name__}")
        fn.num_required_devices = self.num_required_devices

        @wraps(fn)
        def multi_fn(slf, devices, *args, **kwargs):
            # [FUNCTION DOC] HELPER: If the devices list is too short: in Sandcastle/FB prints
            # a message to stderr and returns; otherwise raises a skip. Else runs the test.
            if len(devices) < self.num_required_devices:
                reason = f"fewer than {self.num_required_devices} devices detected"
                if IS_SANDCASTLE or IS_FBCODE:
                    print(
                        f"Skipping {fn.__name__} on sandcastle for following reason: {reason}",
                        file=sys.stderr,
                    )
                    return
                raise unittest.SkipTest(reason)

            return fn(slf, devices, *args, **kwargs)

        return multi_fn


# Only runs the test on the native device type (currently CPU, CUDA, Meta and PRIVATEUSE1)
def onlyNativeDeviceTypes(fn: Callable[_P, _T]) -> Callable[_P, _T]:
    # [FUNCTION DOC] Decorator: wraps the test so it is skipped unless the device type is in
    # NATIVE_DEVICES.
    @wraps(fn)
    def only_fn(self, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        # [FUNCTION DOC] HELPER: Raises a skip if the device type is not native; else runs it.
        if self.device_type not in NATIVE_DEVICES:
            reason = f"onlyNativeDeviceTypes: doesn't run on {self.device_type}"
            raise unittest.SkipTest(reason)

        return fn(self, *args, **kwargs)

    return only_fn


# Only runs the test on the native device types and devices specified in the devices list
def onlyNativeDeviceTypesAnd(devices=None):
    # [FUNCTION DOC] Decorator factory: the produced decorator skips the test unless the device
    # type is native or in the given extra devices collection.
    # KNOWN ISSUE: with the default of None, a non-native device type causes a membership test
    # on None, which raises a type error instead of skipping.
    def decorator(fn):
        # [FUNCTION DOC] HELPER: Wraps the test function with the device check.
        @wraps(fn)
        def only_fn(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Skips if the device type is neither native nor listed.
            if (
                self.device_type not in NATIVE_DEVICES
                and self.device_type not in devices
            ):
                reason = f"onlyNativeDeviceTypesAnd {devices} : doesn't run on {self.device_type}"
                raise unittest.SkipTest(reason)

            return fn(self, *args, **kwargs)

        return only_fn

    return decorator


# Specifies per-dtype precision overrides.
# Ex.
#
# @precisionOverride({torch.half : 1e-2, torch.float : 1e-4})
# @dtypes(torch.half, torch.float, torch.double)
# def test_X(self, device, dtype):
#   ...
#
# When the test is instantiated its class's precision will be set to the
# corresponding override, if it exists.
# self.precision can be accessed directly, and it also controls the behavior of
# functions like self.assertEqual().
#
# Note that self.precision is a scalar value, so if you require multiple
# precisions (or are working with multiple dtypes) they should be specified
# explicitly and computed using self.precision (e.g.
# self.precision *2, max(1, self.precision)).
class precisionOverride:
    # [CLASS DOC] Decorator attaching per-dtype absolute tolerance overrides to a test. State:
    # d, a mapping from torch dtype to a tolerance number. The framework applies the entry for
    # the variant's dtype to the instance precision when the test runs.
    def __init__(self, d):
        # [FUNCTION DOC] Validates and stores the mapping. Raises an error if it is not a
        # mapping or has a key that is not a torch dtype.
        if not isinstance(d, dict):
            raise AssertionError(
                "precisionOverride not given a dtype : precision dict!"
            )
        for dtype in d:
            if not isinstance(dtype, torch.dtype):
                raise AssertionError(f"precisionOverride given unknown dtype {dtype}")

        self.d = d

    def __call__(self, fn):
        # [FUNCTION DOC] Sets the precision_overrides attribute on the test function (mutating
        # it) and returns the same function.
        fn.precision_overrides = self.d
        return fn


# Specifies per-dtype tolerance overrides tol(atol, rtol). It has priority over
# precisionOverride.
# Ex.
#
# @toleranceOverride({torch.float : tol(atol=1e-2, rtol=1e-3},
#                     torch.double : tol{atol=1e-4, rtol = 0})
# @dtypes(torch.half, torch.float, torch.double)
# def test_X(self, device, dtype):
#   ...
#
# When the test is instantiated its class's tolerance will be set to the
# corresponding override, if it exists.
# self.rtol and self.precision can be accessed directly, and they also control
# the behavior of functions like self.assertEqual().
#
# The above example sets atol = 1e-2 and rtol = 1e-3 for torch.float and
# atol = 1e-4 and rtol = 0 for torch.double.
tol = namedtuple("tol", ["atol", "rtol"])


class toleranceOverride:
    # [CLASS DOC] Decorator attaching per-dtype (atol, rtol) tolerance overrides to a test.
    # State: d, a mapping from torch dtype to a tol pair. Takes priority over precisionOverride.
    def __init__(self, d):
        # [FUNCTION DOC] Validates and stores the mapping. Raises an error if it is not a
        # mapping, a key is not a torch dtype, or a value is not a tol pair.
        if not isinstance(d, dict):
            raise AssertionError("toleranceOverride not given a dtype : tol dict!")
        for dtype, prec in d.items():
            if not isinstance(dtype, torch.dtype):
                raise AssertionError(f"toleranceOverride given unknown dtype {dtype}")
            if not isinstance(prec, tol):
                raise AssertionError("toleranceOverride not given a dtype : tol dict!")

        self.d = d

    def __call__(self, fn):
        # [FUNCTION DOC] Sets the tolerance_overrides attribute on the test function (mutating
        # it) and returns the same function.
        fn.tolerance_overrides = self.d
        return fn


# Decorator that instantiates a variant of the test for each given dtype.
# Notes:
#   (1) Tests that accept the dtype argument MUST use this decorator.
#   (2) Can be overridden for CPU or CUDA, respectively, using dtypesIfCPU
#       or dtypesIfCUDA.
#   (3) Can accept an iterable of dtypes or an iterable of tuples
#       of dtypes.
# Examples:
# @dtypes(torch.float32, torch.float64)
# @dtypes((torch.long, torch.float32), (torch.int, torch.float64))
class dtypes:
    # [CLASS DOC] Decorator listing the dtypes (or tuples of dtypes) a test is instantiated
    # for. State: args (the dtype variants) and device_type ("all" or a specific device type
    # for device-specific overrides). The framework reads the per-device mapping stored on the
    # test function under the dtypes attribute.
    def __init__(self, *args, device_type="all"):
        # [FUNCTION DOC] Validates and stores the dtype variants. Either all are torch dtypes,
        # or all are lists/tuples of torch dtypes. Raises an error otherwise.
        if len(args) > 0 and isinstance(args[0], (list, tuple)):
            for arg in args:
                if not isinstance(arg, (list, tuple)):
                    raise AssertionError(
                        "When one dtype variant is a tuple or list, "
                        "all dtype variants must be. "
                        f"Received non-list non-tuple dtype {str(arg)}"
                    )
                if not all(isinstance(dtype, torch.dtype) for dtype in arg):
                    raise AssertionError(f"Unknown dtype in {str(arg)}")
        else:
            if not all(isinstance(arg, torch.dtype) for arg in args):
                raise AssertionError(f"Unknown dtype in {str(args)}")

        self.args = args
        self.device_type = device_type

    def __call__(self, fn):
        # [FUNCTION DOC] Records the dtype variants on the test function in a mapping keyed by
        # device type (mutating the function) and returns the same function. Raises an error
        # if that device type already has dtypes recorded.
        d = getattr(fn, "dtypes", {})
        if self.device_type in d:
            raise AssertionError(f"dtypes redefinition for {self.device_type}")
        d[self.device_type] = self.args
        fn.dtypes = d
        return fn


# Overrides specified dtypes on the CPU.
class dtypesIfCPU(dtypes):
    # [CLASS DOC] dtypes override applying only to the cpu device type.
    def __init__(self, *args):
        # [FUNCTION DOC] Initializes the parent with device type cpu.
        super().__init__(*args, device_type="cpu")


# Overrides specified dtypes on CUDA.
class dtypesIfCUDA(dtypes):
    # [CLASS DOC] dtypes override applying only to the cuda device type.
    def __init__(self, *args):
        # [FUNCTION DOC] Initializes the parent with device type cuda.
        super().__init__(*args, device_type="cuda")


# Overrides specified dtypes on Intel GPU.
class dtypesIfXPU(dtypes):
    # [CLASS DOC] dtypes override applying only to the xpu device type.
    def __init__(self, *args):
        # [FUNCTION DOC] Initializes the parent with device type xpu.
        super().__init__(*args, device_type="xpu")


class dtypesIfMPS(dtypes):
    # [CLASS DOC] dtypes override applying only to the mps device type.
    def __init__(self, *args):
        # [FUNCTION DOC] Initializes the parent with device type mps.
        super().__init__(*args, device_type="mps")


class dtypesIfHPU(dtypes):
    # [CLASS DOC] dtypes override applying only to the hpu device type.
    def __init__(self, *args):
        # [FUNCTION DOC] Initializes the parent with device type hpu.
        super().__init__(*args, device_type="hpu")


class dtypesIfPRIVATEUSE1(dtypes):
    # [CLASS DOC] dtypes override applying only to the registered PrivateUse1 backend's name.
    def __init__(self, *args):
        # [FUNCTION DOC] Initializes the parent with the device type set to the backend name
        # registered at construction time.
        super().__init__(*args, device_type=torch._C._get_privateuse1_backend_name())


def onlyCPU(fn):
    # [FUNCTION DOC] Decorator: runs the test only on the cpu device type (via onlyOn).
    return onlyOn("cpu")(fn)


def onlyCUDA(fn):
    # [FUNCTION DOC] Decorator: runs the test only on the cuda device type (via onlyOn).
    return onlyOn("cuda")(fn)


def onlyMPS(fn):
    # [FUNCTION DOC] Decorator: runs the test only on the mps device type (via onlyOn).
    return onlyOn("mps")(fn)


def onlyXPU(fn):
    # [FUNCTION DOC] Decorator: runs the test only on the xpu device type (via onlyOn).
    return onlyOn("xpu")(fn)


def onlyHPU(fn):
    # [FUNCTION DOC] Decorator: runs the test only on the hpu device type (via onlyOn).
    return onlyOn("hpu")(fn)


def onlyPRIVATEUSE1(fn):
    # [FUNCTION DOC] Decorator: looks up the registered PrivateUse1 backend at decoration time.
    # If torch has no module for it, returns the test marked as unconditionally skipped;
    # otherwise restricts the test to that device type (via onlyOn).
    device_type = torch._C._get_privateuse1_backend_name()
    device_mod = getattr(torch, device_type, None)
    if device_mod is None:
        reason = f"Skip as torch has no module of {device_type}"
        return unittest.skip(reason)(fn)
    return onlyOn(device_type)(fn)


def onlyAccelerator(fn):
    # [FUNCTION DOC] Decorator: wraps the test so it is skipped on cpu and meta device types.
    """Skip test if not running on an accelerator device (i.e., skip on CPU and meta)."""

    @wraps(fn)
    def only_fn(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Raises a skip on cpu or meta; otherwise runs the test.
        if self.device_type in ("cpu", "meta"):
            reason = "onlyAccelerator: doesn't run on CPU or meta devices"
            raise unittest.SkipTest(reason)
        return fn(self, *args, **kwargs)

    return only_fn


def onlyCUDAAndPRIVATEUSE1(fn):
    # [FUNCTION DOC] Decorator: wraps the test so it runs only on cuda or the registered
    # PrivateUse1 backend's device type, skipping otherwise.
    @wraps(fn)
    def only_fn(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Raises a skip for other device types; else runs the test.
        if self.device_type not in ("cuda", torch._C._get_privateuse1_backend_name()):
            reason = f"onlyCUDAAndPRIVATEUSE1: doesn't run on {self.device_type}"
            raise unittest.SkipTest(reason)

        return fn(self, *args, **kwargs)

    return only_fn


def disablecuDNN(fn):
    # [FUNCTION DOC] Decorator: wraps the test so it runs with cuDNN disabled on CUDA devices
    # that have cuDNN; otherwise runs it unchanged.
    @wraps(fn)
    def disable_cudnn(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Runs the test inside a context with cuDNN turned off when the
        # device is cuda and cuDNN exists; else runs it directly. Returns the test's result.
        if self.device_type == "cuda" and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                return fn(self, *args, **kwargs)
        return fn(self, *args, **kwargs)

    return disable_cudnn


def disableMkldnn(fn):
    # [FUNCTION DOC] Decorator: wraps the test so it runs with mkldnn disabled when mkldnn is
    # available; otherwise runs it unchanged.
    @wraps(fn)
    def disable_mkldnn(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Runs the test inside a context with mkldnn turned off when it
        # is available; else runs it directly. Returns the test's result.
        if torch.backends.mkldnn.is_available():
            with torch.backends.mkldnn.flags(enabled=False):
                return fn(self, *args, **kwargs)
        return fn(self, *args, **kwargs)

    return disable_mkldnn


def expectedFailureCPU(fn):
    # [FUNCTION DOC] Decorator: marks the test as an expected failure on cpu.
    return expectedFailure("cpu")(fn)


def expectedFailureCUDA(fn):
    # [FUNCTION DOC] Decorator: marks the test as an expected failure on cuda.
    return expectedFailure("cuda")(fn)


def expectedFailureIfRocm(fn):
    # [FUNCTION DOC] Decorator: marks the test as an expected failure on cuda only when running
    # on ROCm; otherwise returns the test unchanged.
    return expectedFailure("cuda")(fn) if TEST_WITH_ROCM else fn


def expectedFailureXPU(fn):
    # [FUNCTION DOC] Decorator: marks the test as an expected failure on xpu.
    return expectedFailure("xpu")(fn)


def expectedFailureMeta(fn):
    # [FUNCTION DOC] Decorator: marks the test as an expected failure on meta, and also skips it
    # when running under Dynamo.
    return skipIfTorchDynamo()(expectedFailure("meta")(fn))


def expectedFailureXLA(fn):
    # [FUNCTION DOC] Decorator: marks the test as an expected failure on xla.
    return expectedFailure("xla")(fn)


def expectedFailureHPU(fn):
    # [FUNCTION DOC] Decorator: marks the test as an expected failure on hpu.
    return expectedFailure("hpu")(fn)


def expectedFailureMPS(fn):
    # [FUNCTION DOC] Decorator: marks the test as an expected failure on mps.
    return expectedFailure("mps")(fn)


def expectedFailureMPSComplex(fn):
    # [FUNCTION DOC] Decorator: marks the test as an expected failure on mps for complex64 only.
    return expectedFailure("mps", torch.complex64)(fn)


def expectedFailureMPSPre15(fn):
    # [FUNCTION DOC] Decorator: reads the macOS version at decoration time; if it is a known
    # version below 15, marks the test as an expected failure on mps; otherwise (newer macOS,
    # or not macOS) returns the test unchanged.
    import platform

    version = float(".".join(platform.mac_ver()[0].split(".")[:2]) or -1)
    if not version or version < 1.0:  # cpu or other unsupported device
        return fn
    if version < 15.0:
        return expectedFailure("mps")(fn)
    return fn


def expectedFailureMPSPre14(fn):
    # [FUNCTION DOC] Decorator: same as the pre-15 variant but with a threshold of macOS 14.
    import platform

    version = float(".".join(platform.mac_ver()[0].split(".")[:2]) or -1)
    if not version or version < 1.0:  # cpu or other unsupported device
        return fn
    if version < 14.0:
        return expectedFailure("mps")(fn)
    return fn


# Skips a test on CPU if LAPACK is not available.
def skipCPUIfNoLapack(fn):
    # [FUNCTION DOC] Decorator: skips the test on cpu when torch was built without LAPACK.
    return skipCPUIf(not torch._C.has_lapack, "PyTorch compiled without Lapack")(fn)


# Skips a test on CPU if FFT is not available.
def skipCPUIfNoFFT(fn):
    # [FUNCTION DOC] Decorator: skips the test on cpu when torch was built without FFT support.
    return skipCPUIf(not torch._C.has_spectral, "PyTorch is built without FFT support")(
        fn
    )


# Skips a test on CPU if MKL is not available.
def skipCPUIfNoMkl(fn):
    # [FUNCTION DOC] Decorator: skips the test on cpu when torch was built without MKL.
    return skipCPUIf(not TEST_MKL, "PyTorch is built without MKL support")(fn)


# Skips a test on CPU if MKL Sparse is not available (it's not linked on Windows).
def skipCPUIfNoMklSparse(fn):
    # [FUNCTION DOC] Decorator: skips the test on cpu on Windows or when MKL is missing.
    return skipCPUIf(
        IS_WINDOWS or not TEST_MKL, "PyTorch is built without MKL support"
    )(fn)


# Skips a test on CPU if mkldnn is not available.
def skipCPUIfNoMkldnn(fn):
    # [FUNCTION DOC] Decorator: skips the test on cpu when mkldnn is unavailable.
    return skipCPUIf(
        not torch.backends.mkldnn.is_available(),
        "PyTorch is built without mkldnn support",
    )(fn)


# Skips a test on CUDA if MAGMA is not available.
def skipCUDAIfNoMagma(fn):
    # [FUNCTION DOC] Decorator: skips the test on cuda when the class flag no_magma is set, and
    # additionally marks it to skip under non-default-stream mode (always true here).
    return skipCUDAIf("no_magma", "no MAGMA library detected")(
        skipCUDANonDefaultStreamIf(True)(fn)
    )


def has_cusolver():
    # [FUNCTION DOC] Returns true when not running on ROCm (assumes cuSOLVER exists then).
    return not TEST_WITH_ROCM


def has_hipsolver():
    # [FUNCTION DOC] Returns true when running on ROCm (assumes hipSOLVER exists then).
    return TEST_WITH_ROCM


# Skips a test on CUDA if cuSOLVER is not available,
# and on ROCm if MAGMA is not available.
def skipCUDAIfNoCusolverROCMIfNoMagma(fn):
    # [FUNCTION DOC] Decorator: on ROCm requires MAGMA (skip on cuda if missing); on CUDA
    # requires cuSOLVER (via skipCUDAIfNoCusolver).
    if TEST_WITH_ROCM:
        return skipCUDAIfNoMagma(fn)
    else:
        return skipCUDAIfNoCusolver(fn)


# Skips a test on CUDA/ROCM if cuSOLVER/hipSOLVER is not available
def skipCUDAIfNoCusolver(fn):
    # [FUNCTION DOC] Decorator: skips the test on cuda when neither cuSOLVER nor hipSOLVER is
    # available. KNOWN ISSUE: the helpers are exact complements, so this condition is never true.
    return skipCUDAIf(
        not has_cusolver() and not has_hipsolver(), "cuSOLVER not available"
    )(fn)


# Skips a test if both cuSOLVER and MAGMA are not available
def skipCUDAIfNoMagmaAndNoCusolver(fn):
    # [FUNCTION DOC] Decorator: returns the test unchanged when cuSOLVER is assumed present;
    # otherwise applies the MAGMA skip.
    if has_cusolver():
        return fn
    else:
        # cuSolver is disabled on cuda < 10.1.243, tests depend on MAGMA
        return skipCUDAIfNoMagma(fn)


# Skips a test if both cuSOLVER/hipSOLVER and MAGMA are not available
def skipCUDAIfNoMagmaAndNoLinalgsolver(fn):
    # [FUNCTION DOC] Decorator: returns the test unchanged when cuSOLVER or hipSOLVER is
    # assumed present; otherwise applies the MAGMA skip.
    if has_cusolver() or has_hipsolver():
        return fn
    else:
        # cuSolver is disabled on cuda < 10.1.243, tests depend on MAGMA
        return skipCUDAIfNoMagma(fn)


# Skips a test on CUDA when using ROCm.
def skipCUDAIfRocm(func=None, *, msg="test doesn't currently work on the ROCm stack"):
    # [FUNCTION DOC] Decorator usable bare or with a message: skips the test on cuda when
    # running on ROCm. If a function is passed it is decorated directly; otherwise returns the
    # decorator.
    def dec_fn(fn):
        # [FUNCTION DOC] HELPER: Applies the ROCm skip with the given message to the test.
        reason = f"skipCUDAIfRocm: {msg}"
        return skipCUDAIf(TEST_WITH_ROCM, reason=reason)(fn)

    if func:
        return dec_fn(func)
    return dec_fn


# Skips a test on CUDA when not using ROCm.
def skipCUDAIfNotRocm(fn):
    # [FUNCTION DOC] Decorator: skips the test on cuda when not running on ROCm.
    return skipCUDAIf(
        not TEST_WITH_ROCM, "test doesn't currently work on the CUDA stack"
    )(fn)


# Skips a test on CUDA if ROCm is unavailable or its version is lower than requested.
def skipCUDAIfRocmVersionLessThan(version=None):
    # [FUNCTION DOC] Decorator factory: on cuda devices, skips the test if not on ROCm, or the
    # ROCm version is unknown, or version is None, or the ROCm version is below the given
    # version tuple.
    def dec_fn(fn):
        # [FUNCTION DOC] HELPER: Wraps the test with the version check.
        @wraps(fn)
        def wrap_fn(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Raises a skip when the check fails; else runs the test.
            if self.device_type == "cuda":
                if not TEST_WITH_ROCM:
                    reason = "ROCm not available"
                    raise unittest.SkipTest(reason)
                rocm_version_tuple = _get_torch_rocm_version()
                if (
                    rocm_version_tuple is None
                    or version is None
                    or rocm_version_tuple < tuple(version)
                ):
                    reason = (
                        f"ROCm {rocm_version_tuple} is available but {version} required"
                    )
                    raise unittest.SkipTest(reason)

            return fn(self, *args, **kwargs)

        return wrap_fn

    return dec_fn


# Skips a test on CUDA if ROCm hipBLASLt is unavailable or its version is lower than requested.
def skipCUDAIfRocmHipBlasltVersionLessThan(version=None):
    # [FUNCTION DOC] Decorator factory: on cuda devices, skips the test if not on ROCm, or the
    # hipBLASLt version is unknown, or version is None, or it is below the given version tuple.
    def dec_fn(fn):
        # [FUNCTION DOC] HELPER: Wraps the test with the version check.
        @wraps(fn)
        def wrap_fn(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Raises a skip when the check fails; else runs the test.
            if self.device_type == "cuda":
                if not TEST_WITH_ROCM:
                    reason = "ROCm not available"
                    raise unittest.SkipTest(reason)
                hipblaslt_version_tuple = _get_torch_hipblaslt_version()
                if (
                    hipblaslt_version_tuple is None
                    or version is None
                    or hipblaslt_version_tuple < tuple(version)
                ):
                    reason = f"hipBLASLt {hipblaslt_version_tuple} is available but {version} required"
                    raise unittest.SkipTest(reason)

            return fn(self, *args, **kwargs)

        return wrap_fn

    return dec_fn


# Skips a test on CUDA when using ROCm.
def skipCUDAIfNotMiopenSuggestNHWC(fn):
    # [FUNCTION DOC] Decorator: skips the test on cuda unless the MIOpen suggest-NHWC test flag
    # is enabled.
    # KNOWN ISSUE: the comment above this function says it skips on ROCm, which does not match.
    return skipCUDAIf(
        not TEST_WITH_MIOPEN_SUGGEST_NHWC,
        "test doesn't currently work without MIOpen NHWC activation",
    )(fn)


# Skips a test for specified CUDA versions, given in the form of a list of [major, minor]s.
def skipCUDAVersionIn(versions: list[tuple[int, int]] | None = None):
    # [FUNCTION DOC] Decorator factory: skips the test when the installed CUDA version tuple is
    # in the given list. Runs the test normally when the version is (0, 0) (no CUDA or ROCm).
    def dec_fn(fn):
        # [FUNCTION DOC] HELPER: Wraps the test with the version check.
        @wraps(fn)
        def wrap_fn(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Raises a skip when the version is listed; else runs it.
            version = _get_torch_cuda_version()
            if version == (0, 0):  # cpu or rocm
                return fn(self, *args, **kwargs)
            if version in (versions or []):
                reason = f"test skipped for CUDA version {version}"
                raise unittest.SkipTest(reason)
            return fn(self, *args, **kwargs)

        return wrap_fn

    return dec_fn


# Skips a test for CUDA versions less than specified, given in the form of [major, minor].
def skipCUDAIfVersionLessThan(versions: tuple[int, int] | None = None):
    # [FUNCTION DOC] Decorator factory: skips the test when the installed CUDA version tuple is
    # below the given tuple; runs it normally when the version is (0, 0).
    # KNOWN ISSUE: the skip message prints the installed version where the threshold is meant,
    # and a None threshold makes the comparison raise a type error.
    def dec_fn(fn):
        # [FUNCTION DOC] HELPER: Wraps the test with the version check.
        @wraps(fn)
        def wrap_fn(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Raises a skip when the version is too low; else runs it.
            version = _get_torch_cuda_version()
            if version == (0, 0):  # cpu or rocm
                return fn(self, *args, **kwargs)
            if version < versions:
                reason = f"test skipped for CUDA versions < {version}"
                raise unittest.SkipTest(reason)
            return fn(self, *args, **kwargs)

        return wrap_fn

    return dec_fn


# Skips a test on CUDA if cuDNN is unavailable or its version is lower than requested.
def skipCUDAIfCudnnVersionLessThan(version=0):
    # [FUNCTION DOC] Decorator factory: on cuda devices, skips the test if cuDNN is unavailable
    # or its version is unknown or below the given number. Uses class attributes set by the
    # CUDA test base's setUpClass.
    def dec_fn(fn):
        # [FUNCTION DOC] HELPER: Wraps the test with the cuDNN check.
        @wraps(fn)
        def wrap_fn(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Raises a skip when the check fails; else runs the test.
            if self.device_type == "cuda":
                if self.no_cudnn:
                    reason = "cuDNN not available"
                    raise unittest.SkipTest(reason)
                if self.cudnn_version is None or self.cudnn_version < version:
                    reason = f"cuDNN version {self.cudnn_version} is available but {version} required"
                    raise unittest.SkipTest(reason)

            return fn(self, *args, **kwargs)

        return wrap_fn

    return dec_fn


# Skips a test on CUDA if cuSparse generic API is not available
def skipCUDAIfNoCusparseGeneric(fn):
    # [FUNCTION DOC] Decorator: skips the test on cuda when the cuSPARSE generic API is absent.
    return skipCUDAIf(not TEST_CUSPARSE_GENERIC, "cuSparse Generic API not available")(
        fn
    )


def skipCUDAIfNoHipsparseGeneric(fn):
    # [FUNCTION DOC] Decorator: skips the test on cuda when the hipSPARSE generic API is absent.
    return skipCUDAIf(
        not TEST_HIPSPARSE_GENERIC, "hipSparse Generic API not available"
    )(fn)


def skipCUDAIfNoSparseGeneric(fn):
    # [FUNCTION DOC] Decorator: skips the test on cuda when neither the cuSPARSE nor the
    # hipSPARSE generic API is available.
    return skipCUDAIf(
        not (TEST_CUSPARSE_GENERIC or TEST_HIPSPARSE_GENERIC),
        "Sparse Generic API not available",
    )(fn)


def skipCUDAIfNoCudnn(fn):
    # [FUNCTION DOC] Decorator: skips the test on cuda when cuDNN is unavailable (version 0).
    return skipCUDAIfCudnnVersionLessThan(0)(fn)


def skipCUDAIfMiopen(fn):
    # [FUNCTION DOC] Decorator: skips the test on cuda when torch is a HIP (MIOpen) build.
    return skipCUDAIf(torch.version.hip is not None, "Marked as skipped for MIOpen")(fn)


def skipCUDAIfNoMiopen(fn):
    # [FUNCTION DOC] Decorator: skips the test on cuda when not a HIP build, and also applies
    # the no-cuDNN skip.
    return skipCUDAIf(torch.version.hip is None, "MIOpen is not available")(
        skipCUDAIfNoCudnn(fn)
    )


def skipLazy(fn):
    # [FUNCTION DOC] Decorator: unconditionally skips the test on the lazy device type.
    return skipLazyIf(True, "test doesn't work with lazy tensors")(fn)


def skipMeta(fn):
    # [FUNCTION DOC] Decorator: unconditionally skips the test on the meta device type.
    return skipMetaIf(True, "test doesn't work with meta tensors")(fn)


def skipXLA(fn):
    # [FUNCTION DOC] Decorator: unconditionally skips the test on the xla device type.
    return skipXLAIf(True, "Marked as skipped for XLA")(fn)


def skipMPS(fn):
    # [FUNCTION DOC] Decorator: unconditionally skips the test on the mps device type.
    return skipMPSIf(True, "test doesn't work on MPS backend")(fn)


def skipHPU(fn):
    # [FUNCTION DOC] Decorator: unconditionally skips the test on the hpu device type.
    return skipHPUIf(True, "test doesn't work on HPU backend")(fn)


def skipXPU(fn):
    # [FUNCTION DOC] Decorator: unconditionally skips the test on the xpu device type.
    return skipXPUIf(True, "test doesn't work on XPU backend")(fn)


def skipPRIVATEUSE1(fn):
    # [FUNCTION DOC] Decorator: unconditionally skips the test on the registered PrivateUse1
    # backend's device type.
    return skipPRIVATEUSE1If(True, "test doesn't work on privateuse1 backend")(fn)


# TODO: the "all" in the name isn't true anymore for quite some time as we have also have for example XLA and MPS now.
#  This should probably enumerate all available device type test base classes.
def get_all_device_types() -> list[str]:
    # [FUNCTION DOC] Returns ["cpu"], plus "cuda" when CUDA is available.
    # KNOWN ISSUE: despite the name it ignores other device types (see the TODO above).
    return ["cpu"] if not torch.cuda.is_available() else ["cpu", "cuda"]


# skip since currently flex attention requires at least `avx2` support on CPU.
IS_FLEX_ATTENTION_CPU_PLATFORM_SUPPORTED = (
    not IS_MACOS
    and torch.cpu._is_avx2_supported()
    and os.getenv("ATEN_CPU_CAPABILITY") != "default"
)
IS_FLEX_ATTENTION_XPU_PLATFORM_SUPPORTED = (
    torch.xpu.is_available() and torch.utils._triton.has_triton()
)
IS_FLEX_ATTENTION_CUDA_PLATFORM_SUPPORTED = (
    torch.cuda.is_available()
    and torch.utils._triton.has_triton()
    and torch.cuda.get_device_capability() >= (8, 0)
)
IS_FLEX_ATTENTION_MPS_PLATFORM_SUPPORTED = torch.mps.is_available()
flex_attention_supported_platform = unittest.skipUnless(
    IS_FLEX_ATTENTION_XPU_PLATFORM_SUPPORTED
    or (
        IS_FLEX_ATTENTION_CPU_PLATFORM_SUPPORTED
        and not torch.xpu.is_available()
        and not torch.cuda.is_available()
    )
    or IS_FLEX_ATTENTION_CUDA_PLATFORM_SUPPORTED
    or IS_FLEX_ATTENTION_MPS_PLATFORM_SUPPORTED,
    "Requires CUDA and Triton, Intel GPU and triton, MPS, or CPU with avx2 and later",
)
if (
    torch.version.hip
    and torch.cuda.device_count() > 0
    and "gfx94" in torch.cuda.get_device_properties(0).gcnArchName
):
    e4m3_type = torch.float8_e4m3fnuz
    e5m2_type = torch.float8_e5m2fnuz
    E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fnuz).max
    E5M2_MAX_POS = torch.finfo(torch.float8_e5m2fnuz).max
else:
    e4m3_type = torch.float8_e4m3fn
    e5m2_type = torch.float8_e5m2
    E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max
    E5M2_MAX_POS = torch.finfo(torch.float8_e5m2).max
