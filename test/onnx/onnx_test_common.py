# Owner(s): ["module: onnx"]
# [MODULE DOC] Shared helpers for the ONNX export tests: type aliases, paths to the ONNX
# backend test data, a base test case that exports a model and verifies it with ONNX
# Runtime, a helper that runs a model in ONNX Runtime, the tested opset and dtype
# ranges, and the DecorateMeta record with helpers (xfail, skip, skip_slow,
# add_decorate_info) used to mark OpInfo based tests, plus reason-message formatters and
# a context manager implementing skip/xfail behavior. Importing requires onnxruntime,
# pytest, numpy and pytorch_test_common.

from __future__ import annotations

import contextlib
import dataclasses
import io
import os
import unittest
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import onnxruntime
import pytest
import pytorch_test_common

import torch
from torch import export as torch_export
from torch.onnx import _constants
from torch.onnx._internal.torchscript_exporter import verification
from torch.testing._internal import common_utils
from torch.testing._internal.opinfo import core as opinfo_core
from torch.types import Number


_NumericType = Number | torch.Tensor | np.ndarray
_ModelType = torch.nn.Module | Callable | torch_export.ExportedProgram
_InputArgsType = (
    torch.Tensor | int | float | bool | Sequence[Any] | Mapping[str, Any] | None
)
_OutputsType = Sequence[_NumericType]

onnx_model_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "repos",
    "onnx",
    "onnx",
    "backend",
    "test",
    "data",
)
pytorch_converted_dir = os.path.join(onnx_model_dir, "pytorch-converted")
pytorch_operator_dir = os.path.join(onnx_model_dir, "pytorch-operator")


def run_model_test(test_suite: _TestONNXRuntime, *args, **kwargs):
    # [FUNCTION DOC] HELPER: Runs verification for a test suite. Builds verification
    # options from the suite attributes (check_shape and check_dtype when present, and a
    # backend from ort_backend when it is not none), sets the opset version and
    # initializer flag in the keyword arguments, and moves any keyword argument whose name
    # is an options field into the options object (removing it from the keywords). Returns
    # the result of the verification call using the remaining arguments. Mutates the
    # passed keyword dictionary.
    options = verification.VerificationOptions()

    kwargs["opset_version"] = test_suite.opset_version
    kwargs["keep_initializers_as_inputs"] = test_suite.keep_initializers_as_inputs
    if hasattr(test_suite, "check_shape"):
        options.check_shape = test_suite.check_shape
    if hasattr(test_suite, "check_dtype"):
        options.check_dtype = test_suite.check_dtype
    ort_backend = getattr(test_suite, "ort_backend", None)
    if ort_backend is not None:
        options.backend = verification.OnnxBackend(ort_backend)

    names = {f.name for f in dataclasses.fields(options)}
    keywords_to_pop = []
    for k, v in kwargs.items():
        if k in names:
            setattr(options, k, v)
            keywords_to_pop.append(k)
    for k in keywords_to_pop:
        kwargs.pop(k)

    return verification.verify(*args, options=options, **kwargs)


def parameterize_class_name(cls: type, idx: int, input_dicts: Mapping[Any, Any]):
    # [FUNCTION DOC] HELPER: Builds a class name for parameterized test classes by joining
    # the base class name and key_value pairs of the parameter mapping with underscores. The
    # index argument is unused.
    """Combine class name with the parameterized arguments.

    This function is passed to `parameterized.parameterized_class` as the
    `class_name_func` argument.
    """
    suffix = "_".join(f"{k}_{v}" for k, v in input_dicts.items())
    return f"{cls.__name__}_{suffix}"


class _TestONNXRuntime(pytorch_test_common.ExportTestCase):
    # [CLASS DOC] HELPER: Base test case for ONNX Runtime export tests. Class attributes
    # configure the opset (default opset constant), whether initializers are graph inputs,
    # scripting mode, shape and dtype checks, and an optional ORT backend name (none means
    # the verification default). Subclasses call run_test to export and verify models.
    opset_version = _constants.ONNX_DEFAULT_OPSET
    keep_initializers_as_inputs = True  # For IR version 3 type export.
    is_script = False
    check_shape = True
    check_dtype = True
    # Optional ORT execution provider name; None uses the VerificationOptions
    # default backend.
    ort_backend: str | None = None

    def setUp(self):
        # [FUNCTION DOC] Runs the base setup, seeds ONNX Runtime with zero, seeds all
        # devices of the current accelerator (if available) with zero, sets the
        # environment variable that allows only released opsets to zero, and enables
        # script tests on the instance.
        super().setUp()
        onnxruntime.set_seed(0)
        device_type = getattr(
            torch.accelerator.current_accelerator(check_available=True), "type", None
        )
        if device_type:
            torch.get_device_module(device_type).manual_seed_all(0)
        os.environ["ALLOW_RELEASED_ONNX_OPSET_ONLY"] = "0"
        self.is_script_test_enabled = True

    # The exported ONNX model may have less inputs than the pytorch model because of const folding.
    # This mostly happens in unit test, where we widely use torch.size or torch.shape.
    # So the output is only dependent on the input shape, not value.
    # remained_onnx_input_idx is used to indicate which pytorch model input idx is remained in ONNX model.
    def run_test(
        self,
        model,
        input_args,
        input_kwargs=None,
        rtol=1e-3,
        atol=1e-7,
        do_constant_folding=True,
        dynamic_axes=None,
        additional_test_inputs=None,
        input_names=None,
        output_names=None,
        fixed_batch_size=False,
        training=torch.onnx.TrainingMode.EVAL,
        remained_onnx_input_idx=None,
        verbose=False,
    ):
        # [FUNCTION DOC] Exports the model and verifies the ONNX result against PyTorch
        # via the verification helper. The remained input index argument may be a mapping
        # with scripting and tracing entries, or one value used for both. In scripting
        # mode (is_script and script tests enabled) the model is scripted (unless already
        # a script module or function) and verified without flattening and without
        # ignoring none outputs. In tracing mode (not script and the model not already
        # scripted) it is verified directly. Does nothing when neither applies. The
        # remaining arguments are tolerances, export options and naming/dynamic axes
        # settings passed through unchanged.
        def _run_test(m, remained_onnx_input_idx, flatten=True, ignore_none=True):
            # [FUNCTION DOC] HELPER: Calls the module-level verification runner for the
            # given model with all outer arguments (inputs, tolerances, constant folding,
            # dynamic axes, names, batch size, training mode, verbose) plus the given
            # remained input index and flatten and ignore-none flags. Returns its result.
            return run_model_test(
                self,
                m,
                input_args=input_args,
                input_kwargs=input_kwargs,
                rtol=rtol,
                atol=atol,
                do_constant_folding=do_constant_folding,
                dynamic_axes=dynamic_axes,
                additional_test_inputs=additional_test_inputs,
                input_names=input_names,
                output_names=output_names,
                fixed_batch_size=fixed_batch_size,
                training=training,
                remained_onnx_input_idx=remained_onnx_input_idx,
                flatten=flatten,
                ignore_none=ignore_none,
                verbose=verbose,
            )

        if isinstance(remained_onnx_input_idx, dict):
            scripting_remained_onnx_input_idx = remained_onnx_input_idx["scripting"]
            tracing_remained_onnx_input_idx = remained_onnx_input_idx["tracing"]
        else:
            scripting_remained_onnx_input_idx = remained_onnx_input_idx
            tracing_remained_onnx_input_idx = remained_onnx_input_idx

        is_model_script = isinstance(
            model, (torch.jit.ScriptModule, torch.jit.ScriptFunction)
        )

        if self.is_script_test_enabled and self.is_script:
            script_model = model if is_model_script else torch.jit.script(model)
            _run_test(
                script_model,
                scripting_remained_onnx_input_idx,
                flatten=False,
                ignore_none=False,
            )
        if not is_model_script and not self.is_script:
            _run_test(model, tracing_remained_onnx_input_idx)


def run_ort(
    onnx_model: str | torch.onnx.ONNXProgram,
    pytorch_inputs: Sequence[_InputArgsType],
) -> _OutputsType:
    # [FUNCTION DOC] Runs an ONNX model in ONNX Runtime on CPU. A program object is first
    # saved to an in-memory buffer; a path or bytes value is used as is. Logging is limited
    # to errors. Raises an assertion error if the number of PyTorch inputs differs from the
    # number of model inputs. Inputs are converted to numpy arrays in order and matched to
    # input names. Returns the list of output arrays.
    """Run ORT on the given ONNX model and inputs

    Used in test_fx_to_onnx_with_onnxruntime.py

    Args:
        onnx_model (Union[str, torch.onnx.ONNXProgram]): Converter ONNX model
        pytorch_inputs (Sequence[_InputArgsType]): The given torch inputs

    Raises:
        AssertionError: ONNX and PyTorch should have the same input sizes

    Returns:
        _OutputsType: ONNX model predictions
    """
    if isinstance(onnx_model, torch.onnx.ONNXProgram):
        buffer = io.BytesIO()
        onnx_model.save(buffer)
        ort_model = buffer.getvalue()
    else:
        ort_model = onnx_model

    # Suppress floods of warnings from ONNX Runtime
    session_options = onnxruntime.SessionOptions()
    session_options.log_severity_level = 3  # Error
    session = onnxruntime.InferenceSession(
        ort_model, providers=["CPUExecutionProvider"], sess_options=session_options
    )
    input_names = [ort_input.name for ort_input in session.get_inputs()]

    if len(input_names) != len(pytorch_inputs):
        raise AssertionError(
            f"Expected {len(input_names)} inputs, got {len(pytorch_inputs)}"
        )

    ort_input = {
        k: torch.Tensor.numpy(v, force=True)
        for k, v in zip(input_names, pytorch_inputs)
    }
    return session.run(None, ort_input)


# The min onnx opset version to test for
MIN_ONNX_OPSET_VERSION = 9
# The max onnx opset version to test for
MAX_ONNX_OPSET_VERSION = _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET
TESTED_OPSETS = range(MIN_ONNX_OPSET_VERSION, MAX_ONNX_OPSET_VERSION + 1)

BOOL_TYPES = (torch.bool,)

INT_TYPES = (
    # torch.int8,
    # torch.int16,
    torch.int32,
    torch.int64,
    # torch.uint8,
)

QINT_TYPES = (
    torch.qint8,
    torch.quint8,
)

FLOAT_TYPES = (
    torch.float16,
    torch.float32,
    # torch.float64,  ORT doesn't support
)

COMPLEX_TYPES = (
    # torch.complex32,  NOTE: torch.complex32 is experimental in torch
    torch.complex64,
    # torch.complex128,  ORT doesn't support
)

TESTED_DTYPES = (
    # Boolean
    torch.bool,
    # Integers
    *INT_TYPES,
    # Floating types
    *FLOAT_TYPES,
    # Complex types
    *COMPLEX_TYPES,
)


@dataclasses.dataclass
class DecorateMeta:
    # [CLASS DOC] Data record describing one skip or xfail rule for an OpInfo test:
    # operator name, variant name, decorator to apply, opset filters (integers or
    # predicates), dtypes, reason, behavior name, optional input matcher, enable flag and
    # optional torch model type.
    """Information about a test case to skip or xfail.

    Adapted from functorch: functorch/test/common_utils.py

    Attributes:
        op_name: The name of the operator.
        variant_name: The name of the OpInfo variant.
        decorator: The decorator to apply to the test case.
        opsets: The opsets to apply the decorator to.
        dtypes: The dtypes to apply the decorator to.
        reason: The reason for skipping.
        test_behavior: The behavior of the test case. [skip or xfail]
        matcher: The matcher to apply to the test case.
        enabled_if: Whether to enable test behavior. Usually used on onnx/ort version control
        model_type: The type of the torch model. Defaults to None.
    """

    op_name: str
    variant_name: str
    decorator: Callable
    opsets: Collection[int | Callable[[int], bool]] | None
    dtypes: Collection[torch.dtype] | None
    reason: str
    test_behavior: str
    matcher: Callable[[Any], bool] | None = None
    enabled_if: bool = True
    model_type: pytorch_test_common.TorchModelType | None = None

    def contains_opset(self, opset: int) -> bool:
        # [FUNCTION DOC] Returns true when no opset filter is set, otherwise true if the
        # given opset equals any integer filter or satisfies any predicate filter.
        if self.opsets is None:
            return True
        return any(
            opset == opset_spec if isinstance(opset_spec, int) else opset_spec(opset)
            for opset_spec in self.opsets
        )


def xfail(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    opsets: Collection[int | Callable[[int], bool]] | None = None,
    dtypes: Collection[torch.dtype] | None = None,
    matcher: Callable[[Any], bool] | None = None,
    enabled_if: bool = True,
    model_type: pytorch_test_common.TorchModelType | None = None,
):
    # [FUNCTION DOC] Builds a rule that marks a matching test as an expected failure
    # (unittest expected failure decorator, behavior name xfail), forwarding the given
    # filters, reason, matcher, enable flag and model type.
    """Expects a OpInfo test to fail.

    Args:
        op_name: The name of the operator.
        variant_name: The name of the variant.
        opsets: The opsets to expect the failure. e.g. [9, 10] or [opsets_before(11)]
        dtypes: The dtypes to expect the failure.
        reason: The reason for the failure.
        matcher: A function that matches the test sample input. It is used only when
            xfail is in the SKIP_XFAIL_SUBTESTS list.
        enabled_if: Whether to enable xfail. Usually used on onnx/ort version control
        model_type: The type of the torch model. Defaults to None.
    """
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.expectedFailure,
        opsets=opsets,
        dtypes=dtypes,
        enabled_if=enabled_if,
        matcher=matcher,
        reason=reason,
        test_behavior="xfail",
        model_type=model_type,
    )


def skip(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    opsets: Collection[int | Callable[[int], bool]] | None = None,
    dtypes: Collection[torch.dtype] | None = None,
    matcher: Callable[[Any], Any] | None = None,
    enabled_if: bool = True,
    model_type: pytorch_test_common.TorchModelType | None = None,
):
    # [FUNCTION DOC] Builds a rule that skips matching tests (unittest skip decorator with
    # the reason prefixed by Skip, behavior name skip), forwarding the given filters,
    # reason, matcher, enable flag and model type.
    """Skips a test case in OpInfo that we don't care about.

    Likely because ONNX does not support the use case or it is by design.

    Args:
        op_name: The name of the operator.
        variant_name: The name of the variant.
        opsets: The opsets to expect the failure. e.g. [9, 10] or [opsets_before(11)]
        dtypes: The dtypes to expect the failure.
        reason: The reason for the failure.
        matcher: A function that matches the test sample input. It is used only when
            skip is in the SKIP_XFAIL_SUBTESTS list.
        enabled_if: Whether to enable skip. Usually used on onnx/ort version control
        model_type: The type of the torch model. Defaults to None.
    """
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.skip(f"Skip: {reason}"),
        opsets=opsets,
        dtypes=dtypes,
        reason=reason,
        matcher=matcher,
        enabled_if=enabled_if,
        test_behavior="skip",
        model_type=model_type,
    )


def skip_slow(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    opsets: Collection[int | Callable[[int], bool]] | None = None,
    dtypes: Collection[torch.dtype] | None = None,
    matcher: Callable[[Any], Any] | None = None,
    model_type: pytorch_test_common.TorchModelType | None = None,
):
    # [FUNCTION DOC] Builds a skip rule for tests that are too slow, using the slow test
    # decorator and the behavior name skip; enabled only when slow tests are not being run.
    # Forwards the filters, reason, matcher and model type.
    """Skips a test case in OpInfo that is too slow.

    It needs further investigation to understand why it is slow.

    Args:
        op_name: The name of the operator.
        variant_name: The name of the variant.
        opsets: The opsets to expect the failure. e.g. [9, 10] or [opsets_before(11)]
        dtypes: The dtypes to expect the failure.
        reason: The reason for the failure.
        matcher: A function that matches the test sample input. It is used only when
            skip is in the SKIP_XFAIL_SUBTESTS list.
        model_type: The type of the torch model. Defaults to None.
    """
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=common_utils.slowTest,
        opsets=opsets,
        dtypes=dtypes,
        reason=reason,
        matcher=matcher,
        enabled_if=not common_utils.TEST_WITH_SLOW,
        test_behavior="skip",
        model_type=model_type,
    )


def add_decorate_info(
    all_opinfos: Sequence[opinfo_core.OpInfo],
    test_class_name: str,
    base_test_name: str,
    opset: int,
    skip_or_xfails: Iterable[DecorateMeta],
):
    # [FUNCTION DOC] Appends decorators from the rules to the matching OpInfo objects
    # (mutating each opinfo decorators field in place). Rules whose opset filter excludes
    # the given opset are ignored. Raises an assertion error when no OpInfo matches the rule
    # operator and variant, or when a rule carries a model type (those belong in a different
    # list). Returns an identity decorator that leaves the function unchanged.
    """Decorates OpInfo tests with decorators based on the skip_or_xfails list.

    Args:
        all_opinfos: All OpInfos.
        test_class_name: The name of the test class.
        base_test_name: The name of the test method.
        opset: The opset to decorate for.
        skip_or_xfails: DecorateMeta's.
    """
    ops_mapping = {(info.name, info.variant_test_name): info for info in all_opinfos}
    for decorate_meta in skip_or_xfails:
        if not decorate_meta.contains_opset(opset):
            # Skip does not apply to this opset
            continue
        opinfo = ops_mapping.get((decorate_meta.op_name, decorate_meta.variant_name))
        if opinfo is None:
            raise AssertionError(
                f"Couldn't find OpInfo for {decorate_meta}. Did you need to specify variant_name?"
            )
        if decorate_meta.model_type is not None:
            raise AssertionError(
                f"Tested op: {decorate_meta.op_name} in wrong position! "
                "If model_type needs to be specified, it should be "
                "put under SKIP_XFAIL_SUBTESTS_WITH_MATCHER_AND_MODEL_TYPE."
            )
        decorators = list(opinfo.decorators)
        new_decorator = opinfo_core.DecorateInfo(
            decorate_meta.decorator,
            test_class_name,
            base_test_name,
            dtypes=decorate_meta.dtypes,
            active_if=decorate_meta.enabled_if,
        )
        decorators.append(new_decorator)
        opinfo.decorators = tuple(decorators)

    # This decorator doesn't modify fn in any way
    def wrapped(fn):
        # [FUNCTION DOC] HELPER: Identity decorator that returns the function unchanged.
        return fn

    return wrapped


def opsets_before(opset: int) -> Callable[[int], bool]:
    # [FUNCTION DOC] Returns a predicate that is true for opsets strictly lower than
    # the given opset.
    """Returns a comparison function that decides if the given opset is before the specified."""

    def compare(other_opset: int):
        # [FUNCTION DOC] HELPER: True when the other opset is strictly lower than
        # the captured opset.
        return other_opset < opset

    return compare


def opsets_after(opset: int) -> Callable[[int], bool]:
    # [FUNCTION DOC] Returns a predicate that is true for opsets strictly higher than
    # the given opset.
    """Returns a comparison function that decides if the given opset is after the specified."""

    def compare(other_opset: int):
        # [FUNCTION DOC] HELPER: True when the other opset is strictly higher than
        # the captured opset.
        return other_opset > opset

    return compare


def reason_onnx_script_does_not_support(
    operator: str, dtypes: Sequence[str] | None = None
) -> str:
    # [FUNCTION DOC] HELPER: Formats a reason string saying the operator on the given dtypes
    # (or a generic word when none) is not supported by ONNX script.
    """Formats the reason: ONNX script doesn't support the given dtypes."""
    return f"{operator} on {dtypes or 'dtypes'} not supported by ONNX script"


def reason_onnx_runtime_does_not_support(
    operator: str, dtypes: Sequence[str] | None = None
) -> str:
    # [FUNCTION DOC] HELPER: Formats a reason string saying the operator on the given dtypes
    # (or a generic word when none) is not supported by ONNX Runtime.
    """Formats the reason: ONNX Runtime doesn't support the given dtypes."""
    return f"{operator} on {dtypes or 'dtypes'} not supported by ONNX Runtime"


def reason_onnx_does_not_support(
    operator: str, dtypes: Sequence[str] | None = None
) -> str:
    # [FUNCTION DOC] HELPER: Formats a reason string saying the operator on the given dtypes
    # (or certain dtypes when none) is not supported by the ONNX spec.
    """Formats the reason: ONNX doesn't support the given dtypes."""
    return f"{operator} on {dtypes or 'certain dtypes'} not supported by the ONNX Spec"


def reason_dynamo_does_not_support(
    operator: str, dtypes: Sequence[str] | None = None
) -> str:
    # [FUNCTION DOC] HELPER: Formats a reason string saying the operator on the given dtypes
    # (or certain dtypes when none) is not supported by the Dynamo spec.
    """Formats the reason: Dynamo doesn't support the given dtypes."""
    return (
        f"{operator} on {dtypes or 'certain dtypes'} not supported by the Dynamo Spec"
    )


def reason_jit_tracer_error(info: str) -> str:
    # [FUNCTION DOC] HELPER: Formats a reason string for a JIT tracer error on
    # the given info.
    """Formats the reason: JIT tracer errors."""
    return f"JIT tracer error on {info}"


def reason_flaky() -> str:
    # [FUNCTION DOC] HELPER: Returns the fixed reason string for flaky tests.
    """Formats the reason: test is flaky."""
    return "flaky test"


@contextlib.contextmanager
def normal_xfail_skip_test_behaviors(
    test_behavior: str | None = None, reason: str | None = None
):
    # [FUNCTION DOC] Context manager wrapping a test body. Behavior skip skips immediately
    # via pytest. Otherwise the body runs: an exception is re-raised when the behavior is
    # none, converts to a pytest expected failure when the behavior is xfail, and a clean
    # pass under xfail is reported as a failure saying the test unexpectedly passed.
    # KNOWN ISSUE: for any other behavior value an exception raised by the body is silently
    # swallowed rather than re-raised.
    """This context manager is used to handle the different behaviors of xfail and skip.

    Args:
        test_behavior (optional[str]): From DecorateMeta name, can be 'skip', 'xfail', or None.
        reason (optional[str]): The reason for the failure or skip.

    Raises:
        e: Any exception raised by the test case if it's not an expected failure.
    """

    # We need to skip as soon as possible, as SegFault might also be a case.
    if test_behavior == "skip":
        pytest.skip(reason=reason)

    try:
        yield
    # We could use `except (AssertionError, RuntimeError, ...) as e:`, but it needs
    # to go over all test cases to find the right exception type.
    except Exception as e:  # pylint: disable=broad-exception-caught
        if test_behavior is None:
            raise e
        if test_behavior == "xfail":
            pytest.xfail(reason=reason)
    else:
        if test_behavior == "xfail":
            pytest.fail("Test unexpectedly passed")
