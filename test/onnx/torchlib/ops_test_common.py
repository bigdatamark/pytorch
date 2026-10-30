# Owner(s): ["module: onnx"]
# [MODULE DOC] Shared helpers for the torchlib ONNX operator tests. Defines dtype tuples, the test
# opset constant, skip/xfail metadata (DecorateMeta) and helpers that attach it to OpInfo
# decorators, utilities to duplicate OpInfos, conversion of torch values to numpy/ONNX, and an
# executor that traces an onnxscript function into an ONNX graph and runs it in ONNX Runtime.
# Importing it has no side effects beyond imports and building the dtype mapping table.
"""Common utils for testing operators."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import multiprocessing
import os
import pprint
import sys
import unittest
import warnings
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any, TypeVar

import error_reproduction
import numpy as np

import onnx
import onnx_ir as ir
import onnxruntime as ort
import onnxruntime.capi.onnxruntime_pybind11_state
import onnxscript
import onnxscript.evaluator
import pytest

import torch
from torch.onnx._internal.exporter import _building, _ir_passes, _tensors
from torch.testing._internal.opinfo import core as opinfo_core


T = TypeVar("T")


# Convenience tuples for creating dtype lists when skipping or xfailing tests

BOOL_TYPES = (torch.bool,)

INT_TYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)

FLOAT_TYPES = (
    torch.float16,
    torch.float32,
    torch.float64,
)


TEST_OPSET_VERSION = 18
IS_MACOS = sys.platform.startswith("darwin")
IS_WINDOWS = os.name == "nt"


@dataclasses.dataclass
class DecorateMeta:
    # [CLASS DOC] Plain record describing one skip or xfail rule for an OpInfo test: operator
    # name, variant name, the unittest decorator to apply, optional dtype and device filters, the
    # reason, the behavior label (skip or xfail), an optional sample matcher, an enabled flag and
    # an optional test class restriction. Created by the xfail and skip factory functions.
    """A dataclass for storing information about a test case to skip or xfail.

    Adapted from functorch: functorch/test/common_utils.py
    """

    op_name: str
    variant_name: str
    decorator: Callable[..., Any]
    dtypes: Collection[torch.dtype] | None
    device_type: str | None
    reason: str
    test_behavior: str
    matcher: Callable[[Any], bool] | None = None
    enabled_if: bool = True
    # The test_class_name to apply the decorator to. If None, the decorator is
    # applied to all test classes.
    test_class_name: str | None = None


def xfail(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    dtypes: Collection[torch.dtype] | None = None,
    device_type: str | None = None,
    matcher: Callable[[Any], Any] | None = None,
    enabled_if: bool = True,
    test_class_name: str | None = None,
) -> DecorateMeta:
    # [FUNCTION DOC] Builds a DecorateMeta that expects the matching OpInfo test to fail, using
    # the unittest expected-failure decorator and the behavior label xfail. All arguments are just
    # stored; the reason is required and keyword-only. Returns the new record, no side effects.
    """Expects an OpInfo test to fail.

    Args:
        op_name: The name of the operator.
        variant_name: Optional OpInfo variant_test_name.
        reason: The reason for the failure.
        dtypes: The dtypes to expect the failure.
        device_type: Device type. E.g. "cpu", "cuda".
        matcher: A function that matches the test sample input. It is used only when
            the xfail is in the SKIP_XFAIL_SUBTESTS list.
        enabled_if: Whether the xfail is enabled.
        test_class_name: The test class name to apply the xfail to. If None, the
            xfail is applied to all test classes.
    """
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.expectedFailure,
        dtypes=dtypes,
        device_type=device_type,
        matcher=matcher,
        reason=reason,
        enabled_if=enabled_if,
        test_class_name=test_class_name,
        test_behavior="xfail",
    )


def skip(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    dtypes: Collection[torch.dtype] | None = None,
    device_type: str | None = None,
    matcher: Callable[[Any], Any] | None = None,
    enabled_if: bool = True,
    test_class_name: str | None = None,
) -> DecorateMeta:
    # [FUNCTION DOC] Builds a DecorateMeta that skips the matching OpInfo test, using a unittest
    # skip decorator whose message is the reason prefixed with Skip. The behavior label is skip.
    # Other arguments are stored unchanged. Returns the new record, no side effects.
    """Skips an OpInfo test.

    Args:
        op_name: The name of the operator.
        variant_name: Optional OpInfo variant_test_name.
        reason: The reason for skipping.
        dtypes: The dtypes to skip.
        device_type: Device type. E.g. "cpu", "cuda".
        matcher: A function that matches the test sample input. It is used only when
            the skip is in the SKIP_XFAIL_SUBTESTS list.
        enabled_if: Whether the skip is enabled.
        test_class_name: The test class name to apply the skip to. If None, the skip
            is applied to all test classes.
    """
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.skip(f"Skip: {reason}"),
        dtypes=dtypes,
        device_type=device_type,
        reason=reason,
        matcher=matcher,
        enabled_if=enabled_if,
        test_class_name=test_class_name,
        test_behavior="skip",
    )


def add_decorate_info(
    all_opinfos: Sequence[opinfo_core.OpInfo],
    test_class_name: str,
    base_test_name: str,
    skip_or_xfails: Iterable[DecorateMeta],
) -> Callable[[T], T]:
    # [FUNCTION DOC] Registers each DecorateMeta as an extra decorator on the OpInfo with the same
    # op name and variant name found in all_opinfos. The decorator targets the metadata's own test
    # class if set, else test_class_name, and the given base test name. Mutates the matching
    # OpInfo decorators tuple. A missing OpInfo is ignored when the entry is disabled, otherwise
    # an AssertionError is raised. Returns an identity decorator that leaves the test function
    # unchanged.
    """Decorates OpInfo tests with decorators based on the skip_or_xfails list."""
    ops_mapping = {(info.name, info.variant_test_name): info for info in all_opinfos}
    for decorate_meta in skip_or_xfails:
        opinfo = ops_mapping.get((decorate_meta.op_name, decorate_meta.variant_name))
        if opinfo is None and not decorate_meta.enabled_if:
            # If the OpInfo doesn't exist and it is not enabled, we skip the OpInfo
            # because it could be an OpInfo that is in torch-nightly but not older versions.
            continue
        if opinfo is None:
            raise AssertionError(
                f"Couldn't find OpInfo for {decorate_meta}. Did you need to specify variant_name?"
            )
        decorators = list(opinfo.decorators)
        new_decorator = opinfo_core.DecorateInfo(
            decorate_meta.decorator,
            decorate_meta.test_class_name or test_class_name,
            base_test_name,
            dtypes=decorate_meta.dtypes,
            device_type=decorate_meta.device_type,
            active_if=decorate_meta.enabled_if,
        )
        decorators.append(new_decorator)
        opinfo.decorators = tuple(decorators)

    # This decorator doesn't modify fn in any way
    def wrapped(fn):
        # [FUNCTION DOC] HELPER: Identity decorator from add_decorate_info; returns fn as is.
        return fn

    return wrapped


def duplicate_opinfo(
    opinfos: list[opinfo_core.OpInfo], name: str, new_names: tuple[str, ...]
):
    # [FUNCTION DOC] For every OpInfo in the list whose name equals name, appends a deep copy
    # renamed to each entry of new_names. Names already present in the list produce a warning and
    # are skipped. Mutates the list in place after the scan; if name is absent nothing happens.
    """Duplicate an opinfo in the opinfo database and give it a new name."""
    duplicated = []
    all_info_names = {opinfo.name for opinfo in opinfos}
    for opinfo in opinfos:
        if opinfo.name == name:
            for new_name in new_names:
                if new_name in all_info_names:
                    # NOTE: Avoid duplicating an opinfo that already exists in the database.
                    # New opinfos are expected to be added in torch-nightly.
                    warnings.warn(
                        f"OpInfo {new_name} already exists in the database.",
                        stacklevel=1,
                    )
                    continue
                new_opinfo = copy.deepcopy(opinfo)
                new_opinfo.name = new_name
                duplicated.append(new_opinfo)
    opinfos.extend(duplicated)


def duplicate_opinfo_for_prims(
    opinfos: list[opinfo_core.OpInfo], name: str, prims_name: str | None = None
):
    # [FUNCTION DOC] Finds the OpInfo called name, deep copies it, names the copy with the prims_
    # prefix plus prims_name (default: name) and points its op at the matching function in the
    # prims op namespace. Appends it to the list and returns. If the new name already exists a
    # warning is emitted and nothing is added. Raises RuntimeError if no OpInfo has that name.
    """Duplicate an opinfo in the opinfo database for a prims op.

    The function sets the new OpInfo to use the variation torch.ops.prims.
    The new OpInfo will have the name "prims_{prims_name}" where `prims_name` is the
    name of the prims op. If `prims_name` is None, it will be set to "prims_{name}".

    Args:
        opinfos: The list of opinfo_core.OpInfo to add the new opinfo to.
        name: The name of the opinfo to duplicate.
        prims_name: The name of the prims op. If None, it will be set to `name`.
    """
    if prims_name is None:
        prims_name = name
    # The name of the new OpInfo
    new_name = f"prims_{prims_name}"
    all_info_names = {opinfo.name for opinfo in opinfos}
    for opinfo in opinfos:
        if opinfo.name == name:
            if new_name in all_info_names:
                # NOTE: Avoid duplicating an opinfo that already exists in the database.
                warnings.warn(
                    f"OpInfo {new_name} already exists in the database.", stacklevel=1
                )
                continue
            new_opinfo = copy.deepcopy(opinfo)
            new_opinfo.name = new_name
            new_opinfo.op = getattr(torch.ops.prims, prims_name)
            opinfos.append(new_opinfo)  # noqa: B909
            return
    raise RuntimeError(f"OpInfo '{name}' not found in the database.")


def convert_tensor_to_numpy(input: Any) -> Any:
    # [FUNCTION DOC] Converts a test input to something usable by the ONNX path. Tensors are
    # detached, moved to cpu and turned into numpy arrays; complex tensors and complex scalars are
    # first viewed as real pairs. An empty list becomes an empty int64 array, a list holding
    # tensors is converted element by element, and lists of bool, int or float become numpy arrays
    # (int64 for ints, bool dtype for bools). Anything else is returned unchanged.
    if isinstance(input, torch.Tensor):
        if torch.is_complex(input):
            # from complex to real representation
            input = torch.view_as_real(input)
        return input.detach().cpu().numpy()
    if isinstance(input, complex):
        return torch.view_as_real(torch.tensor(input)).detach().cpu().numpy()
    if isinstance(input, list):
        if len(input) == 0:
            return np.array((), dtype=np.int64)
        if any(isinstance(x, torch.Tensor) for x in input):
            # The list can be Optional[Tensor], e.g. [None, Tensor, None] etc.
            return [convert_tensor_to_numpy(x) for x in input]
        if isinstance(input[0], bool):
            return np.array(input, dtype=np.bool_)

        # Just a sequence of numbers
        if isinstance(input[0], int):
            return np.array(input, dtype=np.int64)
        if isinstance(input[0], float):
            return np.array(input)

    return input


def convert_kwargs_for_onnx(kwargs: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC] Returns a new kwargs mapping suited for the ONNX call: the device key is
    # dropped, the dtype value is mapped through the torch to ONNX dtype table (a missing dtype
    # raises KeyError), tensor values become numpy arrays, other values are copied unchanged.
    """Converts kwargs to be compatible with ONNX Runtime."""
    new_kwargs = {}
    for key, value in kwargs.items():
        if key == "device":
            continue
        if key == "dtype":
            value = _TORCH_DTYPE_TO_ONNX_TYPE[value]
        if isinstance(value, torch.Tensor):
            value = np.array(value.cpu())
        new_kwargs[key] = value
    return new_kwargs


class OrtAbortedError(RuntimeError):
    # [CLASS DOC] Runtime error raised when the ONNX Runtime subprocess died without reporting a
    # result (for example because of a crash or abort). Carries no extra state.
    """ONNX Runtime Aborted."""


def _ort_session_run(serialized_model: bytes, ort_inputs: Mapping[str, Any]):
    # [FUNCTION DOC] HELPER: Runs a serialized ONNX model on the cpu provider of ONNX Runtime with
    # all graph optimizations disabled and returns the list of output arrays. Errors from session
    # creation or execution propagate to the caller.
    """Run a model with ONNX Runtime."""

    # Disable all ORT optimizations
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    session = ort.InferenceSession(
        serialized_model, session_options, providers=("CPUExecutionProvider",)
    )
    return session.run(None, ort_inputs)


def _ort_session_run_return_dict(
    serialized_model: bytes, ort_inputs: Mapping[str, Any], return_dict
) -> None:
    # [FUNCTION DOC] HELPER: Process target that runs the model through the plain session helper
    # and stores the outputs under the results key of the shared mapping, with the error key set
    # to None. On any exception results is set to None and the exception goes under the error key.
    """Run a model with ONNX Runtime and store the results in return_dict."""

    try:
        return_dict["results"] = _ort_session_run(serialized_model, ort_inputs)
        return_dict["error"] = None
    except Exception as e:  # pylint: disable=broad-except
        return_dict["results"] = None
        return_dict["error"] = e


def _safe_ort_session_run(serialized_model: bytes, ort_inputs: Mapping[str, Any]):
    # [FUNCTION DOC] HELPER: Runs the model in a separate process so an ONNX Runtime crash cannot
    # kill the test run. Uses a multiprocessing manager mapping to receive results. Returns the
    # outputs; re-raises the exception reported by the child; raises OrtAbortedError when the
    # child left the mapping empty.
    """Run a model with ONNX Runtime in a separate process.

    Args:
        serialized_model: Serialized ONNX model proto.
        ort_inputs: Inputs to the model.

    Returns:
        The inference result.

    Raises:
        OrtAbortedError if the process did not execute successfully.
    """
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    process = multiprocessing.Process(
        target=_ort_session_run_return_dict,
        args=(serialized_model, ort_inputs, return_dict),
    )
    process.start()
    process.join()
    process.close()
    if not return_dict:
        raise OrtAbortedError
    if return_dict["error"] is not None:
        raise return_dict["error"]
    return return_dict["results"]


def _format_model_and_input_information(onnx_model, inputs):
    # [FUNCTION DOC] HELPER: Builds a diagnostic string with the pretty-printed inputs followed by
    # the textual form of the ONNX model, used in error messages.
    return (
        f"Inputs:\n{pprint.pformat(inputs)}\nModel:\n{onnx.printer.to_text(onnx_model)}"
    )


_TORCH_DTYPE_TO_ONNX_TYPE = {
    torch.bfloat16: ir.DataType.BFLOAT16,
    torch.bool: ir.DataType.BOOL,
    torch.complex128: ir.DataType.COMPLEX128,
    torch.complex64: ir.DataType.COMPLEX64,
    torch.float16: ir.DataType.FLOAT16,
    torch.float32: ir.DataType.FLOAT,
    torch.float64: ir.DataType.DOUBLE,
    torch.float8_e4m3fn: ir.DataType.FLOAT8E4M3FN,
    torch.float8_e4m3fnuz: ir.DataType.FLOAT8E4M3FNUZ,
    torch.float8_e5m2: ir.DataType.FLOAT8E5M2,
    torch.float8_e5m2fnuz: ir.DataType.FLOAT8E5M2FNUZ,
    torch.int16: ir.DataType.INT16,
    torch.int32: ir.DataType.INT32,
    torch.int64: ir.DataType.INT64,
    torch.int8: ir.DataType.INT8,
    torch.uint8: ir.DataType.UINT8,
    torch.uint16: ir.DataType.UINT16,
    torch.uint32: ir.DataType.UINT32,
    torch.uint64: ir.DataType.UINT64,
}


def dtype_op_schema_compatible(dtype: torch.dtype, schema) -> bool:
    # [FUNCTION DOC] Decides whether a torch dtype can be used to test an ONNX function. Returns
    # True when the schema has no inputs or its first input is not named self or input. Otherwise
    # returns whether the ONNX equivalent of dtype is among the dtypes allowed by the type
    # constraint of the first input. Raises AssertionError if that input has no type constraint,
    # and KeyError for a torch dtype missing from the mapping table.
    """Checks if the dtype is compatible with the schema.

    When a dtype is "compatible" with the schema, it means we can use the dtype
    to create sample inputs by OpInfo to test the ONNX function and expect outputs to match.

    Args:
        dtype: The torch dtype used to create sample inputs by OpInfo.
        schema: The OpSignature of the function.

    Returns:
        True if the dtype is compatible with the schema.
    """
    inputs = schema.inputs
    if not inputs:
        # If there are no inputs, we can't check compatibility. Assume it is compatible.
        # e.g. aten_randn has only attributes.
        return True
    if inputs[0].name not in {"self", "input"}:
        # If the name of the first input is not "self" or "input",
        # it is usually an input that is not of the same type as the output.
        # We assume support in this case.
        #
        # For example, `aten_ones(size: IntType, dtype: int = FLOAT.dtype)`
        # has the first input as `size`, which is an integer, but it can support
        # any dtype.
        return True

    # Otherwise we check the type constraints of the first input.
    # For example, when dtype=torch.float32, and the op being tested has the schema
    # ```
    # OpSchema(
    #     name='aten_abs',
    #     domain='pkg.onnxscript.torch_lib',
    #     since_version=1,
    #     doc='abs(Tensor self) -> Tensor',
    #     type_constraints=[OpSchema.TypeConstraintParam(type_param_str='TReal',
    # allowed_type_strs=['tensor(float)', 'tensor(int8)', 'tensor(int16)',
    # 'tensor(int32)', 'tensor(int64)', 'tensor(float16)', 'tensor(double)',
    # 'tensor(bfloat16)'], description='')],
    #     inputs=[OpSchema.FormalParameter(name='self', type_str='TReal',
    # description='', param_option=<FormalParameterOption.Single: 0>,
    # is_homogeneous=True, min_arity=1,
    # differentiation_category=<DifferentiationCategory.Unknown: 0>)],
    #     outputs=[OpSchema.FormalParameter(name='return_val',
    # type_str='TReal', description='',
    # param_option=<FormalParameterOption.Single: 0>, is_homogeneous=True,
    # min_arity=1, differentiation_category=<DifferentiationCategory.Unknown: 0>)],
    #     attributes={}
    # )
    # ```
    # we see the first input type is "TReal", corresponding to the type constraint
    # with allowed types ['tensor(float)', 'tensor(int8)', 'tensor(int16)',
    # 'tensor(int32)', 'tensor(int64)', 'tensor(float16)', 'tensor(double)',
    # 'tensor(bfloat16)'].
    # Since torch.float32 (tensor(float)) is in the allowed types, we return True.

    first_input_type_constraint = inputs[0].type_constraint
    if first_input_type_constraint is None:
        raise AssertionError("first_input_type_constraint is None")
    allowed_types = first_input_type_constraint.allowed_types
    # Here we consider seq(tensor(float)) compatible with tensor(float) as well
    allowed_dtypes = {type_.dtype for type_ in allowed_types}
    return _TORCH_DTYPE_TO_ONNX_TYPE[dtype] in allowed_dtypes


def graph_executor(
    test_name: str,
    outputs: Sequence[Any],
    opset_version: int = TEST_OPSET_VERSION,
) -> Callable[[Callable[..., Any], tuple[Any], dict[str, Any]], None]:
    # [FUNCTION DOC] Returns an executor closure for a test: the closure traces an onnxscript
    # function into a graph and runs it. outputs gives the expected torch outputs, used only to
    # set the shapes and types of graph outputs; opset_version is the ONNX opset of the model.
    """Eagerly executes a function."""

    def _capture_graph_and_evaluate_torch_script_evaluator(
        function: Callable, args, kwargs
    ) -> tuple[Any, onnx.ModelProto]:
        # [FUNCTION DOC] HELPER: Executor returned by graph_executor. Turns numpy arguments (and
        # numpy items inside lists or tuples) and keyword arguments into symbolic graph inputs,
        # calls the function under a tracing evaluator, sets output shapes and dtypes from
        # expected outputs, assembles an ONNX model with the recorded nodes and functions,
        # validates it with the ONNX checker, infers shapes and runs it in ONNX Runtime. Returns
        # the pair of outputs and model proto. Raises AssertionError for an invalid model,
        # RuntimeError for ONNX Runtime errors, and OrtAbortedError on aborts. If
        # CATCH_ORT_SEGFAULT or CREATE_REPRODUCTION_REPORT is 1, it runs in a subprocess; with the
        # latter it also writes a reproduction report on failure.
        """Captures the graph of a function and evaluates it using TorchScriptEvaluator."""

        # Initialize the ONNX graph
        graph = ir.Graph(
            (),
            (),
            nodes=(),
            opset_imports={"": opset_version, "pkg.torch.onnx": 1},
            name="main_graph",
        )
        opset = onnxscript.values.Opset("", opset_version)
        tracer = _building.OpRecorder(opset, {})
        ort_inputs = {}
        onnxscript_args: list[Any] = []
        onnxscript_kwargs = {}
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                input_name = f"input_{i}"
                input = _tensors.SymbolicTensor(
                    opset=opset,
                    name=input_name,
                    shape=ir.Shape(arg.shape),
                    type=ir.TensorType(
                        _TORCH_DTYPE_TO_ONNX_TYPE[torch.tensor(arg).dtype]
                    ),
                )
                graph.inputs.append(input)
                onnxscript_args.append(input)
                ort_inputs[input_name] = arg
            elif isinstance(arg, (list, tuple)):
                # str is also a sequence but we do not want to treat it as a tensor
                sequence_input = []
                for j, subarg in enumerate(arg):
                    if isinstance(subarg, np.ndarray):
                        input_name = f"input_{i}_{j}"
                        tensor = torch.tensor(subarg)
                        input = _tensors.SymbolicTensor(
                            opset=opset,
                            name=input_name,
                            shape=ir.Shape(tensor.shape),
                            type=ir.TensorType(_TORCH_DTYPE_TO_ONNX_TYPE[tensor.dtype]),
                        )
                        graph.inputs.append(input)
                        sequence_input.append(input)
                        ort_inputs[input_name] = subarg
                    else:
                        # Include non-numpy inputs as-is
                        # For example, it could be a None value that we want to keep
                        sequence_input.append(subarg)
                onnxscript_args.append(sequence_input)
            else:
                onnxscript_args.append(arg)
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                input = _tensors.SymbolicTensor(
                    opset=opset,
                    name=key,
                    shape=ir.Shape(torch.tensor(value).shape),
                    type=ir.TensorType(
                        _TORCH_DTYPE_TO_ONNX_TYPE[torch.tensor(value).dtype]
                    ),
                )
                graph.inputs.append(input)
                ort_inputs[key] = value
                onnxscript_kwargs[key] = input
            else:
                onnxscript_kwargs[key] = value

        with onnxscript.evaluator.default_as(tracer):
            symbolic_outputs = function(*onnxscript_args, **onnxscript_kwargs)
        if not isinstance(symbolic_outputs, Sequence):
            symbolic_outputs = (symbolic_outputs,)

        # We need to set the size of the output tensors for the ONNX model to be valid
        for output, symbolic_output in zip(outputs, symbolic_outputs):
            if isinstance(output, Sequence):
                # Output is a sequence
                elem_dtype = _TORCH_DTYPE_TO_ONNX_TYPE[output[0].dtype]
                symbolic_output.type = ir.SequenceType(ir.TensorType(elem_dtype))
                continue
            output = (
                output
                if isinstance(output, torch.Tensor)
                else torch.tensor(output, device="cpu")
            )
            symbolic_output.shape = ir.Shape(output.shape)
            symbolic_output.dtype = _TORCH_DTYPE_TO_ONNX_TYPE[output.dtype]

        graph.outputs.extend(symbolic_outputs)
        graph.extend(tracer.nodes)
        onnx_model = ir.Model(graph, ir_version=10, producer_name="torch_test")
        for identifier, onnxscript_function in tracer.functions.items():
            if identifier in onnx_model.functions:
                continue
            if isinstance(onnxscript_function, ir.Function):
                ir_function = onnxscript_function
            else:
                # TODO: Get IR function directly when onnxscript is updated
                proto = onnxscript_function.to_function_proto()
                ir_function = ir.serde.deserialize_function(proto)
            onnx_model.functions[identifier] = ir_function
        _ir_passes.add_opset_imports(onnx_model)
        # Make sure the model is valid
        model_proto = ir.to_proto(onnx_model)
        try:
            onnx.checker.check_model(model_proto, full_check=True)
        except (onnx.checker.ValidationError, onnx.shape_inference.InferenceError) as e:
            raise AssertionError(f"ONNX model is invalid. Model:\n{onnx_model}") from e
        model_proto = onnx.shape_inference.infer_shapes(model_proto, data_prop=True)
        try:
            if (
                os.environ.get("CATCH_ORT_SEGFAULT") == "1"
                or os.environ.get("CREATE_REPRODUCTION_REPORT") == "1"
            ):
                # Use an individual process to run ONNX Runtime to catch segfaults
                return _safe_ort_session_run(
                    model_proto.SerializeToString(), ort_inputs
                ), model_proto

            return _ort_session_run(
                model_proto.SerializeToString(), ort_inputs
            ), model_proto
        except (
            # pylint: disable=c-extension-no-member
            onnxruntime.capi.onnxruntime_pybind11_state.Fail,
            onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException,
            onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument,
            onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph,
            onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented,
            # pylint: enable=c-extension-no-member
        ) as e:
            if os.environ.get("CREATE_REPRODUCTION_REPORT") == "1":
                error_reproduction.create_reproduction_report(
                    test_name,
                    model_proto,
                    ort_inputs,
                    e,
                    "test/onnx/torchlib/test_ops.py",
                )
            raise RuntimeError(
                "ONNX Runtime failed to evaluate:\n"
                + _format_model_and_input_information(model_proto, ort_inputs)
            ) from e
        except OrtAbortedError as e:
            if os.environ.get("CREATE_REPRODUCTION_REPORT") == "1":
                # Save the model and inputs to a file for reproduction
                error_reproduction.create_reproduction_report(
                    test_name,
                    model_proto,
                    ort_inputs,
                    e,
                    "test/onnx/torchlib/test_ops.py",
                )
            raise OrtAbortedError(
                "ONNX Runtime aborted:\n"
                + _format_model_and_input_information(model_proto, ort_inputs)
            ) from e
        except Exception as e:
            if os.environ.get("CREATE_REPRODUCTION_REPORT") == "1":
                error_reproduction.create_reproduction_report(
                    test_name,
                    model_proto,
                    ort_inputs,
                    e,
                    "test/onnx/torchlib/test_ops.py",
                )
            raise

    return _capture_graph_and_evaluate_torch_script_evaluator


@contextlib.contextmanager
def normal_xfail_skip_test_behaviors(
    test_behavior: str | None = None, reason: str | None = None
):
    # [FUNCTION DOC] Context manager applying the skip or xfail behavior to a test body. For skip
    # it calls pytest skip immediately without running the body. For xfail an exception in the
    # body becomes a pytest xfail with the reason, while a body that succeeds fails the test as an
    # unexpected pass. With no behavior, exceptions propagate unchanged.
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
    except Exception:  # pylint: disable=broad-exception-caught
        if test_behavior is None:
            raise
        if test_behavior == "xfail":
            pytest.xfail(reason=reason)
    else:
        if test_behavior == "xfail":
            pytest.fail("Test unexpectedly passed")
