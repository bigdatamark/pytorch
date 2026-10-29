# [PACKAGE DOC] Torchfuzz: a fuzzer that generates random PyTorch operation graphs, converts
# them to runnable Python programs and checks eager vs compiled behavior. This package re-exports
# the main API: code generation (DeviceInfo, FuzzTemplate, template lookup and initialization),
# the operator registry, graph/spec fuzzing (OperationGraph, fuzz_operation_graph, fuzz_spec)
# and the spec types (Spec, TensorSpec, ScalarSpec). Importing it imports codegen, operators,
# ops_fuzzer and tensor_fuzzer, which may load the configured device plugin.
"""Torchfuzz package for generating and testing random PyTorch operations."""

# Make key classes available at package level
from .codegen import (
    DeviceInfo,
    FuzzTemplate,
    get_device_info,
    get_template_names,
    initialize_codegen,
    make_template,
)
from .operators import get_operator, list_operators, register_operator
from .ops_fuzzer import fuzz_operation_graph, fuzz_spec, OperationGraph
from .tensor_fuzzer import ScalarSpec, Spec, TensorSpec


__all__ = [
    "DeviceInfo",
    "FuzzTemplate",
    "OperationGraph",
    "ScalarSpec",
    "Spec",
    "TensorSpec",
    "fuzz_operation_graph",
    "fuzz_spec",
    "get_device_info",
    "get_operator",
    "get_template_names",
    "initialize_codegen",
    "list_operators",
    "make_template",
    "register_operator",
]
