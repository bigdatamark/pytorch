# mypy: allow-untyped-defs
# [MODULE DOC] Jiterator support: compile elementwise CUDA kernels from C++ template code strings at
# call time. Contains a parser for the code string, a callable wrapper that validates arguments and
# launches the native compile-and-launch routine, and two private factories (single output by value,
# multiple outputs by reference). Exports nothing publicly. No import-time side effects.
import re
from collections.abc import Callable

import torch
from torch import Tensor


__all__: list[str] = []


class _CodeParser:
    # [CLASS DOC] HELPER: Parses a single C++ function template string into its parts: template
    # parameters, return type, function name, parameters and body, stored as attributes.
    def __init__(self, code_string: str):
        # [FUNCTION DOC] HELPER: Match the code string against a pattern for a templated function
        # (multi-line allowed) and store the captured pieces. Raises a generic Exception with the code
        # echoed if the string does not match.
        optional_ws = r"\s*"
        required_ws = r"\s+"
        template_params = r"(?P<template_params>\<.+\>)"
        return_type = r"(?P<return_type>\w+)"
        function_name = r"(?P<function_name>\w+)"
        function_params = r"(?P<function_params>\(.+\))"
        function_body = r"(?P<function_body>\{.+\})"

        pattern = (
            optional_ws
            + "template"
            + optional_ws
            + template_params
            + optional_ws
            + return_type
            + required_ws
            + function_name
            + optional_ws
            + function_params
            + optional_ws
            + function_body
            + optional_ws
        )

        result = re.match(
            pattern, code_string, re.DOTALL
        )  # DOTALL for matching multiline

        if result is None:
            raise Exception(  # noqa: TRY002
                f"Couldn't parse code, please check correctness:\n {code_string}"
            )

        self.template_params = result["template_params"]
        self.return_type = result["return_type"]
        self.function_name = result["function_name"]
        self.function_params = result["function_params"]
        self.function_body = result["function_body"]


class _JittedFunction:
    # [CLASS DOC] HELPER: Callable wrapper for a jiterator kernel. Stores the code string, whether
    # outputs are returned by reference, the output count, the parsed kernel name, default keyword
    # arguments and whether CUDA was available at construction. Calling it launches the kernel.
    def __init__(
        self, code_string: str, return_by_ref: bool, num_outputs: int, **kwargs
    ):
        # [FUNCTION DOC] HELPER: Record settings and parse the kernel name from the code string.
        # Raises AssertionError if returning by value with other than one output; parse errors propagate.
        self.code_string = code_string

        if not (return_by_ref or num_outputs == 1):
            raise AssertionError("Return by value only works for single output.")
        self.return_by_ref = return_by_ref
        self.num_outputs = num_outputs

        parsed_code = _CodeParser(code_string)
        self.kernel_name = parsed_code.function_name

        self.kwargs_dict = kwargs
        self.is_cuda_available = torch.cuda.is_available()

    def __call__(self, *tensors: Tensor, **kwargs):
        # [FUNCTION DOC] HELPER: Launch the compiled kernel on the input tensors. Overrides default
        # keyword values with the provided ones. Raises AssertionError if no GPU was available at
        # construction or more than 8 tensors are given, and KeyError for an undeclared keyword.
        # Returns the native launch result (output tensor or tensors).
        # Jiterator follow torch.cuda's lazy initialization behavior
        # Defer checking cuda's availability at the function invocation time
        if not self.is_cuda_available:
            raise AssertionError(
                "Jiterator is only supported on CUDA and ROCm GPUs, none are available."
            )

        if len(tensors) > 8:
            raise AssertionError(
                f"jiterator only supports up to 8 tensor inputs, got {len(tensors)}"
            )

        expanded_kwargs = self.kwargs_dict.copy()
        for key, value in kwargs.items():
            if key in self.kwargs_dict:
                expanded_kwargs[key] = value
            else:
                raise KeyError(f"{key} is not declared in function definition")

        return torch._C._cuda_jiterator_compile_and_launch_kernel(
            self.code_string,
            self.kernel_name,
            self.return_by_ref,
            self.num_outputs,
            tensors,
            expanded_kwargs,
        )


def _create_jit_fn(code_string: str, **kwargs) -> Callable:
    # [FUNCTION DOC] Build a callable for an elementwise CUDA kernel returning a single output by value
    # from the given C++ template code string, with default keyword arguments. Compilation happens at
    # call time. Raises if the code string cannot be parsed.
    """
    Create a jiterator-generated cuda kernel for an elementwise op.

    The code string has to be a valid CUDA function that describes the computation for a single element. The code
    string has to follow the c++ template pattern, as shown in the example below. This function will be inlined
    into elementwise kernel template, and compiled on the fly. Compiled kernel will be cached in memory, as well as
    local temp dir.

    Jiterator-generated kernels accept noncontiguous tensors, and support broadcasting and type promotion.

    Args:
        code_string (str): CUDA code string to be compiled by jiterator. The entry functor must return by value.
        kwargs (Dict, optional): Keyword arguments for generated function

    Example::

        code_string = "template <typename T> T my_kernel(T x, T y, T alpha) { return -x + alpha * y; }"
        jitted_fn = create_jit_fn(code_string, alpha=1.0)
        a = torch.rand(3, device="cuda")
        b = torch.rand(3, device="cuda")
        # invoke jitted function like a regular python function
        result = jitted_fn(a, b, alpha=3.14)

    code_string also allows multiple function definitions, and the last function will be treated as the entry function.

    Example::

        code_string = (
            "template <typename T> T util_fn(T x, T y) { return ::sin(x) + ::cos(y); }"
        )
        code_string += "template <typename T> T my_kernel(T x, T y, T val) { return ::min(val, util_fn(x, y)); }"
        jitted_fn = create_jit_fn(code_string, val=0.0)
        a = torch.rand(3, device="cuda")
        b = torch.rand(3, device="cuda")
        # invoke jitted function like a regular python function
        result = jitted_fn(a, b)  # using default val=0.0

    Jiterator can be used together with python registration to override an operator's cuda kernel.
    Following example is overriding gelu's cuda kernel with relu.

    Example::

        code_string = "template <typename T> T my_gelu(T a) { return a > 0 ? a : 0; }"
        my_gelu = create_jit_fn(code_string)
        my_lib = torch.library.Library("aten", "IMPL")
        my_lib.impl("aten::gelu", my_gelu, "CUDA")
        # torch.nn.GELU and torch.nn.function.gelu are now overridden
        a = torch.rand(3, device="cuda")
        torch.allclose(torch.nn.functional.gelu(a), torch.nn.functional.relu(a))

    .. warning::
        This API is in beta and may change in future releases.

    .. warning::
        This API only supports up to 8 inputs and 1 output

    .. warning::
        All input tensors must live in CUDA device
    """
    return _JittedFunction(code_string, return_by_ref=False, num_outputs=1, **kwargs)


def _create_multi_output_jit_fn(
    code_string: str, num_outputs: int, **kwargs
) -> Callable:
    # [FUNCTION DOC] Build a callable for an elementwise CUDA kernel that writes one or more outputs by
    # reference, with the given number of outputs and default keyword arguments. Raises if the code
    # string cannot be parsed.
    """
    Create a jiterator-generated cuda kernel for an elementwise op that supports returning one or more outputs.

    Args:
        code_string (str): CUDA code string to be compiled by jiterator. The entry functor must return value by reference.
        num_outputs(int): number of outputs return by the kernel
        kwargs (Dict, optional): Keyword arguments for generated function

    Example::

        code_string = "template <typename T> void my_kernel(T x, T y, T alpha, T& out) { out = -x + alpha * y; }"
        jitted_fn = create_jit_fn(code_string, alpha=1.0)
        a = torch.rand(3, device="cuda")
        b = torch.rand(3, device="cuda")
        # invoke jitted function like a regular python function
        result = jitted_fn(a, b, alpha=3.14)

    .. warning::
        This API is in beta and may change in future releases.

    .. warning::
        This API only supports up to 8 inputs and 8 outputs
    """
    return _JittedFunction(
        code_string, return_by_ref=True, num_outputs=num_outputs, **kwargs
    )
