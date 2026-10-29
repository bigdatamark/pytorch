# [MODULE DOC] Helpers for the Python/C++ API parity tests of torch.nn modules and functionals.
# Defines the namedtuples describing a module or functional test (TorchNNModuleTestParams,
# TorchNNFunctionalTestParams) and CppArg, the shared C++ harness source string (file read/write
# of IValues and a non-equal random tensor generator), the failure help message, and utilities
# to build the C++ argument statements, serialize argument dictionaries, register tests on a
# unittest class, and clean up temp folders. Importing has no side effects beyond imports.
import os
import shutil
import unittest
import warnings
from collections import namedtuple

import torch
import torch.testing._internal.common_nn as common_nn
import torch.utils.cpp_extension
from torch.testing._internal.common_cuda import TEST_CUDA


# Note that this namedtuple is for C++ parity test mechanism's internal use.
# For guidance on how to add a new C++ parity test, please see
# NOTE [How to check NN module / functional API parity between Python and C++ frontends]
TorchNNModuleTestParams = namedtuple(
    "TorchNNModuleTestParams",
    [
        # NN module name (e.g. "BCELoss")
        "module_name",
        # Unique identifier for this module config (e.g. "BCELoss_weights_cuda")
        "module_variant_name",
        # An instance of an NN test class (e.g. `CriterionTest`) which stores
        # necessary information (e.g. input / target / extra_args) for running the Python test
        "test_instance",
        # Constructor arguments passed to the C++ module constructor, which must be
        # strictly equivalent to the Python module constructor arguments
        # (e.g. `torch::nn::BCELossOptions().weight(torch::rand(10))`,
        # which is strictly equivalent to passing `torch.rand(10)` to `torch.nn.BCELoss`
        # constructor in Python)
        "cpp_constructor_args",
        # All arguments used in NN module's forward pass.
        # Please see `compute_arg_dict` function for details on how we construct this dict.
        # (e.g.
        # ```
        # arg_dict = {
        #     'input': [python_input_tensor],
        #     'target': [python_target_tensor],
        #     'extra_args': [],
        #     'other': [],
        # }
        # ```
        # )
        "arg_dict",
        # Whether we expect this NN module test to pass the Python/C++ parity test
        # (e.g. `True`)
        "has_parity",
        # Device (e.g. "cuda")
        "device",
        # Temporary folder to store C++ outputs (to be compared with Python outputs later)
        "cpp_tmp_folder",
    ],
)

# Note that this namedtuple is for C++ parity test mechanism's internal use.
# For guidance on how to add a new C++ parity test, please see
# NOTE [How to check NN module / functional API parity between Python and C++ frontends]
TorchNNFunctionalTestParams = namedtuple(
    "TorchNNFunctionalTestParams",
    [
        # NN functional name (e.g. "binary_cross_entropy")
        "functional_name",
        # Unique identifier for this functional config (e.g. "BCELoss_no_reduce_cuda")
        "functional_variant_name",
        # An instance of an NN test class (e.g. `NewModuleTest`) which stores
        # necessary information (e.g. input / target / extra_args) for running the Python test
        "test_instance",
        # The C++ function call that is strictly equivalent to the Python function call
        # (e.g. "F::binary_cross_entropy(
        #            i, t.to(i.options()),F::BinaryCrossEntropyFuncOptions().reduction(torch::kNone))",
        # which is strictly equivalent to `F.binary_cross_entropy(i, t.type_as(i), reduction='none')` in Python)
        "cpp_function_call",
        # All arguments used in NN functional's function call.
        # Please see `compute_arg_dict` function for details on how we construct this dict.
        # (e.g.
        # ```
        # arg_dict = {
        #     'input': [python_input_tensor],
        #     'target': [python_target_tensor],
        #     'extra_args': [],
        #     'other': [],
        # }
        # ```
        # )
        "arg_dict",
        # Whether we expect this NN functional test to pass the Python/C++ parity test
        # (e.g. `True`)
        "has_parity",
        # Device (e.g. "cuda")
        "device",
        # Temporary folder to store C++ outputs (to be compared with Python outputs later)
        "cpp_tmp_folder",
    ],
)

CppArg = namedtuple("CppArg", ["name", "value"])

TORCH_NN_COMMON_TEST_HARNESS = """
#include <torch/script.h>

void write_ivalue_to_file(const torch::IValue& ivalue, const std::string& file_path) {
    auto bytes = torch::jit::pickle_save(ivalue);
    std::ofstream fout(file_path, std::ios::out | std::ios::binary);
    fout.write(bytes.data(), bytes.size());
    fout.close();
}

c10::Dict<std::string, torch::Tensor> load_dict_from_file(const std::string& file_path) {
    c10::Dict<std::string, torch::Tensor> arg_dict;
    auto arg_dict_module = torch::jit::load(file_path);
    for (const auto& p : arg_dict_module.named_buffers(/*recurse=*/false)) {
        arg_dict.insert(p.name, p.value);
    }
    return arg_dict;
}

// Generates rand tensor with non-equal values. This ensures that duplicate
// values won't be causing test failure for modules like MaxPooling.
// size should be small, otherwise randperm fails / long overflows.
torch::Tensor _rand_tensor_non_equal(torch::IntArrayRef size) {
    int64_t total = 1;
    for (int64_t elem : size) {
        total *= elem;
    }
    return torch::randperm(total).view(size).to(torch::kDouble);
}
"""


def compile_cpp_code_inline(name, cpp_sources, functions):
    # [FUNCTION DOC] HELPER: Compiles the given C++ sources into an extension with load_inline
    # (debug symbols on, not verbose) and returns the loaded module exposing the listed
    # functions. Compilation errors propagate from the cpp_extension machinery.
    cpp_module = torch.utils.cpp_extension.load_inline(
        name=name,
        cpp_sources=cpp_sources,
        extra_cflags=[
            "-g"
        ],  # Enable debug symbols by default for debugging test failures.
        functions=functions,
        verbose=False,
    )
    return cpp_module


def compute_temp_file_path(cpp_tmp_folder, variant_name, file_suffix):
    # [FUNCTION DOC] HELPER: Returns the path of the temp file named after the variant name and
    # the suffix, with a pt extension, inside the given folder.
    return os.path.join(cpp_tmp_folder, f"{variant_name}_{file_suffix}.pt")


def is_torch_nn_functional_test(test_params_dict):
    # [FUNCTION DOC] HELPER: Returns true if the test parameter mapping's constructor entry
    # (default empty) has a textual form mentioning wrap_functional, which identifies functional
    # tests as opposed to module tests.
    return "wrap_functional" in str(test_params_dict.get("constructor", ""))


def convert_to_list(python_input):
    # [FUNCTION DOC] HELPER: Normalizes the input to a list; a single tensor becomes a
    # one-element list and any other iterable is converted with list.
    if isinstance(python_input, torch.Tensor):
        return [python_input]
    else:
        return list(python_input)


def set_python_tensors_requires_grad(python_tensors):
    # [FUNCTION DOC] HELPER: Returns a list where every tensor that is not of long dtype has
    # requires_grad enabled in place; long tensors are passed through unchanged.
    return [
        tensor.requires_grad_(True) if tensor.dtype != torch.long else tensor
        for tensor in python_tensors
    ]


def move_python_tensors_to_device(python_tensors, device):
    # [FUNCTION DOC] HELPER: Returns a list of the given tensors moved to the device (new
    # tensors where a copy is needed).
    return [tensor.to(device) for tensor in python_tensors]


def has_test(unit_test_class, test_name):
    # [FUNCTION DOC] HELPER: Returns whether the test class already has an attribute with the
    # given test name.
    return hasattr(unit_test_class, test_name)


def add_test(unit_test_class, test_name, test_fn):
    # [FUNCTION DOC] HELPER: Attaches test_fn to the unittest class under test_name (mutates the
    # class). Raises RuntimeError if an attribute with that name already exists.
    if has_test(unit_test_class, test_name):
        raise RuntimeError("Found two tests with the same name: " + test_name)
    setattr(unit_test_class, test_name, test_fn)


def set_cpp_tensors_requires_grad(cpp_tensor_stmts, python_tensors):
    # [FUNCTION DOC] HELPER: Given C++ tensor statements and the matching (name, tensor) pairs
    # of Python tensors, returns the statements with a requires-grad call appended for each
    # whose Python tensor is not long dtype. Raises an error if the two lengths differ.
    if len(cpp_tensor_stmts) != len(python_tensors):
        raise AssertionError(
            f"Expected len(cpp_tensor_stmts) == len(python_tensors), got {len(cpp_tensor_stmts)} vs {len(python_tensors)}"
        )
    return [
        f"{tensor_stmt}.requires_grad_(true)"
        if tensor.dtype != torch.long
        else tensor_stmt
        for tensor_stmt, (_, tensor) in zip(cpp_tensor_stmts, python_tensors)
    ]


def move_cpp_tensors_to_device(cpp_tensor_stmts, device):
    # [FUNCTION DOC] HELPER: Returns the C++ tensor statements each with a call that moves the
    # tensor to the given device string appended.
    return [f'{tensor_stmt}.to("{device}")' for tensor_stmt in cpp_tensor_stmts]


def is_criterion_test(test_instance):
    # [FUNCTION DOC] HELPER: Returns true if the test instance is a CriterionTest from common_nn
    # (i.e. has a target).
    return isinstance(test_instance, common_nn.CriterionTest)


# This function computes the following:
# - What variable declaration statements should show up in the C++ parity test function
# - What arguments should be passed into the C++ module/functional's forward function
#
# For example, for the "L1Loss" test, the return values from this function are:
# ```
# // Note that `arg_dict` stores all tensor values we transfer from Python to C++
# cpp_args_construction_stmts = [
#   "auto i0 = arg_dict.at("i0").to("cpu").requires_grad_(true)",
#   "auto t0 = arg_dict.at("t0").to("cpu")",
# ],
# cpp_forward_args_symbols = [
#   "i0",
#   "t0",
# ]
# ```
def compute_cpp_args_construction_stmts_and_forward_arg_symbols(test_params):
    # [FUNCTION DOC] HELPER: For the test params, builds the C++ statements that declare each
    # forward argument by looking it up by name in arg_dict (input args also get requires-grad,
    # all get moved to the test device), followed by the declarations of the "other" args.
    # Returns a pair: the full ordered list of declaration statements (input, target, extra,
    # other) and the list of argument names to pass to forward (input, target, extra only;
    # "other" args are excluded).
    device = test_params.device
    cpp_forward_args_symbols = []

    def add_cpp_forward_args(args):
        # [FUNCTION DOC] HELPER: Builds a lookup statement for each (name, value) pair, records
        # the names in the enclosing forward-argument symbol list (mutation), and returns the
        # statements.
        args_stmts = []
        for arg_name, _ in args:
            args_stmts.append(f'auto {arg_name} = arg_dict.at("{arg_name}")')
            cpp_forward_args_symbols.append(arg_name)
        return args_stmts

    cpp_forward_input_args_stmts = set_cpp_tensors_requires_grad(
        move_cpp_tensors_to_device(
            add_cpp_forward_args(test_params.arg_dict["input"]), device
        ),
        test_params.arg_dict["input"],
    )
    cpp_forward_target_args_stmts = move_cpp_tensors_to_device(
        add_cpp_forward_args(test_params.arg_dict["target"]), device
    )
    cpp_forward_extra_args_stmts = move_cpp_tensors_to_device(
        add_cpp_forward_args(test_params.arg_dict["extra_args"]), device
    )

    # Build the list of other arguments needed
    cpp_other_args_stmts = []
    for arg_name, _ in test_params.arg_dict["other"]:
        cpp_other_args_stmts.append(f'auto {arg_name} = arg_dict.at("{arg_name}")')
    cpp_other_args_stmts = move_cpp_tensors_to_device(cpp_other_args_stmts, device)

    cpp_args_construction_stmts = (
        cpp_forward_input_args_stmts
        + cpp_forward_target_args_stmts
        + cpp_forward_extra_args_stmts
        + cpp_other_args_stmts
    )

    return cpp_args_construction_stmts, cpp_forward_args_symbols


def serialize_arg_dict_as_script_module(arg_dict):
    # [FUNCTION DOC] HELPER: Flattens the input, target, extra_args and other entries of
    # arg_dict into one name-to-tensor mapping, registers each as a buffer on a fresh empty
    # module, and returns it scripted with TorchScript so C++ can load it. Raises an error if
    # any value is not a tensor.
    arg_dict_flat = dict(
        arg_dict["input"]
        + arg_dict["target"]
        + arg_dict["extra_args"]
        + arg_dict["other"]
    )
    arg_dict_module = torch.nn.Module()
    for arg_name, arg_value in arg_dict_flat.items():
        if not isinstance(arg_value, torch.Tensor):
            raise AssertionError(
                f"Expected arg_value to be Tensor, got {type(arg_value)} for arg '{arg_name}'"
            )
        arg_dict_module.register_buffer(arg_name, arg_value)

    return torch.jit.script(arg_dict_module)


# NOTE: any argument symbol used in `cpp_constructor_args` / `cpp_options_args` / `cpp_function_call`
# must have a mapping in `cpp_var_map`.
#
# The mapping can take one of the following formats:
#
# 1. `argument_name` -> Python value
# 2. `argument_name` -> '_get_input()' (which means `argument_name` in C++ will be bound to `test_instance._get_input()`)
#
# For example:
# ```
# def bceloss_weights_no_reduce_test():
#     t = torch.randn(15, 10).gt(0).double()
#     weights = torch.rand(10)
#     return dict(
#         fullname='BCELoss_weights_no_reduce',
#         constructor=wrap_functional(
#             lambda i: F.binary_cross_entropy(i, t.type_as(i),
#                                              weight=weights.type_as(i), reduction='none')),
#         cpp_function_call='''F::binary_cross_entropy(
#                              i, t.to(i.options()),
#                              F::BinaryCrossEntropyFuncOptions()
#                              .weight(weights.to(i.options()))
#                              .reduction(torch::kNone))''',
#         input_fn=lambda: torch.rand(15, 10).clamp_(2.8e-2, 1 - 2.8e-2),
#         cpp_var_map={'i': '_get_input()', 't': t, 'weights': weights},
#         reference_fn=lambda i, p, m: -(t * i.log() + (1 - t) * (1 - i).log()) * weights,
#     )
# ```
def compute_arg_dict(test_params_dict, test_instance):
    # [FUNCTION DOC] HELPER: Builds the argument dictionary (keys input, target, extra_args,
    # other, each a list of CppArg) for a test. Inputs are named i0, i1, ..., targets t0...
    # (criterion tests only), extra args e0...; entries of the optional cpp_var_map from the
    # params dict are appended to "other" when the value is a tensor or the special string
    # meaning the test's input. Raises RuntimeError for any other string or value type.
    arg_dict = {
        "input": [],
        "target": [],
        "extra_args": [],
        "other": [],
    }

    def put_args_into_arg_dict(arg_type, arg_type_prefix, args):
        # [FUNCTION DOC] HELPER: Appends a CppArg named by the prefix plus the index for each
        # argument to the list under arg_type in the enclosing arg_dict (mutation).
        for i, arg in enumerate(args):
            arg_dict[arg_type].append(CppArg(name=arg_type_prefix + str(i), value=arg))

    put_args_into_arg_dict("input", "i", convert_to_list(test_instance._get_input()))
    if is_criterion_test(test_instance):
        put_args_into_arg_dict(
            "target", "t", convert_to_list(test_instance._get_target())
        )
    if test_instance.extra_args:
        put_args_into_arg_dict(
            "extra_args", "e", convert_to_list(test_instance.extra_args)
        )

    cpp_var_map = test_params_dict.get("cpp_var_map", {})
    for arg_name, arg_value in cpp_var_map.items():
        if isinstance(arg_value, str):
            if arg_value == "_get_input()":
                arg_dict["other"].append(
                    CppArg(name=arg_name, value=test_instance._get_input())
                )
            else:
                raise RuntimeError(
                    f"`{arg_name}` has unsupported string value: {arg_value}"
                )
        elif isinstance(arg_value, torch.Tensor):
            arg_dict["other"].append(CppArg(name=arg_name, value=arg_value))
        else:
            raise RuntimeError(f"`{arg_name}` has unsupported value: {arg_value}")

    return arg_dict


def decorate_test_fn(test_fn, test_cuda, has_impl_parity, device):
    # [FUNCTION DOC] HELPER: Wraps test_fn with unittest decorators and returns it: for cuda
    # device, skip when CUDA is unavailable or when the test is excluded from CUDA; if
    # has_impl_parity is false, mark as expected failure.
    if device == "cuda":
        test_fn = unittest.skipIf(not TEST_CUDA, "CUDA unavailable")(test_fn)
        test_fn = unittest.skipIf(not test_cuda, "Excluded from CUDA tests")(test_fn)

    # If `Implementation Parity` entry in parity table for this module is `No`,
    # or `has_parity` entry in test params dict is `False`, we mark the test as
    # expected failure.
    if not has_impl_parity:
        test_fn = unittest.expectedFailure(test_fn)

    return test_fn


MESSAGE_HOW_TO_FIX_CPP_PARITY_TEST_FAILURE = """
What should I do when C++ API parity test is failing?

- If you are changing the implementation of an existing `torch.nn` module / `torch.nn.functional` function:
Answer: Ideally you should also change the C++ API implementation for that module / function
(you can start by searching for the module / function name in `torch/csrc/api/` folder).

- If you are adding a new test for an existing `torch.nn` module / `torch.nn.functional` function:
Answer: Ideally you should fix the C++ API implementation for that module / function
to exactly match the Python API implementation (you can start by searching for the module /
function name in `torch/csrc/api/` folder).

- If you are adding a test for a *new* `torch.nn` module / `torch.nn.functional` function:
Answer: Ideally you should add the corresponding C++ API implementation for that module / function,
and it should exactly match the Python API implementation. (We have done a large effort on this
which is tracked at https://github.com/pytorch/pytorch/issues/25883.)

However, if any of the above is proven to be too complicated, you can just add
`test_cpp_api_parity=False` to any failing test in `torch/testing/_internal/common_nn.py`,
and the C++ API parity test will be skipped accordingly. Note that you should
also file an issue when you do this.

For more details on how to add a C++ API parity test, please see:
NOTE [How to check NN module / functional API parity between Python and C++ frontends]
"""


def generate_error_msg(name, cpp_value, python_value):
    # [FUNCTION DOC] HELPER: Builds the parity-failure message naming the value, the C++ and
    # Python values, and the how-to-fix guidance text.
    return (
        f"Parity test failed: {name} in C++ has value: {cpp_value}, "
        f"which does not match the corresponding value in Python: {python_value}.\n{MESSAGE_HOW_TO_FIX_CPP_PARITY_TEST_FAILURE}"
    )


def try_remove_folder(folder_path):
    # [FUNCTION DOC] HELPER: Removes the folder tree if it exists; any error during removal is
    # only reported as a warning, never raised.
    if os.path.exists(folder_path):
        # Don't block the process if this fails, but show the error message as warning.
        try:
            shutil.rmtree(folder_path)
        except Exception as e:
            warnings.warn(
                f"Non-blocking folder removal fails with the following error:\n{str(e)}"
            )
