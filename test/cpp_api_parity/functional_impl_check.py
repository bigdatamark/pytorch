# [MODULE DOC] Implementation-parity check between Python torch.nn.functional functions and
# the C++ torch::nn::functional counterparts. From test parameter dicts (common_nn) it builds
# per-device test params, generates a C++ forward-test function per functional from a string
# template, compiles all of them together inline, and registers unit tests on a test class.
# Each test runs the Python functional, serializes its inputs, runs the C++ one on them and
# compares the saved outputs. Main entry points: write_test_to_test_class and build_cpp_tests.
# The template constant has no import-time side effects; importing pulls in cpp_api_parity
# helpers.
# The purpose of this test is to check that we have implementation parity between
# a Python `torch.nn.functional` function and its corresponding C++ `torch::nn::functional`
# function. Concretely, this test does the following:
#
# 1. Get a test params dict from common_nn.py, run forward pass on the Python functional
# created using the test params.
#
# 2. Serialize the Python functional's forward input arguments, deserialize them
# in C++ and use them as input for the C++ functional's forward pass.
#
# 3. Run the forward pass on the C++ functional, and serialize the C++ functional's
# forward output.
#
# 4. Compare Python/C++ functional's forward output. If they are the same, then we
# have implementation parity between Python/C++ module.

import os
import pprint
import re
import tempfile
from string import Template

import torch
from cpp_api_parity.sample_functional import SAMPLE_FUNCTIONAL_CPP_SOURCE
from cpp_api_parity.utils import (
    add_test,
    compile_cpp_code_inline,
    compute_arg_dict,
    compute_cpp_args_construction_stmts_and_forward_arg_symbols,
    compute_temp_file_path,
    decorate_test_fn,
    generate_error_msg,
    is_torch_nn_functional_test,
    move_python_tensors_to_device,
    serialize_arg_dict_as_script_module,
    set_python_tensors_requires_grad,
    TORCH_NN_COMMON_TEST_HARNESS,
    TorchNNFunctionalTestParams,
    try_remove_folder,
)


# Expected substitutions:
#
# ${functional_variant_name}  (e.g. `BCELoss_no_reduce`)
# ${cpp_args_construction_stmts}
# ${cpp_function_call}
TORCH_NN_FUNCTIONAL_TEST_FORWARD = Template(
    """
void ${functional_variant_name}_test_forward(
    const std::string& arg_dict_file_path,
    const std::string& forward_output_file_path) {
  pybind11::gil_scoped_release no_gil;

  namespace F = torch::nn::functional;

  // Declare arguments
  auto arg_dict = load_dict_from_file(arg_dict_file_path);
  ${cpp_args_construction_stmts};

  // Some functionals (such as `F::rrelu`) create random tensors in their call path.
  // To make sure the random tensors created are the same in Python/C++, we need
  // to set the RNG seed manually.
  torch::manual_seed(0);

  // Run function with arguments
  auto cpp_output = ${cpp_function_call};

  // Save the output into a file to be compared in Python later
  write_ivalue_to_file(torch::IValue(cpp_output), forward_output_file_path);
}
"""
)


def run_forward(unit_test_class, test_params):
    # [FUNCTION DOC] Runs the Python functional of a test once and returns its output. Takes the
    # input, target and extra_args entries from the arg dict, moves them to the test device (input
    # tensors also get requires_grad set), seeds the RNG with 0 so random functionals match C++,
    # then calls the constructed functional with the combined inputs. unit_test_class is unused.
    device = test_params.device

    inputs = set_python_tensors_requires_grad(
        move_python_tensors_to_device(
            [arg_value for _, arg_value in test_params.arg_dict["input"]], device
        )
    )
    inputs += move_python_tensors_to_device(
        [arg_value for _, arg_value in test_params.arg_dict["target"]], device
    )
    inputs += move_python_tensors_to_device(
        [arg_value for _, arg_value in test_params.arg_dict["extra_args"]], device
    )

    # Some functionals (such as `F.rrelu`) create random tensors in their call path.
    # To make sure the random tensors created are the same in Python/C++, we need
    # to set the RNG seed manually.
    torch.manual_seed(0)
    python_output = test_params.test_instance.constructor()(*inputs)

    return python_output


def test_forward(unit_test_class, test_params):
    # [FUNCTION DOC] Runs one parity test for a functional variant. Recreates an empty temp
    # folder, computes the Python output, saves the serialized arguments, looks up the compiled
    # C++ test function by variant name on the test class module, runs it to write the C++ output
    # file, loads that output and asserts equality with the Python output using the test class
    # assertEqual (failure message from generate_error_msg). Removes the temp folder at the end
    # (only on success).
    functional_variant_name = test_params.functional_variant_name
    cpp_tmp_folder = test_params.cpp_tmp_folder
    # Remove the temporary folder if it exists already
    try_remove_folder(cpp_tmp_folder)
    os.mkdir(cpp_tmp_folder)

    # Run forward on Python functional
    python_output = run_forward(unit_test_class, test_params)

    # Save Python arguments to be used from C++ function
    arg_dict_file_path = compute_temp_file_path(
        cpp_tmp_folder, functional_variant_name, "arg_dict"
    )
    serialize_arg_dict_as_script_module(test_params.arg_dict).save(arg_dict_file_path)

    cpp_test_name = f"{test_params.functional_variant_name}_test_forward"
    cpp_test_fn = getattr(
        unit_test_class.functional_impl_check_cpp_module, cpp_test_name
    )

    def run_cpp_test_fn_and_check_output():
        # [FUNCTION DOC] HELPER: Runs the C++ test function with the argument file and
        # an output path, loads the saved result and asserts it equals the Python output
        # from the enclosing scope.
        forward_output_file_path = compute_temp_file_path(
            cpp_tmp_folder, functional_variant_name, "forward_output"
        )

        cpp_test_fn(arg_dict_file_path, forward_output_file_path)
        cpp_output = torch.load(forward_output_file_path)

        # Check that forward outputs are equal
        unit_test_class.assertEqual(
            python_output,
            cpp_output,
            msg=generate_error_msg("forward output", cpp_output, python_output),
        )

    run_cpp_test_fn_and_check_output()

    # Remove temporary folder that stores C++ outputs
    try_remove_folder(cpp_tmp_folder)


def compute_functional_name(test_params_dict):
    # [FUNCTION DOC] Returns the snake-case Python functional name for a test params dict. Uses
    # the options-args entry when present (strips the namespace prefix and the options suffix from
    # the name before the opening parenthesis, then converts camel case) else the function-call
    # entry (text before the parenthesis without the namespace prefix). Raises RuntimeError if
    # neither entry exists.
    def camel_case_to_snake_case(camel_case_str):
        # [FUNCTION DOC] HELPER: Converts a camel case string to snake case, lower-
        # cased, inserting an underscore before each capital except at the start.
        return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_case_str).lower()

    if "cpp_options_args" in test_params_dict:
        # Expected format for `cpp_options_args`: `F::FunctionalFuncOptions(...)`
        # Example output: `binary_cross_entropy`
        return camel_case_to_snake_case(
            test_params_dict["cpp_options_args"]
            .split("(")[0]
            .replace("F::", "")
            .replace("FuncOptions", "")
        )
    elif "cpp_function_call" in test_params_dict:
        # Expected format for `cpp_function_call`: `F::functional_name(...)`
        # Example output: `binary_cross_entropy`
        return test_params_dict["cpp_function_call"].split("(")[0].replace("F::", "")
    else:
        raise RuntimeError(
            "`cpp_options_args` or `cpp_function_call` entry must be present in test params dict:\n"
            f"{pprint.pformat(test_params_dict)}"
        )


def compute_cpp_function_call(test_params_dict, arg_dict, functional_name):
    # [FUNCTION DOC] Returns the C++ call expression for a functional. If the params dict has an
    # explicit function-call entry it is returned unchanged; if it has options args, builds a call
    # of the functional name with all input, target and extra argument names followed by the
    # options. Raises RuntimeError if neither entry is present.
    if "cpp_function_call" in test_params_dict:
        return test_params_dict["cpp_function_call"]
    elif "cpp_options_args" in test_params_dict:
        cpp_forward_args_symbols = [
            arg_name
            for arg_name, _ in arg_dict["input"]
            + arg_dict["target"]
            + arg_dict["extra_args"]
        ]
        return "F::{}({}, {})".format(
            functional_name,
            ", ".join(cpp_forward_args_symbols),
            test_params_dict["cpp_options_args"],
        )
    else:
        raise RuntimeError(
            "`cpp_options_args` or `cpp_function_call` entry must be present in test params dict:\n"
            f"{pprint.pformat(test_params_dict)}"
        )


def process_test_params_for_functional(test_params_dict, device, test_instance_class):
    # [FUNCTION DOC] Builds the TorchNNFunctionalTestParams for one device from a params dict:
    # creates the test instance, derives the functional name, checks the test name starts with the
    # test prefix (else AssertionError), makes the variant name by dropping that prefix and adding
    # the device suffix for non-cpu devices, computes the arg dict and C++ call, reads the
    # optional parity flag (default true), and allocates a new temporary folder (a side effect on
    # disk).
    test_instance = test_instance_class(**test_params_dict)
    functional_name = compute_functional_name(test_params_dict)
    if not test_instance.get_name().startswith("test_"):
        raise AssertionError(
            f"Expected test name to start with 'test_', got '{test_instance.get_name()}'"
        )
    # Example output: `BCELoss_no_reduce_cuda`
    functional_variant_name = test_instance.get_name()[5:] + (
        ("_" + device) if device != "cpu" else ""
    )
    arg_dict = compute_arg_dict(test_params_dict, test_instance)

    return TorchNNFunctionalTestParams(
        functional_name=functional_name,
        functional_variant_name=functional_variant_name,
        test_instance=test_instance,
        cpp_function_call=compute_cpp_function_call(
            test_params_dict, arg_dict, functional_name
        ),
        arg_dict=arg_dict,
        has_parity=test_params_dict.get("has_parity", True),
        device=device,
        cpp_tmp_folder=tempfile.mkdtemp(),
    )


def write_test_to_test_class(
    unit_test_class, test_params_dict, test_instance_class, parity_table, devices
):
    # [FUNCTION DOC] Validates one functional test params dict and registers a unit test per
    # device on unit_test_class. Raises AssertionError if it is not a functional test, if neither
    # or both of the options-args and function-call entries exist, if torch.nn.functional lacks
    # the function or the parity table lacks its entry. For each device it builds params, deletes
    # the freshly made temp folder, stores the params in the class map keyed by unit test name,
    # wraps a test function with decorate_test_fn (cuda flag, impl-parity flag from the table and
    # the params dict, device) and adds it to the class.
    if not is_torch_nn_functional_test(test_params_dict):
        raise AssertionError("Expected torch.nn.functional test")

    if not (
        "cpp_options_args" in test_params_dict
        or "cpp_function_call" in test_params_dict
    ):
        raise AssertionError(
            "To enable C++ API parity test, "
            "`cpp_options_args` or `cpp_function_call` entry must be present in test params dict:\n"
            f"{pprint.pformat(test_params_dict)}. \n"
            "If you are interested in adding the C++ API parity test, please see:\n"
            "NOTE [How to check NN module / functional API parity between Python and C++ frontends]. \n"
            "If not, please add `test_cpp_api_parity=False` to the test params dict and file an issue about this."
        )

    if (
        "cpp_options_args" in test_params_dict
        and "cpp_function_call" in test_params_dict
    ):
        raise AssertionError(
            "Only one of `cpp_options_args` and `cpp_function_call` entries "
            f"should be present in test params dict:\n{pprint.pformat(test_params_dict)}"
        )

    functional_name = compute_functional_name(test_params_dict)

    if not hasattr(torch.nn.functional, functional_name):
        raise AssertionError(
            f"`torch.nn.functional` doesn't have function `{functional_name}`. "
            f"(Discovered while processing\n{pprint.pformat(test_params_dict)}.)"
        )

    functional_full_name = "F::" + functional_name

    if functional_full_name not in parity_table["torch::nn::functional"]:
        raise AssertionError(
            f"Please add `{functional_full_name}` entry to `torch::nn::functional` "
            "section of `test/cpp_api_parity/parity-tracker.md`. "
            f"(Discovered while processing\n{pprint.pformat(test_params_dict)}.)"
        )

    for device in devices:
        test_params = process_test_params_for_functional(
            test_params_dict=test_params_dict,
            device=device,
            test_instance_class=test_instance_class,
        )
        try_remove_folder(test_params.cpp_tmp_folder)
        unit_test_name = (
            f"test_torch_nn_functional_{test_params.functional_variant_name}"
        )
        unit_test_class.functional_test_params_map[unit_test_name] = test_params

        def test_fn(self):
            # [FUNCTION DOC] HELPER: Generated unit test body; looks up the stored params
            # for the running test method name and runs the forward parity check.
            test_forward(
                unit_test_class=self,
                test_params=unit_test_class.functional_test_params_map[
                    self._testMethodName
                ],
            )

        test_fn = decorate_test_fn(
            test_fn=test_fn,
            test_cuda=test_params_dict.get("test_cuda", True),
            has_impl_parity=parity_table["torch::nn::functional"][functional_full_name][
                0
            ]
            and test_params_dict.get("has_parity", True),
            device=device,
        )

        add_test(unit_test_class, unit_test_name, test_fn)


def generate_test_cpp_sources(test_params, template):
    # [FUNCTION DOC] Returns the C++ source text for one test by substituting the variant name,
    # the argument construction statements (joined with semicolons and newlines) and the function
    # call into the template. The forward argument symbols from the helper are ignored.
    (
        cpp_args_construction_stmts,
        _,
    ) = compute_cpp_args_construction_stmts_and_forward_arg_symbols(test_params)

    test_cpp_sources = template.substitute(
        functional_variant_name=test_params.functional_variant_name,
        cpp_args_construction_stmts=";\n  ".join(cpp_args_construction_stmts),
        cpp_function_call=test_params.cpp_function_call,
    )
    return test_cpp_sources


# Build all C++ tests together, instead of once per test.
def build_cpp_tests(unit_test_class, print_cpp_source=False):
    # [FUNCTION DOC] Compiles all registered functional tests together into one inline C++ module
    # (once, not per test), using the common harness, sample functional source and a source per
    # params entry. Optionally prints the full source. Stores the compiled module on the test
    # class and returns nothing. Raises AssertionError if the params map is empty.
    if len(unit_test_class.functional_test_params_map) <= 0:
        raise AssertionError(
            "Expected functional_test_params_map to have at least one entry"
        )
    cpp_sources = TORCH_NN_COMMON_TEST_HARNESS + SAMPLE_FUNCTIONAL_CPP_SOURCE
    functions = []
    for test_params in unit_test_class.functional_test_params_map.values():
        cpp_sources += generate_test_cpp_sources(
            test_params=test_params, template=TORCH_NN_FUNCTIONAL_TEST_FORWARD
        )
        functions.append(f"{test_params.functional_variant_name}_test_forward")
    if print_cpp_source:
        print(cpp_sources)

    cpp_module = compile_cpp_code_inline(
        name="functional_impl_check", cpp_sources=cpp_sources, functions=functions
    )
    unit_test_class.functional_impl_check_cpp_module = cpp_module
