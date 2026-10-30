# [MODULE DOC] Script that serializes TorchScript modules with forward hooks and pre-hooks to disk
# for the C++ JIT hooks tests. It imports module factory functions from the sibling jit hooks test
# module (adding the test directory to the import path at import time) and, when run as a program,
# saves each scripted model under a path prefix given on the command line.
import argparse
import os
import sys

import torch


# grab modules from test_jit_hooks.cpp
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from jit.test_hooks_modules import (
    create_forward_tuple_input,
    create_module_forward_multiple_inputs,
    create_module_forward_single_input,
    create_module_hook_return_nothing,
    create_module_multiple_hooks_multiple_inputs,
    create_module_multiple_hooks_single_input,
    create_module_no_forward_input,
    create_module_same_hook_repeated,
    create_submodule_forward_multiple_inputs,
    create_submodule_forward_single_input,
    create_submodule_hook_return_nothing,
    create_submodule_multiple_hooks_multiple_inputs,
    create_submodule_multiple_hooks_single_input,
    create_submodule_same_hook_repeated,
    create_submodule_to_call_directly_with_hooks,
)


# Create saved modules for JIT forward hooks and pre-hooks
def main():
    # [FUNCTION DOC] Entry point. Parses the required option naming the output path prefix, stores
    # the prefix with a trailing underscore in a module-level global, builds a list of fifteen
    # named hook-test modules (submodule and top-level variants with single or multiple inputs and
    # hooks, hooks returning nothing, repeated hooks, no forward input, tuple input, and calling a
    # submodule directly), scripts each, and saves it as a file named prefix, test name, then the
    # pt suffix. Prints a completion message. Exits via argparse if the option is missing.
    parser = argparse.ArgumentParser(
        description="Serialize a script modules with hooks attached"
    )
    parser.add_argument("--export-script-module-to", required=True)
    options = parser.parse_args()
    global save_name
    save_name = options.export_script_module_to + "_"

    tests = [
        (
            "test_submodule_forward_single_input",
            create_submodule_forward_single_input(),
        ),
        (
            "test_submodule_forward_multiple_inputs",
            create_submodule_forward_multiple_inputs(),
        ),
        (
            "test_submodule_multiple_hooks_single_input",
            create_submodule_multiple_hooks_single_input(),
        ),
        (
            "test_submodule_multiple_hooks_multiple_inputs",
            create_submodule_multiple_hooks_multiple_inputs(),
        ),
        ("test_submodule_hook_return_nothing", create_submodule_hook_return_nothing()),
        ("test_submodule_same_hook_repeated", create_submodule_same_hook_repeated()),
        ("test_module_forward_single_input", create_module_forward_single_input()),
        (
            "test_module_forward_multiple_inputs",
            create_module_forward_multiple_inputs(),
        ),
        (
            "test_module_multiple_hooks_single_input",
            create_module_multiple_hooks_single_input(),
        ),
        (
            "test_module_multiple_hooks_multiple_inputs",
            create_module_multiple_hooks_multiple_inputs(),
        ),
        ("test_module_hook_return_nothing", create_module_hook_return_nothing()),
        ("test_module_same_hook_repeated", create_module_same_hook_repeated()),
        ("test_module_no_forward_input", create_module_no_forward_input()),
        ("test_forward_tuple_input", create_forward_tuple_input()),
        (
            "test_submodule_to_call_directly_with_hooks",
            create_submodule_to_call_directly_with_hooks(),
        ),
    ]

    for name, model in tests:
        m_scripted = torch.jit.script(model)
        filename = save_name + name + ".pt"
        torch.jit.save(m_scripted, filename)

    print("OK: completed saving modules with hooks!")


if __name__ == "__main__":
    main()
