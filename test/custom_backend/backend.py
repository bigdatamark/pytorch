# [MODULE DOC] HELPER: builds and exports a module lowered to the custom_backend test backend.
# Provides the library path lookup, a lowering helper and a tiny model; run as a program with
# --export-module-to to load the shared library, lower the model and save it. No import-time
# side effects.
import argparse
import os.path
import sys

import torch


def get_custom_backend_library_path():
    # [FUNCTION DOC] HELPER: returns the absolute path of the custom backend shared library under
    # the build directory of the current directory; file name depends on platform (dll on
    # Windows, dylib on macOS, so otherwise). Raises AssertionError if the file is missing.
    """
    Get the path to the library containing the custom backend.

    Return:
        The path to the custom backend object, customized by platform.
    """
    if sys.platform.startswith("win32"):
        library_filename = "custom_backend.dll"
    elif sys.platform.startswith("darwin"):
        library_filename = "libcustom_backend.dylib"
    else:
        library_filename = "libcustom_backend.so"
    path = os.path.abspath(f"build/{library_filename}")
    if not os.path.exists(path):
        raise AssertionError(f"Library not found: {path}")
    return path


def to_custom_backend(module):
    # [FUNCTION DOC] HELPER: lowers a scripted module to the backend named custom_backend via the
    # private C++ binding, compiling only forward with an empty compile spec; returns the lowered
    # module.
    # KNOWN ISSUE: the docstring names the binding for the test backend, but the code calls the
    # general backend lowering binding with the backend name custom_backend.
    """
    This is a helper that wraps torch._C._jit_to_test_backend and compiles
    only the forward method with an empty compile spec.

    Args:
        module: input ScriptModule.

    Returns:
        The module, lowered so that it can run on TestBackend.
    """
    lowered_module = torch._C._jit_to_backend(
        "custom_backend", module, {"forward": {"": ""}}
    )
    return lowered_module


class Model(torch.nn.Module):
    # [CLASS DOC] HELPER: stateless module computing the sum and difference of two inputs.
    """
    Simple model used for testing that to_backend API supports saving, loading,
    and executing in C++.
    """

    def forward(self, a, b):
        # [FUNCTION DOC] HELPER: returns a tuple of (a + b, a - b).
        return (a + b, a - b)


def main():
    # [FUNCTION DOC] HELPER: parses the required --export-module-to option, loads the backend
    # library into torch.ops (raises AssertionError if it is not listed as loaded), lowers a
    # scripted Model and saves it to the given path.
    parser = argparse.ArgumentParser(description="Lower a Module to a custom backend")
    parser.add_argument("--export-module-to", required=True)
    options = parser.parse_args()

    # Load the library containing the custom backend.
    library_path = get_custom_backend_library_path()
    torch.ops.load_library(library_path)
    if library_path not in torch.ops.loaded_libraries:
        raise AssertionError(f"Library not loaded: {library_path}")

    # Lower an instance of Model to the custom backend  and export it
    # to the specified location.
    lowered_module = to_custom_backend(torch.jit.script(Model()))
    torch.jit.save(lowered_module, options.export_module_to)


if __name__ == "__main__":
    main()
