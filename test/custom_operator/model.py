# [MODULE DOC] HELPER: serializes a script module that uses the custom op library. Provides the
# library path lookup and a Model; run as a program with --export-script-module-to to save it.
import argparse
import os.path
import sys

import torch


def get_custom_op_library_path():
    # [FUNCTION DOC] HELPER: returns the absolute path of the custom ops shared library under the
    # build directory of the current directory (platform-specific file name). Raises
    # AssertionError if the file does not exist.
    if sys.platform.startswith("win32"):
        library_filename = "custom_ops.dll"
    elif sys.platform.startswith("darwin"):
        library_filename = "libcustom_ops.dylib"
    else:
        library_filename = "libcustom_ops.so"
    path = os.path.abspath(f"build/{library_filename}")
    if not os.path.exists(path):
        raise AssertionError(f"Library not found: {path}")
    return path


class Model(torch.jit.ScriptModule):
    # [CLASS DOC] HELPER: script module with a 5x5 identity parameter p; forward uses the custom
    # operator and so needs the library loaded first.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: registers parameter p as a 5x5 identity matrix.
        super().__init__()
        self.p = torch.nn.Parameter(torch.eye(5))

    @torch.jit.script_method
    def forward(self, input):
        # [FUNCTION DOC] HELPER: calls the custom op with defaults, takes the first element of its
        # result list and adds one.
        return torch.ops.custom.op_with_defaults(input)[0] + 1


def main():
    # [FUNCTION DOC] HELPER: parses the required --export-script-module-to path, loads the custom
    # op library and saves a new Model there.
    parser = argparse.ArgumentParser(
        description="Serialize a script module with custom ops"
    )
    parser.add_argument("--export-script-module-to", required=True)
    options = parser.parse_args()

    torch.ops.load_library(get_custom_op_library_path())

    model = Model()
    model.save(options.export_script_module_to)


if __name__ == "__main__":
    main()
