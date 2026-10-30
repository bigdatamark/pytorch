# [MODULE DOC] Build helper invoked by CMake that turns a Metal shader source file into a C++
# header. The header holds a static Metal shader library object built from the shader text, with
# local include directives already inlined by the embed_headers helper from the torch utils
# directory (added to the import path at import time). Run with an input and an output filename;
# prints usage and exits with status 1 if fewer arguments are given.
"""Used by CMake to transform *.metal into *_metallib.h."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "torch" / "utils"))
from _cpp_embed_headers import embed_headers


def write_metallib_headers(metal_filename: str, output_filename: str):
    # [FUNCTION DOC] Writes the output header file: an include of the MPS operation utilities,
    # then a static shader library object initialized with the embedded text of the Metal source
    # as a raw string literal. Overwrites the output file. Raises OSError on file errors.
    embedded_headers = embed_headers(metal_filename)

    with open(output_filename, "w") as out:
        out.writelines(
            [
                "#include <ATen/native/mps/OperationUtils.h>\n",
                'static ::at::native::mps::MetalShaderLibrary lib(R"SHDR(\n',
                embedded_headers,
                ')SHDR");\n',
            ]
        )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage:\n {sys.argv[0]} metal_filename output_filename")
        sys.exit(1)

    write_metallib_headers(sys.argv[1], sys.argv[2])
