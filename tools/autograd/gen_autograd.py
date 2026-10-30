# [MODULE DOC] Driver for the autograd C++ and Python code generators. It delegates to the sibling
# generators for Node subclasses, VariableType, InplaceOrView, TraceType, variable factories, view
# functions and Python bindings. Provides a function for the C++ outputs, one for the
# Python-facing outputs, and a command line main. Importing has no side effects beyond imports.
"""
To run this file by hand from the root of the PyTorch
repository, run:

python -m tools.autograd.gen_autograd \
       aten/src/ATen/native/native_functions.yaml \
       aten/src/ATen/native/tags.yaml \
       $OUTPUT_DIR \
       tools/autograd

Where $OUTPUT_DIR is where you would like the files to be
generated.  In the full build system, OUTPUT_DIR is
torch/csrc/autograd/generated/
"""

# gen_autograd.py generates C++ autograd functions and Python bindings.
#
# It delegates to the following scripts:
#
#  gen_autograd_functions.py: generates subclasses of torch::autograd::Node
#  gen_variable_type.py: generates VariableType.h which contains all tensor methods
#  gen_python_functions.py: generates Python bindings to THPVariable
#

from __future__ import annotations

import argparse
import os

from torchgen.api import cpp
from torchgen.api.autograd import (
    match_differentiability_info,
    NativeFunctionWithDifferentiabilityInfo,
)
from torchgen.gen import parse_native_yaml
from torchgen.selective_build.selector import SelectiveBuilder

from . import gen_python_functions
from .gen_autograd_functions import (
    gen_autograd_functions_lib,
    gen_autograd_functions_python,
)
from .gen_inplace_or_view_type import gen_inplace_or_view_type
from .gen_trace_type import gen_trace_type
from .gen_variable_factories import gen_variable_factories
from .gen_variable_type import gen_variable_type
from .gen_view_funcs import gen_view_funcs
from .load_derivatives import load_derivatives


def gen_autograd(
    native_functions_path: str,
    tags_path: str,
    out: str,
    autograd_dir: str,
    operator_selector: SelectiveBuilder,
    disable_autograd: bool = False,
) -> None:
    # [FUNCTION DOC] Generates the C++ autograd sources into the output directory. Loads
    # derivatives.yaml from the autograd directory, parses the native functions, keeps those the
    # operator selector selects for training (sorted by C++ name) and pairs them with their
    # differentiability info. Unless disable_autograd is set it writes VariableType, InplaceOrView
    # and TraceType files (tracing uses all native functions, unfiltered). It always writes the
    # autograd Functions library, the variable factories and the view functions. Returns nothing;
    # writes files as a side effect.
    # Parse and load derivatives.yaml
    differentiability_infos, used_dispatch_keys = load_derivatives(
        os.path.join(autograd_dir, "derivatives.yaml"), native_functions_path, tags_path
    )

    template_path = os.path.join(autograd_dir, "templates")

    native_funcs = parse_native_yaml(native_functions_path, tags_path).native_functions
    fns = sorted(
        filter(
            operator_selector.is_native_function_selected_for_training, native_funcs
        ),
        key=lambda f: cpp.name(f.func),
    )
    fns_with_diff_infos: list[NativeFunctionWithDifferentiabilityInfo] = (
        match_differentiability_info(fns, differentiability_infos)
    )

    # Generate VariableType.h/cpp
    if not disable_autograd:
        gen_variable_type(
            out,
            native_functions_path,
            tags_path,
            fns_with_diff_infos,
            template_path,
            used_dispatch_keys,
        )

        gen_inplace_or_view_type(
            out, native_functions_path, tags_path, fns_with_diff_infos, template_path
        )

        # operator filter not applied as tracing sources are excluded in selective build
        gen_trace_type(out, native_funcs, template_path)
    # Generate Functions.h/cpp
    gen_autograd_functions_lib(out, differentiability_infos, template_path)

    # Generate variable_factories.h
    gen_variable_factories(out, native_functions_path, tags_path, template_path)

    # Generate ViewFuncs.h/cpp
    gen_view_funcs(out, fns_with_diff_infos, template_path)


def gen_autograd_python(
    native_functions_path: str,
    tags_path: str,
    out: str,
    autograd_dir: str,
) -> None:
    # [FUNCTION DOC] Generates the Python-facing autograd outputs: loads derivatives.yaml, writes
    # the Python part of the autograd Functions files and the Python bindings (using
    # deprecated.yaml and the templates folder), all into the output directory. Returns nothing.
    differentiability_infos, _ = load_derivatives(
        os.path.join(autograd_dir, "derivatives.yaml"), native_functions_path, tags_path
    )

    template_path = os.path.join(autograd_dir, "templates")

    # Generate Functions.h/cpp
    gen_autograd_functions_python(out, differentiability_infos, template_path)

    # Generate Python bindings
    deprecated_path = os.path.join(autograd_dir, "deprecated.yaml")
    gen_python_functions.gen(
        out, native_functions_path, tags_path, deprecated_path, template_path
    )


def main() -> None:
    # [FUNCTION DOC] Command line entry point: parses four positional paths (native functions
    # yaml, tags yaml, output directory, autograd directory) and runs the C++ generation with a
    # selector that selects everything. The tags argument reuses the metavar NATIVE, which only
    # affects help text.
    parser = argparse.ArgumentParser(description="Generate autograd C++ files script")
    parser.add_argument(
        "native_functions", metavar="NATIVE", help="path to native_functions.yaml"
    )
    parser.add_argument("tags", metavar="NATIVE", help="path to tags.yaml")
    parser.add_argument("out", metavar="OUT", help="path to output directory")
    parser.add_argument(
        "autograd", metavar="AUTOGRAD", help="path to autograd directory"
    )
    args = parser.parse_args()
    gen_autograd(
        args.native_functions,
        args.tags,
        args.out,
        args.autograd,
        SelectiveBuilder.get_nop_selector(),
    )


if __name__ == "__main__":
    main()
