# [MODULE DOC] Driver for the autograd, Python-binding and related code generation used by the
# PyTorch build. Exposes generate_code (runs the autograd, pybinding and annotated-args
# generators for a chosen subset), selector helpers for selective build, and a command-line main
# that also generates functionalization view-meta classes and optionally the lazy TorchScript
# backend. Import-time: imports yaml and prefers its C safe loader.
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, cast

import yaml


try:
    # use faster C loader if available
    from yaml import CSafeLoader as YamlLoader
except ImportError:
    from yaml import SafeLoader as YamlLoader  # type: ignore[assignment, misc]


NATIVE_FUNCTIONS_PATH = "aten/src/ATen/native/native_functions.yaml"
TAGS_PATH = "aten/src/ATen/native/tags.yaml"


def generate_code(
    gen_dir: Path,
    native_functions_path: str | None = None,
    tags_path: str | None = None,
    install_dir: str | None = None,
    subset: str | None = None,
    disable_autograd: bool = False,
    force_schema_registration: bool = False,
    operator_selector: Any = None,
) -> None:
    # [FUNCTION DOC] Run the code generators and write outputs under gen_dir. Args: gen_dir;
    # native_functions_path and tags_path (default to the in-repo yaml files); install_dir
    # (default is the torch/csrc directory under gen_dir, with Python outputs under the
    # generated testing directory; when given it is used for both); subset, one of
    # pybindings, libtorch or python to run only that generator (any other falsy value runs
    # all three); disable_autograd; operator_selector (default is the no-op selector that
    # keeps all operators). Creates output directories and writes generated files. Returns
    # None.
    # KNOWN ISSUE: force_schema_registration is accepted but never used. An unrecognized
    # non-empty subset silently generates nothing.
    from tools.autograd.gen_annotated_fn_args import gen_annotated
    from tools.autograd.gen_autograd import gen_autograd, gen_autograd_python

    from torchgen.selective_build.selector import SelectiveBuilder

    # Build ATen based Variable classes
    if install_dir is None:
        install_dir = os.fspath(gen_dir / "torch/csrc")
        python_install_dir = os.fspath(gen_dir / "torch/testing/_internal/generated")
    else:
        python_install_dir = install_dir
    autograd_gen_dir = os.path.join(install_dir, "autograd", "generated")
    for d in (autograd_gen_dir, python_install_dir):
        os.makedirs(d, exist_ok=True)
    autograd_dir = os.fspath(Path(__file__).parent.parent / "autograd")

    if subset == "pybindings" or not subset:
        gen_autograd_python(
            native_functions_path or NATIVE_FUNCTIONS_PATH,
            tags_path or TAGS_PATH,
            autograd_gen_dir,
            autograd_dir,
        )

    if operator_selector is None:
        operator_selector = SelectiveBuilder.get_nop_selector()

    if subset == "libtorch" or not subset:
        gen_autograd(
            native_functions_path or NATIVE_FUNCTIONS_PATH,
            tags_path or TAGS_PATH,
            autograd_gen_dir,
            autograd_dir,
            disable_autograd=disable_autograd,
            operator_selector=operator_selector,
        )

    if subset == "python" or not subset:
        gen_annotated(
            native_functions_path or NATIVE_FUNCTIONS_PATH,
            tags_path or TAGS_PATH,
            python_install_dir,
            autograd_dir,
        )


def get_selector_from_legacy_operator_selection_list(
    selected_op_list_path: str,
) -> Any:
    # [FUNCTION DOC] Build an operator selector from a legacy YAML list of selected operator
    # names. Reads the file at the given path, strips the overload suffix from every name,
    # and marks all operators as root operators and as used for training. Returns the
    # selector. Raises yaml or file errors on bad input.
    with open(selected_op_list_path) as f:
        # strip out the overload part
        # It's only for legacy config - do NOT copy this code!
        selected_op_list = {
            opname.split(".", 1)[0] for opname in yaml.load(f, Loader=YamlLoader)
        }

    # Internal build doesn't use this flag any more. Only used by OSS
    # build now. Every operator should be considered a root operator
    # (hence generating unboxing code for it, which is consistent with
    # the current behavior), and also be considered as used for
    # training, since OSS doesn't support training on mobile for now.
    #
    is_root_operator = True
    is_used_for_training = True

    from torchgen.selective_build.selector import SelectiveBuilder

    selector = SelectiveBuilder.from_legacy_op_registration_allow_list(
        selected_op_list,
        is_root_operator,
        is_used_for_training,
    )

    return selector


def get_selector(
    selected_op_list_path: str | None,
    operators_yaml_path: str | None,
) -> Any:
    # [FUNCTION DOC] Choose the selective-build operator selector. Args:
    # selected_op_list_path (legacy list) and operators_yaml_path (model yaml); at most one
    # may be given, otherwise AssertionError is raised. With neither, returns the no-op
    # selector (all operators kept). Side effect: inserts the repository root at the front
    # of the module search path.
    # cwrap depends on pyyaml, so we can't import it earlier
    REPO_ROOT = Path(__file__).absolute().parents[2]
    sys.path.insert(0, str(REPO_ROOT))

    from torchgen.selective_build.selector import SelectiveBuilder

    if selected_op_list_path is not None and operators_yaml_path is not None:
        raise AssertionError(
            "Expected at most one of selected_op_list_path and "
            "operators_yaml_path to be set."
        )

    if selected_op_list_path is None and operators_yaml_path is None:
        return SelectiveBuilder.get_nop_selector()
    elif selected_op_list_path is not None:
        return get_selector_from_legacy_operator_selection_list(selected_op_list_path)
    else:
        return SelectiveBuilder.from_yaml_path(cast(str, operators_yaml_path))


def main() -> None:
    # [FUNCTION DOC] Command-line entry point. Parses options (native functions and tags
    # paths, gen dir, deprecated install dir, subset, disable-autograd, selection files,
    # lazy backend and per-operator header flags), builds the selector, calls generate_code,
    # then generates functionalization view-meta classes and, if requested, the lazy
    # TorchScript backend through the torchgen lazy generator. Creates directories and
    # writes generated sources. Raises AssertionError if an expected directory or input file
    # is missing.
    # KNOWN ISSUE: the aten path is derived from native_functions_path, which is
    # dereferenced before use, so omitting that option fails with a TypeError. The subset
    # help text omits the python value. The force-schema-registration option is parsed but
    # has no effect.
    parser = argparse.ArgumentParser(description="Autogenerate code")
    parser.add_argument("--native-functions-path")
    parser.add_argument("--tags-path")
    parser.add_argument(
        "--gen-dir",
        type=Path,
        default=Path("."),
        help="Root directory where to install files. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--install-dir",
        "--install_dir",
        help=(
            "Deprecated. Use --gen-dir instead. The semantics are different, do not change "
            "blindly."
        ),
    )
    parser.add_argument(
        "--subset",
        help='Subset of source files to generate. Can be "libtorch" or "pybindings". Generates both when omitted.',
    )
    parser.add_argument(
        "--disable-autograd",
        default=False,
        action="store_true",
        help="It can skip generating autograd related code when the flag is set",
    )
    parser.add_argument(
        "--selected-op-list-path",
        help="Path to the YAML file that contains the list of operators to include for custom build.",
    )
    parser.add_argument(
        "--operators-yaml-path",
        "--operators_yaml_path",
        help="Path to the model YAML file that contains the list of operators to include for custom build.",
    )
    parser.add_argument(
        "--force-schema-registration",
        "--force_schema_registration",
        action="store_true",
        help="force it to generate schema-only registrations for ops that are not"
        "listed on --selected-op-list",
    )
    parser.add_argument(
        "--gen-lazy-ts-backend",
        "--gen_lazy_ts_backend",
        action="store_true",
        help="Enable generation of the torch::lazy TorchScript backend",
    )
    parser.add_argument(
        "--per-operator-headers",
        "--per_operator_headers",
        action="store_true",
        help="Build lazy tensor ts backend with per-operator ATen headers, must match how ATen was built",
    )
    options = parser.parse_args()

    # Path: aten/src/ATen
    aten_path = os.path.dirname(os.path.dirname(options.native_functions_path))
    operator_selector = get_selector(
        options.selected_op_list_path, options.operators_yaml_path
    )

    generate_code(
        options.gen_dir,
        options.native_functions_path,
        options.tags_path,
        options.install_dir,
        options.subset,
        options.disable_autograd,
        options.force_schema_registration,
        # options.selected_op_list
        operator_selector=operator_selector,
    )

    # Generate the python bindings for functionalization's `ViewMeta` classes.
    from torchgen.gen_functionalization_type import (
        gen_functionalization_view_meta_classes,
    )

    functionalization_templates_dir = os.path.join(aten_path, "templates")
    install_dir = options.install_dir or os.fspath(options.gen_dir / "torch/csrc")
    functionalization_install_dir = os.path.join(
        install_dir, "functionalization", "generated"
    )

    os.makedirs(functionalization_install_dir, exist_ok=True)
    if not os.path.isdir(functionalization_install_dir):
        raise AssertionError(f"Not a directory: {functionalization_install_dir}")
    if not os.path.isdir(functionalization_templates_dir):
        raise AssertionError(f"Not a directory: {functionalization_templates_dir}")

    gen_functionalization_view_meta_classes(
        options.native_functions_path or NATIVE_FUNCTIONS_PATH,
        options.tags_path or TAGS_PATH,
        selector=operator_selector,
        install_dir=functionalization_install_dir,
        template_dir=functionalization_templates_dir,
    )

    if options.gen_lazy_ts_backend:
        ts_backend_yaml = os.path.join(aten_path, "native/ts_native_functions.yaml")
        ts_native_functions = "torch/csrc/lazy/ts_backend/ts_native_functions.cpp"
        ts_node_base = "torch/csrc/lazy/ts_backend/ts_node.h"
        lazy_install_dir = os.path.join(install_dir, "lazy", "generated")
        os.makedirs(lazy_install_dir, exist_ok=True)

        if not os.path.isfile(ts_backend_yaml):
            raise AssertionError(f"Unable to access ts_backend_yaml: {ts_backend_yaml}")
        if not os.path.isfile(ts_native_functions):
            raise AssertionError(f"Unable to access {ts_native_functions}")
        from torchgen.dest.lazy_ir import GenTSLazyIR
        from torchgen.gen_lazy_tensor import run_gen_lazy_tensor

        run_gen_lazy_tensor(
            aten_path=aten_path,
            source_yaml=ts_backend_yaml,
            backend_name="TorchScript",
            output_dir=lazy_install_dir,
            dry_run=False,
            impl_path=ts_native_functions,
            node_base="TsNode",
            node_base_hdr=ts_node_base,
            build_in_tree=True,
            lazy_ir_generator=GenTSLazyIR,
            per_operator_headers=options.per_operator_headers,
            gen_forced_fallback_code=True,
        )


if __name__ == "__main__":
    main()
