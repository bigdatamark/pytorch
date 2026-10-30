# [MODULE DOC] Script that generates two csv files, listing core ATen operators and
# prim operators with their schemas, in the build/ir directory (relative to the
# current directory) for inclusion in the IR documentation. Parses
# native_functions.yaml and tags.yaml from the repo root. Importing has no side
# effects beyond imports; running as main writes the files.
import os
from collections import OrderedDict
from pathlib import Path

import torch
import torch._prims as prims
from torchgen.gen import parse_native_yaml


ROOT = Path(__file__).absolute().parents[3]
NATIVE_FUNCTION_YAML_PATH = ROOT / "aten/src/ATen/native/native_functions.yaml"
TAGS_YAML_PATH = ROOT / "aten/src/ATen/native/tags.yaml"

BUILD_DIR = "build/ir"
ATEN_OPS_CSV_FILE = "aten_ops.csv"
PRIMS_OPS_CSV_FILE = "prims_ops.csv"


def get_aten():
    # [FUNCTION DOC] Parses the native functions and tags yaml files and returns a list of
    # pairs of operator name (prefixed with the aten namespace) and schema string, for
    # functions tagged core, sorted by name. Asterisks in schemas are escaped for rst.
    # Reads files from disk.
    parsed_yaml = parse_native_yaml(NATIVE_FUNCTION_YAML_PATH, TAGS_YAML_PATH)
    native_functions = parsed_yaml.native_functions

    aten_ops = OrderedDict()
    for function in native_functions:
        if "core" in function.tags:
            op_name = str(function.func.name)
            aten_ops[op_name] = function

    op_schema_pairs = []
    for key, op in sorted(aten_ops.items()):
        op_name = f"aten.{key}"
        schema = str(op.func).replace("*", r"\*")

        op_schema_pairs.append((op_name, schema))

    return op_schema_pairs


def get_prims():
    # [FUNCTION DOC] Walks the exported names of the prims module and returns a list of
    # pairs of operator name (default overload suffix stripped) and schema string, for
    # those that are operator overloads. Asterisks in schemas are escaped for rst. Non-
    # overload names are skipped.
    op_schema_pairs = []
    for op_name in prims.__all__:
        op_overload = getattr(prims, op_name, None)

        if not isinstance(op_overload, torch._ops.OpOverload):
            continue

        op_overloadpacket = op_overload.overloadpacket

        op_name = str(op_overload).replace(".default", "")
        schema = op_overloadpacket.schema.replace("*", r"\*")

        op_schema_pairs.append((op_name, schema))

    return op_schema_pairs


def main():
    # [FUNCTION DOC] Collects the ATen and prims operator lists, creates the build/ir
    # directory if needed and writes one csv file for each (header line Operator,Schema
    # followed by one quoted row per operator, with the operator name wrapped in rst
    # literal markup). Overwrites existing files.
    aten_ops_list = get_aten()
    prims_ops_list = get_prims()

    os.makedirs(BUILD_DIR, exist_ok=True)

    with open(os.path.join(BUILD_DIR, ATEN_OPS_CSV_FILE), "w") as f:
        f.write("Operator,Schema\n")
        f.writelines(f'"``{name}``","{schema}"\n' for name, schema in aten_ops_list)

    with open(os.path.join(BUILD_DIR, PRIMS_OPS_CSV_FILE), "w") as f:
        f.write("Operator,Schema\n")
        f.writelines(f'"``{name}``","{schema}"\n' for name, schema in prims_ops_list)


if __name__ == "__main__":
    main()
