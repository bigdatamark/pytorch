#!/usr/bin/env python3

# [MODULE DOC]
# Build script for Vulkan compute shaders. Expands GLSL template files (using yaml parameter
# files with variants and an embedded preprocessor), optionally compiles them to SPIR-V with
# glslc, reads shader metadata from header comments (tile size, storage types, registration
# keys, descriptor layouts), and emits a C++ source that embeds the binaries and registers
# them with the shader registry. Import-time side effects: appends the repo root to the
# module search path; defines default environment and type mapping tables.
from __future__ import annotations

import argparse
import array
import codecs
import copy
import glob
import io
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader  # type: ignore[assignment, misc]


REPO_ROOT = Path(__file__).absolute().parent.parent
sys.path.append(str(REPO_ROOT))

CPP_H_NAME = "spv.h"
CPP_SRC_NAME = "spv.cpp"

DEFAULT_ENV: dict[str, Any] = {
    "PRECISION": "highp",
    "FLOAT_IMAGE_FORMAT": "rgba16f",
    "INT_IMAGE_FORMAT": "rgba32i",
    "UINT_IMAGE_FORMAT": "rgba32ui",
}

TYPES_ENV: dict[str, Any] = {
    "IMAGE_FORMAT": {
        "float": "rgba32f",
        "half": "rgba16f",
        "int": "rgba32i",
        "uint": "rgba32ui",
        "int8": "rgba8i",
        "uint8": "rgba8ui",
    },
    "IMAGE_T": {
        3: {
            "float": "image3D",
            "half": "image3D",
            "int": "iimage3D",
            "uint": "uimage3D",
        },
        2: {
            "float": "image2D",
            "half": "image2D",
            "int": "iimage2D",
            "uint": "uimage2D",
        },
    },
    "SAMPLER_T": {
        3: {
            "float": "sampler3D",
            "half": "sampler3D",
            "int": "isampler3D",
            "uint": "usampler3D",
        },
        2: {
            "float": "sampler2D",
            "half": "sampler2D",
            "int": "isampler2D",
            "uint": "usampler2D",
        },
    },
    "VEC4_T": {
        "float": "vec4",
        "half": "vec4",
        "int": "ivec4",
        "uint": "uvec4",
        "int8": "vec4",
        "uint8": "uvec4",
    },
    "T": {
        "float": "float",
        "half": "float",
        "int": "int",
        "uint": "uint",
        "int8": "int",
        "uint8": "uint8",
    },
}

FUNCS_ENV: dict[str, Any] = {
    "GET_POS": {
        3: lambda pos: pos,
        2: lambda pos: f"{pos}.xy",
    }
}


def extract_filename(path: str, keep_ext: bool = True) -> Any:
    # [FUNCTION DOC] HELPER: Returns the final path component, with the extension kept or, if
    # keep_ext is false, with everything after the first dot removed.
    if keep_ext:
        return os.path.basename(path)
    else:
        return os.path.basename(path).split(".")[0]


############################
#  SPIR-V Code Generation  #
############################


# https://gist.github.com/pypt/94d747fe5180851196eb
# pyrefly: ignore [invalid-inheritance]
class UniqueKeyLoader(Loader):
    # [CLASS DOC] HELPER: Yaml loader that rejects duplicate keys in a mapping, which the
    # stock loader would silently overwrite. Stateless apart from the base loader.
    def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: Builds a mapping from a yaml mapping node. Raises a
        # constructor error if the node is not a mapping, a key is unhashable, or a key
        # appears twice. Returns the plain mapping of constructed keys to values.
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None,
                None,
                f"expected a mapping node, but found {node.id}",
                node.start_mark,
            )
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)  # type: ignore[no-untyped-call]
            try:
                hash(key)
            except TypeError as e:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found unacceptable key ",
                    key_node.start_mark,
                ) from e
            # check for duplicate keys
            if key in mapping:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found duplicate key",
                    key_node.start_mark,
                )
            value = self.construct_object(value_node, deep=deep)  # type: ignore[no-untyped-call]
            mapping[key] = value
        return mapping


# https://github.com/google/XNNPACK/blob/master/tools/xngen.py
def extract_leading_whitespace(line: str) -> str:
    # [FUNCTION DOC] HELPER: Returns the run of whitespace at the start of the line (possibly
    # empty).
    match = re.match(r"\s*", line)
    return match.group(0) if match else ""


# https://github.com/google/XNNPACK/blob/master/tools/xngen.py
def escape(line: str) -> str:
    # [FUNCTION DOC] HELPER: Converts a template text line into a concatenation expression
    # for generated Python code: literal parts become quoted strings (double quotes escaped)
    # and each dollar-brace placeholder becomes a string conversion of its inner expression.
    # Raises a value error if a placeholder is not closed. Returns the joined expression text.
    output_parts = []
    while "${" in line:
        start_pos = line.index("${")
        end_pos = line.index("}", start_pos + 2)
        if start_pos != 0:
            output_parts.append('"' + line[:start_pos].replace('"', '\\"') + '"')
        output_parts.append("str(" + line[start_pos + 2 : end_pos] + ")")
        line = line[end_pos + 1 :]
    if line:
        output_parts.append('"' + line.replace('"', '\\"') + '"')
    return " + ".join(output_parts)


# https://github.com/google/XNNPACK/blob/master/tools/xngen.py
def preprocess(
    input_text: str, variables: dict[str, Any], input_path: str = "codegen"
) -> str:
    # [FUNCTION DOC] Expands a template into text. Lines whose first non-blank character is a
    # dollar sign (but not a placeholder) are Python statements with the sign removed; their
    # indentation structure is tracked so nested blocks work; other lines are printed with
    # placeholders substituted. Blank lines are preserved and lines containing the LINT marker
    # are dropped. The generated program is compiled and executed with variables as globals,
    # collecting output in a string buffer which is returned. Raises an assertion error on
    # inconsistent indentation; executing the template runs arbitrary code.
    # KNOWN ISSUE: if the template has no non-blank lines but has trailing blank lines, the
    # final loop uses an indent variable that was never assigned.
    input_lines = input_text.splitlines()
    python_lines = []

    blank_lines = 0

    last_indent = ""

    # List of tuples (total_index, python_indent)
    indent_stack = [("", "")]

    # Indicates whether this is the first line inside Python
    # code block (i.e. for, while, if, elif, else)
    python_block_start = True
    for input_line in input_lines:
        if input_line == "":
            blank_lines += 1
            continue
        # Skip lint markers.
        if "LINT" in input_line:
            continue

        input_indent = extract_leading_whitespace(input_line)
        if python_block_start:
            if not input_indent.startswith(last_indent):
                raise AssertionError("input_indent must start with last_indent")
            extra_python_indent = input_indent[len(last_indent) :]
            python_indent = indent_stack[-1][1] + extra_python_indent
            indent_stack.append((input_indent, python_indent))
            if not input_indent.startswith(indent_stack[-1][0]):
                raise AssertionError("input_indent must start with indent_stack top")
        else:
            while not input_indent.startswith(indent_stack[-1][0]):
                del indent_stack[-1]
        python_block_start = False

        python_indent = indent_stack[-1][1]
        stripped_input_line = input_line.strip()
        if stripped_input_line.startswith("$") and not stripped_input_line.startswith(
            "${"
        ):
            if stripped_input_line.endswith(":"):
                python_block_start = True
            while blank_lines != 0:
                python_lines.append(python_indent + "print(file=OUT_STREAM)")
                blank_lines -= 1
            python_lines.append(python_indent + stripped_input_line.replace("$", ""))
        else:
            if not input_line.startswith(python_indent):
                raise AssertionError("input_line must start with python_indent")
            while blank_lines != 0:
                python_lines.append(python_indent + "print(file=OUT_STREAM)")
                blank_lines -= 1
            python_lines.append(
                python_indent
                + f"print({escape(input_line[len(python_indent) :])}, file=OUT_STREAM)"
            )
        last_indent = input_indent

    while blank_lines != 0:
        # pyrefly: ignore [unbound-name]
        python_lines.append(python_indent + "print(file=OUT_STREAM)")
        blank_lines -= 1

    exec_globals = dict(variables)
    output_stream = io.StringIO()
    exec_globals["OUT_STREAM"] = output_stream

    python_bytecode = compile("\n".join(python_lines), input_path, "exec")
    exec(python_bytecode, exec_globals)

    return output_stream.getvalue()


class SPVGenerator:
    # [CLASS DOC] Discovers GLSL sources and yaml template parameter files under source
    # directories, expands them into a map of output shader name to (source file, parameter
    # mapping), and generates GLSL and SPIR-V outputs. State: src_dir_paths, env, glslc_path,
    # glsl_src_files (name to path), template_yaml_files, shader_template_params (template name
    # to list of variant parameter mappings) and output_shader_map. The constructor does all
    # discovery and parsing; generateSPV does the file output.
    def __init__(
        self,
        src_dir_paths: str | list[str],
        env: dict[Any, Any],
        glslc_path: str | None,
    ) -> None:
        # [FUNCTION DOC] Stores the source directories (a single string becomes a list), the
        # base environment mapping and the optional glslc compiler path, then scans the
        # directories, parses every template yaml file and builds the output shader map.
        # Raises on duplicate templates, invalid variant keys or missing GLSL sources.
        if isinstance(src_dir_paths, str):
            self.src_dir_paths = [src_dir_paths]
        else:
            self.src_dir_paths = src_dir_paths

        self.env = env
        self.glslc_path = glslc_path

        self.glsl_src_files: dict[str, str] = {}
        self.template_yaml_files: list[str] = []

        self.addSrcAndYamlFiles(self.src_dir_paths)
        self.shader_template_params: dict[Any, Any] = {}
        for yaml_file in self.template_yaml_files:
            self.parseTemplateYaml(yaml_file)

        self.output_shader_map: dict[str, tuple[str, dict[str, str]]] = {}
        self.constructOutputMap()

    def addSrcAndYamlFiles(self, src_dir_paths: list[str]) -> None:
        # [FUNCTION DOC] Recursively finds files whose names contain the glsl marker (recorded
        # by base name without extension, later directories overwrite earlier same-named
        # ones) and yaml files (appended to the template yaml list) under each directory.
        for src_path in src_dir_paths:
            # Collect glsl source files
            glsl_files = glob.glob(
                os.path.join(src_path, "**", "*.glsl*"), recursive=True
            )
            for file in glsl_files:
                if len(file) > 1:
                    self.glsl_src_files[extract_filename(file, keep_ext=False)] = file
            # Collect template yaml files
            yaml_files = glob.glob(
                os.path.join(src_path, "**", "*.yaml"), recursive=True
            )
            for file in yaml_files:
                if len(file) > 1:
                    self.template_yaml_files.append(file)

    def generateVariantCombinations(
        self,
        iterated_params: dict[str, Any],
        exclude_params: set[str] | None = None,
    ) -> list[Any]:
        # [FUNCTION DOC] Builds the cartesian product of the iterated parameters. For every
        # parameter not in exclude_params, each of its values becomes a (parameter name, name
        # suffix, value) triple, with the suffix defaulting to the value. Returns a list of
        # tuples with one triple per parameter.
        if exclude_params is None:
            exclude_params = set()
        all_iterated_params = []
        for param_name, value_list in iterated_params.items():
            if param_name not in exclude_params:
                param_values = []
                for value in value_list:
                    suffix = value.get("SUFFIX", value["VALUE"])
                    param_values.append((param_name, suffix, value["VALUE"]))
                all_iterated_params.append(param_values)

        return list(product(*all_iterated_params))

    def parseTemplateYaml(self, yaml_file: str) -> None:
        # [FUNCTION DOC] Loads one yaml file (duplicate keys rejected) and, for each template
        # in it, appends to shader_template_params a list of full parameter mappings, one per
        # shader variant, starting from the template defaults. A variant that has (or inherits)
        # a generate_variant_forall section expands to one mapping per combination, appending
        # the chosen value suffixes to the variant name. Raises a key error if a template name
        # is defined twice and an assertion error if a variant uses an undeclared parameter.
        with open(yaml_file) as f:
            contents = yaml.load(f, Loader=UniqueKeyLoader)
            for template_name, params_dict in contents.items():
                if template_name in self.shader_template_params:
                    raise KeyError(f"{template_name} params file is defined twice")

                default_params = params_dict["parameter_names_with_default_values"]
                params_names = set(default_params.keys()).union({"NAME"})

                self.shader_template_params[template_name] = []

                default_iterated_params = params_dict.get(
                    "generate_variant_forall", None
                )

                for variant in params_dict["shader_variants"]:
                    variant_params_names = set(variant.keys())
                    invalid_keys = (
                        variant_params_names
                        - params_names
                        - {"generate_variant_forall"}
                    )
                    if len(invalid_keys) != 0:
                        raise AssertionError(f"Invalid keys found: {invalid_keys}")

                    iterated_params = variant.get(
                        "generate_variant_forall", default_iterated_params
                    )

                    if iterated_params is not None:
                        variant_combinations = self.generateVariantCombinations(
                            iterated_params, variant_params_names
                        )

                        for combination in variant_combinations:
                            default_params_copy = copy.deepcopy(default_params)
                            for key in variant:
                                if key != "generate_variant_forall":
                                    default_params_copy[key] = variant[key]

                            variant_name = variant["NAME"]
                            for param_value in combination:
                                default_params_copy[param_value[0]] = param_value[2]
                                if len(param_value[1]) > 0:
                                    variant_name = f"{variant_name}_{param_value[1]}"

                            default_params_copy["NAME"] = variant_name

                            self.shader_template_params[template_name].append(
                                default_params_copy
                            )
                    else:
                        default_params_copy = copy.deepcopy(default_params)
                        for key in variant:
                            default_params_copy[key] = variant[key]

                        self.shader_template_params[template_name].append(
                            default_params_copy
                        )

    def create_shader_params(
        self, variant_params: dict[str, Any] | None = None
    ) -> dict[str, str]:
        # [FUNCTION DOC] Returns a copy of the base environment overlaid with the variant
        # parameters, plus a FORMAT entry derived from the DTYPE parameter (default float):
        # integer and float dtypes map to matching image formats, unknown dtypes use the
        # default float image format.
        if variant_params is None:
            variant_params = {}
        shader_params = copy.deepcopy(self.env)
        for key, value in variant_params.items():
            shader_params[key] = value

        shader_dtype = shader_params.get("DTYPE", "float")

        if shader_dtype == "int":
            shader_params["FORMAT"] = self.env["INT_IMAGE_FORMAT"]
        elif shader_dtype == "uint":
            shader_params["FORMAT"] = self.env["UINT_IMAGE_FORMAT"]
        elif shader_dtype == "int32":
            shader_params["FORMAT"] = "rgba32i"
        elif shader_dtype == "uint32":
            shader_params["FORMAT"] = "rgba32ui"
        elif shader_dtype == "int8":
            shader_params["FORMAT"] = "rgba8i"
        elif shader_dtype == "uint8":
            shader_params["FORMAT"] = "rgba8ui"
        elif shader_dtype == "float32":
            shader_params["FORMAT"] = "rgba32f"
        # Assume float by default
        else:
            shader_params["FORMAT"] = self.env["FLOAT_IMAGE_FORMAT"]

        return shader_params

    def constructOutputMap(self) -> None:
        # [FUNCTION DOC] Fills output_shader_map: every template variant maps its name to the
        # GLSL file named like the template plus its parameters; every GLSL file without a
        # template maps its own name to itself with default parameters. Raises a key error if
        # a template has no GLSL source.
        for shader_name, params in self.shader_template_params.items():
            for variant in params:
                source_glsl = self.glsl_src_files[shader_name]

                self.output_shader_map[variant["NAME"]] = (
                    source_glsl,
                    self.create_shader_params(variant),
                )

        for shader_name, source_glsl in self.glsl_src_files.items():
            if shader_name not in self.shader_template_params:
                self.output_shader_map[shader_name] = (
                    source_glsl,
                    self.create_shader_params(),
                )

    def generateSPV(self, output_dir: str) -> dict[str, str]:
        # [FUNCTION DOC] For every output shader, expands its GLSL template with the shader
        # parameters and writes the result as a glsl file in output_dir. If a glslc path was
        # given, also compiles it to SPIR-V (compute stage, Vulkan 1.0 target, warnings as
        # errors, include dirs from the sources) and prints the command. Returns a mapping
        # from each produced SPIR-V path to its generated GLSL path (empty without glslc).
        # Raises if the compiler fails.
        output_file_map = {}
        for shader_name in self.output_shader_map:
            source_glsl = self.output_shader_map[shader_name][0]
            shader_params = self.output_shader_map[shader_name][1]

            with codecs.open(source_glsl, "r", encoding="utf-8") as input_file:
                input_text = input_file.read()
                output_text = preprocess(input_text, shader_params)

            glsl_out_path = os.path.join(output_dir, f"{shader_name}.glsl")
            with codecs.open(glsl_out_path, "w", encoding="utf-8") as output_file:
                output_file.write(output_text)

            # If no GLSL compiler is specified, then only write out the generated GLSL shaders.
            # This is mainly for testing purposes.
            if self.glslc_path is not None:
                spv_out_path = os.path.join(output_dir, f"{shader_name}.spv")

                cmd = [
                    self.glslc_path,
                    "-fshader-stage=compute",
                    glsl_out_path,
                    "-o",
                    spv_out_path,
                    "--target-env=vulkan1.0",
                    "-Werror",
                ] + [
                    arg
                    for src_dir_path in self.src_dir_paths
                    for arg in ["-I", src_dir_path]
                ]

                print("glslc cmd:", cmd)
                subprocess.check_call(cmd)

                output_file_map[spv_out_path] = glsl_out_path

        return output_file_map


##############################################
#  Shader Info and Shader Registry Handling  #
##############################################


@dataclass
class ShaderInfo:
    # [CLASS DOC] Metadata scraped from a shader source: tile_size, descriptor layouts
    # (Vulkan descriptor type names, one per binding), optional weight and bias storage type
    # names, and an optional registration pair of op name and list of registry keys.
    tile_size: list[int]
    layouts: list[str]
    weight_storage_type: str = ""
    bias_storage_type: str = ""
    register_for: tuple[str, list[str]] | None = None


def getName(filePath: str) -> str:
    # [FUNCTION DOC] HELPER: Returns the file's base name with dots replaced by underscores
    # so it can be used as a C++ identifier part.
    return os.path.basename(filePath).replace("/", "_").replace(".", "_")


def isDescriptorLine(lineStr: str) -> bool:
    # [FUNCTION DOC] HELPER: True if the line starts with a layout declaration with a set
    # qualifier (a descriptor binding line).
    descriptorLineId = r"^layout\(set"
    return re.search(descriptorLineId, lineStr) is not None


def isTileSizeLine(lineStr: str) -> bool:
    # [FUNCTION DOC] HELPER: True if the line is a header comment line declaring TILE_SIZE.
    tile_size_id = r"^ \* TILE_SIZE = \("
    return re.search(tile_size_id, lineStr) is not None


def findTileSizes(lineStr: str) -> list[int]:
    # [FUNCTION DOC] HELPER: Parses a TILE_SIZE comment line holding three integers and
    # returns them as a list. Raises an assertion error if the line does not match.
    tile_size_id = r"^ \* TILE_SIZE = \(([0-9]+), ([0-9]+), ([0-9]+)\)"
    matches = re.search(tile_size_id, lineStr)
    if matches is None:
        raise AssertionError("matches is None in findTileSizes")
    return [int(matches.group(1)), int(matches.group(2)), int(matches.group(3))]


def isWeightStorageTypeLine(lineStr: str) -> bool:
    # [FUNCTION DOC] HELPER: True if the line is a header comment line declaring
    # WEIGHT_STORAGE.
    weight_storage_id = r"^ \* WEIGHT_STORAGE = "
    return re.search(weight_storage_id, lineStr) is not None


def getWeightStorageType(lineStr: str) -> str:
    # [FUNCTION DOC] HELPER: Extracts the storage type token (letters, underscore, digit and
    # D) from a WEIGHT_STORAGE comment line. Raises an assertion error if it does not match.
    weight_storage_id = r"^ \* WEIGHT_STORAGE = ([a-zA-Z]+_\dD)"
    matches = re.search(weight_storage_id, lineStr)
    if matches is None:
        raise AssertionError("matches is None in getWeightStorageType")
    return matches.group(1)


def isBiasStorageTypeLine(lineStr: str) -> bool:
    # [FUNCTION DOC] HELPER: True if the line is a header comment line declaring BIAS_STORAGE.
    weight_storage_id = r"^ \* BIAS_STORAGE = "
    return re.search(weight_storage_id, lineStr) is not None


def getBiasStorageType(lineStr: str) -> str:
    # [FUNCTION DOC] HELPER: Extracts the storage type token from a BIAS_STORAGE comment
    # line. Raises an assertion error if it does not match.
    weight_storage_id = r"^ \* BIAS_STORAGE = ([a-zA-Z]+_\dD)"
    matches = re.search(weight_storage_id, lineStr)
    if matches is None:
        raise AssertionError("matches is None in getBiasStorageType")
    return matches.group(1)


def isRegisterForLine(lineStr: str) -> bool:
    # [FUNCTION DOC] HELPER: True if the line is a REGISTER_FOR comment line holding a quoted
    # shader name and a list with at least one quoted registry key.
    # Check for Shader Name and a list of at least one Registry Key
    register_for_id = (
        r"^ \* REGISTER_FOR = \('([A-Za-z0-9_]+)'\s*,\s*\['([A-Za-z0-9_]+)'.*\]\)"
    )
    return re.search(register_for_id, lineStr) is not None


def findRegisterFor(lineStr: str) -> tuple[str, list[str]]:
    # [FUNCTION DOC] HELPER: Collects all single-quoted tokens in the line and returns the
    # first as the op name and the rest as registry keys.
    # KNOWN ISSUE: the none check can never trigger because findall returns a list, and its
    # error text names the wrong function (getBiasStorageType).
    register_for_pattern = r"'([A-Za-z0-9_]+)'"
    matches = re.findall(register_for_pattern, lineStr)
    if matches is None:
        raise AssertionError("matches is None in getBiasStorageType")
    matches_list = list(matches)
    return (matches_list[0], matches_list[1:])


typeIdMapping = {
    r"image[123]D\b": "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE",
    r"sampler[123]D\b": "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER",
    r"\bbuffer\b": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
    r"\buniform\b": "VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER",
}

storageTypeToEnum = {
    "TEXTURE_2D": "api::StorageType::TEXTURE_2D",
    "TEXTURE_3D": "api::StorageType::TEXTURE_3D",
    "BUFFER": "api::StorageType::BUFFER",
    "": "api::StorageType::UNKNOWN",
}


def determineDescriptorType(lineStr: str) -> str:
    # [FUNCTION DOC] HELPER: Returns the Vulkan descriptor type name for a layout line by
    # testing, in order, image, sampler, buffer and uniform keywords. Raises an assertion
    # error if none matches.
    for identifier, typeNum in typeIdMapping.items():
        if re.search(identifier, lineStr):
            return typeNum
    raise AssertionError(
        "No matching descriptor type for " + lineStr + " in determineDescriptorType"
    )


def getShaderInfo(srcFilePath: str) -> ShaderInfo:
    # [FUNCTION DOC] Reads a shader source file line by line and returns a ShaderInfo with
    # descriptor layouts (in file order) and the tile size, storage types and registration
    # pair found in header comments; later matching lines overwrite earlier ones.
    shader_info = ShaderInfo([], [], "")
    with open(srcFilePath) as srcFile:
        for line in srcFile:
            if isDescriptorLine(line):
                shader_info.layouts.append(determineDescriptorType(line))
            if isTileSizeLine(line):
                shader_info.tile_size = findTileSizes(line)
            if isWeightStorageTypeLine(line):
                shader_info.weight_storage_type = getWeightStorageType(line)
            if isBiasStorageTypeLine(line):
                shader_info.bias_storage_type = getBiasStorageType(line)
            if isRegisterForLine(line):
                shader_info.register_for = findRegisterFor(line)

    return shader_info


##########################
#  C++ File Generation  #
#########################

cpp_template = """
#include <ATen/native/vulkan/api/ShaderRegistry.h>
#include <stdint.h>
#include <vector>

using namespace at::native::vulkan;

namespace at {{
namespace native {{
namespace vulkan {{

namespace {{

{spv_bin_arrays}

}}

static void register_fn() {{

{register_shader_infos}

{shader_info_registry}

}}

static const api::ShaderRegisterInit register_shaders(&register_fn);

}}
}}
}}

"""


def generateSpvBinStr(spvPath: str, name: str) -> tuple[int, str]:
    # [FUNCTION DOC] Reads a SPIR-V binary as 32-bit unsigned words and returns its size in
    # bytes plus C++ text defining a constant word array named after name with a bin suffix.
    # Reads the file; a size not divisible by four raises an error.
    with open(spvPath, "rb") as fr:
        next_bin = array.array("I", fr.read())
        sizeBytes = 4 * len(next_bin)
        spv_bin_str = "const uint32_t {}_bin[] = {{\n{}\n}};".format(
            name,
            textwrap.indent(",\n".join(str(x) for x in next_bin), "  "),
        )

    return sizeBytes, spv_bin_str


def generateShaderInfoStr(shader_info: ShaderInfo, name: str, sizeBytes: int) -> str:
    # [FUNCTION DOC] Returns indented C++ text registering the shader in the shader registry,
    # passing its name, binary array, byte size, descriptor layouts, tile size (an empty
    # vector if unknown) and the weight and bias storage type enum names (unknown if unset).
    tile_size = (
        f"{{{', '.join(str(x) for x in shader_info.tile_size)}}}"
        if (len(shader_info.tile_size) > 0)
        else "std::vector<uint32_t>()"
    )

    shader_info_layouts = "{{{}}}".format(",\n ".join(shader_info.layouts))

    shader_info_args = [
        f'"{name}"',
        f"{name}_bin",
        str(sizeBytes),
        shader_info_layouts,
        tile_size,
        storageTypeToEnum[shader_info.weight_storage_type],
        storageTypeToEnum[shader_info.bias_storage_type],
    ]

    shader_info_str = textwrap.indent(
        "api::shader_registry().register_shader(\n  api::ShaderInfo(\n{args}));\n".format(
            args=textwrap.indent(",\n".join(shader_info_args), "     "),
        ),
        "    ",
    )

    return shader_info_str


def generateShaderDispatchStr(shader_info: ShaderInfo, name: str) -> str:
    # [FUNCTION DOC] Returns indented C++ text registering an op dispatch entry for the
    # shader, or an empty string when the shader has no registration info.
    # KNOWN ISSUE: the loop over registry keys reassigns the result each time, so only the
    # last key is registered, and an empty key list would leave the result unassigned.
    if shader_info.register_for is None:
        return ""

    (op_name, registry_keys) = shader_info.register_for
    for registry_key in registry_keys:
        shader_dispatch_str = textwrap.indent(
            f'api::shader_registry().register_op_dispatch("{op_name}", api::DispatchKey::{registry_key.upper()}, "{name}");',
            "    ",
        )

    # pyrefly: ignore [unbound-name]
    return shader_dispatch_str


def genCppFiles(
    spv_files: dict[str, str], cpp_header_path: str, cpp_src_file_path: str
) -> None:
    # [FUNCTION DOC] For every compiled shader (SPIR-V path mapped to its GLSL path), builds
    # the binary array text, shader registration text and optional dispatch registration
    # text, then fills the C++ template and writes it to cpp_src_file_path.
    # KNOWN ISSUE: cpp_header_path is accepted but never used; no header is written.
    spv_bin_strs = []
    register_shader_info_strs = []
    shader_registry_strs = []

    for spvPath, srcPath in spv_files.items():
        name = getName(spvPath).replace("_spv", "")

        sizeBytes, spv_bin_str = generateSpvBinStr(spvPath, name)
        spv_bin_strs.append(spv_bin_str)

        shader_info = getShaderInfo(srcPath)

        register_shader_info_strs.append(
            generateShaderInfoStr(shader_info, name, sizeBytes)
        )

        if shader_info.register_for is not None:
            shader_registry_strs.append(generateShaderDispatchStr(shader_info, name))

    spv_bin_arrays = "\n".join(spv_bin_strs)
    register_shader_infos = "\n".join(register_shader_info_strs)
    shader_info_registry = "\n".join(shader_registry_strs)

    cpp = cpp_template.format(
        spv_bin_arrays=spv_bin_arrays,
        register_shader_infos=register_shader_infos,
        shader_info_registry=shader_info_registry,
    )

    with open(cpp_src_file_path, "w") as fw:
        fw.write(cpp)


##########
#  Main  #
##########


def parse_arg_env(items: dict[Any, Any]) -> dict[Any, Any]:
    # [FUNCTION DOC] HELPER: Converts command line key=value strings (the annotation says
    # mapping but a list of strings is expected) into a mapping with stripped keys and values.
    # Returns an empty mapping when items is empty or missing.
    # KNOWN ISSUE: splitting on every equals sign drops anything after a second one, and an
    # item lacking an equals sign raises an index error.
    d = {}
    if items:
        for item in items:
            tokens = item.split("=")
            key = tokens[0].strip()
            value = tokens[1].strip()
            d[key] = value
    return d


def main(argv: list[str]) -> int:
    # [FUNCTION DOC] Command line driver with options for GLSL search paths, glslc path, temp
    # dir, output dir and extra key=value environment entries. Merges the type and function
    # tables and user entries into the module-level default environment (mutating it), creates
    # missing directories, generates GLSL and SPIR-V into the temp dir, and writes the C++
    # source into the output dir. Returns 0.
    # KNOWN ISSUE: argv is ignored; arguments are parsed from the process command line.
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-i",
        "--glsl-paths",
        nargs="+",
        help='List of paths to look for GLSL source files, separated by spaces. Ex: --glsl-paths "path1 path2 path3"',
        default=["."],
    )
    parser.add_argument("-c", "--glslc-path", required=True, help="")
    parser.add_argument("-t", "--tmp-dir-path", required=True, help="/tmp")
    parser.add_argument("-o", "--output-path", required=True, help="")
    parser.add_argument(
        "--env", metavar="KEY=VALUE", nargs="*", help="Set a number of key-value pairs"
    )
    options = parser.parse_args()

    DEFAULT_ENV.update(TYPES_ENV)
    DEFAULT_ENV.update(FUNCS_ENV)
    env = DEFAULT_ENV

    for key, value in parse_arg_env(options.env).items():
        env[key] = value

    if not os.path.exists(options.output_path):
        os.makedirs(options.output_path)

    if not os.path.exists(options.tmp_dir_path):
        os.makedirs(options.tmp_dir_path)

    shader_generator = SPVGenerator(options.glsl_paths, env, options.glslc_path)
    output_spv_files = shader_generator.generateSPV(options.tmp_dir_path)

    genCppFiles(
        output_spv_files,
        f"{options.output_path}/{CPP_H_NAME}",
        f"{options.output_path}/{CPP_SRC_NAME}",
    )

    return 0


def invoke_main() -> None:
    # [FUNCTION DOC] Runs main with the process arguments and exits with its return status.
    sys.exit(main(sys.argv))


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
