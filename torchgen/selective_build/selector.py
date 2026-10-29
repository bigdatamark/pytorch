# [MODULE DOC]
# Defines SelectiveBuilder, the in-memory form of a selective build YAML specification, which
# answers whether operators, native functions and kernel dtypes are selected, whether they
# are root or training operators, and serializes back to a dictionary. Also provides helpers
# to merge builders and kernel metadata and to compute an operator name from a NativeFunction.
# Imports yaml; no other import-time side effects.
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml

from torchgen.selective_build.operator import (
    merge_debug_info,
    merge_operator_dicts,
    SelectiveBuildOperator,
    strip_operator_overload_name,
)


if TYPE_CHECKING:
    from torchgen.model import NativeFunction


# A SelectiveBuilder holds information extracted from the selective build
# YAML specification.
#
# It includes information about the build's selectivity, the debug_info
# associated with this selective build (opaque string), and the set of
# operators that should be included in the build.
#
@dataclass(frozen=True)
class SelectiveBuilder:
    # [CLASS DOC]
    # Immutable description of a selective build. State: include_all_operators (build is not
    # selective), private debug info tuple, operators (mapping of name to
    # SelectiveBuildOperator), kernel_metadata (kernel tag to dtype list),
    # et_kernel_metadata (ExecuTorch kernel tag to list of dtype/dim-order keys),
    # custom_classes and build_features (sets), and include_all_non_op_selectives (keep all
    # dtypes and classes). Key methods: constructors from YAML, the is_*_selected queries and
    # to_dict.
    # If true, then the build is not selective, and includes all
    # operators.
    include_all_operators: bool

    # Debug Information at the selective/custom build level.
    _debug_info: tuple[str, ...] | None

    # A dictionary of operator -> operator metadata.
    operators: dict[str, SelectiveBuildOperator]

    # A dictionary of selected kernel tags and dtypes. Typically a
    # PyTorch Operator Kernel (function) may have many code paths
    # that are specialized for many many Tensor dtypes, so it's not
    # one per kernel function, but there could be many per kernel
    # function. The tag isn't a kernel function name, but some fragment
    # of the kernel function implementation itself.
    kernel_metadata: dict[str, list[str]]

    # ExecuTorch only. A dictionary of kernel tag -> list of (list of input
    # dtypes for tensor-like input args).
    # This is from selective.yaml
    et_kernel_metadata: dict[str, list[str]]

    # A set of all the custom torch bind classes used by the selected models
    # Stored as a set internally to remove duplicates proactively, but written
    # as a list to yamls
    custom_classes: set[str]

    # A set of all the build features used by the selected models
    # Stored as a set internally to remove duplicates proactively, but written
    # as a list to yamls
    build_features: set[str]

    # If true, then fragments for all dtypes for all kernel functions
    # are included as well as all custom classes. This is typically set when any one of the
    # operator lists is generated from a mechanism other than
    # tracing based selective build.
    include_all_non_op_selectives: bool

    @staticmethod
    def get_nop_selector() -> SelectiveBuilder:
        # [FUNCTION DOC]
        # Returns a builder that is not selective: it includes all operators.
        return SelectiveBuilder.from_yaml_dict({"include_all_operators": True})

    @staticmethod
    def from_yaml_dict(data: dict[str, object]) -> SelectiveBuilder:
        # [FUNCTION DOC]
        # Builds a SelectiveBuilder from a parsed YAML mapping. All keys are optional; the
        # two include_all flags default to false, debug_info becomes a tuple of strings,
        # operators are parsed into SelectiveBuildOperator records, kernel_metadata values
        # are stringified, and custom_classes/build_features become sets. Raises a generic
        # Exception on unknown top level keys and AssertionError on wrong value types.
        # KNOWN ISSUE: et_kernel_metadata is only type-checked as a mapping; its values are
        # not validated or converted.
        valid_top_level_keys = {
            "include_all_non_op_selectives",
            "include_all_operators",
            "debug_info",
            "operators",
            "kernel_metadata",
            "et_kernel_metadata",
            "custom_classes",
            "build_features",
        }
        top_level_keys = set(data.keys())
        if len(top_level_keys - valid_top_level_keys) > 0:
            raise Exception(  # noqa: TRY002
                "Got unexpected top level keys: {}".format(
                    ",".join(top_level_keys - valid_top_level_keys),
                )
            )
        include_all_operators = data.get("include_all_operators", False)
        if not isinstance(include_all_operators, bool):
            raise AssertionError(
                f"Expected 'include_all_operators' to be bool, got {type(include_all_operators)}"
            )

        debug_info = None
        if "debug_info" in data:
            di_list = data["debug_info"]
            if not isinstance(di_list, list):
                raise AssertionError(
                    f"Expected 'debug_info' to be list, got {type(di_list)}"
                )

            debug_info = tuple(str(x) for x in di_list)

        operators = {}
        operators_dict = data.get("operators", {})
        if not isinstance(operators_dict, dict):
            raise AssertionError(
                f"Expected 'operators' to be dict, got {type(operators_dict)}"
            )

        for k, v in operators_dict.items():
            operators[k] = SelectiveBuildOperator.from_yaml_dict(k, v)

        kernel_metadata = {}
        kernel_metadata_dict = data.get("kernel_metadata", {})
        if not isinstance(kernel_metadata_dict, dict):
            raise AssertionError(
                f"Expected 'kernel_metadata' to be dict, got {type(kernel_metadata_dict)}"
            )

        for k, v in kernel_metadata_dict.items():
            kernel_metadata[str(k)] = [str(dtype) for dtype in v]

        et_kernel_metadata = data.get("et_kernel_metadata", {})
        if not isinstance(et_kernel_metadata, dict):
            raise AssertionError(
                f"Expected 'et_kernel_metadata' to be dict, got {type(et_kernel_metadata)}"
            )

        custom_classes = data.get("custom_classes", [])
        if not isinstance(custom_classes, Iterable):
            raise AssertionError(
                f"Expected 'custom_classes' to be Iterable, got {type(custom_classes)}"
            )
        custom_classes = set(custom_classes)

        build_features = data.get("build_features", [])
        if not isinstance(build_features, Iterable):
            raise AssertionError(
                f"Expected 'build_features' to be Iterable, got {type(build_features)}"
            )
        build_features = set(build_features)

        include_all_non_op_selectives = data.get("include_all_non_op_selectives", False)
        if not isinstance(include_all_non_op_selectives, bool):
            raise AssertionError(
                f"Expected 'include_all_non_op_selectives' to be bool, "
                f"got {type(include_all_non_op_selectives)}"
            )

        return SelectiveBuilder(
            include_all_operators,
            debug_info,
            operators,
            kernel_metadata,
            et_kernel_metadata,
            custom_classes,  # type: ignore[arg-type]
            build_features,  # type: ignore[arg-type]
            include_all_non_op_selectives,
        )

    @staticmethod
    def from_yaml_str(config_contents: str) -> SelectiveBuilder:
        # [FUNCTION DOC]
        # Parses YAML text with the safe loader and builds a SelectiveBuilder from it.
        contents = yaml.safe_load(config_contents)
        return SelectiveBuilder.from_yaml_dict(contents)

    @staticmethod
    def from_yaml_path(config_path: str) -> SelectiveBuilder:
        # [FUNCTION DOC]
        # Reads the YAML file at config_path (I/O) and builds a SelectiveBuilder from it.
        # Raises the usual file errors if the file cannot be opened.
        with open(config_path) as f:
            contents = yaml.safe_load(f)
            return SelectiveBuilder.from_yaml_dict(contents)

    @staticmethod
    def from_legacy_op_registration_allow_list(
        allow_list: set[str], is_root_operator: bool, is_used_for_training: bool
    ) -> SelectiveBuilder:
        # [FUNCTION DOC]
        # Builds a SelectiveBuilder from a legacy set of operator names: every operator gets
        # the given root and training flags and covers all overloads, and all non-operator
        # selectives are included.
        operators = {}
        for op in allow_list:
            operators[op] = {
                "name": op,
                "is_root_operator": is_root_operator,
                "is_used_for_training": is_used_for_training,
                "include_all_overloads": True,
            }
        return SelectiveBuilder.from_yaml_dict(
            {
                "operators": operators,
                "include_all_non_op_selectives": True,
            }
        )

    def is_operator_selected(self, name: str) -> bool:
        # [FUNCTION DOC]
        # True if the build includes all operators, the exact name is listed, or the name
        # without overload is listed with include_all_overloads set.
        if self.include_all_operators:
            return True

        if name in self.operators:
            return True
        name = strip_operator_overload_name(name)
        return name in self.operators and self.operators[name].include_all_overloads

    def is_native_function_selected(self, func: NativeFunction) -> bool:
        # [FUNCTION DOC]
        # Checks selection of a native function by its namespaced operator name.
        op_name = op_name_from_native_function(func)
        return self.is_operator_selected(op_name)

    def is_operator_selected_for_training(self, name: str) -> bool:
        # [FUNCTION DOC]
        # True if the operator is selected and either all operators are included, the exact
        # entry is used for training, or the base-name entry covers all overloads and is used
        # for training. Unlisted entries count as not used for training.
        if not self.is_operator_selected(name):
            return False
        if self.include_all_operators:
            return True

        not_training_op = SelectiveBuildOperator(
            name="",
            is_root_operator=False,
            is_used_for_training=False,
            include_all_overloads=False,
            _debug_info=None,
        )
        op = not_training_op
        if name in self.operators:
            op = self.operators[name]

        name = strip_operator_overload_name(name)
        base_op = not_training_op
        if name in self.operators:
            base_op = self.operators[name]

        return op.is_used_for_training or (
            base_op.include_all_overloads and base_op.is_used_for_training
        )

    def is_native_function_selected_for_training(self, func: NativeFunction) -> bool:
        # [FUNCTION DOC]
        # Training-selection check for a native function by its namespaced operator name.
        op_name = op_name_from_native_function(func)
        return self.is_operator_selected_for_training(op_name)

    def is_root_operator(self, name: str) -> bool:
        # [FUNCTION DOC]
        # True if the operator is selected and is a root operator: always true when all
        # operators are included; otherwise taken from the exact entry, or from the base-name
        # entry when it covers all overloads.
        if not self.is_operator_selected(name):
            return False
        if self.include_all_operators:
            return True

        if name in self.operators:
            op: SelectiveBuildOperator = self.operators[name]
            return op.is_root_operator
        name = strip_operator_overload_name(name)
        if name not in self.operators:
            return False
        base_op: SelectiveBuildOperator = self.operators[name]
        return base_op.include_all_overloads and base_op.is_root_operator

    def is_kernel_dtype_selected(self, kernel_tag: str, dtype: str) -> bool:
        # [FUNCTION DOC]
        # True if all operators or all non-operator selectives are included, or the dtype is
        # listed under the kernel tag in kernel_metadata.
        if self.include_all_operators or self.include_all_non_op_selectives:
            return True

        return (
            kernel_tag in self.kernel_metadata
            and dtype in self.kernel_metadata[kernel_tag]
        )

    def et_get_selected_kernels(self, op_name: str, kernel_key: list[str]) -> list[str]:
        # [FUNCTION DOC]
        # ExecuTorch only. Chooses which of the candidate kernel keys cover the models using
        # op_name. Without metadata for the op it returns all keys if all operators are
        # included, else an empty list. Otherwise each model key is matched to a candidate
        # with the same second path segment (version ignored), falling back to "default".
        # Raises a generic Exception if no match and no default exists. Result order is
        # unspecified.
        """
        Return a list of kernel keys that cover the used ops
        """
        # If no kernel metadata, either it's implied by include_all_operators=True or the op is not used.
        if op_name not in self.et_kernel_metadata:
            return kernel_key if self.include_all_operators else []
        # Otherwise, only return the specific kernel keys.

        result_set = set()

        for model_kernel_keys in self.et_kernel_metadata[op_name]:
            key_found = False
            for key in kernel_key:
                # Don't compare the version for now
                if (
                    key != "default"
                    and key.split("/")[1] == model_kernel_keys.split("/")[1]
                ):
                    result_set.add(key)
                    key_found = True
                    break
            if not key_found:
                if "default" not in kernel_key:
                    raise Exception("Missing kernel for the model")  # noqa: TRY002
                else:
                    result_set.add("default")

        return list(result_set)

    def to_dict(self) -> dict[str, object]:
        # [FUNCTION DOC]
        # Serializes to a mapping: both include flags, operators as dictionaries, sorted
        # debug info when present, kernel metadata with sorted dtype lists, the ExecuTorch
        # metadata as stored, and sorted custom classes and build features.
        ret: dict[str, object] = {
            "include_all_non_op_selectives": self.include_all_non_op_selectives,
            "include_all_operators": self.include_all_operators,
        }
        operators = {}
        for op_name, op in self.operators.items():
            operators[op_name] = op.to_dict()
        ret["operators"] = operators

        if self._debug_info is not None:
            ret["debug_info"] = sorted(self._debug_info)

        ret["kernel_metadata"] = {
            k: sorted(v) for (k, v) in self.kernel_metadata.items()
        }

        ret["et_kernel_metadata"] = self.et_kernel_metadata

        ret["custom_classes"] = sorted(self.custom_classes)

        ret["build_features"] = sorted(self.build_features)

        return ret


def merge_kernel_metadata(
    lhs: dict[str, list[str]],
    rhs: dict[str, list[str]],
) -> dict[str, list[str]]:
    # [FUNCTION DOC] HELPER: Returns a new mapping of kernel tag to the de-duplicated union of
    # dtypes from both inputs; list order is unspecified.
    kernel_metadata: dict[str, list[str]] = {}
    for tag_name, dtypes in list(lhs.items()) + list(rhs.items()):
        dtypes_copy = set(dtypes)
        if tag_name in kernel_metadata:
            dtypes_copy |= set(kernel_metadata[tag_name])

        kernel_metadata[tag_name] = list(dtypes_copy)

    return kernel_metadata


def merge_et_kernel_metadata(
    lhs: dict[str, list[str]],
    rhs: dict[str, list[str]],
) -> dict[str, list[str]]:
    # [FUNCTION DOC] HELPER: Returns a new mapping of op name to the sorted union of kernel
    # keys from both inputs.
    merge_et_kernel_metadata: dict[str, set[str]] = defaultdict(set)
    for op in list(lhs.keys()) + list(rhs.keys()):
        merge_et_kernel_metadata[op].update(lhs.get(op, []))
        merge_et_kernel_metadata[op].update(rhs.get(op, []))

    return {op: sorted(val) for op, val in merge_et_kernel_metadata.items()}


def combine_selective_builders(
    lhs: SelectiveBuilder, rhs: SelectiveBuilder
) -> SelectiveBuilder:
    # [FUNCTION DOC]
    # Merges two builders into a new one: flags are or-ed, and debug info, operators, kernel
    # metadata, ExecuTorch metadata, custom classes and build features are unioned/merged.
    include_all_operators = lhs.include_all_operators or rhs.include_all_operators
    debug_info = merge_debug_info(lhs._debug_info, rhs._debug_info)
    operators = merge_operator_dicts(lhs.operators, rhs.operators)
    kernel_metadata = merge_kernel_metadata(lhs.kernel_metadata, rhs.kernel_metadata)
    et_kernel_metadata = merge_et_kernel_metadata(
        lhs.et_kernel_metadata, rhs.et_kernel_metadata
    )
    include_all_non_op_selectives = (
        lhs.include_all_non_op_selectives or rhs.include_all_non_op_selectives
    )
    custom_classes = lhs.custom_classes.union(rhs.custom_classes)
    build_features = lhs.build_features.union(rhs.build_features)
    return SelectiveBuilder(
        include_all_operators,
        debug_info,
        operators,
        kernel_metadata,
        et_kernel_metadata,
        custom_classes,
        build_features,
        include_all_non_op_selectives,
    )


def op_name_from_native_function(f: NativeFunction) -> str:
    # [FUNCTION DOC] HELPER: Returns the operator name as namespace, a double colon, then the
    # function's full name including overload.
    # This was originally read from the 'operator_name_with_overload' field in the
    # declaration dict, which was the part before the first '(' in 'schema_string'.
    return f"{f.namespace}::{f.func.name}"
