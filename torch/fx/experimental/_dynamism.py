# [MODULE DOC] Utilities for detecting which tensor dimensions and scalar values vary across a
# set of example inputs (to infer dynamic shapes). Exports a source-name normalizer, a
# module-to-nested-mapping converter, the dynamism tracker, and a clone-to-meta helper.
# No import-time side effects.
import re
from collections.abc import Callable
from typing import Any

import torch
from torch.utils._pytree import tree_flatten_with_path, tree_map


KeyPath = tuple[Any, ...]
NonTensorShapeFn = Callable[[int | float], tuple[Any, ...]]

__all__ = [
    "normalize_source_name",
    "module_to_nested_dict",
    "track_dynamism_across_examples",
    "clone_and_convert_to_meta",
]


def normalize_source_name(name: str) -> str:
    # [FUNCTION DOC] Rewrite every dotted attribute access in a source name string into a
    # bracketed quoted-key access (a dot followed by an identifier becomes a bracketed single-
    # quoted key). Returns the new string; pure.
    # Match attribute access like .x and replace with ['x']
    return re.sub(r"\.([a-zA-Z_][a-zA-Z0-9_]*)", r"['\1']", name)


def module_to_nested_dict(module: torch.nn.Module) -> dict[str, Any]:
    # [FUNCTION DOC] Convert a module into a nested mapping with two special entries (private
    # names for parameters and for child modules) plus one entry per public, non-callable
    # attribute whose value is an int, float or tensor (booleans and modules excluded; attributes
    # raising NotImplementedError on access are skipped). Direct parameters AND buffers are
    # both stored under the parameters entry; children are converted recursively. Pure apart
    # from attribute access on the module.
    # KNOWN ISSUE: docstring says keys are 'parameters' and 'modules', but the code uses keys
    # with a leading underscore; buffers are also placed under the parameters entry.
    """Recursively converts an nn.Module into a nested dictionary with explicit 'parameters' and 'modules' keys."""
    self_dict: dict[str, Any] = {}

    parameters: dict[str, torch.Tensor] = {}
    modules: dict[str, dict[str, Any]] = {}
    self_dict["_parameters"] = parameters
    self_dict["_modules"] = modules

    for attr_name in dir(module):
        try:
            if not attr_name.startswith("_") and not callable(
                getattr(module, attr_name)
            ):
                attr_value = getattr(module, attr_name)
                if (
                    not isinstance(attr_value, torch.nn.Module)
                    and isinstance(attr_value, (int, float, torch.Tensor))
                    and type(attr_value) is not bool
                ):
                    self_dict[attr_name] = attr_value
        except NotImplementedError:
            # Skip attributes that raise NotImplementedError since they won't
            # contain any dynamism anyways.
            continue

    for name, param in module.named_parameters(recurse=False):
        self_dict["_parameters"][name] = param
    for name, buffer in module.named_buffers(recurse=False):
        self_dict["_parameters"][name] = buffer

    for name, submodule in module.named_children():
        self_dict["_modules"][name] = module_to_nested_dict(submodule)

    return self_dict


def track_dynamism_across_examples(
    example_inputs: list[Any],
) -> dict[Any, Any]:
    # [FUNCTION DOC] Given a list of example-input mappings, record per leaf path (int, float or
    # tensor leaves only) the set of observed sizes for each dimension (a scalar counts as a
    # one-dimension shape). Returns a mapping from the top-level key to a mapping from a path
    # string (prefixed with "L") to a tuple of booleans, true where more than one distinct
    # size was seen. Side effect: if an example has a "self" entry that is a module, that entry
    # is replaced in place by its nested-mapping form. Dimension lists grow if later examples
    # have more dimensions.
    # KNOWN ISSUE: a mismatch between tensor and non-tensor leaves at the same path is detected
    # but ignored; also the example inputs are mutated despite the docstring.
    """
    This function analyzes a list of example inputs to determine the dynamism of their shapes.
    It tracks whether the dimensions of tensors or non-tensor values change across
    different examples. The function returns a dictionary where each key represents
    a path to a value in the input examples, and the corresponding value is a tuple
    indicating which dimensions are dynamic (i.e., change across examples). This
    helps in understanding how the structure of data varies across different instances.
    """
    tracking: dict[KeyPath, tuple[list[set[Any]], bool]] = {}

    for ex in example_inputs:
        if "self" in ex and isinstance(ex["self"], torch.nn.Module):
            ex["self"] = module_to_nested_dict(ex["self"])
        leaves_with_paths, _ = tree_flatten_with_path(ex)
        for key_path, value in leaves_with_paths:
            if not isinstance(value, (int, float, torch.Tensor)):
                continue
            if isinstance(value, torch.Tensor):
                shape: tuple[int | float, ...] = tuple(value.shape)
                is_tensor = True
            else:
                shape = (value,)
                is_tensor = False
            if key_path not in tracking:
                tracking[key_path] = ([set() for _ in range(len(shape))], is_tensor)
            else:
                dim_sets, flag = tracking[key_path]
                if flag != is_tensor:
                    pass
                while len(dim_sets) < len(shape):
                    dim_sets.append(set())
            for i, dim in enumerate(shape):
                tracking[key_path][0][i].add(dim)

    output: dict[Any, Any] = {}
    for key_path, (dim_sets, _is_tensor) in tracking.items():
        final_dyn = tuple(len(s) > 1 for s in dim_sets)
        key_str = "L" + "".join(f"{str(k)}" for k in key_path)
        key = key_path[0].key  # type: ignore[attr-defined]
        if key not in output:
            output[key] = {}
        output[key][key_str] = final_dyn
    return output


def clone_and_convert_to_meta(example_input: Any) -> Any:
    # [FUNCTION DOC] Map over a nested structure (any pytree, not only a list) and return a
    # new structure where each tensor is cloned and moved to the meta device; other leaves are
    # kept by reference.
    """
    This function takes a list of example inputs and for each tensor, clones it and converts it to device=meta.
    For non-tensor values, it keeps the reference. It uses pytree to handle nested structures recursively.
    """

    def transform_fn(value: Any) -> Any:
        # [FUNCTION DOC] HELPER: Return a meta-device clone for a tensor, else the value itself.
        if isinstance(value, torch.Tensor):
            return value.clone().to(device="meta")
        return value

    return tree_map(transform_fn, example_input)
