# [MODULE DOC] Helpers to fetch attributes of known leaf nn.Module types for
# lowering. Contains module_fetch_book (mapping from module class to a version,
# a list of attribute names, and a name-matching function), default_matching,
# extract_attrs_for_lowering and lift_lowering_attrs_to_nodes. No import-time
# side effects besides building the book.
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule


__all__ = [
    "default_matching",
    "extract_attrs_for_lowering",
    "lift_lowering_attrs_to_nodes",
]


# Matching method matches the attribute name of current version to the attribute name of `target_version`
@compatibility(is_backward_compatible=False)
def default_matching(name: str, target_version: int) -> str:
    """Default matching method"""
    # [FUNCTION DOC] Identity name matcher: returns the attribute name unchanged;
    # the target version is ignored.
    return name


# This dict maps the nn.Module class name to the attribute name list that we want to fetch for lowering.
# The first integer in the tuple is the version number of the nn.Module class when we create the parameter list.
# If there's a version mismatch then it means the parameter names in the book might be mismatched with nn.Module.
module_fetch_book: dict[type, tuple[int, list[str], Callable[[str, int], str]]] = {
    torch.nn.modules.linear.Linear: (1, ["weight", "bias"], default_matching),
    torch.nn.modules.conv.Conv2d: (
        1,
        [
            "weight",
            "bias",
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "groups",
            "padding_mode",
        ],
        default_matching,
    ),
    torch.nn.modules.batchnorm.BatchNorm2d: (
        2,
        ["weight", "bias", "running_mean", "running_var", "eps"],
        default_matching,
    ),
    torch.nn.modules.pooling.AdaptiveAvgPool2d: (1, [], default_matching),
    torch.nn.modules.pooling.MaxPool2d: (
        1,
        ["kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"],
        default_matching,
    ),
    torch.nn.modules.activation.ReLU: (1, ["inplace"], default_matching),
}


@compatibility(is_backward_compatible=False)
def extract_attrs_for_lowering(mod: nn.Module) -> dict[str, Any]:
    """If `mod` is in `module_fetch_book`, fetch the mod's attributes that in the `module_fetch_book`
    after checking module's version is compatible with the `module_fetch_book`.
    """
    # [FUNCTION DOC] Build a mapping of attribute name to value for a supported
    # module. Always includes the key name holding the module's qualified type
    # name. Looks up the module's exact type in the fetch book, then reads each
    # listed attribute (after mapping its name for the module's version).
    # Raises RuntimeError if the book version is older than the module's version
    # or if the module type is not in the book. No mutation.
    attrs_for_lowering: dict[str, Any] = {}
    attrs_for_lowering["name"] = torch.typename(mod)

    if type(mod) in module_fetch_book:
        version, param_to_fetch, matching_method = module_fetch_book[type(mod)]
        if version < mod._version:
            raise RuntimeError(
                f"Fetcher version {version} try to fetch {torch.typename(mod)} version {mod._version}, "
                "please upgrade the module_fetch_book, open an issue and @842974287 "
                "or report a bug to AIACC team directly."
            )
        for attr in param_to_fetch:
            attrs_for_lowering[attr] = getattr(mod, matching_method(attr, mod._version))
    else:
        raise RuntimeError(
            f"{torch.typename(mod)} is not in the module_fetch_book yet, "
            "please add it to the module_fetch_book, open an issue and @842974287 "
            "or report a bug to AIACC team directly."
        )
    return attrs_for_lowering


@compatibility(is_backward_compatible=False)
def lift_lowering_attrs_to_nodes(fx_module: GraphModule) -> None:
    """Recursively traverse all `fx_module` nodes and fetch the module's attributes if the node is a leaf module."""
    # [FUNCTION DOC] Walk the graph's nodes; for each call_module node, recurse into
    # nested GraphModules, otherwise set an attribute on the node holding the
    # extracted lowering attributes of the target submodule. Mutates node objects.
    # Raises RuntimeError for unsupported leaf module types.
    submodules = dict(fx_module.named_modules())

    for node in fx_module.graph.nodes:
        if node.op == "call_module":
            if isinstance(submodules[node.target], GraphModule):
                lift_lowering_attrs_to_nodes(submodules[node.target])
            else:
                node.attrs_for_lowering = extract_attrs_for_lowering(
                    submodules[node.target]
                )
