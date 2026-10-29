# Copyright (c) Meta Platforms, Inc. and affiliates
# [MODULE DOC] HELPER: Utility that regroups a flat FX graph into nested submodules using the
# torch.export unflatten machinery, based on node module-hierarchy metadata.
from collections import defaultdict

import torch
from torch.export.unflatten import _ModuleFrame, _SubmoduleEntry


def _outline_submodules(orig_graph: torch.fx.Graph) -> torch.fx.GraphModule:
    # [FUNCTION DOC] HELPER: Builds a new GraphModule whose nested submodules are outlined from the
    # given graph by running the export unflatten module-frame pass over all its nodes. Lints the
    # resulting graph and recompiles it before returning it. Does not modify the input graph.
    # Create an empty GraphModule to hold the outlined modules
    new_module = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
    seen_nodes: dict[str, torch.fx.Node] = {}
    seen_modules: dict[int, list[_SubmoduleEntry]] = defaultdict(list)
    seen_attrs: dict[str, set[str]] = defaultdict(set)
    created_modules: dict[str, torch.nn.Module] = {}
    _ModuleFrame(
        orig_graph,
        tuple(orig_graph.nodes),
        seen_nodes,
        seen_modules,
        seen_attrs,
        created_modules,
        None,
        [("", None, 0)],
        "",
        {},
        module=new_module,
    ).run_outer()
    new_module.graph.lint()
    new_module.recompile()
    return new_module
