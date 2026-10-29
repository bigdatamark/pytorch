# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC]
# Provides an unsafe pass that undoes auto_functionalized higher order ops in an
# ExportedProgram, restoring the original in-place mutating ops, plus a helper that
# removes redundant self-copies. No import-time side effects beyond imports.

import torch
from torch._higher_order_ops.auto_functionalize import (
    auto_functionalized,
    auto_functionalized_v2,
)
from torch._inductor.fx_passes.post_grad import decompose_auto_functionalized
from torch.export import ExportedProgram
from torch.fx import Graph


def remove_self_clone(graph: Graph) -> None:
    # [FUNCTION DOC]
    # Mutates the given fx graph in place: for every aten copy_ node whose destination and
    # source are the same node, replaces all uses of the copy with that node and erases the copy.
    # Returns nothing.
    # KNOWN ISSUE: erases nodes while iterating over graph.nodes; fx tolerates this, but it
    # is fragile.
    for node in graph.nodes:
        if node.target is torch.ops.aten.copy_.default and node.args[0] == node.args[1]:
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)


def unsafe_remove_auto_functionalized_pass(
    ep: ExportedProgram,
) -> ExportedProgram:
    # [FUNCTION DOC]
    # Mutates and returns the given ExportedProgram. Under the graph signature's replace hook,
    # marks each auto_functionalized / auto_functionalized_v2 call node so that no tensors are
    # cloned (full re-inplacing), decomposes them back into the original mutating ops, removes
    # self copies and dead code. Raises AssertionError if a node's first arg is not an OpOverload.
    # No safety check that in-place mutation is valid.
    # KNOWN ISSUE: the loop over submodules ignores the module variable and always operates on
    # ep.graph, so nested GraphModules are never processed and the main graph is processed once
    # per submodule.
    """
    This pass removes an instances of the higher order op 'auto_functionalized',
    and modifies the calling EP inplace to have the original mutator op.
    This pass doesn't perform safety checks to make sure that this inplace mutation is safe.
    """

    with ep.graph_module._set_replace_hook(ep.graph_signature.get_replace_hook()):
        for module in ep.graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in ep.graph.nodes:
                if (
                    node.op == "call_function" and node.target is auto_functionalized
                ) or (
                    node.op == "call_function" and node.target is auto_functionalized_v2
                ):
                    func = node.args[0]
                    if not isinstance(func, torch._ops.OpOverload):
                        raise AssertionError(
                            f"Expected func to be an OpOverload, but got {type(func)}"
                        )
                    # re-inplace everything
                    node.meta["only_clone_these_tensors"] = []
            decompose_auto_functionalized(ep.graph)
            remove_self_clone(ep.graph)
            ep.graph.eliminate_dead_code()

    return ep
