# [MODULE DOC]
# Defines _RemoveRuntimeAssertionsPass, which deletes unused runtime assertion
# nodes from an exported graph module. No import-time side effects.
import torch
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class _RemoveRuntimeAssertionsPass(PassBase):
    # [CLASS DOC] HELPER: Stateless pass that strips runtime assertion nodes
    # previously inserted by the inline-constraints assertion pass.
    """
    Remove runtime assertions inserted by the
    _AddRuntimeAssertionsForInlineConstraintsPass.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        # [FUNCTION DOC] Walk every GraphModule (including submodules) and erase assertion
        # nodes (async/scalar asserts, range constraints, tensor metadata asserts)
        # that have no users. Mutates graphs in place; no DCE or recompile is done.
        # Returns PassResult with the same module and whether anything was erased.
        modified = False
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if node.target in [
                    torch.ops.aten._assert_async.msg,
                    torch.ops.aten._assert_scalar.default,
                    torch.ops.aten.sym_constrain_range_for_size.default,
                    torch.ops.aten.sym_constrain_range.default,
                    torch.ops.aten._assert_tensor_metadata.default,
                ]:
                    assert_async_node = node
                    if len(assert_async_node.users) > 0:
                        continue
                    module.graph.erase_node(assert_async_node)
                    # the upstream scalar_tensor <- {le, ge} <- sym_size
                    # linear chain of nodes of nodes is removed by the
                    # downstream dead code elimination
                    modified = True

        # We don't necessarily want to run DCE here because it could affect
        # nodes that are in the module_call_graph attribute of the exported
        # program. We will leave it to the pass caller to call DCE.
        return PassResult(graph_module, modified)
