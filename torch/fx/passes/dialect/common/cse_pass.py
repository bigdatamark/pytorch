# [MODULE DOC] Common subexpression elimination (CSE) pass for FX graphs that is dialect agnostic and
# based purely on node connectivity. Defines the sets of banned (random and in-place) aten ops, a helper
# returning their union, and the CSEPass class. Import-time side effect: resolves the aten op namespace.
from typing import Any

import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.node import Target
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.utils._pytree import tree_flatten


aten = torch.ops.aten


# stateful ops are banned from CSE
rand_ops = {
    aten.dropout,
    aten._fused_dropout,
    aten._standard_gamma,
    aten.bernoulli,
    aten.multinomial,
    aten.native_dropout,
    aten.normal,
    aten.poisson,
    aten.binomial,
    aten.rrelu,
    aten.rand_like,
    aten.rand,
    aten.randint,
    aten.randn,
    aten.randperm,
}

inplace_ops = {
    aten.add_,
    aten.sub_,
    aten.mul_,
    aten.div_,
    aten.pow_,
    aten.lerp_,
    aten.relu_,
    aten.sigmoid_,
    aten.tanh_,
}


@torch.fx._compatibility.compatibility(is_backward_compatible=False)
def get_CSE_banned_ops() -> set[torch._ops.OpOverloadPacket]:
    # [FUNCTION DOC] Returns a new set holding every op that CSE must not merge: the union of the
    # random (stateful) ops and the in-place ops defined in this module. No arguments, no side effects.
    return rand_ops.union(inplace_ops)


@torch.fx._compatibility.compatibility(is_backward_compatible=False)
class CSEPass(PassBase):
    # [CLASS DOC] Pass that removes duplicate call_function nodes (same target and same substituted
    # arguments) from a graph module, producing a new module. State: banned_ops, the set of op
    # packets that are never deduplicated. Only safe for functional graphs.
    def __init__(
        self, banned_ops: set[torch._ops.OpOverloadPacket] | None = None
    ) -> None:
        # [FUNCTION DOC] Stores the banned op set (an empty set when None is given) and runs the base
        # initializer. Note the default does not include the module-level random or in-place op sets;
        # callers wanting those must pass get_CSE_banned_ops explicitly.
        """
        This version of CSE Pass aims to be dialect agnostic, and it's implemented purely based on the connectivity between fx.Node.

        For functional dialects, user would only need to specify the random ops in ban list.

        Warning: CSE Pass cannot be safely applied on a FX graph in non-functional dialects.
        If your dialect contains stateful operators, please customized the banned_ops.

        """
        if banned_ops is None:
            banned_ops = set()
        self.banned_ops = banned_ops
        super().__init__()

    def call(self, graph_module: GraphModule) -> PassResult:
        # [FUNCTION DOC] Builds a new graph by copying nodes in order. Placeholder, output, get_attr and
        # banned-op nodes are copied unchanged. Other nodes have their arguments remapped to the new
        # graph; a node whose target and remapped args/kwargs match an earlier node (compared via a hash
        # then full equality of a token) is dropped and uses are redirected to the earlier node.
        # Returns a PassResult with the original module and False if nothing was removed, else a new
        # GraphModule and True. The input module is not mutated.
        # KNOWN ISSUE: the docstring says the result is always a new copy, but the unmodified case returns
        # the input module itself. Hash collisions with a differing token are copied but not recorded.
        """
        Return a new copy of torch.fx.GraphModule with CSE applied to the input graph

        Example usage:

        from torch.fx.experimental.proxy_tensor import make_fx
        def f(a):
            b = a * a
            c = a * a
            return b+c

        p = CSEPass()
        traced_graph = make_fx(f)(torch.tensor(1))
        print(traced_graph)
        result = p(traced_graph)
        print(result.graph_module)
        """

        def get_aten_target(node: Node) -> Target:
            # [FUNCTION DOC] HELPER: Returns the overload packet of the node's target if it has one,
            # otherwise the target itself; used to match against the banned op set.
            if hasattr(node.target, "overloadpacket"):
                return node.target.overloadpacket
            return node.target

        modified = False
        new_graph = Graph()
        env: dict[
            Node, Node
        ] = {}  # map from node in the old graph to node in the new graph
        hash_env: dict[
            tuple[Target, int], Node
        ] = {}  # map from hash to a node in the new graph
        token_map: dict[
            tuple[Target, int], dict[str, Any]
        ] = {}  # map from hash to token
        for n in graph_module.graph.nodes:
            # The placeholder, output, and get_attr nodes are copied to the new graph without change
            # do not CSE away random operations
            if (
                n.op == "placeholder"
                or n.op == "output"
                or n.op == "get_attr"
                or get_aten_target(n) in self.banned_ops
            ):
                new_node = new_graph.node_copy(n, lambda x: env[x])
                env[n] = new_node
            else:  # n.op == 'call_function', should never see n.op == 'call_module' or 'call_method'
                # substitute args and kwargs members to their mapping in env if exists
                # specs can be used to reconstruct nested list/dictionaries
                def substitute(arg_list: Any) -> tuple[tuple[Any, ...], Any]:
                    # [FUNCTION DOC] HELPER: Flattens a nested args/kwargs structure, replaces every
                    # Node leaf found in the old-to-new mapping by its new node, and returns the flat
                    # tuple together with the structure spec needed to rebuild it.
                    arg_list, spec = tree_flatten(arg_list)
                    for i in range(len(arg_list)):
                        v = arg_list[i]
                        if isinstance(v, Node) and v in env:
                            arg_list[i] = env[v]
                    return tuple(arg_list), spec

                args, args_spec = substitute(n.args)
                kwargs, kwargs_spec = substitute(n.kwargs)

                # each token corresponds to a unique node
                # nodes with the same token can be substituted
                token = {
                    "target": n.target,
                    "args": args,
                    "args_spec": args_spec,
                    "kwargs": kwargs,
                    "kwargs_spec": kwargs_spec,
                }

                # hash substituted args to a number, do not hash specs because specs are not hashable
                hash_arg = hash((args, kwargs))
                hash_val = (n.target, hash_arg)

                # check if a node has a substitute and can be eliminated
                hash_val_in_hash_env = hash_val in hash_env
                if hash_val_in_hash_env and token_map[hash_val] == token:
                    modified = True  # substitution happens and the graph is modified
                    env[n] = hash_env[hash_val]
                    continue

                new_node = new_graph.node_copy(n, lambda x: env[x])
                env[n] = new_node
                if not hash_val_in_hash_env:
                    hash_env[hash_val] = new_node
                    token_map[hash_val] = token

        if not modified:
            return PassResult(graph_module, False)
        csed_gm = GraphModule(graph_module, new_graph)
        return PassResult(csed_gm, True)
