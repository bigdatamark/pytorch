# [MODULE DOC]
# Constant folding utilities for FX graphs: ConstantFolder interpreter, constant_fold,
# constant_graph_tag and run_and_get_constant_graph, plus the MODULE_TYPE tag constants
# used to split a module into main and constant subgraphs. No import-time side effects
# beyond resolving torch.ops.aten.
# mypy: allow-untyped-defs
import collections
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch
import torch.utils._pytree as pytree


aten = torch.ops.aten

# We would like to split modules into two subgraphs for runtime weight updates to work correctly.
# The use case and more information could be found at:
# https://docs.google.com/document/d/1inZC-8KarJ6gKB7G9egmYLx1V_dKX_apxon0w4zPC0Q/edit?usp=sharing
META_TAG = "MODULE_TYPE"
MODULE_TAG = "_MAIN_MODULE"
CONST_MODULE_TAG = "_CONST_MODULE"


def replace_node_with_constant(gm, node, constant, name=None):
    # [FUNCTION DOC] Replace node in gm's graph by a get_attr node referencing constant.
    # The attribute name is name if given, else the first unused _frozen_param<N>
    # name (tracked via a _frozen_param_count counter attribute on gm). The new node
    # takes over all uses and the old node's meta, and the old node is erased. The
    # constant is registered on gm as a buffer and attribute. Mutates gm; returns None.
    g = gm.graph

    if name:
        qualname = name
    else:
        if not hasattr(gm, "_frozen_param_count"):
            gm._frozen_param_count = 0
        i = gm._frozen_param_count

        while True:
            qualname = f"_frozen_param{i}"
            if not hasattr(gm, qualname):
                break
            i += 1

        gm._frozen_param_count = i + 1

    with g.inserting_before(node):
        new_input_node = g.create_node("get_attr", qualname, (), {})
        node.replace_all_uses_with(new_input_node)
        new_input_node.meta.update(node.meta)
        g.erase_node(node)

    # needed to suppress `does not reference an nn.Module, nn.Parameter, or buffer` warning
    gm.register_buffer(qualname, constant)
    setattr(gm, qualname, constant)


class ConstantFolder(torch.fx.Interpreter):
    # [CLASS DOC] Interpreter that evaluates a graph with placeholders as unknown and
    # records which nodes produce concrete tensors, i.e. are constant foldable.
    # State: node_replacements (node to computed tensor), replaced_uses (count of how
    # many folded consumers use each node), unknown_value (sentinel for non-constant
    # values), skip_constructors (leave nodes with no tensor inputs unfolded) and
    # user_to_last_uses (for freeing env values early). Key methods: run, run_node,
    # is_impure, insertable_tensor_check, add_node_replacement (the last two are hooks
    # for subclasses).
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        skip_constructors: bool = False,
    ):
        # [FUNCTION DOC] Set up the interpreter over gm, initialize replacement tracking,
        # the unknown sentinel and the skip_constructors flag, and precompute last-use
        # info so values are freed once only the output consumes them.
        super().__init__(gm)
        self.node_replacements: dict[torch.fx.Node, Any] = {}
        self.replaced_uses: dict[torch.fx.Node, int] = collections.Counter()
        self.unknown_value = object()
        self.skip_constructors: bool = skip_constructors

        # overwrite this to deallocate env values if their only remaining use
        # is the output
        self.user_to_last_uses = self.node_to_last_non_output_use()

    def is_impure(self, node: torch.fx.Node) -> bool:
        # [FUNCTION DOC] Return True for nodes that must not be folded: an int8 get_attr
        # converted to bfloat16, and dequantize ops (so quantize-dequantize pairs stay
        # fusable). Otherwise False.
        if (
            node.target is torch.ops.prims.convert_element_type.default
            and node.args[0].op == "get_attr"  # type: ignore[union-attr]
            and node.args[0].meta["val"].dtype == torch.int8  # type: ignore[union-attr]
            and node.args[1] == torch.bfloat16
        ):
            # For int8_weight -> dq -> bf16_weight
            return True
        if node.target in [
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
            torch.ops.pt2e_quant.dequantize_affine,
        ]:
            # For the pattern fp32_weight -> q -> dq
            # We only folding fp32_weight -> q
            # int8_weight and leave dq in graph to be fused
            return True
        return False

    def node_to_last_non_output_use(self):
        # [FUNCTION DOC] Walk the graph in reverse and return a mapping from each node to
        # the list of input nodes for which it is the last (non-output) user, plus the
        # node itself if its only user is the output node. Used to free interpreter env
        # entries. Does not mutate the graph.
        last_non_output_use = collections.defaultdict(list)
        seen_uses = set()
        output_node = next(iter(reversed(self.module.graph.nodes)))  # type: ignore[arg-type, union-attr]

        for node in reversed(self.module.graph.nodes):  # type: ignore[arg-type, union-attr]
            if node.target == "output":
                continue

            def add_use(inp):
                # [FUNCTION DOC] HELPER: Record inp as last-used by the current node unless
                # a later node already used it.
                if inp in seen_uses:
                    return

                seen_uses.add(inp)
                last_non_output_use[node].append(inp)

            # In-place is fine since we don't mutate
            pytree.tree_map_only_(torch.fx.Node, add_use, (node.args, node.kwargs))

            # if this node is only used in output, we want to gc it right away
            if len(node.users) == 1 and output_node in node.users:
                last_non_output_use[node].append(node)

        return last_non_output_use

    def run_node(self, node):
        # [FUNCTION DOC] Interpret one node. The output node re-registers its arguments as
        # unknown before delegating. Any node with an unknown input, plus efficient zero
        # tensor, triton kernel wrapper, constructor-like nodes (when skip_constructors),
        # and seeded nondeterministic ops evaluate to the unknown sentinel. Otherwise the
        # node is run; if it yields a non-meta tensor (not get_attr) that passes the
        # insertable check and is not impure, it is recorded as a replacement, its input
        # nodes get their replaced use counts bumped, and earlier replacements whose
        # uses are all folded away are dropped. Returns the computed value or sentinel.
        if node.target == "output":
            # because we remove nodes from env on last non output use,
            # re-define them now or we'll get error in interpreter
            def set_env(arg):
                # [FUNCTION DOC] HELPER: Mark arg as unknown in the interpreter env.
                self.env[arg] = self.unknown_value

            # In-place is fine since we don't mutate
            pytree.tree_map_only_(torch.fx.Node, set_env, node.args)
            return super().run_node(node)

        args, kwargs = self.fetch_args_kwargs_from_env(node)
        flattened_inputs = pytree.arg_tree_leaves(*args, **kwargs)

        # We need to do this weird thing because in cases where flattened_inputs
        # contains a ScriptObject, equality checking results in a type error if
        # the types are different.
        if any(
            type(self.unknown_value) is type(input_) and self.unknown_value == input_
            for input_ in flattened_inputs
        ):
            return self.unknown_value

        # TODO - fix errors with this
        if (
            node.op == "call_function"
            and node.target is aten._efficientzerotensor.default
        ):
            return self.unknown_value

        # TODO - constant folding triton kernel returns the inputs -- fix this
        if (
            node.op == "call_function"
            and node.name == "triton_kernel_wrapper_functional_proxy"
        ):
            return self.unknown_value

        # skip constructors, since inductor generates optimal code for them already
        # and turning into tensor would result in an additional global memory read
        # TODO - more complicated strategy
        if (
            self.skip_constructors
            and node.op != "get_attr"
            and not any(isinstance(e, torch.Tensor) for e in flattened_inputs)
        ):
            return self.unknown_value

        # All mutations should either be removed or on inputs which we did not make constant
        if (
            isinstance(node.target, torch._ops.OpOverload)
            and torch.Tag.nondeterministic_seeded in node.target.tags
        ):
            return self.unknown_value

        out = super().run_node(node)

        if node.op != "get_attr" and isinstance(out, torch.Tensor):
            if out.device.type == "meta":
                return out

            if not self.insertable_tensor_check(out):
                return out

            if self.is_impure(node):
                return self.unknown_value

            self.add_node_replacement(node, out)

            flattened_node_inps = pytree.arg_tree_leaves(*node.args, **node.kwargs)

            for n in flattened_node_inps:
                if not isinstance(n, torch.fx.Node):
                    continue

                self.replaced_uses[n] += 1

            for to_delete in self.user_to_last_uses.get(node, []):
                if self.replaced_uses[to_delete] == len(to_delete.users):
                    self.node_replacements.pop(to_delete, None)

        return out

    def insertable_tensor_check(self, tensor: torch.Tensor) -> bool:
        # [FUNCTION DOC] Hook: whether a folded tensor may be inserted; always True here.
        return True

    def add_node_replacement(self, node: torch.fx.Node, tensor: torch.Tensor) -> None:
        # [FUNCTION DOC] Hook: record tensor as the constant replacing node.
        self.node_replacements[node] = tensor

    def run(self):  # type: ignore[override]
        # [FUNCTION DOC] Run the interpreter with all placeholders unknown and return the
        # graph result (output values are unknown or concrete).
        env = {}
        for n in self.module.graph.find_nodes(op="placeholder"):  # type: ignore[operator, union-attr]
            env[n] = self.unknown_value
        return super().run(initial_env=env)


def constant_fold(
    gm: torch.fx.GraphModule,
    constraint_fn: Callable[[torch.fx.Node], bool] | None = None,
):
    # [FUNCTION DOC] Fold constants in gm in place. Runs ConstantFolder (skipping
    # constructors) with torch modes disabled, replaces each foldable node (optionally
    # filtered by constraint_fn returning True) with a frozen get_attr constant, deletes
    # now-unused get_attr nodes and their attributes, then eliminates dead code, lints
    # and recompiles. Mutates gm; returns None.
    with torch.utils._python_dispatch._disable_current_modes():
        cf = ConstantFolder(gm, skip_constructors=True)
        cf.run()

        for node, constant in cf.node_replacements.items():
            if constraint_fn is not None and not constraint_fn(node):
                continue
            replace_node_with_constant(gm, node, constant)

        erased_params = []
        # Get all attr users by looking up the graph instead from node.users, because in this case
        # _tensor_constant0 and _tensor_constant0_1 are actually referring to the same tensor.

        #     opcode         name                 target            args                         kwargs
        # -------------  -------------------  ----------------  ---------------------------  --------
        # placeholder    arg0_1               arg0              ()                           {}
        # get_attr       _tensor_constant0    state             ()                           {}
        # call_function  add                  aten.add.Tensor   (arg0_1, _tensor_constant0)  {}
        # get_attr       _tensor_constant0_1  state             ()                           {}
        # call_function  add_                 aten.add_.Tensor  (_tensor_constant0_1, 1)     {}
        # output         output               output            ([add],)                     {}

        get_attr_node_users = defaultdict(list)
        for node in gm.graph.nodes:
            if node.op == "get_attr":
                get_attr_node_users[node.target].extend(node.users.keys())
        for node in gm.graph.find_nodes(op="get_attr"):
            if node.op == "get_attr" and len(get_attr_node_users[node.target]) == 0:
                if hasattr(gm, node.target):
                    delattr(gm, node.target)
                erased_params.append(node)
        for node in erased_params:
            gm.graph.erase_node(node)

        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()


def constant_graph_tag(gm: torch.fx.GraphModule) -> None:
    # [FUNCTION DOC] Run constant folding analysis on gm and tag every node's meta
    # under META_TAG as CONST_MODULE_TAG (get_attr, folded nodes, or nodes feeding a
    # fold) or MODULE_TAG otherwise. Does not change the graph structure.
    with torch.utils._python_dispatch._disable_current_modes():
        cf = ConstantFolder(gm, skip_constructors=True)
        cf.run()

        for node in gm.graph.nodes:
            if (
                node.op == "get_attr"
                or node in cf.node_replacements
                or node in cf.replaced_uses
            ):
                node.meta[META_TAG] = CONST_MODULE_TAG
            else:
                node.meta[META_TAG] = MODULE_TAG


def run_and_get_constant_graph(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # [FUNCTION DOC] Return a new GraphModule containing only the constant-foldable part
    # of gm. Tags nodes, retags get_attr nodes not consumed by any const-tagged user as
    # main, copies const-tagged nodes into a new graph, and outputs a tuple of those
    # nodes that are used by main-tagged users. Tags in gm meta are mutated.
    """
    Construct a GraphModule which corresponds to the part which could be
    constant folded in provided gm.
    """

    constant_graph_tag(gm)
    # We rewrite the tags, if it's a constant being directly consumed, without
    # any folding opportunity, we keep it in main gm.
    for node in gm.graph.find_nodes(op="get_attr"):
        used_to_fold = False
        for u in node.users:
            if u.meta[META_TAG] == CONST_MODULE_TAG:
                used_to_fold = True
                break
        if not used_to_fold:
            node.meta[META_TAG] = MODULE_TAG

    new_graph = torch.fx.Graph()

    node_remapping: dict[torch.fx.Node, torch.fx.Node] = {}
    output_nodes = []
    for node in gm.graph.nodes:
        if node.meta[META_TAG] == MODULE_TAG:
            continue

        new_node = new_graph.node_copy(node, lambda x: node_remapping[x])
        node_remapping[node] = new_node

        for user in node.users:
            if user.meta[META_TAG] == MODULE_TAG:
                output_nodes.append(new_node)
                break

    new_graph.output(tuple(output_nodes))
    new_graph.lint()
    new_gm = torch.fx.GraphModule(gm, new_graph)

    return new_gm
