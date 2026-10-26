# mypy: allow-untyped-defs
# [MODULE DOC] Group and batch fusion passes over FX graphs (pre-grad torch level and post-grad aten level).
# Each fusion rule is a class with match (return a hashable group key for a node or None) and fuse (merge a
# list of independent nodes sharing a key, typically by stacking inputs, running one batched op and
# unbinding/selecting the results). Contents: rule base classes, registries PRE_GRAD_FUSIONS and
# POST_GRAD_FUSIONS filled by the register_fusion decorator, linear / layernorm / pointwise / math rules,
# candidate search (BFS) and greedy independent-subset selection, and the driver group_batch_fusion_passes.
# Import-time side effects: tries to import the fbgemm inductor lowerings to set has_fbgemm; registers all
# rules in the registries; defines global search option defaults.
import collections
import logging
import operator
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from typing import Any

import torch
from torch._dynamo.utils import counters, is_node_meta_valid
from torch._logging import trace_structured
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.symbolic_shapes import free_symbols
from torch.fx.passes.graph_transform_observer import GraphTransformObserver
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..pattern_matcher import (
    CallFunctionVarArgs,
    CallMethodVarArgs,
    get_arg_value,
    MatchResult,
    stable_topological_sort,
)
from ..utils import OPTIMUS_EXCLUDE_POST_GRAD


try:
    # importing this will register fbgemm lowerings for inductor
    import deeplearning.fbgemm.fbgemm_gpu.fb.inductor_lowerings  # noqa: F401

    has_fbgemm = True
except Exception:
    has_fbgemm = False

aten = torch.ops.aten

log = logging.getLogger(__name__)

DEFAULT_BETA = 1
DEFAULT_ALPHA = 1

MIN_FUSE_SET_SIZE = 5
MAX_FUSE_SET_SIZE = 300
MAX_FUSE_SEARCH_DEPTH = 5
# The maximum tensor size that can go into the fusion group
MAX_FUSE_TENSOR_SIZE_GROUP_LINEAR = 4096
# Whether we only fuse nodes with same parent node
FUSE_NODES_WITH_SAME_PARENT = False
# Whether we enable the add broadcast in batch linear
SHAPE_BROADCAST_BATCH_LINEAR = False
# Whether we enable the fuse nodes with same users
Fuse_NODES_WITH_SAME_USERS = False

# exclude these nodes from BFS
# excluding get item improves optimizer compilation time by 60s
SEARCH_EXCLUSIONS = OrderedSet([operator.getitem])


default_graph_search_options = {
    "min_fuse_set_size": MIN_FUSE_SET_SIZE,
    "max_fuse_set_size": MAX_FUSE_SET_SIZE,
    "max_fuse_search_depth": MAX_FUSE_SEARCH_DEPTH,
    "max_fuse_tensor_size_group_linear": MAX_FUSE_TENSOR_SIZE_GROUP_LINEAR,
    "fuse_nodes_with_same_parent": FUSE_NODES_WITH_SAME_PARENT,
    "shape_broadcast_batch_linear": SHAPE_BROADCAST_BATCH_LINEAR,
    "fuse_nodes_with_same_users": Fuse_NODES_WITH_SAME_USERS,
}

graph_search_options = default_graph_search_options


def update_stack_example_value(node, metadata, dim=0, op=torch.stack):
    # [FUNCTION DOC] Set the example_value meta of a stack or unbind node, computed by running torch.stack or torch.unbind
    # on the given tensor metadata and dim. Does nothing for a None node, a node without meta, or any other op.
    """
    Update the example value of the node in the graph to enable followup split cat opt.
    """
    if node is not None and hasattr(node, "meta"):
        if op is torch.stack:
            example_value = torch.stack(metadata, dim=dim)
        elif op is torch.unbind:
            example_value = torch.unbind(metadata, dim=dim)  # type: ignore[assignment]
        else:
            return
        node.meta["example_value"] = example_value


def update_pointwise_example_value(pointwise_node, input, other, op):
    # [FUNCTION DOC] Set the example_value meta of a pointwise node (add or mul only) by computing the op on the given
    # tensors. Does nothing for a None node, a node without meta, or another op.
    """
    Update the example value of the add node in the graph to enable followup split cat opt.
    """
    if pointwise_node is not None and hasattr(pointwise_node, "meta"):
        if op is torch.add:
            example_value = torch.add(input, other)
        elif op is torch.mul:
            example_value = torch.mul(input, other)
        else:
            return
        pointwise_node.meta["example_value"] = example_value


class GroupBatchFusionBase:
    # [CLASS DOC] Base class of all fusion rules. Holds graph_search_options (from the graph_search_options keyword or
    # the module defaults). Subclasses implement match(node), returning a group key or None, and
    # fuse(graph, subset), rewriting a list of independent nodes sharing a key.
    def __init__(self, **kwargs) -> None:
        # [FUNCTION DOC] Pop graph_search_options from the keyword arguments (defaulting to the module defaults) and store it.
        self.graph_search_options = kwargs.pop(
            "graph_search_options", default_graph_search_options
        )

    def match(self, node):
        # [FUNCTION DOC] Abstract: return a group key for the node or None. Raises NotImplementedError in the base class.
        raise NotImplementedError("match called on base")

    def fuse(self, graph, subset):
        # [FUNCTION DOC] Abstract: fuse the nodes in subset in graph. Raises NotImplementedError in the base class.
        raise NotImplementedError("fuse called on base")


PRE_GRAD_FUSIONS: dict[str, GroupBatchFusionBase] = {}
POST_GRAD_FUSIONS: dict[str, GroupBatchFusionBase] = {}


def register_fusion(name: str, pre_grad=True):
    # [FUNCTION DOC] Class decorator factory: registers a fusion class under name in the pre-grad or post-grad registry
    # (global dictionaries, mutated at import time) and returns the class unchanged.
    def decorator(fusion_cls: GroupBatchFusionBase):
        # [FUNCTION DOC] HELPER: Do the registration and return the class.
        if pre_grad:
            PRE_GRAD_FUSIONS[name] = fusion_cls
        else:
            POST_GRAD_FUSIONS[name] = fusion_cls
        return fusion_cls

    return decorator


def list_group_batch_fusions(pre_grad=True) -> list[str]:
    # [FUNCTION DOC] Return the list of registered fusion names for pre-grad or post-grad.
    if pre_grad:
        return list(PRE_GRAD_FUSIONS.keys())
    else:
        return list(POST_GRAD_FUSIONS.keys())


def decompose_stack(graph: torch.fx.GraphModule, input_tensors: list[Any]) -> Any:
    # [FUNCTION DOC] Emulate stacking tensors in post-grad (aten) form: unsqueeze each input at dim 0 then concatenate along
    # dim 0, setting val meta on each created node from the inputs' val meta. Returns the concatenated node.
    # Inserts nodes at the current graph insertion point.
    unsqueezed_inputs = []
    unsqueezed_inputs_meta = []
    for input_tensor in input_tensors:
        unsqueezed_input = graph.call_function(  # type: ignore[operator]
            aten.unsqueeze, args=(input_tensor,), kwargs={"dim": 0}
        )
        unsqueezed_inputs.append(unsqueezed_input)
        unsqueezed_input.meta["val"] = aten.unsqueeze(input_tensor.meta["val"], dim=0)  # type: ignore[assignment]
        unsqueezed_inputs_meta.append(unsqueezed_input.meta["val"])
    stacked_inputs = graph.call_function(  # type: ignore[operator]
        aten.cat, args=(unsqueezed_inputs,), kwargs={"dim": 0}
    )
    stacked_inputs.meta["val"] = aten.cat(unsqueezed_inputs_meta, dim=0)  # type: ignore[assignment]
    return stacked_inputs


class GroupFusion(GroupBatchFusionBase):
    # [CLASS DOC] Marker base for rules that fuse ops of arbitrary shapes in a group way (for example with fbgemm gmm).
    """
    Fuse ops in a group way, e.g, fuse mm/addmm of arbitrary input shapes with fbgemm.gmm.
    """


class BatchFusion(GroupBatchFusionBase):
    # [CLASS DOC] Marker base for rules that fuse ops of identical shapes in a batch way (for example with bmm).
    """
    Fuse ops in a batch way, e.g, fuse mm/addmm of same input shapes with bmm.
    """


class BatchPointwiseOpsFusionFactory(BatchFusion):
    # [CLASS DOC] Base for batched pointwise-op rules; stores the op being batched in the op attribute.
    def __init__(self, op, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Store the op and forward options to the base class.
        super().__init__(**kwargs)
        self.op = op


@register_fusion("batch_linear_post_grad", pre_grad=False)
class PostGradBatchLinearFusion(BatchFusion):
    # [CLASS DOC] Post-grad (aten) rule batching mm and addmm nodes with identical 2D shapes into a single bmm over
    # stacked operands, then selecting each result (adding the bias after, optionally broadcast). Group key
    # contains m, k, n, whether a bias exists, and optionally the users' targets. Registered as
    # batch_linear_post_grad (post-grad).
    """
    Fuse ops in a batch way in post grad (aten level).
    """

    def _addmm_node_can_be_fused(self, node: torch.fx.Node) -> bool:
        # [FUNCTION DOC] HELPER: True if the addmm has default beta and alpha.
        # pyre-fixme[7]: Incompatible return type
        return (
            # pyrefly: ignore [bad-return]
            node.kwargs.get("beta", DEFAULT_BETA) == DEFAULT_BETA
            and node.kwargs.get("alpha", DEFAULT_ALPHA) == DEFAULT_ALPHA  # type: ignore[return-value]
        )

    def _is_input_2d(self, input: torch.fx.Node) -> bool:
        # [FUNCTION DOC] HELPER: True if the node's val is a rank-2 tensor with static int dims.
        input_shapes = input.meta["val"].shape
        return (
            len(input_shapes) == 2
            and isinstance(input_shapes[0], int)
            and isinstance(input_shapes[1], int)
        )

    def match(self, node: torch.fx.Node) -> tuple[str, int, int, int, bool, str] | None:
        # [FUNCTION DOC] Return a group key for mm or fusable addmm nodes with static 2D input and weight, or None.
        if CallFunctionVarArgs(aten.mm).match(node):
            input_m, weight_m = node.args
            bias_m = None

        elif CallFunctionVarArgs(aten.addmm.default).match(
            node
        ) and self._addmm_node_can_be_fused(node):
            bias_m, input_m, weight_m = node.args
        else:
            return None
        # get the user of the node
        if self.graph_search_options.get("fuse_nodes_with_same_users", False):
            users = [user.target for user in node.users]
        else:
            users = ""  # type: ignore[assignment]
        # only handle the cases where inputs are 2D tensors
        if not self._is_input_2d(input_m) or not self._is_input_2d(weight_m):  # type: ignore[arg-type]
            return None
        m, k = input_m.meta["val"].shape  # type: ignore[union-attr]
        n = weight_m.meta["val"].shape[1]  # type: ignore[union-attr]
        batch_key = ("batch_linear_post_grad", m, k, n, bias_m is not None, str(users))
        return batch_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        # [FUNCTION DOC] Fuse mm/addmm nodes: stack inputs and weights, run one bmm inserted before the last node, then for each
        # original node insert a select of its result, plus the bias add (with optional broadcast per the search option)
        # when a bias exists; reroute uses, copy meta, erase originals; increments the batch_linear_post_grad counter.
        batch_inputs = []
        batch_weights = []
        batch_biases = []
        batch_nodes = []
        batch_inputs_meta = []
        batch_weights_meta = []
        batch_biases_meta = []

        for node in subset:
            if CallFunctionVarArgs(aten.addmm.default).match(node):
                bias, input, weight = node.args
            elif CallFunctionVarArgs(aten.mm.default).match(node):
                input, weight = node.args
                bias = None
            batch_nodes.append(node)
            batch_inputs.append(input)  # type: ignore[possibly-undefined]
            batch_weights.append(weight)  # type: ignore[possibly-undefined]
            batch_biases.append(bias)  # type: ignore[possibly-undefined]
            batch_inputs_meta.append(input.meta)  # type: ignore[possibly-undefined, union-attr]
            batch_weights_meta.append(weight.meta)  # type: ignore[possibly-undefined, union-attr]
            if bias is not None:  # type: ignore[possibly-undefined]
                batch_biases_meta.append(bias.meta)  # type: ignore[possibly-undefined, union-attr]
            else:
                batch_biases_meta.append(None)

        with graph.inserting_before(subset[-1]):  # type: ignore[operator]
            fused_inputs = decompose_stack(graph, batch_inputs)
            fused_weights = decompose_stack(graph, batch_weights)
            fused_inputs_meta_val = torch.stack(
                [input["val"] for input in batch_inputs_meta]
            )
            fused_weights_meta_val = torch.stack(
                [weight["val"] for weight in batch_weights_meta]
            )
            fused_bmm = graph.call_function(  # type: ignore[operator]
                aten.bmm,
                args=(fused_inputs, fused_weights),
            )
            fused_bmm.meta["val"] = aten.bmm(
                fused_inputs_meta_val, fused_weights_meta_val
            )
        for i, original_mm in enumerate(batch_nodes):
            has_bias = False
            with graph.inserting_after(fused_bmm):  # type: ignore[operator]
                new_mm = graph.call_function(aten.select, args=((fused_bmm, 0, i)))  # type: ignore[operator]
                new_mm.meta["val"] = aten.select(fused_bmm.meta["val"], 0, i)
                if batch_biases[i]:
                    has_bias = True
                    # broadcast the bias to the same shape as the mm output
                    if self.graph_search_options.get(
                        "shape_broadcast_batch_linear", False
                    ):
                        broadcast_shape = torch.broadcast_shapes(
                            batch_biases_meta[i]["val"].shape, new_mm.meta["val"].shape
                        )
                        broadcast_bias = graph.call_function(  # type: ignore[operator]
                            aten.broadcast_to.default,
                            args=(batch_biases[i],),
                            kwargs={"size": broadcast_shape},
                        )
                        broadcast_bias.meta["val"] = aten.broadcast_to(
                            batch_biases_meta[i]["val"], broadcast_shape
                        )  # type: ignore[assignment]
                        new_bias_add = graph.call_function(  # type: ignore[operator]
                            aten.add.Tensor, args=((broadcast_bias, new_mm))
                        )
                        new_bias_add.meta["val"] = aten.add.Tensor(
                            broadcast_bias.meta["val"], new_mm.meta["val"]
                        )
                    else:
                        new_bias_add = graph.call_function(  # type: ignore[operator]
                            aten.add, args=((batch_biases[i], new_mm))
                        )
                        new_bias_add.meta["val"] = aten.add.Tensor(
                            batch_biases_meta[i]["val"], new_mm.meta["val"]
                        )
            new_mm_cont = new_bias_add if has_bias else new_mm  # type: ignore[possibly-undefined]
            original_mm.replace_all_uses_with(new_mm_cont)
            new_mm_cont.meta.update(original_mm.meta)
            graph.erase_node(original_mm)  # type: ignore[operator]
        counters["inductor"]["batch_linear_post_grad"] += 1


@register_fusion("group_linear", pre_grad=False)
class GroupLinearFusion(GroupFusion):
    # [CLASS DOC] Post-grad rule grouping mm/addmm nodes (even dims, bounded size, 2D) for the fbgemm grouped matmul op.
    # Registered as group_linear (post-grad).
    def _addmm_node_can_be_fused(self, node: torch.fx.Node):
        # [FUNCTION DOC] HELPER: True for an addmm with default alpha and beta, 2D operands, even dims and dims no larger than
        # the configured group-linear maximum.
        input_shape = node.args[1].meta["val"].shape  # type: ignore[union-attr]
        weight_shape = node.args[2].meta["val"].shape  # type: ignore[union-attr]
        return (
            node.kwargs.get("beta", DEFAULT_BETA) == DEFAULT_BETA
            and node.kwargs.get("alpha", DEFAULT_ALPHA) == DEFAULT_ALPHA
            and len(input_shape) == 2
            and len(weight_shape) == 2
            and all(x % 2 == 0 for x in input_shape + weight_shape)
            and all(
                shape <= self.graph_search_options["max_fuse_tensor_size_group_linear"]
                for shape in input_shape + weight_shape
            )
        )

    def _mm_node_can_be_fused(self, node: torch.fx.Node):
        # [FUNCTION DOC] HELPER: True for an mm with 2D operands, even dims and dims no larger than the configured maximum.
        input_shape = node.args[0].meta["val"].shape  # type: ignore[union-attr]
        weight_shape = node.args[1].meta["val"].shape  # type: ignore[union-attr]
        return (
            len(input_shape) == 2
            and len(weight_shape) == 2
            and all(x % 2 == 0 for x in input_shape + weight_shape)
            and all(
                shape <= self.graph_search_options["max_fuse_tensor_size_group_linear"]
                for shape in input_shape + weight_shape
            )
        )

    def match(self, node: torch.fx.Node) -> tuple[str, bool] | None:
        # [FUNCTION DOC] Return a group key for fusable mm or addmm nodes (second element states whether there is no bias), else None.
        if CallFunctionVarArgs(aten.mm.default).match(
            node
        ) and self._mm_node_can_be_fused(node):
            group_key = ("group_linear", True)
        elif CallFunctionVarArgs(aten.addmm.default).match(
            node
        ) and self._addmm_node_can_be_fused(node):
            bias = node.args[0]
            group_key = ("group_linear", bias is None)
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        # [FUNCTION DOC] Replace the nodes with a single fbgemm gmm call (smart_fused) inserted before the first node, taking
        # lists of inputs, weights and biases (biases None if all absent), and a getitem per original node.
        # Raises AssertionError if a node is neither addmm nor mm. Increments the group_linear counter.
        group_inputs = []
        group_weights = []
        group_biases = []
        group_nodes = []
        for node in subset:
            if CallFunctionVarArgs(aten.addmm.default).match(node):
                bias, input, weight = node.args
            else:
                if not CallFunctionVarArgs(aten.mm.default).match(node):
                    raise AssertionError(f"expected aten.mm node, got {node}")
                input, weight = node.args
                bias = None

            group_nodes.append(node)
            group_inputs.append(input)
            group_weights.append(weight)
            group_biases.append(bias)

        if all(bias is None for bias in group_biases):
            group_biases = None  # type: ignore[assignment]

        with graph.inserting_before(subset[0]):  # type: ignore[operator]
            fused_mm = graph.call_function(  # type: ignore[operator]
                torch.ops.fbgemm.gmm.default,
                args=(group_inputs, group_weights, group_biases),
                kwargs={"smart_fused": True},
            )

        for i, original_mm in enumerate(group_nodes):
            with graph.inserting_after(fused_mm):  # type: ignore[operator]
                new_mm = graph.call_function(operator.getitem, args=(fused_mm, i))  # type: ignore[operator]
            original_mm.replace_all_uses_with(new_mm)
            new_mm.meta.update(original_mm.meta)
            graph.erase_node(original_mm)  # type: ignore[operator]
        counters["inductor"]["group_linear"] += 1


class BatchPointwiseMathOpsPostGradFusion(BatchPointwiseOpsFusionFactory):
    # [CLASS DOC] Post-grad rule batching same-shaped pointwise binary math ops (add, sub, mul, div) by stacking both
    # operands, applying the op once and selecting each result. The op attribute holds the aten overload.
    """
    Batch pointwise math operator (e.g., add, mul) in post grad pass.
    """

    def __init__(self, op, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Store the op and forward options to the base class.
        super().__init__(op, **kwargs)
        self.op = op

    def _pointwise_node_can_be_fused(self, node: torch.fx.Node):
        # [FUNCTION DOC] HELPER: True when both arguments are nodes with valid meta holding tensors of equal shape
        # (scalars and symbolic values are rejected).
        # note: we only consider the case where the inputs are tensors
        # for mixed precision training, we need to make sure the inputs
        # of the aten.cat when do the stack should be the same dtype
        # otherwise, the output of the aten.cat may be not the same as
        # its inputs, and cause dtype not same error in mm or addmm
        input, other = node.args
        return (
            input.meta["val"].shape == other.meta["val"].shape  # type: ignore[union-attr]
            # input and other can be scalars, where they have no attribute 'meta'
            if hasattr(input, "meta")
            and hasattr(other, "meta")
            and is_node_meta_valid(input)  # type: ignore[arg-type, union-attr]
            and is_node_meta_valid(other)  # type: ignore[arg-type, union-attr]
            # torch.SymInt or torch.SymFloat object has no attribute 'shape'
            and isinstance(input.meta["val"], torch.Tensor)  # type: ignore[union-attr]
            and isinstance(other.meta["val"], torch.Tensor)  # type: ignore[union-attr]
            else False
        )

    def match(self, node: torch.fx.Node):
        # [FUNCTION DOC] Return a group key (op name, shape, dtypes, alpha, rounding mode, optional shared select parent) or None.
        if CallFunctionVarArgs(self.op).match(
            node
        ) and self._pointwise_node_can_be_fused(node):
            alpha = node.kwargs.get("alpha", DEFAULT_ALPHA)
            rounding_mode = node.kwargs.get("rounding_mode", None)
            input, other = node.args
            shape = list(input.meta["val"].shape)  # type: ignore[union-attr]
            if self.graph_search_options.get("fuse_nodes_with_same_parent", False):
                # only consider the linear case so far
                # pyre-fixme[16]
                if input.target is aten.select or other.target is aten.select:  # type: ignore[union-attr]
                    parent = (
                        # pyre-fixme[16]
                        input.args[0]  # type: ignore[union-attr]
                        # pyre-fixme[16]
                        if input.target is aten.select  # type: ignore[union-attr]
                        else other.args[0]  # type: ignore[union-attr]
                    )
                else:
                    parent = ""
            else:
                parent = ""
            group_key = (
                "batch_aten_" + self.op.__name__.lower().split(".")[0],
                str(shape),
                str(input.meta["val"].dtype),  # type: ignore[union-attr]
                str(other.meta["val"].dtype),  # type: ignore[union-attr]
                str(alpha),
                str(rounding_mode),
                str(parent),
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        # [FUNCTION DOC] Stack both operands, run the op once, then select each result and replace the originals; increments the
        # per-op batch counter.
        # KNOWN ISSUE: alpha is forwarded only for the add overload, and rounding_mode is never forwarded, although
        # both are part of the group key, so sub with a non-default alpha is fused without it.
        batch_inputs, batch_others = [], []
        alpha = subset[0].kwargs.get("alpha", DEFAULT_ALPHA)
        batch_inputs_meta, batch_others_meta = [], []

        for node in subset:
            input, other = node.args
            batch_inputs.append(input)
            batch_others.append(other)
            batch_inputs_meta.append(input.meta)  # type: ignore[possibly-undefined, union-attr]
            batch_others_meta.append(other.meta)  # type: ignore[possibly-undefined, union-attr]

        with graph.inserting_before(subset[0]):  # type: ignore[operator]
            stack_inputs = decompose_stack(graph, batch_inputs)
            stack_others = decompose_stack(graph, batch_others)
            stack_inputs_meta = torch.stack(
                [input["val"] for input in batch_inputs_meta]
            )
            stack_others_meta = torch.stack(
                [other["val"] for other in batch_others_meta]
            )

            batch_op = graph.call_function(  # type: ignore[operator]
                self.op,
                args=(stack_inputs, stack_others),
                kwargs={"alpha": alpha} if self.op == aten.add.Tensor else {},
            )
            batch_op.meta["val"] = self.op(stack_inputs_meta, stack_others_meta)
            for i, original_add in enumerate(subset):
                with graph.inserting_after(batch_op):  # type: ignore[operator]
                    new_add = graph.call_function(  # type: ignore[operator]
                        torch.ops.aten.select, args=((batch_op, 0, i))
                    )
                original_add.replace_all_uses_with(new_add)
                new_add.meta.update(original_add.meta)
                graph.erase_node(original_add)  # type: ignore[operator]
        counters["inductor"][
            "batch_aten_" + self.op.__name__.lower().split(".")[0]
        ] += 1


@register_fusion("batch_linear_lhs")
class BatchLinearLHSFusion(BatchFusion):
    # [CLASS DOC] Pre-grad rule batching linear calls sharing the same input into one mm (or addmm) with concatenated
    # transposed weights, then splitting the output along dim 1. Skips tensor-subclass weights that cannot be
    # concatenated. Registered as batch_linear_lhs (pre-grad).
    """
    Batch linear left-hand side fusion. This pass tries to fuse the following patterns:

        torch.nn.functional.linear(x, w1), linear(x, w2),... * linear(x, wn)
        -> torch.mm(x, torch.cat([w1, w2,... * wn]).transpose(0, 1))

    We have a separate pass to eliminate contiguous transpose in a generic way.
    """

    def match(self, node: torch.fx.Node) -> tuple[str, int | None, Any] | None:
        # [FUNCTION DOC] Return a group key (name, bias rank, shared input node) for fusable linear calls, or None. Probes
        # concatenation on subclass weights and declines on failure.
        if CallFunctionVarArgs([torch.nn.functional.linear, torch._C._nn.linear]).match(
            node
        ) and is_linear_node_can_be_fused(node):
            input = get_arg_value(node, 0, "input")
            weight = get_arg_value(node, 1, "weight")
            bias = get_arg_value(node, 2, "bias")
            # Skip fusion when weight is a tensor subclass that can't handle aten.cat.
            # fuse() calls torch.cat on the weight example_values, which fails for
            # subclasses lacking aten.cat dispatch.  Probe it here so subclasses that
            # do implement aten.cat (e.g. future torchao versions) still get fused.
            weight_val = weight.meta.get("example_value", weight.meta.get("val"))
            if weight_val is not None and type(weight_val) not in (
                torch.Tensor,
                FakeTensor,
            ):
                try:
                    _ = torch.cat([weight_val[:1], weight_val[:1]], dim=0)
                except (RuntimeError, TypeError):
                    return None
            bias_tensor = None
            if bias is not None:
                bias_tensor = bias.meta.get("val", bias.meta.get("example_value"))
            bias_dim = None if bias_tensor is None else bias_tensor.ndim  # type: ignore[union-attr]
            group_key = ("batch_linear_lhs", bias_dim, input)
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        # [FUNCTION DOC] Concatenate weights (and biases), transpose, run one mm or addmm on the shared input, split the result
        # into per-linear sections and getitem each; replace and erase the originals; sets example_value meta along the
        # way. Raises AssertionError if the inputs differ. Increments the batch_linear_lhs counter.
        # KNOWN ISSUE: biases are collected only for linears that have one, relying on the group key (bias rank) to keep them uniform.
        batch_nodes = []
        batch_input = None
        batch_weights, batch_weights_meta = [], []
        batch_biases, batch_biases_meta = [], []
        split_sections = []
        for node in subset:
            input = get_arg_value(node, 0, "input")
            weight = get_arg_value(node, 1, "weight")
            bias = get_arg_value(node, 2, "bias")
            batch_nodes.append(node)
            if batch_input is None:
                batch_input = input
            else:
                if batch_input is not input:
                    raise AssertionError(
                        f"expected batch_input to be input, got {batch_input}"
                    )
            batch_weights.append(weight)
            batch_weights_meta.append(weight.meta["example_value"])
            if bias:
                batch_biases.append(bias)
                batch_biases_meta.append(bias.meta["example_value"])
            split_sections.append(weight.meta["example_value"].shape[0])

        with graph.inserting_before(subset[0]):  # type: ignore[operator]
            cat_weights = graph.call_function(  # type: ignore[operator]
                torch.cat, args=(batch_weights,), kwargs={"dim": 0}
            )
            cat_weights.meta["example_value"] = torch.cat(batch_weights_meta, dim=0)
            transposed_weights = graph.call_function(  # type: ignore[operator]
                torch.transpose, args=(cat_weights, 0, 1)
            )
            transposed_weights.meta["example_value"] = torch.transpose(
                cat_weights.meta["example_value"], 0, 1
            )
            if len(batch_biases) > 0:
                cat_biases = graph.call_function(  # type: ignore[operator]
                    torch.cat, args=(batch_biases,), kwargs={"dim": 0}
                )
                cat_biases.meta["example_value"] = torch.cat(batch_biases_meta, dim=0)
                fused_lhs = graph.call_function(  # type: ignore[operator]
                    torch.addmm,
                    args=(cat_biases, batch_input, transposed_weights),
                )
                fused_lhs.meta["example_value"] = torch.addmm(
                    cat_biases.meta["example_value"],
                    batch_input.meta["example_value"],  # type: ignore[union-attr]
                    transposed_weights.meta["example_value"],
                )
            else:
                fused_lhs = graph.call_function(  # type: ignore[operator]
                    torch.mm,
                    args=(batch_input, transposed_weights),
                )
                fused_lhs.meta["example_value"] = torch.mm(
                    batch_input.meta["example_value"],  # type: ignore[union-attr]
                    transposed_weights.meta["example_value"],
                )
            fused_lhs_list = graph.call_function(  # type: ignore[operator]
                torch.split, args=(fused_lhs, split_sections), kwargs={"dim": 1}
            )

        for i, node in enumerate(batch_nodes):
            with graph.inserting_after(fused_lhs_list):  # type: ignore[operator]
                new_node = graph.call_function(  # type: ignore[operator]
                    operator.getitem, args=(fused_lhs_list, i)
                )
            node.replace_all_uses_with(new_node)
            new_node.meta.update(node.meta)
            graph.erase_node(node)  # type: ignore[operator]
        counters["inductor"]["batch_linear_lhs"] += 1


# Poor person's check for if a node in the graph mutates its input.
# (the graph is torch IR, so we will see torch fns and python operators)
def _is_mutable_node(tgt):
    # [FUNCTION DOC] HELPER: Heuristic test whether a call target mutates its input: name ends with an underscore, or an
    # in-place python operator (module _operator, name starting with i).
    if str(tgt).endswith("_"):
        # e.g. torch.mul_, torch.Tensor.mul_
        return True
    if (
        hasattr(tgt, "__module__")
        and tgt.__module__ == "_operator"
        and tgt.__name__.startswith("i")
    ):
        # e.g. operator.iand, operator.imul
        return True
    return False


def is_linear_node_can_be_fused(node: torch.fx.Node):
    # [FUNCTION DOC] True if a linear node is fusable: valid example_value meta for node, input and weight, both input and
    # weight rank 2, and no user is a mutating op.
    input = get_arg_value(node, 0, "input")
    weight = get_arg_value(node, 1, "weight")
    return (
        is_node_meta_valid(node)
        and is_node_meta_valid(input)
        and is_node_meta_valid(weight)
        and len(input.meta["example_value"].shape) == 2
        and len(weight.meta["example_value"].shape) == 2
        # the mm -> bmm transform adds an unbind() op,
        # which is not safe for autograd when the output of the mm is mutated.
        # don't pattern match if any users of the mm mutate the input.
        and not any(_is_mutable_node(user.target) for user in node.users)
    )


@register_fusion("batch_linear")
class PreGradBatchLinearFusion(BatchFusion):
    # [CLASS DOC] Pre-grad rule batching linear calls with equal input and weight shapes into a stacked bmm (or baddbmm
    # with unsqueezed biases), then unbind and getitem per original. Registered as batch_linear (pre-grad).
    """
    Batch linear fusion in pre grad pass.
    Fuse linear with same size with torch.baddbmm
    """

    def _getitem_args(self, getitem_node: torch.fx.Node):
        # [FUNCTION DOC] HELPER: Return the first argument of a getitem call node, or None if the node is not one.
        if getitem_node.target != operator.__getitem__ or (
            getitem_node.op != "call_function"
        ):
            return None
        return getitem_node.args[0]

    def match(self, node: torch.fx.Node):
        # [FUNCTION DOC] Return a group key (parent of getitem input, input and weight shapes, bias absent flag, users) for
        # fusable linear calls, else None.
        if CallFunctionVarArgs([torch.nn.functional.linear, torch._C._nn.linear]).match(
            node
        ) and is_linear_node_can_be_fused(node):
            input = get_arg_value(node, 0, "input")
            weight = get_arg_value(node, 1, "weight")
            bias = get_arg_value(node, 2, "bias")
            if self.graph_search_options.get("fuse_nodes_with_same_users", False):
                users = [user.target for user in node.users]
            else:
                users = ""  # type: ignore[assignment]
            group_key = (
                "batch_linear",
                self._getitem_args(input),
                str(input.meta["example_value"].shape),
                str(weight.meta["example_value"].shape),
                bias is None,
                str(users),
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        # [FUNCTION DOC] Stack inputs and weights, transpose weights, run bmm (no biases) or baddbmm (with stacked, unsqueezed
        # biases; meta update failures, for example with dynamic shapes, are logged and skipped), unbind, replace each
        # linear with a getitem, erase originals; increments the batch_linear counter.
        batch_nodes = []
        batch_inputs = []
        batch_weights = []
        batch_biases = []
        batch_inputs_metadata = []
        batch_weights_metadata = []
        batch_biases_metadata = []
        for node in subset:
            batch_nodes.append(node)
            input = get_arg_value(node, 0, "input")
            batch_inputs.append(input)
            batch_inputs_metadata.append(input.meta["example_value"])
            weight = get_arg_value(node, 1, "weight")
            batch_weights.append(weight)
            batch_weights_metadata.append(weight.meta["example_value"])
            bias = get_arg_value(node, 2, "bias")
            batch_biases.append(bias)
            if bias is not None and hasattr(bias, "meta"):
                batch_biases_metadata.append(bias.meta["example_value"])

        with graph.inserting_before(subset[0]):  # type: ignore[operator]
            stack_inputs = graph.call_function(  # type: ignore[operator]
                torch.stack, args=(batch_inputs,), kwargs={"dim": 0}
            )
            update_stack_example_value(stack_inputs, batch_inputs_metadata)
            stack_weights = graph.call_function(  # type: ignore[operator]
                torch.stack, args=(batch_weights,), kwargs={"dim": 0}
            )
            update_stack_example_value(stack_weights, batch_weights_metadata)
            transpose_weight = graph.call_function(  # type: ignore[operator]
                torch.transpose, args=(stack_weights, 1, 2)
            )
            transpose_weight.meta["example_value"] = torch.transpose(
                stack_weights.meta["example_value"], 1, 2
            )
            if all(bias is None for bias in batch_biases):
                bmm = graph.call_function(  # type: ignore[operator]
                    torch.bmm,
                    args=(stack_inputs, transpose_weight),
                )
                bmm.meta["example_value"] = torch.bmm(
                    stack_inputs.meta["example_value"],
                    transpose_weight.meta["example_value"],
                )
                bmm_meta = bmm.meta["example_value"]
            else:
                stack_biases = graph.call_function(  # type: ignore[operator]
                    torch.stack, args=(batch_biases,), kwargs={"dim": 0}
                )
                update_stack_example_value(stack_biases, batch_biases_metadata)
                unsqueeze_biases = graph.call_function(  # type: ignore[operator]
                    torch.unsqueeze, args=(stack_biases, 1)
                )
                unsqueeze_biases.meta["example_value"] = torch.unsqueeze(
                    stack_biases.meta["example_value"], 1
                )
                bmm = graph.call_function(  # type: ignore[operator]
                    torch.baddbmm,
                    args=(unsqueeze_biases, stack_inputs, transpose_weight),
                )
                try:
                    # it will have runtime error to broadcast when it has dynamic shape included
                    # in the meta data, so we need to skip the update meta data
                    bmm.meta["example_value"] = torch.baddbmm(
                        unsqueeze_biases.meta["example_value"],
                        stack_inputs.meta["example_value"],
                        transpose_weight.meta["example_value"],
                    )
                    bmm_meta = bmm.meta["example_value"]
                except Exception as e:
                    log.debug(
                        f" exception when update bmm meta data with stack error traceback {e}"  # noqa: G004
                    )
                    bmm_meta = None

            bmm = graph.call_function(torch.unbind, args=(bmm,), kwargs={"dim": 0})  # type: ignore[operator]
            if bmm_meta is not None:
                bmm.meta["example_value"] = torch.unbind(bmm_meta, dim=0)
            for i, linear in enumerate(batch_nodes):
                with graph.inserting_after(bmm):  # type: ignore[operator]
                    getitem = graph.call_function(operator.getitem, args=(bmm, i))  # type: ignore[operator]
                linear.replace_all_uses_with(getitem)
                getitem.meta.update(linear.meta)
                graph.erase_node(linear)  # type: ignore[operator]
        counters["inductor"]["batch_linear"] += 1


# Profitability gates for cat_linear (opt-in; numbers come from the sweep in the
# PR description). The split only wins for a narrow output fed by a large,
# memory-bound concat, so gate on output width and concat traffic.
# Widest win output is 96; first loss at N=128.
CAT_LINEAR_MAX_OUTPUT_WIDTH = 96
# Only the concat's M*K_total*dtype_bytes of HBM traffic is removed, so it has to
# cover the per-piece launch + reduce. Bytes, so fp32 trips it at half the bf16
# element count.
CAT_LINEAR_CAT_TRAFFIC_FLOOR_BYTES = 512 * 1024 * 1024
# Skip slivers: a piece this thin is a launch-bound matmul with no upside.
CAT_LINEAR_MIN_PIECE_WIDTH = 8
# More parts means more launches and reduce-adds; >3 lost in the sweep.
CAT_LINEAR_MAX_PARTS = 3


def _weight_hoist_anchor(graph, weight):
    # [FUNCTION DOC] HELPER: Choose the node after which hoisted weight slices are inserted: the weight itself, or the last
    # leading placeholder when the weight is a placeholder.
    # Insert hoisted slices after the weight's own def, or after the last
    # placeholder when it's a lifted input (can't insert a call between placeholders).
    if weight.op == "placeholder":
        anchor = weight
        for n in graph.nodes:
            if n.op == "placeholder":
                anchor = n
            else:
                break
        return anchor
    return weight


def _meta_shape(node):
    # [FUNCTION DOC] HELPER: Return the static shape of a node as a tuple of ints from example_value or val meta, or None if
    # the node is invalid, has no shape, or has symbolic dims.
    if not isinstance(node, torch.fx.Node) or not is_node_meta_valid(node):
        return None
    val = node.meta.get("example_value")
    if val is None:
        val = node.meta.get("val")
    if val is None or not hasattr(val, "shape"):
        return None
    shape = val.shape
    # static shapes only: bail on symbolic so we never specialize a backed
    # SymInt or trip over an unbacked one (same free_symbols guard split_cat uses)
    if free_symbols(shape):
        return None
    return tuple(int(s) for s in shape)


def match_cat_linear(node: torch.fx.Node):
    # [FUNCTION DOC] Match linear applied to a last-dim concatenation of 2 to CAT_LINEAR_MAX_PARTS nodes. Returns
    # (parts, weight, bias, offsets) or None. Requires: single-user cat, static shapes, cat on last dim, every
    # piece at least the minimum width, matching weight contraction dim, output width within the maximum, and cat
    # traffic in bytes at least the floor constant.
    """Match linear(cat([parts...], dim=-1), weight, bias).

    Returns (parts, weight, bias, offsets) or None. Conservative on purpose:
    last-dim cat only, total width and part count capped, no thin slivers, and
    a weight whose contraction dim lines up with the cat. The cat must have a
    single user so it can actually be erased once the linear is rewritten.
    """
    cat_in = get_arg_value(node, 0, "input")
    weight = get_arg_value(node, 1, "weight")
    bias = get_arg_value(node, 2, "bias")
    if not isinstance(cat_in, torch.fx.Node) or not CallFunctionVarArgs(
        [torch.cat]
    ).match(cat_in):
        return None
    if len(cat_in.users) != 1:
        return None

    parts = get_arg_value(cat_in, 0, "tensors")
    if not isinstance(parts, (list, tuple)) or not (
        2 <= len(parts) <= CAT_LINEAR_MAX_PARTS
    ):
        return None
    if not all(isinstance(p, torch.fx.Node) for p in parts):
        return None

    cat_shape = _meta_shape(cat_in)
    if cat_shape is None or len(cat_shape) < 2:
        return None
    rank = len(cat_shape)
    cat_dim = get_arg_value(cat_in, 1, "dim")
    if cat_dim is None:
        cat_dim = 0
    if not isinstance(cat_dim, int):
        return None
    if (cat_dim + rank if cat_dim < 0 else cat_dim) != rank - 1:
        return None

    k_total = int(cat_shape[-1])

    offsets = [0]
    for p in parts:
        sh = _meta_shape(p)
        if sh is None or len(sh) != rank:
            return None
        k_i = int(sh[-1])
        if k_i < CAT_LINEAR_MIN_PIECE_WIDTH:
            return None
        offsets.append(offsets[-1] + k_i)
    if offsets[-1] != k_total:
        return None

    w_shape = _meta_shape(weight)
    if w_shape is None or len(w_shape) != 2 or int(w_shape[-1]) != k_total:
        return None

    # N is the profit variable; wide outputs are compute-bound and lose the split.
    n_out = int(w_shape[0])
    if n_out > CAT_LINEAR_MAX_OUTPUT_WIDTH:
        return None

    # traffic = prod(leading dims) * k_total * dtype_bytes.
    cat_val = cat_in.meta.get("example_value")
    if cat_val is None:
        cat_val = cat_in.meta.get("val")
    item_bytes = getattr(getattr(cat_val, "dtype", None), "itemsize", 2)
    m_lead = 1
    for d in cat_shape[:-1]:
        m_lead *= int(d)
    if m_lead * k_total * item_bytes < CAT_LINEAR_CAT_TRAFFIC_FLOOR_BYTES:
        return None

    return parts, weight, bias, offsets


@register_fusion("cat_linear")
class CatLinearFusion(BatchFusion):
    # [CLASS DOC] Pre-grad rule rewriting linear of a concatenation into a sum of linears on contiguous column slices of
    # the weight, avoiding materializing the concatenation. Per-node rewrite (min_fuse_set_size forced to 1; each
    # match is its own group). State: _piece_cache maps (weight, start, end) to the sliced contiguous weight so
    # tied weights are sliced once. Opt-in as cat_linear.
    """
    Rewrite linear(cat([x0, x1, ...], dim=-1), weight, bias) into a sum of
    per-piece linears on contiguous last-dim weight slices, so the concatenated
    activation is never materialised in forward or backward.

    Unlike the other fusions here this is a per-node rewrite (one linear(cat)
    splits into several smaller linears) rather than a batch of like-shaped ops,
    so every match forms its own group of one. Off by default; opt in via
    pre_grad_fusion_options["cat_linear"].
    """

    def __init__(self, **kwargs) -> None:
        # [FUNCTION DOC] Initialize with min_fuse_set_size forced to 1 and an empty weight-piece cache.
        super().__init__(**kwargs)
        self.graph_search_options = {
            **self.graph_search_options,
            "min_fuse_set_size": 1,
        }
        # Dedup hoisted weight slices across the several fuse() calls this rule
        # makes on one graph (weight-tied blocks match at multiple sites). A fresh
        # rule instance is built per compile in generate_fusion_from_config and
        # only ever runs on a single graph, so an instance-level cache is safe.
        self._piece_cache: dict = {}

    def match(self, node: torch.fx.Node):
        # [FUNCTION DOC] Return a unique-per-node group key if the node is a linear with valid meta that match_cat_linear accepts, else None. Side-effect free.
        if not CallFunctionVarArgs(
            [torch.nn.functional.linear, torch._C._nn.linear]
        ).match(node):
            return None
        if not is_node_meta_valid(node):
            return None
        # match() runs on every linear during the candidate BFS, including nodes
        # that never fuse, so keep it side-effect-free: recompute in fuse() rather
        # than stashing on node.meta (which would leave stale meta behind).
        if match_cat_linear(node) is None:
            return None
        return ("cat_linear", node)

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        # [FUNCTION DOC] For each node, re-run the match; hoist a slice plus clone of the weight per piece (cached), emit a
        # linear per part (bias only on the first piece), set example values, and chain the partial results with add;
        # replace the original node and erase it; increments the cat_linear counter.
        # Slice+clone each weight once and hoist to entry, so a weight matched by
        # several cat_linears (weight-tied blocks) isn't re-sliced at every site.
        cache = self._piece_cache

        for node in subset:
            matched = match_cat_linear(node)
            if matched is None:
                continue
            parts, weight, bias, offsets = matched
            weight_val = weight.meta.get("example_value", weight.meta.get("val"))
            bias_val = None
            if bias is not None:
                bias_val = bias.meta.get("example_value", bias.meta.get("val"))
            anchor = _weight_hoist_anchor(graph, weight)

            with graph.inserting_before(node):  # type: ignore[operator]
                partials = []
                for i, part in enumerate(parts):
                    key = (weight, offsets[i], offsets[i + 1])
                    w_cont = cache.get(key)
                    if w_cont is None:
                        # slice is a differentiable view of the full weight; the
                        # clone makes each GEMM operand contiguous. Folds to a const
                        # when frozen; in training it's a cheap weight-only op.
                        # pyrefly: ignore [not-callable]
                        with graph.inserting_after(anchor):
                            w_slice = graph.call_function(  # type: ignore[operator]
                                aten.slice.Tensor,
                                args=(weight, 1, offsets[i], offsets[i + 1]),
                            )
                        # pyrefly: ignore [not-callable]
                        with graph.inserting_after(w_slice):
                            w_cont = graph.call_function(  # type: ignore[operator]
                                aten.clone.default,
                                args=(w_slice,),
                                kwargs={"memory_format": torch.contiguous_format},
                            )
                        if weight_val is not None:
                            sv = aten.slice.Tensor(  # type: ignore[operator]
                                weight_val, 1, offsets[i], offsets[i + 1]
                            )
                            w_slice.meta["example_value"] = sv
                            w_cont.meta["example_value"] = sv.contiguous()  # type: ignore[operator]
                        cache[key] = w_cont
                    # bias rides on the first piece only; folding it into the
                    # reduce-add would just be an extra node.
                    out_i = graph.call_function(  # type: ignore[operator]
                        torch.nn.functional.linear,
                        args=(part, w_cont, bias if i == 0 else None),
                    )
                    # Each synthesized node gets its own fake val computed from its
                    # own operands, the way BatchLinearLHSFusion does. The old code
                    # aliased the whole linear's fake tensor onto every piece and
                    # every reduce-add, sharing one FakeTensor object across
                    # distinct nodes.
                    part_val = part.meta.get("example_value", part.meta.get("val"))
                    w_cont_val = w_cont.meta.get("example_value")
                    if part_val is not None and w_cont_val is not None:
                        out_i.meta["example_value"] = torch.nn.functional.linear(
                            part_val, w_cont_val, bias_val if i == 0 else None
                        )
                    partials.append(out_i)

                acc = partials[0]
                for out_i in partials[1:]:
                    prev = acc
                    acc = graph.call_function(operator.add, args=(prev, out_i))  # type: ignore[operator]
                    prev_val = prev.meta.get("example_value")
                    out_val = out_i.meta.get("example_value")
                    if prev_val is not None and out_val is not None:
                        acc.meta["example_value"] = torch.add(prev_val, out_val)

            node.replace_all_uses_with(acc)
            graph.erase_node(node)  # type: ignore[operator]
            counters["inductor"]["cat_linear"] += 1


@register_fusion("batch_layernorm")
class BatchLayernormFusion(BatchFusion):
    # [CLASS DOC] Pre-grad rule batching layer_norm calls with equal shapes, normalized shape and eps by stacking inputs
    # (and weights, biases), running one layer_norm, applying weight multiplication and bias addition, then unbind and getitem.
    # Registered as batch_layernorm.
    """
    Batch layer norm fusion in pre grad pass
    """

    def match(self, node: torch.fx.Node):
        # [FUNCTION DOC] Return a group key (shapes of input, weight, bias, normalized shape, eps, users) for layer_norm calls
        # with available example values, else None.
        if CallFunctionVarArgs(torch.nn.functional.layer_norm).match(node):
            input = get_arg_value(node, 0, "input")
            weight = get_arg_value(node, 2, "weight")
            bias = get_arg_value(node, 3, "bias")
            if self.graph_search_options.get("fuse_nodes_with_same_users", False):
                users = [user.target for user in node.users]
            else:
                users = ""  # type: ignore[assignment]
            group_key = (
                (
                    "batch_layernorm",
                    str(input.meta["example_value"].shape),
                    str(weight.meta["example_value"].shape)
                    if weight is not None
                    else "",
                    str(bias.meta["example_value"].shape) if bias is not None else "",
                    str(get_arg_value(node, 1, "normalized_shape")),
                    str(get_arg_value(node, 4, "eps")),
                    str(users),
                )
                if "example_value" in input.meta
                and is_node_meta_valid(weight)
                and is_node_meta_valid(bias)
                else None
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        # [FUNCTION DOC] Stack inputs along a negative dim derived from the normalized shape, stack weights and biases when
        # present, run a layer_norm, apply mul by weight and add of bias as appropriate, unbind, getitem per original, and erase
        # originals; increments the batch_layernorm counter. Raises AssertionError if the eps values differ.
        group_inputs = []
        group_shapes = []
        group_weights = []
        group_biases = []
        group_epss = []
        group_nodes = []
        group_inputs_metadata = []
        group_biases_metadata = []
        group_weights_metadata = []
        for node in subset:
            group_nodes.append(node)
            input = get_arg_value(node, 0, "input")
            group_inputs.append(input)
            group_inputs_metadata.append(input.meta["example_value"])
            group_shapes.append(get_arg_value(node, 1, "normalized_shape"))
            weight = get_arg_value(node, 2, "weight")
            group_weights.append(weight)
            if weight is not None and hasattr(weight, "meta"):
                group_weights_metadata.append(weight.meta["example_value"])
            bias = get_arg_value(node, 3, "bias")
            group_biases.append(bias)
            if bias is not None and hasattr(bias, "meta"):
                group_biases_metadata.append(bias.meta["example_value"])
            eps = get_arg_value(node, 4, "eps")
            if eps is None:
                eps = 1e-5
            group_epss.append(eps)
        stack_dim = -1 - len(group_shapes[-1])

        if all(bias is None for bias in group_biases):
            group_biases = None  # type: ignore[assignment]
        if all(weight is None for weight in group_weights):
            group_weights = None  # type: ignore[assignment]
        if not all(eps == group_epss[0] for eps in group_epss):
            raise AssertionError("all epsilon values must be equal")

        with graph.inserting_before(subset[0]):  # type: ignore[operator]
            stack_input = graph.call_function(  # type: ignore[operator]
                torch.stack, args=(group_inputs,), kwargs={"dim": stack_dim}
            )
            update_stack_example_value(stack_input, group_inputs_metadata, stack_dim)
            if group_weights is not None:
                stack_weight = graph.call_function(  # type: ignore[operator]
                    torch.stack, args=(group_weights,), kwargs={"dim": 0}
                )
                update_stack_example_value(stack_weight, group_weights_metadata)
            else:
                stack_weight = None
            if group_biases is not None:
                stack_bias = graph.call_function(  # type: ignore[operator]
                    torch.stack, args=(group_biases,), kwargs={"dim": 0}
                )
                update_stack_example_value(stack_bias, group_biases_metadata)
            else:
                stack_bias = None

            batch_layer_norm = graph.call_function(  # type: ignore[operator]
                torch.nn.functional.layer_norm,
                args=(stack_input, group_shapes[-1]),
                kwargs={"eps": group_epss[-1]},
            )
            batch_layer_norm.meta["example_value"] = stack_input.meta["example_value"]

            if group_weights is not None and group_biases is not None:
                previous_batch_layer_norm_meta = batch_layer_norm.meta["example_value"]
                batch_layer_norm = graph.call_function(  # type: ignore[operator]
                    torch.mul, args=(stack_weight, batch_layer_norm)
                )
                update_pointwise_example_value(
                    batch_layer_norm,
                    # pyrefly: ignore [missing-attribute]
                    stack_weight.meta["example_value"],
                    previous_batch_layer_norm_meta,
                    torch.mul,
                )
                previous_batch_layer_norm_meta = batch_layer_norm.meta["example_value"]
                batch_layer_norm = graph.call_function(  # type: ignore[operator]
                    torch.add, args=(stack_bias, batch_layer_norm)
                )
                update_pointwise_example_value(
                    batch_layer_norm,
                    # pyrefly: ignore [missing-attribute]
                    stack_bias.meta["example_value"],
                    previous_batch_layer_norm_meta,
                    torch.add,
                )
            elif group_weights is not None and group_biases is None:
                previous_batch_layer_norm_meta = batch_layer_norm.meta["example_value"]
                # pyrefly: ignore [not-callable]
                batch_layer_norm = graph.call_function(
                    torch.mul, args=(stack_weight, batch_layer_norm)
                )
                update_pointwise_example_value(
                    batch_layer_norm,
                    # pyrefly: ignore [missing-attribute]
                    stack_weight.meta["example_value"],
                    previous_batch_layer_norm_meta,
                    torch.mul,
                )
            elif group_weights is None and group_biases is not None:
                previous_batch_layer_norm_meta = batch_layer_norm.meta["example_value"]
                # pyrefly: ignore [not-callable]
                batch_layer_norm = graph.call_function(
                    torch.add, args=(stack_bias, batch_layer_norm)
                )
                update_pointwise_example_value(
                    batch_layer_norm,
                    # pyrefly: ignore [missing-attribute]
                    stack_bias.meta["example_value"],
                    previous_batch_layer_norm_meta,
                    torch.add,
                )

            batch_layer_norm_unbind = graph.call_function(  # type: ignore[operator]
                torch.unbind,
                args=(batch_layer_norm,),
                kwargs={"dim": stack_dim},
            )
            update_stack_example_value(
                batch_layer_norm_unbind,
                batch_layer_norm.meta["example_value"],
                op=torch.unbind,
                dim=stack_dim,
            )

        for i, node in enumerate(group_nodes):
            with graph.inserting_after(batch_layer_norm_unbind):  # type: ignore[operator]
                new_node = graph.call_function(  # type: ignore[operator]
                    operator.getitem, args=(batch_layer_norm_unbind, i)
                )
            node.replace_all_uses_with(new_node)
            new_node.meta.update(node.meta)
            graph.erase_node(node)  # type: ignore[operator]
        counters["inductor"]["batch_layernorm"] += 1


class BatchPointwiseOpsPreGradFusion(BatchPointwiseOpsFusionFactory):
    # [CLASS DOC] Pre-grad rule batching unary pointwise ops (such as sigmoid, relu, tanh) of equal input shape by stacking
    # inputs, applying the op once, unbinding. The op attribute is the torch function.
    """
    Batch pointwise ops (e.g., sigmoid, relu, tanh) fusion in pre grad pass.
    We fuse it in random place, and the introduced stack node may be merged in split cat.
    """

    def __init__(self, op, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Store the op and forward options to the base class.
        super().__init__(op, **kwargs)
        self.op = op

    def match(self, node: torch.fx.Node):
        # [FUNCTION DOC] Return a group key (op name, input shape, inplace flag, optional parent target) or None.
        input = get_arg_value(node, 0, "input")
        if CallFunctionVarArgs(self.op).match(node) and is_node_meta_valid(node):
            if self.graph_search_options.get("fuse_nodes_with_same_parent", False):
                # pyre-fixme[16]
                parent = node.args[0]
                parent = parent.target if parent is not None else ""  # type: ignore[union-attr]
            else:
                parent = ""
            # for relu op, we also use the inplace to construct the key
            group_key = (
                "batch_" + self.op.__name__.lower().split(".")[0],
                str(input.meta["example_value"].shape),
                str(node.kwargs.get("inplace", False)),
                str(parent),
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        # [FUNCTION DOC] Stack inputs, apply the op once (relu keeps the inplace flag), unbind, getitem per original, replace and erase the originals;
        # increments the per-op batch counter.
        batch_nodes = []
        batch_inputs = []
        batch_inputs_metadata = []

        for node in subset:
            batch_nodes.append(node)
            input = get_arg_value(node, 0, "input")
            batch_inputs.append(input)
            batch_inputs_metadata.append(input.meta["example_value"])

        with graph.inserting_before(subset[0]):  # type: ignore[operator]
            stack_inputs = graph.call_function(  # type: ignore[operator]
                torch.stack, args=(batch_inputs,), kwargs={"dim": 0}
            )
            update_stack_example_value(stack_inputs, batch_inputs_metadata)
            if self.op is torch.nn.functional.relu:
                batch_op = graph.call_function(  # type: ignore[operator]
                    self.op,
                    args=(stack_inputs,),
                    kwargs={"inplace": subset[0].kwargs.get("inplace", False)},
                )
                batch_op.meta["example_value"] = self.op(
                    stack_inputs.meta["example_value"],
                    # pyrefly: ignore [bad-argument-type]
                    inplace=subset[0].kwargs.get("inplace", False),
                )
            else:
                batch_op = graph.call_function(  # type: ignore[operator]
                    self.op,
                    args=(stack_inputs,),
                )
                batch_op.meta["example_value"] = self.op(
                    stack_inputs.meta["example_value"]
                )
            unbind_op = graph.call_function(  # type: ignore[operator]
                torch.unbind, args=(batch_op,), kwargs={"dim": 0}
            )
            unbind_op.meta["example_value"] = torch.unbind(
                batch_op.meta["example_value"], dim=0
            )
            for i, node in enumerate(batch_nodes):
                with graph.inserting_after(unbind_op):  # type: ignore[operator]
                    getitem = graph.call_function(operator.getitem, args=(unbind_op, i))  # type: ignore[operator]
                node.replace_all_uses_with(getitem)
                getitem.meta.update(node.meta)
                graph.erase_node(node)  # type: ignore[operator]
        counters["inductor"]["batch_" + self.op.__name__.lower().split(".")[0]] += 1


class BatchPointwiseOpsPostGradFusion(BatchPointwiseOpsFusionFactory):
    # [CLASS DOC] Post-grad rule batching unary aten pointwise ops of equal input shape by stacking (via decompose_stack),
    # applying the op once and selecting each result.
    """
    Batch pointwise ops (e.g., sigmoid, relu, tanh) fusion in post grad pass.
    The introduced stack node may be merged in split cat.
    """

    def __init__(self, op, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Store the op and forward options to the base class.
        super().__init__(op, **kwargs)
        self.op = op

    def match(self, node: torch.fx.Node):
        # [FUNCTION DOC] Return a group key (op name, input val shape, inplace flag, optional parent target) or None.
        input = get_arg_value(node, 0, "input")
        if CallFunctionVarArgs(self.op).match(node) and is_node_meta_valid(node):
            # for relu op, we also use the inplace to construct the key
            # we batch the ops with same parent to enable followup split cat
            parent = node.args[0]
            parent = (
                parent.target  # type: ignore[union-attr]
                if self.graph_search_options.get("fuse_nodes_with_same_parent", False)
                else ""
            )
            group_key = (
                "batch_aten_" + self.op.__name__.lower().split(".")[0],
                str(input.meta["val"].shape),
                str(node.kwargs.get("inplace", False)),
                # pyre-fixme[16]
                str(parent),
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        # [FUNCTION DOC] Stack inputs by unsqueeze and cat, apply the op once, select each result with aten select, replace and erase the originals;
        # increments the per-op batch counter.
        batch_nodes = []
        batch_inputs = []
        batch_inputs_metadata = []

        for node in subset:
            batch_nodes.append(node)
            input = get_arg_value(node, 0, "input")
            batch_inputs.append(input)
            batch_inputs_metadata.append(input.meta["val"])

        with graph.inserting_before(subset[0]):  # type: ignore[operator]
            stack_inputs = decompose_stack(graph, batch_inputs)
            update_stack_example_value(stack_inputs, batch_inputs_metadata)
            batch_op = graph.call_function(  # type: ignore[operator]
                self.op,
                args=(stack_inputs,),
            )
            for i, node in enumerate(batch_nodes):
                with graph.inserting_after(batch_op):  # type: ignore[operator]
                    getitem = graph.call_function(aten.select, args=(batch_op, 0, i))  # type: ignore[operator]
                node.replace_all_uses_with(getitem)
                getitem.meta.update(node.meta)
                graph.erase_node(node)  # type: ignore[operator]
        counters["inductor"][
            "batch_aten_" + self.op.__name__.lower().split(".")[0]
        ] += 1


class BatchMathOpsPreGradFusion(BatchPointwiseOpsFusionFactory):
    # [CLASS DOC] Pre-grad rule batching simple math ops (detach, nan_to_num, clamp, dropout) with the same input shape and
    # extra arguments by stacking, applying the op, unbinding. Subclasses choose the op; _match_op can be overridden.
    """
    Batch simple math related ops such as nan_to_num in pre grad pass.
    """

    def __init__(self, op, **kwargs):
        # [FUNCTION DOC] HELPER: Store the op and forward options to the base class.
        super().__init__(op, **kwargs)
        self.op = op

    def match(self, node: torch.fx.Node):
        # [FUNCTION DOC] Return a group key built from input shape, extra args, kwargs and the first user's target, or None.
        # KNOWN ISSUE: takes the first user of the node, which fails with StopIteration if the node has no users.
        input = get_arg_value(node, 0, "input")
        if self._match_op(node) and is_node_meta_valid(node):
            # check the input has the same shape and its users have the same target
            # check all clamp operators have the same min and max values, and
            # nan_to_num operators use the same default value.
            child = next(iter(node.users.keys()))
            group_key = (
                str(input.meta["example_value"].shape)
                + str(node.args[1:])
                + str(node.kwargs)
                + str(child.target)
            )
        else:
            group_key = None
        return group_key

    def _match_op(self, node: torch.fx.Node) -> MatchResult:
        # [FUNCTION DOC] HELPER: Return whether node calls the rule's op; subclasses may broaden this.
        return CallFunctionVarArgs(self.op).match(node)

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        # [FUNCTION DOC] Stack inputs, call the op once with the first node's extra args and kwargs, unbind, getitem per original,
        # replace and erase the originals; increments the per-op batch counter.
        batch_nodes = []
        batch_inputs = []
        batch_inputs_metadata = []
        args = subset[0].args[1:]
        kwargs = subset[0].kwargs

        for node in subset:
            batch_nodes.append(node)
            input = get_arg_value(node, 0, "input")
            batch_inputs.append(input)
            batch_inputs_metadata.append(input.meta["example_value"])

        with graph.inserting_before(subset[0]):  # type: ignore[operator]
            stack_inputs = graph.call_function(  # type: ignore[operator]
                torch.stack, args=(batch_inputs,), kwargs={"dim": 0}
            )
            update_stack_example_value(stack_inputs, batch_inputs_metadata)
            batch_op = graph.call_function(  # type: ignore[operator]
                self.op,
                args=(stack_inputs, *args),
                kwargs=kwargs,
            )
            batch_op.meta["example_value"] = self.op(
                stack_inputs.meta["example_value"], *args, **kwargs
            )
            unbind_op = graph.call_function(  # type: ignore[operator]
                torch.unbind, args=(batch_op,), kwargs={"dim": 0}
            )
            unbind_op.meta["example_value"] = torch.unbind(
                batch_op.meta["example_value"], dim=0
            )
            for i, node in enumerate(batch_nodes):
                with graph.inserting_after(unbind_op):  # type: ignore[operator]
                    getitem = graph.call_function(operator.getitem, args=(unbind_op, i))  # type: ignore[operator]
                node.replace_all_uses_with(getitem)
                getitem.meta.update(node.meta)
                graph.erase_node(node)  # type: ignore[operator]
        counters["inductor"]["batch_" + self.op.__name__.lower().split(".")[0]] += 1


@register_fusion("batch_tanh")
class BatchTanhPreGradFusion(BatchPointwiseOpsPreGradFusion):
    # [CLASS DOC] Pre-grad batch rule for torch.tanh (registered batch_tanh).
    def __init__(self, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Fix the batched op for this rule and forward options to the parent class.
        super().__init__(torch.tanh, **kwargs)


@register_fusion("batch_sigmoid")
class BatchSigmoidPreGradFusion(BatchPointwiseOpsPreGradFusion):
    # [CLASS DOC] Pre-grad batch rule for torch.sigmoid (registered batch_sigmoid).
    def __init__(self, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Fix the batched op for this rule and forward options to the parent class.
        super().__init__(torch.sigmoid, **kwargs)


@register_fusion("batch_relu")
class BatchReLuPreGradFusion(BatchPointwiseOpsPreGradFusion):
    # [CLASS DOC] Pre-grad batch rule for relu (registered batch_relu).
    def __init__(self, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Fix the batched op for this rule and forward options to the parent class.
        super().__init__(torch.nn.functional.relu, **kwargs)


@register_fusion("batch_detach")
class BatchDetachPreGradFusion(BatchMathOpsPreGradFusion):
    # [CLASS DOC] Pre-grad batch rule for detach (registered batch_detach); also matches the detach method.
    def __init__(self, **kwargs):
        # [FUNCTION DOC] HELPER: Fix the batched op for this rule and forward options to the parent class.
        super().__init__(torch.detach, **kwargs)

    def _match_op(self, node: torch.fx.Node) -> MatchResult:
        # [FUNCTION DOC] HELPER: Match detach as either the torch function or the tensor method call.
        return CallFunctionVarArgs(torch.detach).match(node) or CallMethodVarArgs(
            "detach"
        ).match(node)


@register_fusion("batch_nan_to_num")
class BatchNanToNumPreGradFusion(BatchMathOpsPreGradFusion):
    # [CLASS DOC] Pre-grad batch rule for nan_to_num (registered batch_nan_to_num).
    def __init__(self, **kwargs):
        # [FUNCTION DOC] HELPER: Fix the batched op for this rule and forward options to the parent class.
        super().__init__(torch.nan_to_num, **kwargs)


@register_fusion("batch_clamp")
class BatchClampPreGradFusion(BatchMathOpsPreGradFusion):
    # [CLASS DOC] Pre-grad batch rule for clamp (registered batch_clamp).
    def __init__(self, **kwargs):
        # [FUNCTION DOC] HELPER: Fix the batched op for this rule and forward options to the parent class.
        super().__init__(torch.clamp, **kwargs)


@register_fusion("batch_dropout")
class BatchDropoutPreGradFusion(BatchMathOpsPreGradFusion):
    # [CLASS DOC] Pre-grad batch rule for dropout (registered batch_dropout).
    def __init__(self, **kwargs):
        # [FUNCTION DOC] HELPER: Fix the batched op for this rule and forward options to the parent class.
        super().__init__(torch.nn.functional.dropout, **kwargs)


@register_fusion("batch_aten_tanh", pre_grad=False)
class BatchTanhPostGradFusion(BatchPointwiseOpsPostGradFusion):
    # [CLASS DOC] Post-grad batch rule for aten tanh (registered batch_aten_tanh).
    def __init__(self, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Fix the batched op for this rule and forward options to the parent class.
        super().__init__(aten.tanh.default, **kwargs)


@register_fusion("batch_aten_sigmoid", pre_grad=False)
class BatchSigmoidPostGradFusion(BatchPointwiseOpsPostGradFusion):
    # [CLASS DOC] Post-grad batch rule for aten sigmoid (registered batch_aten_sigmoid).
    def __init__(self, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Fix the batched op for this rule and forward options to the parent class.
        super().__init__(aten.sigmoid.default, **kwargs)


@register_fusion("batch_aten_relu", pre_grad=False)
class BatchReLuPostGradFusion(BatchPointwiseOpsPostGradFusion):
    # [CLASS DOC] Post-grad batch rule for aten relu (registered batch_aten_relu).
    def __init__(self, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Fix the batched op for this rule and forward options to the parent class.
        super().__init__(aten.relu.default, **kwargs)


@register_fusion("batch_aten_add", pre_grad=False)
class BatchAddPostGradFusion(BatchPointwiseMathOpsPostGradFusion):
    # [CLASS DOC] Post-grad batch rule for aten add (registered batch_aten_add).
    def __init__(self, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Fix the batched op for this rule and forward options to the parent class.
        super().__init__(aten.add.Tensor, **kwargs)


@register_fusion("batch_aten_sub", pre_grad=False)
class BatchSubPostGradFusion(BatchPointwiseMathOpsPostGradFusion):
    # [CLASS DOC] Post-grad batch rule for aten sub (registered batch_aten_sub).
    def __init__(self, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Fix the batched op for this rule and forward options to the parent class.
        super().__init__(aten.sub.Tensor, **kwargs)


@register_fusion("batch_aten_div", pre_grad=False)
class BatchDivPostGradFusion(BatchPointwiseMathOpsPostGradFusion):
    # [CLASS DOC] Post-grad batch rule for aten div (registered batch_aten_div).
    def __init__(self, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Fix the batched op for this rule and forward options to the parent class.
        super().__init__(aten.div.Tensor, **kwargs)


@register_fusion("batch_aten_mul", pre_grad=False)
class BatchMulPostGradFusion(BatchPointwiseMathOpsPostGradFusion):
    # [CLASS DOC] Post-grad batch rule for aten mul (registered batch_aten_mul).
    def __init__(self, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Fix the batched op for this rule and forward options to the parent class.
        super().__init__(aten.mul.Tensor, **kwargs)


class _OrderedSet:
    # [CLASS DOC] HELPER: Minimal insertion-ordered set backed by an ordered mapping; supports membership, length, append and iteration in insertion order.
    def __init__(self, param=None) -> None:
        # [FUNCTION DOC] HELPER: Create from an optional iterable (empty/None gives an empty set).
        if param:
            self.rep = OrderedDict(dict.fromkeys(param))
        else:
            self.rep = OrderedDict()

    def __contains__(self, o) -> bool:
        # [FUNCTION DOC] HELPER: Membership test.
        return o in self.rep

    def __len__(self) -> int:
        # [FUNCTION DOC] HELPER: Number of elements.
        return self.rep.__len__()

    def append(self, o):
        # [FUNCTION DOC] HELPER: Add an element, keeping first-insertion order.
        self.rep[o] = None

    def __iter__(self):
        # [FUNCTION DOC] HELPER: Iterate elements in insertion order.
        return self.rep.keys().__iter__()


def find_independent_subset_greedy(
    node_list: Iterable[torch.fx.Node],
    graph_search_options: dict[str, Any],
) -> Iterator[Iterable[torch.fx.Node]]:
    # [FUNCTION DOC] Generator yielding lists of mutually independent nodes (none depends on another in the same list),
    # preserving input order, from the candidate nodes. Repeats rounds over the leftover nodes. Lists are capped
    # at max_fuse_set_size and only yielded if at least min_fuse_set_size long (both from the search options).
    # Dependency sets are cached between rounds and invalidated when their members are yielded. The caller may
    # mutate only the returned nodes during iteration.
    """
    Yields a list of subsets of `node_list` where no element in the subset
    depends on any other element in the subset. This results in a set of
    independent nodes which can be fused together.

    The order of `node_list` is preserved within each subset so we can benefit
    from split-cat elimination in later passes.

    During iteration it is only safe to mutate the graph by changing the nodes
    that have been returned.

    graph_search_options:
      - min_fuse_set_size: Minimum size of the subset to consider. Subsets below
        this size will be ignored.
      - max_fuse_set_size: Maximum size of the subset to consider. Subsets will
        be broken to be at most this size.
    """

    # Compute all the children of `node` which are members of
    # `interesting_nodes`.
    def find_dependent_nodes(node, interesting_nodes):
        # [FUNCTION DOC] HELPER: Return the set of interesting nodes that node transitively depends on (walk over input nodes).
        visited_node_set = OrderedSet[torch.fx.Node]()
        dep_set = OrderedSet[torch.fx.Node]()

        work = [node]
        while work:
            node = work.pop()
            for input_node in node.all_input_nodes:
                if input_node in interesting_nodes:
                    dep_set.add(input_node)

                if input_node not in visited_node_set:
                    visited_node_set.add(input_node)
                    work.append(input_node)

        return dep_set

    min_fuse_set_size = graph_search_options["min_fuse_set_size"]
    max_fuse_set_size = graph_search_options["max_fuse_set_size"]

    # node_list needs to be a set because we only track the nodes that are left
    # in it (and we want to do the `in` on a set, not a list). But we want to
    # keep the correct order.
    node_list = _OrderedSet(node_list)

    cache: dict[torch.fx.Node, OrderedSet[torch.fx.Node]] = {}
    while node_list:
        subset: list[torch.fx.Node] = []
        subset_deps = OrderedSet[torch.fx.Node]()

        next_round_node_list = _OrderedSet()
        for node in node_list:
            if len(subset) >= max_fuse_set_size or node in subset_deps:
                next_round_node_list.append(node)
                continue

            dep_set = cache.pop(node, None)
            if dep_set is None:
                dep_set = find_dependent_nodes(node, node_list)

            if not dep_set.intersection(subset):
                subset.append(node)
                subset_deps.update(dep_set)
            else:
                next_round_node_list.append(node)
                cache[node] = dep_set

        if len(subset) >= min_fuse_set_size:
            # Careful here - the caller uses the subsets to fuse nodes together
            # so we need to clear any cache entry that contains one of the
            # returned nodes because the dependency list could be different
            # (larger) after the merge.
            cache = {k: v for k, v in cache.items() if v.isdisjoint(subset)}
            yield subset

        node_list = next_round_node_list


def get_fusion_candidates(
    rule: GroupBatchFusionBase,
    root_node: torch.fx.Node,
    fused_set: OrderedSet[torch.fx.Node],
) -> collections.defaultdict[Any, list[torch.fx.Node]]:
    # [FUNCTION DOC] Breadth-first search backward from root_node's inputs, up to the max_fuse_search_depth option, asking
    # the rule to match each node; returns a mapping from group key to the list of matching nodes. Nodes already
    # fused are skipped, matching nodes are not expanded further, and a root in SEARCH_EXCLUSIONS (getitem) yields an empty mapping.
    """
    Search fusion candidates for a specific rule using BFS starting from the root node.
    We only search the subgraph within graph_search_options["max_fuse_search_depth"].
    """
    q: collections.deque[tuple[int, torch.fx.Node]] = collections.deque()

    candidate_dict: collections.defaultdict[Any, list[torch.fx.Node]] = (
        collections.defaultdict(list)
    )

    if root_node.target in SEARCH_EXCLUSIONS:
        return candidate_dict

    visited_set = OrderedSet[torch.fx.Node]()

    for next_node in root_node.all_input_nodes:
        q.append((1, next_node))
        visited_set.add(next_node)

    while len(q) > 0:
        depth, node = q.popleft()

        if node in fused_set:
            continue

        key = rule.match(node)
        if key is not None:
            candidate_nodes = candidate_dict[key]
            if node not in candidate_nodes:
                candidate_nodes.append(node)
        else:
            if depth < rule.graph_search_options["max_fuse_search_depth"]:
                for next_node in node.all_input_nodes:
                    if next_node not in visited_set:
                        visited_set.add(next_node)
                        q.append((depth + 1, next_node))

    return candidate_dict


def apply_group_batch_fusion(graph: torch.fx.GraphModule, rule: GroupBatchFusionBase):
    # [FUNCTION DOC] Apply one fusion rule to the whole graph: sort topologically, walk nodes in reverse, collect candidates
    # per root, and for each group of at least min_fuse_set_size nodes fuse each independent subset, recording fused nodes.
    # If anything fused, recompile the owning module and emit a structured trace artifact with the readable graph
    # (named optimus_<RuleClass>, or cff_ prefix for MTIA rules). Mutates the graph.
    stable_topological_sort(graph)  # type: ignore[arg-type]
    fused_set = OrderedSet[torch.fx.Node]()
    log_to_scuba = False

    for node in reversed(graph.nodes):  # type: ignore[arg-type]
        candidates = get_fusion_candidates(rule, node, fused_set)

        for key, candidate_nodes in candidates.items():
            if len(candidate_nodes) < rule.graph_search_options["min_fuse_set_size"]:
                continue

            for subset in find_independent_subset_greedy(
                candidate_nodes, rule.graph_search_options
            ):
                rule.fuse(graph, subset)
                fused_set.update(subset)
                log.debug(
                    f"{rule.__class__.__name__}: key = {key}; subset size = {len(list(subset))}"  # noqa: G004
                )
                log_to_scuba = True
    if log_to_scuba:
        from torch.fx._lazy_graph_module import _LazyGraphModule

        # Force graph to re-compile otherwise the output python code may be broken
        gm = graph._owning_module
        if isinstance(gm, _LazyGraphModule):
            _LazyGraphModule.recompile()
        else:
            if not isinstance(gm, torch.fx.GraphModule):
                raise AssertionError(f"expected torch.fx.GraphModule, got {type(gm)}")
            gm.recompile()
        graph_str = gm.print_readable(
            print_output=False, include_stride=True, include_device=True
        )

        name = f"optimus_{str(rule.__class__.__name__)}"
        if "MTIA" in name:
            name = f"cff_{str(rule.__class__.__name__)}"
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": name,
                "encoding": "string",
            },
            payload_fn=lambda: graph_str,
        )


def generate_fusion_from_config(config_options: dict[str, Any], pre_grad=True):
    # [FUNCTION DOC] Instantiate fusion rules from a config mapping of name to options. Names not in the registries are
    # skipped (they belong to pattern_matcher passes). Options are merged over the global search options, ignoring a
    # devices key. Returns the list of rule instances.
    # KNOWN ISSUE: a name registered only in the other registry (pre vs post grad) raises KeyError.
    fusions: list[GroupBatchFusionBase] = []
    for name, options in config_options.items():
        # we skip all patterns from pattern_matcher passes (e.g., split_cat)
        if name not in PRE_GRAD_FUSIONS and name not in POST_GRAD_FUSIONS:
            continue
        fusion_cls = PRE_GRAD_FUSIONS[name] if pre_grad else POST_GRAD_FUSIONS[name]
        _options = graph_search_options.copy()
        _options.update({k: v for k, v in options.items() if k != "devices"})
        fusions.append(fusion_cls(graph_search_options=_options))  # type: ignore[operator]
    return fusions


def group_batch_fusion_passes(
    graph: torch.fx.Graph, pre_grad=True, fusion_options=None
):
    # [FUNCTION DOC] Driver: build fusion rules and apply each in order via apply_group_batch_fusion, each under a
    # graph transform observer. Pre-grad uses fusion_options or the pre_grad_fusion_options config. Post-grad
    # splits post_grad_fusion_options into rules that require fbgemm (applied only if fbgemm is present, and
    # not excluded) and the rest. Mutates the graph; returns nothing.
    fusions: list[GroupBatchFusionBase] = []
    # we keep all current pre grad fusions to keep
    # current implementation, will remove this later
    if pre_grad:
        fusions += generate_fusion_from_config(
            fusion_options
            if fusion_options is not None
            else config.pre_grad_fusion_options,
            pre_grad=True,
        )
    else:
        fbgemm_fusion_keys = [
            x
            for x in config.post_grad_fusion_options
            if (
                x not in OPTIMUS_EXCLUDE_POST_GRAD
                and config.post_grad_fusion_options[x].get("require_fbgemm", False)
            )
        ]
        fbgemm_fusions = {
            fusion: config.post_grad_fusion_options[fusion]
            for fusion in fbgemm_fusion_keys
        }
        non_fbgemm_fusions = {
            fusion: config.post_grad_fusion_options[fusion]
            for fusion in config.post_grad_fusion_options
            if fusion not in fbgemm_fusion_keys
        }
        fusions += generate_fusion_from_config(non_fbgemm_fusions, pre_grad=False)
        if has_fbgemm:
            fusions += generate_fusion_from_config(fbgemm_fusions, pre_grad=False)

    for i, rule in enumerate(fusions):
        with GraphTransformObserver(
            graph.owning_module,  # pyrefly: ignore[bad-argument-type]
            f"group_batch_fusion_{i}",
        ):
            apply_group_batch_fusion(graph, rule)  # type: ignore[arg-type]
