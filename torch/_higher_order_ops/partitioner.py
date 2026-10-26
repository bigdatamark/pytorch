# [MODULE DOC]
# Utilities to build a joint forward/backward graph for a higher order op body
# and split it into forward and backward graph modules using the min-cut
# rematerialization partitioner. Contains HopJointGraph (normalizes the joint
# graph), HopPartitionedGraph (validated partition result), the
# create_hop_joint_graph builder and the HopGraphMinCutPartitioner entry point.
# Import-time side effect: creates the module logger only.
import logging
from collections.abc import Callable
from typing import Any

import torch
from torch._higher_order_ops.utils import create_bw_fn, materialize_as_graph


logger: logging.Logger = logging.getLogger(__name__)


def _find_hop_subgraph_outputs(gm: torch.fx.GraphModule) -> tuple[torch.fx.Node]:
    # [FUNCTION DOC] HELPER: Returns the tuple of nodes returned by the graph's
    # (first) output node. Raises AssertionError if the output node's args are not
    # a tuple.
    output_node_args = gm.graph.find_nodes(op="output")[0].args
    if not isinstance(output_node_args, tuple):
        raise AssertionError(
            f"expected output_node_args to be tuple, got {type(output_node_args)}"
        )
    return output_node_args[0]


def is_complex_expr(expr: Any) -> bool:
    # [FUNCTION DOC] HELPER: Returns True if a symbolic expression is neither a
    # bare symbol nor a constant (i.e. a compound expression such as a sum).
    return not expr.is_symbol and not expr.is_constant()


class HopPartitionedGraph:
    # [CLASS DOC] Result of partitioning a HOP joint graph. Holds fw_gm (outputs:
    # original forward outputs followed by saved intermediates), bw_gm (inputs:
    # intermediates followed by output gradients), the count of forward outputs and
    # of intermediates, and a flag forbidding complex symbolic expressions at the
    # boundary. Construction reorders fw_gm outputs to match bw_gm inputs and then
    # validates the boundary, raising on mismatch.
    def __init__(
        self,
        fw_gm: torch.fx.GraphModule,
        bw_gm: torch.fx.GraphModule,
        n_fw_outputs: int,
        n_intermediates: int,
        no_complex_exprs_at_boundary: bool,
    ):
        # [FUNCTION DOC] Stores the two graph modules and counts, then normalizes
        # fw_gm's output order and checks the partition boundary. Raises
        # RuntimeError (from the check) if the partition is invalid. Mutates
        # fw_gm in place (output order, recompile).
        self.fw_gm = fw_gm
        self.bw_gm = bw_gm
        self.n_fw_outputs = n_fw_outputs
        self.n_intermediates = n_intermediates
        self.no_complex_exprs_at_boundary = no_complex_exprs_at_boundary
        self._reorder_fw_output()
        self._check_partition_boundary()

    def _check_partition_boundary(self) -> None:
        # [FUNCTION DOC] HELPER: Validates the partition. Collects reasons when: a fw
        # output lacks a val meta or is not a SymInt/Tensor; a SymInt output is a
        # complex expression while that is disallowed; the fw output count differs
        # from forward outputs plus intermediates; fw output count differs from bw
        # input count; fw intermediates differ from bw intermediates in name or
        # size/expression; or forward outputs differ from bw gradient inputs in
        # size/expression. Raises RuntimeError listing all reasons. No mutation.
        # KNOWN ISSUE: with zero forward outputs the slices of bw placeholders by
        # negative n_fw_outputs would select everything or nothing incorrectly.
        """check partitioned graph is in valid state."""
        invalid_reasons = []
        fw_outputs = _find_hop_subgraph_outputs(self.fw_gm)
        for i, out in enumerate(fw_outputs):
            if "val" not in out.meta:
                invalid_reasons.append(f"fw_gm output[{i}] doesn't have a 'val' meta.")
            elif not isinstance(out.meta["val"], (torch.SymInt, torch.Tensor)):
                invalid_reasons.append(
                    f"fw_gm output[{i}] is of type {type(out.meta['val'])} but only SymInt or Tensor are allowed."
                )

            elif (
                isinstance(out.meta["val"], torch.SymInt)
                and is_complex_expr(out.meta["val"].node.expr)
                and self.no_complex_exprs_at_boundary
            ):
                invalid_reasons.append(
                    f"fw_gm output[{i}] must be of type SymInt with basic symbols or "
                    f"Tensor but got {type(out.meta['val'])} {out.meta['val']}"
                )

        if len(fw_outputs) != self.n_fw_outputs + self.n_intermediates:
            invalid_reasons.append(
                f"len(fw_outputs) ({len(fw_outputs)}) != n_fw_outputs ({self.n_fw_outputs}) + n_intermediates ({self.n_intermediates})"
            )

        bw_phs = list(self.bw_gm.graph.find_nodes(op="placeholder"))

        if len(fw_outputs) != len(bw_phs):
            invalid_reasons.append(
                f"Expect number of fw_gm's output to be the same as bw_gm's input but "
                f"fw_gm has {len(fw_outputs)} outputs, bw_gm takes {len(bw_phs)} inputs."
            )

        original_forward_outputs = fw_outputs[: self.n_fw_outputs]
        fw_intermediates = fw_outputs[self.n_fw_outputs :]

        bw_intermediates = bw_phs[: -self.n_fw_outputs]
        bw_grads = bw_phs[-self.n_fw_outputs :]

        def _match_size_or_expr(
            val1: torch.SymInt | torch.Tensor,
            val2: torch.SymInt | torch.Tensor,
        ) -> bool:
            # [FUNCTION DOC] HELPER: True if both values are the same kind and
            # match: SymInts compare by symbolic expression, tensors by size.
            if type(val1) is not type(val2):
                return False

            if isinstance(val1, torch.SymInt) and isinstance(val2, torch.SymInt):
                return val1.node.expr == val2.node.expr
            elif isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                return val1.size() == val2.size()

            return False

        for fw, bw in zip(fw_intermediates, bw_intermediates):
            if fw.name != bw.name or not _match_size_or_expr(
                fw.meta["val"], bw.meta["val"]
            ):
                invalid_reasons.append("fw intermediates don't match bw intermediates")

        for fw_out, bw_grad in zip(original_forward_outputs, bw_grads):
            if not _match_size_or_expr(fw_out.meta["val"], bw_grad.meta["val"]):
                invalid_reasons.append("fw outputs don't match bw gradients")

        if len(invalid_reasons) > 0:
            newline = "\n"
            raise RuntimeError(
                f"Invalid HopPartitionedGraph. Reasons:\n{newline.join(invalid_reasons)}"
            )

    def _reorder_fw_output(self) -> None:
        # [FUNCTION DOC] HELPER: Reorders fw_gm's trailing intermediate outputs to
        # follow the order of the first n_intermediates bw_gm placeholders, matching
        # by node name. Rewrites the output node args, lints and recompiles fw_gm.
        # No-op when there are no intermediates. Raises KeyError if a bw name has
        # no matching fw intermediate.
        """
        Before the pass, fw_gm returns (*fw_outputs, *intermediates1)
        and bw_gm takes (*intermediates2, *grad_fw_outputs) as input.
        intermediates1 and intermediates2 share the same node names but
        they might be in different order. E.g. this could happen if there
        are inputs that contain symints.

        To simplify downstream processing, this graph pass normalizes the output of fw_gm
        to be consistent with the backward inputs:

        fw_gm:
          - input: fw_args
          - output: (*fw_outputs, *intermediates)

        bw_gm:
          - input: (*intermediates, *grad_fw_outputs)
          - output: grad_fw_args

        Example:

        def fw_gm(x, y, z):
           a, b, c = f(x), g(y), k(z)
           return a, b, c, f_tmp, g_tmp, k_tmp

        , where a, b, c are fw_outputs, f_tmp, g_tmp, k_tmp are intermediates

        The corresponding bw_gm has the following signature:

        def bw_gm(f_tmp, g_tmp, k_tmp, grad_a, grad_b, grac):
          return grad_x, grad_y, grad_z
        """
        fw_gm_output_nodes = _find_hop_subgraph_outputs(self.fw_gm)
        fw_outputs_nodes = fw_gm_output_nodes[: self.n_fw_outputs]
        fw_intermediates_nodes = fw_gm_output_nodes[self.n_fw_outputs :]
        if len(fw_intermediates_nodes) > 0:
            fw_intermediates_name_to_node = {n.name: n for n in fw_intermediates_nodes}

            # First n_intermediates placeholders
            bw_names: list[str] = [
                ph.name
                for ph in list(self.bw_gm.graph.find_nodes(op="placeholder"))[
                    : self.n_intermediates
                ]
            ]
            new_fw_outputs = list(fw_outputs_nodes) + [
                fw_intermediates_name_to_node[name] for name in bw_names
            ]

            output_node = self.fw_gm.graph.find_nodes(op="output")[0]
            output_node.args = (tuple(new_fw_outputs),)

            self.fw_gm.graph.lint()
            self.fw_gm.recompile()


class HopJointGraph:
    # [CLASS DOC] Wrapper around a joint forward+backward GraphModule. Attributes:
    # joint_gm, n_primals (leading placeholders that are primal inputs), n_fw_outputs,
    # functionalized flag, and n_tangents (set while renaming placeholders). Construction
    # renames placeholders and removes redundant sym_size nodes; partition() splits the
    # graph into a HopPartitionedGraph.
    def __init__(
        self,
        joint_gm: torch.fx.GraphModule,
        n_primals: int,
        n_fw_outputs: int,
        *,
        functionalized: bool,
    ):
        # [FUNCTION DOC] Stores the joint graph and counts, then renames
        # placeholders (primals_N / tangents_N) and removes redundant sym_size ops,
        # mutating joint_gm in place.
        self.joint_gm = joint_gm
        self.n_primals = n_primals
        self.n_fw_outputs = n_fw_outputs
        self.functionalized = functionalized

        self._rename_phs()
        self._remove_redundant_sym_size_ops()

    def _rename_phs(self) -> None:
        # [FUNCTION DOC] HELPER: Renames the first n_primals placeholders to
        # primals_i and the rest to tangents_j (target and name), counts tangents
        # into n_tangents, then lints and recompiles the graph module. Mutates
        # joint_gm and sets n_tangents.
        """
        Rename the placeholders for joint_gm so that the partitioner
        could recognize which inputs are primals and which are tangents.
        """
        self.n_tangents = 0
        for i, ph in enumerate(self.joint_gm.graph.find_nodes(op="placeholder")):
            if i < self.n_primals:
                ph.target = f"primals_{i}"
                ph.name = f"primals_{i}"
            else:
                self.n_tangents += 1
                ph.target = f"tangents_{i - self.n_primals}"
                ph.name = f"tangents_{i - self.n_primals}"

        self.joint_gm.graph.lint()
        self.joint_gm.compile()

    def _remove_redundant_sym_size_ops(self) -> None:
        # [FUNCTION DOC] HELPER: Maps symbolic expressions of SymInt placeholders to
        # their nodes; every sym_size.int call whose value expression equals one
        # of those is replaced by the placeholder and erased. Raises AssertionError
        # if a sym_size node lacks a val meta. Lints and recompiles joint_gm.
        """
        Deletes torch.ops.sym_size.int operators whose output is a
        corresponding placeholder that holds the same symbol, and replace all usage
        of the sym_size node to be directly using the placeholders.

        This is to make sure all basic symbols come from inputs.
        """
        placeholder_exprs = {}
        for node in self.joint_gm.graph.nodes:
            if (
                isinstance(node, torch.fx.Node)
                and node.op == "placeholder"
                and hasattr(node, "meta")
                and "val" in node.meta
            ):
                val = node.meta["val"]
                if isinstance(val, torch.SymInt):
                    placeholder_exprs[val.node.expr] = node

        nodes_to_remove = []
        for node in self.joint_gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.sym_size.int
        ):
            if not (hasattr(node, "meta") and "val" in node.meta):
                raise AssertionError(
                    f"node {node} must have 'meta' attribute with 'val' key"
                )
            val = node.meta["val"]
            expr = val.node.expr
            if expr in placeholder_exprs:
                placeholder_node = placeholder_exprs[expr]
                node.replace_all_uses_with(placeholder_node)
                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            self.joint_gm.graph.erase_node(node)

        self.joint_gm.graph.lint()
        self.joint_gm.recompile()

    def _mark_complex_exprs_as_must_recompute(self) -> None:
        # [FUNCTION DOC] HELPER: Sets the recompute meta of every call_function node
        # whose val is a SymInt with a complex expression to MUST_RECOMPUTE so such
        # values are not saved across the partition boundary. Raises AssertionError
        # if such a node already has a recompute policy. Lints and recompiles.
        """
        For control flow operators such as scan, we don't want to
        have symint in the partitioning boundaries because otherwise we would need to support stacking
        the symints up, which causes more entropy in the stack.

        By marking the recompute polify for complex nodes as MUST_RECOMPUTE, the partitioning boundary
        no longer contains complex expressions.

        Note that this pass doesn't exclude basic symbols from partitioning boundary
        and it's up to the downstream to decide whether to return the basic symbol
        or have a separate graph pass to remove them.
        """

        from torch._functorch.partitioners import CheckpointPolicy

        for n in (
            node for node in self.joint_gm.graph.nodes if node.op == "call_function"
        ):
            if "val" not in n.meta:
                continue
            val = n.meta["val"]
            if isinstance(val, torch.SymInt) and is_complex_expr(val.node.expr):
                if n.meta.get("recompute", None) is not None:
                    raise AssertionError(
                        f"node {n} with complex SymInt expression should not have recompute policy set"
                    )

                n.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE

        self.joint_gm.graph.lint()
        self.joint_gm.recompile()

    def partition(
        self, partition_fn: Callable, always_recompute_complex_exprs: bool
    ) -> HopPartitionedGraph:
        # [FUNCTION DOC] Splits the joint graph with partition_fn (called with the
        # joint graph, no profiling info, and the forward output count). When
        # always_recompute_complex_exprs is set, first marks complex SymInt nodes
        # as must-recompute. Computes the number of intermediates as the forward
        # graph output count minus forward outputs and returns a validated
        # HopPartitionedGraph. Logs graphs at debug level.
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "before min_cut_partition:\n%s",
                self.joint_gm.print_readable(print_output=False),
            )

        if always_recompute_complex_exprs:
            self._mark_complex_exprs_as_must_recompute()

        fw_gm, bw_gm = partition_fn(
            self.joint_gm, None, num_fwd_outputs=self.n_fw_outputs
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("after partition_fn:")
            logger.debug("fw_gm:\n%s", fw_gm.print_readable(print_output=False))
            logger.debug("bw_gm:\n%s", bw_gm.print_readable(print_output=False))

        n_intermediates = len(_find_hop_subgraph_outputs(fw_gm)) - self.n_fw_outputs

        return HopPartitionedGraph(
            fw_gm,
            bw_gm,
            self.n_fw_outputs,
            n_intermediates,
            always_recompute_complex_exprs,
        )


def create_hop_joint_graph(
    fw_fn: Callable,
    fw_args: tuple[torch.Tensor | torch.SymInt, ...],
    functionalize: bool,
) -> HopJointGraph:
    # [FUNCTION DOC] Traces fw_fn with fw_args (grad enabled), checks every output
    # is a node carrying a tensor val (AssertionError otherwise), builds zero
    # example gradients per output, traces the backward-with-forward-outputs
    # function on args plus gradients into a joint graph and, if requested,
    # functionalizes it (removing mutations and views) after tracing. Returns
    # a HopJointGraph with primals count equal to len(fw_args).
    fw_gm = materialize_as_graph(fw_fn, fw_args, force_enable_grad=True)
    fw_gm_output_nodes = _find_hop_subgraph_outputs(fw_gm)

    if not all(
        isinstance(n, torch.fx.Node) and "val" in n.meta for n in fw_gm_output_nodes
    ):
        raise AssertionError(
            "all fw_gm output nodes must be torch.fx.Node with 'val' in meta"
        )
    fw_gm_output_vals = tuple(n.meta["val"] for n in fw_gm_output_nodes)  # type: ignore[arg-type]

    if not all(isinstance(val, torch.Tensor) for val in fw_gm_output_vals):
        raise AssertionError(
            f"all fw_gm output values must be torch.Tensor, got {[type(v) for v in fw_gm_output_vals]}"
        )
    example_grads = tuple(torch.zeros_like(val) for val in fw_gm_output_vals)

    joint_fn = create_bw_fn(fw_fn, fw_args, return_fw_outputs=True)
    joint_gm = materialize_as_graph(
        joint_fn, fw_args + example_grads, force_enable_grad=True
    )
    if functionalize:
        # Need to first trace out the joint_fn with autograd info on
        # then functionalize the graph otherwise the grad information is lost
        joint_gm = materialize_as_graph(
            # pyrefly: ignore [bad-argument-type]
            torch.func.functionalize(joint_gm, remove="mutations_and_views"),
            fw_args + example_grads,
        )

    return HopJointGraph(
        joint_gm,
        len(fw_args),
        len(fw_gm_output_nodes),
        functionalized=functionalize,
    )


class HopGraphMinCutPartitioner:
    # [CLASS DOC] Stateless namespace exposing the min-cut based partitioning entry point.
    @staticmethod
    def create_partitioned_graph(
        fw_fn: Callable,
        fw_args: tuple[torch.Tensor | torch.SymInt, ...],
        *,
        always_recompute_complex_exprs: bool = False,
    ) -> HopPartitionedGraph:
        # [FUNCTION DOC] Builds a functionalized joint graph for fw_fn and fw_args
        # and partitions it with the min-cut rematerialization partitioner.
        # always_recompute_complex_exprs keyword forces complex symbolic
        # expressions to be recomputed in backward. Returns a HopPartitionedGraph.
        """
        Inputs:
            - fw_fn: the forward function that we'll use to create a joint graph and partition
            - fw_args: the flat_args to fw_fn
            - always_recompute_complex_exprs: when set to True, the bw_gm will do a re-compute
              for inputs that are complex expressions such that the partitioning boundary
              only consists of basic symbols and tensors.

        Returns a HopPartitionedGraph
        """
        from torch._functorch.partitioners import min_cut_rematerialization_partition

        joint_graph: HopJointGraph = create_hop_joint_graph(
            fw_fn, fw_args, functionalize=True
        )
        return joint_graph.partition(
            min_cut_rematerialization_partition, always_recompute_complex_exprs
        )
