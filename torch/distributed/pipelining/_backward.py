# mypy: allow-untyped-defs
# [MODULE DOC] Backward-pass helpers for pipeline stages. Provides autograd-graph analysis
# (reverse graph, reverse closure, parameter grouping) used to split backward into an
# input-gradient step and a weight-gradient step, plus stage_backward_input,
# stage_backward_weight, the combined stage_backward, and a null-tolerant accumulate helper.
# Import-time side effect: creates the module logger.
# Copyright (c) Meta Platforms, Inc. and affiliates
import collections
import logging
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch.autograd.graph import GradientEdge, Node
from torch.nn import Parameter

from ._debug import map_debug_info


logger = logging.getLogger(__name__)


def _get_grad_fn_or_grad_acc(t: torch.Tensor) -> Node | None:
    # [FUNCTION DOC] HELPER: Returns the autograd node for a tensor: its grad_fn, or for a leaf
    # requiring grad its grad accumulator (obtained through a temporary view because accumulators
    # are created lazily). Raises RuntimeError if a view of a grad-requiring leaf has no grad_fn
    # (for example under no-grad mode).
    """
    Get the grad function or grad accumulator for a tensor.

    Accumulate grad nodes are lazily created, so we need to a
    dummy view in order to trigger its creation.
    """
    if t.requires_grad and t.grad_fn is None:
        # if no grad function (leaf tensors) we use view
        viewed_t = t.view_as(t)
        grad_fn = viewed_t.grad_fn
        if grad_fn is not None:
            return grad_fn.next_functions[0][0]
        else:
            raise RuntimeError(
                "Attempted to get grad_fn, but got None."
                "Is this being created in a no-grad context?"
            )
    else:
        return t.grad_fn


def reverse_closure(
    roots: list[Node], target_nodes: set[Node], reverse_edges_dict
) -> tuple[set[Node], set[Node]]:
    # [FUNCTION DOC] Breadth-first walk over the reverse edges starting from the roots. Returns a
    # pair: the set of visited nodes, and the set of target nodes that were reached. Target nodes
    # are recorded but not expanded or added to the closure; None entries are skipped.
    """
    This function returns the reverse closure of the given roots,
    i.e. the set of nodes that can be reached from the roots by following the
    reverse edges of the graph. The target_nodes are the nodes that we want to
    include in the closure.
    """
    # Recurse until we reach a target node
    closure: set[Node] = set()
    visited_target_nodes = set()
    q: collections.deque[Node] = collections.deque()
    for node in roots:
        if node is not None and node not in closure:
            closure.add(node)
            q.append(node)
    while q:
        node = q.popleft()
        reverse_edges = reverse_edges_dict[node]
        for fn in reverse_edges:
            if fn in closure or fn is None:
                continue
            if fn in target_nodes:
                visited_target_nodes.add(fn)
                continue
            closure.add(fn)
            q.append(fn)
    return closure, visited_target_nodes


def construct_reverse_graph(roots: list[Node]) -> dict[Node, list[Node]]:
    # [FUNCTION DOC] Builds a mapping from each autograd node to the list of nodes that consume
    # it (reverse of next_functions) by walking the graph breadth-first from the roots. None roots
    # and None edges are skipped.
    q: collections.deque[Node] = collections.deque()
    root_seen: set[Node] = set()
    reverse_edges_dict: dict[Node, list[Node]] = collections.defaultdict(list)
    for node in roots:
        if node is not None and node not in root_seen:
            q.append(node)
            root_seen.add(node)
    while q:
        node = q.popleft()
        for fn, _ in node.next_functions:
            if fn is not None:
                if len(reverse_edges_dict[fn]) == 0:
                    q.append(fn)
                reverse_edges_dict[fn].append(node)
    return reverse_edges_dict


def get_param_groups(
    inputs: list[Node], params: list[Node], reverse_edges_dict
) -> list[dict[str, Any]]:
    # [FUNCTION DOC] Groups parameter gradient nodes by the intermediate nodes they connect to.
    # First computes the closure of the input nodes, then for each parameter finds which closure
    # nodes it reaches; parameters sharing an intermediate are merged into one group. Returns a
    # list of unique dictionaries with a set of params and a set of intermediates. Parameters that
    # reach no intermediate (e.g. when inputs do not require grad) are not in any group.
    """
    Given a list of inputs and a list of parameters, return a list of parameter
    groups, where each group contains the parameters and the intermediates that
    are connected to the parameters.

    The returned list of parameter groups is a list of dictionaries, where each
    dictionary contains the following keys:
    - "params": a set of parameters
    - "intermediates": a set of intermediates

    The returned list of parameter groups is a list of dictionaries,
    """
    # reverse graph that starts with inputs, and goes up to the dOutput or the loss,
    # but omits weights and any subgraphs connecting weights to this closure
    inputs_closure, _ = reverse_closure(inputs, set(), reverse_edges_dict)
    param_groups: dict[Node, dict[str, set]] = dict()  # keyed on intermediates
    for param in params:
        closure, intersected = reverse_closure(
            [param], inputs_closure, reverse_edges_dict
        )
        param_group: dict[str, set] = {
            "params": {param},
            "intermediates": intersected,
        }
        for input_node in intersected:
            existing = param_groups.get(input_node)
            if existing is not None:
                existing["params"] = existing["params"].union(param_group["params"])
                existing["intermediates"] = existing["intermediates"].union(
                    param_group["intermediates"]
                )
                param_group = existing
            else:
                param_groups[input_node] = param_group

    # Sanity check: union of all param_groups params should be equal to all params
    union_params: set[Node] = set()
    seen_ids: set[int] = set()
    unique_param_groups = []
    for param_group in param_groups.values():
        if id(param_group) not in seen_ids:
            seen_ids.add(id(param_group))
            unique_param_groups.append(param_group)
            union_params = union_params.union(param_group["params"])

    # The assert will only be true if the input tensor requires gradients,
    # otherwise the autograd graph will miss the first layer of inputs
    # assert union_params == set(params)
    return unique_param_groups


def _autograd_grad_for_inputs(
    outputs: Sequence[torch.Tensor],
    inputs: Sequence[Any],
    grad_outputs: Sequence[torch.Tensor | None] | None = None,
    retain_graph: bool = False,
    allow_unused: bool = False,
) -> tuple[torch.Tensor | None, ...]:
    # [FUNCTION DOC] HELPER: Computes gradients of outputs with respect to only those inputs that
    # are tensors requiring grad, calling autograd grad on that subset. Returns a tuple aligned
    # with the inputs, with None for the others (all None without calling autograd if none qualify).
    """Compute input gradients, returning ``None`` for non-grad inputs."""
    # Some inputs may not be used or may not require gradients, so we filter them out
    # before calling autograd.grad and place None for those positions in the result.
    grad_indices: list[int] = []
    inputs_requiring_grad: list[torch.Tensor] = []
    for i, inp in enumerate(inputs):
        if isinstance(inp, torch.Tensor) and inp.requires_grad:
            grad_indices.append(i)
            inputs_requiring_grad.append(inp)

    if not inputs_requiring_grad:
        return tuple(None for _ in inputs)

    grads = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs_requiring_grad,
        grad_outputs=grad_outputs,
        retain_graph=retain_graph,
        allow_unused=allow_unused,
    )

    result: list[torch.Tensor | None] = [None] * len(inputs)
    for idx, g in zip(grad_indices, grads, strict=True):
        result[idx] = g
    return tuple(result)


def stage_backward_input(
    stage_outputs_or_loss: list[torch.Tensor],
    output_grads: list[torch.Tensor] | None,
    input_values: list[Any],
    weights: Iterator[Parameter],
) -> tuple[tuple[torch.Tensor | None, ...], list[dict[str, Any]]]:
    # [FUNCTION DOC] Computes the gradients of the stage outputs (or loss) with respect to the stage
    # inputs only, for the zero-bubble input-backward step. Outputs that need no grad are
    # skipped. Registers prehooks on the intermediate nodes of each parameter group to save their
    # incoming gradients, runs autograd grad keeping the graph, accumulates results into the
    # inputs' grad fields, then detaches the outputs in place and removes the hooks. Returns the
    # input gradients and the parameter groups for stage_backward_weight. Raises RuntimeError
    # with debug info on any failure.
    """
    Compute the gradients for only the stage inputs with
    respect to the stage outputs (if non-last stage) or loss (if last stage)

    After computing input gradients, we save the intermediate nodes in `param_groups`
    for later use in stage_backward_weight. We don't need to save any other intermediate nodes
    that aren't needed for dW because when we do dW calculation, we start from saved intermediates.
    Detaching the stage_outputs_or_loss at the end of this function is important as
    it frees up the memory that the autograd graph is anticipating to be used later (but doesn't actually need).
    """
    valid_outputs: list[torch.Tensor] = []
    valid_output_grads: list[torch.Tensor | None] = []
    for i, stage_output in enumerate(stage_outputs_or_loss):
        if not stage_output.requires_grad and stage_output.grad_fn is None:
            continue
        valid_outputs.append(stage_output)
        valid_output_grads.append(
            torch.ones_like(stage_output) if output_grads is None else output_grads[i]
        )

    stage_output_grad_fns: list[Node] = list(
        filter(None, map(_get_grad_fn_or_grad_acc, valid_outputs))
    )
    stage_input_grad_fns: list[Node] = list(
        filter(
            None,
            (
                _get_grad_fn_or_grad_acc(inp)
                for inp in input_values
                if isinstance(inp, torch.Tensor)
            ),
        )
    )
    weight_grad_fns: list[Node] = list(
        filter(None, map(_get_grad_fn_or_grad_acc, weights))
    )

    reverse_edges_dict = construct_reverse_graph(stage_output_grad_fns)
    param_groups = get_param_groups(
        stage_input_grad_fns, weight_grad_fns, reverse_edges_dict
    )

    handles = []
    try:
        for param_group in param_groups:
            intermediates = list(param_group["intermediates"])
            param_group["intermediates"] = intermediates
            for i, intermediate in enumerate(intermediates):

                def get_hook(param_group, i):
                    # [FUNCTION DOC] HELPER: Returns a prehook closure bound to a parameter group and intermediate index.
                    def hook(grad_inputs):
                        # [FUNCTION DOC] HELPER: Prehook that stores the gradient inputs of the intermediate into the
                        # group's list of grads at the bound index, creating that list on first use.
                        if param_group.get("grads", None) is None:
                            param_group["grads"] = [None] * len(
                                param_group["intermediates"]
                            )
                        param_group["grads"][i] = grad_inputs

                    return hook

                handle = intermediate.register_prehook(get_hook(param_group, i))
                handles.append(handle)

        if valid_outputs:
            dinputs = _autograd_grad_for_inputs(
                valid_outputs,
                input_values,
                valid_output_grads,
                retain_graph=True,
                allow_unused=True,
            )
        else:
            dinputs = tuple(None for _ in input_values)

        for inp, dinput in zip(input_values, dinputs):
            if isinstance(inp, torch.Tensor) and dinput is not None:
                if inp.grad is None:
                    inp.grad = dinput
                else:
                    inp.grad += dinput

        # drop output side graph state we no longer need
        for t in stage_outputs_or_loss:
            t.detach_()

        return dinputs, param_groups
    except Exception as e:
        exc_msg = f"""
        Failed to run stage backward input:
        Stage output: {map_debug_info(stage_outputs_or_loss)}
        Output gradient: {map_debug_info(output_grads)}
        Input: {map_debug_info(input_values)}
        """
        raise RuntimeError(exc_msg) from e
    finally:
        for handle in handles:
            handle.remove()


def stage_backward_weight(
    weights: Iterator[Parameter], param_groups: list[dict[str, Any]], retain_graph=False
) -> tuple[torch.Tensor | None, ...]:
    # [FUNCTION DOC] Computes weight gradients from the state saved by stage_backward_input. For
    # each parameter group, builds gradient edges from the saved intermediate grads (clamping them
    # with temporary prehooks when a group has several intermediates), then runs autograd grad to
    # the group's params and accumulates results into each weight's grad field. Removes the
    # intermediates and grads entries from the groups, and removes the hooks. Returns a tuple of
    # the weights' pre-existing grads (captured before accumulation).
    # map weights to param_group_weights
    grad_acc_to_weight = {}
    weight_grads: list[torch.Tensor | None] = []
    for index, weight in enumerate(weights):
        grad_acc = _get_grad_fn_or_grad_acc(weight)
        grad_acc_to_weight[grad_acc] = weight, index
        weight_grads.append(weight.grad)

    for param_group in param_groups:
        valid_edges = []
        valid_grad_outputs: list[torch.Tensor] = []

        for grads_tuple, intermediate in zip(
            param_group["grads"], param_group["intermediates"]
        ):
            for i, grad in enumerate(grads_tuple):
                if grad is not None:
                    valid_edges.append(GradientEdge(intermediate, i))
                    # pyrefly: ignore [bad-argument-type]
                    valid_grad_outputs.append(grad)

        # clamp saved grads to avoid double counting through descendant intermediates.
        handles = []
        try:
            if len(param_group["intermediates"]) > 1:
                for grads_tuple, intermediate in zip(
                    param_group["grads"],
                    param_group["intermediates"],
                    strict=True,
                ):
                    handles.append(
                        intermediate.register_prehook(
                            lambda grad_outputs, sg=grads_tuple: sg
                        )
                    )

            # break the hook -> param_group -> intermediates cycle
            del param_group["intermediates"]

            if valid_edges:
                weights_edges = tuple(GradientEdge(w, 0) for w in param_group["params"])
                dweights = torch.autograd.grad(
                    valid_edges,
                    weights_edges,
                    grad_outputs=valid_grad_outputs,
                    retain_graph=retain_graph,
                )

                del param_group["grads"]

                for grad_acc, dw in zip(param_group["params"], dweights):
                    weight, index = grad_acc_to_weight[grad_acc]
                    if weight.grad is None:
                        weight.grad = dw
                    else:
                        weight.grad += dw
        finally:
            for handle in handles:
                handle.remove()
    return tuple(weight_grads)


def stage_backward(
    stage_output,
    output_grads,
    input_values,
    outputs_with_grads_idxs: list[int] | None = None,  # deprecated, not used
) -> tuple[torch.Tensor | None, ...]:
    # [FUNCTION DOC] Runs the full backward for a stage: flattens the (possibly nested tuple, list
    # or dict) output and its gradients into tensor lists, runs autograd backward on them which
    # accumulates parameter grads, and returns the input gradients as a tuple (None for non
    # tensors), clearing each input's grad field so memory can be freed early. The
    # outputs_with_grads_idxs argument is deprecated and only selects subsets of the outputs.
    # Raises RuntimeError with debug info on any failure.
    """
    This is a helper function to:
    1. compute the gradients for the stage inputs, and
    2. accumulate gradients for the stage module's parameters.

    Given the input value(s) and the corresponding gradient for the output
    value(s), compute and accumulate gradients for all parameter values (leaves
    in the autograd trace) as well as return a list of the gradients for the
    input values
    """
    if outputs_with_grads_idxs is not None:
        # Deprecated, not used in runtime calls, only exists in compiler
        stage_output = [stage_output[i] for i in outputs_with_grads_idxs]
        output_grads = [output_grads[i] for i in outputs_with_grads_idxs]

    try:
        # stage_output may be a composite datatype like dict. Extract all individual
        # tensor values here
        stage_output_tensors: list[torch.Tensor] = []
        output_grad_tensors: list[torch.Tensor | None] = []

        def extract_tensors_with_grads(
            output_val,
            grad_val,
            # Don't delete me- see [Note: ref cycle]
            extract_tensors_with_grads,
        ):
            # [FUNCTION DOC] HELPER: Recursively collects output tensors that need grad and their matching
            # gradients from nested tuples, lists and dicts, appending to the enclosing lists. It takes
            # itself as an argument to avoid a reference cycle. Raises AssertionError on mismatched
            # types, lengths or keys; non-tensor leaves are ignored.
            if isinstance(output_val, torch.Tensor):
                if not output_val.requires_grad and output_val.grad_fn is None:
                    return
                if not isinstance(grad_val, (torch.Tensor, type(None))):
                    raise AssertionError(
                        f"Expected Tensor or None gradient but got {type(grad_val)}"
                    )
                stage_output_tensors.append(output_val)
                output_grad_tensors.append(grad_val)
            elif isinstance(output_val, (tuple, list)):
                if grad_val is None:
                    return
                if not isinstance(grad_val, (tuple, list)):
                    raise AssertionError(
                        f"grad_value expected to have type {type(output_val)} but got {type(grad_val)}"
                    )
                if not len(output_val) == len(grad_val):
                    raise AssertionError(
                        f"Expected len(output_val) == len(grad_val), got {len(output_val)} != {len(grad_val)}"
                    )
                for ov, gv in zip(output_val, grad_val):
                    extract_tensors_with_grads(
                        ov,
                        gv,
                        extract_tensors_with_grads,
                    )
            elif isinstance(output_val, dict):
                if grad_val is None:
                    return
                if not isinstance(grad_val, dict):
                    raise AssertionError(f"Expected dict, got {type(grad_val)}")
                if not set(output_val.keys()) == set(grad_val.keys()):
                    raise AssertionError(
                        f"Expected keys {set(output_val.keys())}, got {set(grad_val.keys())}"
                    )
                for k in output_val:
                    extract_tensors_with_grads(
                        output_val[k], grad_val[k], extract_tensors_with_grads
                    )
            else:
                # Output is a non-tensor type; just ignore it
                pass

        # Note: ref cycle
        # break a ref cycle that would keep tensors alive until GC runs
        # 1. extract_tensors_with_grads refers to a cell that holds refs to any vars defined in stage_backward
        #    and used in extract_tensors_with_grads
        # 2. extract_tensors_with_grads referred to both stage_output_tensors, output_grad_tensors,
        #    and to itself (extract_tensors_with_grads) since it makes a recursive call
        # 3. stage_output_tensors was kept alive by the above refcycle, and it holds activation tensors, which is bad
        # fix -> explicitly pass in the ref to the fn, so there is no gc cycle anymore
        extract_tensors_with_grads(
            stage_output, output_grads, extract_tensors_with_grads
        )

        torch.autograd.backward(
            stage_output_tensors,
            grad_tensors=output_grad_tensors,  # type: ignore[arg-type]
        )

        # Extract gradients wrt the input values
        grad_inputs: list[torch.Tensor | None] = []
        for val in input_values:
            if isinstance(val, torch.Tensor):
                grad_inputs.append(val.grad)
                # Since gradients that will pass back to previous stages do not require gradient accumulation,
                # by decrementing the gradients' reference count at this point, the memory of gradients will be
                # returned to the allocator as soon as the next micro batch's get_bwd_send_ops comes and current
                # asynchronous send completes.
                # This prevents the gradients from persisting in GPU memory for the entire duration of step_microbatches
                # until clear_runtime_states() is called.
                val.grad = None
            else:
                grad_inputs.append(None)

        # Alternative impl: `torch.autograd.grad`.
        # Note that `torch.autograd.grad` will not accumulate gradients into the
        # model's parameters.
        """
        inputs_with_grad = []
        for val in input_values:
            if isinstance(val, torch.Tensor) and val.requires_grad:
                inputs_with_grad.append(val)

        grad_inputs = torch.autograd.grad(
            stage_output_tensors, inputs_with_grad, output_grad_tensors,  # type: ignore[arg-type]
        )
        """

    except Exception as e:
        exc_msg = f"""
        Failed to run stage backward:
        Stage output: {map_debug_info(stage_output)}
        Output gradient: {map_debug_info(output_grads)}
        Input: {map_debug_info(input_values)}
        """
        raise RuntimeError(exc_msg) from e

    return tuple(grad_inputs)


# TODO: handling requires_grad=False dynamically. Can we analyze this during initial
# IR emission?
def _null_coalesce_accumulate(lhs, rhs):
    # [FUNCTION DOC] HELPER: Returns the non-None one of two values, or their sum if both are present.
    """
    Coalesce two values, even if one of them is null, returning the non-null
    value.
    """
    if lhs is None:
        return rhs
    elif rhs is None:
        return lhs
    else:
        return torch.add(lhs, rhs)
