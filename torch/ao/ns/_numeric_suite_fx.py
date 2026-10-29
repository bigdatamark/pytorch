# mypy: allow-untyped-defs
# [MODULE DOC] FX graph mode Numeric Suite public API: compares weights and activations of two
# FX-traceable models (typically float vs quantized). Offers weight extraction, adding
# loggers to both models (with error propagation), shadow-logger models (without error
# propagation), extraction of logger results into the NSResultsType mapping, comparison
# helpers, and the n-shadows workflow (multiple qconfigs shadowing each subgraph).
# Defines OutputLogger, OutputComparisonLogger and NSTracer. No import-time side effects
# beyond imports and defining the RNNReturnType alias.
"""
This module contains tooling to compare weights and activations
across models. Example usage::

    import copy
    import torch
    import torch.ao.quantization.quantize_fx as quantize_fx
    import torch.ao.ns._numeric_suite_fx as ns

    m = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1)).eval()
    mp = quantize_fx.prepare_fx(m, {"": torch.ao.quantization.default_qconfig})
    # We convert a copy because we need the original prepared model
    # to be available for comparisons, and `quantize_fx.convert_fx` is inplace.
    mq = quantize_fx.convert_fx(copy.deepcopy(mp))

    #
    # Comparing weights
    #

    # extract weight pairs
    weight_comparison = ns.extract_weights("a", mp, "b", mq)

    # add SQNR for each comparison, inplace
    ns.extend_logger_results_with_comparison(
        weight_comparison, "a", "b", torch.ao.ns.fx.utils.compute_sqnr, "sqnr"
    )

    # weight_comparison contains the weights from `mp` and `mq` stored
    # in pairs, and can be used for further analysis.


    #
    # Comparing activations, with error propagation
    #

    # add loggers
    mp_ns, mq_ns = ns.add_loggers(
        "a", copy.deepcopy(mp), "b", copy.deepcopy(mq), ns.OutputLogger
    )

    # send an example datum to capture intermediate activations
    datum = torch.randn(1, 1, 1, 1)
    mp_ns(datum)
    mq_ns(datum)

    # extract intermediate activations
    act_comparison = ns.extract_logger_info(mp_ns, mq_ns, ns.OutputLogger, "b")

    # add SQNR for each comparison, inplace
    ns.extend_logger_results_with_comparison(
        act_comparison, "a", "b", torch.ao.ns.fx.utils.compute_sqnr, "sqnr"
    )

    # act_comparison contains the activations from `mp_ns` and `mq_ns` stored
    # in pairs, and can be used for further analysis.

    #
    # Comparing activations, without error propagation
    #

    # create shadow model
    mp_shadows_mq = ns.add_shadow_loggers(
        "a", copy.deepcopy(mp), "b", copy.deepcopy(mq), ns.OutputLogger
    )

    # send an example datum to capture intermediate activations
    datum = torch.randn(1, 1, 1, 1)
    mp_shadows_mq(datum)

    # extract intermediate activations
    shadow_act_comparison = ns.extract_shadow_logger_info(
        mp_shadows_mq, ns.OutputLogger, "b"
    )

    # add SQNR for each comparison, inplace
    ns.extend_logger_results_with_comparison(
        shadow_act_comparison, "a", "b", torch.ao.ns.fx.utils.compute_sqnr, "sqnr"
    )

    # shadow_act_comparison contains the activations from `mp_ns` and `mq_ns` stored
    # in pairs, and can be used for further analysis.

"""

import collections
from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import torch
import torch.ao.quantization.quantize_fx as quantize_fx
import torch.nn as nn
from torch.ao.ns.fx.graph_matcher import get_matching_subgraph_pairs
from torch.ao.ns.fx.mappings import get_base_name_to_sets_of_related_ops
from torch.ao.ns.fx.n_shadows_utils import (
    _get_dedup_subgraphs,
    create_add_loggers_graph,
    create_n_transformed_and_logged_copies_of_subgraph,
    create_results_comparison,
    extract_weight_comparison,
    group_results_by_subgraph,
    OutputProp,
    print_n_shadows_summary,
    SHADOW_WRAPPER_NODE_NAME_PREFIX,
)
from torch.ao.ns.fx.qconfig_multi_mapping import QConfigMultiMapping
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.backend_config.utils import (
    get_fusion_pattern_to_root_node_getter,
)
from torch.ao.quantization.fx.graph_module import _get_observed_graph_module_attr
from torch.ao.quantization.fx.match_utils import _find_matches
from torch.ao.quantization.fx.qconfig_mapping_utils import (
    _generate_node_name_to_qconfig,
)
from torch.ao.quantization.fx.quantize_handler import _get_pattern_to_quantize_handlers
from torch.fx import GraphModule
from torch.fx.graph import Node

from .fx.graph_passes import add_loggers_to_model, create_a_shadows_b
from .fx.ns_types import NSNodeTargetType, NSResultsType, NSSingleResultValuesType
from .fx.utils import (
    get_target_type_str,
    maybe_add_missing_fqns,
    rekey_logger_info_on_node_name_of_model,
)
from .fx.weight_utils import extract_weight_from_node


if TYPE_CHECKING:
    from torch.ao.quantization.qconfig import QConfigAny

RNNReturnType = tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]


class OutputLogger(nn.Module):
    # [CLASS DOC] Module inserted into a graph to record the values flowing through it. It
    # keeps tensors in stats (and RNN-style tuples in stats_rnn) plus metadata describing
    # which node and model it logs (ref names, target types, results type, argument
    # indices, fqn, qconfig string). Flags: enabled (no recording when off) and
    # save_activations (skip storing to save memory). Marked impure so graph dead-code
    # elimination keeps it. forward returns its input unchanged.
    """
    Base class for capturing intermediate values.
    """

    stats: list[torch.Tensor]
    stats_rnn: list[RNNReturnType]

    # Mark as impure so that calls to it will not be removed during DCE.
    _is_impure = True

    def __init__(
        self,
        ref_node_name: str,
        prev_node_name: str,
        model_name: str,
        ref_name: str,
        prev_node_target_type: str,
        ref_node_target_type: str,
        results_type: str,
        index_within_arg: int,
        index_of_arg: int,
        fqn: str | None,
        qconfig_str: str | None = "",
    ):
        # [FUNCTION DOC] Stores the identifying metadata arguments as attributes, creates empty
        # stats and stats_rnn lists, and sets enabled and save_activations to true.
        super().__init__()
        self.stats: list[torch.Tensor] = []
        self.stats_rnn: list[RNNReturnType] = []

        # name of the node which was responsible for adding this logger
        # Note:
        # - if we are logging node outputs, this is the same as prev_node_name
        # - if we are logging node inputs, this is the name of the node
        #   whose input this logger is logging.
        #
        # example, where logger1 is logging input of op1 and logger2 is logging
        #    the output of op1:
        #
        #  x1 -> logger1 -> op1 -> logger2 -> x2
        #
        # in this example,
        #   - logger1's prev_node_name is x1 and ref_node_name is op1
        #   - logger2's prev_node_name is op1 and ref_node_name is op1
        self.ref_node_name = ref_node_name
        # name of the node whose output this Logger is capturing
        self.prev_node_name = prev_node_name

        # name of the model from which the node originated from
        self.model_name = model_name
        # reference name, used to match loggers from separate models
        # to each other
        self.ref_name = ref_name
        # type of the target of the node whose output this logger is logging
        self.prev_node_target_type = prev_node_target_type
        # type of the target of the node which was responsible for adding this
        # logger
        self.ref_node_target_type = ref_node_target_type
        # what kind of values are inside of stats
        self.results_type = results_type
        # index of this node within the arg of the input/output node
        # for example, in cat([x1, x2, x3], dim=0), x2 would have index_within_arg == 1
        self.index_within_arg = index_within_arg
        # index of this node within the args of the input/output node
        # for example, in add(x1, x2), x2 would have index_of_arg == 1
        self.index_of_arg = index_of_arg
        # fully qualified name
        self.fqn = fqn
        # if loggers are added before prepare_fx, but we do not want
        # collect results of calibration, only results after convert_fx
        # so, we add a flag to control whether this logger collects data
        self.enabled = True
        # string representation of qconfig
        self.qconfig_str = qconfig_str
        # this can be turned off to reduce memory usage during calibration
        self.save_activations = True

    # Note: cannot annotate the type of x because TorchScript does not support
    #   the Union type.
    def forward(self, x):
        # [FUNCTION DOC] Returns x unchanged. When enabled and saving activations, appends a
        # detached copy of a tensor to stats, or a detached copy of an RNN-style
        # (output, (hidden, cell)) tuple to stats_rnn. Other input kinds are not recorded.
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        # TODO(future PR): consider designing this better, as the difference
        # between these two flags is subtle and not obvious.
        if not self.enabled:
            return x
        if not self.save_activations:
            return x
        # TODO(future PR): consider refactoring this to better reuse the parent
        # class
        if isinstance(x, torch.Tensor):
            self.stats.append(x.detach())
        elif isinstance(x, tuple) and len(x) == 2 and len(x[1]) == 2:
            new_res = (x[0].detach(), (x[1][0].detach(), x[1][1].detach()))
            self.stats_rnn.append(new_res)
        return x

    def __repr__(self):
        # [FUNCTION DOC] Returns a string showing the public instance attributes (skipping
        # the training flag and underscore-prefixed names).
        clean_dict = {
            k: v
            for k, v in self.__dict__.items()
            # skip nn.Module keys
            if (k != "training") and not k.startswith("_")
        }
        return f"OutputLogger({clean_dict})"


class OutputComparisonLogger(OutputLogger):
    # [CLASS DOC] OutputLogger variant that also receives the reference activation and, at
    # each call, computes a comparison (SQNR by default) between its input and the reference,
    # appending results to the comparisons list.
    """
    Same as OutputLogger, but also requires the original activation
    in order to calculate the comparison at calibration time
    """

    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] Passes all arguments to OutputLogger and sets the comparison function
        # to the SQNR computation, its name to sqnr, and an empty comparisons list.
        super().__init__(*args, **kwargs)
        # TODO(future PR): make the comparison function configurable
        self.comparison_fn = torch.ao.ns.fx.utils.compute_sqnr
        self.comparison_fn_name = "sqnr"
        # precalculated comparisons of logger output versus reference
        self.comparisons = []
        # precalculated comparisons function

    def forward(self, x, x_ref):  # type: ignore[override]
        # [FUNCTION DOC] Returns x unchanged. If enabled: requires x to be a tensor (raises
        # AssertionError otherwise), stores a detached copy in stats when save_activations is
        # set, and appends the comparison of x against x_ref to comparisons. When disabled,
        # returns x without recording (and without checking the type).
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        if not self.enabled:
            return x
        if not isinstance(x, torch.Tensor):
            raise AssertionError("non-tensor inputs not yet supported")
        if self.save_activations:
            # save the activation, for debugging
            self.stats.append(x.detach())
        # save the comparison
        self.comparisons.append(self.comparison_fn(x, x_ref))
        return x

    def __repr__(self):
        # [FUNCTION DOC] Same as the base repr but labeled OutputComparisonLogger.
        clean_dict = {
            k: v
            for k, v in self.__dict__.items()
            # skip nn.Module keys
            if (k != "training") and not k.startswith("_")
        }
        return f"OutputComparisonLogger({clean_dict})"


class NSTracer(quantize_fx.QuantizationTracer):
    # [CLASS DOC] FX quantization tracer that additionally treats observer and fake-quantize
    # modules as leaf modules so they are kept as single call nodes.
    """
    Just like a regular FX quantization tracer, but treats observers and fake_quantize
    modules as leaf modules.
    """

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        # [FUNCTION DOC] Returns true for observers and fake-quantize modules; otherwise defers
        # to the parent tracer's decision.
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        if isinstance(m, torch.ao.quantization.ObserverBase):
            return True
        elif isinstance(m, torch.ao.quantization.FakeQuantizeBase):
            return True
        return super().is_leaf_module(m, module_qualified_name)


def _extract_weights_one_model(
    model_name: str,
    model: GraphModule,
    nodes_and_names_to_instrument: list[tuple[Node, str]],
    results: NSResultsType,
    op_to_type_to_weight_extraction_fn: dict[str, dict[Callable, Callable]]
    | None = None,
) -> None:
    # [FUNCTION DOC] HELPER: For each (node, reference name) pair, extracts the node's weight
    # (using the optional extraction-function override) and, if one exists, stores it as a
    # one-element list in the results mapping under reference name, weight result type and
    # model name. Mutates results.
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite_fx._extract_weights_one_model"
    )
    for node, ref_name in nodes_and_names_to_instrument:
        res_type = NSSingleResultValuesType.WEIGHT.value
        extracted_weight = extract_weight_from_node(
            node, model, op_to_type_to_weight_extraction_fn
        )
        if extracted_weight:
            if ref_name not in results:
                results[ref_name] = {res_type: {}}
            results[ref_name][res_type][model_name] = [extracted_weight]


def _extract_weights_impl(
    model_name_a: str,
    gm_a: GraphModule,
    model_name_b: str,
    gm_b: GraphModule,
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] | None = None,
    unmatchable_types_map: dict[str, set[NSNodeTargetType]] | None = None,
    op_to_type_to_weight_extraction_fn: dict[str, dict[Callable, Callable]]
    | None = None,
) -> NSResultsType:
    # [FUNCTION DOC] HELPER: Matches subgraphs between the two graph modules, extracts weights
    # at the base op node of each matched subgraph for both models, fills in missing fqns,
    # re-keys the results by node names of model B, and returns the results mapping.
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite_fx._extract_weights_impl"
    )
    matched_subgraph_pairs = get_matching_subgraph_pairs(
        gm_a, gm_b, base_name_to_sets_of_related_ops, unmatchable_types_map
    )

    # split the subgraph pairs into one data structure for each model
    nodes_and_names_to_instrument_a: list[tuple[Node, str]] = []
    nodes_and_names_to_instrument_b: list[tuple[Node, str]] = []
    for match_name, match in matched_subgraph_pairs.items():
        subgraph_a, subgraph_b = match
        nodes_and_names_to_instrument_a.append((subgraph_a.base_op_node, match_name))
        nodes_and_names_to_instrument_b.append((subgraph_b.base_op_node, match_name))

    # populate the results, one model at a time
    results: NSResultsType = {}
    _extract_weights_one_model(
        model_name_a,
        gm_a,
        nodes_and_names_to_instrument_a,
        results,
        op_to_type_to_weight_extraction_fn,
    )
    _extract_weights_one_model(
        model_name_b,
        gm_b,
        nodes_and_names_to_instrument_b,
        results,
        op_to_type_to_weight_extraction_fn,
    )

    # fill in missing fqn entries
    maybe_add_missing_fqns(results)

    # rekey on names of nodes in gm_b
    results = rekey_logger_info_on_node_name_of_model(results, model_name_b)

    return results


def extract_weights(
    model_name_a: str,
    model_a: nn.Module,
    model_name_b: str,
    model_b: nn.Module,
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] | None = None,
    unmatchable_types_map: dict[str, set[NSNodeTargetType]] | None = None,
    op_to_type_to_weight_extraction_fn: dict[str, dict[Callable, Callable]]
    | None = None,
) -> NSResultsType:
    # [FUNCTION DOC] Public entry point: traces both models into graph modules with NSTracer
    # (carrying over any recorded node-name-to-scope attribute), then extracts and pairs
    # weights of matching subgraphs. Optional arguments override the related-ops sets,
    # unmatchable types and weight extraction functions. Returns the results mapping.
    """
    Extract weights from model A and model B, and return a comparison.

    Args:
        model_name_a: string name of model A to use in results
        model_a: model A
        model_name_b: string name of model B to use in results
        model_b: model B
        base_name_to_sets_of_related_ops: optional override of subgraph base nodes, subject to change
        unmatchable_types_map: optional override of unmatchable types, subject to change
        op_to_type_to_weight_extraction_fn: optional override of function which extracts weight
            from a type, subject to change

    Return:
        NSResultsType, containing the weight comparisons
    """

    torch._C._log_api_usage_once("quantization_api._numeric_suite_fx.extract_weights")
    if base_name_to_sets_of_related_ops is None:
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()

    # TODO(future PR): expose these
    skipped_module_names: list[str] = []
    skipped_module_classes: list[Callable] = []
    tracer_a = NSTracer(skipped_module_names, skipped_module_classes)
    tracer_b = NSTracer(skipped_module_names, skipped_module_classes)
    gm_a = GraphModule(model_a, tracer_a.trace(model_a))
    maybe_model_a_node_name_to_scope = _get_observed_graph_module_attr(
        model_a, "node_name_to_scope"
    )
    if maybe_model_a_node_name_to_scope is not None:
        gm_a._node_name_to_scope = maybe_model_a_node_name_to_scope
    gm_b = GraphModule(model_b, tracer_b.trace(model_b))
    maybe_model_b_node_name_to_scope = _get_observed_graph_module_attr(
        model_b, "node_name_to_scope"
    )
    if maybe_model_b_node_name_to_scope is not None:
        gm_b._node_name_to_scope = maybe_model_b_node_name_to_scope
    return _extract_weights_impl(
        model_name_a,
        gm_a,
        model_name_b,
        gm_b,
        base_name_to_sets_of_related_ops,
        unmatchable_types_map,
        op_to_type_to_weight_extraction_fn,
    )


def _add_loggers_one_model(
    model_name: str,
    model: GraphModule,
    nodes_and_names_to_instrument_inputs: list[tuple[Node, str, str]],
    nodes_and_names_to_instrument_outputs: list[tuple[Node, str, str]],
    logger_cls: Callable,
) -> nn.Module:
    # [FUNCTION DOC] HELPER: Converts the input and output instrumentation lists (node,
    # reference name, node type) into mappings keyed by node and inserts loggers of
    # logger_cls into the model; returns the instrumented model.
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite_fx._add_loggers_one_model"
    )

    # TODO(future PR): do not observe nodes we do not care
    #   about (both fp32, denylist, etc)
    node_to_instrument_inputs_to_ref_name: dict[Node, tuple[str, str]] = {}
    node_to_instrument_outputs_to_ref_name: dict[Node, tuple[str, str]] = {}
    for node, ref_name, ref_node_type in nodes_and_names_to_instrument_inputs:
        node_to_instrument_inputs_to_ref_name[node] = (ref_name, ref_node_type)
    for node, ref_name, ref_node_type in nodes_and_names_to_instrument_outputs:
        node_to_instrument_outputs_to_ref_name[node] = (ref_name, ref_node_type)

    model = add_loggers_to_model(
        model,
        node_to_instrument_inputs_to_ref_name,
        node_to_instrument_outputs_to_ref_name,
        logger_cls,
        model_name,
    )
    return model


def _add_loggers_impl(
    name_a: str,
    gm_a: GraphModule,
    name_b: str,
    gm_b: GraphModule,
    logger_cls: Callable,
    should_log_inputs: bool,
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] | None = None,
    unmatchable_types_map: dict[str, set[NSNodeTargetType]] | None = None,
) -> tuple[nn.Module, nn.Module]:
    # [FUNCTION DOC] HELPER: Matches subgraphs between both graph modules and instruments each
    # matched pair: outputs are logged at each subgraph end node, inputs (only if
    # should_log_inputs) at each start node. Returns the two instrumented models.
    torch._C._log_api_usage_once("quantization_api._numeric_suite_fx._add_loggers_impl")
    matched_subgraph_pairs = get_matching_subgraph_pairs(
        gm_a, gm_b, base_name_to_sets_of_related_ops, unmatchable_types_map
    )
    nodes_and_names_to_instrument_inputs_a = []
    nodes_and_names_to_instrument_inputs_b = []
    nodes_and_names_to_instrument_outputs_a = []
    nodes_and_names_to_instrument_outputs_b = []
    for match_name, (subgraph_a, subgraph_b) in matched_subgraph_pairs.items():
        ref_node_type_a = get_target_type_str(subgraph_a.base_op_node, gm_a)
        ref_node_type_b = get_target_type_str(subgraph_b.base_op_node, gm_b)
        # Note: for matching inputs we use start_node, such as observing
        # the input of linear in linear-relu
        if should_log_inputs:
            nodes_and_names_to_instrument_inputs_a.append(
                (subgraph_a.start_node, match_name, ref_node_type_a)
            )
            nodes_and_names_to_instrument_inputs_b.append(
                (subgraph_b.start_node, match_name, ref_node_type_b)
            )
        # Note: for matching activations we always use end_node,
        # such as observing the output of relu in linear-relu
        nodes_and_names_to_instrument_outputs_a.append(
            (subgraph_a.end_node, match_name, ref_node_type_a)
        )
        nodes_and_names_to_instrument_outputs_b.append(
            (subgraph_b.end_node, match_name, ref_node_type_b)
        )

    new_model_a = _add_loggers_one_model(
        name_a,
        gm_a,
        nodes_and_names_to_instrument_inputs_a,
        nodes_and_names_to_instrument_outputs_a,
        logger_cls,
    )
    new_model_b = _add_loggers_one_model(
        name_b,
        gm_b,
        nodes_and_names_to_instrument_inputs_b,
        nodes_and_names_to_instrument_outputs_b,
        logger_cls,
    )
    return (new_model_a, new_model_b)


def add_loggers(
    name_a: str,
    model_a: nn.Module,
    name_b: str,
    model_b: nn.Module,
    logger_cls: Callable,
    should_log_inputs: bool = False,
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] | None = None,
    unmatchable_types_map: dict[str, set[NSNodeTargetType]] | None = None,
) -> tuple[nn.Module, nn.Module]:
    # [FUNCTION DOC] Public entry point: traces both models to graph modules with NSTracer and
    # adds loggers of logger_cls at matching locations (optionally also on inputs).
    # Modifies models in place and returns the two instrumented models.
    """
    Instrument model A and model B with loggers.

    Args:
        name_a: string name of model A to use in results
        model_a: model A
        name_b: string name of model B to use in results
        model_b: model B
        logger_cls: class of Logger to use
        base_name_to_sets_of_related_ops: optional override of subgraph base nodes, subject to change
        unmatchable_types_map: optional override of unmatchable types, subject to change

    Return:
        Returns a tuple of (model_a_with_loggers, model_b_with_loggers).  Modifies both models inplace.
    """

    torch._C._log_api_usage_once("quantization_api._numeric_suite_fx.add_loggers")
    # TODO(future PR): expose these
    skipped_module_names: list[str] = []
    skipped_module_classes: list[Callable] = []
    tracer_a = NSTracer(skipped_module_names, skipped_module_classes)
    tracer_b = NSTracer(skipped_module_names, skipped_module_classes)
    gm_a = GraphModule(model_a, tracer_a.trace(model_a))
    maybe_model_a_node_name_to_scope = _get_observed_graph_module_attr(
        model_a, "node_name_to_scope"
    )
    if maybe_model_a_node_name_to_scope is not None:
        gm_a._node_name_to_scope = maybe_model_a_node_name_to_scope
    gm_b = GraphModule(model_b, tracer_b.trace(model_b))
    maybe_model_b_node_name_to_scope = _get_observed_graph_module_attr(
        model_b, "node_name_to_scope"
    )
    if maybe_model_b_node_name_to_scope is not None:
        gm_b._node_name_to_scope = maybe_model_b_node_name_to_scope
    return _add_loggers_impl(
        name_a,
        gm_a,
        name_b,
        gm_b,
        logger_cls,
        should_log_inputs=should_log_inputs,
        base_name_to_sets_of_related_ops=base_name_to_sets_of_related_ops,
        unmatchable_types_map=unmatchable_types_map,
    )


def _extract_logger_info_one_model(
    model: nn.Module,
    results: NSResultsType,
    logger_cls: Callable,
) -> None:
    # [FUNCTION DOC] HELPER: Scans all submodules of model for loggers (instances of
    # logger_cls, or scripted modules originally named OutputLogger) and appends one record
    # per logger to results, nested by reference name, results type and model name. A record
    # holds the logged values (RNN stats if any, else plain stats), node names/types,
    # argument indices, fqn, qconfig string and comparisons (empty if the logger has none).
    # Each list is kept sorted by argument index. Raises AssertionError if the logger's model
    # name already appears directly under the reference name key. Mutates results.
    # KNOWN ISSUE: that duplicate check looks at the results-type level, not the model-name
    # level, so it only triggers for unusual result layouts.
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite_fx._extract_logger_info_one_model"
    )
    for _gm_name, mod in model.named_modules():
        # TODO(future PR): better check when scripted
        is_logger = isinstance(mod, logger_cls) or (  # type: ignore[arg-type]
            isinstance(mod, torch.jit.RecursiveScriptModule)
            and mod.original_name == "OutputLogger"
        )
        if is_logger:
            key = mod.ref_name
            if key not in results:
                results[key] = {}
            if mod.model_name in results[key]:
                raise AssertionError(f"{mod.model_name} is already present in results")
            if mod.results_type not in results[key]:
                results[key][mod.results_type] = {}
            if mod.model_name not in results[key][mod.results_type]:
                results[key][mod.results_type][mod.model_name] = []
            stats_to_use = mod.stats
            if len(mod.stats_rnn) > 0:
                stats_to_use = mod.stats_rnn
            data = {
                "type": mod.results_type,
                "values": stats_to_use,
                "ref_node_name": mod.ref_node_name,
                "ref_node_target_type": mod.ref_node_target_type,
                "prev_node_name": mod.prev_node_name,
                "prev_node_target_type": mod.prev_node_target_type,
                "index_within_arg": mod.index_within_arg,
                "index_of_arg": mod.index_of_arg,
                "fqn": mod.fqn,
                "qconfig_str": mod.qconfig_str,
            }
            if hasattr(mod, "comparisons"):
                data["comparisons"] = mod.comparisons
                data["comparison_fn_name"] = mod.comparison_fn_name
            else:
                data["comparisons"] = []
                data["comparison_fn_name"] = ""
            results[key][mod.results_type][mod.model_name].append(data)
            # ensure the list stays sorted
            results[key][mod.results_type][mod.model_name].sort(
                key=lambda res: f"{res['index_of_arg']}:{res['index_within_arg']}"
            )


# TODO(future PR): align on naming
# this is equivalent of just the comparison extraction part of `ns.compare_model_outputs`
def extract_logger_info(
    model_a: nn.Module,
    model_b: nn.Module,
    logger_cls: Callable,
    model_name_to_use_for_layer_names: str,
) -> NSResultsType:
    # [FUNCTION DOC] Collects logger records from both instrumented models into one results
    # mapping, fills in missing fqns, and re-keys entries by the node names of the model named
    # by model_name_to_use_for_layer_names. Returns the results mapping.
    """
    Traverse all loggers in `model_a` and `model_b`, and extract the logged
    information.

    Args:
        model_a: model A
        model_b: model B
        logger_cls: class of Logger to use
        model_name_to_use_for_layer_names: string name of model to use for
          layer names in the output

    Return:
        NSResultsType, containing the logged comparisons
    """
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite_fx.extract_logger_info"
    )
    results: NSResultsType = {}
    for model in (model_a, model_b):
        _extract_logger_info_one_model(model, results, logger_cls)
    # fill in missing fqn entries
    maybe_add_missing_fqns(results)
    # rekey on the name of model b
    results = rekey_logger_info_on_node_name_of_model(
        results, model_name_to_use_for_layer_names
    )
    return results


def _add_shadow_loggers_impl(
    name_a: str,
    gm_a: GraphModule,
    name_b: str,
    gm_b: GraphModule,
    logger_cls: Callable,
    should_log_inputs: bool,
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] | None = None,
    node_type_to_io_type_map: dict[str, set[NSNodeTargetType]] | None = None,
    unmatchable_types_map: dict[str, set[NSNodeTargetType]] | None = None,
) -> nn.Module:
    # [FUNCTION DOC] HELPER: Matches subgraphs between the two graph modules and builds a
    # single model in which each matched subgraph of B is added as a shadow of the matching
    # subgraph of A, with loggers; returns that combined graph module.
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite_fx._add_shadow_loggers_impl"
    )
    matched_subgraph_pairs = get_matching_subgraph_pairs(
        gm_a, gm_b, base_name_to_sets_of_related_ops, unmatchable_types_map
    )
    gm_a_shadows_b = create_a_shadows_b(
        name_a,
        gm_a,
        name_b,
        gm_b,
        matched_subgraph_pairs,
        logger_cls,
        should_log_inputs=should_log_inputs,
        node_type_to_io_type_map=node_type_to_io_type_map,
    )
    return gm_a_shadows_b


def add_shadow_loggers(
    name_a: str,
    model_a: nn.Module,
    name_b: str,
    model_b: nn.Module,
    logger_cls: Callable,
    should_log_inputs: bool = False,
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] | None = None,
    node_type_to_io_type_map: dict[str, set[NSNodeTargetType]] | None = None,
    unmatchable_types_map: dict[str, set[NSNodeTargetType]] | None = None,
) -> nn.Module:
    # [FUNCTION DOC] Public entry point: traces both models to graph modules with NSTracer and
    # returns one model that runs A with B's matching subgraphs shadowing it, with loggers
    # (optionally on inputs too). Inputs of B's subgraphs come from A's activations, so there
    # is no error propagation.
    """
    Instrument model A and model B with shadow loggers.

    Args:
        name_a: string name of model A to use in results
        model_a: model A
        name_b: string name of model B to use in results
        model_b: model B
        logger_cls: class of Logger to use
        should_log_inputs: whether to log inputs
        base_name_to_sets_of_related_ops: optional override of subgraph base nodes, subject to change
        unmatchable_types_map: optional override of unmatchable types, subject to change
    """
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite_fx.add_shadow_loggers"
    )
    # TODO(future PR): expose these
    skipped_module_names: list[str] = []
    skipped_module_classes: list[Callable] = []
    tracer_a = NSTracer(skipped_module_names, skipped_module_classes)
    tracer_b = NSTracer(skipped_module_names, skipped_module_classes)
    gm_a = GraphModule(model_a, tracer_a.trace(model_a))
    maybe_model_a_node_name_to_scope = _get_observed_graph_module_attr(
        model_a, "node_name_to_scope"
    )
    if maybe_model_a_node_name_to_scope is not None:
        gm_a._node_name_to_scope = maybe_model_a_node_name_to_scope
    gm_b = GraphModule(model_b, tracer_b.trace(model_b))
    maybe_model_b_node_name_to_scope = _get_observed_graph_module_attr(
        model_b, "node_name_to_scope"
    )
    if maybe_model_b_node_name_to_scope is not None:
        gm_b._node_name_to_scope = maybe_model_b_node_name_to_scope
    return _add_shadow_loggers_impl(
        name_a,
        gm_a,
        name_b,
        gm_b,
        logger_cls,
        should_log_inputs=should_log_inputs,
        base_name_to_sets_of_related_ops=base_name_to_sets_of_related_ops,
        node_type_to_io_type_map=node_type_to_io_type_map,
        unmatchable_types_map=unmatchable_types_map,
    )


def extract_shadow_logger_info(
    model_a_shadows_b: nn.Module,
    logger_cls: Callable,
    model_name_to_use_for_layer_names: str,
) -> NSResultsType:
    # [FUNCTION DOC] Extracts logger records from the shadow model into a results mapping,
    # fills in missing fqns, re-keys by node names of the named model, and returns a plain
    # mapping.
    """
    Traverse all loggers in a shadow model, and extract the logged
    information.

    Args:
        model_a_shadows_b: shadow model
        logger_cls: class of Logger to use
        model_name_to_use_for_layer_names: string name of model to use for
          layer names in the output

    Return:
        NSResultsType, containing the logged comparisons
    """
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite_fx.extract_shadow_logger_info"
    )
    results: NSResultsType = collections.defaultdict(dict)
    _extract_logger_info_one_model(model_a_shadows_b, results, logger_cls)
    # fill in missing fqn entries
    maybe_add_missing_fqns(results)
    # rekey on the name of model b
    results = rekey_logger_info_on_node_name_of_model(
        results, model_name_to_use_for_layer_names
    )
    return dict(results)


def extend_logger_results_with_comparison(
    results: NSResultsType,
    model_name_1: str,
    model_name_2: str,
    comparison_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    comparison_name: str,
) -> None:
    # [FUNCTION DOC] For every result group, pairs each record of model 2 with the record of
    # model 1 sharing the same argument indices, applies comparison_fn to each pair of logged
    # values, and stores the list of comparison results in the model 2 record under
    # comparison_name. Raises AssertionError if either model name is missing in a group or
    # no matching record of model 1 exists. Mutates results in place.
    """
    Compares the logged values from `model_name_2` against the corresponding
    values in `model_name_1`, using `comparison_fn`. Records the result
    in `model_name_2`'s results under `comparison_name`. Modifies `results` inplace.

    Args:
        results: the result data structure from `extract_logger_info` or
          `extract_shadow_logger_info`.
        model_name_1: string name of model 1
        model_name_2: string name of model 2
        comparison_fn: function to compare two Tensors
        comparison_name: string name of model to use for
          layer names in the output
    """
    for results_type_to_results in results.values():
        for model_name_to_results in results_type_to_results.values():
            if model_name_1 not in model_name_to_results:
                raise AssertionError(f"{model_name_1} not found in results")
            if model_name_2 not in model_name_to_results:
                raise AssertionError(f"{model_name_2} not found in results")

            results_1 = model_name_to_results[model_name_1]
            results_2 = model_name_to_results[model_name_2]

            for result_2 in results_2:
                index_within_arg_2 = result_2["index_within_arg"]
                index_of_arg_2 = result_2["index_of_arg"]
                # find corresponding result_1
                result_1 = None
                for cur_result_1 in results_1:
                    index_within_arg_1 = cur_result_1["index_within_arg"]
                    index_of_arg_1 = cur_result_1["index_of_arg"]
                    if (index_within_arg_1 == index_within_arg_2) and (
                        index_of_arg_1 == index_of_arg_2
                    ):
                        result_1 = cur_result_1
                        break
                if result_1 is None:
                    raise AssertionError("Expected result_1 to be not None")

                values_1 = result_1["values"]
                values_2 = result_2["values"]
                result_2[comparison_name] = []
                for value_1, value_2 in zip(values_1, values_2):
                    comparison_result = comparison_fn(value_1, value_2)
                    result_2[comparison_name].append(comparison_result)


def prepare_n_shadows_model(
    model: torch.nn.Module,
    example_inputs: Any,
    qconfig_multi_mapping: QConfigMultiMapping,
    backend_config: BackendConfig,
    custom_prepare_fn: Callable | None = None,
    custom_prepare_kwargs: dict[str, Any] | None = None,
    custom_tracer: Any = None,
) -> GraphModule:
    # [FUNCTION DOC] Traces the model (with the custom tracer if given), propagates example
    # inputs to record shapes/values, finds fusion/quantization pattern matches using the
    # backend config, deduplicates them into subgraphs, computes a node-to-qconfig mapping per
    # qconfig mapping, and for every subgraph inserts N transformed (prepared) copies with
    # loggers. Returns the new graph module; the original model is traced, not copied.
    """
    Given a model with a graph with M ops such as


      args_kwargs_m -> op_m -> output_m


    And a set of N qconfigs for each op, creates a new model, with
    each of the subgraph of `op_m` transformed into

    .. code::

           |---------> op_m_n -> log_m_n
           |                     /
      args_kwargs_m ---------> op_m -> log_m_0

    Where op_m_n is op_m wrapped in a submodule and transformed with
    qconfig_n, and its inner graph looks like

    .. code::

      args_m -------- op_m_prepared_with_qconfig_n -> out_m_n
                  /
      kwargs_m ---

    This is useful for testing different quantization of multiple layers in
    a single pass through the model.

    High level TODOs for future PRs:
    * figure out a better way to name the output structure
    * return a results data structure instead of printing it out
    * add examples to docblocks
    """

    if custom_tracer is None:
        tracer = quantize_fx.QuantizationTracer([], [])
    else:
        tracer = custom_tracer
    mt = torch.fx.GraphModule(model, tracer.trace(model))
    # this is necessary to ensure logger FQNs get populated
    mt._node_name_to_scope = tracer.node_name_to_scope  # type: ignore[assignment]

    # run example input propagation, we need this to call prepare_fx on
    # individual subgraphs
    output_prop = OutputProp(mt)
    output_prop.propagate(*example_inputs)

    # Find the set of subgraphs in the original graph which we need to
    # consider.
    modules = dict(mt.named_modules(remove_duplicate=False))
    patterns = _get_pattern_to_quantize_handlers(backend_config)
    root_node_getter_mapping = get_fusion_pattern_to_root_node_getter(backend_config)
    standalone_module_names: list[str] = []
    standalone_module_classes: list[type] = []
    custom_module_classes: list[type] = []
    matches = _find_matches(
        mt.graph,
        modules,
        patterns,
        root_node_getter_mapping,
        standalone_module_names,
        standalone_module_classes,
        custom_module_classes,
    )
    subgraphs_dedup: dict[str, list[Node]] = _get_dedup_subgraphs(matches)

    # generate node to qconfig for each subgraph
    # TODO(future PR): deduplicate repeating entries
    list_of_node_name_to_qconfig: list[dict[str, QConfigAny]] = []
    for qconfig_mapping in qconfig_multi_mapping.qconfig_mappings_list:
        node_name_to_qconfig = _generate_node_name_to_qconfig(
            mt, modules, mt.graph, qconfig_mapping, tracer.node_name_to_scope
        )
        list_of_node_name_to_qconfig.append(node_name_to_qconfig)

    # For each region in the model, do the following:
    #   For each qconfig for that region, do the following:
    #     1. create a copy of the region wrapped in a module
    #     2. pass original args, original kwargs, and expected output to module
    #     3. add an output comparison logger and hook it up to compare
    #        actual output to expected output
    #     4. run `prepare_fx` on the module
    for subgraph_idx, (match_name, nodes_in_this_subgraph) in enumerate(
        subgraphs_dedup.items()
    ):
        create_n_transformed_and_logged_copies_of_subgraph(
            mt,
            subgraph_idx,
            match_name,
            nodes_in_this_subgraph,
            qconfig_multi_mapping.qconfig_mappings_list,
            list_of_node_name_to_qconfig,
            custom_prepare_fn,
            custom_prepare_kwargs,  # type: ignore[arg-type]
        )

    return mt


# TODO(future PR): we should rethink the names of all the PNP APIs
def _prepare_n_shadows_add_loggers_model(
    model: torch.nn.Module,
    example_inputs: Any,
    qconfig_mapping: QConfigMapping,
    backend_config: BackendConfig,
) -> torch.nn.Module:
    # [FUNCTION DOC] HELPER: Traces the model, propagates example inputs, finds and
    # deduplicates matched subgraphs, computes the node-to-qconfig mapping for the single
    # qconfig mapping, then rewrites the graph into an add-loggers graph that compares a
    # quantized copy against the original at each subgraph. Returns the graph module.
    r"""
    Note: this API is not recommended for wide usage, it is only
    provided for customers who need to migrate from the `add_loggers`
    API.

    This creates a model which provides logging for the following
    problem: if we quantize `model` with `qconfig_mapping` and feed
    the same input through both models, log the comparisons of
    corresponding intermediate layers.

    The problem is solved with a single model.  Specifically, we
    partition `model` into N subgraphs, create a copy of each relevant
    subgraph, wrap it in a module, apply the quantization API to that
    module, and hook up loggers to measure the comparisons.

    Example starting graph:

      x0 -> op0 -> x1 -> op1 -> x2

    Example config: quantize op0 to int8, do nothing to op1.
    The following graph will be created:

    .. code::

      x0_0 -> op0_0 -> x1_0 -> log -----> op1_0 -> x2_0 -> log
       \                        \                           \       # noqa: W605
         ---> op0_1 -> x1_1 ----> clog -> op1_0 -> x2_1 ----> clog

    Where op0_0 is op0, op0_1 is op0 wrapped in a submodule and quantized
    to int8, op1_0 is op1 (appearing in the graph twice), log is a logger,
    and clog is a comparison logger.
    """

    tracer = quantize_fx.QuantizationTracer([], [])
    mt = torch.fx.GraphModule(model, tracer.trace(model))
    # this is necessary to ensure logger FQNs get populated
    mt._node_name_to_scope = tracer.node_name_to_scope  # type: ignore[assignment]

    # run example input propagation, we need this to call prepare_fx on
    # individual subgraphs
    output_prop = OutputProp(mt)
    output_prop.propagate(*example_inputs)

    # Find the set of subgraphs in the original graph which we need to
    # consider.
    modules = dict(mt.named_modules(remove_duplicate=False))
    patterns = _get_pattern_to_quantize_handlers(backend_config)
    root_node_getter_mapping = get_fusion_pattern_to_root_node_getter(backend_config)
    standalone_module_names: list[str] = []
    standalone_module_classes: list[type] = []
    custom_module_classes: list[type] = []
    matches = _find_matches(
        mt.graph,
        modules,
        patterns,
        root_node_getter_mapping,
        standalone_module_names,
        standalone_module_classes,
        custom_module_classes,
    )
    subgraphs_dedup: dict[str, list[Node]] = _get_dedup_subgraphs(matches)

    # generate node to qconfig for each subgraph
    node_name_to_qconfig = _generate_node_name_to_qconfig(
        mt, modules, mt.graph, qconfig_mapping, tracer.node_name_to_scope
    )

    # Now, mutate the graph to be the add_loggers graph with propagation
    # error.
    create_add_loggers_graph(mt, subgraphs_dedup, qconfig_mapping, node_name_to_qconfig)

    return mt


# TODO(future PR): we should rethink the names of all the PNP APIs
def _n_shadows_compare_weights(
    model: torch.nn.Module,
    example_inputs: Any,
    qconfig_mapping: QConfigMapping,
    backend_config: BackendConfig,
) -> NSResultsType:
    # [FUNCTION DOC] HELPER: Builds an n-shadows model from a single qconfig mapping, runs the
    # example inputs through it to populate weight observers, converts it, and returns the
    # extracted weight comparison.
    """
    Note: this API is not recommended for wide usage, it is only
    provided for customers who need to migrate from the `add_loggers`
    API.
    """
    qconfig_multi_mapping = QConfigMultiMapping.from_list_qconfig_mapping(
        [qconfig_mapping]
    )
    mp = prepare_n_shadows_model(
        model, example_inputs, qconfig_multi_mapping, backend_config
    )
    # passing inputs through the model is necessary to populate
    # observers which observe weights with real values
    mp(*example_inputs)
    mq = convert_n_shadows_model(mp)
    weight_comparison = extract_weight_comparison(mq)
    return weight_comparison


# TODO(future PR): consider aligning API signature with other similar quantization
# functions (enable_fake_quant, etc)
def loggers_set_enabled(model: torch.nn.Module, enabled: bool) -> None:
    # [FUNCTION DOC] Sets the enabled flag of every OutputLogger in the model in place.
    """
    Sets the `enabled` setting on a `model`'s loggers
    """
    for _, child in model.named_modules():
        if isinstance(child, OutputLogger):
            child.enabled = enabled


# TODO(future PR): consider aligning API signature with other similar quantization
# functions (enable_fake_quant, etc)
def loggers_set_save_activations(
    model: torch.nn.Module,
    save_activations: bool,
) -> None:
    # [FUNCTION DOC] Sets the save_activations flag of every OutputLogger in the model in place.
    """
    Sets the `save_activations` setting on a `model`'s loggers
    """
    for _name, child in model.named_modules():
        if isinstance(child, OutputLogger):
            child.save_activations = save_activations


def convert_n_shadows_model(
    model: GraphModule,
    custom_convert_fn: Callable | None = None,
    custom_convert_kwargs: dict[str, Any] | None = None,
) -> GraphModule:
    # [FUNCTION DOC] For each graph node whose name starts with the shadow-wrapper prefix,
    # converts the corresponding submodule with the default convert step (or the custom
    # convert function called with the optional keyword arguments) and replaces the attribute
    # on the model. Mutates and returns the model.
    """
    Given a model from `prepare_n_shadows_model`, runs `convert_fx`
    on each shadow submodule.
    """
    for node in model.graph.nodes:
        # TODO(future PR): consider matching in a safer way than
        # node name string match
        if node.name.startswith(SHADOW_WRAPPER_NODE_NAME_PREFIX):
            orig_mod = getattr(model, node.name)
            if custom_convert_fn is None:
                converted_mod = torch.ao.quantization.quantize_fx.convert_fx(orig_mod)
            else:
                if custom_convert_kwargs is None:
                    custom_convert_kwargs = {}
                converted_mod = custom_convert_fn(orig_mod, **custom_convert_kwargs)
            setattr(model, node.name, converted_mod)

    return model


def extract_results_n_shadows_model(model: torch.nn.Module) -> NSResultsType:
    # [FUNCTION DOC] Returns a new results mapping built from all OutputLogger modules in the model.
    """
    Extracts logger results from `model`.
    """
    results: NSResultsType = {}
    _extract_logger_info_one_model(model, results, OutputLogger)
    return results


def print_comparisons_n_shadows_model(results: NSResultsType) -> None:
    # [FUNCTION DOC] Groups results by subgraph, builds a comparison table, and prints a
    # summary to standard output. Returns None.
    """
    Prints a summary of extracted `results`.
    """
    results_grouped = group_results_by_subgraph(results)
    results_comparison = create_results_comparison(results_grouped)
    print_n_shadows_summary(results_comparison)
