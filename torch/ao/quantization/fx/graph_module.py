# mypy: allow-untyped-defs
# [MODULE DOC] GraphModule subclasses used by FX graph-mode quantization: FusedGraphModule,
# ObservedGraphModule, ObservedStandaloneGraphModule and QuantizedGraphModule. They keep a set of
# preserved attribute names across construction and deepcopy, and the quantized variant exposes
# packed weights through state_dict. Also helpers to detect observed modules.
import copy
from typing import Any

import torch
from torch.fx import GraphModule
from torch.fx.graph import Graph


__all__ = [
    "FusedGraphModule",
    "ObservedGraphModule",
    "ObservedStandaloneGraphModule",
    "QuantizedGraphModule",
]


class FusedGraphModule(GraphModule):
    # [CLASS DOC] GraphModule produced by the fusion pass. Remembers preserved_attr_names, the
    # attribute names copied from the root module onto the new instance (also on deepcopy).
    def __init__(
        self,
        root: torch.nn.Module | dict[str, Any],
        graph: Graph,
        preserved_attr_names: set[str],
    ):
        # [FUNCTION DOC] Store preserved_attr_names, snapshot those attributes that exist on root,
        # run GraphModule construction with root and graph, then set the snapshot attributes on self.
        self.preserved_attr_names = preserved_attr_names
        preserved_attrs = {
            attr: getattr(root, attr)
            for attr in self.preserved_attr_names
            if hasattr(root, attr)
        }
        super().__init__(root, graph)
        for attr in preserved_attrs:
            setattr(self, attr, preserved_attrs[attr])

    # GraphModule does not copy attributes which are not in the __dict__
    # of vanilla nn.Module.  So, we override __deepcopy__ in order
    # to copy the quantization specific attributes correctly.
    def __deepcopy__(self, memo):
        # [FUNCTION DOC] Deep copy: builds a plain Module holding a deep copy of this object's
        # attribute dictionary, then constructs a new FusedGraphModule from it, a deep copy of the graph
        # and a deep copy of preserved_attr_names, so quantization-specific attributes survive.
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        return FusedGraphModule(
            fake_mod,
            copy.deepcopy(self.graph),
            copy.deepcopy(self.preserved_attr_names),
        )


class ObservedGraphModule(GraphModule):
    # [CLASS DOC] GraphModule produced by the prepare step (observers inserted). Always preserves a
    # fixed set of prepare-time bookkeeping attributes (qconfig maps, patterns, scope map, etc.)
    # in addition to caller-provided names, in preserved_attr_names.
    def __init__(
        self,
        root: torch.nn.Module | dict[str, Any],
        graph: Graph,
        preserved_attr_names: set[str],
    ):
        # [FUNCTION DOC] Set preserved_attr_names to the built-in prepare-time attribute names unioned
        # with the given names, snapshot those present on root, run GraphModule construction, then
        # set the snapshot attributes on self.
        self.preserved_attr_names = {
            "_activation_post_process_map",
            "_activation_post_process_indexes",
            "_patterns",
            "_node_name_to_qconfig",
            "_prepare_custom_config",
            "_equalization_node_name_to_qconfig",
            "_node_name_to_scope",
            "_qconfig_mapping",
            "_is_qat",
            "_observed_node_names",
        }.union(preserved_attr_names)
        preserved_attrs = {
            attr: getattr(root, attr)
            for attr in self.preserved_attr_names
            if hasattr(root, attr)
        }
        super().__init__(root, graph)
        for attr in preserved_attrs:
            setattr(self, attr, preserved_attrs[attr])

    # GraphModule does not copy attributes which are not in the __dict__
    # of vanilla nn.Module.  So, we override __deepcopy__ in order
    # to copy the quantization specific attributes correctly.
    def __deepcopy__(self, memo):
        # [FUNCTION DOC] Deep copy: builds a plain Module holding a deep copy of this object's
        # attribute dictionary, then constructs a new ObservedGraphModule from it, a deep copy of the graph
        # and a deep copy of preserved_attr_names, so quantization-specific attributes survive.
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        return ObservedGraphModule(
            fake_mod,
            copy.deepcopy(self.graph),
            copy.deepcopy(self.preserved_attr_names),
        )


def _is_observed_module(module: Any) -> bool:
    # [FUNCTION DOC] HELPER: true if module has a meta mapping containing the observed graph
    # module attributes entry, false otherwise.
    return hasattr(module, "meta") and "_observed_graph_module_attrs" in module.meta


def _get_observed_graph_module_attr(
    model: torch.nn.Module | GraphModule, attr_name: str
) -> Any:
    # [FUNCTION DOC] HELPER: read attribute attr_name from the observed-graph-module attributes
    # stored in model.meta; returns None if the model has no such meta entry.
    if hasattr(model, "meta") and "_observed_graph_module_attrs" in model.meta:  # type: ignore[operator, index]
        return getattr(model.meta["_observed_graph_module_attrs"], attr_name)  # type: ignore[index]
    return None


class ObservedStandaloneGraphModule(ObservedGraphModule):
    # [CLASS DOC] Observed graph module for standalone submodules; additionally preserves the
    # input/output quantized index attributes of the standalone module.
    def __init__(
        self,
        root: torch.nn.Module | dict[str, Any],
        graph: Graph,
        preserved_attr_names: set[str],
    ):
        # [FUNCTION DOC] Add the standalone input/output quantized-index attribute names to the
        # preserved names and delegate to the parent constructor.
        preserved_attr_names = preserved_attr_names.union(
            {
                "_standalone_module_input_quantized_idxs",
                "_standalone_module_output_quantized_idxs",
            }
        )
        super().__init__(root, graph, preserved_attr_names)

    def __deepcopy__(self, memo):
        # [FUNCTION DOC] Deep copy: builds a plain Module holding a deep copy of this object's
        # attribute dictionary, then constructs a new ObservedStandaloneGraphModule from it, a deep copy of the graph
        # and a deep copy of preserved_attr_names, so quantization-specific attributes survive.
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        return ObservedStandaloneGraphModule(
            fake_mod,
            copy.deepcopy(self.graph),
            copy.deepcopy(self.preserved_attr_names),
        )


def _is_observed_standalone_module(module: Any) -> bool:
    # [FUNCTION DOC] HELPER: true if module is an observed module whose stored attributes flag it
    # as an observed standalone module.
    return (
        _is_observed_module(module)
        and module.meta["_observed_graph_module_attrs"].is_observed_standalone_module
    )


def _save_packed_weight(self, destination, prefix, keep_vars):
    # [FUNCTION DOC] HELPER: state_dict hook. For every attribute whose name contains the packed
    # weight marker and whose value is a script object, adds it to destination under the prefix
    # plus the attribute name (mutates destination). keep_vars is unused.
    for attr_name in dir(self):
        if "_packed_weight" in attr_name and isinstance(
            getattr(self, attr_name), torch._C.ScriptObject
        ):  # type: ignore[attr-defined]
            packed_weight = getattr(self, attr_name)
            destination[prefix + attr_name] = packed_weight


class QuantizedGraphModule(GraphModule):
    # [CLASS DOC] GraphModule for lowered quantized models. Registers a state_dict hook so packed
    # parameter script objects appear in state_dict, and restores them on load. Preserves
    # preserved_attr_names from the root module.
    """This class is created to make sure PackedParams
    (e.g. LinearPackedParams, Conv2dPackedParams) to appear in state_dict
    so that we can serialize and deserialize quantized graph module with
    torch.save(m.state_dict()) and m.load_state_dict(state_dict)
    """

    def __init__(
        self,
        root: torch.nn.Module | dict[str, Any],
        graph: Graph,
        preserved_attr_names: set[str],
    ):
        # [FUNCTION DOC] Store preserved_attr_names, snapshot those present on root, run GraphModule
        # construction, restore the snapshot attributes and register the packed-weight state_dict hook.
        self.preserved_attr_names = preserved_attr_names
        preserved_attrs = {
            attr: getattr(root, attr)
            for attr in self.preserved_attr_names
            if hasattr(root, attr)
        }
        super().__init__(root, graph)
        for attr in preserved_attrs:
            setattr(self, attr, preserved_attrs[attr])
        self._register_state_dict_hook(_save_packed_weight)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # [FUNCTION DOC] Load hook: entries in state_dict whose key starts with the packed weight
        # marker and whose value is a script object are set directly as attributes on self and
        # removed from state_dict (mutated); the remainder goes to the standard loading logic.
        attrs_to_pop = []
        for attr_name in state_dict:
            if attr_name.startswith("_packed_weight") and isinstance(
                state_dict[attr_name], torch._C.ScriptObject
            ):  # type: ignore[attr-defined]
                setattr(self, attr_name, state_dict[attr_name])
                attrs_to_pop.append(attr_name)

        # pop the packed param attributes
        for attr_name in attrs_to_pop:
            state_dict.pop(attr_name)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def __deepcopy__(self, memo):
        # [FUNCTION DOC] Deep copy: builds a plain Module holding a deep copy of this object's
        # attribute dictionary, then constructs a new QuantizedGraphModule from it, a deep copy of the graph
        # and a deep copy of preserved_attr_names, so quantization-specific attributes survive.
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        return QuantizedGraphModule(
            fake_mod,
            copy.deepcopy(self.graph),
            copy.deepcopy(self.preserved_attr_names),
        )
