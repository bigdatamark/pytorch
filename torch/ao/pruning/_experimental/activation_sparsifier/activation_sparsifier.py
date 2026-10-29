# mypy: allow-untyped-defs
# [MODULE DOC] Experimental activation sparsifier. Defines ActivationSparsifier, which attaches
# forward pre-hooks to layers of a model to aggregate their input activations over time, then
# derive and apply a sparsity mask from the aggregate via user supplied aggregate, reduce and
# mask functions. Supports per-feature masks and state save/restore. No import-time side effects.
import copy
import warnings
from collections import defaultdict
from typing import Any

import torch
from torch import nn
from torch.ao.pruning.sparsifier.utils import fqn_to_module, module_to_fqn


__all__ = ["ActivationSparsifier"]


class ActivationSparsifier:
    # [CLASS DOC] Manages activation sparsification for chosen layers of a model.
    # State: model; defaults (default functions, features, feature_dim and sparse_config);
    # data_groups (layer fqn -> per-layer config incl. layer, hook, hook_state, accumulated data);
    # state (layer fqn -> mask, a tensor, or a list of tensors when features are used, None until
    # the first forward). Workflow: register_layer, run forward passes, step to compute masks,
    # then squash_mask to swap the aggregate hook for a hook that multiplies inputs by the mask.
    r"""
    The Activation sparsifier class aims to sparsify/prune activations in a neural
    network. The idea is to attach the sparsifier to a layer (or layers) and it
    zeroes out the activations based on the mask_fn (or sparsification function)
    input by the user.
    The mask_fn is applied once all the inputs are aggregated and reduced i.e.
    mask = mask_fn(reduce_fn(aggregate_fn(activations)))

    Note::
        The sparsification mask is computed on the input **before it goes through the attached layer**.

    Args:
        model (nn.Module):
            The model whose layers will be sparsified. The layers that needs to be
            sparsified should be added separately using the register_layer() function
        aggregate_fn (Optional, Callable):
            default aggregate_fn that is used if not specified while registering the layer.
            specifies how inputs should be aggregated over time.
            The aggregate_fn should usually take 2 torch tensors and return the aggregated tensor.
            Example
                def add_agg_fn(tensor1, tensor2):  return tensor1 + tensor2
                reduce_fn (Optional, Callable):
                    default reduce_fn that is used if not specified while registering the layer.
                    reduce_fn will be called on the aggregated tensor i.e. the tensor obtained after
                    calling agg_fn() on all inputs.
                    Example
                def mean_reduce_fn(agg_tensor):    return agg_tensor.mean(dim=0)
                mask_fn (Optional, Callable):
                    default mask_fn that is used to create the sparsification mask using the tensor obtained after
                    calling the reduce_fn(). This is used by default if a custom one is passed in the
                    register_layer().
                    Note that the mask_fn() definition should contain the sparse arguments that is passed in sparse_config
                    arguments.
                features (Optional, list):
                    default selected features to sparsify.
                    If this is non-empty, then the mask_fn will be applied for each feature of the input.
                    For example,
                mask = [mask_fn(reduce_fn(aggregated_fn(input[feature])) for feature in features]
                feature_dim (Optional, int):
                    default dimension of input features. Again, features along this dim will be chosen
                    for sparsification.
                sparse_config (Dict):
                    Default configuration for the mask_fn. This config will be passed
                    with the mask_fn()

    Example:
        >>> # xdoctest: +SKIP
        >>> model = SomeModel()
        >>> act_sparsifier = ActivationSparsifier(...)  # init activation sparsifier
        >>> # Initialize aggregate_fn
        >>> def agg_fn(x, y):
        >>>     return x + y
        >>>
        >>> # Initialize reduce_fn
        >>> def reduce_fn(x):
        >>>     return torch.mean(x, dim=0)
        >>>
        >>> # Initialize mask_fn
        >>> def mask_fn(data):
        >>>     return torch.eye(data.shape).to(data.device)
        >>>
        >>>
        >>> act_sparsifier.register_layer(
        ...     model.some_layer,
        ...     aggregate_fn=agg_fn,
        ...     reduce_fn=reduce_fn,
        ...     mask_fn=mask_fn,
        ... )
        >>>
        >>> # start training process
        >>> for _ in [...]:
        >>> # epoch starts
        >>> # model.forward(), compute_loss() and model.backwards()
        >>> # epoch ends
        >>>     act_sparsifier.step()
        >>> # end training process
        >>> sparsifier.squash_mask()
    """

    def __init__(
        self,
        model: nn.Module,
        aggregate_fn=None,
        reduce_fn=None,
        mask_fn=None,
        features=None,
        feature_dim=None,
        **sparse_config,
    ):
        # [FUNCTION DOC] Stores the model and default settings (aggregate_fn, reduce_fn, mask_fn, features,
        # feature_dim, and extra keyword args kept as the default sparse_config) and creates empty
        # per-layer data_groups and state mappings. Nothing is registered or hooked yet.
        self.model = model
        self.defaults: dict[str, Any] = defaultdict()
        self.defaults["sparse_config"] = sparse_config

        # functions
        self.defaults["aggregate_fn"] = aggregate_fn
        self.defaults["reduce_fn"] = reduce_fn
        self.defaults["mask_fn"] = mask_fn

        # default feature and feature_dim
        self.defaults["features"] = features
        self.defaults["feature_dim"] = feature_dim

        self.data_groups: dict[str, dict] = defaultdict(
            dict
        )  # contains all relevant info w.r.t each registered layer

        self.state: dict[str, Any] = defaultdict(dict)  # layer name -> mask

    @staticmethod
    def _safe_rail_checks(args):
        # [FUNCTION DOC] HELPER: Validates a layer config mapping. Raises an error if features is given
        # without feature_dim, or if aggregate_fn, reduce_fn or mask_fn is not callable (so each must be
        # supplied either as default or at registration).
        """Makes sure that some of the functions and attributes are not passed incorrectly"""

        # if features are not None, then feature_dim must not be None
        features, feature_dim = args["features"], args["feature_dim"]
        if features is not None:
            if feature_dim is None:
                raise AssertionError("need feature dim to select features")

        # all the *_fns should be callable
        fn_keys = ["aggregate_fn", "reduce_fn", "mask_fn"]
        for key in fn_keys:
            fn = args[key]
            if not callable(fn):
                raise AssertionError(f"{fn} must be callable")

    def _aggregate_hook(self, name):
        # [FUNCTION DOC] HELPER: Build the forward pre-hook that aggregates inputs of the named layer.
        # Reads feature settings and aggregate_fn from the layer config at build time. The hook takes the
        # first input; without features it combines the stored data (zeros on first call) with the input
        # via aggregate_fn; with features it does so per selected feature slice taken along feature_dim.
        # On the first call it initializes the mask entries to ones in matching shape. Result is stored
        # in the layer config under data. The hook returns nothing and does not modify the input.
        """Returns hook that computes aggregate of activations passing through."""

        # gather some data
        feature_dim = self.data_groups[name]["feature_dim"]
        features = self.data_groups[name]["features"]
        agg_fn = self.data_groups[name]["aggregate_fn"]

        def hook(module, input) -> None:
            # [FUNCTION DOC] HELPER: Hook body described on the enclosing method; updates data_groups and state for this layer.
            input_data = input[0]

            data = self.data_groups[name].get("data")  # aggregated data
            if features is None:
                # no features associated, data should not be a list
                if data is None:
                    data = torch.zeros_like(input_data)
                    self.state[name]["mask"] = torch.ones_like(input_data)
                out_data = agg_fn(data, input_data)
            else:
                # data should be a list [aggregated over each feature only]
                if data is None:
                    out_data = [
                        0 for _ in range(len(features))
                    ]  # create one in case of 1st forward
                    self.state[name]["mask"] = [0 for _ in range(len(features))]
                else:
                    out_data = data  # a list

                # compute aggregate over each feature
                for feature_idx in range(len(features)):
                    # each feature is either a list or scalar, convert it to torch tensor
                    feature_tensor = (
                        torch.Tensor([features[feature_idx]])
                        .long()
                        .to(input_data.device)
                    )
                    data_feature = torch.index_select(
                        input_data, feature_dim, feature_tensor
                    )
                    if data is None:
                        curr_data = torch.zeros_like(data_feature)
                        self.state[name]["mask"][feature_idx] = torch.ones_like(
                            data_feature
                        )
                    else:
                        curr_data = data[feature_idx]
                    out_data[feature_idx] = agg_fn(curr_data, data_feature)
            self.data_groups[name]["data"] = out_data

        return hook

    def register_layer(
        self,
        layer: nn.Module,
        aggregate_fn=None,
        reduce_fn=None,
        mask_fn=None,
        features=None,
        feature_dim=None,
        **sparse_config,
    ):
        # [FUNCTION DOC] Registers a layer (must be inside self.model) for sparsification. The name is its
        # fully qualified name in the model; raises an error if not found. If already registered, warns and
        # unregisters first. Builds the layer config from defaults overridden by non-None arguments
        # (sparse_config is merged), validates it, stores it, attaches the aggregate forward pre-hook, sets
        # the mask to None and marks hook_state as aggregate.
        r"""
        Registers a layer for sparsification. The layer should be part of self.model.
        Specifically, registers a pre-forward hook to the layer. The hook will apply the aggregate_fn
        and store the aggregated activations that is input over each step.

        Note::
            - There is no need to pass in the name of the layer as it is automatically computed as per
              the fqn convention.

            - All the functions (fn) passed as argument will be called at a dim, feature level.
        """
        name = module_to_fqn(self.model, layer)
        if name is None:
            raise AssertionError("layer not found in the model")

        if name in self.data_groups:  # unregister layer if already present
            warnings.warn(
                "layer already attached to the sparsifier, deregistering the layer and registering with new config",
                stacklevel=2,
            )
            self.unregister_layer(name=name)

        local_args = copy.deepcopy(self.defaults)
        update_dict = {
            "aggregate_fn": aggregate_fn,
            "reduce_fn": reduce_fn,
            "mask_fn": mask_fn,
            "features": features,
            "feature_dim": feature_dim,
            "layer": layer,
        }
        local_args.update(
            (arg, val) for arg, val in update_dict.items() if val is not None
        )
        local_args["sparse_config"].update(sparse_config)

        self._safe_rail_checks(local_args)

        self.data_groups[name] = local_args
        agg_hook = layer.register_forward_pre_hook(self._aggregate_hook(name=name))

        self.state[name]["mask"] = (
            None  # mask will be created when model forward is called.
        )

        # attach agg hook
        self.data_groups[name]["hook"] = agg_hook

        # for serialization purposes, we know whether aggregate_hook is attached
        # or sparsify_hook()
        self.data_groups[name]["hook_state"] = "aggregate"  # aggregate hook is attached

    def get_mask(self, name: str | None = None, layer: nn.Module | None = None):
        # [FUNCTION DOC] Return the mask for a layer given by name or by module object (name derived from
        # the model). Raises an error if neither given or module not in the model; raises ValueError if the
        # layer is unknown or its mask is still None (no forward pass yet).
        """
        Returns mask associated to the layer.

        The mask is
            - a torch tensor is features for that layer is None.
            - a list of torch tensors for each feature, otherwise

        Note::
            The shape of the mask is unknown until model.forward() is applied.
            Hence, if get_mask() is called before model.forward(), an
            error will be raised.
        """
        if name is None and layer is None:
            raise AssertionError("Need at least name or layer obj to retrieve mask")

        if name is None:
            if layer is None:
                raise AssertionError("layer must be provided when name is None")
            name = module_to_fqn(self.model, layer)
            if name is None:
                raise AssertionError("layer not found in the specified model")

        if name not in self.state:
            raise ValueError("Error: layer with the given name not found")

        mask = self.state[name].get("mask", None)

        if mask is None:
            raise ValueError(
                "Error: shape unknown, call layer() routine at least once to infer mask"
            )
        return mask

    def unregister_layer(self, name):
        # [FUNCTION DOC] Remove the layer's hook and delete its state and data_groups entries.
        # Raises KeyError if the name is not registered.
        """Detaches the sparsifier from the layer"""

        # detach any hooks attached
        self.data_groups[name]["hook"].remove()

        # pop from the state dict
        self.state.pop(name)

        # pop from the data groups
        self.data_groups.pop(name)

    def step(self):
        # [FUNCTION DOC] Without gradients, for each registered layer calls update_mask with its accumulated
        # data, then discards that accumulated data. Raises KeyError if a layer has seen no forward pass
        # (no data entry).
        """Internally calls the update_mask() function for each layer"""
        with torch.no_grad():
            for name, configs in self.data_groups.items():
                data = configs["data"]
                self.update_mask(name, data, configs)

                self.data_groups[name].pop("data")  # reset the accumulated data

    def update_mask(self, name, data, configs):
        # [FUNCTION DOC] Update the stored mask of layer name from aggregated data. Applies reduce_fn then
        # mask_fn (with the layer's sparse_config) to the data, per feature when features are set, and
        # assigns the result into the existing mask tensor(s) in place. Returns nothing.
        """
        Called for each registered layer and does the following-
            1. apply reduce_fn on the aggregated activations
            2. use mask_fn to compute the sparsification mask

        Note:
            the reduce_fn and mask_fn is called for each feature, dim over the data
        """
        mask = self.get_mask(name)
        sparse_config = configs["sparse_config"]
        features = configs["features"]
        reduce_fn = configs["reduce_fn"]
        mask_fn = configs["mask_fn"]
        if features is None:
            data = reduce_fn(data)
            mask.data = mask_fn(data, **sparse_config)
        else:
            for feature_idx in range(len(features)):
                data_feature = reduce_fn(data[feature_idx])
                mask[feature_idx].data = mask_fn(data_feature, **sparse_config)

    def _sparsify_hook(self, name):
        # [FUNCTION DOC] HELPER: Build the forward pre-hook that applies the layer's mask. Fetches the mask now
        # (so raises if it is not yet known). The hook multiplies the input by the mask, or with features,
        # overwrites each selected feature slice of the input in place with its masked values; returns the
        # modified input, which replaces the layer input.
        """Returns hook that applies sparsification mask to input entering the attached layer"""
        mask = self.get_mask(name)
        features = self.data_groups[name]["features"]
        feature_dim = self.data_groups[name]["feature_dim"]

        def hook(module, input):
            # [FUNCTION DOC] HELPER: Hook body described on the enclosing method.
            input_data = input[0]
            if features is None:
                # apply to all the features
                return input_data * mask
            else:
                # apply per feature, feature_dim
                for feature_idx in range(len(features)):
                    feature = (
                        torch.Tensor([features[feature_idx]])
                        .long()
                        .to(input_data.device)
                    )
                    sparsified = (
                        torch.index_select(input_data, feature_dim, feature)
                        * mask[feature_idx]
                    )
                    input_data.index_copy_(feature_dim, feature, sparsified)
                return input_data

        return hook

    def squash_mask(self, attach_sparsify_hook=True, **kwargs):
        # [FUNCTION DOC] Finalize masks: for every registered layer remove the aggregate hook and, if
        # attach_sparsify_hook is true, attach the sparsify pre-hook. kwargs are ignored.
        # KNOWN ISSUE: hook_state is set to sparsify even when no sparsify hook is attached, and the hook
        # entry is then missing, so a later unregister_layer fails with KeyError.
        """
        Unregisters aggregate hook that was applied earlier and registers sparsification hooks if
        attach_sparsify_hook = True.
        """
        for name, configs in self.data_groups.items():
            # unhook agg hook
            configs["hook"].remove()
            configs.pop("hook")
            self.data_groups[name]["hook_state"] = "None"
            if attach_sparsify_hook:
                configs["hook"] = configs["layer"].register_forward_pre_hook(
                    self._sparsify_hook(name)
                )
            configs["hook_state"] = (
                "sparsify"  # signals that sparsify hook is now attached
            )

    def _get_serializable_data_groups(self):
        # [FUNCTION DOC] HELPER: Return a copy of data_groups with the hook and layer entries removed
        # (those cannot be serialized). Functions are kept as is.
        """Exclude hook and layer from the config keys before serializing

        TODO: Might have to treat functions (reduce_fn, mask_fn etc) in a different manner while serializing.
              For time-being, functions are treated the same way as other attributes
        """
        data_groups: dict[str, Any] = defaultdict()
        for name, config in self.data_groups.items():
            new_config = {
                key: value
                for key, value in config.items()
                if key not in ["hook", "layer"]
            }
            data_groups[name] = new_config
        return data_groups

    def _convert_mask(self, states_dict, sparse_coo=True):
        # [FUNCTION DOC] HELPER: Return a deep copy of the state mapping with each mask (or list of masks)
        # converted to sparse COO format when sparse_coo is true, otherwise to dense. None masks are left.
        r"""Converts the mask to sparse coo or dense depending on the `sparse_coo` argument.
        If `sparse_coo=True`, then the mask is stored as sparse coo else dense tensor
        """
        states = copy.deepcopy(states_dict)
        for state in states.values():
            if state["mask"] is not None:
                if isinstance(state["mask"], list):
                    for idx in range(len(state["mask"])):
                        if sparse_coo:
                            state["mask"][idx] = state["mask"][idx].to_sparse_coo()
                        else:
                            state["mask"][idx] = state["mask"][idx].to_dense()
                else:
                    if sparse_coo:
                        state["mask"] = state["mask"].to_sparse_coo()
                    else:
                        state["mask"] = state["mask"].to_dense()
        return states

    def state_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC] Return a mapping with keys state (masks converted to sparse), data_groups
        # (serializable configs) and defaults.
        r"""Returns the state of the sparsifier as a :class:`dict`.

        It contains:
        * state - contains name -> mask mapping.
        * data_groups - a dictionary containing all config information for each
            layer
        * defaults - the default config while creating the constructor
        """
        data_groups = self._get_serializable_data_groups()
        state = self._convert_mask(self.state)
        return {"state": state, "data_groups": data_groups, "defaults": self.defaults}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # [FUNCTION DOC] Restore the sparsifier from a mapping with state, data_groups and defaults by
        # delegating to the set-state routine. Returns nothing.
        r"""The load_state_dict() restores the state of the sparsifier based on the state_dict

        Args:
        * state_dict - the dictionary to which the current sparsifier needs to be restored to
        """
        state = state_dict["state"]
        data_groups, defaults = state_dict["data_groups"], state_dict["defaults"]

        self.__set_state__(
            {"state": state, "data_groups": data_groups, "defaults": defaults}
        )

    def __get_state__(self) -> dict[str, Any]:
        # [FUNCTION DOC] Return a mapping with defaults, state (sparse masks) and serializable data_groups.
        # KNOWN ISSUE: the name uses the pattern get-state with underscores around it, not the standard
        # pickle protocol name, so pickling does not use it.
        data_groups = self._get_serializable_data_groups()
        state = self._convert_mask(self.state)
        return {
            "defaults": self.defaults,
            "state": state,
            "data_groups": data_groups,
        }

    def __set_state__(self, state: dict[str, Any]) -> None:
        # [FUNCTION DOC] Replace instance attributes from a state mapping, converting masks back to dense,
        # then for each data group look up the layer in the model by name (raises an error if missing),
        # re-attach the hook matching hook_state (aggregate or sparsify) and store layer and hook in the config.
        # KNOWN ISSUE: if hook_state is neither value, no hook is created and the hook variable is unbound
        # (or reuses the previous iteration's hook). Not the standard pickle protocol name either.
        state["state"] = self._convert_mask(
            state["state"], sparse_coo=False
        )  # convert mask to dense tensor
        self.__dict__.update(state)

        # need to attach layer and hook info into the data_groups
        for name, config in self.data_groups.items():
            # fetch layer
            layer = fqn_to_module(self.model, name)
            if layer is None:
                raise AssertionError(f"layer {name} not found in the model")

            # if agg_mode is True, then layer in aggregate mode
            if "hook_state" in config and config["hook_state"] == "aggregate":
                hook = layer.register_forward_pre_hook(self._aggregate_hook(name))

            elif "hook_state" in config and config["hook_state"] == "sparsify":
                hook = layer.register_forward_pre_hook(self._sparsify_hook(name))

            config["layer"] = layer
            config["hook"] = hook  # type: ignore[possibly-undefined]

    def __repr__(self):
        # [FUNCTION DOC] Return a readable multi-line string listing each data group (name and config
        # entries) excluding data, hook and the three function entries.
        format_string = self.__class__.__name__ + " ("
        for name, config in self.data_groups.items():
            format_string += "\n"
            format_string += "\tData Group\n"
            format_string += f"\t    name: {name}\n"
            for key in sorted(config.keys()):
                if key in ["data", "hook", "reduce_fn", "mask_fn", "aggregate_fn"]:
                    continue
                format_string += f"\t    {key}: {config[key]}\n"
        format_string += ")"
        return format_string
