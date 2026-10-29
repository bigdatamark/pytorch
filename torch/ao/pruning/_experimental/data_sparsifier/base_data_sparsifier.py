# mypy: allow-untyped-defs
# [MODULE DOC] Defines BaseDataSparsifier, a sparsifier for raw tensors, parameters and embeddings
# (rather than module weights), plus the internal container module that owns the data and its mask
# parametrizations. Also defines EMBEDDING_TYPES and SUPPORTED_TYPES. Import-time side effect:
# if no Python warning options are set, sets the warnings filter to show each warning once.
import abc
import copy
import sys
import warnings
from collections import defaultdict
from typing import Any

import torch
from torch import nn
from torch.ao.pruning.sparsifier import base_sparsifier, utils
from torch.nn.utils import parametrize


if not sys.warnoptions:
    # to suppress repeated warnings when being used in a training loop.
    warnings.simplefilter("once")

__all__ = ["BaseDataSparsifier"]

EMBEDDING_TYPES = {
    nn.Embedding,
    nn.EmbeddingBag,
}

SUPPORTED_TYPES = {
    torch.Tensor,
    nn.Parameter,
    *EMBEDDING_TYPES,
}


class _Container(nn.Module):
    # [CLASS DOC] HELPER: Empty module used as the holder of buffers and parametrizations for sparsified data.
    pass


class BaseDataSparsifier(base_sparsifier.BaseSparsifier):
    # [CLASS DOC] Abstract base for data sparsifiers. State: _container (a holder module with one buffer per
    # named data item, parametrized with a sparsity mask), data_groups (name -> config), state (name ->
    # mask), plus defaults inherited from the base sparsifier. The class owns masks and parametrizations,
    # not the user. Subclasses implement update_mask. Key methods: add_data, get_data, get_mask, step,
    # squash_mask, state_dict / load_state_dict and pickle hooks.
    r"""
    Base Data Sparsifier class for all Data sparsifiers.
    The abstract class accepts raw torch tensors / embedding / embedding bags (refer to SUPPORTED_TYPES above)
    to prepare for sparsification.
    In this case, mask (and parametrizations) is owned by the class and not by the user.
    Specifically, the container object inside the class maintains the mask and parametrizations of the input data

    Args:
        data_list (list of tuples)
            list of (name, data) tuples to sparsify. Lookup SUPPORTED_TYPES
            for type of data. Internally, a container module handles the data sparsification.

        defaults (dict)
            default configurations will be attached to the
            configuration. Only the keys that don't exist in the `config` will
            be updated.
    Example::
        >>> # xdoctest: +SKIP
        >>> data_list = [('tensor_1', torch.randn(3,3)), ('tensor_2', torch.randn(4,4))]
        >>> defaults = {'sparsity_level': 0.7}
        >>> sparsifier = DerivedDataSparsifier(data_list = data_list, **defaults) # Some sparsifier that inherits BaseDataSparsifier
        >>> new_tensor_to_add = {'name': 'tensor_3', 'data': torch.randn(5,5), 'sparsity_level': 0.3}
        >>> sparsifier.add_data(**new_tensor_to_add)
        >>> # tensor_1 and tensor_2 will have sparsity_level of 0.7 but tensor_3 will have sparsity_level=0.3
    """

    def __init__(self, data_list: list[tuple[str, Any]] | None = None, **defaults):
        # [FUNCTION DOC] Create the sparsifier with default configuration kwargs and an empty container, then
        # add each (name, data) pair of data_list (if given) using the defaults.
        super().__init__(defaults=defaults)

        self._container = _Container()

        self.data_groups: dict[str, dict] = defaultdict(dict)  # name -> {**config}
        if data_list is not None:
            # add data with default config here
            [self.add_data(name, data, **self.defaults) for name, data in data_list]

    def prepare(self, model, config):
        # [FUNCTION DOC] Not supported: always raises NotImplementedError.
        raise NotImplementedError("this function is undefined for this class")

    def _extract_weight(self, data):
        # [FUNCTION DOC] HELPER: Return the tensor itself for a tensor or parameter, or the weight of an
        # embedding / embedding bag. Returns None for any other type.
        # extract the weight parameter instead of underlying data
        if type(data) in [torch.Tensor, nn.Parameter]:
            return data
        elif type(data) in EMBEDDING_TYPES:
            return data.weight

    def add_data(self, name: str, data, reuse_mask=True, **config):
        # [FUNCTION DOC] Add (or replace) a named data item to sparsify. Data must be exactly one of the supported
        # types, else raises an error. Config is the defaults overridden by the given keyword config; the mask
        # defaults to all ones (config may supply mask or parametrization class). If the name exists, warns,
        # starts from the old config updated with the new one and, if reuse_mask, keeps the old mask (raises an
        # error if shapes differ), then deletes the old item. Registers the weight as a container buffer,
        # parametrizes it with the mask, records state and config, and returns the parametrized container attribute.
        r"""Configures and parametrizes the internal container model with name and data.

        **Note**:
            1. If the data with name already exists, it replaces the data.
            2. While replacing, the old mask is reused when `reuse_mask=True`
            3. If `reuse_mask=True`, then the replacing data needs to have the same shape as that of old data.
            4. By default, the config of the replaced data is used as config for the replacing data, unless something
               is specified in the config dictionary.
        """
        if type(data) not in SUPPORTED_TYPES:
            raise AssertionError(
                f"specified data type:{type(data)} not supported at the moment"
            )
        local_args = copy.deepcopy(self.defaults)
        local_args.update(config)
        weight = self._extract_weight(data)

        # Bookkeeping in the container class
        mask = local_args.get("mask", torch.ones_like(weight))
        param_class = local_args.get("parametrization", utils.FakeSparsity)

        if name in self.state:
            # If the named data already exists - replace
            warnings.warn(
                "Replacing existing data of the same name. - Did you mean a different name?",
                stacklevel=2,
            )

            # reuse old config
            old_args = self.data_groups[name]
            local_args = copy.deepcopy(old_args)
            local_args.update(config)

            if reuse_mask:
                current_data = self.get_data(name=name)
                if weight.shape != current_data.shape:
                    raise AssertionError(
                        "to retain the old mask, the shape of the new data must be the same as the previous one"
                    )
                mask = self.get_mask(
                    name=name
                )  # reuse mask instead of creating a new one

            self._delete_data(name=name)

        # parameter creates a deepcopy of the weight inside, so create a buffer
        self._container.register_buffer(name=name, tensor=weight)
        parametrize.register_parametrization(self._container, name, param_class(mask))
        self.state[name]["mask"] = mask
        self.data_groups[name] = local_args
        return getattr(self._container, name)

    def get_data(self, name: str, return_original: bool = True):
        # [FUNCTION DOC] Return the data registered under name: the original unparametrized tensor if
        # return_original (raises ValueError if parametrization was removed by squashing), otherwise the
        # sparsified (parametrized) view. Raises ValueError for an unknown name.
        r"""Returns weight tensor (or data)
        Args:
            - name: name of the data to be returned
            - return_original returns weight tensor without applying parametrization if True
                else - returns the sparsified version (parametrized)
        """
        if name not in self.data_groups:
            raise ValueError("data with specified name does not exist")

        if return_original:
            if not parametrize.is_parametrized(self._container, name):
                raise ValueError("mask squashed - original mask value does not exist")
            data = getattr(self._container.parametrizations, name).original
            return data
        else:
            return getattr(self._container, name)

    def _convert_mask(self, states, sparse_coo=True):
        # [FUNCTION DOC] HELPER: Deep copy the state mapping and convert every mask to sparse COO (or dense when
        # sparse_coo is false). Returns the copy.
        r"""Converts the mask to sparse coo or dense tensors depending on the `sparse_coo` argument."""
        states = copy.deepcopy(states)
        for state in states.values():
            if sparse_coo:
                state["mask"] = state["mask"].to_sparse_coo()
            else:
                state["mask"] = state["mask"].to_dense()

        return states

    def state_dict(self):
        # [FUNCTION DOC] Return a mapping with state (masks as sparse COO), data_groups (the live mapping, not a
        # copy) and _container (the container module state dictionary).
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains:
        * state - contains name -> mask mapping.
        * data_groups - a list containing all sparsity configuration groups
            with the key name specifying the name of the data
        * container_state_dict - the state dictionary of the internal
            container model used for sparsification
        """
        state = self._convert_mask(self.state)
        return {
            "state": state,
            "data_groups": self.data_groups,
            "_container": self._container.state_dict(),
        }

    def _load_container_from_state(self, states, data_groups, container_state_dict):
        # [FUNCTION DOC] HELPER: Rebuild container contents from saved pieces. For each name in states, requires
        # a config in data_groups and a matching entry in the container state (plain or parametrized original),
        # else raises RuntimeError. Registers it as a buffer and, if it was parametrized, re-registers the
        # parametrization with the saved mask (default all ones).
        # KNOWN ISSUE: the parametrization class is looked up as a top-level key of data_groups (keyed by data
        # name), so it practically always falls back to the default FakeSparsity.
        r"""This restores the state of the container specifically based on the data present in state and data_groups
        If the data was parametrized, then the data would be added to the container and then parametrized,
        else it would just add the attribute to the container.
        """
        for name, state in states.items():
            config_name = data_groups.get(name, None)
            if config_name is None:
                raise RuntimeError(f"Error loading {name}")

            # check if the data with such a name was parametrized, if so parametrize
            # otherwise just set the attribute and continue
            parametrized_name = f"parametrizations.{name}.original"
            parametrized = False
            data = container_state_dict.get(name, None)
            if name in container_state_dict:
                # the parametrization was probably removed for this
                data = container_state_dict.get(name)

            elif parametrized_name in container_state_dict:
                # so the weight was parametrized
                data = container_state_dict.get(parametrized_name)
                parametrized = True

            else:
                raise RuntimeError(f"Error loading {name}")

            self._container.register_buffer(name=name, tensor=data)

            if parametrized:
                # register parameter if parametrized
                mask = state.get("mask", torch.ones_like(data))
                param_class = data_groups.get(
                    "parametrization", utils.FakeSparsity
                )  # change once public_api for utils is fixed!
                parametrize.register_parametrization(
                    self._container, name, param_class(mask)
                )

    def load_state_dict(self, state_dict, strict=True):
        # [FUNCTION DOC] Restore from a state mapping (state, data_groups, _container). Works on deep copies;
        # converts masks to dense. If strict, replaces the container with a fresh one first; rebuilds the
        # container, then (if not strict) merges the current state and groups over the loaded ones, and
        # finally applies the result via the set-state routine.
        # KNOWN ISSUE: in non-strict mode the current entries override loaded entries of the same name.
        r"""The load_state_dict() restores the state of the sparsifier based on the state_dict

        Args:
        * state_dict - the dictionary to which the current sparsifier needs to be restored to
        * strict - If True - the sparsifier is reset and is restored exactly to the state in state_dict.
            If False - the current sparsifier is not reset before loading the state_dict i.e. data added
            before loading the state_dict is not erased.
        """
        states = copy.deepcopy(state_dict["state"])
        data_groups = copy.deepcopy(state_dict["data_groups"])
        container_state_dict = copy.deepcopy(state_dict["_container"])

        states = self._convert_mask(
            states, sparse_coo=False
        )  # convert sparse coo mask to dense
        if strict:
            # if strict load -> then reset container
            self._container = _Container()

        self._load_container_from_state(states, data_groups, container_state_dict)

        if not strict:
            states.update(self.state)
            data_groups.update(self.data_groups)

        self.__setstate__({"state": states, "data_groups": data_groups})

    def __setstate__(self, state):
        # [FUNCTION DOC] Pickle support: if state has a container entry, rebuilds a fresh container from it
        # (masks converted to dense) and consumes that entry; then updates the instance attributes from state.
        # Without a container entry masks are not converted.
        if "_container" in state:  # If container object is in state then load model
            container_dict = state.pop("_container")
            self._container = _Container()
            state["state"] = self._convert_mask(
                state["state"], sparse_coo=False
            )  # convert sparse coo mask to dense
            self._load_container_from_state(
                state["state"], state["data_groups"], container_dict
            )

        self.__dict__.update(state)

    def __getstate__(self):
        # [FUNCTION DOC] Pickle support: return a mapping with defaults, state (sparse masks), data_groups and
        # the container state dictionary.
        state = self._convert_mask(self.state)
        return {
            "defaults": self.defaults,
            "state": state,
            "data_groups": self.data_groups,
            "_container": self._container.state_dict(),
        }

    def __repr__(self):  # type:ignore[override]
        # [FUNCTION DOC] Return a readable multi-line string listing each data group name and its config, skipping the data entry.
        format_string = self.__class__.__name__ + " ("
        for name, sparse_args in self.data_groups.items():
            format_string += "\n"
            format_string += "\tData Group\n"
            format_string += f"\t    name: {name}\n"
            for key in sorted(sparse_args.keys()):
                if key == "data":
                    continue
                format_string += f"\t    {key}: {sparse_args[key]}\n"
        format_string += ")"
        return format_string

    def get_mask(self, name: str):
        # [FUNCTION DOC] Return the mask for the named data; raises ValueError if the name is unknown.
        if name not in self.state:
            raise ValueError("data with specified name does not exist")
        return self.state[name]["mask"]

    def squash_mask(self, *args, leave_parametrized=True, names=None, **kwargs):
        # [FUNCTION DOC] Remove mask parametrizations from the container for names (default all data groups),
        # keeping the parametrized (masked) values if leave_parametrized else restoring the originals.
        # Extra args are ignored.
        # KNOWN ISSUE: the docstring mentions a sparsified option that does not exist.
        r"""Squashes the sparse masks into the appropriate tensors. Also, accepts list of strings
        to squash mask for. If none, squashes mask for all the keys
        kwargs:
            * names: list of strings to squash mask for
            * sparsified: if true - applies the mask before squashing
                          if false - does not apply the mask before squashing
        """
        if names is None:
            names = list(self.data_groups.keys())
        for name in names:
            parametrize.remove_parametrizations(
                self._container, name, leave_parametrized=leave_parametrized
            )

    def step(self):  # type:ignore[override]
        # [FUNCTION DOC] If mask updates are enabled, for each data group fetch the original data and call
        # update_mask with the name, data and the group config, without gradients. No-op otherwise.
        if not self.enable_mask_update:
            return
        with torch.no_grad():
            for name, config in self.data_groups.items():
                # get non-sparsified data
                data = self.get_data(name)
                # need name for the mask otherwise can directly pass mask?
                self.update_mask(name, data, **config)

    @abc.abstractmethod
    def update_mask(self, name, data, **kwargs):  # type: ignore[override]
        # [FUNCTION DOC] Abstract hook for subclasses to compute and store the mask for the named data from
        # the given data and config kwargs. The base body does nothing.
        pass

    def _delete_data(self, name):
        # [FUNCTION DOC] HELPER: Remove named data: squashes its parametrization without applying the mask,
        # deletes the container attribute, and drops its state and data_groups entries.
        """Detaches some data from the sparsifier.

        Args:
            name (str)
                Name of the data to be removed from the sparsifier

        Note:
            Currently private. Kind of used as a helper function when replacing data of the same name
        """
        self.squash_mask(
            names=[name], leave_parametrized=False
        )  # do not apply the mask while deleting
        delattr(self._container, name)
        self.state.pop(name)
        self.data_groups.pop(name)
