# [MODULE DOC] Utilities that turn a stateful nn.Module into a functional form whose
# parameters and buffers are passed explicitly (make_functional, make_functional_with_buffers,
# combine_state_for_ensemble, plus deprecated v1 helpers and functional_init variants).
# Handles parameter tying via a name map. No import-time side effects.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
from typing import Any, NoReturn, TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils._named_member_accessor import NamedMemberAccessor


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

# Utilities to make nn.Module "functional"
# In particular the goal is to be able to provide a function that takes as input
# the parameters and evaluate the nn.Module using fixed inputs.


def raise_parameter_tying_error() -> NoReturn:
    # [FUNCTION DOC] Always raises a RuntimeError stating that models with tied
    # (shared) parameters are unsupported by make_functional. Takes no arguments.
    raise RuntimeError(
        "make_functional(module): we don't yet support models that "
        "do parameter tying (also sometimes known as weight sharing). "
        "Please try to rewrite your model by replacing all instances of the "
        "tied parameter with another and/or comment your support in "
        "https://github.com/pytorch/functorch/issues/446"
    )


def create_names_map(
    named_params: dict[str, Tensor] | Iterable[tuple[str, Tensor]],
    tied_named_params: dict[str, Tensor] | Iterable[tuple[str, Tensor]],
) -> dict[str, list[str]]:
    # [FUNCTION DOC] Builds a mapping from each unique parameter name to the list of all
    # names (including tied aliases) that refer to the same tensor. Both inputs may be
    # mappings or iterables of name/tensor pairs. Tensors are matched by identity hash.
    # Raises AssertionError if unique names are not a subset of all names, or a tied
    # tensor is not among the unique tensors. Pure function, no side effects.
    """
    named_params is a dictionary of tensors: {'A': A, 'B': B}
    tied_named_params is another dictionary of tensors {'A': A, 'B': B, 'B_tied': B}
    with potentially tied (or 'duplicated') tensors

    This function creates a mapping from the names in named_params to the
    names in tied_named_params: {'A': ['A'], 'B': ['B', 'B_tied']}.
    """
    named_params_dict = dict(named_params)
    tied_named_params_dict = dict(tied_named_params)

    tensors_dict_keys = set(named_params_dict.keys())
    tied_tensors_dict_keys = set(tied_named_params_dict.keys())
    if not tensors_dict_keys.issubset(tied_tensors_dict_keys):
        raise AssertionError(
            f"tensors_dict_keys {tensors_dict_keys} is not a subset of "
            f"tied_tensors_dict_keys {tied_tensors_dict_keys}"
        )

    tensor_to_mapping: dict[Tensor, tuple[str, list[str]]] = {}
    for key, tensor in named_params_dict.items():
        tensor_to_mapping[tensor] = (key, [])
    for key, tensor in tied_named_params_dict.items():
        if tensor not in tensor_to_mapping:
            raise AssertionError(
                f"tensor for key '{key}' not found in tensor_to_mapping"
            )
        tensor_to_mapping[tensor][1].append(key)
    return dict(tensor_to_mapping.values())


def _extract_members(
    mod: nn.Module,
    named_members: Callable[..., Iterable[tuple[str, Tensor]]],
    subclass: Callable[[Tensor], Tensor],
) -> tuple[tuple[Tensor, ...], tuple[str, ...], dict[str, list[str]]]:
    # [FUNCTION DOC] HELPER: Strips parameters or buffers out of a module in place.
    # named_members is the module's bound named_parameters/named_buffers; subclass wraps
    # each meta-device placeholder (Parameter for weights, identity for buffers).
    # Every member (including tied aliases) is replaced by a meta placeholder, with tied
    # members sharing one placeholder. Returns (unique tensors, unique names, names map).
    all_named_members = tuple(named_members(remove_duplicate=False))
    unique_named_members = tuple(named_members(remove_duplicate=True))
    names_map = create_names_map(unique_named_members, all_named_members)

    # Remove all the members in the model
    # pyrefly: ignore [implicit-any]
    memo = {}
    accessor = NamedMemberAccessor(mod)
    for name, p in all_named_members:
        if p not in memo:
            memo[p] = subclass(torch.empty_like(p, device="meta"))
        replacement = memo[p]
        accessor.set_tensor(name, replacement)

    if len(unique_named_members) == 0:
        names, params = (), ()
    else:
        names, params = zip(*unique_named_members)  # type: ignore[assignment]
    return params, names, names_map


def extract_weights(
    mod: nn.Module,
) -> tuple[tuple[Tensor, ...], tuple[str, ...], dict[str, list[str]]]:
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    # [FUNCTION DOC] Removes all parameters from mod in place (replacing them with meta
    # placeholders) and returns (params tuple, names tuple, tied-names map).
    return _extract_members(mod, mod.named_parameters, nn.Parameter)


def extract_buffers(
    mod: nn.Module,
) -> tuple[tuple[Tensor, ...], tuple[str, ...], dict[str, list[str]]]:
    # [FUNCTION DOC] Same as extract_weights but for buffers: replaces each buffer in mod
    # with a meta placeholder in place and returns (buffers, names, tied-names map).
    return _extract_members(mod, mod.named_buffers, lambda x: x)


def load_weights(
    mod: nn.Module,
    names: Sequence[str],
    params: Sequence[Tensor],
    as_params: bool = False,
) -> None:
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    # [FUNCTION DOC] Sets the tensors named by names on mod to the given params, in place.
    # If as_params is true each tensor is first wrapped in a Parameter. Returns nothing.
    accessor = NamedMemberAccessor(mod)
    if as_params:
        params = [nn.Parameter(p) for p in params]
    accessor.set_tensors(names, params)


def _swap_state(
    mod: nn.Module, names_map: dict[str, list[str]], elems: Iterable[Tensor]
) -> list[Tensor]:
    # [FUNCTION DOC] HELPER: Pairs each entry of names_map with the next element of elems.
    # For each group of aliased names, the first name is swapped (its old tensor is
    # collected in the result list) and the remaining aliases are set to the same new
    # tensor. Mutates mod; returns the replaced tensors, one per group.
    result: list[Tensor] = []
    accessor = NamedMemberAccessor(mod)
    for (_, attr_names), elem in zip(names_map.items(), elems):
        for i, attr_name in enumerate(attr_names):
            if i == 0:
                result.append(accessor.swap_tensor(attr_name, elem))
            else:
                accessor.set_tensor(attr_name, elem)
    return result


def load_buffers(
    mod: nn.Module,
    names: Sequence[str],
    buffers: Sequence[Tensor],
    as_params: bool = False,
) -> None:
    # [FUNCTION DOC] Sets the named buffers on mod in place to the given tensors.
    # KNOWN ISSUE: the as_params argument is accepted but ignored.
    accessor = NamedMemberAccessor(mod)
    accessor.set_tensors(names, buffers)


def load_state(
    model: nn.Module,
    weights: Sequence[Tensor],
    weight_names: Sequence[str],
    buffers: Sequence[Tensor] = (),
    buffer_names: Sequence[str] = (),
) -> nn.Module:
    """load_state(model, weights, weight_names, buffers=(), buffer_names=()) -> model

    load_state takes `weights` and `buffers` and assigns them to the model.
    This is the inverse operation of `make_functional_deprecated_v1`.
    """
    # [FUNCTION DOC] Loads weights (and optionally buffers) back into model in place and
    # returns the model. Raises AssertionError if a names sequence length differs from its
    # tensors sequence length. Buffers are only loaded when non-empty.
    if len(weight_names) != len(weights):
        raise AssertionError(
            f"len(weight_names)={len(weight_names)} != len(weights)={len(weights)}"
        )
    load_weights(model, weight_names, weights)
    if len(buffers) > 0:
        if len(buffer_names) != len(buffers):
            raise AssertionError(
                f"len(buffer_names)={len(buffer_names)} != len(buffers)={len(buffers)}"
            )
        load_buffers(model, buffer_names, buffers)
    return model


def make_functional_deprecated_v1(
    model: nn.Module,
) -> tuple[tuple[Tensor, ...], Callable[..., Any], tuple[str, ...]]:
    """make_functional_deprecated_v1(model) -> weights, func, weight_names

    Given an nn.Module, make_functional_deprecated_v1 extracts the state (weights)
    and returns a functional version of the model, `func`. This makes
    it so that it is possible use transforms over the parameters of
    `model`.

    `func` can be invoked as follows:
    ```
    x = torch.randn(4, 3)
    model = nn.Linear(3, 3)
    weights, func, _ = make_functional_deprecated_v1(model)
    func(weights, (x,))
    ```

    And here is an example of applying the grad transform:
    ```
    x = torch.randn(4, 3)
    model = nn.Linear(3, 3)
    weights, _, func = make_functional_deprecated_v1(model)
    grad_weights = grad(func)(weights, (x,))
    ```

    To put the state back into a model, use `load_state`.
    """
    # [FUNCTION DOC] Deprecated. Extracts the weights from model in place (model is left
    # without parameters) and returns (weights, func, weight_names). func takes (weights,
    # data tuple), deep-copies the model, loads the weights and calls it on the data.
    # Raises RuntimeError if the model has buffers.
    buffers = list(model.buffers())
    if len(buffers) > 0:
        raise RuntimeError(
            "make_functional_deprecated_v1(model): `model` has buffers. Please use "
            "make_functional_with_buffers_deprecated_v1(model) instead."
        )
    weights, descriptors, _ = extract_weights(model)

    def fun(weights: tuple[Tensor, ...], data: tuple[Any, ...]) -> Any:
        # [FUNCTION DOC] HELPER: Runs a deep copy of the stripped model with the given
        # weights loaded, calling it with the positional args in data; returns its output.
        mutable_model = copy.deepcopy(model)
        load_weights(mutable_model, descriptors, weights)
        return mutable_model(*data)

    return weights, fun, descriptors


def make_functional_with_buffers_deprecated_v1(
    model: nn.Module,
) -> tuple[
    tuple[Tensor, ...],
    tuple[Tensor, ...],
    Callable[..., Any],
    tuple[str, ...],
    tuple[str, ...],
]:
    """make_functional_with_buffers_deprecated_v1(model) -> weights, buffers, func, weight_names, buffer_names

    Given an nn.Module, make_functional_with_buffers_deprecated_v1 extracts the state (weights and buffers)
    and returns a functional version of the model, `func`.

    `func` can be invoked as follows:
    ```
    x = torch.randn(4, 3)
    model = nn.Linear(3, 3)
    weights, buffers, func, _, _ = make_functional_with_buffers_deprecated_v1(model)
    func(weights, buffers, (x,))
    ```

    And here is an example of applying the grad transform:
    ```
    x = torch.randn(4, 3)
    model = nn.Linear(3, 3)
    weights, buffers, func, _, _ = make_functional_with_buffers_deprecated_v1(model)
    func(weights, buffers, (x,))
    grad_weights = grad(func)(weights, buffers, (x,))
    ```

    To put the state back into a model, use `load_state`.
    """
    # [FUNCTION DOC] Deprecated. Extracts weights and buffers from model in place and
    # returns (weights, buffers, func, weight_names, buffer_names). func takes (weights,
    # buffers, data tuple), deep-copies the model, loads both and calls it on the data.
    weights, weight_descriptors, _ = extract_weights(model)
    buffers, buf_descriptors, _ = extract_buffers(model)

    def fun(
        weights: tuple[Tensor, ...],
        buffers: tuple[Tensor, ...],
        data: tuple[Any, ...],
    ) -> Any:
        # [FUNCTION DOC] HELPER: Runs a deep copy of the stripped model with the given
        # weights and buffers loaded, calling it with the positional args in data.
        mutable_model = copy.deepcopy(model)
        load_weights(mutable_model, weight_descriptors, weights)
        load_buffers(mutable_model, buf_descriptors, buffers)
        return mutable_model(*data)

    return weights, buffers, fun, weight_descriptors, buf_descriptors


class FunctionalModuleWithBuffers(nn.Module):
    # [CLASS DOC] Callable wrapper around a stateless (meta-placeholder) copy of a model.
    # State: stateless_model, param_names, buffer_names, and all_names_map (parameter and
    # buffer alias groups merged; params first). Calling it with params and buffers
    # temporarily installs them on the model, runs forward, then restores placeholders.
    """
    This is the callable object returned by :func:`make_functional_with_buffers`.
    """

    def __init__(
        self,
        stateless_model: nn.Module,
        param_names: tuple[str, ...],
        buffer_names: tuple[str, ...],
        param_names_map: dict[str, list[str]],
        buffer_names_map: dict[str, list[str]],
    ) -> None:
        # [FUNCTION DOC] Stores the stateless model and name tuples, and builds
        # all_names_map by merging the parameter map with the buffer map.
        super().__init__()
        self.stateless_model = stateless_model
        self.param_names = param_names
        self.buffer_names = buffer_names

        self.all_names_map = dict(param_names_map)
        self.all_names_map.update(buffer_names_map)

    @staticmethod
    def _create_from(
        model: nn.Module, disable_autograd_tracking: bool = False
    ) -> tuple[FunctionalModuleWithBuffers, tuple[Tensor, ...], tuple[Tensor, ...]]:
        # [FUNCTION DOC] Deep-copies model, strips its params and buffers from the copy and
        # returns (wrapper module, params, buffers). The original model is untouched. If
        # disable_autograd_tracking is true, the params get requires_grad turned off.
        # TODO: We don't need to copy the model to create a stateless copy
        model_copy = copy.deepcopy(model)
        params, param_names, param_names_map = extract_weights(model_copy)
        buffers, buffer_names, buffer_names_map = extract_buffers(model_copy)
        if disable_autograd_tracking:
            for param in params:
                param.requires_grad_(False)
        return (
            FunctionalModuleWithBuffers(
                model_copy, param_names, buffer_names, param_names_map, buffer_names_map
            ),
            params,
            buffers,
        )

    def forward(
        self,
        params: Iterable[Tensor],
        buffers: Iterable[Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # [FUNCTION DOC] Runs the model with the given params and buffers (params first, in
        # name-map order) swapped in, passing extra args and kwargs through. Returns the
        # model output; the placeholders are restored afterwards even if an error occurs.
        # Temporarily load the state back onto self.stateless_model
        old_state = _swap_state(
            self.stateless_model,
            self.all_names_map,
            tuple(params) + tuple(buffers),
        )
        try:
            return self.stateless_model(*args, **kwargs)
        finally:
            # Remove the loaded state on self.stateless_model
            _swap_state(self.stateless_model, self.all_names_map, old_state)


class FunctionalModule(nn.Module):
    # [CLASS DOC] Callable wrapper around a stateless copy of a buffer-free model.
    # State: stateless_model, param_names, names_map (alias groups). Calling it with
    # params temporarily installs them, runs forward, then restores placeholders.
    """
    This is the callable object returned by :func:`make_functional`.
    """

    def __init__(
        self,
        stateless_model: nn.Module,
        param_names: tuple[str, ...],
        names_map: dict[str, list[str]],
    ) -> None:
        # [FUNCTION DOC] Stores the stateless model, parameter names and alias map.
        super().__init__()
        self.stateless_model = stateless_model
        self.param_names = param_names
        self.names_map = names_map

    @staticmethod
    def _create_from(
        model: nn.Module, disable_autograd_tracking: bool = False
    ) -> tuple[FunctionalModule, tuple[Tensor, ...]]:
        # [FUNCTION DOC] Deep-copies model, strips its params from the copy and returns
        # (wrapper module, params). Original model untouched. If disable_autograd_tracking
        # is true, the params get requires_grad turned off.
        # TODO: We don't need to copy the model to create a stateless copy
        model_copy = copy.deepcopy(model)
        params, param_names, names_map = extract_weights(model_copy)
        if disable_autograd_tracking:
            for param in params:
                param.requires_grad_(False)
        return FunctionalModule(model_copy, param_names, names_map), params

    def forward(self, params: Iterable[Tensor], *args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] Runs the model with the given params swapped in, passing extra args
        # and kwargs through. Returns the output; placeholders are restored in a finally step.
        # Temporarily load the state back onto self.stateless_model
        old_state = _swap_state(self.stateless_model, self.names_map, params)
        try:
            return self.stateless_model(*args, **kwargs)
        finally:
            # Remove the loaded state on self.stateless_model
            _swap_state(self.stateless_model, self.names_map, old_state)


def make_functional(
    model: nn.Module, disable_autograd_tracking: bool = False
) -> tuple[FunctionalModule, tuple[Tensor, ...]]:
    """make_functional(model, disable_autograd_tracking=False) -> func, params

    Given a ``torch.nn.Module``, :func:`make_functional` extracts the state
    (params) and returns a functional version of the model, ``func``. This
    makes it so that it is possible use transforms over the parameters of
    ``model``.

    ``func`` can be invoked as follows:

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional

        x = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params = make_functional(model)
        func(params, x)

    And here is an example of applying the grad transform over the parameters
    of a model.

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional, grad

        x = torch.randn(4, 3)
        t = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params = make_functional(model)


        def compute_loss(params, x, t):
            y = func(params, x)
            return nn.functional.mse_loss(y, t)


        grad_weights = grad(compute_loss)(params, x, t)

    If the model has any buffers, please use :func:`make_functional_with_buffers` instead.

    Args:
        model (torch.nn.Module): Input model.
        disable_autograd_tracking (bool): Flag to disable gradients tracking for output parameters.
            The returned params are unrelated to the set of params from the original model. If False (default),
            the params will have ``requires_grad=True`` on them (aka they will be trackable with regular
            PyTorch autograd), matching the requires_grad-ness of the params from the original model.
            Otherwise, the returned params will have ``requires_grad=False``. Default, False.
            If you plan on using regular PyTorch autograd (e.g., if you want to call ``.backward()`` or
            ``torch.autograd.grad()``, then set ``disable_autograd_tracking=False``.
            Otherwise, if you're only planning on using functorch's gradient transforms,
            then please set ``disable_autograd_tracking=True`` to avoid unnecessarily tracking
            history with PyTorch autograd.

    """
    # [FUNCTION DOC] Returns (FunctionalModule, params) for a model without buffers; the
    # original model is not modified (a copy is stripped). Raises RuntimeError if the
    # model has any buffers.
    buffers = list(model.buffers())
    if len(buffers) > 0:
        raise RuntimeError(
            "make_functional(model): `model` has buffers. Please use "
            "make_functional_with_buffers(model) instead."
        )
    return FunctionalModule._create_from(
        model, disable_autograd_tracking=disable_autograd_tracking
    )


def make_functional_with_buffers(
    model: nn.Module, disable_autograd_tracking: bool = False
) -> tuple[FunctionalModuleWithBuffers, tuple[Tensor, ...], tuple[Tensor, ...]]:
    """make_functional_with_buffers(model, disable_autograd_tracking=False) -> func, params, buffers

    Given a ``torch.nn.Module``, make_functional_with_buffers extracts the
    state (params and buffers) and returns a functional version of the model
    ``func`` that can be invoked like a function.

    ``func`` can be invoked as follows:

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional_with_buffers

        x = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params, buffers = make_functional_with_buffers(model)
        func(params, buffers, x)

    And here is an example of applying the grad transform over the parameters
    of a model:

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional_with_buffers, grad

        x = torch.randn(4, 3)
        t = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params, buffers = make_functional_with_buffers(model)


        def compute_loss(params, buffers, x, t):
            y = func(params, buffers, x)
            return nn.functional.mse_loss(y, t)


        grad_weights = grad(compute_loss)(params, buffers, x, t)

    Args:
        model (torch.nn.Module): Input model.
        disable_autograd_tracking (bool): Flag to disable gradients tracking for output parameters.
            The returned params are unrelated to the set of params from the original model. If False (default),
            the params will have ``requires_grad=True`` on them (aka they will be trackable with regular
            PyTorch autograd), matching the requires_grad-ness of the params from the original model.
            Otherwise, the returned params will have ``requires_grad=False``. Default, False.
            If you plan on using regular PyTorch autograd (e.g., if you want to call ``.backward()`` or
            ``torch.autograd.grad()``, then set ``disable_autograd_tracking=False``.
            Otherwise, if you're only planning on using functorch's gradient transforms,
            then please set ``disable_autograd_tracking=True`` to avoid unnecessarily tracking
            history with PyTorch autograd.

    """
    # [FUNCTION DOC] Returns (FunctionalModuleWithBuffers, params, buffers) built from a
    # copy of model; the original is not modified.
    return FunctionalModuleWithBuffers._create_from(
        model, disable_autograd_tracking=disable_autograd_tracking
    )


def transpose_stack(
    tuple_of_tuple_of_tensors: tuple[tuple[Tensor, ...], ...],
) -> tuple[Tensor, ...]:
    # [FUNCTION DOC] Transposes a tuple (per model) of tuples (per tensor) and stacks each
    # group along a new leading dimension, detaching the result. Returns one stacked
    # tensor per position.
    tuple_of_tuple_of_tensors = tuple(zip(*tuple_of_tuple_of_tensors))
    results = tuple(
        torch.stack(shards).detach() for shards in tuple_of_tuple_of_tensors
    )
    return results


def combine_state_for_ensemble(
    models: Sequence[nn.Module],
) -> tuple[FunctionalModuleWithBuffers, tuple[Tensor, ...], tuple[Tensor, ...]]:
    """combine_state_for_ensemble(models) -> func, params, buffers

    Prepares a list of torch.nn.Modules for ensembling with :func:`vmap`.

    Given a list of ``M`` ``nn.Modules`` of the same class, stacks all of their
    parameters and buffers together to make ``params`` and ``buffers``.
    Each parameter and buffer in the result will have an additional dimension
    of size ``M``.

    :func:`combine_state_for_ensemble` also returns ``func``, a functional
    version of one of the models in :attr:`models`. One cannot directly run
    ``func(params, buffers, *args, **kwargs)`` directly, you probably want to
    use ``vmap(func, ...)(params, buffers, *args, **kwargs)``

    Here's an example of how to ensemble over a very simple model:

    .. code-block:: python

        num_models = 5
        batch_size = 64
        in_features, out_features = 3, 3
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        data = torch.randn(batch_size, 3)

        fmodel, params, buffers = combine_state_for_ensemble(models)
        output = vmap(fmodel, (0, 0, None))(params, buffers, data)

        assert output.shape == (num_models, batch_size, out_features)

    .. warning::
        All of the modules being stacked together must be the same (except for
        the values of their parameters/buffers). For example, they should be in the
        same mode (training vs eval).

        This API is subject to change -- we're investigating better ways to
        create ensembles and would love your feedback how to improve this.
    """
    # [FUNCTION DOC] Makes each model functional (with buffers) and stacks their params and
    # buffers along a new leading dimension. Returns (functional module of the first model,
    # stacked params, stacked buffers). Raises RuntimeError if models is empty, training
    # modes differ, or classes differ.
    if len(models) == 0:
        raise RuntimeError(
            "combine_state_for_ensemble: Expected at least one model, got 0."
        )
    if not (all(m.training for m in models) or all(not m.training for m in models)):
        raise RuntimeError(
            "combine_state_for_ensemble: Expected all models to "
            "have the same training/eval mode."
        )
    model0_typ = type(models[0])
    if not all(type(m) is model0_typ for m in models):
        raise RuntimeError(
            "combine_state_for_ensemble: Expected all models to be of the same class."
        )
    funcs, params, buffers = zip(
        *[make_functional_with_buffers(model) for model in models]
    )
    params = transpose_stack(params)
    buffers = transpose_stack(buffers)
    return funcs[0], params, buffers


def functional_init(
    model_class: type[nn.Module],
    ensemble_shape: tuple[()] | tuple[int, ...] = (),
    device: torch.types.Device = "cpu",
) -> Callable[..., tuple[tuple[Tensor, ...], Callable[..., Any], tuple[str, ...]]]:
    # [FUNCTION DOC] Returns a factory that builds model_class instances (moved to device)
    # and returns (weights, func, names) in the deprecated v1 form. With an empty
    # ensemble_shape one model is built; with one element, that many models are built
    # and their weights stacked.
    def wrapped(
        *args: Any, **kwargs: Any
    ) -> tuple[tuple[Tensor, ...], Callable[..., Any], tuple[str, ...]]:
        # [FUNCTION DOC] HELPER: Constructs the model(s) with args and kwargs. Raises
        # ValueError if ensemble_shape has 2+ entries or the model count is not positive.
        if len(ensemble_shape) >= 2:
            raise ValueError("NYI: ensemble_shape with more than 1 element")
        if len(ensemble_shape) == 0:
            model = model_class(*args, **kwargs).to(device)
            return make_functional_deprecated_v1(model)
        num_models = ensemble_shape[0]  # type: ignore[misc]
        if num_models <= 0:
            raise ValueError(f"num_models {num_models} should be > 0")
        # NB: Not very efficient, more of a POC
        models = tuple(
            model_class(*args, **kwargs).to(device) for _ in range(num_models)
        )
        _, fn, names = make_functional_deprecated_v1(model_class(*args, **kwargs))
        weights = tuple(make_functional_deprecated_v1(model)[0] for model in models)
        weights = tuple(zip(*weights))
        weights = tuple(torch.stack(shards).detach() for shards in weights)
        return weights, fn, names

    return wrapped


def functional_init_with_buffers(
    model_class: type[nn.Module],
    ensemble_shape: tuple[()] | tuple[int, ...] = (),
    device: torch.types.Device = "cpu",
) -> Callable[
    ...,
    tuple[
        tuple[Tensor, ...],
        tuple[Tensor, ...],
        Callable[..., Any],
        tuple[str, ...],
        tuple[str, ...],
    ]
    | tuple[tuple[Tensor, ...], Callable[..., Any], tuple[str, ...]],
]:
    # [FUNCTION DOC] Like functional_init but uses the buffer-aware deprecated v1 form for
    # ensembles, returning (weights, buffers, func, weight_names, buffer_names).
    def wrapped(
        *args: Any, **kwargs: Any
    ) -> (
        tuple[
            tuple[Tensor, ...],
            tuple[Tensor, ...],
            Callable[..., Any],
            tuple[str, ...],
            tuple[str, ...],
        ]
        | tuple[tuple[Tensor, ...], Callable[..., Any], tuple[str, ...]]
    ):
        # [FUNCTION DOC] HELPER: Constructs the model(s) with args and kwargs. Raises
        # ValueError if ensemble_shape has 2+ entries or the model count is not positive.
        # KNOWN ISSUE: with an empty ensemble_shape it uses the buffer-less v1 function, so
        # it returns a 3-tuple and raises RuntimeError for models that have buffers.
        if len(ensemble_shape) >= 2:
            raise ValueError("NYI: ensemble_shape with more than 1 element")
        if len(ensemble_shape) == 0:
            model = model_class(*args, **kwargs).to(device)
            return make_functional_deprecated_v1(model)
        num_models = ensemble_shape[0]  # type: ignore[misc]
        if num_models <= 0:
            raise ValueError(f"num_models {num_models} should be > 0")
        # NB: Not very efficient, more of a POC
        models = tuple(
            model_class(*args, **kwargs).to(device) for _ in range(num_models)
        )
        (
            _,
            _,
            fn,
            weight_names,
            buffer_names,
        ) = make_functional_with_buffers_deprecated_v1(model_class(*args, **kwargs))
        weights, buffers = zip(
            *tuple(
                make_functional_with_buffers_deprecated_v1(model)[:2]
                for model in models
            )
        )
        weights = tuple(zip(*weights))
        weights = tuple(torch.stack(shards).detach() for shards in weights)
        buffers = tuple(zip(*buffers))
        buffers = tuple(torch.stack(shards).detach() for shards in buffers)
        return weights, buffers, fn, weight_names, buffer_names

    return wrapped
