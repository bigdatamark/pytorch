# mypy: allow-untyped-defs
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] Auto-wrapping policies and utilities for FSDP. Provides callable
# policies (always, lambda, transformer-layer and size based), policy
# classes (ModuleWrapPolicy, CustomPolicy) that map modules to wrapping
# kwargs, the enable_wrap context manager with the wrap annotation, and
# the private recursive wrapping machinery. Import-time side effect:
# attaches default exclude and force-leaf module sets as attributes of
# size_based_auto_wrap_policy.
import contextlib
import copy
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable, Sequence
from typing import Any, cast

import torch.nn as nn


__all__ = [
    "always_wrap_policy",
    "lambda_auto_wrap_policy",
    "transformer_auto_wrap_policy",
    "size_based_auto_wrap_policy",
    "enable_wrap",
    "wrap",
    "CustomPolicy",
    "ModuleWrapPolicy",
]


# NOTE: We intentionally keep this function simple and isolate the complexity
# to `fn` to enable using this function generically. We may move this to a
# non-FSDP-specific folder and/or make it public in the future.
def _post_order_apply(
    root_module: nn.Module,
    fn: Callable[[nn.Module], nn.Module | None],
):
    # [FUNCTION DOC] HELPER: Applies fn to every module under root_module in post-
    # order (children first), visiting each shared module once. If fn
    # returns a module, it replaces the original in its parent by
    # attribute assignment; None leaves it unchanged. Raises
    # AssertionError if a replacement is requested for the root (no
    # parent or name) or fn returns something other than None or a
    # module.
    """
    This applies ``fn`` to every module in the module tree of ``root_module``
    following a post-order traversal. If ``fn`` returns an :class:`nn.Module`,
    then this replaces the original module with the newly returned one in the
    tree. Otherwise, ``fn`` should return ``None``, in which case the module is
    not changed.
    """
    # Track visited modules to avoid visiting shared modules multiple times
    visited_modules: set[nn.Module] = {root_module}

    def _post_order_apply_inner(
        module: nn.Module,
        module_name: str,
        parent_module: nn.Module | None,
    ):
        # [FUNCTION DOC] HELPER: Recursive step of the post-order traversal: processes
        # unvisited children first, then applies fn to the module and
        # swaps in a returned replacement on the parent. Raises
        # AssertionError on invalid replacement requests.
        for child_module_name, child_module in module.named_children():
            if child_module not in visited_modules:
                visited_modules.add(child_module)
                _post_order_apply_inner(child_module, child_module_name, module)
        optional_module = fn(module)
        if optional_module is not None:
            if not isinstance(parent_module, nn.Module):
                raise AssertionError(
                    "Non-root modules should have their parent module set but got "
                    f"{parent_module} for {module}"
                )
            if not module_name:
                raise AssertionError(
                    "Non-root modules should have their module name set but got "
                    f"an empty module name for {module}"
                )
            if not isinstance(optional_module, nn.Module):
                raise AssertionError(
                    f"fn should return None or an nn.Module but got {optional_module}"
                )
            setattr(parent_module, module_name, optional_module)

    _post_order_apply_inner(root_module, "", None)


def _construct_wrap_fn(
    root_module: nn.Module,
    target_module_to_kwargs: dict[nn.Module, dict[str, Any]],
    fsdp_fn: Callable,
) -> Callable[[nn.Module], nn.Module | None]:
    # [FUNCTION DOC] HELPER: Builds the function used with the post-order traversal:
    # for a module that is a key of the target mapping and is not the
    # root, it calls fsdp_fn with the module and its kwargs and returns
    # the wrapped result; otherwise returns None. The root is never
    # wrapped here because the caller handles it.
    """
    This constructs the "wrap" function to pass to :func:`_post_order_apply`
    based on ``target_module_to_kwargs``, which should be constructed from the
    wrapping policy.
    """

    def fn(module: nn.Module) -> nn.Module | None:
        # [FUNCTION DOC] HELPER: Returns the wrapped module for targeted non-root
        # modules, else None.
        # Explicitly avoid wrapping the root module since for FSDP, it is
        # handled by the caller
        if module in target_module_to_kwargs and module is not root_module:
            kwargs = target_module_to_kwargs[module]
            return fsdp_fn(module, **kwargs)
        return None

    return fn


def _run_mixed_precision_override_policy(
    root_module: nn.Module,
    module_classes: Iterable[type[nn.Module]],
    ignored_modules: set[nn.Module],
    root_kwargs: dict[str, Any],
    target_module_to_kwargs: dict[nn.Module, dict[str, Any]],
):
    # [FUNCTION DOC] HELPER: Mixed-precision override policy: for every non-ignored
    # module (found by walking the root) that is an instance of the
    # given classes, ensures an entry in the target mapping (new
    # entries reuse the root kwargs object, not a copy) and sets that
    # entry's mixed_precision to None. Mutates and returns the target
    # mapping.
    module_classes_tuple = tuple(set(module_classes))
    for module in root_module.modules():
        if module in ignored_modules:
            continue
        elif isinstance(module, module_classes_tuple):
            # This policy overrides any existing policy
            if module not in target_module_to_kwargs:
                # Only inherit from the root kwargs if not already specified
                target_module_to_kwargs[module] = root_kwargs
            target_module_to_kwargs[module]["mixed_precision"] = None
    return target_module_to_kwargs


def always_wrap_policy(*args, **kwargs) -> bool:
    # [FUNCTION DOC] Auto wrap policy that ignores all arguments and always returns
    # True, so every submodule is wrapped.
    """
    A simple recursive wrap policy that always returns ``True``. This means
    that every submodule is wrapped by the wrapper class in
    :func:`_recursive_wrap`.
    """
    return True


class _Policy(ABC):
    # [CLASS DOC] HELPER: Abstract base for wrapping policies. A policy decides, for a
    # root module, ignored modules and root kwargs, which modules get the
    # API applied and with which kwargs, via _run_policy.
    """
    This defines an abstract base class that represents a policy for applying
    a module-level API.
    """

    @abstractmethod
    def _run_policy(
        self,
        root_module: nn.Module,
        ignored_modules: set[nn.Module],
        root_kwargs: dict[str, Any],
    ) -> dict[nn.Module, dict[str, Any]]:
        # [FUNCTION DOC] HELPER: Abstract: subclasses return a mapping from each
        # target module to its kwargs.
        """
        This should return a dict ``target_module_to_kwargs`` that maps from
        each target module to wrap to its kwargs.
        """
        ...


def _module_wrap_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    module_classes: set[type[nn.Module]],
) -> bool:
    # [FUNCTION DOC] HELPER: Auto wrap policy: while recursing it returns True so
    # traversal continues; when deciding for a module it returns
    # whether the module is an instance of any given module class. The
    # nonwrapped_numel argument is unused.
    """
    This auto wrap policy wraps every module that is an instance of any type in
    ``module_classes`` as its own FSDP instance. The root module given by
    ``module`` is always wrapped as an FSDP instance regardless. Since the
    wrapping proceeds bottom up, each FSDP instance manages the parameters in
    its subtree excluding any already managed by a child FSDP instance.

    Args:
        module (nn.Module): Current module being considered.
        recurse (bool): If ``False``, then this function must decide whether
            ``module`` should be wrapped as an FSDP instance or not. If
            ``True``, then the function is still recursing down the module
            tree as a part of the DFS.
        nonwrapped_numel (int): Parameter numel not yet wrapped.
        module_classes (Set[Type[nn.Module]]): Set of module classes that are
            wrapped as FSDP instances.

    Returns:
        ``True`` if ``recurse=True``, and whether ``module`` should be wrapped
        if ``recurse=False``.
    """
    if recurse:
        return True  # always recurse
    return isinstance(module, tuple(module_classes))


class ModuleWrapPolicy(_Policy):
    # [CLASS DOC] Policy that targets every non-ignored module that is an instance of
    # the given module classes, each getting a shallow copy of the root
    # kwargs. Stores the class set and its string form. Also callable as a
    # legacy auto wrap policy function.
    """
    This policy applies to every module of the specified module classes,
    passing in the kwargs given to the root.
    """

    def __init__(self, module_classes: Iterable[type[nn.Module]]):
        # [FUNCTION DOC] Stores the module classes as a set (and its string form for
        # repr).
        module_classes_set = set(module_classes)
        self._module_classes = module_classes_set
        self._module_classes_str = str(module_classes_set)

    def _run_policy(
        self,
        root_module: nn.Module,
        ignored_modules: set[nn.Module],
        root_kwargs: dict[str, Any],
    ) -> dict[nn.Module, dict[str, Any]]:
        # [FUNCTION DOC] Walks the root's modules, skipping ignored ones, and returns
        # a mapping from each instance of the configured classes to a
        # shallow copy of the root kwargs.
        module_classes = tuple(self._module_classes)
        target_module_to_kwargs: dict[nn.Module, dict[str, Any]] = {}
        for module in root_module.modules():
            if module in ignored_modules:
                continue
            elif isinstance(module, module_classes):
                # Shallow copy to avoid coupling changes across modules
                target_module_to_kwargs[module] = copy.copy(root_kwargs)
        return target_module_to_kwargs

    def __call__(self, module, recurse, *args, **kwargs):
        # [FUNCTION DOC] Legacy callable form: returns True when recursing, otherwise
        # whether the module is an instance of the configured classes.
        # Extra arguments are ignored.
        # nonwrapped_numel is not used.
        return _module_wrap_policy(
            module, recurse, nonwrapped_numel=-1, module_classes=self._module_classes
        )

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns the default representation followed by the string
        # form of the module class set.
        return super().__repr__() + f"({self._module_classes_str})"


class CustomPolicy(_Policy):
    # [CLASS DOC] Policy driven by a user function mapping each module to False
    # (skip), True (apply with the root kwargs) or a kwargs dictionary
    # (apply with the root kwargs overridden by it). An empty dictionary
    # counts as skip.
    """
    This policy takes in a lambda function that maps a given ``nn.Module`` to
    either ``False``, ``True``, or a kwarg dictionary.
    - If the function returns ``False`` or an empty dictionary, then the module
      does not have the API applied.
    - If the function returns ``True``, then the module has the API applied
      with the root's kwargs.
    - If the function returns a non-empty dictionary, then the module has the
      API applied, and the dictionary overrides the root's kwargs.

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> model = init_transformer_model(...)
        >>> def lambda_fn(module: nn.Module):
        >>>     if module is model.lm_head:
        >>>         return {"sharding_strategy": ShardingStrategy.SHARD_GRAD_OP}
        >>>     elif isinstance(module, TransformerBlock):
        >>>         return True
        >>>     return False
        >>> policy = CustomPolicy(lambda_fn)
        >>> fsdp_model = FSDP(model, auto_wrap_policy=policy)
    """

    def __init__(self, lambda_fn: Callable[[nn.Module], bool | dict[str, Any]]):
        # [FUNCTION DOC] Stores the user function.
        self._lambda_fn = lambda_fn

    def _run_policy(
        self,
        root_module: nn.Module,
        ignored_modules: set[nn.Module],
        root_kwargs: dict[str, Any],
    ) -> dict[nn.Module, dict[str, Any]]:
        # [FUNCTION DOC] Calls the user function on each non-ignored module and builds
        # the mapping of targeted modules to kwargs (a copy of the root
        # kwargs, updated by a returned dictionary). Raises ValueError
        # if the function returns something other than a boolean or
        # dictionary.
        target_module_to_kwargs: dict[nn.Module, dict[str, Any]] = {}
        for module in root_module.modules():
            if module in ignored_modules:
                continue
            res = self._lambda_fn(module)
            if not isinstance(res, (dict, bool)):
                raise ValueError(
                    "The lambda_fn passed to CustomPolicy should return "
                    f"False/True or a kwarg dict, but it returned {res}"
                )
            if not res:
                continue
            kwargs = copy.copy(root_kwargs)
            if isinstance(res, dict):
                # Override the root kwargs with the ones specified by the
                # lambda function
                kwargs.update(res)
            target_module_to_kwargs[module] = kwargs
        return target_module_to_kwargs


def lambda_auto_wrap_policy(
    module: nn.Module, recurse: bool, nonwrapped_numel: int, lambda_fn: Callable
) -> bool:
    # [FUNCTION DOC] Legacy auto wrap policy: returns True while recursing; otherwise
    # returns the result of the user function on the module.
    # nonwrapped_numel is unused.
    """
    A convenient auto wrap policy to wrap submodules based on an arbitrary user
    function. If `lambda_fn(submodule) == True``, the submodule will be wrapped as
    a `wrapper_cls` unit.

    Return if a module should be wrapped during auto wrapping.

    The first three parameters are required by :func:`_recursive_wrap`.

    Args:
        module (nn.Module): Current module being considered.
        recurse (bool): If ``False``, then this function must decide whether
            ``module`` should be wrapped as an FSDP instance or not. If
            ``True``, then the function is still recursing down the module
            tree as a part of the DFS.
        nonwrapped_numel (int): Parameter numel not yet wrapped.

        lambda_fn (Callable[[nn.Module], bool]): If this returns ``True``, then
            this module will be wrapped.
    """
    if recurse:
        return True  # always recurse
    return lambda_fn(module)


def transformer_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    transformer_layer_cls: set[type[nn.Module]],
) -> bool:
    # [FUNCTION DOC] Legacy auto wrap policy equivalent to the module-class policy
    # with the transformer layer classes; useful so that shared
    # embeddings end up in the same FSDP instance. Returns True while
    # recursing, else whether the module is an instance of those
    # classes.
    """
    See :func:`_module_wrap_policy`, where ``transformer_layer_cls`` is the
    same as ``module_classes``. Note that shared parameters must be wrapped in
    the same FSDP instance, so this auto wrap policy can help wrap shared
    embeddings into the same FSDP instance for transformer models.
    """
    return _module_wrap_policy(module, recurse, nonwrapped_numel, transformer_layer_cls)


def _wrap_module_cls_individually(
    module: nn.Module, module_classes: Sequence[type], recurse: bool, *args, **kwargs
):
    # [FUNCTION DOC] HELPER: Legacy auto wrap policy: returns True while recursing,
    # otherwise whether the module is an instance of any of the given
    # classes. Extra arguments are ignored.
    if recurse:
        # always recurse
        return True
    else:
        # if not recursing, decide whether we should wrap based on whether the type of module
        # is in `module_classes`.
        return isinstance(module, tuple(module_classes))


def _or_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    policies,
) -> bool:
    # [FUNCTION DOC] HELPER: Combines several legacy policies: returns True if any
    # policy, called with the module, recurse flag and unwrapped
    # parameter count, returns True.
    """
    A policy that wraps ``module`` if any policy in the passed in iterable of
    ``policies`` returns ``True``.
    """
    return any(
        policy(module=module, recurse=recurse, nonwrapped_numel=nonwrapped_numel)
        for policy in policies
    )


def size_based_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    # Additional custom arguments
    min_num_params: int = int(1e8),
    force_leaf_modules: set[type[nn.Module]] | None = None,
    exclude_wrap_modules: set[type[nn.Module]] | None = None,
) -> bool:
    # [FUNCTION DOC] Size based auto wrap policy. Uses the given or default (function
    # attribute) force-leaf and exclude sets. A module is large when
    # its not-yet-wrapped parameter count is at least min_num_params.
    # While recursing it returns True if the module is large and not a
    # force-leaf type (keep descending); when deciding it returns True
    # if the module is large and not an excluded type (wrap it).
    """
    A size-based auto wrap policy.

    Args:
        module (nn.Module): Current module being considered.
        recurse (bool): If ``False``, then this function must decide whether
            ``module`` should be wrapped as an FSDP instance or not. If
            ``True``, then the function is still recursing down the module
            tree as a part of the DFS.
        nonwrapped_numel (int): Parameter numel not yet wrapped.

        min_num_params (int): Customizable policy input that controls the size
            threshold over which a module is ready to be wrapped. This is in
            units of numel.
        force_leaf_modules (Optional[set[type[nn.Module]]]): Set of module types to keep
            as leaves, i.e. their children will never be wrapped.
        exclude_wrap_modules (Optional[set[type[nn.Module]]]): Set of module types to be
            excluded in wrapping.

    Returns:
        Whether ``module`` should be wrapped.
    """
    force_leaf_modules = (
        size_based_auto_wrap_policy.FORCE_LEAF_MODULES  # type: ignore[attr-defined]
        if force_leaf_modules is None
        else force_leaf_modules
    )
    exclude_wrap_modules = (
        size_based_auto_wrap_policy.EXCLUDE_WRAP_MODULES  # type: ignore[attr-defined]
        if exclude_wrap_modules is None
        else exclude_wrap_modules
    )

    # Keep the argument `min_num_params` for BC for now, but it represents the
    # minimum non-wrapped *numel* before triggering a wrapping
    min_nonwrapped_numel = min_num_params
    is_large = nonwrapped_numel >= min_nonwrapped_numel
    if recurse:
        # We should recurse if the module is big enough but not in force_leaf_modules list.
        return is_large and not isinstance(module, tuple(force_leaf_modules))
    else:
        # If we are not recursing, determine if we should wrap.
        return is_large and not isinstance(module, tuple(exclude_wrap_modules))


# Set those defaults to the size_based_auto_wrap_policy function. Make them easy to be imported.
size_based_auto_wrap_policy.EXCLUDE_WRAP_MODULES = {nn.ModuleList, nn.ModuleDict}  # type: ignore[attr-defined]
size_based_auto_wrap_policy.FORCE_LEAF_MODULES = {nn.MultiheadAttention}  # type: ignore[attr-defined]


@contextlib.contextmanager
def enable_wrap(
    *, wrapper_cls: Any, **wrapper_kwargs: Any
) -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager that turns on the global auto-wrap context with
    # the given wrapper class and keyword arguments, so wrap calls
    # inside the block wrap modules with them. Restores the global
    # state on exit. Raises NotImplementedError if nested inside
    # another such context.
    """
    Context manager to wrap modules using a wrapper.

    Useful for when you'd like to apply the same configuration arguments to all
    child modules that you wrap. A particularly important use case is wrapping
    large layers so that they get sharded (in-place) during initialization, to
    avoid running out of system memory. Large layers can indicate that they
    should be sharded via the ``wrap`` annotation and this context manager can
    provide the exact configuration for these nested instances.

    Usage::

        with enable_wrap(wrapper_cls, **params):
            # Wraps layer in FSDP by default if within context
            self.l1 = wrap(torch.nn.Linear(5, 5))

    Args:
        wrapper_cls:
            Class that `wrap` annotation will `wrap` modules with, such as
            `FullyShardedDataParallel`.
        **wrapper_kwargs:
            Configuration settings that will be passed to all ``wrap``
            instances inside the context
    """
    kwargs = {
        "wrapper_cls": wrapper_cls,
        **wrapper_kwargs,
    }
    with _ConfigAutoWrap(**kwargs):
        yield


def wrap(module: nn.Module, **wrap_overrides: Any) -> nn.Module:
    # [FUNCTION DOC] Wraps the module with the active context's wrapper class, using
    # the context kwargs updated by the given overrides; outside an
    # enable_wrap context returns the module unchanged. Raises
    # AssertionError if the context has no wrapper class.
    """
    Annotate that a module should be wrapped. Annotated modules will only be
    wrapped if inside of an :func:`enable_wrap` context manager. This allows
    a module to be initialized both with and without a wrapper without code
    change.

    The class that this function wraps the passed in ``nn.Module`` with is the
    passed in ``wrapper_cls`` argument into ``enable_wrap``. Both
    ``enable_wrap`` and ``wrap`` can take in kwargs specifying how to construct
    the ``wrapper_cls`` instance. In the case of duplicate kwargs in
    ``enable_wrap`` and ``wrap``, the argument passed into ``wrap`` will be
    respected.

    Usage::

        with enable_wrap(wrapper_cls=FSDP, **fsdp_config):
            # Wraps layer in FSDP by default if within context
            self.l1 = wrap(torch.nn.Linear(5, 5))

    Args:
        module (nn.Module): module to wrap (if in :func:`enable_wrap` context)
        **wrap_overrides: configuration overrides that will take priority over
            the values provided by the :func:`enable_wrap` context
    """
    if _ConfigAutoWrap.in_autowrap_context:
        if _ConfigAutoWrap.wrapper_cls is None:
            raise AssertionError("Expected _ConfigAutoWrap.wrapper_cls to be set")

        wrap_overrides = {**_ConfigAutoWrap.kwargs, **wrap_overrides}
        return _wrap(
            module,
            _ConfigAutoWrap.wrapper_cls,
            **wrap_overrides,
        )
    return module


def _wrap(module: nn.Module, wrapper_cls: Callable, **kwargs) -> nn.Module:
    # [FUNCTION DOC] HELPER: Constructs the wrapper around the module with the given
    # kwargs; if the module has a _wrap_overrides attribute, those
    # values take priority over the kwargs. Raises AssertionError if no
    # wrapper class is given.
    if wrapper_cls is None:
        raise AssertionError("Expected wrapper_cls to be set")
    if hasattr(module, "_wrap_overrides"):
        # If module has a _wrap_overrides attribute, we force overriding the
        # FSDP config with these attributes for this module. Currently this
        # is only used to disable mixed precision for BatchNorm when
        # auto_wrapping.
        overrides = {**kwargs, **module._wrap_overrides}  # type: ignore[arg-type, dict-item]
        return wrapper_cls(module, **overrides)

    return wrapper_cls(module, **kwargs)


def _recursive_wrap(
    module: nn.Module,
    auto_wrap_policy: Callable,
    wrapper_cls: Callable,
    ignored_modules: set[nn.Module],
    ignored_params: set[nn.Parameter],
    only_wrap_children: bool = False,
    **kwargs: Any,
) -> tuple[nn.Module, int]:
    # [FUNCTION DOC] HELPER: Recursively wraps submodules for which the auto wrap
    # policy says to wrap, bottom-up, skipping ignored modules and
    # replacing children in place by attribute assignment. Counts
    # parameters not yet wrapped; after wrapping children, the module
    # itself is wrapped when the policy accepts the remaining count
    # (unless only_wrap_children). Returns the resulting module and the
    # number of parameter elements that were wrapped. Raises
    # AssertionError if the policy or wrapper class is missing or a
    # child is already an instance of the wrapper class.
    """
    Wraps submodules of ``module`` for which ``auto_wrap_policy`` returns
    ``True`` with ``wrapper_cls``.

    Args:
        module (nn.Module): Module to recursively wrap.
        auto_wrap_policy (Callable): A callable representing a policy that
            determines which modules to recursively wrap with ``wrapper_cls``.
        ignored_modules (set[torch.nn.Module]): Modules to ignore when
            wrapping.
        ignored_params (set[torch.nn.Parameter]): Parameters to ignore when
            wrapping; these should be the parameters contained in the modules
            in ``ignored_modules``.
    Returns:
        (nn.Module, int):
            ``module`` after wrapping and the numel recursively wrapped.
    """
    if auto_wrap_policy is None:
        raise AssertionError("Must specify auto_wrap_policy.")
    if wrapper_cls is None:
        raise AssertionError("Must specify wrapper_cls")
    # Make sure no child is already wrapped.
    for _, child in module.named_modules():
        if child in ignored_modules:
            continue
        try:
            if isinstance(child, cast(type, wrapper_cls)):
                raise AssertionError(
                    f"Child module {child} is already wrapped by {wrapper_cls}"
                )
        except TypeError:
            # wrapper_cls is a function as opposed to a class type, just bypass above check.
            pass

    # We count all params, assuming none of them are already wrapped.
    nonwrapped_numel = sum(
        p.numel() for p in module.parameters() if p not in ignored_params
    )

    if auto_wrap_policy is None:
        raise AssertionError("Expected auto_wrap_policy to be set")
    if auto_wrap_policy(module=module, recurse=True, nonwrapped_numel=nonwrapped_numel):
        total_wrapped_numel = 0
        # Iterate through the children, recursively wrap if necessary
        for name, child in module.named_children():
            if child in ignored_modules:
                continue
            wrapped_child, num_wrapped_params = _recursive_wrap(
                module=child,
                auto_wrap_policy=auto_wrap_policy,
                wrapper_cls=wrapper_cls,
                ignored_modules=ignored_modules,
                ignored_params=ignored_params,
                **kwargs,
            )
            setattr(module, name, wrapped_child)
            # Keep track of how many parameters have been wrapped
            total_wrapped_numel += num_wrapped_params
        # decide if we need to wrap the current module,
        # since the left over parameters exceed the number of params to wrap
        remainder = nonwrapped_numel - total_wrapped_numel
        if not only_wrap_children and auto_wrap_policy(
            module=module, recurse=False, nonwrapped_numel=remainder
        ):
            # Leaf node or final wrapping of the remainder both happen here.
            return _wrap(module, wrapper_cls, **kwargs), nonwrapped_numel
        else:
            return module, total_wrapped_numel
    return module, 0


class _ConfigAutoWrap:
    # [CLASS DOC] HELPER: Context manager holding the global auto-wrap state as class
    # attributes: whether a context is active, the wrapper class and the
    # wrapper kwargs. Entering enables the context with the instance's
    # kwargs; exiting resets the state. Only one context may be active at
    # a time.
    """
    Helper class to wrap modules based on default config args via a context manager.
    See :func:`enable_wrap` for more information.
    """

    in_autowrap_context: bool = False  # Context flag
    wrapper_cls: Callable | None = None  # The wrapper class
    kwargs: dict[str, Any] = {}  # Wrapper's args

    def __init__(self, **kwargs: dict[str, Any]):
        # [FUNCTION DOC] HELPER: Stores the keyword arguments (including wrapper_cls)
        # for use on enter.
        self.kwargs = kwargs

    @staticmethod
    def enable_autowrap_context(kwargs: Any) -> None:
        # [FUNCTION DOC] HELPER: Activates the global context: records the wrapper
        # class taken out of the kwargs (removing it from them) and the
        # remaining kwargs. Raises NotImplementedError if already
        # active and AssertionError if wrapper_cls is absent. Mutates
        # the passed kwargs and class-level state.
        if _ConfigAutoWrap.in_autowrap_context:
            raise NotImplementedError(
                "You are already within an autowrap context and we currently do not supported nested autowrap."
            )
        _ConfigAutoWrap.in_autowrap_context = True
        # Get and save the wrapper cls for the context.
        if "wrapper_cls" not in kwargs:
            raise AssertionError(
                "Expected to pass in wrapper_cls arg into _ConfigAutoWrap."
            )
        _ConfigAutoWrap.wrapper_cls = cast(Callable, kwargs["wrapper_cls"])
        del kwargs["wrapper_cls"]
        # Save the rest.
        _ConfigAutoWrap.kwargs = kwargs

    @staticmethod
    def disable_autowrap_context() -> None:
        # [FUNCTION DOC] HELPER: Resets the global context: inactive, no wrapper
        # class, empty kwargs.
        _ConfigAutoWrap.in_autowrap_context = False
        _ConfigAutoWrap.wrapper_cls = None
        _ConfigAutoWrap.kwargs = {}

    def __enter__(self) -> None:
        # [FUNCTION DOC] HELPER: Enables the global auto-wrap context using this
        # instance's kwargs.
        self.enable_autowrap_context(self.kwargs)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # [FUNCTION DOC] HELPER: Disables the global auto-wrap context; does not
        # suppress exceptions.
        self.disable_autowrap_context()
