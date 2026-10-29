# mypy: allow-untyped-defs
# [MODULE DOC] Implements the contract decorator for composable distributed APIs: it stores
# per-module state and a registry of applied APIs in module attribute dicts under random
# unique keys, enforces single application per module, and verifies FQNs are unchanged.
# Import-time side effect: generates the random STATE_KEY and REGISTRY_KEY.
import uuid
from collections import OrderedDict
from collections.abc import Callable
from functools import wraps
from typing import Concatenate, Generic, Protocol
from typing_extensions import ParamSpec, TypeVar

import torch
import torch.nn as nn
from torch.distributed._composable_state import _State
from torch.distributed.utils import _get_root_modules


_T = TypeVar("_T", covariant=True)
_P = ParamSpec("_P")


def generate_state_key(string="__composable_api_state_key"):
    # [FUNCTION DOC] Returns the given prefix joined with a fresh random UUID text, giving a
    # unique attribute key.
    return f"{string}_{str(uuid.uuid4())}"


STATE_KEY = generate_state_key()
REGISTRY_KEY = generate_state_key()


# TODO: we can add additional info to RegistryItem to share across APIs. E.g.,
# we can add args and kwargs here, and then we can detect whether fully_shard
# is combined with reentrant activation checkpointing and error out with a clear
# message.
class RegistryItem:
    # [CLASS DOC] Empty marker object recorded in a module's registry per applied composable
    # API (one instance shared by all modules passed in one call); room for future shared info.
    pass


_TState = TypeVar("_TState", bound="_State", covariant=True)
_M = TypeVar("_M", nn.Module, list[nn.Module])


class _ContractFn(Protocol, Generic[_P, _T, _TState]):
    # [CLASS DOC] HELPER: Typing protocol for a function decorated by contract: callable like
    # the original and exposing a state method that returns its per-module state object.
    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _T: ...

    def state(self, module: nn.Module) -> _TState: ...


def contract(
    state_cls: type[_TState] = _State,  # type: ignore[assignment]
) -> Callable[
    [Callable[Concatenate[_M, _P], _M]],
    _ContractFn[Concatenate[_M, _P], _M, _TState],
]:
    # [FUNCTION DOC] Decorator factory taking a state class (default _State) and returning a
    # decorator. The decorated function becomes a wrapper that, per call, collects the input
    # module(s) (root modules only when a sequence is given), creates one shared state and
    # registry item, records them in each module's attribute dict, runs the function, then
    # checks that parameter, buffer and module FQNs (and their order) are unchanged.
    # Wrapper raises AssertionError if the API was already applied to a module, if stored
    # dicts are corrupted, or if the module count changed; RuntimeError on FQN changes.
    # A None return from the function is treated as the input module. The wrapper gains a
    # state attribute that returns the state stored for a module (or None).
    r"""
    Decorate a function as a composable distributed API, where the first
    argument of the function must be an :class:`nn.Module` instance or sequence
    of :class:`nn.Module` instances.

    The decorator verifies that the decorated function does not modify
    fully-qualified names (FQNs) for parameters, buffers, or modules. The
    decorated function can return different module instances than the input
    modules; the FQN invariant will be enforced following the input order.

    When a function ``func`` is decorated by ``@contract()``, a
    ``.state(module: nn.Module)`` method will be installed to the decorated
    function. Then you can retrieve and modify the state on a module by calling
    ``func.state(module)``.

    Example::
        >>> # xdoctest: +SKIP
        >>> import torch.nn as nn
        >>>
        >>> class MyModel(nn.Module):
        >>>     def __init__(self) -> None:
        >>>         super().__init__()
        >>>         self.l1 = nn.Linear(10, 10)
        >>>         self.l2 = nn.Linear(10, 10)
        >>>
        >>>     def forward(self, x):
        >>>         return self.l2(self.l1(x))
        >>>
        >>> @contract()
        >>> def my_feature(module: nn.Module) -> nn.Module:
        >>>     my_feature.state(module).some_state = "any value"
        >>>     return module
        >>>
        >>> model = MyModel()
        >>> my_feature(model.l1)
        >>> assert my_feature.state(model.l1).some_state == "any value"
        >>> my_feature(model.l2)
        >>> model(torch.randn(2, 10)).sum().backward()
    """

    # wraps will make functions decorated with contract() pickleable - needed for integration with torch.package
    @wraps(state_cls)  # type: ignore[arg-type]
    def inner(
        func: Callable[Concatenate[_M, _P], _M],
    ) -> _ContractFn[Concatenate[_M, _P], _M, _TState]:
        # [FUNCTION DOC] HELPER: The actual decorator; wraps func with the FQN-checking
        # wrapper and attaches the state accessor, returning the wrapper.
        @wraps(func)
        def wrapper(
            module: _M,
            *args: _P.args,
            **kwargs: _P.kwargs,
        ) -> _M:
            # [FUNCTION DOC] HELPER: See contract. Registers state and registry entries,
            # calls func, validates unchanged FQNs and returns func's result (or the input).
            inp_module = module
            modules: list[nn.Module]
            if isinstance(module, nn.Module):
                modules = [module]
            else:
                # If the user passes a sequence of modules, then we assume that
                # we only need to insert the state object on the root modules
                # (i.e. those without a parent) among the passed-in modules.
                # pyrefly: ignore [no-matching-overload]
                modules = _get_root_modules(list(module))
            state = state_cls()  # shared across all modules
            registry_item = RegistryItem()  # shared across all modules

            # `func` is allowed to return different module instances than the
            # input modules as long as FQNs are preserved following the input
            # module order
            all_orig_named_params: list[dict[str, nn.Parameter]] = []
            all_orig_named_buffers: list[dict[str, torch.Tensor]] = []
            all_orig_named_modules: list[dict[str, nn.Module]] = []

            for module in modules:
                default_all_state: dict[Callable, _State] = OrderedDict()
                default_registry: dict[str, RegistryItem] = OrderedDict()
                all_state: dict[Callable, _State] = module.__dict__.setdefault(  # type: ignore[call-overload]
                    STATE_KEY, default_all_state
                )
                if not isinstance(all_state, dict):
                    raise AssertionError(
                        f"Distributed composable API states corrupted: {all_state}"
                    )
                registry: dict[str, RegistryItem] = module.__dict__.setdefault(  # type: ignore[call-overload]
                    REGISTRY_KEY, default_registry
                )
                if not isinstance(registry, dict):
                    raise AssertionError(
                        f"Distributed composable API registry corrupted: {registry}"
                    )
                if func in all_state or func.__name__ in registry:
                    raise AssertionError(
                        "Each distinct composable distributed API can only be applied to a "
                        f"module once. {func.__name__} has already been applied to the "
                        f"following module:\n{module}"
                    )
                all_state.setdefault(func, state)
                registry.setdefault(func.__name__, registry_item)

                all_orig_named_params.append(OrderedDict(module.named_parameters()))
                all_orig_named_buffers.append(OrderedDict(module.named_buffers()))
                all_orig_named_modules.append(OrderedDict(module.named_modules()))

            updated = func(inp_module, *args, **kwargs)
            if updated is None:
                updated = inp_module  # type: ignore[assignment]
            updated_modules: list[nn.Module]
            if isinstance(updated, nn.Module):
                updated_modules = [updated]
            else:
                updated_modules = _get_root_modules(list(inp_module))  # type: ignore[arg-type, call-overload]

            all_new_named_params: list[dict[str, nn.Parameter]] = []
            all_new_named_buffers: list[dict[str, torch.Tensor]] = []
            all_new_named_modules: list[dict[str, nn.Module]] = []
            for module in updated_modules:
                all_new_named_params.append(OrderedDict(module.named_parameters()))
                all_new_named_buffers.append(OrderedDict(module.named_buffers()))
                all_new_named_modules.append(OrderedDict(module.named_modules()))

            num_orig_modules = len(all_orig_named_modules)
            num_new_modules = len(all_new_named_modules)
            if num_orig_modules != num_new_modules:
                raise AssertionError(
                    f"{func.__name__} should return the same number of modules as input modules"
                    f"Inputs: {num_orig_modules} modules\n"
                    f"Outputs: {num_new_modules} modules"
                )

            def check_fqn(orig_fqns: list[str], new_fqns: list[str], check_key: str):
                # [FUNCTION DOC] HELPER: Returns silently if the two name lists are equal;
                # otherwise raises RuntimeError reporting added/removed names, or an order change.
                # KNOWN ISSUE: the order-change message prints the (empty) difference sets
                # instead of the original and new FQN lists.
                if orig_fqns == new_fqns:
                    return

                orig_fqn_set, new_fqn_set = set(orig_fqns), set(new_fqns)
                orig_only = orig_fqn_set - new_fqn_set
                new_only = new_fqn_set - orig_fqn_set
                if len(orig_only) or len(new_only):
                    raise RuntimeError(
                        f"{check_key}"
                        "Composable distributed API implementations cannot modify FQNs.\n"
                        f"FQNs only in original: {orig_only}\n"
                        f"FQNs only in new: {new_only}"
                    )
                else:
                    raise RuntimeError(
                        f"{check_key}"
                        "Composable distributed API implementations cannot modify "
                        "the order of FQNs.\n"
                        f"Original FQNs: {orig_only}\n"
                        f"New FQNs: {new_only}"
                    )

            for orig_named_params, new_named_params in zip(
                all_orig_named_params, all_new_named_params
            ):
                check_fqn(
                    list(orig_named_params.keys()),
                    list(new_named_params.keys()),
                    "Checking parameters: ",
                )
            for orig_named_buffers, new_named_buffers in zip(
                all_orig_named_buffers, all_new_named_buffers
            ):
                check_fqn(
                    list(orig_named_buffers.keys()),
                    list(new_named_buffers.keys()),
                    "Checking buffers: ",
                )
            for orig_named_modules, new_named_modules in zip(
                all_orig_named_modules, all_new_named_modules
            ):
                check_fqn(
                    list(orig_named_modules.keys()),
                    list(new_named_modules.keys()),
                    "Checking modules: ",
                )

            # TODO: verify that installed distributed paradigms are compatible with
            # each other.

            return updated

        def get_state(module: nn.Module) -> _State:
            # [FUNCTION DOC] HELPER: Returns the state object stored for this API (func) on
            # the module, or None; creates an empty state mapping on the module if absent.
            return module.__dict__.setdefault(  # type: ignore[call-overload]
                STATE_KEY,
                {},  # TODO(@yhcharles): this is a temporary fix, need a better way
            ).get(func)  # type: ignore[call-overload]

        wrapper.state = get_state  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    return inner  # type: ignore[return-value]


def _get_registry(module: nn.Module) -> dict[str, RegistryItem] | None:
    # [FUNCTION DOC] HELPER: Returns the mapping of applied composable API names to registry
    # items for the module, or None if none were applied.
    r"""
    Get an ``OrderedDict`` of composable APIs that have been applied to the
    ``module``, indexed by the API name. If no API has been applied, then this
    returns ``None``.
    """
    return getattr(module, REGISTRY_KEY, None)
