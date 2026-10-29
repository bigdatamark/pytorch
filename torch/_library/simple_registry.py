# [MODULE DOC]
# Registry for the "simple" torch.library APIs (fake impl, effects, torch_dispatch rules,
# symmetric-memory args) that do not go through the dispatcher table. Defines the global
# registry singleton keyed by qualified operator name, per-operator entries, and holders.
# No import-time side effects other than creating the singleton.
from collections.abc import Callable
from typing import Any

from torch.utils._ordered_set import OrderedSet

from .effects import EffectHolder
from .fake_impl import FakeImplHolder
from .utils import RegistrationHandle


__all__ = [
    "SimpleLibraryRegistry",
    "SimpleOperatorEntry",
    "singleton",
    "SymmMemArgsHolder",
]


class SimpleLibraryRegistry:
    # [CLASS DOC]
    # Mapping from fully qualified operator name (with overload) to its SimpleOperatorEntry.
    # Entries are created lazily by find; get is a non-creating lookup.
    """Registry for the "simple" torch.library APIs

    The "simple" torch.library APIs are a higher-level API on top of the
    raw PyTorch DispatchKey registration APIs that includes:
    - fake impl

    Registrations for these APIs do not go into the PyTorch dispatcher's
    table because they may not directly involve a DispatchKey. For example,
    the fake impl is a Python function that gets invoked by FakeTensor.
    Instead, we manage them here.

    SimpleLibraryRegistry is a mapping from a fully qualified operator name
    (including the overload) to SimpleOperatorEntry.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the empty internal mapping from qualified name to entry.
        self._data: dict[str, SimpleOperatorEntry] = {}

    def find(self, qualname: str) -> "SimpleOperatorEntry":
        # [FUNCTION DOC] Returns the entry for qualname, creating and storing a new
        # SimpleOperatorEntry if none exists (mutates the registry).
        res = self._data.get(qualname, None)
        if res is None:
            self._data[qualname] = res = SimpleOperatorEntry(qualname)
        return res

    def get(self, qualname: str) -> "SimpleOperatorEntry | None":
        # [FUNCTION DOC] Returns the entry for qualname, or None if it was never created. Does not create.
        return self._data.get(qualname, None)


singleton: SimpleLibraryRegistry = SimpleLibraryRegistry()


class SimpleOperatorEntry:
    # [CLASS DOC]
    # One entry per operator overload. Holds the holders where kernels/rules are registered:
    # fake_impl, torch_dispatch_rules, effect and symm_mem_args, all keyed by the same qualname.
    """This is 1:1 to an operator overload.

    The fields of SimpleOperatorEntry are Holders where kernels can be
    registered to.
    """

    def __init__(self, qualname: str) -> None:
        # [FUNCTION DOC] Stores qualname and constructs the four holders for that operator.
        self.qualname: str = qualname
        self.fake_impl: FakeImplHolder = FakeImplHolder(qualname)
        self.torch_dispatch_rules: GenericTorchDispatchRuleHolder = (
            GenericTorchDispatchRuleHolder(qualname)
        )

        self.effect: EffectHolder = EffectHolder(qualname)
        self.symm_mem_args: SymmMemArgsHolder = SymmMemArgsHolder(qualname)

    # For compatibility reasons. We can delete this soon.
    @property
    def abstract_impl(self) -> FakeImplHolder:
        # [FUNCTION DOC] Property kept for compatibility; returns the fake_impl holder.
        return self.fake_impl


class GenericTorchDispatchRuleHolder:
    # [CLASS DOC]
    # Holds per-operator __torch_dispatch__ rules, a mapping from a tensor subclass type to a
    # callable. At most one rule per class is allowed.
    def __init__(self, qualname: str) -> None:
        # [FUNCTION DOC] Initializes an empty class-to-rule mapping and stores qualname.
        self._data: dict[type, Callable[..., Any]] = {}
        self.qualname: str = qualname

    def register(
        self, torch_dispatch_class: type, func: Callable[..., Any]
    ) -> RegistrationHandle:
        # [FUNCTION DOC] Registers func as the rule for torch_dispatch_class.
        # Raises RuntimeError if the class already has a rule. Returns a RegistrationHandle whose
        # deregister removes the rule.
        if self.find(torch_dispatch_class):
            raise RuntimeError(
                f"{torch_dispatch_class} already has a `__torch_dispatch__` rule registered for {self.qualname}"
            )
        self._data[torch_dispatch_class] = func

        def deregister() -> None:
            del self._data[torch_dispatch_class]

        return RegistrationHandle(deregister)

    def find(self, torch_dispatch_class: type) -> Callable[..., Any] | None:
        # [FUNCTION DOC] Returns the rule registered for the exact class, or None (no subclass lookup).
        return self._data.get(torch_dispatch_class, None)


class SymmMemArgsHolder:
    # [CLASS DOC]
    # Tracks which argument names of an operator need symmetric-memory allocation (used by
    # Inductor lowering). State is an ordered set of names, or None when unregistered.
    """Tracks which arguments of an operator require symmetric memory allocation.

    Used by Inductor during lowering to automatically realize tensors as
    symmetric memory buffers.
    """

    def __init__(self, qualname: str) -> None:
        # [FUNCTION DOC] Initializes with no registered args and stores qualname.
        self._symm_mem_args: OrderedSet[str] | None = None
        self.qualname: str = qualname

    def register(self, arg_names: list[str]) -> RegistrationHandle:
        # [FUNCTION DOC] Registers arg_names as the symmetric-memory args. Raises ValueError for an
        # empty list. If already registered, logs a warning and overwrites. Returns a handle whose
        # deregister resets the state to unregistered.
        if not arg_names:
            raise ValueError(
                f"Cannot register empty arg_names list for {self.qualname}"
            )

        if self._symm_mem_args is not None:
            import logging

            log = logging.getLogger(__name__)
            log.warning(
                "Overwriting symm_mem arg registration for %s. Old: %s, New: %s",
                self.qualname,
                self._symm_mem_args,
                arg_names,
            )

        self._symm_mem_args = OrderedSet(arg_names)

        def deregister() -> None:
            self._symm_mem_args = None

        return RegistrationHandle(deregister)

    def get(self) -> OrderedSet[str] | None:
        # [FUNCTION DOC] Returns the ordered set of registered arg names, or None.
        return self._symm_mem_args

    def is_registered(self) -> bool:
        # [FUNCTION DOC] Returns True if any arg names are registered.
        return self._symm_mem_args is not None

    def is_symm_mem_arg(self, arg_name: str) -> bool:
        # [FUNCTION DOC] Returns True if arg_name is among the registered symmetric-memory args.
        return self._symm_mem_args is not None and arg_name in self._symm_mem_args


def find_torch_dispatch_rule(
    op: Any, torch_dispatch_class: type
) -> Callable[..., Any] | None:
    # [FUNCTION DOC] Looks up (creating the entry if needed) the singleton entry for op's
    # qualified name and returns the __torch_dispatch__ rule for the given class, or None.
    return singleton.find(op.__qualname__).torch_dispatch_rules.find(
        torch_dispatch_class
    )
