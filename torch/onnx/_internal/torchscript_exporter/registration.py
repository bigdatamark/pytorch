# mypy: allow-untyped-defs
# [MODULE DOC] Registry of ONNX symbolic functions keyed by qualified name and opset version.
# Contains the opset dispatch helper, OverrideDict (base plus override layers), the per-op
# _SymbolicFunctionGroup, SymbolicRegistry, and the onnx_symbolic and custom_onnx_symbolic
# decorators. Import time creates the global registry instance used by the decorators.
"""Module for handling symbolic function registration."""

import warnings
from collections.abc import Callable, Collection, Sequence
from typing import Generic, TypeVar
from typing_extensions import ParamSpec

from torch.onnx import _constants, errors


OpsetVersion = int


def _dispatch_opset_version(
    target: OpsetVersion, registered_opsets: Collection[OpsetVersion]
) -> OpsetVersion | None:
    # [FUNCTION DOC] HELPER: Picks which registered opset serves a target opset. Returns None if
    # nothing is registered. For targets at or above the base opset, returns the highest
    # registered version not above the target (or None). For lower targets (legacy), returns the
    # smallest registered version between the target and the base opset (or None).
    """Finds the registered opset given a target opset version and the available opsets.

    Args:
        target: The target opset version.
        registered_opsets: The available opsets.

    Returns:
        The registered opset version.
    """
    if not registered_opsets:
        return None

    descending_registered_versions = sorted(registered_opsets, reverse=True)
    # Linear search for the opset version, which is fine since the number of opset
    # versions is small.

    if target >= _constants.ONNX_BASE_OPSET:
        # Always look down toward opset 1 when the target is >= ONNX_BASE_OPSET (opset 9).
        # When a custom op is register at opset 1, we want to be able to discover it as a
        # fallback for all opsets >= ONNX_BASE_OPSET.
        for version in descending_registered_versions:
            if version <= target:
                return version
        return None

    # target < opset 9. This is the legacy behavior to support opset 7 and opset 8.
    # for caffe2 support. We search up toward opset 9.
    for version in reversed(descending_registered_versions):
        # Count back up until _constants.ONNX_BASE_OPSET
        if target <= version <= _constants.ONNX_BASE_OPSET:
            return version

    return None


_K = TypeVar("_K")
_V = TypeVar("_V")
_R = TypeVar("_R")
_P = ParamSpec("_P")


class OverrideDict(Collection[_K], Generic[_K, _V]):
    # [CLASS DOC] Mapping-like collection with three internal dictionaries: base entries,
    # overrides, and the merged view (overrides win over base). All read operations (lookup,
    # get, contains, iterate, length, truthiness) use the merged view. Invariant: merged holds
    # every override plus every base key without an override.
    """A dictionary that merges built-in and custom symbolic functions.

    It supports overriding and un-overriding built-in symbolic functions with custom
    ones.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates empty base, override and merged dictionaries.
        self._base: dict[_K, _V] = {}
        self._overrides: dict[_K, _V] = {}
        self._merged: dict[_K, _V] = {}

    def set_base(self, key: _K, value: _V) -> None:
        # [FUNCTION DOC] Stores a base entry (replacing any previous base value). Updates the
        # merged view only if the key is not currently overridden.
        self._base[key] = value
        if key not in self._overrides:
            self._merged[key] = value

    def in_base(self, key: _K) -> bool:
        # [FUNCTION DOC] True if the key has a base entry (regardless of overrides).
        """Checks if a key is in the base dictionary."""
        return key in self._base

    def override(self, key: _K, value: _V) -> None:
        # [FUNCTION DOC] Records an override and makes it visible in the merged view, even if the
        # key has no base entry.
        """Overrides a base key-value with a new pair."""
        self._overrides[key] = value
        self._merged[key] = value

    def remove_override(self, key: _K) -> None:
        # [FUNCTION DOC] Drops any override for the key (silently if absent) and restores the
        # base value in the merged view; if there is no base entry the key disappears entirely.
        """Un-overrides a key-value pair."""
        self._overrides.pop(key, None)  # type: ignore[arg-type]
        self._merged.pop(key, None)  # type: ignore[arg-type]
        if key in self._base:
            self._merged[key] = self._base[key]

    def overridden(self, key: _K) -> bool:
        # [FUNCTION DOC] True if the key currently has an override.
        """Checks if a key-value pair is overridden."""
        return key in self._overrides

    def __getitem__(self, key: _K) -> _V:
        # [FUNCTION DOC] HELPER: Returns the merged value for the key; raises key error if absent.
        return self._merged[key]

    def get(self, key: _K, default: _V | None = None):
        # [FUNCTION DOC] HELPER: Returns the merged value for the key, or default if absent.
        return self._merged.get(key, default)

    def __contains__(self, key: object) -> bool:
        # [FUNCTION DOC] HELPER: Membership test against the merged view.
        return key in self._merged

    def __iter__(self):
        # [FUNCTION DOC] HELPER: Iterates over the keys of the merged view.
        return iter(self._merged)

    def __len__(self) -> int:
        # [FUNCTION DOC] HELPER: Number of keys in the merged view.
        return len(self._merged)

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Text form showing the base and override dictionaries.
        return f"OverrideDict(base={self._base}, overrides={self._overrides})"

    def __bool__(self) -> bool:
        # [FUNCTION DOC] HELPER: True if the merged view is non-empty.
        return bool(self._merged)


class _SymbolicFunctionGroup:
    # [CLASS DOC] HELPER: All registered implementations of one op name, keyed by opset version
    # in an OverrideDict (built-in as base, custom as overrides). Lookup finds the best version
    # for a target opset through the dispatch helper. Holds the op name for messages only.
    """Different versions of symbolic functions registered to the same name.

    O(number of registered versions of an op) search is performed to find the most
    recent version of the op.

    The registration is delayed until op is used to improve startup time.

    Function overloads with different arguments are not allowed.
    Custom op overrides are supported.
    """

    def __init__(self, name: str) -> None:
        # [FUNCTION DOC] HELPER: Stores the op name; creates an empty version-keyed function map.
        self._name = name
        # A dictionary of functions, keyed by the opset version.
        self._functions: OverrideDict[OpsetVersion, Callable] = OverrideDict()

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Text form with the op name and registered functions.
        return f"_SymbolicFunctionGroup({self._name}, registered={self._functions})"

    def __getitem__(self, key: OpsetVersion) -> Callable:
        # [FUNCTION DOC] HELPER: Like get, but raises a key error when no version applies.
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    # TODO(justinchuby): Add @functools.lru_cache(maxsize=None) if lookup time becomes
    # a problem.
    def get(self, opset: OpsetVersion) -> Callable | None:
        # [FUNCTION DOC] Returns the function registered for the dispatched opset version for the
        # given target opset, or None if no registered version applies. Read-only.
        """Find the most recent version of the function."""
        version = _dispatch_opset_version(opset, self._functions)
        if version is None:
            return None

        return self._functions[version]

    def add(self, func: Callable, opset: OpsetVersion) -> None:
        # [FUNCTION DOC] Registers a built-in function at an opset version. If a built-in already
        # exists there, emits an exporter warning (asking for a bug report) and replaces it.
        """Adds a symbolic function.

        Args:
            func: The function to add.
            opset: The opset version of the function to add.
        """
        if self._functions.in_base(opset):
            warnings.warn(
                f"Symbolic function '{self._name}' already registered for opset {opset}. "
                f"Replacing the existing function with new function. This is unexpected. "
                f"Please report it on {_constants.PYTORCH_GITHUB_ISSUES_URL}.",
                errors.OnnxExporterWarning,
                stacklevel=2,
            )
        self._functions.set_base(opset, func)

    def add_custom(self, func: Callable, opset: OpsetVersion) -> None:
        # [FUNCTION DOC] Registers a custom function that overrides any built-in at that opset.
        """Adds a custom symbolic function.

        Args:
            func: The symbolic function to register.
            opset: The corresponding opset version.
        """
        self._functions.override(opset, func)

    def remove_custom(self, opset: OpsetVersion) -> None:
        # [FUNCTION DOC] Removes the custom override at an opset version, restoring any built-in.
        # If there is no custom function there, emits a plain warning and does nothing else.
        """Removes a custom symbolic function.

        Args:
            opset: The opset version of the custom function to remove.
        """
        if not self._functions.overridden(opset):
            warnings.warn(
                f"No custom function registered for '{self._name}' opset {opset}",
                stacklevel=2,
            )
            return
        self._functions.remove_override(opset)

    def get_min_supported(self) -> OpsetVersion:
        # [FUNCTION DOC] Returns the smallest registered opset version. Raises a value error if
        # nothing is registered.
        # KNOWN ISSUE: the docstring says built-in only, but custom overrides are included too.
        """Returns the lowest built-in opset version supported by the function."""
        return min(self._functions)


class SymbolicRegistry:
    # [CLASS DOC] Mapping from qualified op names (domain::op) to function groups. Supports
    # registering built-in or custom functions, unregistering custom ones, group lookup, and
    # queries of registration status and names. A module-level instance named registry is shared.
    """Registry for symbolic functions.

    The registry maintains a mapping from qualified names to symbolic functions.
    It is used to register new symbolic functions and to dispatch calls to
    the appropriate function.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates an empty name-to-group mapping.
        self._registry: dict[str, _SymbolicFunctionGroup] = {}

    def register(
        self, name: str, opset: OpsetVersion, func: Callable, custom: bool = False
    ) -> None:
        # [FUNCTION DOC] Adds a function for a qualified name and opset, creating the group on
        # first use; custom functions go through the override path, others through the built-in
        # path (which may warn on duplicates). Raises a value error if the name lacks the double
        # colon separator. Mutates the registry.
        """Registers a symbolic function.

        Args:
            name: The qualified name of the function to register. In the form of 'domain::op'.
                E.g. 'aten::add'.
            opset: The opset version of the function to register.
            func: The symbolic function to register.
            custom: Whether the function is a custom function that overrides existing ones.

        Raises:
            ValueError: If the separator '::' is not in the name.
        """
        if "::" not in name:
            raise ValueError(
                f"The name must be in the form of 'domain::op', not '{name}'"
            )
        symbolic_functions = self._registry.setdefault(
            name, _SymbolicFunctionGroup(name)
        )
        if custom:
            symbolic_functions.add_custom(func, opset)
        else:
            symbolic_functions.add(func, opset)

    def unregister(self, name: str, opset: OpsetVersion) -> None:
        # [FUNCTION DOC] Removes the custom function for the name and opset. Does nothing if the
        # name is unknown; otherwise the group warns if no custom function was registered.
        # Built-in functions are never removed.
        """Unregisters a symbolic function.

        Args:
            name: The qualified name of the function to unregister.
            opset: The opset version of the function to unregister.
        """
        if name not in self._registry:
            return
        self._registry[name].remove_custom(opset)

    def get_function_group(self, name: str) -> _SymbolicFunctionGroup | None:
        # [FUNCTION DOC] Returns the function group for a qualified name, or None if unknown.
        """Returns the function group for the given name."""
        return self._registry.get(name)

    def is_registered_op(self, name: str, version: int) -> bool:
        # [FUNCTION DOC] True if a function exists for the name that serves the given opset
        # version (using opset dispatch, so lower registered versions count).
        """Returns whether the given op is registered for the given opset version."""
        functions = self.get_function_group(name)
        if functions is None:
            return False
        return functions.get(version) is not None

    def all_functions(self) -> set[str]:
        # [FUNCTION DOC] Returns a new set of all registered qualified op names.
        """Returns the set of all registered function names."""
        return set(self._registry)


def onnx_symbolic(
    name: str,
    opset: OpsetVersion | Sequence[OpsetVersion],
    decorate: Sequence[Callable] | None = None,
    custom: bool = False,
) -> Callable:
    # [FUNCTION DOC] Decorator factory for registering symbolic functions. The returned decorator
    # applies each decorate function in order to the target, registers the result in the global
    # registry for every given opset version (a single int is accepted), and returns the original
    # undecorated function. Raises a value error for a name lacking the double colon separator
    # (at decoration time). Side effect: mutates the global registry.
    """Registers a symbolic function.

    Usage::

    ```
    @onnx_symbolic(
        "aten::symbolic_b",
        opset=10,
        decorate=[quantized_aten_handler(scale=1 / 128, zero_point=0)],
    )
    @symbolic_helper.parse_args("v", "v", "b")
    def symbolic_b(g: _C.Graph, x: _C.Value, y: _C.Value, arg1: bool) -> _C.Value: ...
    ```

    Args:
        name: The qualified name of the function in the form of 'domain::op'.
            E.g. 'aten::add'.
        opset: The opset versions of the function to register at.
        decorate: A sequence of decorators to apply to the function.
        custom: Whether the function is a custom symbolic function.

    Raises:
        ValueError: If the separator '::' is not in the name.
    """

    def wrapper(func: Callable[_P, _R]) -> Callable[_P, _R]:
        # [FUNCTION DOC] HELPER: The decorator: applies decorators, registers for each opset and
        # returns func unchanged. Normalizes a single int opset to a tuple in the enclosing scope.
        decorated = func
        if decorate is not None:
            for decorate_func in decorate:
                decorated = decorate_func(decorated)

        global registry
        nonlocal opset
        if isinstance(opset, OpsetVersion):
            opset = (opset,)
        for opset_version in opset:
            registry.register(name, opset_version, decorated, custom=custom)

        # Return the original function because the decorators in "decorate" are only
        # specific to the instance being registered.
        return func

    return wrapper


def custom_onnx_symbolic(
    name: str,
    opset: OpsetVersion | Sequence[OpsetVersion],
    decorate: Sequence[Callable] | None = None,
) -> Callable:
    # [FUNCTION DOC] Same as the symbolic decorator factory but always registers as custom (an
    # override of built-ins). Returns the decorator; raises a value error on a bad name.
    """Registers a custom symbolic function.

    Args:
        name: the qualified name of the function.
        opset: the opset version of the function.
        decorate: a sequence of decorators to apply to the function.

    Returns:
        The decorator.

    Raises:
        ValueError: If the separator '::' is not in the name.
    """
    return onnx_symbolic(name, opset, decorate, custom=True)


# The registry for all symbolic functions.
registry = SymbolicRegistry()
