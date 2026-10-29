# [PACKAGE DOC] User-facing control of the Python-native DSL operation overrides (triton, cutedsl and other registered DSLs) defined in torch._native: enable or disable whole DSLs, individual operations, or dispatch keys, with context managers for temporary changes.
# Exports (as attributes of the replacement module object): available_dsls, all_dsls, get_dsl_operations, disable/enable_operations, disable/enable_dispatch_keys, operations_disabled, is_operation_disabled, is_dsl_disabled, and one DSLController attribute per registered DSL name.
# Import-time side effect: replaces this module in the module registry with a PythonNativeModule instance. Registry access is lazy to avoid circular imports.
"""
User-facing API for controlling DSL operation overrides.

The torch.backends.python_native module provides control over DSL (Domain Specific Language)
operation overrides defined in torch._native. This allows users to selectively enable or disable
high-performance implementations from various DSLs like Triton and CuteDSL.

The module supports both coarse-grained control (entire DSLs) and fine-grained control
(individual operations or dispatch keys). All control operations support context managers
for temporary state changes.

Example usage::

    import torch.backends.python_native as pn

    # DSL-level control
    pn.triton.enabled = False  # Disable all triton ops
    pn.cutedsl.enabled = True  # Enable all cutedsl ops

    # Individual operation control
    pn.disable_operations("scaled_mm")  # Disable specific op across all DSLs
    pn.enable_operations("scaled_mm")  # Re-enable specific op

    # Context manager support
    with pn.triton.disabled():
        result = some_computation()  # Triton ops disabled here

    # Query capabilities
    print(pn.available_dsls)  # ['triton', 'cutedsl']
    print(pn.get_dsl_operations("triton"))  # Operations for triton
"""

import functools
import sys
import types
from contextlib import contextmanager

from torch.backends import ContextProp, flags_frozen, PropModule


@contextmanager
def _preserve_filter_state():
    # [FUNCTION DOC] HELPER: Context manager that snapshots the registry filter state (disabled DSL names, operation symbols, dispatch keys), yields the live filter state, and on exit restores all three sets to the snapshot, even on error. Mutates the shared filter state.
    """Context manager to save and restore registry filter state."""
    filter_state = _get_filter_state()

    # Save original state
    original_state = (
        set(filter_state._dsl_names),
        set(filter_state._op_symbols),
        set(filter_state._dispatch_keys),
    )

    try:
        yield filter_state
    finally:
        # Restore original state
        filter_state._dsl_names.clear()
        filter_state._op_symbols.clear()
        filter_state._dispatch_keys.clear()

        filter_state._dsl_names.update(original_state[0])
        filter_state._op_symbols.update(original_state[1])
        filter_state._dispatch_keys.update(original_state[2])


def _get_dsl_registry():
    # [FUNCTION DOC] HELPER: Imports lazily and returns the DSL registry object (lazy to avoid circular imports).
    """Lazy import to avoid circular imports."""
    from torch._native.dsl_registry import dsl_registry

    return dsl_registry


def _get_registry_functions():
    # [FUNCTION DOC] HELPER: Lazily imports and returns a 4-tuple from the native registry: the function that disables op overrides, the function that re-enables them, the graph registry, and the filter state.
    """Lazy import of registry functions."""
    from torch._native.registry import (
        _filter_state,
        _graphs,
        deregister_op_overrides,
        reenable_op_overrides,
    )

    return deregister_op_overrides, reenable_op_overrides, _graphs, _filter_state


def _get_filter_state():
    # [FUNCTION DOC] HELPER: Returns the registry's global filter state object (fourth entry of the registry tuple).
    """Direct access to filter state."""
    return _get_registry_functions()[3]


def _get_dsl_module(dsl_name: str):
    # [FUNCTION DOC] HELPER: Looks up a registered DSL module by name through the DSL registry; raises ValueError listing all known DSLs if the name is not registered.
    """Get the registered DSL module for direct control.

    Uses the DSL registry to dynamically look up DSL modules instead of
    hard-coding the mapping. This makes the function automatically extensible
    for new DSLs without code changes.

    Args:
        dsl_name (str): Name of the DSL to retrieve.

    Returns:
        DSLModuleProtocol: The registered DSL module.

    Raises:
        ValueError: If the DSL is not registered.
    """
    registry = _get_dsl_registry()

    # Use the public API to get the DSL module
    dsl_module = registry.get_dsl_module(dsl_name)
    if dsl_module is not None:
        return dsl_module
    else:
        raise ValueError(
            f"Unknown DSL: {dsl_name}. Available DSLs: {registry.list_all_dsls()}"
        )


class DSLController:
    # [CLASS DOC] Handle for one named DSL, offering availability, version, backend and enabled state queries plus enable, disable and a temporary-disable context manager.
    # State: only the DSL name; all real state lives in the shared registry.
    """Controller for a specific DSL."""

    def __init__(self, dsl_name: str):
        # [FUNCTION DOC] Stores the DSL name.
        self._dsl_name = dsl_name

    @property
    def name(self) -> str:
        # [FUNCTION DOC] Property returning the DSL name.
        return self._dsl_name

    @property
    def available(self) -> bool:
        # [FUNCTION DOC] Property: asks the registry whether this DSL's runtime is available.
        """Check if DSL runtime is available."""
        registry = _get_dsl_registry()
        return registry.is_dsl_available(self._dsl_name)

    @property
    def version(self):
        # [FUNCTION DOC] Property: returns the DSL version reported by the registry.
        """Get DSL version."""
        registry = _get_dsl_registry()
        return registry.get_dsl_version(self._dsl_name)

    @property
    def backend(self) -> str:
        # [FUNCTION DOC] Property: returns the chosen lowering backend by calling the DSL module's chooser function; raises AttributeError if the DSL module has no such function.
        """Chosen lowering backend for DSLs that have one (e.g. helion)."""
        chosen = getattr(_get_dsl_module(self._dsl_name), "_chosen_backend", None)
        if chosen is None:
            raise AttributeError(f"{self._dsl_name} DSL has no configurable backend")
        return chosen()

    @property
    def enabled(self) -> bool:
        # [FUNCTION DOC] Property: True unless this DSL's name is in the filter state's set of disabled DSLs.
        """Check if DSL is currently enabled."""
        filter_state = _get_filter_state()
        return self._dsl_name not in filter_state._dsl_names

    @enabled.setter
    def enabled(self, value: bool):
        # [FUNCTION DOC] Property setter: enables (true value) or disables (false value) the DSL. Raises RuntimeError if global flags have been frozen.
        """Enable or disable the DSL."""
        if flags_frozen():
            raise RuntimeError(
                f"not allowed to set {self._dsl_name} DSL flags "
                "after disable_global_flags; please use flags() context manager instead"
            )
        if value:
            self.enable()
        else:
            self.disable()

    def disable(self):
        # [FUNCTION DOC] Disables all operation overrides of this DSL by asking the DSL module to deregister them (mutates registry state).
        """Disable all operations for this DSL."""
        dsl_module = _get_dsl_module(self._dsl_name)
        dsl_module.deregister_op_overrides()

    def enable(self):
        # [FUNCTION DOC] Re-enables all operation overrides of this DSL through the registry (mutates registry state).
        """Re-enable all operations for this DSL."""
        reenable_op_overrides = _get_registry_functions()[1]
        reenable_op_overrides(enable_dsl_names=self._dsl_name)

    @contextmanager
    def disabled(self):
        # [FUNCTION DOC] Context manager: records whether the DSL was enabled, disables it for the body, and on exit re-enables it only if it had been enabled before.
        # KNOWN ISSUE: unlike the public enabled setter, this does not check whether flags are frozen.
        """Context manager to temporarily disable DSL."""
        original_state = self.enabled
        try:
            self.disable()
            yield
        finally:
            if original_state:
                self.enable()

    def __repr__(self):
        # [FUNCTION DOC] Returns a text form with the DSL name, availability and enabled status.
        status = "available" if self.available else "unavailable"
        enabled_status = "enabled" if self.enabled else "disabled"
        return f"DSLController({self._dsl_name}, {status}, {enabled_status})"


class PythonNativeModule(PropModule):
    # [CLASS DOC] Module replacement object for this package, built on the shared property-module base. Provides DSL listing, per-operation and per-dispatch-key enable/disable, and dynamic attribute access that returns a cached DSLController for any registered DSL name (and exposes a few private helpers for tests).
    """Main module for python_native DSL control."""

    def __init__(self, original_module):
        # [FUNCTION DOC] Wraps the original module, reusing its name.
        super().__init__(original_module, original_module.__name__)

    @property
    def available_dsls(self) -> list[str]:
        # [FUNCTION DOC] Property: names of DSLs whose runtime is available, as a list.
        """Get list of available DSLs."""
        registry = _get_dsl_registry()
        result = registry.list_available_dsls()
        return list(result) if not isinstance(result, list) else result

    @property
    def all_dsls(self) -> list[str]:
        # [FUNCTION DOC] Property: names of all registered DSLs, as a list.
        """Get list of all registered DSLs."""
        registry = _get_dsl_registry()
        result = registry.list_all_dsls()
        return list(result) if not isinstance(result, list) else result

    def get_dsl_operations(self, dsl_name: str) -> list[str]:
        # [FUNCTION DOC] Returns the sorted list of operation names registered by the named DSL, taken from the native registry.
        """Get list of operations registered by a specific DSL.

        Args:
            dsl_name (str): Name of the DSL to query (e.g., 'triton', 'cutedsl').

        Returns:
            list[str]: Sorted list of operation names registered by the DSL.

        Example::

            ops = torch.backends.python_native.get_dsl_operations("triton")
            print(ops)  # ['triton_to_mxfp8_dim0', ...]
        """
        from torch._native.registry import get_dsl_operations

        return get_dsl_operations(dsl_name)

    def disable_operations(self, *op_symbols: str):
        # [FUNCTION DOC] Disables the named operations across all DSLs (mutates registry filter state).
        """Disable specific operations across all DSLs.

        Args:
            *op_symbols (str): Names of operations to disable.

        Example::

            # Disable scaled matrix multiply across all DSLs
            torch.backends.python_native.disable_operations("scaled_mm")

            # Disable multiple operations
            torch.backends.python_native.disable_operations(
                "scaled_mm", "flash_attention"
            )
        """
        deregister_op_overrides = _get_registry_functions()[0]
        deregister_op_overrides(disable_op_symbols=list(op_symbols))

    def enable_operations(self, *op_symbols: str):
        # [FUNCTION DOC] Re-enables the named operations across all DSLs (mutates registry filter state).
        """Re-enable specific operations across all DSLs.

        Args:
            *op_symbols (str): Names of operations to re-enable.

        Example::

            # Re-enable previously disabled operations
            torch.backends.python_native.enable_operations(
                "scaled_mm", "flash_attention"
            )
        """
        reenable_op_overrides = _get_registry_functions()[1]
        reenable_op_overrides(enable_op_symbols=list(op_symbols))

    def disable_dispatch_keys(self, *dispatch_keys: str):
        # [FUNCTION DOC] Disables operation overrides at the given dispatch keys such as CUDA or CPU (mutates registry filter state).
        """Disable operations at specific dispatch keys.

        Args:
            *dispatch_keys (str): Dispatch keys to disable (e.g., 'CUDA', 'CPU').

        Example::

            # Disable all native operations on CUDA
            torch.backends.python_native.disable_dispatch_keys("CUDA")
        """
        deregister_op_overrides = _get_registry_functions()[0]
        deregister_op_overrides(disable_dispatch_keys=list(dispatch_keys))

    def enable_dispatch_keys(self, *dispatch_keys: str):
        # [FUNCTION DOC] Re-enables operation overrides at the given dispatch keys (mutates registry filter state).
        """Re-enable operations at specific dispatch keys.

        Args:
            *dispatch_keys (str): Dispatch keys to re-enable (e.g., 'CUDA', 'CPU').

        Example::

            # Re-enable native operations on CUDA
            torch.backends.python_native.enable_dispatch_keys("CUDA")
        """
        reenable_op_overrides = _get_registry_functions()[1]
        reenable_op_overrides(enable_dispatch_keys=list(dispatch_keys))

    @contextmanager
    def operations_disabled(self, *op_symbols: str):
        # [FUNCTION DOC] Context manager that disables the named operations for the body and afterwards re-enables only those that were not already disabled before entry, even on error.
        """Context manager to temporarily disable operations.

        Args:
            *op_symbols (str): Names of operations to temporarily disable.

        Example::

            with torch.backends.python_native.operations_disabled("scaled_mm"):
                # scaled_mm is disabled across all DSLs
                result = model(input)
            # scaled_mm is automatically re-enabled here
        """
        filter_state = _get_filter_state()
        previously_disabled_ops = {
            op for op in op_symbols if op in filter_state._op_symbols
        }

        self.disable_operations(*op_symbols)
        try:
            yield
        finally:
            # Only re-enable operations that weren't already disabled
            ops_to_reenable = [
                op for op in op_symbols if op not in previously_disabled_ops
            ]
            if ops_to_reenable:
                self.enable_operations(*ops_to_reenable)

    @functools.lru_cache(maxsize=16)  # noqa: B019
    def _get_dsl_controller(self, name: str) -> "DSLController":
        # [FUNCTION DOC] HELPER: Returns a DSLController for the name; results are cached per (module, name) with a bounded cache of 16 entries.
        """Get or create a DSL controller (cached)."""
        return DSLController(name)

    def _get_registry_functions(self):
        # [FUNCTION DOC] HELPER: Exposes the registry function tuple for tests.
        """Expose registry functions for testing."""
        return _get_registry_functions()

    def is_operation_disabled(self, op_symbol: str) -> bool:
        # [FUNCTION DOC] Returns True if the operation symbol is currently in the disabled set.
        """Check if an operation is currently disabled."""
        filter_state = _get_filter_state()
        return op_symbol in filter_state._op_symbols

    def is_dsl_disabled(self, dsl_name: str) -> bool:
        # [FUNCTION DOC] Returns True if the DSL name is currently in the disabled set.
        """Check if a DSL is currently disabled."""
        filter_state = _get_filter_state()
        return dsl_name in filter_state._dsl_names

    def __getattr__(self, name: str):
        # [FUNCTION DOC] Fallback attribute lookup: raises AttributeError for double-underscore names (avoids circular imports during torch startup), returns a cached DSLController for registered DSL names, returns certain private test helpers by name, and otherwise raises AttributeError.
        """Dynamic attribute access for DSL controllers."""
        # Skip dunder attributes to avoid triggering DSL registry lookups
        # during torch initialization. inspect.getmodule() calls
        # hasattr(module, '__file__') which would otherwise cause a circular
        # import through _get_dsl_registry() while torch is still loading.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"module '{self.__name__}' has no attribute '{name}'")

        if name in self.all_dsls:
            return self._get_dsl_controller(name)

        # Expose private functions for testing
        if name == "_get_dsl_module":
            return _get_dsl_module
        if name == "_get_registry_functions":
            return self._get_registry_functions
        if name == "_get_filter_state":
            return _get_filter_state

        raise AttributeError(f"module '{self.__name__}' has no attribute '{name}'")

    def __dir__(self):
        # [FUNCTION DOC] Returns a sorted list of attribute names: the base ones, the public methods listed here, and all registered DSL names (DSL names are skipped silently if the registry cannot be queried).
        """Return available attributes including DSL names."""
        attrs = set(super().__dir__())
        attrs.update(
            {
                "available_dsls",
                "all_dsls",
                "get_dsl_operations",
                "disable_operations",
                "enable_operations",
                "disable_dispatch_keys",
                "enable_dispatch_keys",
                "operations_disabled",
                "is_operation_disabled",
                "is_dsl_disabled",
            }
        )

        # Add DSL names
        try:
            attrs.update(self.all_dsls)
        except Exception:
            # If registry not available yet, skip DSL names
            pass

        return sorted(attrs)


# Replace the current module with our enhanced version
sys.modules[__name__] = PythonNativeModule(sys.modules[__name__])
