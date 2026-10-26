# [MODULE DOC] Shared heuristic registry for Inductor. Holds one module-level mapping from
# (name, device_type, op_name) keys to heuristic classes, used by both template heuristics
# (compile-time config generation) and codegen heuristics (pointwise/reduction configs), plus an
# instance cache. Lookup cascades from most to least specific key. Also offers decorators,
# getters, a test clear helper and a context manager to override template heuristics.
# Import-time side effects: creates the registry and cache dicts and a module logger only.
"""
Heuristic registry system for PyTorch Inductor.

Provides centralized registration for both:
- Template heuristics (compile-time, keyed by template_name/device_type/op_name)
- Codegen heuristics (runtime, keyed by name/device_type)

Both share one underlying registry dict and cascading fallback lookup.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator

    from .template.base import TemplateConfigHeuristics


_HEURISTIC_REGISTRY: dict[tuple[str | None, ...], Any] = {}
# Alias so tests can snapshot/restore the shared registry via the old name.
# This intentionally covers both template and codegen entries.
_TEMPLATE_HEURISTIC_REGISTRY = _HEURISTIC_REGISTRY
_HEURISTIC_CACHE: dict[tuple[str | None, ...], Any] = {}

log = logging.getLogger(__name__)


# -----------------------------------------------------------------
# Shared internals
# -----------------------------------------------------------------


def _register(
    key: tuple[str | None, ...],
    target: Any,
) -> None:
    # [FUNCTION DOC] HELPER: Store the given heuristic class in the shared registry under the given key, replacing any existing entry. Mutates global state; returns nothing.
    _HEURISTIC_REGISTRY[key] = target


def _lookup(name: str, device_type: str, op_name: str | None) -> Any | None:
    # [FUNCTION DOC] HELPER: Cascading lookup of a registered class. Tries (name, device, op), then (name, any device, op),
    # then (name, device, any op), then (name, any, any); returns the first match or None if none exist.
    """Cascading fallback lookup: (name, device, op) -> (name, None, op) -> (name, device, None) -> (name, None, None)."""
    keys = [
        (name, device_type, op_name),
        (name, None, op_name),
        (name, device_type, None),
        (name, None, None),
    ]
    for key in keys:
        if key in _HEURISTIC_REGISTRY:
            return _HEURISTIC_REGISTRY[key]
    return None


# -----------------------------------------------------------------
# Template heuristic API
# -----------------------------------------------------------------


def register_template_heuristic(
    template_name: str,
    device_type: str | None,
    register: bool = True,
    op_name: str | None = None,
) -> Any:
    # [FUNCTION DOC] Build a decorator that registers a TemplateConfigHeuristics subclass under (template_name, device_type, op_name).
    # A device_type or op_name of None acts as a wildcard in lookup. If register is false the decorator is a no-op.
    # When registering, it overwrites any previous entry for the key and logs at info level. The class is returned unchanged.
    """
    Decorator to register template heuristic classes.

    Args:
        template_name: Name of the template (e.g., "mm", "bmm", "scaled_mm")
        device_type: Device type ("cuda", "cpu", "xpu").
            None means applicable to all device types.
        register: Whether to register this heuristic.
        op_name: Operator name for per-op specialization.

    Example:
        @register_template_heuristic("mm", "cuda", register=torch.version.hip is None)
        class CUDAMMTemplateConfigHeuristic(CUDAConfigHeuristic):
            pass
    """

    def decorator(
        cls: type[TemplateConfigHeuristics],
    ) -> type[TemplateConfigHeuristics]:
        # [FUNCTION DOC] HELPER: The actual decorator. Registers the class if enabled and always returns the class unchanged.
        if register:
            key: tuple[str | None, ...] = (template_name, device_type, op_name)
            _register(key, cls)
            log.info(
                "Registered template heuristic: %s for template_name=%s, device_type=%s, op_name=%s",
                cls.__name__,
                template_name,
                device_type,
                op_name,
            )
        return cls

    return decorator


def get_template_heuristic(
    template_name: str, device_type: str, op_name: str
) -> TemplateConfigHeuristics:
    # [FUNCTION DOC] Return a heuristic instance for the template/device/op using cascading lookup, with a per-key instance cache.
    # If nothing is registered, logs an error listing the available keys and returns a fresh base
    # TemplateConfigHeuristics (not cached, so the lookup repeats on each call). Does not raise for missing entries.
    """
    Retrieve a template heuristic instance for the given template and device type.

    Returns a cached instance. Falls back to TemplateConfigHeuristics() if not found.
    """
    cache_key: tuple[str | None, ...] = (template_name, device_type, op_name)
    if cache_key in _HEURISTIC_CACHE:
        return _HEURISTIC_CACHE[cache_key]

    heuristic_class = _lookup(template_name, device_type, op_name)

    if heuristic_class is None:
        from .template.base import TemplateConfigHeuristics as _Base

        log.error(
            "No template heuristic found - template_name=%s, device_type=%s, op_name=%s. "
            "Available: %s. Using fallback.",
            template_name,
            device_type,
            op_name,
            list(_HEURISTIC_REGISTRY.keys()),
        )
        return _Base()

    instance = heuristic_class()
    _HEURISTIC_CACHE[cache_key] = instance
    return instance


def get_registered_heuristic_class(
    template_name: str, device_type: str, op_name: str
) -> None | type[TemplateConfigHeuristics]:
    # [FUNCTION DOC] Return the registered heuristic class (not an instance) for the template/device/op using cascading lookup, or None if absent.
    """Get the registered class (not instance) for the given template/device/op."""
    return _lookup(template_name, device_type, op_name)


# -----------------------------------------------------------------
# Codegen heuristic API
# -----------------------------------------------------------------


class CodegenConfigHeuristics:
    # [CLASS DOC] Base class for codegen (pointwise, reduction, etc.) config heuristics. Subclasses implement get_configs; no state is kept here.
    """Base class for codegen heuristics (pointwise, reduction, etc.)."""

    def get_configs(self, *args: Any, **kwargs: Any) -> list[Any]:
        # [FUNCTION DOC] Return a list of configs for a codegen heuristic. Abstract here: always raises NotImplementedError; subclasses override.
        raise NotImplementedError


def register_codegen_heuristic(
    name: str,
    device_type: str | None = None,
    register: bool = True,
) -> Any:
    # [FUNCTION DOC] Build a decorator that registers a CodegenConfigHeuristics subclass under (name, device_type, None).
    # device_type None marks the default/fallback for all devices. If register is false the decorator is a no-op.
    # Registration overwrites any previous entry for the key and logs at info level.
    """
    Decorator to register a codegen heuristic class.

    Args:
        name: Heuristic name (e.g., "pointwise", "reduction").
        device_type: Device type ("hip", "xpu", "cuda").
            None means default/fallback for all devices.
        register: Whether to actually register.

    Example:
        @register_codegen_heuristic("pointwise", "hip", register=torch.version.hip is not None)
        class ROCmPointwiseHeuristic(CodegenConfigHeuristics):
            ...
    """

    def decorator(cls: type[CodegenConfigHeuristics]) -> type[CodegenConfigHeuristics]:
        # [FUNCTION DOC] HELPER: The actual decorator. Registers the class if enabled and returns the class unchanged.
        if register:
            key: tuple[str | None, ...] = (name, device_type, None)
            _register(key, cls)
            log.info(
                "Registered codegen heuristic: %s for name=%s, device_type=%s",
                cls.__name__,
                name,
                device_type,
            )
        return cls

    return decorator


def get_codegen_heuristic(name: str, device_type: str) -> CodegenConfigHeuristics:
    # [FUNCTION DOC] Return a cached codegen heuristic instance for the name and device type, falling back from (name, device) to (name, any device).
    # If not found, lazily imports the triton_codegen heuristics module (which registers classes) and retries.
    # Raises ValueError listing available keys if still not found. Caches the created instance.
    """
    Retrieve a codegen heuristic instance for the given name and device type.

    Returns a cached instance. Uses cascading fallback:
        (name, device_type, None) -> (name, None, None)
    """
    cache_key: tuple[str | None, ...] = (name, device_type, None)
    if cache_key in _HEURISTIC_CACHE:
        return _HEURISTIC_CACHE[cache_key]

    heuristic_class = _lookup(name, device_type, None)

    if heuristic_class is None:
        # Lazily import codegen heuristics to trigger registration
        import torch._inductor.heuristics.triton_codegen  # noqa: F401

        heuristic_class = _lookup(name, device_type, None)

    if heuristic_class is None:
        raise ValueError(
            f"No codegen heuristic found - name={name}, device_type={device_type}. "
            f"Available: {list(_HEURISTIC_REGISTRY.keys())}"
        )

    instance = heuristic_class()
    _HEURISTIC_CACHE[cache_key] = instance
    return instance


# -----------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------


def clear_registry() -> None:
    # [FUNCTION DOC] Remove every registered heuristic and every cached instance. Intended for tests; mutates global state.
    """Clear all registered heuristics. Primarily for testing."""
    _HEURISTIC_REGISTRY.clear()
    _HEURISTIC_CACHE.clear()


@contextlib.contextmanager
def override_template_heuristics(
    device_type: str,
    template_op_pairs: list[tuple[str, str]],
    override_heuristic_class: type[TemplateConfigHeuristics] | None = None,
) -> Iterator[None]:
    # [FUNCTION DOC] Context manager that temporarily registers override_heuristic_class (default: the base TemplateConfigHeuristics, i.e. no configs)
    # for each (template_name, op_name) pair on the given device type. The shared instance cache is cleared on entry and exit.
    # On exit, the overridden keys are removed and any previous entries restored. Raises AssertionError if an op_name is None.
    # Yields nothing. Mutates global registry state.
    """
    Context manager to temporarily override template heuristics.

    Args:
        device_type: Device type ("cuda", "cpu", "xpu")
        template_op_pairs: List of (template_name, op_name) pairs to override.
        override_heuristic_class: Heuristic class to use for the override.
    """
    if override_heuristic_class is None:
        from .template.base import TemplateConfigHeuristics as _Base

        override_heuristic_class = _Base
    original_entries: dict[tuple[str | None, ...], Any] = {}
    new_keys: list[tuple[str | None, ...]] = []
    # Clears the shared cache (both template and codegen instances).
    # Codegen instances are stateless so re-creation is cheap.
    _HEURISTIC_CACHE.clear()
    try:
        for template_name, op_name in template_op_pairs:
            if op_name is None:
                raise AssertionError("op_name must not be None")
            key: tuple[str | None, ...] = (template_name, device_type, op_name)
            if key in _HEURISTIC_REGISTRY:
                original_entries[key] = _HEURISTIC_REGISTRY[key]
            _HEURISTIC_REGISTRY[key] = override_heuristic_class
            new_keys.append(key)
        yield
    finally:
        for key in new_keys:
            _HEURISTIC_REGISTRY.pop(key, None)
            if key in original_entries:
                _HEURISTIC_REGISTRY[key] = original_entries[key]
        # Same shared-cache clear on exit; see entry comment above.
        _HEURISTIC_CACHE.clear()
