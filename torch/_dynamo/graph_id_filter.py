# [MODULE DOC]
# Per-graph override machinery for torch.compile. Parses configuration strings of the form
# "filter:value;filter:value" where a filter selects graph (frame) IDs by single ID, inclusive
# range, or comparison operator. Provides GraphIdFilter (ID matching), a generic router base,
# GraphBackendRouter (ID -> backend), GraphConfigRouter (ID -> merged config mapping), cached
# router factories, config-key validators, and the get_*_override_for_compile_id entry points.
# No import-time side effects besides creating the module logger.
# mypy: disallow-untyped-defs

from __future__ import annotations

import functools
import logging
import re
import warnings
from typing import Any, Generic, TYPE_CHECKING, TypeVar


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._guards import CompileId


log = logging.getLogger(__name__)


class GraphIdFilter:
    # [CLASS DOC] Predicate over integer graph IDs built from a comma-separated spec.
    # Holds an immutable set of explicit IDs (single IDs and expanded inclusive ranges) and a list
    # of (operator, threshold) comparison conditions. Membership is true if the ID is explicit or
    # satisfies any condition. An empty or blank spec yields a filter that matches nothing.
    """
    A filter for matching graph IDs based on various conditions.
    Supports individual IDs, ranges, and comparison operators.
    """

    def __init__(self, filter_str: str) -> None:
        # [FUNCTION DOC] Initialize empty ID set and condition list, then parse the given spec string.
        self._explicit_ids: frozenset[int] = frozenset()
        self._conditions: list[tuple[str, int]] = []
        self._parse(filter_str)

    def _parse(self, filter_str: str) -> None:
        # [FUNCTION DOC] HELPER: Parse a comma-separated spec. Each part is a comparison (>=, >, <=, <
        # followed by digits), an inclusive range "a-b", or a single integer. Blank parts are skipped;
        # unparseable parts are logged as warnings and ignored. A range with start greater than end
        # contributes no IDs. Stores results in the explicit-ID set and condition list.
        if not filter_str or not filter_str.strip():
            return

        explicit_ids: set[int] = set()
        conditions: list[tuple[str, int]] = []

        # Pattern for comparison operators (>=, >, <=, <) followed by a number
        cmp_pattern = re.compile(r"^(>=|>|<=|<)(\d+)$")
        # Pattern for ranges like "10-20"
        range_pattern = re.compile(r"^(\d+)-(\d+)$")

        for part in filter_str.split(","):
            part = part.strip()
            if not part:
                continue

            if match := cmp_pattern.match(part):
                conditions.append((match.group(1), int(match.group(2))))
            elif match := range_pattern.match(part):
                start, end = int(match.group(1)), int(match.group(2))
                explicit_ids.update(range(start, end + 1))
            else:
                try:
                    explicit_ids.add(int(part))
                except ValueError:
                    log.warning("Invalid graph ID filter: %s", part)

        self._explicit_ids = frozenset(explicit_ids)
        self._conditions = conditions

    def __contains__(self, graph_id: int) -> bool:
        # [FUNCTION DOC] Return True if graph_id is an explicit ID or satisfies any comparison condition.
        """Check if the given graph ID matches this filter."""
        if graph_id in self._explicit_ids:
            return True

        for op, val in self._conditions:
            if op == ">" and graph_id > val:
                return True
            elif op == ">=" and graph_id >= val:
                return True
            elif op == "<" and graph_id < val:
                return True
            elif op == "<=" and graph_id <= val:
                return True

        return False

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Return a debug string listing sorted explicit IDs and conditions, or "empty".
        # pyrefly: ignore [implicit-any]
        parts = []
        if self._explicit_ids:
            parts.append(f"ids={sorted(self._explicit_ids)}")
        if self._conditions:
            parts.append(f"conditions={self._conditions}")
        return f"GraphIdFilter({', '.join(parts) if parts else 'empty'})"


T = TypeVar("T")


class _GraphRouterBase(Generic[T]):
    # [CLASS DOC] HELPER: Generic base mapping graph IDs to values of type T from a rule string.
    # State: ordered list of (GraphIdFilter, value) rules, a precomputed per-ID value table for IDs
    # 0..max referenced ID, and a single overflow value for all larger IDs (valid because beyond the
    # largest referenced ID only unbounded comparisons can match). Subclasses supply value parsing
    # (_parse_value_str) and may override _match_rules. Default semantics: first matching rule wins.
    """
    Base class for routing graphs to different values based on their IDs.

    The router parses a configuration string with rules in the format:
        "filter1:value1;filter2:value2;..."

    Rules are evaluated in order, and the first matching rule wins.
    """

    def __init__(self, config_str: str, rule_type: str) -> None:
        # [FUNCTION DOC] Set up empty rule/value tables, remember rule_type (used in warnings), parse the
        # config string into rules, then precompute the per-ID lookup table. Subclass errors from
        # value parsing or rule matching propagate from here.
        self._rules: list[tuple[GraphIdFilter, T]] = []
        self._values: list[T | None] = []
        self._overflow_value: T | None = None
        self._rule_type = rule_type
        self._parse(config_str)
        self._precompute()

    def _parse_value_str(self, value_str: str) -> T | None:
        # [FUNCTION DOC] HELPER: Abstract hook; convert a value string to T, or None to skip the rule.
        # Base implementation raises NotImplementedError.
        """Parse a value string into the appropriate type. Returns None to skip."""
        raise NotImplementedError

    def _parse(self, config_str: str) -> None:
        # [FUNCTION DOC] HELPER: Split the config on semicolons into rules, each split at the first colon
        # into a filter and a value string. Rules with a missing colon or empty side are logged and
        # skipped. Values that the subclass parser returns as None are dropped. Appends accepted rules
        # to the rule list in order.
        if not config_str or not config_str.strip():
            return

        rule_strs = config_str.split(";")
        for rule_str in rule_strs:
            rule_str = rule_str.strip()
            if not rule_str:
                continue

            colon_idx = rule_str.find(":")
            if colon_idx == -1:
                log.warning(
                    "Invalid %s override rule (missing ':'): %s",
                    self._rule_type,
                    rule_str,
                )
                continue

            filter_str = rule_str[:colon_idx].strip()
            value_str = rule_str[colon_idx + 1 :].strip()

            if not filter_str or not value_str:
                log.warning("Invalid %s override rule: %s", self._rule_type, rule_str)
                continue

            value = self._parse_value_str(value_str)
            if value is not None:
                self._rules.append((GraphIdFilter(filter_str), value))

    def _precompute(self) -> None:
        # [FUNCTION DOC] HELPER: With no rules, do nothing. Otherwise find the largest ID or threshold
        # referenced by any filter, fill the value table for IDs 0..that max by matching rules, and
        # store the result for the next ID as the overflow value. May raise from _match_rules on conflicts.
        if not self._rules:
            return

        # Find max ID from explicit IDs and comparison thresholds
        max_id = 0
        for f, _ in self._rules:
            if f._explicit_ids:
                max_id = max(max_id, *f._explicit_ids)
            for _, val in f._conditions:
                max_id = max(max_id, val)

        # Pre-compute values for IDs 0 to max_id
        for i in range(max_id + 1):
            self._values.append(self._match_rules(i))

        # For IDs > max_id, the result is constant (only unbounded conditions apply)
        self._overflow_value = self._match_rules(max_id + 1)

    def _match_rules(self, graph_id: int) -> T | None:
        # [FUNCTION DOC] HELPER: Return the value of the first rule whose filter contains graph_id, else None.
        for f, value in self._rules:
            if graph_id in f:
                return value
        return None

    def get_value_for_graph(self, graph_id: int) -> T | None:
        # [FUNCTION DOC] Look up the precomputed value for graph_id (table for small IDs, overflow value
        # otherwise). Returns None if no rule applies. No side effects.
        """Get the value for a given graph ID. Returns None if no rule matches."""
        if graph_id < len(self._values):
            return self._values[graph_id]
        return self._overflow_value

    def is_empty(self) -> bool:
        # [FUNCTION DOC] Return True when no valid rules were parsed.
        """Check if no rules are configured."""
        return len(self._rules) == 0


class GraphBackendRouter(_GraphRouterBase[Any]):
    # [CLASS DOC] Router mapping graph IDs to torch.compile backend callables. Keeps a mapping from
    # backend object identity to the name it was given in the config, used for conflict messages.
    # Overlapping rules that resolve to different backends raise a ValueError during precompute.
    # Parsed backends are registered in the dynamo cached_backends so reset() reaches them.
    """
    Routes graphs to different backends based on their IDs.

    The router parses a configuration string with rules in the format:
        "filter1:backend1;filter2:backend2;..."

    If a graph ID matches multiple rules with different backends, a ValueError
    is raised.

    Examples:
        "0-5:eager;>5:inductor"     - IDs 0-5 use eager, rest use inductor
        ">10:aot_eager"             - IDs > 10 use aot_eager
        "<=3:eager;4-10:aot_eager"  - IDs 0-3 use eager, 4-10 use aot_eager

    Supported backends include "eager", "aot_eager", "aot_eager_decomp_partition",
    "inductor", and any other registered backend.
    """

    def __init__(self, config_str: str) -> None:
        # [FUNCTION DOC] Create the backend-name table before the base initializer parses the config
        # (parsing fills it). Raises as the base class does on invalid backends or conflicts.
        self._backend_names: dict[int, str] = {}
        super().__init__(config_str, "backend")

    def _parse_value_str(self, value_str: str) -> Any | None:
        # [FUNCTION DOC] HELPER: Resolve a backend name through the backend registry. Raises an
        # AssertionError if the lookup returns nothing. Side effects: registers the backend in the
        # global cached_backends mapping (if absent) and records its name by object identity.
        """Look up a backend by name."""
        from .backends.registry import lookup_backend
        from .eval_frame import cached_backends

        backend = lookup_backend(value_str)

        # Register the backend so its reset() is called during torch._dynamo.reset()
        if backend is None:
            raise AssertionError("Invalid override backend: " + value_str)
        cached_backends.setdefault(id(backend), backend)
        self._backend_names[id(backend)] = value_str
        return backend

    def _match_rules(self, graph_id: int) -> Any | None:
        # [FUNCTION DOC] HELPER: Collect distinct backends (by identity) from all rules matching graph_id.
        # Raises ValueError if more than one distinct backend matches; returns the single backend, or
        # None if none match. Differs from the base class in not using first-match-wins.
        """Match rules with conflict detection for overlapping filters."""
        matches = {id(backend): backend for f, backend in self._rules if graph_id in f}
        if len(matches) > 1:
            names = [self._backend_names[bid] for bid in matches]
            raise ValueError(
                f"Conflicting backend override for graph {graph_id}: matched {names}"
            )
        if matches:
            return next(iter(matches.values()))
        return None

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string showing the rules, or "empty".
        if not self._rules:
            return "GraphBackendRouter(empty)"
        return f"GraphBackendRouter({self._rules})"


class GraphConfigRouter(_GraphRouterBase[dict[str, Any]]):
    # [CLASS DOC] Router mapping graph IDs to config-override mappings (string key to scalar value).
    # Unlike first-match routing, all matching rules are merged; the same key with differing values
    # across matching rules raises a ValueError. Values are parsed from comma-separated key=value items.
    """
    Routes graphs to different inductor configs based on their IDs.

    The router parses a configuration string with rules in the format:
        "filter1:config1;filter2:config2;..."

    All matching rules are aggregated: configs from all matching rules are merged
    into a single dict. Conflicting keys (same key, different value) raise an error.
    Config format is "key=value" or "key1=value1,key2=value2" for multiple settings.

    Examples:
        "0-5:triton.cudagraph_skip_dynamic_graphs=False"
        ">10:triton.cudagraphs=False,triton.cudagraph_trees=False"

        With "0:a=1;>=0:b=2", graph 0 gets {"a": 1, "b": 2} (both rules match).
        With "0:a=1;>=0:a=2", graph 0 raises an error (conflicting values for "a").
        With "0:a=1;>=0:a=1", graph 0 gets {"a": 1} (same value is not a conflict).
    """

    def __init__(self, config_str: str) -> None:
        # [FUNCTION DOC] Initialize the base router with rule type "config" and parse config_str.
        super().__init__(config_str, "config")

    @staticmethod
    def _parse_scalar_value(value_str: str) -> Any:
        # [FUNCTION DOC] HELPER: Convert text to a scalar after trimming: true/false (any case) to booleans,
        # "none" to the null value, text containing a dot to a float, else an integer; if numeric
        # conversion fails the trimmed string is returned unchanged.
        """Parse a string value into the appropriate Python type."""
        value_str = value_str.strip()
        if value_str.lower() == "true":
            return True
        if value_str.lower() == "false":
            return False
        if value_str.lower() == "none":
            return None
        try:
            if "." in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            return value_str

    def _match_rules(self, graph_id: int) -> dict[str, Any] | None:
        # [FUNCTION DOC] HELPER: Merge the config mappings of every rule matching graph_id into one new
        # mapping. Raises ValueError if a key appears with different values. Returns None when the
        # merged mapping is empty.
        """Aggregate configs from all matching rules. Conflicts raise an error."""
        result: dict[str, Any] = {}
        for f, value in self._rules:
            if graph_id in f:
                for k, v in value.items():
                    if k in result and result[k] != v:
                        raise ValueError(
                            f"Conflicting config override for graph {graph_id}: "
                            f"key '{k}' has value {result[k]!r} and {v!r}"
                        )
                    result[k] = v
        return result if result else None

    def _parse_value_str(self, value_str: str) -> dict[str, Any] | None:
        # [FUNCTION DOC] HELPER: Parse comma-separated key=value items into a mapping, splitting at the
        # first equals sign and converting values with the scalar parser. Items lacking an equals sign
        # are logged and skipped. Returns None if no items were parsed (so the rule is dropped).
        """Parse a config string like 'key1=val1,key2=val2' into a dict."""
        result: dict[str, Any] = {}
        for item in value_str.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                log.warning("Invalid config item (missing '='): %s", item)
                continue
            key, value = item.split("=", 1)
            result[key.strip()] = self._parse_scalar_value(value)
        return result if result else None

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string showing the rules, or "empty".
        if not self._rules:
            return "GraphConfigRouter(empty)"
        return f"GraphConfigRouter({self._rules})"


def _get_override_for_compile_id(
    compile_id: CompileId | None,
    config_str: str,
    create_router: Callable[[str], _GraphRouterBase[T]],
    label: str,
) -> T | None:
    # [FUNCTION DOC] HELPER: Shared lookup. Returns None if compile_id or config_str is missing or the
    # compile id has no frame_id. Otherwise builds a router via create_router (callers pass cached
    # factories), returns the value for the frame id, and logs it at info level using label.
    """
    Get the override value for a given CompileId.

    Returns the value from the router, or None if no override applies.
    """
    if compile_id is None or not config_str:
        return None

    graph_id = compile_id.frame_id
    if graph_id is None:
        return None

    router = create_router(config_str)
    value = router.get_value_for_graph(graph_id)
    if value is not None:
        log.info("Overriding %s: %s", label, value)
    return value


@functools.lru_cache
def _create_backend_router(config_str: str) -> GraphBackendRouter:
    # [FUNCTION DOC] HELPER: Memoized (unbounded cache) factory for a backend router per config string.
    """Create and cache GraphBackendRouter instances based on config string."""
    return GraphBackendRouter(config_str)


@functools.lru_cache
def _validate_backend_names(config_str: str) -> str | None:
    # [FUNCTION DOC] HELPER: Memoized. Check each backend name in the rule string against the registry.
    # Returns an error message naming the first unresolvable backend, or None if all are valid or
    # the string is blank. Rules without a colon or with empty names are skipped.
    """Return an error message if any backend name is invalid, else None."""
    if not config_str or not config_str.strip():
        return None
    from .backends.registry import lookup_backend

    for rule_str in config_str.split(";"):
        rule_str = rule_str.strip()
        if not rule_str or ":" not in rule_str:
            continue
        backend_name = rule_str[rule_str.find(":") + 1 :].strip()
        if not backend_name:
            continue
        try:
            lookup_backend(backend_name)
        except Exception:
            return (
                f"TORCH_COMPILE_OVERRIDE_BACKENDS: "
                f"'{backend_name}' is not a valid backend, "
                f"see `torch._dynamo.list_backends()` for available backends"
            )
    return None


@functools.lru_cache
def _validate_inductor_config_keys(config_str: str) -> str | None:
    # [FUNCTION DOC] HELPER: Memoized. Parse the string into a config router and verify every key exists
    # as an attribute of the inductor config module. Returns an error message for the first unknown
    # key, else None.
    """Return an error message if any config key is invalid, else None."""
    router = GraphConfigRouter(config_str)
    from torch._inductor import config

    for _, config_dict in router._rules:
        for key in config_dict:
            if not hasattr(config, key):
                return (
                    f"TORCH_COMPILE_OVERRIDE_INDUCTOR_CONFIGS: "
                    f"'{key}' is not a valid torch._inductor.config option"
                )
    return None


@functools.lru_cache
def _validate_dynamo_config_keys(config_str: str) -> str | None:
    # [FUNCTION DOC] HELPER: Same as the inductor key validator but checks the dynamo config module.
    """Return an error message if any config key is invalid, else None."""
    router = GraphConfigRouter(config_str)
    from torch._dynamo import config

    for _, config_dict in router._rules:
        for key in config_dict:
            if not hasattr(config, key):
                return (
                    f"TORCH_COMPILE_OVERRIDE_DYNAMO_CONFIGS: "
                    f"'{key}' is not a valid torch._dynamo.config option"
                )
    return None


@functools.lru_cache
def _create_inductor_config_router(config_str: str) -> GraphConfigRouter:
    # [FUNCTION DOC] HELPER: Memoized factory for a config router for inductor overrides.
    """Create and cache GraphConfigRouter for inductor config overrides."""
    return GraphConfigRouter(config_str)


@functools.lru_cache
def _create_dynamo_config_router(config_str: str) -> GraphConfigRouter:
    # [FUNCTION DOC] HELPER: Memoized factory for a config router for dynamo overrides. Emits a Python
    # warning (once per distinct string, due to caching) when the router has rules, since overrides
    # are keyed by frame ID and dynamo configs can shift frame IDs.
    """Create and cache GraphConfigRouter for dynamo config overrides.

    Warns that dynamo config overrides are keyed by frame ID and some configs
    can affect graph breaks, which may shift frame IDs.
    """
    router = GraphConfigRouter(config_str)
    if not router.is_empty():
        warnings.warn(
            "TORCH_COMPILE_OVERRIDE_DYNAMO_CONFIGS is set. Dynamo config overrides are "
            "keyed by frame ID. Some dynamo configs can affect graph breaks, "
            "which may alter the number of frames and shift frame IDs, causing "
            "overrides to target the wrong graphs.",
        )
    return router


def get_backend_override_for_compile_id(
    compile_id: CompileId | None,
    config_str: str,
) -> Any:
    # [FUNCTION DOC] Public entry: return the backend callable configured for the compile id's frame
    # id by the override string, or None when no override applies. May raise ValueError on
    # conflicting backend rules or AssertionError for unknown backends.
    """
    Get the backend override for a given CompileId.

    Returns the backend function to use, or None if no override applies.
    """
    return _get_override_for_compile_id(
        compile_id,
        config_str,
        _create_backend_router,
        "torch.compile backend",
    )


def get_inductor_config_override_for_compile_id(
    compile_id: CompileId | None,
    config_str: str,
) -> dict[str, Any] | None:
    # [FUNCTION DOC] Public entry: return the merged inductor config patch mapping for the compile id's
    # frame id, or None. May raise ValueError on conflicting values for a key.
    """
    Get the inductor config override for a given CompileId.

    Returns a dict of config patches to apply, or None if no override applies.
    """
    return _get_override_for_compile_id(
        compile_id,
        config_str,
        _create_inductor_config_router,  # type: ignore[arg-type]
        "inductor config",
    )


def get_dynamo_config_override_for_compile_id(
    compile_id: CompileId | None,
    config_str: str,
) -> dict[str, Any] | None:
    # [FUNCTION DOC] Public entry: return the merged dynamo config patch mapping for the compile id's
    # frame id, or None. Warns (once per string) that frame IDs may shift. May raise ValueError on conflicts.
    """
    Get the dynamo config override for a given CompileId.

    Returns a dict of config patches to apply, or None if no override applies.
    """
    return _get_override_for_compile_id(
        compile_id,
        config_str,
        _create_dynamo_config_router,  # type: ignore[arg-type]
        "dynamo config",
    )


# Keep old name for backwards compatibility
_create_router = _create_backend_router
