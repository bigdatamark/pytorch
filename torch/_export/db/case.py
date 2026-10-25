# [MODULE DOC] ExportDB case registry: tag taxonomy, SupportLevel enum,
# the frozen ExportCase dataclass describing one example, global registries of
# cases/conflicts/rewrites, and decorators to register them. Import-time side
# effect: only defines empty module-level registries.
# mypy: allow-untyped-defs
import inspect
import re
import string
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from types import ModuleType

import torch

_TAGS: dict[str, dict[str, Any]] = {
    "torch": {
        "cond": {},
        "dynamic-shape": {},
        "escape-hatch": {},
        "map": {},
        "dynamic-value": {},
        "operator": {},
        "mutation": {},
    },
    "python": {
        "assert": {},
        "builtin": {},
        "closure": {},
        "context-manager": {},
        "control-flow": {},
        "data-structure": {},
        "standard-library": {},
        "object-model": {},
    },
}


class SupportLevel(Enum):
    # [CLASS DOC] Enum marking whether an example's feature is handled by export:
    # SUPPORTED (1) or NOT_SUPPORTED_YET (0).
    """
    Indicates at what stage the feature
    used in the example is handled in export.
    """

    SUPPORTED = 1
    NOT_SUPPORTED_YET = 0


ArgsType = tuple[Any, ...]


def check_inputs_type(args, kwargs):
    # [FUNCTION DOC] Validates example inputs: args must be a tuple, kwargs a
    # mapping, and every kwargs key a string. Raises ValueError otherwise.
    if not isinstance(args, tuple):
        raise ValueError(
            f"Expecting args type to be a tuple, got: {type(args)}"
        )
    if not isinstance(kwargs, dict):
        raise ValueError(
            f"Expecting kwargs type to be a dict, got: {type(kwargs)}"
        )
    for key in kwargs:
        if not isinstance(key, str):
            raise ValueError(
                f"Expecting kwargs keys to be a string, got: {type(key)}"
            )

def _validate_tag(tag: str):
    # [FUNCTION DOC] HELPER: checks a dotted tag against the nested tag registry.
    # Each segment must consist only of lowercase letters and hyphens (else
    # AssertionError) and must exist at its level of the registry (else ValueError).
    parts = tag.split(".")
    t = _TAGS
    for part in parts:
        if not set(part) <= set(string.ascii_lowercase + "-"):
            raise AssertionError(f"Tag contains invalid characters: {part}")
        if part in t:
            t = t[part]
        else:
            raise ValueError(f"Tag {tag} is not found in registered tags.")


@dataclass(frozen=True)
class ExportCase:
    # [CLASS DOC] Immutable record describing one ExportDB example: the model,
    # example args/kwargs, optional extra args for generalization testing, a
    # unique name, description, tags, support level and dynamic shapes.
    # Invariants checked on construction: input types valid, tags registered,
    # description a non-empty string.
    example_args: ArgsType
    description: str  # A description of the use case.
    model: torch.nn.Module
    name: str
    example_kwargs: dict[str, Any] = field(default_factory=dict)
    extra_args: ArgsType | None = None  # For testing graph generalization.
    # Tags associated with the use case. (e.g dynamic-shape, escape-hatch)
    tags: set[str] = field(default_factory=set)
    support_level: SupportLevel = SupportLevel.SUPPORTED
    dynamic_shapes: dict[str, Any] | None = None

    def __post_init__(self):
        # [FUNCTION DOC] Validates the case after construction: checks example
        # args/kwargs (and extra_args if given) types, validates every tag, and
        # requires a non-empty string description. Raises ValueError or
        # AssertionError on failure; no mutation.
        check_inputs_type(self.example_args, self.example_kwargs)
        if self.extra_args is not None:
            check_inputs_type(self.extra_args, {})

        for tag in self.tags:
            _validate_tag(tag)

        if not isinstance(self.description, str) or len(self.description) == 0:
            raise ValueError(f'Invalid description: "{self.description}"')


_EXAMPLE_CASES: dict[str, ExportCase] = {}
_MODULES: set[ModuleType] = set()
_EXAMPLE_CONFLICT_CASES: dict[str, list[ExportCase]] = {}
_EXAMPLE_REWRITE_CASES: dict[str, list[ExportCase]] = {}


def register_db_case(case: ExportCase) -> None:
    """
    Registers a user provided ExportCase into example bank.
    """
    # [FUNCTION DOC] Adds case to the global registry keyed by name. If the name
    # is already registered, the case is not stored there; instead it is appended
    # to the conflict list for that name (seeded with the original case).
    # Mutates module-level registries; returns None.
    if case.name in _EXAMPLE_CASES:
        if case.name not in _EXAMPLE_CONFLICT_CASES:
            _EXAMPLE_CONFLICT_CASES[case.name] = [_EXAMPLE_CASES[case.name]]
        _EXAMPLE_CONFLICT_CASES[case.name].append(case)
        return

    _EXAMPLE_CASES[case.name] = case


def to_snake_case(name):
    # [FUNCTION DOC] Converts a CamelCase identifier to lowercase snake_case by
    # inserting underscores at word boundaries (before capitalized words and
    # between a lowercase letter/digit and an uppercase letter). Returns the string.
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def _make_export_case(m, name, configs):
    # [FUNCTION DOC] HELPER: builds an ExportCase for module class m under the
    # given name from the configs mapping. Raises TypeError if m is not an
    # nn.Module; if configs has no description, falls back to m's docstring
    # (AssertionError if that is missing). Returns the new ExportCase.
    if not isinstance(m, torch.nn.Module):
        raise TypeError("Export case class should be a torch.nn.Module.")

    if "description" not in configs:
        # Fallback to docstring if description is missing.
        if m.__doc__ is None:
            raise AssertionError(
                f"Could not find description or docstring for export case: {m}"
            )
        configs = {**configs, "description": m.__doc__}
    # pyrefly: ignore [bad-argument-type]
    return ExportCase(**{**configs, "model": m, "name": name})


def export_case(**kwargs):
    """
    Decorator for registering a user provided case into example bank.
    """
    # [FUNCTION DOC] Decorator factory: keyword arguments become ExportCase
    # fields. The returned decorator registers the decorated module class as a
    # case named after its defining module's last name component and returns
    # the ExportCase (replacing the class).

    def wrapper(m):
        # [FUNCTION DOC] HELPER: performs the registration. Raises RuntimeError
        # if the defining module already registered a case (one per file) and
        # AssertionError if the module cannot be determined. Records the module
        # in the global module set and registers the case.
        configs = kwargs
        module = inspect.getmodule(m)
        if module in _MODULES:
            raise RuntimeError("export_case should only be used once per example file.")

        if module is None:
            raise AssertionError("module must not be None")
        _MODULES.add(module)
        module_name = module.__name__.split(".")[-1]
        case = _make_export_case(m, module_name, configs)
        register_db_case(case)
        return case

    return wrapper


def export_rewrite_case(**kwargs):
    # [FUNCTION DOC] Decorator factory for registering a rewritten (fixed)
    # variant of an existing case. Keyword arguments must include parent, an
    # ExportCase; the rest become ExportCase fields.
    def wrapper(m):
        # [FUNCTION DOC] HELPER: removes parent from the options (mutating the
        # shared kwargs mapping), asserts it is an ExportCase, copies its example
        # args, builds a case named by the snake-case class name, appends it to
        # the rewrite list under the parent's name and returns it.
        configs = kwargs

        parent = configs.pop("parent")
        if not isinstance(parent, ExportCase):
            raise AssertionError(f"expected ExportCase, got {type(parent)}")
        key = parent.name
        if key not in _EXAMPLE_REWRITE_CASES:
            _EXAMPLE_REWRITE_CASES[key] = []

        configs["example_args"] = parent.example_args
        case = _make_export_case(m, to_snake_case(m.__name__), configs)
        _EXAMPLE_REWRITE_CASES[key].append(case)
        return case

    return wrapper
