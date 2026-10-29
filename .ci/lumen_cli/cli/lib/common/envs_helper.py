# [MODULE DOC] Helpers that read environment variables as strings, booleans and paths,
# dataclass field factories that defer reading the environment until instantiation, and
# utilities that render dataclass defaults as help text. No import-time side effects.
"""
Environment Variables and Dataclasses Utility helpers for CLI tasks.
"""

from __future__ import annotations

import os
from dataclasses import field, fields, is_dataclass, MISSING
from pathlib import Path
from textwrap import indent

from cli.lib.common.utils import str2bool


def get_env(name: str, default: str = "") -> str:
    # [FUNCTION DOC] Return the environment variable value, or default when it is unset
    # or empty (an empty value counts as missing).
    """Get environment variable with default fallback."""
    return os.environ.get(name) or default


def env_path_optional(
    name: str,
    default: str | Path | None = None,
    resolve: bool = True,
) -> Path | None:
    # [FUNCTION DOC] Read an environment variable as a path, falling back to default.
    # Returns None when both are empty; otherwise a Path, made absolute and symlink
    # resolved when resolve is true.
    """Get environment variable as optional Path."""
    val = get_env(name) or default
    if not val:
        return None

    path = Path(val)
    return path.resolve() if resolve else path


def env_path(
    name: str,
    default: str | Path | None = None,
    resolve: bool = True,
) -> Path:
    # [FUNCTION DOC] Like env_path_optional but never returns None: raises ValueError
    # when neither the variable nor the default provides a value.
    """Get environment variable as Path, raise if missing."""
    path = env_path_optional(name, default, resolve)
    if not path:
        raise ValueError(f"Missing path value for {name}")
    return path


def env_bool(
    name: str,
    default: bool = False,
) -> bool:
    # [FUNCTION DOC] Read an environment variable as a boolean via str2bool; returns
    # default when the variable is unset or empty.
    val = get_env(name)
    if not val:
        return default
    return str2bool(val)


def env_bool_field(
    name: str,
    default: bool = False,
):
    # [FUNCTION DOC] Build a dataclass field whose default is computed at instance
    # creation time from the boolean environment variable (see env_bool).
    return field(default_factory=lambda: env_bool(name, default))


def env_path_field(
    name: str,
    default: str | Path = "",
    *,
    resolve: bool = True,
) -> Path:
    # [FUNCTION DOC] Build a dataclass field whose default is computed at instance
    # creation time with env_path (so it raises ValueError if no value is available).
    # KNOWN ISSUE: the return annotation says Path but a dataclass field object is
    # returned (same for the sibling field helpers).
    return field(default_factory=lambda: env_path(name, default, resolve=resolve))


def env_str_field(
    name: str,
    default: str = "",
) -> str:
    # [FUNCTION DOC] Build a dataclass field whose default is the string environment
    # variable (or default) read at instance creation time.
    return field(default_factory=lambda: get_env(name, default))


def generate_dataclass_help(cls) -> str:
    # [FUNCTION DOC] Render one line per field of a dataclass as name padded to 22
    # columns, an equals sign and the repr of the default, indented four spaces.
    # Raises TypeError if cls is not a dataclass.
    """Auto-generate help text for dataclass fields."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    def get_value(f):
        # [FUNCTION DOC] HELPER: Default of a field: the plain default, else the result
        # of calling the default factory (an error placeholder string if it raises),
        # else the placeholder "<required>".
        if f.default is not MISSING:
            return f.default
        if f.default_factory is not MISSING:
            try:
                return f.default_factory()
            except Exception as e:
                return f"<error: {e}>"
        return "<required>"

    lines = [f"{f.name:<22} = {repr(get_value(f))}" for f in fields(cls)]
    return indent("\n".join(lines), "    ")


def with_params_help(params_cls: type, title: str = "Parameter defaults"):
    # [FUNCTION DOC] Build a class decorator that appends a titled table of the
    # defaults of params_cls (a dataclass) to the decorated class docstring. Raises
    # TypeError immediately if params_cls is not a dataclass. The table is generated
    # when the decorator is applied, so environment based defaults are evaluated then.
    """
    Class decorator that appends a help table generated from another dataclass
    (e.g., VllmParameters) to the decorated class's docstring.
    """
    if not is_dataclass(params_cls):
        raise TypeError(f"{params_cls} must be a dataclass")

    def _decorator(cls: type) -> type:
        # [FUNCTION DOC] HELPER: Append the help block to the docstring of cls (mutates
        # cls) and return it.
        block = generate_dataclass_help(params_cls)
        cls.__doc__ = (cls.__doc__ or "") + f"\n\n{title}:\n{block}"
        return cls

    return _decorator
