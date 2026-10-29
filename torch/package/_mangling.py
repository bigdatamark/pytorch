# mypy: allow-untyped-defs
# [MODULE DOC] Name mangling for modules loaded from a package, so they never collide with real
# modules. Provides PackageMangler (per-importer unique prefix) and functions to test, strip and
# extract mangle prefixes. Module-level state: a global counter of manglers created.
"""Import mangling.
See mangling.md for details.
"""

import re


_mangle_index = 0


class PackageMangler:
    # [CLASS DOC] Gives all modules imported by one importer a shared unique mangled parent name
    # of the form angle-bracketed torch_package plus a number. State: this instance's index and
    # parent name; the index comes from a process-wide counter incremented per instance.
    """
    Used on import, to ensure that all modules imported have a shared mangle parent.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Takes the next value of the global counter as this mangler's index,
        # increments the counter (global side effect) and builds the parent name from it.
        global _mangle_index
        self._mangle_index = _mangle_index
        # Increment the global index
        _mangle_index += 1
        # Angle brackets are used so that there is almost no chance of
        # confusing this module for a real module. Plus, it is Python's
        # preferred way of denoting special modules.
        self._mangle_parent = f"<torch_package_{self._mangle_index}>"

    def mangle(self, name) -> str:
        # [FUNCTION DOC] Returns the parent name, a dot, then name. Raises an error if name is
        # empty.
        if len(name) == 0:
            raise AssertionError("name must not be empty")
        return self._mangle_parent + "." + name

    def demangle(self, mangled: str) -> str:
        # [FUNCTION DOC] Strips this mangler's prefix (everything up to the first dot) from a
        # name it produced; names with any other prefix are returned unchanged.
        """
        Note: This only demangles names that were mangled by this specific
        PackageMangler. It will pass through names created by a different
        PackageMangler instance.
        """
        if mangled.startswith(self._mangle_parent + "."):
            return mangled.partition(".")[2]

        # wasn't a mangled name
        return mangled

    def parent_name(self):
        # [FUNCTION DOC] Returns this mangler's parent name string.
        return self._mangle_parent


def is_mangled(name: str) -> bool:
    # [FUNCTION DOC] Returns whether name begins with the angle-bracketed torch_package prefix
    # followed by digits (any mangler).
    return bool(re.match(r"<torch_package_\d+>", name))


def demangle(name: str) -> str:
    # [FUNCTION DOC] Removes the mangle prefix from any mangled name, returning the rest
    # after the first dot, or an empty string if only the prefix is present. Unmangled
    # names are returned as is.
    """
    Note: Unlike PackageMangler.demangle, this version works on any
    mangled name, irrespective of which PackageMangler created it.
    """
    if is_mangled(name):
        _first, sep, last = name.partition(".")
        # If there is only a base mangle prefix, e.g. '<torch_package_0>',
        # then return an empty string.
        return last if len(sep) != 0 else ""
    return name


def get_mangle_prefix(name: str) -> str:
    # [FUNCTION DOC] Returns the first dotted component of a mangled name (the mangle prefix),
    # or the whole name if it is not mangled.
    return name.partition(".")[0] if is_mangled(name) else name
