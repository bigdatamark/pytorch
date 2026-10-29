# [MODULE DOC] Importer abstractions for torch.package. Defines two error types, the abstract
# Importer (import a module by name, and find the module-relative name of an object), the
# default sys_importer instance and OrderedImporter, which tries several importers in turn.
# Import-time side effect: creates a module logger and the sys_importer singleton.
import importlib
import logging
import sys
from abc import ABC, abstractmethod
from pickle import (
    _getattribute,  # pyrefly: ignore [missing-module-attribute]
    _Pickler,
    whichmodule as _pickle_whichmodule,  # pyrefly: ignore [missing-module-attribute]
)
from types import ModuleType
from typing import Any

from ._mangling import demangle, get_mangle_prefix, is_mangled


__all__ = ["ObjNotFoundError", "ObjMismatchError", "Importer", "OrderedImporter"]
log = logging.getLogger(__name__)


class ObjNotFoundError(Exception):
    # [CLASS DOC] Error raised when an importer cannot locate an object by its name.
    """Raised when an importer cannot find an object by searching for its name."""


class ObjMismatchError(Exception):
    # [CLASS DOC] Error raised when looking up an object's name in an importer yields a
    # different object, typically because the same name exists in two environments.
    """Raised when an importer found a different object with the same name as the user-provided one."""


class Importer(ABC):
    # [CLASS DOC] Abstract environment from which modules can be imported. Subclasses supply
    # import_module and may override whichmodule; they are expected to provide a mapping of
    # module names to module objects as the modules attribute (used by the default
    # whichmodule). The provided get_name performs the round trip check: find the module and
    # qualified name of an object, re-import and confirm it is the same object, else raise
    # ObjNotFoundError or ObjMismatchError.
    """Represents an environment to import modules from.

    By default, you can figure out what module an object belongs by checking
    __module__ and importing the result using __import__ or importlib.import_module.

    torch.package introduces module importers other than the default one.
    Each PackageImporter introduces a new namespace. Potentially a single
    name (e.g. 'foo.bar') is present in multiple namespaces.

    It supports two main operations:
        import_module: module_name -> module object
        get_name: object -> (parent module name, name of obj within module)

    The guarantee is that following round-trip will succeed or throw an ObjNotFoundError/ObjMisMatchError.
        module_name, obj_name = env.get_name(obj)
        module = env.import_module(module_name)
        obj2 = getattr(module, obj_name)
        assert obj1 is obj2
    """

    modules: dict[str, ModuleType]

    @abstractmethod
    def import_module(self, module_name: str) -> ModuleType:
        # [FUNCTION DOC] Abstract: imports and returns the named module from this
        # environment, following the contract of the standard library import function
        # (raising ModuleNotFoundError or ImportError on failure).
        """Import `module_name` from this environment.

        The contract is the same as for importlib.import_module.
        """

    def get_name(self, obj: Any, name: str | None = None) -> tuple[str, str]:
        # [FUNCTION DOC] Returns a pair of (module name, attribute path) under which obj can be
        # retrieved from this environment. The attribute path is the explicit name if given,
        # else a string returned by calling the object's reduce method (only for types without
        # a pickler dispatch entry; failures ignored), else its qualified name, else its
        # name. The module comes from whichmodule and is demangled, imported through this
        # importer and the attribute resolved. Raises ObjNotFoundError if import or lookup
        # fails, and ObjMismatchError (with a message explaining importer ordering) if a
        # different object is found.
        """Given an object, return a name that can be used to retrieve the
        object from this environment.

        Args:
            obj: An object to get the module-environment-relative name for.
            name: If set, use this name instead of looking up __name__ or __qualname__ on `obj`.
                This is only here to match how Pickler handles __reduce__ functions that return a string,
                don't use otherwise.
        Returns:
            A tuple (parent_module_name, attr_name) that can be used to retrieve `obj` from this environment.
            Use it like:
                mod = importer.import_module(parent_module_name)
                obj = getattr(mod, attr_name)

        Raises:
            ObjNotFoundError: we couldn't retrieve `obj by name.
            ObjMisMatchError: we found a different object with the same name as `obj`.
        """
        if name is None and obj and _Pickler.dispatch.get(type(obj)) is None:
            # Honor the string return variant of __reduce__, which will give us
            # a global name to search for in this environment.
            # TODO: I guess we should do copyreg too?
            reduce = getattr(obj, "__reduce__", None)
            if reduce is not None:
                try:
                    rv = reduce()
                    if isinstance(rv, str):
                        name = rv
                except Exception:
                    pass
        if name is None:
            name = getattr(obj, "__qualname__", None)
        if name is None:
            name = obj.__name__

        orig_module_name = self.whichmodule(obj, name)
        # Demangle the module name before importing. If this obj came out of a
        # PackageImporter, `__module__` will be mangled. See mangling.md for
        # details.
        module_name = demangle(orig_module_name)

        # Check that this name will indeed return the correct object
        try:
            module = self.import_module(module_name)
            if sys.version_info >= (3, 14):
                # pickle._getattribute signature changes in 3.14
                # to take iterable and return just one object
                obj2 = _getattribute(module, name.split("."))
            else:
                obj2, _ = _getattribute(module, name)
        except (ImportError, KeyError, AttributeError):
            raise ObjNotFoundError(
                f"{obj} was not found as {module_name}.{name}"
            ) from None

        if obj is obj2:
            return module_name, name

        def get_obj_info(obj):
            # [FUNCTION DOC] HELPER: For the given object returns a tuple of its module name,
            # a human readable origin (the package mangle prefix or the current environment)
            # and the name of the importer that would provide it. Raises AssertionError if
            # the enclosing name is None.
            if name is None:
                raise AssertionError("name must not be None")
            module_name = self.whichmodule(obj, name)
            is_mangled_ = is_mangled(module_name)
            location = (
                get_mangle_prefix(module_name)
                if is_mangled_
                else "the current Python environment"
            )
            importer_name = (
                f"the importer for {get_mangle_prefix(module_name)}"
                if is_mangled_
                else "'sys_importer'"
            )
            return module_name, location, importer_name

        obj_module_name, obj_location, obj_importer_name = get_obj_info(obj)
        obj2_module_name, obj2_location, obj2_importer_name = get_obj_info(obj2)
        msg = (
            f"\n\nThe object provided is from '{obj_module_name}', "
            f"which is coming from {obj_location}."
            f"\nHowever, when we import '{obj2_module_name}', it's coming from {obj2_location}."
            "\nTo fix this, make sure this 'PackageExporter's importer lists "
            f"{obj_importer_name} before {obj2_importer_name}."
        )
        raise ObjMismatchError(msg)

    def whichmodule(self, obj: Any, name: str) -> str:
        # [FUNCTION DOC] Returns the module name an object belongs to: its module attribute if
        # present, else the first loaded module (from a copy of the modules mapping, skipping
        # main modules and empty entries) that holds the same object at the given name, else
        # the main module name. Pure apart from attribute access on modules.
        """Find the module name an object belongs to.

        This should be considered internal for end-users, but developers of
        an importer can override it to customize the behavior.

        Taken from pickle.py, but modified to exclude the search into sys.modules
        """
        module_name = getattr(obj, "__module__", None)
        if module_name is not None:
            return module_name

        # Protect the iteration by using a list copy of self.modules against dynamic
        # modules that trigger imports of other modules upon calls to getattr.
        for module_name, module in self.modules.copy().items():
            if (
                module_name == "__main__"
                or module_name == "__mp_main__"  # bpo-42406
                or module is None
            ):
                continue
            try:
                if _getattribute(module, name)[0] is obj:
                    return module_name
            except AttributeError:
                pass

        return "__main__"


class _SysImporter(Importer):
    # [CLASS DOC] HELPER: Importer that behaves like normal Python: modules come from the
    # standard import machinery. A single instance is exported as sys_importer.
    """An importer that implements the default behavior of Python."""

    def import_module(self, module_name: str):
        # [FUNCTION DOC] HELPER: Imports and returns the module with the standard library
        # import function (may populate the interpreter module cache; raises on failure).
        return importlib.import_module(module_name)

    def whichmodule(self, obj: Any, name: str) -> str:
        # [FUNCTION DOC] HELPER: Returns the module attribute of obj when present (so mangled
        # package names are not imported); otherwise falls back to the pickle library's module
        # search for the given name.
        # In Python 3.14+, pickle.whichmodule tries to import the module,
        # which fails for mangled package names like '<torch_package_0>'.
        # Check __module__ first before calling pickle.whichmodule.
        module_name = getattr(obj, "__module__", None)
        if module_name is not None:
            return module_name
        return _pickle_whichmodule(obj, name)


sys_importer = _SysImporter()


class OrderedImporter(Importer):
    # [CLASS DOC] Importer that wraps an ordered list of other importers (kept in a private
    # list) and delegates each operation to them in order, first success wins. Used by the
    # exporter and importer to combine a package with the system environment.
    """A compound importer that takes a list of importers and tries them one at a time.

    The first importer in the list that returns a result "wins".
    """

    def __init__(self, *args):
        # [FUNCTION DOC] Stores the given importers, in order, as a list.
        self._importers: list[Importer] = list(args)

    def _is_torchpackage_dummy(self, module):
        # [FUNCTION DOC] HELPER: Returns True if the module is a package-flagged namespace
        # placeholder: marked as a torch package module, has a path attribute, and either
        # lacks a file attribute or has it set to None. False otherwise.
        """Returns true iff this module is an empty PackageNode in a torch.package.

        If you intern `a.b` but never use `a` in your code, then `a` will be an
        empty module with no source. This can break cases where we are trying to
        re-package an object after adding a real dependency on `a`, since
        OrderedImporter will resolve `a` to the dummy package and stop there.

        See: https://github.com/pytorch/pytorch/pull/71520#issuecomment-1029603769
        """
        if not getattr(module, "__torch_package__", False):
            return False
        if not hasattr(module, "__path__"):
            return False
        if not hasattr(module, "__file__"):
            return True
        return module.__file__ is None

    def get_name(self, obj: Any, name: str | None = None) -> tuple[str, str]:
        # [FUNCTION DOC] Asks each importer in order for the name of obj and returns the first
        # successful (module name, attribute path) pair. Failures from an importer are logged
        # as warnings and skipped. Raises ObjNotFoundError if none of them succeeds.
        for importer in self._importers:
            try:
                return importer.get_name(obj, name)
            except (ObjNotFoundError, ObjMismatchError) as e:
                warning_message = (
                    f"Tried to call get_name with obj {obj}, "
                    f"and name {name} on {importer} and got {e}"
                )
                log.warning(warning_message)
        raise ObjNotFoundError(
            f"Could not find obj {obj} and name {name} in any of the importers {self._importers}"
        )

    def import_module(self, module_name: str) -> ModuleType:
        # [FUNCTION DOC] Tries each importer in order and returns the first module found that
        # is not an empty package placeholder. Raises TypeError if an entry is not an
        # Importer (checked lazily when reached), re-raises the last ModuleNotFoundError if
        # every importer failed, or raises a new one if all results were placeholders.
        last_err = None
        for importer in self._importers:
            if not isinstance(importer, Importer):
                raise TypeError(
                    f"{importer} is not an Importer. "
                    "All importers in OrderedImporter must inherit from Importer."
                )
            try:
                module = importer.import_module(module_name)
                if self._is_torchpackage_dummy(module):
                    continue
                return module
            except ModuleNotFoundError as err:
                last_err = err

        if last_err is not None:
            raise last_err
        else:
            raise ModuleNotFoundError(module_name)

    def whichmodule(self, obj: Any, name: str) -> str:
        # [FUNCTION DOC] Returns the first module name reported by the importers in order
        # that is not the main module name; returns the main module name if none qualifies.
        for importer in self._importers:
            module_name = importer.whichmodule(obj, name)
            if module_name != "__main__":
                return module_name

        return "__main__"
