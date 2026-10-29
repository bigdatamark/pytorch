# [MODULE DOC] HELPER: Picklers for torch.package. Defines a base pickler with an overridable
# persistent id hook, PackagePickler (looks up objects through an Importer rather than the
# standard import system) and the create_pickler factory. No import-time side effects.
import sys
from pickle import (
    _compat_pickle,  # pyrefly: ignore [missing-module-attribute]
    _extension_registry,  # pyrefly: ignore [missing-module-attribute]
    _getattribute,  # pyrefly: ignore [missing-module-attribute]
    _Pickler,
    EXT1,
    EXT2,
    EXT4,
    GLOBAL,
    PicklingError,
    STACK_GLOBAL,
)
from struct import pack
from types import FunctionType

from .importer import Importer, ObjMismatchError, ObjNotFoundError, sys_importer


class _PyTorchLegacyPickler(_Pickler):
    # [CLASS DOC] HELPER: Pure-Python pickler whose persistent id behavior can be replaced at
    # runtime by assigning a callable to the private persistent id attribute (None by
    # default, meaning the base behavior is used).
    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Forwards all arguments to the base pickler and initializes
        # the custom persistent id callable to None.
        super().__init__(*args, **kwargs)
        self._persistent_id = None

    def persistent_id(self, obj):
        # [FUNCTION DOC] HELPER: Returns the persistent id for obj. Uses the base class
        # behavior if no custom callable was set, otherwise delegates to the custom callable.
        if self._persistent_id is None:
            return super().persistent_id(obj)
        return self._persistent_id(obj)


class PackagePickler(_PyTorchLegacyPickler):
    # [CLASS DOC] Pickler that saves classes and functions by name, resolving them through an
    # Importer (public attribute importer) so objects from loaded packages can be re-pickled.
    # Its dispatch table is a fresh copy of the base one with functions routed to the
    # overridden save_global, which is a copy of the stdlib logic with the import step changed.
    """Package-aware pickler.

    This behaves the same as a normal pickler, except it uses an `Importer`
    to find objects and modules to save.
    """

    def __init__(self, importer: Importer, *args, **kwargs):
        # [FUNCTION DOC] Stores the importer, initializes the base pickler with the remaining
        # arguments, then replaces the instance dispatch table with a fresh copy of the base
        # table (guarding against third-party mutation) mapping function objects to save_global.
        self.importer = importer
        super().__init__(*args, **kwargs)

        # Make sure the dispatch table copied from _Pickler is up-to-date.
        # Previous issues have been encountered where a library (e.g. dill)
        # mutate _Pickler.dispatch, PackagePickler makes a copy when this lib
        # is imported, then the offending library removes its dispatch entries,
        # leaving PackagePickler with a stale dispatch table that may cause
        # unwanted behavior.
        self.dispatch = _Pickler.dispatch.copy()  # type: ignore[misc]
        self.dispatch[FunctionType] = PackagePickler.save_global  # type: ignore[assignment]

    def save_global(self, obj, name=None):
        # [FUNCTION DOC] Writes a reference to a global (class or function) obj to the pickle
        # stream. Asks the importer for the module name and qualified name, imports that module
        # via the importer, and finds the parent object. Then emits an extension-registry code
        # if registered, else a stack-global (protocol 4+), a getattr reduce for nested names,
        # or a global opcode (older protocols, with Python 2 name mapping when fix_imports is
        # on). Memoizes obj. Raises PicklingError if the object is not found or mismatches,
        # a local object is pickled (Python 3.14+), or the names cannot be ASCII-encoded for
        # old protocols. Side effects: writes to the output stream and memo.
        # ruff: noqa: F841
        # unfortunately the pickler code is factored in a way that
        # forces us to copy/paste this function. The only change is marked
        # CHANGED below.
        write = self.write  # type: ignore[attr-defined]
        memo = self.memo  # type: ignore[attr-defined]

        # CHANGED: import module from module environment instead of __import__
        try:
            module_name, name = self.importer.get_name(obj, name)
        except (ObjNotFoundError, ObjMismatchError) as err:
            raise PicklingError(f"Can't pickle {obj}: {str(err)}") from err

        module = self.importer.import_module(module_name)
        if sys.version_info >= (3, 14):
            # pickle._getattribute signature changes in 3.14
            # to take iterable and return just the object (not tuple)
            # We need to get the parent object that contains the attribute
            name_parts = name.split(".")
            if "<locals>" in name_parts:
                raise PicklingError(f"Can't pickle local object {obj!r}")
            if len(name_parts) == 1:
                parent = module
            else:
                parent = _getattribute(module, name_parts[:-1])
        else:
            _, parent = _getattribute(module, name)
        # END CHANGED

        if self.proto >= 2:  # type: ignore[attr-defined]
            code = _extension_registry.get((module_name, name))
            if code:
                if code <= 0:
                    raise AssertionError(
                        f"expected positive extension code, got {code}"
                    )
                if code <= 0xFF:
                    write(EXT1 + pack("<B", code))
                elif code <= 0xFFFF:
                    write(EXT2 + pack("<H", code))
                else:
                    write(EXT4 + pack("<i", code))
                return
        lastname = name.rpartition(".")[2]
        if parent is module:
            name = lastname
        # Non-ASCII identifiers are supported only with protocols >= 3.
        if self.proto >= 4:  # type: ignore[attr-defined]
            self.save(module_name)  # type: ignore[attr-defined]
            self.save(name)  # type: ignore[attr-defined]
            write(STACK_GLOBAL)
        elif parent is not module:
            self.save_reduce(getattr, (parent, lastname))  # type: ignore[attr-defined]
        elif self.proto >= 3:  # type: ignore[attr-defined]
            write(
                GLOBAL
                + bytes(module_name, "utf-8")
                + b"\n"
                + bytes(name, "utf-8")
                + b"\n"
            )
        else:
            if self.fix_imports:  # type: ignore[attr-defined]
                r_name_mapping = _compat_pickle.REVERSE_NAME_MAPPING
                r_import_mapping = _compat_pickle.REVERSE_IMPORT_MAPPING
                if (module_name, name) in r_name_mapping:
                    module_name, name = r_name_mapping[(module_name, name)]
                elif module_name in r_import_mapping:
                    module_name = r_import_mapping[module_name]
            try:
                write(
                    GLOBAL
                    + bytes(module_name, "ascii")
                    + b"\n"
                    + bytes(name, "ascii")
                    + b"\n"
                )
            except UnicodeEncodeError as exc:
                raise PicklingError(
                    f"can't pickle global identifier '{module}.{name}' using "
                    f"pickle protocol {self.proto:d}"  # type: ignore[attr-defined]
                ) from exc

        self.memoize(obj)  # type: ignore[attr-defined]


def create_pickler(data_buf, importer, protocol=4):
    # [FUNCTION DOC] HELPER: Factory returning a pickler that writes to data_buf. If the
    # importer is the default system importer, returns the plain legacy pickler (faster path);
    # otherwise returns a PackagePickler bound to the importer. Protocol defaults to 4.
    if importer is sys_importer:
        # if we are using the normal import library system, then
        # we can use the C implementation of pickle which is faster
        return _PyTorchLegacyPickler(data_buf, protocol=protocol)
    else:
        return PackagePickler(importer, data_buf, protocol=protocol)
