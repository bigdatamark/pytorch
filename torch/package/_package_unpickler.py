# mypy: allow-untyped-defs
# [MODULE DOC] HELPER: Defines PackageUnpickler, a pure-Python unpickler that resolves global
# names through a torch.package Importer. No import-time side effects.
import _compat_pickle
import pickle

from .importer import Importer


class PackageUnpickler(pickle._Unpickler):  # type: ignore[name-defined]
    # [CLASS DOC] Unpickler subclass that looks up classes and functions via an Importer
    # instead of the normal import system. Holds the importer in a private attribute. The key
    # method is find_class, which also applies the Python 2 name remapping when needed.
    """Package-aware unpickler.

    This behaves the same as a normal unpickler, except it uses `importer` to
    find any global names that it encounters while unpickling.
    """

    def __init__(self, importer: Importer, *args, **kwargs):
        # [FUNCTION DOC] Initializes the base unpickler with the remaining arguments and stores
        # the importer used later to resolve global names.
        super().__init__(*args, **kwargs)
        self._importer = importer

    def find_class(self, module, name):
        # [FUNCTION DOC] Resolves a (module, name) global reference found in the pickle. For old
        # protocols (below 3) with fix_imports enabled, first remaps Python 2 names via the
        # compat tables. Then imports the module through the importer and returns the named
        # attribute. Raises whatever the importer raises on missing modules, or AttributeError
        # if the attribute does not exist.
        # Subclasses may override this.
        if self.proto < 3 and self.fix_imports:  # type: ignore[attr-defined]
            if (module, name) in _compat_pickle.NAME_MAPPING:
                module, name = _compat_pickle.NAME_MAPPING[(module, name)]
            elif module in _compat_pickle.IMPORT_MAPPING:
                module = _compat_pickle.IMPORT_MAPPING[module]
        mod = self._importer.import_module(module)
        return getattr(mod, name)
