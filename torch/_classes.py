# [MODULE DOC] Implements the torch.classes pseudo-module used to access
# TorchScript custom (C++-registered) classes by namespace and name, and to load
# shared libraries that register such classes. Main contents: _ClassNamespace,
# _Classes, and the module-level singleton instance named classes. Import-time
# side effect: imports torch._C and creates the singleton.
import types
from typing import Any

import torch._C


class _ClassNamespace(types.ModuleType):
    # [CLASS DOC] HELPER: Module-like object for one custom-class namespace.
    # Stores its namespace name; attribute lookups for unknown names are resolved
    # lazily through the native registry, returning the Python wrapper (proxy)
    # for the registered class.
    def __init__(self, name: str) -> None:
        # [FUNCTION DOC] HELPER: Initializes the module with a torch.classes
        # prefixed name and stores the namespace name. No separator is added
        # between the prefix and name.
        super().__init__("torch.classes" + name)
        self.name = name

    def __getattr__(self, attr: str) -> Any:
        # [FUNCTION DOC] HELPER: Called only for attributes not found normally.
        # Looks up the custom class wrapper in this namespace via the native
        # registry and returns it; raises RuntimeError if not registered.
        proxy = torch._C._get_custom_class_python_wrapper(self.name, attr)
        if proxy is None:
            raise RuntimeError(f"Class {self.name}.{attr} not registered!")
        return proxy


class _Classes(types.ModuleType):
    # [CLASS DOC] Module-like root of the torch.classes API. Unknown attributes
    # become cached _ClassNamespace children; exposes the set of loaded libraries
    # and a loader for shared libraries registering custom classes.
    __file__ = "_classes.py"

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes the module with the name torch.classes.
        super().__init__("torch.classes")

    def __getattr__(self, name: str) -> _ClassNamespace:
        # [FUNCTION DOC] HELPER: Creates a namespace object for the requested
        # name, caches it as an attribute on this object (mutation) so later
        # lookups bypass this method, and returns it. Never raises.
        namespace = _ClassNamespace(name)
        setattr(self, name, namespace)
        return namespace

    @property
    def loaded_libraries(self) -> Any:
        # [FUNCTION DOC] Property returning the set of library paths loaded so
        # far, shared with the torch.ops loader.
        return torch.ops.loaded_libraries

    def load_library(self, path: str) -> None:
        # [FUNCTION DOC] Loads the shared library at the given path into the
        # process by delegating to the torch.ops loader. Static registration code
        # in the library may register custom classes (global side effect); the
        # path is recorded in the loaded libraries set. Errors from loading
        # propagate.
        """
        Loads a shared library from the given path into the current process.

        The library being loaded may run global initialization code to register
        custom classes with the PyTorch JIT runtime. This allows dynamically
        loading custom classes. For this, you should compile your class
        and the static registration code into a shared library object, and then
        call ``torch.classes.load_library('path/to/libcustom.so')`` to load the
        shared object.

        After the library is loaded, it is added to the
        ``torch.classes.loaded_libraries`` attribute, a set that may be inspected
        for the paths of all libraries loaded using this function.

        Args:
            path (str): A path to a shared library to load.
        """
        torch.ops.load_library(path)


# The classes "namespace"
classes = _Classes()
