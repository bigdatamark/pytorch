# [MODULE DOC] Defines is_from_package, a check for whether an object came from a loaded
# torch.package archive (detected through mangled module names).
from types import ModuleType
from typing import Any

from .._mangling import is_mangled


def is_from_package(obj: Any) -> bool:
    # [FUNCTION DOC] Returns True if the object is a module, or an instance of a type, whose
    # module name is mangled (i.e. it was imported from a package). Modules are checked by
    # their own name, other objects by the module of their type. Externed modules are not
    # mangled so give False. Pure.
    """
    Return whether an object was loaded from a package.

    Note: packaged objects from externed modules will return ``False``.
    """
    if type(obj) is ModuleType:
        return is_mangled(obj.__name__)
    else:
        return is_mangled(type(obj).__module__)
