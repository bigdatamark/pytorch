# mypy: allow-untyped-defs
# [PACKAGE DOC] Backward-compatibility switches for old broadcasting and keepdim behavior.
# Exports broadcast_warning and keepdim_warning, Warning objects whose enabled property reads
# and writes the corresponding native warning flags. Imports private getters and setters from
# torch._C.
from torch._C import (
    _get_backcompat_broadcast_warn,
    _get_backcompat_keepdim_warn,
    _set_backcompat_broadcast_warn,
    _set_backcompat_keepdim_warn,
)


class Warning:
    # [CLASS DOC] Property-style wrapper around a native setter and getter pair. State: setter
    # and getter callables. The enabled property reads or writes the flag. Note the name
    # shadows the built-in Warning.
    def __init__(self, setter, getter) -> None:
        # [FUNCTION DOC] Store the setter and getter callables.
        self.setter = setter
        self.getter = getter

    def set_enabled(self, value) -> None:
        # [FUNCTION DOC] Set the native flag by calling the setter with value.
        self.setter(value)

    def get_enabled(self):
        # [FUNCTION DOC] Return the current native flag value from the getter.
        return self.getter()

    enabled = property(get_enabled, set_enabled)


broadcast_warning = Warning(
    _set_backcompat_broadcast_warn, _get_backcompat_broadcast_warn
)
keepdim_warning = Warning(_set_backcompat_keepdim_warn, _get_backcompat_keepdim_warn)
