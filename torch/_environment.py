# [MODULE DOC] Environment detection stubs for the open-source build. Internal
# builds replace this module; here both predicates always report False.
# No import-time side effects.
from typing import Literal


def is_fbcode() -> Literal[False]:
    # [FUNCTION DOC] Reports whether running inside the internal fbcode build.
    # Takes no arguments; always returns False in the open-source tree.
    return False


def is_prod() -> Literal[False]:
    # [FUNCTION DOC] Reports whether running in an internal production environment.
    # Takes no arguments; always returns False in the open-source tree.
    return False
