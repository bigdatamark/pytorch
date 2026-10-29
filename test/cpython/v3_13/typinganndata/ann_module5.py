# [MODULE DOC] HELPER: fixture with postponed annotations using Final, at module level (name) and
# in a class (value); used to check Final inside forward references.
# Used by test_typing to verify that Final wrapped in ForwardRef works.

from __future__ import annotations

from typing import Final

name: Final[str] = "final"

class MyClass:
    # [CLASS DOC] HELPER: class with one attribute annotated as bare Final, value 3000.
    value: Final = 3000
