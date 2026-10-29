# [MODULE DOC] HELPER: fixture with postponed annotations declaring a module-level name as
# ClassVar, which type-hint resolution must reject. Sets wrong to 1 at import.
# Tests that top-level ClassVar is not allowed

from __future__ import annotations

from typing import ClassVar

wrong: ClassVar[int] = 1
