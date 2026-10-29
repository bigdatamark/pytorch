# [MODULE DOC]
# Leaf module defining the two base types of the CUPTI user-defined-record schema: the Ctype
# enumeration (how to interpret a field's bytes) and the Field integer subclass (a field id
# that also carries its Ctype). Has no intra-package imports, so the generated stubs module
# and the curated records module can both import it without a cycle. No import-time side
# effects beyond defining the types.
"""Base types for the CUPTI user-defined-record schema: :class:`Ctype` and :class:`Field`.

Kept in a tiny leaf module (no intra-package imports) so both the generated
``_cupti_stubs`` module and the hand-curated ``records`` module can import it
without a cycle.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing_extensions import Self


class Ctype(Enum):
    # [CLASS DOC]
    # Enumeration of decode interpretations for a record field: unsigned int (value "u", also
    # the default for enums), signed int ("i"), float ("f"), and C string pointer ("cstr").
    # The value is the numpy kind letter for numeric members. Byte width is never stored here;
    # it always comes from CUPTI's captured record layout.
    """How to interpret a field's bytes at decode time. The byte *width* always comes
    from CUPTI's captured record layout (``ppRecordLayouts``); ``Ctype`` only says
    whether those bytes are unsigned/signed/float, or a ``const char*`` to dereference.
    Struct/opaque fields need no ctype -- the decoder skips any field whose layout size
    is not 1/2/4/8."""

    UINT = "u"  # unsigned integer (also the default for enums)
    INT = "i"  # signed integer
    FLOAT = "f"  # float / double
    CSTR = "cstr"  # const char* -- dereferenced to a Python str

    def numpy(self, size: int) -> str:
        # [FUNCTION DOC]
        # Builds the little-endian numpy dtype string for this ctype at the given byte size
        # (a "<" prefix, the kind letter, then the size). Pure; no validation, so calling it
        # on the CSTR member yields a string numpy cannot use as a numeric dtype.
        """Little-endian numpy dtype string for a numeric ctype at ``size`` bytes."""
        return f"<{self.value}{size}"


class Field(int):
    # [CLASS DOC]
    # Integer subclass: the integer value is the CUPTI field id, so a Field works anywhere a
    # plain id is expected (set members, dict keys, selections). Extra attribute: ctype, the
    # Ctype used when decoding that field. Immutable by convention; set once in construction.
    # Exposes id and string properties and a descriptive repr.
    """A CUPTI user-defined-record field. A ``Field`` *is* its ``CUpti_Activity*FieldIds``
    id -- it subclasses ``int`` -- so it can be used directly anywhere the integer id is
    expected (selection element, column key, set/dict member) without ``.id``/``int()``.
    It additionally carries its :class:`Ctype` for decode."""

    ctype: Ctype

    def __new__(cls, id: int, ctype: Ctype) -> Self:
        # [FUNCTION DOC]
        # Creates a Field whose integer value is the given field id and attaches the given
        # Ctype as the ctype attribute. Returns the new instance. No validation of arguments.
        self = super().__new__(cls, id)
        self.ctype = ctype
        return self

    @property
    def id(self) -> int:
        # [FUNCTION DOC]
        # Property returning the field id as a plain int (same value as the instance itself).
        """The field id (== ``int(self)``); kept for call sites that read ``.id``."""
        return int(self)

    @property
    def string(self) -> bool:
        # [FUNCTION DOC]
        # Property: true when this field's ctype is the C string pointer kind, meaning the
        # decoder must dereference the stored pointer into a text value; false otherwise.
        """True for ``const char*`` fields (dereferenced to str during decode)."""
        return self.ctype is Ctype.CSTR

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns a debug string showing the class name, the integer id, and the ctype member
        # name.
        return f"Field({int(self)}, {self.ctype.name})"
