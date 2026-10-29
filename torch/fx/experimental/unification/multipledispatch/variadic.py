# [MODULE DOC] Variadic signature support for multiple dispatch. Defines the
# metaclass-based types used to say "any number of arguments of these types" in
# a signature: VariadicSignatureType (the generated type), VariadicSignatureMeta
# (provides subscripting), Variadic (user-facing entry point), and isvariadic.
from __future__ import annotations

from .utils import typename


__all__ = ["VariadicSignatureType", "isvariadic", "VariadicSignatureMeta", "Variadic"]


class VariadicSignatureType(type):
    # [CLASS DOC] Metaclass of generated variadic types. Each instance carries a
    # variadic_type attribute (tuple of accepted element types). Customizes
    # subclass checks, equality and hashing so variadic types with the same set
    # of element types compare equal.
    # checking if subclass is a subclass of self
    def __subclasscheck__(cls, subclass: type) -> bool:
        # [FUNCTION DOC] True if subclass is cls itself, or every element type of
        # subclass (its variadic types if variadic, else the class itself) is a
        # subclass of one of the element types of cls.
        other_type = subclass.variadic_type if isvariadic(subclass) else (subclass,)
        return subclass is cls or all(
            issubclass(other, cls.variadic_type)  # type: ignore[attr-defined]
            for other in other_type
        )

    def __eq__(cls, other: object) -> bool:
        # [FUNCTION DOC] True if other is variadic and has the same set of
        # element types (order and duplicates ignored).
        """
        Return True if other has the same variadic type
        Parameters
        ----------
        other : object (type)
            The object (type) to check
        Returns
        -------
        bool
            Whether or not `other` is equal to `self`
        """
        return isvariadic(other) and set(cls.variadic_type) == set(other.variadic_type)  # type: ignore[attr-defined]

    def __hash__(cls) -> int:
        # [FUNCTION DOC] Hash of the metaclass and the frozen set of element
        # types, consistent with equality.
        return hash((type(cls), frozenset(cls.variadic_type)))  # type: ignore[attr-defined]


def isvariadic(obj: type) -> bool:
    # [FUNCTION DOC] True if obj is an instance of VariadicSignatureType.
    """Check whether the type `obj` is variadic.
    Parameters
    ----------
    obj : type
        The type to check
    Returns
    -------
    bool
        Whether or not `obj` is variadic
    Examples
    --------
    >>> # xdoctest: +SKIP
    >>> isvariadic(int)
    False
    >>> isvariadic(Variadic[int])
    True
    """
    return isinstance(obj, VariadicSignatureType)


class VariadicSignatureMeta(type):
    # [CLASS DOC] Metaclass of Variadic. Its subscript operation creates a new
    # VariadicSignatureType for a type or tuple of types.
    """A metaclass that overrides ``__getitem__`` on the class. This is used to
    generate a new type for Variadic signatures. See the Variadic class for
    examples of how this behaves.
    """

    def __getitem__(
        cls, variadic_type: type | tuple[type, ...]
    ) -> VariadicSignatureType:
        # [FUNCTION DOC] Build a new variadic type named after the element types;
        # a single type is wrapped into a one-element tuple and stored as the
        # variadic_type attribute. Intended to raise ValueError for non-types.
        # KNOWN ISSUE: the validity check ends with an always-truthy term (the
        # type of the argument), so the ValueError can never be raised.
        if not (isinstance(variadic_type, (type, tuple)) or type(variadic_type)):
            raise ValueError(
                "Variadic types must be type or tuple of types"
                " (Variadic[int] or Variadic[(int, float)]"
            )

        if not isinstance(variadic_type, tuple):
            variadic_type = (variadic_type,)
        return VariadicSignatureType(
            f"Variadic[{typename(variadic_type)}]",
            (),
            dict(variadic_type=variadic_type, __slots__=()),
        )


class Variadic(metaclass=VariadicSignatureMeta):
    # [CLASS DOC] Placeholder base used only through subscripting to declare
    # variadic signature entries (any number of arguments of the given types).
    # Holds no state of its own.
    """A class whose getitem method can be used to generate a new type
    representing a specific variadic signature.
    Examples
    --------
    >>> # xdoctest: +SKIP
    >>> Variadic[int]  # any number of int arguments
    <class 'multipledispatch.variadic.Variadic[int]'>
    >>> Variadic[(int, str)]  # any number of one of int or str arguments
    <class 'multipledispatch.variadic.Variadic[(int, str)]'>
    >>> issubclass(int, Variadic[int])
    True
    >>> issubclass(int, Variadic[(int, str)])
    True
    >>> issubclass(str, Variadic[(int, str)])
    True
    >>> issubclass(float, Variadic[(int, str)])
    False
    """
