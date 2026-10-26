# [MODULE DOC] Support for tagged-union ("one field set at a time") schema classes used by export serde.
# Provides _Union (base class), _union_dataclass (decorator for subclasses) and _UnionTag
# (a str subclass that validates comparisons against the union's field names).
# No import-time side effects besides defining a TypeVar bound to _Union.
import functools
from collections.abc import Hashable
from dataclasses import dataclass, fields
from typing import Any, TypeVar
from typing_extensions import dataclass_transform


T = TypeVar("T", bound="_Union")


class _UnionTag(str):
    # [CLASS DOC] HELPER: String naming the active field of a union instance. Remembers the
    # union class (_cls) it belongs to so equality checks can reject names that are not
    # fields of that class. Hashes like a plain string.
    __slots__ = ("_cls",)
    _cls: Hashable

    @staticmethod
    def create(t: str, cls: type["_Union"]) -> "_UnionTag":
        # [FUNCTION DOC] HELPER: Build a tag string t bound to union class cls and return it.
        # Raises AssertionError if the new tag somehow already carries a class attribute.
        tag = _UnionTag(t)
        if hasattr(tag, "_cls"):
            raise AssertionError("tag already has _cls attribute")
        tag._cls = cls
        return tag

    def __eq__(self, cmp: object) -> bool:
        # [FUNCTION DOC] HELPER: Compare the tag with a string. Raises AssertionError if cmp is
        # not a string or is not a valid field name of the bound union class (so typos in
        # comparisons are caught); otherwise returns whether the strings are equal.
        if not isinstance(cmp, str):
            raise AssertionError(f"expected str, got {type(cmp)}")
        other = str(cmp)
        if other not in _get_field_names(self._cls):
            raise AssertionError(
                f"{other} is not a valid tag for {self._cls}. Available tags: {_get_field_names(self._cls)}"
            )
        return str(self) == other

    def __hash__(self) -> int:
        # [FUNCTION DOC] HELPER: Hash of the underlying string (needed because __eq__ is overridden).
        return hash(str(self))


@functools.cache
def _get_field_names(cls: type["_Union"]) -> set[str]:
    # [FUNCTION DOC] HELPER: Return the set of dataclass field names of a union class. Cached per class.
    # pyrefly: ignore[bad-argument-type]  # TODO _Union subclasses are dataclasses at runtime via _union_dataclass
    return {f.name for f in fields(cls)}


# If you turn a schema class that inherits from union into a dataclass, please use
# this decorator to configure it. It's safe, faster and allows code sharing.
#
# For example, _union_dataclass customizes the __eq__ method to only check the type
# and value property instead of default implementation of dataclass which goes
# through every field in the dataclass.
@dataclass_transform(eq_default=False)
def _union_dataclass(cls: type[T]) -> type[T]:
    # [FUNCTION DOC] HELPER: Class decorator that turns a _Union subclass into a dataclass without
    # generated repr or eq (the _Union versions are used instead). Raises AssertionError if
    # cls does not inherit from _Union. Returns the decorated class.
    if not issubclass(cls, _Union):
        raise AssertionError(f"{cls} must inherit from {_Union}.")
    return dataclass(repr=False, eq=False)(cls)


class _Union:
    # [CLASS DOC] Base class for tagged unions: a dataclass whose fields are alternatives and
    # exactly one is set. State: the field values plus _type, the tag naming the active field.
    # Instances must be built with create. Reading an unset (None) field that is not the
    # active one raises AttributeError. Equality compares active tag and value only.
    _type: _UnionTag

    @classmethod
    def create(cls: type[T], **kwargs: object) -> T:
        # [FUNCTION DOC] Construct a union instance with exactly one field set from keyword arguments
        # (all other fields become None) and record that field as the active tag.
        # Raises AssertionError if the number of keyword arguments is not one.
        if len(kwargs) != 1:
            raise AssertionError(f"expected exactly 1 kwarg, got {len(kwargs)}")
        obj = cls(**{**{f.name: None for f in fields(cls)}, **kwargs})  # type: ignore[arg-type]
        obj._type = _UnionTag.create(next(iter(kwargs.keys())), cls)
        return obj

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Dataclass hook that validates the schema: raises AssertionError if any field is
        # named type, _type, create or value, since those names are reserved by this class.
        if any(
            f.name in ("type", "_type", "create", "value")
            for f in fields(self)  # type: ignore[arg-type, misc]
        ):
            raise AssertionError(
                "field names 'type', '_type', 'create', 'value' are reserved"
            )

    @property
    def type(self) -> str:
        # [FUNCTION DOC] Return the active tag (name of the set field). Raises RuntimeError if the
        # instance was not built through create and so has no tag.
        try:
            return self._type
        except AttributeError as e:
            raise RuntimeError(
                f"Please use {type(self).__name__}.create to instantiate the union type."
            ) from e

    @property
    def value(self) -> Any:
        # [FUNCTION DOC] Return the value stored in the active field.
        return getattr(self, self.type)

    def __getattribute__(self, name: str) -> Any:
        # [FUNCTION DOC] Attribute access hook: returns the attribute normally, but raises AttributeError
        # when a union field that is None (unset) other than the active one is read.
        attr = super().__getattribute__(name)
        if attr is None and name in _get_field_names(type(self)) and name != self.type:
            raise AttributeError(f"Field {name} is not set.")
        return attr

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is also a _Union with the same active tag and equal value;
        # False for non-union objects. Note the class itself is not compared.
        if not isinstance(other, _Union):
            return False
        return self.type == other.type and self.value == other.value

    def __str__(self) -> str:
        # [FUNCTION DOC] String form, identical to the repr.
        return self.__repr__()

    def __repr__(self) -> str:
        # [FUNCTION DOC] Render as ClassName(tag=value) showing only the active field.
        return f"{type(self).__name__}({self.type}={getattr(self, self.type)})"
