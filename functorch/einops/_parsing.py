# [MODULE DOC]
# Parser for einops-style rearrange patterns (adapted from the einops project). Provides
# AnonymousAxis, ParsedExpression (one side of a pattern), parse_pattern for a full pattern,
# validation specific to rearrange, and comma_separate for building first-class dim lists.
# No import-time side effects beyond defining the ellipsis marker constant.
"""Adapted from https://github.com/arogozhnikov/einops/blob/36c7bb16e57d6e57f8f3050f9e07abdf3f00469f/einops/parsing.py.

MIT License

Copyright (c) 2018 Alex Rogozhnikov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import keyword
import warnings
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Collection, Mapping


_ellipsis: str = "\u2026"  # NB, this is a single unicode symbol. String is used as it is not a list, but can be iterated


class AnonymousAxis:
    # [CLASS DOC] Represents a numeric axis length (at least 1) with no name inside a pattern.
    # State: value, the integer length. Instances compare by identity, not by value.
    """Used by `ParsedExpression` to represent an axis with a size (> 1), but no associated identifier.

    Note: Different instances of this class are not equal to each other, even if they have the same value.
    """

    def __init__(self, value: str) -> None:
        # [FUNCTION DOC] Converts the text value to an integer and stores it. Raises
        # ValueError if the integer is below 1 (or if the text is not numeric).
        self.value = int(value)
        if self.value < 1:
            raise ValueError(
                f"Anonymous axis should have positive length, not {self.value}"
            )

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns the length followed by the word axis, joined with a hyphen.
        return f"{self.value}-axis"


class ParsedExpression:
    # [CLASS DOC] Parsed form of one side of a pattern. Attributes: has_ellipsis,
    # has_ellipsis_parenthesized (None if no ellipsis), identifiers (set of axis names, anonymous
    # axes and the ellipsis marker), has_non_unitary_anonymous_axes, and composition (list with
    # one entry per output axis: a list of identifiers for a named or grouped axis, an empty
    # list for a length-1 axis, or the ellipsis marker). Built entirely in the constructor.
    """Structure containing information about one side of an `einops`-style pattern (e.g. 'b c (h w)')."""

    def __init__(
        self,
        expression: str,
        *,
        allow_underscore: bool = False,
        allow_duplicates: bool = False,
    ) -> None:
        # [FUNCTION DOC] Tokenizes the expression character by character into axis names,
        # numbers, one-level parenthesized groups and a single ellipsis (three dots become one
        # marker character), filling the attributes. Raises ValueError for stray dots,
        # repeated ellipsis, duplicate names (unless allowed), invalid names, nested or
        # unbalanced parentheses, and unknown characters.
        """Parse the expression and store relevant metadata.

        Args:
            expression (str): the `einops`-pattern to parse
            allow_underscore (bool): whether to allow axis identifier names to begin with an underscore
            allow_duplicates (bool): whether to allow an identifier to appear more than once in the expression
        """
        self.has_ellipsis: bool = False
        self.has_ellipsis_parenthesized: bool | None = None
        self.identifiers: set[str | AnonymousAxis] = set()
        # that's axes like 2, 3, 4 or 5. Axes with size 1 are exceptional and replaced with empty composition
        self.has_non_unitary_anonymous_axes: bool = False
        # composition keeps structure of composite axes, see how different corner cases are handled in tests
        self.composition: list[list[str | AnonymousAxis] | str] = []
        if "." in expression:
            if "..." not in expression:
                raise ValueError(
                    "Expression may contain dots only inside ellipsis (...)"
                )
            if str.count(expression, "...") != 1 or str.count(expression, ".") != 3:
                raise ValueError(
                    "Expression may contain dots only inside ellipsis (...); only one ellipsis for tensor "
                )
            expression = expression.replace("...", _ellipsis)
            self.has_ellipsis = True

        bracket_group: list[str | AnonymousAxis] | None = None

        def add_axis_name(x: str) -> None:
            # [FUNCTION DOC] HELPER: Registers one finished token: checks for duplicates, records
            # the ellipsis, ignores or records a length-1 axis as an empty group, validates names,
            # creates an anonymous axis for numbers, and appends to the current group or to the
            # composition. Raises ValueError on duplicates or invalid identifiers.
            if x in self.identifiers:
                if not (allow_underscore and x == "_") and not allow_duplicates:
                    raise ValueError(
                        f"Indexing expression contains duplicate dimension '{x}'"
                    )
            if x == _ellipsis:
                self.identifiers.add(_ellipsis)
                if bracket_group is None:
                    self.composition.append(_ellipsis)
                    self.has_ellipsis_parenthesized = False
                else:
                    bracket_group.append(_ellipsis)
                    self.has_ellipsis_parenthesized = True
            else:
                is_number = str.isdecimal(x)
                if is_number and int(x) == 1:
                    # handling the case of anonymous axis of length 1
                    if bracket_group is None:
                        self.composition.append([])
                    else:
                        pass  # no need to think about 1s inside parenthesis
                    return
                is_axis_name, reason = self.check_axis_name_return_reason(
                    x, allow_underscore=allow_underscore
                )
                if not (is_number or is_axis_name):
                    raise ValueError(f"Invalid axis identifier: {x}\n{reason}")
                axis_name: str | AnonymousAxis = AnonymousAxis(x) if is_number else x
                self.identifiers.add(axis_name)
                if is_number:
                    self.has_non_unitary_anonymous_axes = True
                if bracket_group is None:
                    self.composition.append([axis_name])
                else:
                    bracket_group.append(axis_name)

        current_identifier = None
        for char in expression:
            if char in "() ":
                if current_identifier is not None:
                    add_axis_name(current_identifier)
                current_identifier = None
                if char == "(":
                    if bracket_group is not None:
                        raise ValueError(
                            "Axis composition is one-level (brackets inside brackets not allowed)"
                        )
                    bracket_group = []
                elif char == ")":
                    if bracket_group is None:
                        raise ValueError("Brackets are not balanced")
                    self.composition.append(bracket_group)
                    bracket_group = None
            elif str.isalnum(char) or char in ["_", _ellipsis]:
                if current_identifier is None:
                    current_identifier = char
                else:
                    current_identifier += char
            else:
                raise ValueError(f"Unknown character '{char}'")

        if bracket_group is not None:
            raise ValueError(f"Imbalanced parentheses in expression: '{expression}'")
        if current_identifier is not None:
            add_axis_name(current_identifier)

    @staticmethod
    def check_axis_name_return_reason(
        name: str, allow_underscore: bool = False
    ) -> tuple[bool, str]:
        # [FUNCTION DOC] Returns a pair (valid, reason). Invalid if not an identifier or if it
        # starts or ends with an underscore (a lone underscore is valid when allowed). Valid
        # names that are keywords emit a RuntimeWarning, and the name axis emits a
        # FutureWarning (side effect).
        """Check if the given axis name is valid, and a message explaining why if not.

        Valid axes names are python identifiers except keywords, and should not start or end with an underscore.

        Args:
            name (str): the axis name to check
            allow_underscore (bool): whether axis names are allowed to start with an underscore

        Returns:
            tuple[bool, str]: whether the axis name is valid, a message explaining why if not
        """
        if not str.isidentifier(name):
            return False, "not a valid python identifier"
        elif name[0] == "_" or name[-1] == "_":
            if name == "_" and allow_underscore:
                return True, ""
            return False, "axis name should should not start or end with underscore"
        else:
            if keyword.iskeyword(name):
                warnings.warn(
                    f"It is discouraged to use axes names that are keywords: {name}",
                    RuntimeWarning,
                )
            if name == "axis":
                warnings.warn(
                    "It is discouraged to use 'axis' as an axis name and will raise an error in future",
                    FutureWarning,
                )
            return True, ""

    @staticmethod
    def check_axis_name(name: str) -> bool:
        # [FUNCTION DOC] Returns only the validity flag from the detailed name check (with
        # default options); may emit the same warnings.
        """Check if the name is a valid axis name.

        Args:
            name (str): the axis name to check

        Returns:
            bool: whether the axis name is valid
        """
        is_valid, _ = ParsedExpression.check_axis_name_return_reason(name)
        return is_valid


def parse_pattern(
    pattern: str, axes_lengths: Mapping[str, int]
) -> tuple[ParsedExpression, ParsedExpression]:
    # [FUNCTION DOC] Splits the pattern on the arrow into left and right text, parses each into
    # a ParsedExpression and returns both. Raises ValueError if the arrow is not present
    # exactly once, the ellipsis marker is a key of axes_lengths, the right side has an
    # ellipsis but the left does not, or the left ellipsis is inside parentheses.
    """Parse an `einops`-style pattern into a left-hand side and right-hand side `ParsedExpression` object.

    Args:
        pattern (str): the `einops`-style rearrangement pattern
        axes_lengths (Mapping[str, int]): any additional length specifications for dimensions

    Returns:
       tuple[ParsedExpression, ParsedExpression]: a tuple containing the left-hand side and right-hand side expressions
    """
    # adapted from einops.einops._prepare_transformation_recipe
    # https://github.com/arogozhnikov/einops/blob/230ac1526c1f42c9e1f7373912c7f8047496df11/einops/einops.py
    try:
        left_str, right_str = pattern.split("->")
    except ValueError:
        raise ValueError("Pattern must contain a single '->' separator") from None

    if _ellipsis in axes_lengths:
        raise ValueError(f"'{_ellipsis}' is not an allowed axis identifier")

    left = ParsedExpression(left_str)
    right = ParsedExpression(right_str)

    if not left.has_ellipsis and right.has_ellipsis:
        raise ValueError(
            f"Ellipsis found in right side, but not left side of a pattern {pattern}"
        )
    if left.has_ellipsis and left.has_ellipsis_parenthesized:
        raise ValueError(
            f"Ellipsis is parenthesis in the left side is not allowed: {pattern}"
        )

    return left, right


def validate_rearrange_expressions(
    left: ParsedExpression, right: ParsedExpression, axes_lengths: Mapping[str, int]
) -> None:
    # [FUNCTION DOC] Checks rearrange-specific rules. Raises TypeError if any given length is
    # not exactly an int; ValueError if either side has numbers other than 1 as axes, if
    # identifiers are present on only one side, or if a given length names an axis missing
    # from the left side.
    """Perform expression validations that are specific to the `rearrange` operation.

    Args:
        left (ParsedExpression): left-hand side expression
        right (ParsedExpression): right-hand side expression
        axes_lengths (Mapping[str, int]): any additional length specifications for dimensions
    """
    for length in axes_lengths.values():
        if (length_type := type(length)) is not int:
            raise TypeError(
                f"rearrange axis lengths must be integers, got: {length_type}"
            )

    if left.has_non_unitary_anonymous_axes or right.has_non_unitary_anonymous_axes:
        raise ValueError("rearrange only supports unnamed axes of size 1")

    difference = set.symmetric_difference(left.identifiers, right.identifiers)
    if len(difference) > 0:
        raise ValueError(
            f"Identifiers only on one side of rearrange expression (should be on both): {difference}"
        )

    unmatched_axes = axes_lengths.keys() - left.identifiers
    if len(unmatched_axes) > 0:
        raise ValueError(
            f"Identifiers not found in rearrange expression: {unmatched_axes}"
        )


def comma_separate(collection: Collection[str | Collection[str]]) -> str:
    # [FUNCTION DOC] Joins items with comma and space. Strings are used as is; nested
    # collections become a parenthesized recursive join, with a trailing comma when the
    # nested collection has exactly one item. Returns the string.
    """Convert a collection of strings representing first class dims into a comma-separated string.

    Args:
        collection (Collection[Union[str, Collection[str]]]): the collection of strings to convert

    Returns:
        str: the comma-separated string

    Examples:
        >>> comma_separate(("d0",))
        'd0'

        >>> comma_separate(("d0", "d1", "d2", "d3"))
        'd0, d1, d2, d3'

        >>> comma_separate([("d1", "d4")])
        '(d1, d4)'

        >>> comma_separate([("d0",), (), ("d1",), ("d2",), ("d3", "d4")])
        '(d0,), (), (d1,), (d2,), (d3, d4)'
    """
    return ", ".join(
        item
        if isinstance(item, str)
        else f"({comma_separate(item)}{',' if len(item) == 1 else ''})"
        for item in collection
    )
