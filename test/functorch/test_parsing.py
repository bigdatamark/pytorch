# Owner(s): ["module: functorch"]

# [MODULE DOC] Tests for the einops-style pattern parsing in functorch.einops._parsing:
# AnonymousAxis, ParsedExpression, parse_pattern and validate_rearrange_expressions. Adapted
# from the einops project test suite under the MIT license quoted below. Defines a small
# mocked equality helper and runs under the common test runner.
"""Adapted from https://github.com/arogozhnikov/einops/blob/230ac1526c1f42c9e1f7373912c7f8047496df11/tests/test_parsing.py.

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

from typing import Any
from unittest import mock

from functorch.einops._parsing import (
    _ellipsis,
    AnonymousAxis,
    parse_pattern,
    ParsedExpression,
    validate_rearrange_expressions,
)
from torch.testing._internal.common_utils import run_tests, TestCase


def mock_anonymous_axis_eq(self: AnonymousAxis, other: object) -> bool:
    # [FUNCTION DOC] HELPER: Replacement for the equality method of AnonymousAxis used
    # with a mock patch: returns True when the other object is also an AnonymousAxis with
    # an equal value, so equality is by value instead of identity.
    return isinstance(other, AnonymousAxis) and self.value == other.value


class TestAnonymousAxis(TestCase):
    # [CLASS DOC] Tests for AnonymousAxis, which by default compares by identity.
    def test_anonymous_axes(self) -> None:
        # [FUNCTION DOC] Asserts two anonymous axes with the same value are unequal by
        # default. Then, with equality patched to compare by value, checks axes with
        # matching values compare equal, different values unequal, and that list
        # comparison works with an integer mixed in.
        a, b = AnonymousAxis("2"), AnonymousAxis("2")
        self.assertNotEqual(a, b)

        with mock.patch.object(AnonymousAxis, "__eq__", mock_anonymous_axis_eq):
            c, d = AnonymousAxis("2"), AnonymousAxis("3")
            self.assertEqual(a, c)
            self.assertEqual(b, c)
            self.assertNotEqual(a, d)
            self.assertNotEqual(b, d)
            self.assertListEqual([a, 2, b], [c, 2, c])


class TestParsedExpression(TestCase):
    # [CLASS DOC] Tests for ParsedExpression, the parser of one side of
    # an einops-style pattern.
    def test_elementary_axis_name(self) -> None:
        # [FUNCTION DOC] Checks the static axis-name validity check: a list of
        # ordinary names (letters, digits after the first character, camel case, inner
        # underscores) is accepted, while empty, leading digit, all digits, leading or
        # trailing underscore, a single underscore, three dots and the ellipsis marker
        # constant are rejected.
        for name in [
            "a",
            "b",
            "h",
            "dx",
            "h1",
            "zz",
            "i9123",
            "somelongname",
            "Alex",
            "camelCase",
            "u_n_d_e_r_score",
            "unreasonablyLongAxisName",
        ]:
            self.assertTrue(ParsedExpression.check_axis_name(name))

        for name in [
            "",
            "2b",
            "12",
            "_startWithUnderscore",
            "endWithUnderscore_",
            "_",
            "...",
            _ellipsis,
        ]:
            self.assertFalse(ParsedExpression.check_axis_name(name))

    def test_invalid_expressions(self) -> None:
        # [FUNCTION DOC] Checks that ParsedExpression raises ValueError for repeated
        # ellipsis, ellipsis in nested or doubled parentheses, unbalanced or empty or
        # nested parentheses, and invalid identifiers (leading digit, leading
        # underscore, dots attached to a name), while valid neighbouring inputs parse
        # silently, including a non-ASCII letter.
        # double ellipsis should raise an error
        ParsedExpression("... a b c d")
        with self.assertRaises(ValueError):
            ParsedExpression("... a b c d ...")
        with self.assertRaises(ValueError):
            ParsedExpression("... a b c (d ...)")
        with self.assertRaises(ValueError):
            ParsedExpression("(... a) b c (d ...)")

        # double/missing/enclosed parenthesis
        ParsedExpression("(a) b c (d ...)")
        with self.assertRaises(ValueError):
            ParsedExpression("(a)) b c (d ...)")
        with self.assertRaises(ValueError):
            ParsedExpression("(a b c (d ...)")
        with self.assertRaises(ValueError):
            ParsedExpression("(a) (()) b c (d ...)")
        with self.assertRaises(ValueError):
            ParsedExpression("(a) ((b c) (d ...))")

        # invalid identifiers
        ParsedExpression("camelCase under_scored cApiTaLs \u00df ...")
        with self.assertRaises(ValueError):
            ParsedExpression("1a")
        with self.assertRaises(ValueError):
            ParsedExpression("_pre")
        with self.assertRaises(ValueError):
            ParsedExpression("...pre")
        with self.assertRaises(ValueError):
            ParsedExpression("pre...")

    @mock.patch.object(AnonymousAxis, "__eq__", mock_anonymous_axis_eq)
    def test_parse_expression(self, *mocks: mock.MagicMock) -> None:
        # [FUNCTION DOC] Checks the parse results with equality of anonymous axes
        # patched to compare by value (the patch passes no extra argument, so the
        # variadic mocks is empty). Cases cover named axes, empty parentheses, the
        # digit one dropped as unit axes, anonymous numeric axes, ellipsis outside and
        # inside parentheses. Verifies the identifier set, the composition list of
        # groups, and the flags for non-unitary anonymous axes, ellipsis and
        # parenthesized ellipsis.
        parsed = ParsedExpression("a1  b1   c1    d1")
        self.assertSetEqual(parsed.identifiers, {"a1", "b1", "c1", "d1"})
        self.assertListEqual(parsed.composition, [["a1"], ["b1"], ["c1"], ["d1"]])
        self.assertFalse(parsed.has_non_unitary_anonymous_axes)
        self.assertFalse(parsed.has_ellipsis)

        parsed = ParsedExpression("() () () ()")
        self.assertSetEqual(parsed.identifiers, set())
        self.assertListEqual(parsed.composition, [[], [], [], []])
        self.assertFalse(parsed.has_non_unitary_anonymous_axes)
        self.assertFalse(parsed.has_ellipsis)

        parsed = ParsedExpression("1 1 1 ()")
        self.assertSetEqual(parsed.identifiers, set())
        self.assertListEqual(parsed.composition, [[], [], [], []])
        self.assertFalse(parsed.has_non_unitary_anonymous_axes)
        self.assertFalse(parsed.has_ellipsis)

        parsed = ParsedExpression("5 (3 4)")
        self.assertEqual(len(parsed.identifiers), 3)
        self.assertSetEqual(
            {
                i.value if isinstance(i, AnonymousAxis) else i
                for i in parsed.identifiers
            },
            {3, 4, 5},
        )
        self.assertListEqual(
            parsed.composition,
            [[AnonymousAxis("5")], [AnonymousAxis("3"), AnonymousAxis("4")]],
        )
        self.assertTrue(parsed.has_non_unitary_anonymous_axes)
        self.assertFalse(parsed.has_ellipsis)

        parsed = ParsedExpression("5 1 (1 4) 1")
        self.assertEqual(len(parsed.identifiers), 2)
        self.assertSetEqual(
            {
                i.value if isinstance(i, AnonymousAxis) else i
                for i in parsed.identifiers
            },
            {4, 5},
        )
        self.assertListEqual(
            parsed.composition, [[AnonymousAxis("5")], [], [AnonymousAxis("4")], []]
        )

        parsed = ParsedExpression("name1 ... a1 12 (name2 14)")
        self.assertEqual(len(parsed.identifiers), 6)
        self.assertEqual(
            len(parsed.identifiers - {"name1", _ellipsis, "a1", "name2"}), 2
        )
        self.assertListEqual(
            parsed.composition,
            [
                ["name1"],
                _ellipsis,
                ["a1"],
                [AnonymousAxis("12")],
                ["name2", AnonymousAxis("14")],
            ],
        )
        self.assertTrue(parsed.has_non_unitary_anonymous_axes)
        self.assertTrue(parsed.has_ellipsis)
        self.assertFalse(parsed.has_ellipsis_parenthesized)

        parsed = ParsedExpression("(name1 ... a1 12) name2 14")
        self.assertEqual(len(parsed.identifiers), 6)
        self.assertEqual(
            len(parsed.identifiers - {"name1", _ellipsis, "a1", "name2"}), 2
        )
        self.assertListEqual(
            parsed.composition,
            [
                ["name1", _ellipsis, "a1", AnonymousAxis("12")],
                ["name2"],
                [AnonymousAxis("14")],
            ],
        )
        self.assertTrue(parsed.has_non_unitary_anonymous_axes)
        self.assertTrue(parsed.has_ellipsis)
        self.assertTrue(parsed.has_ellipsis_parenthesized)


class TestParsingUtils(TestCase):
    # [CLASS DOC] Tests for the pattern splitting function parse_pattern.
    def test_parse_pattern_number_of_arrows(self) -> None:
        # [FUNCTION DOC] Checks parse_pattern raises ValueError when the pattern has
        # more than one or no arrow separator and accepts exactly one.
        axes_lengths: dict[str, int] = {}

        too_many_arrows_pattern = "a -> b -> c -> d"
        with self.assertRaises(ValueError):
            parse_pattern(too_many_arrows_pattern, axes_lengths)

        too_few_arrows_pattern = "a"
        with self.assertRaises(ValueError):
            parse_pattern(too_few_arrows_pattern, axes_lengths)

        just_right_arrows = "a -> a"
        parse_pattern(just_right_arrows, axes_lengths)

    def test_ellipsis_invalid_identifier(self) -> None:
        # [FUNCTION DOC] Checks parse_pattern raises ValueError when the axes lengths
        # mapping contains the ellipsis marker as a key.
        axes_lengths: dict[str, int] = {"a": 1, _ellipsis: 2}
        pattern = f"a {_ellipsis} -> {_ellipsis} a"
        with self.assertRaises(ValueError):
            parse_pattern(pattern, axes_lengths)

    def test_ellipsis_matching(self) -> None:
        # [FUNCTION DOC] Checks that an ellipsis only on the right raises ValueError,
        # while an ellipsis only on the left, or on both sides, is accepted by
        # parse_pattern (the left-only case is rejected later by
        # the rearrange validation).
        axes_lengths: dict[str, int] = {}

        pattern = "a -> a ..."
        with self.assertRaises(ValueError):
            parse_pattern(pattern, axes_lengths)

        # raising an error on this pattern is handled by the rearrange expression validation
        pattern = "a ... -> a"
        parse_pattern(pattern, axes_lengths)

        pattern = "a ... -> ... a"
        parse_pattern(pattern, axes_lengths)

    def test_left_parenthesized_ellipsis(self) -> None:
        # [FUNCTION DOC] Checks that parse_pattern raises ValueError when the left
        # side holds a parenthesized ellipsis.
        axes_lengths: dict[str, int] = {}

        pattern = "(...) -> ..."
        with self.assertRaises(ValueError):
            parse_pattern(pattern, axes_lengths)


class MaliciousRepr:
    # [CLASS DOC] HELPER: Test object whose repr text looks like executable code; used to
    # verify that axis lengths are type checked and never evaluated.
    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a fixed string that looks like
        # a print statement.
        return "print('hello world!')"


class TestValidateRearrangeExpressions(TestCase):
    # [CLASS DOC] Tests for validate_rearrange_expressions, which checks parsed left and
    # right expressions and the axes lengths mapping for a rearrange operation.
    def test_validate_axes_lengths_are_integers(self) -> None:
        # [FUNCTION DOC] Checks validation passes with integer lengths and raises
        # TypeError when a length value is a non-integer object.
        axes_lengths: dict[str, Any] = {"a": 1, "b": 2, "c": 3}
        pattern = "a b c -> c b a"
        left, right = parse_pattern(pattern, axes_lengths)
        validate_rearrange_expressions(left, right, axes_lengths)

        axes_lengths = {"a": 1, "b": 2, "c": MaliciousRepr()}
        left, right = parse_pattern(pattern, axes_lengths)
        with self.assertRaises(TypeError):
            validate_rearrange_expressions(left, right, axes_lengths)

    def test_non_unitary_anonymous_axes_raises_error(self) -> None:
        # [FUNCTION DOC] Checks validation raises ValueError for a non-unit anonymous
        # axis on the left side and on the right side.
        axes_lengths: dict[str, int] = {}

        left_non_unitary_axis = "a 2 -> 1 1 a"
        left, right = parse_pattern(left_non_unitary_axis, axes_lengths)
        with self.assertRaises(ValueError):
            validate_rearrange_expressions(left, right, axes_lengths)

        right_non_unitary_axis = "1 1 a -> a 2"
        left, right = parse_pattern(right_non_unitary_axis, axes_lengths)
        with self.assertRaises(ValueError):
            validate_rearrange_expressions(left, right, axes_lengths)

    def test_identifier_mismatch(self) -> None:
        # [FUNCTION DOC] Checks validation raises ValueError when the identifiers on
        # the two sides differ, for an extra name on either side.
        axes_lengths: dict[str, int] = {}

        mismatched_identifiers = "a -> a b"
        left, right = parse_pattern(mismatched_identifiers, axes_lengths)
        with self.assertRaises(ValueError):
            validate_rearrange_expressions(left, right, axes_lengths)

        mismatched_identifiers = "a b -> a"
        left, right = parse_pattern(mismatched_identifiers, axes_lengths)
        with self.assertRaises(ValueError):
            validate_rearrange_expressions(left, right, axes_lengths)

    def test_unexpected_axes_lengths(self) -> None:
        # [FUNCTION DOC] Checks validation raises ValueError when the lengths mapping
        # names an axis that does not appear in the pattern.
        axes_lengths: dict[str, int] = {"c": 2}

        pattern = "a b -> b a"
        left, right = parse_pattern(pattern, axes_lengths)
        with self.assertRaises(ValueError):
            validate_rearrange_expressions(left, right, axes_lengths)


if __name__ == "__main__":
    run_tests()
