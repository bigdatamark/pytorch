# Owner(s): ["oncall: jit"]
# [MODULE DOC] TorchScript tests for percent-style string formatting in scripted functions. Each
# test either compares scripted and eager results via checkScript or scripts a function and
# checks the RuntimeError (with highlighted source) raised when it is called with a bad format.
# Importing the module appends the test directory to the system path.

import os
import sys
from typing import List

import torch


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestStringFormatting(JitTestCase):
    # [CLASS DOC] JitTestCase suite for percent formatting in TorchScript: integer remainder,
    # string, integer, exponent, float and character placeholders, escaped percent, tuples and
    # the error messages for malformed or mismatched formats.
    def test_modulo_operator(self):
        # [FUNCTION DOC] Checks that the integer remainder operator works in a scripted function
        # on two ints.
        def fn(dividend: int, divisor: int) -> int:
            # [FUNCTION DOC] HELPER: Returns the remainder of the dividend divided by the
            # divisor, both ints.
            return dividend % divisor

        self.checkScript(fn, (5, 2))

    def test_string_interpolation_with_string_placeholder_and_string_variable(self):
        # [FUNCTION DOC] Checks that a string placeholder fills in a string variable, compared
        # with eager mode.
        def fn(arg1: str):
            # [FUNCTION DOC] HELPER: Formats the string argument into a template with one string
            # placeholder.
            return "%s in template" % arg1

        self.checkScript(fn, ("foo",))

    def test_string_interpolation_with_string_placeholder_and_format_string_variable(
        self,
    ):
        # [FUNCTION DOC] Checks formatting where the format string is the variable and the value
        # is a constant string.
        def fn(arg1: str):
            # [FUNCTION DOC] HELPER: Applies the string argument as a format string to the
            # constant text foo.
            return arg1 % "foo"

        self.checkScript(fn, ("%s in template",))

    def test_string_interpolation_with_double_percent_in_string(self):
        # [FUNCTION DOC] Checks that a doubled percent sign in the template is emitted as a
        # literal percent.
        def fn(arg1: str):
            # [FUNCTION DOC] HELPER: Formats the argument into a template that ends with an
            # escaped percent sign.
            return "%s in template %%" % arg1

        self.checkScript(fn, ("foo",))

    def test_string_interpolation_with_percent_in_string(self):
        # [FUNCTION DOC] Checks that a template ending in a lone percent sign fails at run time
        # with a RuntimeError saying Incomplete format specifier, highlighting the formatting
        # expression.
        @torch.jit.script
        def fn(arg1: str) -> str:
            # [FUNCTION DOC] HELPER: Scripted at definition time; formats a string into a
            # template ending in a lone percent.
            return "%s in template %" % arg1  # noqa: F501

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Incomplete format specifier", '"%s in template %" % arg1'
        ):
            fn("foo")

    def test_string_interpolation_with_string_placeholder_and_digit_variable(self):
        # [FUNCTION DOC] Checks that a string placeholder accepts an int variable.
        def fn(arg1: int) -> str:
            # [FUNCTION DOC] HELPER: Formats an int argument into a template with one string
            # placeholder.
            return "%s in template" % arg1

        self.checkScript(fn, (1,))

    def test_string_interpolation_with_digit_placeholder_and_digit_variable(self):
        # [FUNCTION DOC] Checks that an integer placeholder accepts an int variable.
        def fn(arg1: int) -> str:
            # [FUNCTION DOC] HELPER: Formats an int argument into a template with one decimal
            # placeholder.
            return "%d in template" % arg1

        self.checkScript(fn, (1,))

    def test_string_interpolation_with_alternate_digit_placeholder(self):
        # [FUNCTION DOC] Checks that the alternate integer placeholder accepts an int variable.
        def fn(arg1: int) -> str:
            # [FUNCTION DOC] HELPER: Formats an int argument into a template with an alternate
            # integer placeholder.
            return "%i in template" % arg1

        self.checkScript(fn, (1,))

    def test_string_interpolation_with_digit_placeholder_and_string_variable(self):
        # [FUNCTION DOC] Checks that an integer placeholder given a string raises a RuntimeError
        # saying a number is required but a String was received, highlighting the formatting
        # expression.
        @torch.jit.script
        def fn(arg1: str) -> str:
            # [FUNCTION DOC] HELPER: Scripted at definition time; formats a string into a
            # template with a decimal placeholder.
            return "%d in template" % arg1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "%d requires a number for formatting, but got String",
            '"%d in template" % arg1',
        ):
            fn("1")

    def test_string_interpolation_with_exponent_placeholder_and_string_variable(self):
        # [FUNCTION DOC] Checks that an exponent placeholder given a string raises a
        # RuntimeError saying a number is required but a String was received, highlighting the
        # formatting expression.
        @torch.jit.script
        def fn(arg1: str) -> str:
            # [FUNCTION DOC] HELPER: Scripted at definition time; formats a string into a
            # template with an exponent placeholder.
            return "%e in template" % arg1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "%e requires a number for formatting, but got String",
            '"%e in template" % arg1',
        ):
            fn("1")

    def test_string_interpolation_with_lowercase_exponent_placeholder_and_digit_variable(
        self,
    ):
        # [FUNCTION DOC] Checks that a lowercase exponent placeholder accepts an int variable.
        def fn(arg1: int) -> str:
            # [FUNCTION DOC] HELPER: Formats an int argument into a template with a lowercase
            # exponent placeholder.
            return "%e in template" % arg1

        self.checkScript(fn, (1,))

    def test_string_interpolation_with_capital_exponent_placeholder_and_digit_variable(
        self,
    ):
        # [FUNCTION DOC] Checks that an uppercase exponent placeholder accepts an int variable.
        def fn(arg1: int) -> str:
            # [FUNCTION DOC] HELPER: Formats an int argument into a template with an uppercase
            # exponent placeholder.
            return "%E in template" % arg1

        self.checkScript(fn, (1,))

    def test_string_interpolation_with_float_placeholder_and_float_variable(self):
        # [FUNCTION DOC] Checks that a float placeholder accepts a float variable.
        def fn(arg1: float) -> str:
            # [FUNCTION DOC] HELPER: Formats a float argument into a template with one float
            # placeholder.
            return "%f in template" % arg1

        self.checkScript(fn, (1.0,))

    def test_string_interpolation_with_float_placeholder_and_digit_variable(self):
        # [FUNCTION DOC] Checks that a float placeholder accepts an int variable.
        def fn(arg1: int) -> str:
            # [FUNCTION DOC] HELPER: Formats an int argument into a template with one float
            # placeholder.
            return "%f in template" % arg1

        self.checkScript(fn, (1,))

    def test_string_interpolation_with_char_placeholder_and_char_variable(self):
        # [FUNCTION DOC] Checks that a character placeholder accepts a one-character string
        # variable.
        def fn(arg1: str) -> str:
            # [FUNCTION DOC] HELPER: Formats a string argument into a template with one
            # character placeholder.
            return "%c in template" % arg1

        self.checkScript(fn, ("a",))

    def test_string_interpolation_with_char_placeholder_and_digit_variable(self):
        # [FUNCTION DOC] Checks that a character placeholder accepts an int variable holding a
        # character code.
        def fn(arg1: int) -> str:
            # [FUNCTION DOC] HELPER: Formats an int argument into a template with one character
            # placeholder.
            return "%c in template" % arg1

        self.checkScript(fn, (97,))

    def test_string_interpolation_with_char_placeholder_and_true_string_variable(self):
        # [FUNCTION DOC] Checks that a character placeholder given a multi-character string
        # raises a RuntimeError saying an int or char is required but a String was received,
        # highlighting the expression.
        @torch.jit.script
        def fn(arg1: str) -> str:
            # [FUNCTION DOC] HELPER: Scripted at definition time; formats a string into a
            # template with a character placeholder.
            return "%c in template" % arg1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "%c requires an int or char for formatting, but got String",
            '"%c in template" % arg1',
        ):
            fn("foo")

    def test_string_interpolation_with_multiple_placeholders(self):
        # [FUNCTION DOC] Checks formatting with a tuple of a string, an int and a float for
        # three placeholders.
        def fn(arg1: str, arg2: int, arg3: float) -> str:
            # [FUNCTION DOC] HELPER: Formats a string, an int and a float into a template with
            # three matching placeholders.
            return "%s %d %f in template" % (arg1, arg2, arg3)

        self.checkScript(fn, ("foo", 1, 1))

    def test_string_interpolation_with_subscript(self):
        # [FUNCTION DOC] Checks that a list element can be the formatted value.
        def fn(arg1: List[str]) -> str:
            # [FUNCTION DOC] HELPER: Formats the first element of a string list into a template
            # with a string placeholder.
            return "%s in template" % arg1[0]

        self.checkScript(fn, (["foo", "bar"],))

    def test_string_interpolation_with_too_few_arguments(self):
        # [FUNCTION DOC] Checks that a template with two placeholders given one non-tuple
        # argument raises a RuntimeError saying there are too few arguments, highlighting the
        # expression.
        @torch.jit.script
        def fn(arg1: str) -> str:
            # [FUNCTION DOC] HELPER: Scripted at definition time; formats one string into a
            # template with two placeholders.
            return "%s %s in template" % arg1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "Too few arguments for format string",
            '"%s %s in template" % arg1',
        ):
            fn("foo")

    def test_string_interpolation_with_too_many_arguments(self):
        # [FUNCTION DOC] Checks that a template with one placeholder given a two-element tuple
        # raises a RuntimeError saying there are too many arguments, highlighting the
        # expression.
        @torch.jit.script
        def fn(arg1: str, arg2: str) -> str:
            # [FUNCTION DOC] HELPER: Scripted at definition time; formats two strings into a
            # template with one placeholder.
            return "%s in template" % (arg1, arg2)  # noqa: F507

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "Too many arguments for format string",
            '"%s in template" % (arg1, arg2',
        ):
            fn("foo", "bar")

    def test_string_interpolation_with_unknown_format_specifier(self):
        # [FUNCTION DOC] Checks that an unsupported specifier raises a RuntimeError saying that
        # specifier is not supported in TorchScript format strings, highlighting the expression.
        @torch.jit.script
        def fn(arg1: str) -> str:
            # [FUNCTION DOC] HELPER: Scripted at definition time; formats a string into a
            # template with an unsupported specifier.
            return "%a in template" % arg1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "The specifier %a is not supported in TorchScript format strings",
            '"%a in template" % arg1',
        ):
            fn("foo")


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
