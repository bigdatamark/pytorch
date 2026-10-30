# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests TorchScript support for Python string methods. A single test class compiles
# small functions with the JIT and compares scripted results against eager Python via the
# checkScript helper. Runs the test runner only on Python older than 3.14.

import sys
from test_jit import JitTestCase
from torch.testing._internal.common_utils import run_tests


class TestScript(JitTestCase):
    # [CLASS DOC] Test case (derived from the JIT test base class) holding the scripted string
    # tests: one big method for string methods and operators and one for string slicing.
    def test_str_ops(self):
        # [FUNCTION DOC] Scripts many small string functions and checks each against eager Python.
        # Covers character class predicates, case conversion, strip variants, center, ljust,
        # rjust, zfill, count, endswith, startswith, expandtabs, find family, replace, partition
        # family, split family, splitlines, comparisons, join and truthiness of strings. Loops
        # over a fixed list of sample strings and several widths. A two-character fill string is
        # expected to raise an error when called in eager mode.
        def test_str_is(s: str) -> tuple[bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool]:
            # [FUNCTION DOC] HELPER: Returns a tuple of eleven booleans from the is-upper, lower,
            # digit, space, alnum, alpha, decimal, numeric, identifier, title and printable
            # predicates of the input string.
            return s.isupper(), s.islower(), s.isdigit(), s.isspace(), \
                s.isalnum(), s.isalpha(), s.isdecimal(), s.isnumeric(), \
                s.isidentifier(), s.istitle(), s.isprintable()

        def test_str_to(s: str) -> tuple[str, str, str, str, str]:
            # [FUNCTION DOC] HELPER: Returns a tuple of five strings: upper, lower, capitalize,
            # title and swapcase versions of the input.
            return s.upper(), s.lower(), s.capitalize(), s.title(), s.swapcase()

        def test_str_strip(s: str) -> tuple[str, str, str]:
            # [FUNCTION DOC] HELPER: Returns left-stripped, right-stripped and fully stripped
            # versions of the input using default whitespace.
            return (
                s.lstrip(),
                s.rstrip(),
                s.strip(),
            )

        def test_str_strip_char_set(s: str, char_set: str) -> tuple[str, str, str]:
            # [FUNCTION DOC] HELPER: Returns left, right and both-side stripped versions of the
            # input, removing any characters found in the given character set string.
            return (
                s.lstrip(char_set),
                s.rstrip(char_set),
                s.strip(char_set),
            )

        inputs = ["", "12a", "!B", "12", "a", "B", "aB", "$12", "B12", "AB ",
                  "  \t", "  \n", "\na", "abc", "123.3", "s a", "b12a ",
                  "more strings with spaces", "Titular Strings", "\x0acan'tprintthis",
                  "spaces at the end ", " begin"]

        def test_str_center(i: int, s: str) -> str:
            # [FUNCTION DOC] HELPER: Centers the string in a field of the given width using the
            # default space fill.
            return s.center(i)

        def test_str_center_fc(i: int, s: str) -> str:
            # [FUNCTION DOC] HELPER: Centers the string in the given width using a star as the
            # fill character.
            return s.center(i, '*')

        def test_str_center_error(s: str) -> str:
            # [FUNCTION DOC] HELPER: Centers with a two-character fill string, which is invalid
            # and must raise an error; only used by the error check.
            return s.center(10, '**')

        def test_ljust(s: str, i: int) -> str:
            # [FUNCTION DOC] HELPER: Left-justifies the string to the given width with the default
            # space fill.
            return s.ljust(i)

        def test_ljust_fc(s: str, i: int, fc: str) -> str:
            # [FUNCTION DOC] HELPER: Left-justifies the string to the given width using the
            # supplied fill character.
            return s.ljust(i, fc)

        def test_ljust_fc_err(s: str) -> str:
            # [FUNCTION DOC] HELPER: Left-justifies with a two-character fill string, which is
            # invalid; defined but never called by the test.
            return s.ljust(10, '**')

        def test_rjust(s: str, i: int) -> str:
            # [FUNCTION DOC] HELPER: Right-justifies the string to the given width with the
            # default space fill.
            return s.rjust(i)

        def test_rjust_fc(s: str, i: int, fc: str) -> str:
            # [FUNCTION DOC] HELPER: Right-justifies the string to the given width using the
            # supplied fill character.
            return s.rjust(i, fc)

        def test_rjust_fc_err(s: str) -> str:
            # [FUNCTION DOC] HELPER: Right-justifies with a two-character fill string, which is
            # invalid; defined but never called by the test.
            return s.rjust(10, '**')

        def test_zfill(s: str, i: int) -> str:
            # [FUNCTION DOC] HELPER: Zero-fills the string on the left to the given width.
            return s.zfill(i)

        for input in inputs:
            self.checkScript(test_str_is, (input,))
            self.checkScript(test_str_to, (input,))
            self.checkScript(test_str_strip, (input,))
            for char_set in ["abc", "123", " ", "\t"]:
                self.checkScript(test_str_strip_char_set, (input, char_set))
            for i in range(7):
                self.checkScript(test_str_center, (i, input,))
                self.checkScript(test_str_center_fc, (i, input,))
                self.checkScript(test_ljust, (input, i))
                self.checkScript(test_ljust_fc, (input, i, '*'))
                self.checkScript(test_rjust, (input, i))
                self.checkScript(test_rjust_fc, (input, i, '*'))
                self.checkScript(test_zfill, (input, i))

        with self.assertRaises(Exception):
            test_str_center_error("error")
            test_ljust("error")

        def test_count() -> tuple[int, int, int, int, int, int, int, int, int, int, int, int]:
            # [FUNCTION DOC] HELPER: Returns a tuple of twelve counts of a single-character and a
            # multi-character substring in a literal word, with default, negative, out-of-range
            # and bounded start and end positions.
            return (
                "hello".count("h"),
                "hello".count("h", 0, 1),
                "hello".count("h", -3),
                "hello".count("h", -10, 1),
                "hello".count("h", 0, -10),
                "hello".count("h", 0, 10),
                "hello".count("ell"),
                "hello".count("ell", 0, 1),
                "hello".count("ell", -3),
                "hello".count("ell", -10, 1),
                "hello".count("ell", 0, -10),
                "hello".count("ell", 0, 10)
            )
        self.checkScript(test_count, ())

        def test_endswith() -> tuple[bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool]:
            # [FUNCTION DOC] HELPER: Returns fourteen booleans from checking suffix matches on a
            # literal word, for a two-character and a one-character suffix, with various start and
            # end positions including negative and out-of-range ones.
            return (
                "hello".endswith("lo"),
                "hello".endswith("lo", 0),
                "hello".endswith("lo", -2),
                "hello".endswith("lo", -8),
                "hello".endswith("lo", 0, -5),
                "hello".endswith("lo", -2, 3),
                "hello".endswith("lo", -8, 4),
                "hello".endswith("l"),
                "hello".endswith("l", 0),
                "hello".endswith("l", -2),
                "hello".endswith("l", -8),
                "hello".endswith("l", 0, -5),
                "hello".endswith("l", -2, 3),
                "hello".endswith("l", -8, 4)
            )
        self.checkScript(test_endswith, ())

        def test_startswith() -> tuple[bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool]:
            # [FUNCTION DOC] HELPER: Returns fourteen booleans from checking
            # prefix matches on a literal word, using the same patterns and start
            # and end positions as the suffix test.
            return (
                "hello".startswith("lo"),
                "hello".startswith("lo", 0),
                "hello".startswith("lo", -2),
                "hello".startswith("lo", -8),
                "hello".startswith("lo", 0, -5),
                "hello".startswith("lo", -2, 3),
                "hello".startswith("lo", -8, 4),
                "hello".startswith("l"),
                "hello".startswith("l", 0),
                "hello".startswith("l", -2),
                "hello".startswith("l", -8),
                "hello".startswith("l", 0, -5),
                "hello".startswith("l", -2, 3),
                "hello".startswith("l", -8, 4)
            )
        self.checkScript(test_startswith, ())

        def test_expandtabs() -> tuple[str, str, str, str, str, str]:
            # [FUNCTION DOC] HELPER: Returns six strings produced by expanding tab characters with
            # default, positive and negative tab sizes.
            return (
                'xyz\t82345\tabc'.expandtabs(),
                'xyz\t32345\tabc'.expandtabs(3),
                'xyz\t52345\tabc'.expandtabs(5),
                'xyz\t62345\tabc'.expandtabs(6),
                'xyz\t72345\tabc'.expandtabs(7),
                'xyz\t62345\tabc'.expandtabs(-5),
            )
        self.checkScript(test_expandtabs, ())

        def test_rfind() -> tuple[int, int, int, int, int, int, int, int, int]:
            # [FUNCTION DOC] HELPER: Returns nine integers from searching a literal string from
            # the right for substrings with optional start and end bounds, including negative and
            # reversed bounds.
            return (
                "hello123abc".rfind("llo"),
                "hello123abc".rfind("12"),
                "hello123abc".rfind("ab"),
                "hello123abc".rfind("ll", -1),
                "hello123abc".rfind("12", 4),
                "hello123abc".rfind("ab", -7),
                "hello123abc".rfind("ll", -1, 8),
                "hello123abc".rfind("12", 4, -4),
                "hello123abc".rfind("ab", -7, -20),
            )
        self.checkScript(test_rfind, ())

        def test_find() -> tuple[int, int, int, int, int, int, int, int, int]:
            # [FUNCTION DOC] HELPER: Returns nine integers from searching a literal string from
            # the left for substrings, using the same bounds as the right-search test.
            return (
                "hello123abc".find("llo"),
                "hello123abc".find("12"),
                "hello123abc".find("ab"),
                "hello123abc".find("ll", -1),
                "hello123abc".find("12", 4),
                "hello123abc".find("ab", -7),
                "hello123abc".find("ll", -1, 8),
                "hello123abc".find("12", 4, -4),
                "hello123abc".find("ab", -7, -20),
            )
        self.checkScript(test_find, ())

        def test_index() -> tuple[int, int, int, int, int, int]:
            # [FUNCTION DOC] HELPER: Returns six integers from the index method (left search,
            # raising if absent) with various bounds; all chosen substrings are present.
            return (
                "hello123abc".index("llo"),
                "hello123abc".index("12"),
                "hello123abc".index("ab"),
                "hello123abc".index("12", 4),
                "hello123abc".index("ab", -7),
                "hello123abc".index("12", 4, -4),
            )
        self.checkScript(test_index, ())

        def test_rindex() -> tuple[int, int, int, int, int, int]:
            # [FUNCTION DOC] HELPER: Returns six integers from the rindex method (right search,
            # raising if absent) with various bounds; all chosen substrings are present.
            return (
                "hello123abc".rindex("llo"),
                "hello123abc".rindex("12"),
                "hello123abc".rindex("ab"),
                "hello123abc".rindex("12", 4),
                "hello123abc".rindex("ab", -7),
                "hello123abc".rindex("12", 4, -4),
            )
        self.checkScript(test_rindex, ())

        def test_replace() -> tuple[str, str, str, str, str, str, str]:
            # [FUNCTION DOC] HELPER: Returns seven strings from replace calls with and without a
            # maximum count, including zero matches and a negative count.
            return (
                "hello123abc".replace("llo", "sdf"),
                "ff".replace("f", "ff"),
                "abc123".replace("a", "testing"),
                "aaaaaa".replace("a", "testing", 3),
                "bbb".replace("a", "testing", 3),
                "ccc".replace("c", "ccc", 3),
                "cc".replace("c", "ccc", -3),
            )
        self.checkScript(test_replace, ())

        def test_partition() -> tuple[tuple[str, str, str], tuple[str, str, str], tuple[str, str, str],
                                      tuple[str, str, str], tuple[str, str, str], tuple[str, str, str],
                                      tuple[str, str, str]]:
            # [FUNCTION DOC] HELPER: Returns a tuple of seven partition results
            # (head, separator, tail triples) for several strings, including the
            # case where the separator is missing.
            return (
                "hello123abc".partition("llo"),
                "ff".partition("f"),
                "abc123".partition("a"),
                "aaaaaa".partition("testing"),
                "bbb".partition("a"),
                "ccc".partition("ccc"),
                "cc".partition("ccc"),
            )
        self.checkScript(test_partition, ())

        def test_rpartition() -> tuple[tuple[str, str, str], tuple[str, str, str], tuple[str, str, str],
                                       tuple[str, str, str], tuple[str, str, str], tuple[str, str, str],
                                       tuple[str, str, str]]:
            # [FUNCTION DOC] HELPER: Returns a tuple of seven right-partition results for the same
            # inputs as the partition test.
            return (
                "hello123abc".rpartition("llo"),
                "ff".rpartition("f"),
                "abc123".rpartition("a"),
                "aaaaaa".rpartition("testing"),
                "bbb".rpartition("a"),
                "ccc".rpartition("ccc"),
                "cc".rpartition("ccc"),
            )
        self.checkScript(test_rpartition, ())

        def test_split() -> tuple[list[str], list[str], list[str], list[str], list[str],
                                  list[str], list[str], list[str], list[str], list[str], list[str]]:
            # [FUNCTION DOC] HELPER: Returns eleven lists of strings.
            # KNOWN ISSUE: the body returns literal lists and never calls split, so it only checks
            # that list results round trip through scripting, not split behavior.
            return (
                ["a", "a", "a", "a", "a"],
                ["a", "a", "a", "a", "a"],
                ["a", "a", "a", "a", "a"],
                ["", "a", "a", "a", "a", "a", ""],
                ["a", "a", "a", "a", "a", ""],
                ["a", "a", "a", "a", "a", ""],
                ["a", "a", "a", "a a "],
                [" a a a a a "],
                [" a", "a a", "a a"],
                [" a", "a a", "a a "],
                [" ", "a ", "a a "],
            )
        self.checkScript(test_split, ())

        # test raising error for empty separator
        def test_split_empty_separator():
            # [FUNCTION DOC] HELPER: Splits a string with an empty separator, which must raise an
            # error whose message mentions an empty separator.
            s = "test"
            return s.split("")

        self.checkScriptRaisesRegex(test_split_empty_separator, (), Exception,
                                    "empty separator")

        def test_rsplit() -> tuple[list[str], list[str], list[str], list[str], list[str],
                                   list[str], list[str], list[str], list[str]]:
            # [FUNCTION DOC] HELPER: Returns nine lists of strings.
            # KNOWN ISSUE: the body returns literal lists and never calls rsplit, so it does not
            # exercise right split behavior.
            return (
                ["a", "a", "a", "a", "a"],
                ["", "a", "a", "a", "a", "a", ""],
                ["a", "a", "a", "a", "a", ""],
                ["a", "a", "a", "a", "a", ""],
                ["a a a", "a", "a", ""],
                [" a a a a a "],
                [" a", "a a", "a a "],
                [" a", "a a", "a a "],
                [" ", "a ", "a a"],
            )
        self.checkScript(test_rsplit, ())

        def test_splitlines() -> tuple[list[str], list[str], list[str], list[str],
                                       list[str], list[str]]:
            # [FUNCTION DOC] HELPER: Returns six lists from splitting literal strings into lines,
            # using newline, vertical tab and form feed boundaries and trailing newlines.
            return (
                "hello\ntest".splitlines(),
                "hello\n\ntest\n".splitlines(),
                "hello\ntest\n\n".splitlines(),
                "hello\vtest".splitlines(),
                "hello\v\f\ntest".splitlines(),
                "hello\ftest".splitlines(),
            )
        self.checkScript(test_splitlines, ())

        def test_str_cmp(a: str, b: str) -> tuple[bool, bool, bool, bool, bool, bool]:
            # [FUNCTION DOC] HELPER: Returns six booleans comparing two strings with not-equal,
            # equal, less, greater, less-equal and greater-equal.
            return a != b, a == b, a < b, a > b, a <= b, a >= b

        for i in range(len(inputs) - 1):
            self.checkScript(test_str_cmp, (inputs[i], inputs[i + 1]))

        def test_str_join():
            # [FUNCTION DOC] HELPER: Returns six strings produced by joining lists of strings with
            # various separators, including an empty separator.
            return (
                ",".join(["a"]),
                ",".join(["a", "b", "c"]),
                ",".join(["aa", "bb", "cc"]),
                ",".join(["a,a", "bb", "c,c"]),
                "**a**".join(["b", "c", "d", "e"]),
                "".join(["a", "b", "c"]),
            )
        self.checkScript(test_str_join, ())

        def test_bool_conversion(a: str):
            # [FUNCTION DOC] HELPER: Returns the input string if it is non-empty, otherwise the
            # literal default string; tests string truthiness.
            if a:
                return a
            else:
                return "default"

        self.checkScript(test_bool_conversion, ("nonempty",))
        self.checkScript(test_bool_conversion, ("",))

    def test_string_slice(self):
        # [FUNCTION DOC] Scripts a function slicing a string with several start, stop and step
        # combinations (including negative start) and checks it against eager Python.
        def test_slice(a: str) -> tuple[str, str, str, str, str]:
            # [FUNCTION DOC] HELPER: Returns a tuple of five slices of the input string with
            # different start, stop and step values.
            return (
                a[0:1:2],
                a[0:6:1],
                a[4:1:2],
                a[0:3:2],
                a[-1:1:3],
            )

        self.checkScript(test_slice, ("hellotest",))

if __name__ == '__main__':
    if sys.version_info < (3, 14):
        run_tests()
