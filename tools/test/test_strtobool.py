# [MODULE DOC] Unit tests for tools.strtobool.strtobool covering accepted true and false
# spellings (case-insensitive) and rejection of other strings.
from __future__ import annotations

import unittest

from tools.strtobool import strtobool


class TestStrtobool(unittest.TestCase):
    # [CLASS DOC] Test case for strtobool; stateless.
    def test_truthy(self) -> None:
        # [FUNCTION DOC] Checks y, yes, t, true, on and 1 in several cases return exactly True.
        for val in (
            "y",
            "Y",
            "yes",
            "YES",
            "Yes",
            "t",
            "T",
            "true",
            "TRUE",
            "on",
            "ON",
            "1",
        ):
            self.assertIs(strtobool(val), True, msg=val)

    def test_falsy(self) -> None:
        # [FUNCTION DOC] Checks n, no, f, false, off and 0 in several cases return exactly
        # False.
        for val in (
            "n",
            "N",
            "no",
            "NO",
            "No",
            "f",
            "F",
            "false",
            "FALSE",
            "off",
            "OFF",
            "0",
        ):
            self.assertIs(strtobool(val), False, msg=val)

    def test_invalid_raises(self) -> None:
        # [FUNCTION DOC] Checks the empty string and unrecognized words raise a value error.
        for val in ("", "maybe", "2", "yeah", "nope"):
            with self.assertRaises(ValueError, msg=val):
                strtobool(val)


if __name__ == "__main__":
    unittest.main()
