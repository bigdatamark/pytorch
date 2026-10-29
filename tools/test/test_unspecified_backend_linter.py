# mypy: ignore-errors
# [MODULE DOC] Tests for the unspecified-backend linter, which flags torch.compile uses (calls
# or decorators) that do not pass an explicit backend. Each test writes a source file into a
# temporary directory and inspects the lint messages from check_file.
import tempfile
import unittest
from pathlib import Path

from tools.linter.adapters.unspecified_backend_linter import (
    check_file,
    LINTER_CODE,
    LintSeverity,
)


class TestUnspecifiedBackendLinter(unittest.TestCase):
    # [CLASS DOC] Test case for the linter. State: tmpdir (temporary directory, cleaned up
    # after each test) and path (a callsite.py file inside it that tests overwrite).
    """Test the torch.compile explicit-backend linter."""

    def setUp(self):
        # [FUNCTION DOC] Creates the temporary directory, registers its cleanup, and sets the
        # path of the file that sources are written to.
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.path = Path(self.tmpdir.name) / "callsite.py"

    def _check(self, source: str):
        # [FUNCTION DOC] HELPER: Writes the source text to the test file and returns the list
        # of lint messages the linter reports for it.
        self.path.write_text(source)
        return check_file(str(self.path))

    def test_bare_call_without_backend_is_flagged(self):
        # [FUNCTION DOC] Checks a compile call with no backend yields one error message with
        # the implicit-inductor-backend name at line 4, column 5.
        messages = self._check(
            "import torch\ndef f(x):\n    return x\ng = torch.compile(f)\n"
        )
        self.assertEqual(len(messages), 1)
        msg = messages[0]
        self.assertEqual(msg.path, str(self.path))
        self.assertEqual(msg.line, 4)
        self.assertEqual(msg.char, 5)
        self.assertEqual(msg.code, LINTER_CODE)
        self.assertEqual(msg.severity, LintSeverity.ERROR)
        self.assertEqual(msg.name, "implicit-inductor-backend")

    def test_call_with_backend_is_ok(self):
        # [FUNCTION DOC] Checks a compile call passing a backend keyword yields no messages.
        messages = self._check(
            "import torch\n"
            "def f(x):\n"
            "    return x\n"
            'g = torch.compile(f, backend="eager")\n'
        )
        self.assertEqual(messages, [])

    def test_bare_decorator_is_flagged(self):
        # [FUNCTION DOC] Checks the compile decorator used without a call is flagged once, at
        # the decorator line.
        messages = self._check(
            "import torch\n@torch.compile\ndef f(x):\n    return x\n"
        )
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].line, 2)
        self.assertEqual(messages[0].name, "implicit-inductor-backend")

    def test_call_decorator_without_backend_is_flagged(self):
        # [FUNCTION DOC] Checks the compile decorator called with no arguments is flagged once.
        messages = self._check(
            "import torch\n@torch.compile()\ndef f(x):\n    return x\n"
        )
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].line, 2)

    def test_call_decorator_with_backend_is_ok(self):
        # [FUNCTION DOC] Checks the compile decorator with a backend keyword is not flagged.
        messages = self._check(
            "import torch\n"
            '@torch.compile(backend="aot_eager")\n'
            "def f(x):\n"
            "    return x\n"
        )
        self.assertEqual(messages, [])

    def test_noqa_same_line_suppresses(self):
        # [FUNCTION DOC] Checks a same-line noqa marker suppresses the message.
        messages = self._check(
            "import torch\n"
            "def f(x):\n"
            "    return x\n"
            "g = torch.compile(f)  # noqa: UNSPECIFIED_BACKEND\n"
        )
        self.assertEqual(messages, [])

    def test_noqa_on_any_line_of_multiline_call_suppresses(self):
        # [FUNCTION DOC] Checks a noqa marker on any line of a multi-line call suppresses it.
        messages = self._check(
            "import torch\n"
            "def f(x):\n"
            "    return x\n"
            "g = torch.compile(\n"
            "    f,  # noqa: UNSPECIFIED_BACKEND\n"
            ")\n"
        )
        self.assertEqual(messages, [])

    def test_unrelated_compile_is_ignored(self):
        # [FUNCTION DOC] Checks a compile call from the regular expression module is ignored.
        messages = self._check('import re\np = re.compile("x")\n')
        self.assertEqual(messages, [])

    def test_multiple_offenders(self):
        # [FUNCTION DOC] Checks two offending calls give two messages on lines 4 and 5.
        messages = self._check(
            "import torch\n"
            "def f(x):\n"
            "    return x\n"
            "a = torch.compile(f)\n"
            "b = torch.compile(f)\n"
        )
        self.assertEqual(len(messages), 2)
        self.assertEqual([m.line for m in messages], [4, 5])

    def test_syntax_error_reports_message(self):
        # [FUNCTION DOC] Checks unparseable source yields one error message named syntax-error.
        messages = self._check("def f(:\n    pass\n")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].name, "syntax-error")
        self.assertEqual(messages[0].severity, LintSeverity.ERROR)


if __name__ == "__main__":
    unittest.main()
