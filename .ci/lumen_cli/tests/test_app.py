# tests/test_cli.py
# [MODULE DOC] Unit tests for the top-level lumen CLI entry point: that the build
# external vllm command constructs and runs the vLLM build runner, and that build help
# exits cleanly and mentions the external subcommand.
import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from cli.run import main


class TestArgparseCLI(unittest.TestCase):
    # [CLASS DOC] Test case for the argparse driven entry point; patches the runner so
    # nothing real is built.
    @patch("cli.build_cli.register_build.VllmBuildRunner.run", return_value=None)
    @patch("cli.build_cli.register_build.VllmBuildRunner.__init__", return_value=None)
    def test_cli_run_build_external(self, mock_init, mock_run):
        # [FUNCTION DOC] Run main with arguments for the build external vllm command with
        # the runner constructor and run method mocked; expects each called exactly once.
        # Any SystemExit from parsing is swallowed.
        from cli.run import main  # import after patches if needed

        test_args = ["cli.run", "build", "external", "vllm"]
        with patch.object(sys, "argv", test_args):
            # argparse may call sys.exit on error; capture to avoid test aborts
            try:
                main()
            except SystemExit:
                pass
        mock_init.assert_called_once()  # got constructed
        mock_run.assert_called_once_with()  # run() called

    def test_build_help(self):
        # [FUNCTION DOC] Run main with the build help option while capturing output;
        # expects a SystemExit with code zero and usage text that mentions external.
        test_args = ["cli.run", "build", "--help"]

        with patch.object(sys, "argv", test_args):
            stdout = io.StringIO()
            stderr = io.StringIO()

            # --help always raises SystemExit(0)
            with self.assertRaises(SystemExit) as cm:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    main()

            self.assertEqual(cm.exception.code, 0)

            output = stdout.getvalue()
            self.assertIn("usage", output)
            self.assertIn("external", output)


if __name__ == "__main__":
    unittest.main()
