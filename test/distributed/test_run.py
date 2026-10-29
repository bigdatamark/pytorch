#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]
# [MODULE DOC] Tests for the torchrun argument parser and launcher glue in
# torch.distributed.run and the launcher API: the signals-to-handle option and its environment
# variable, rendezvous defaults for single-node versus multi-node runs, the log line prefix
# template option, and the shell completion printing option. Elastic launch is mocked where a
# run is simulated. No import-time side effects.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from importlib.util import find_spec
from unittest.mock import MagicMock, patch

import torch.distributed.run as run
from torch.distributed.launcher.api import launch_agent, LaunchConfig
from torch.testing._internal.common_utils import run_tests, TestCase


class RunTest(TestCase):
    # [CLASS DOC] TestCase for torchrun CLI behavior. setUp records the original value of the
    # signals-to-handle environment variable and tearDown restores (or removes) it, since
    # tests may change it. Includes helpers to parse arguments, read the log prefix template,
    # and capture the output of the print-completion flag.
    def setUp(self):
        # [FUNCTION DOC] Run the base setUp and remember the current value of the
        # signals-to-handle environment variable (None if unset).
        super().setUp()
        # Save original environment variable if it exists
        self.original_signals_env = os.environ.get(
            "TORCHELASTIC_SIGNALS_TO_HANDLE", None
        )

    def tearDown(self):
        # [FUNCTION DOC] Restore the signals-to-handle environment variable to its saved
        # value, or delete it if it was originally unset and is now present.
        # KNOWN ISSUE: does not call the base class tearDown.
        # Restore original environment variable
        if self.original_signals_env is not None:
            os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"] = self.original_signals_env
        elif "TORCHELASTIC_SIGNALS_TO_HANDLE" in os.environ:
            del os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"]

    def test_signals_to_handle_default(self):
        # [FUNCTION DOC] Parsing a lone script name yields the default signals string
        # SIGTERM,SIGINT,SIGHUP,SIGQUIT.
        """Test that the default value for signals_to_handle is correctly set."""
        parser = run.get_args_parser()
        args = parser.parse_args(["dummy_script.py"])
        self.assertEqual(args.signals_to_handle, "SIGTERM,SIGINT,SIGHUP,SIGQUIT")

    def test_signals_to_handle_custom(self):
        # [FUNCTION DOC] An explicit signals option is parsed to the exact given comma
        # separated string.
        """Test that a custom value for signals_to_handle is correctly parsed."""
        parser = run.get_args_parser()
        args = parser.parse_args(
            ["--signals-to-handle=SIGTERM,SIGUSR1,SIGUSR2", "dummy_script.py"]
        )
        self.assertEqual(args.signals_to_handle, "SIGTERM,SIGUSR1,SIGUSR2")

    def test_config_from_args_signals_to_handle(self):
        # [FUNCTION DOC] The signals option parsed from arguments is carried into the
        # LaunchConfig produced by config_from_args.
        """Test that the signals_to_handle argument is correctly passed to LaunchConfig."""
        parser = run.get_args_parser()
        args = parser.parse_args(
            ["--signals-to-handle=SIGTERM,SIGUSR1,SIGUSR2", "dummy_script.py"]
        )
        config, _, _ = run.config_from_args(args)
        self.assertEqual(config.signals_to_handle, "SIGTERM,SIGUSR1,SIGUSR2")

    @patch("torch.distributed.launcher.api.LocalElasticAgent")
    @patch("torch.distributed.launcher.api.rdzv_registry.get_rendezvous_handler")
    def test_launch_agent_sets_environment_variable(self, mock_get_handler, mock_agent):
        # [FUNCTION DOC] With the local elastic agent and rendezvous handler mocked,
        # launch_agent on a config with custom signals sets the signals environment variable
        # to that string. The variable is removed first and a mock agent run result reports
        # success.
        """Test that launch_agent sets the TORCHELASTIC_SIGNALS_TO_HANDLE environment variable."""
        # Setup
        config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=1,
            signals_to_handle="SIGTERM,SIGUSR1,SIGUSR2",
        )
        entrypoint = "dummy_script.py"
        args = []

        # Make sure the environment variable doesn't exist before the test
        if "TORCHELASTIC_SIGNALS_TO_HANDLE" in os.environ:
            del os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"]

        # Mock agent.run() to return a MagicMock
        mock_agent_instance = MagicMock()
        mock_agent_instance.run.return_value = MagicMock(
            is_failed=lambda: False, return_values={}
        )
        mock_agent.return_value = mock_agent_instance

        # Call launch_agent
        launch_agent(config, entrypoint, args)

        # Verify that the environment variable was set correctly
        self.assertEqual(
            os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"], "SIGTERM,SIGUSR1,SIGUSR2"
        )

    def _parse_args(self, args):
        # [FUNCTION DOC] HELPER: Parse a list of command line arguments with the torchrun
        # parser and return the namespace.
        parser = run.get_args_parser()
        return parser.parse_args(args)

    @patch("torch.distributed.run.elastic_launch")
    def test_single_node_defaults_to_free_port(self, mock_launch):
        # [FUNCTION DOC] With elastic_launch mocked, running a single-node job without a
        # master port (port stays None) leaves args with rendezvous backend c10d and endpoint
        # localhost:0.
        """Single-node with no explicit port should auto-switch to c10d with localhost:0."""
        args = self._parse_args(["dummy_script.py"])
        self.assertIsNone(args.master_port)
        run.run(args)
        self.assertEqual(args.rdzv_backend, "c10d")
        self.assertEqual(args.rdzv_endpoint, "localhost:0")

    @patch("torch.distributed.run.elastic_launch")
    def test_single_node_explicit_port_uses_static(self, mock_launch):
        # [FUNCTION DOC] With elastic_launch mocked, an explicit master port (12345) is kept
        # and the rendezvous backend ends up static.
        """Single-node with explicit --master-port should keep static backend."""
        args = self._parse_args(["--master-port=12345", "dummy_script.py"])
        self.assertEqual(args.master_port, 12345)
        run.run(args)
        self.assertEqual(args.rdzv_backend, "static")

    @patch("torch.distributed.run.elastic_launch")
    def test_multi_node_defaults_to_static(self, mock_launch):
        # [FUNCTION DOC] With elastic_launch mocked, a two-node job ends with static backend
        # and master port 29500.
        """Multi-node should keep static backend with default port 29500."""
        args = self._parse_args(["--nnodes=2", "dummy_script.py"])
        run.run(args)
        self.assertEqual(args.rdzv_backend, "static")
        self.assertEqual(args.master_port, 29500)

    @patch("torch.distributed.run.elastic_launch")
    def test_standalone_still_works(self, mock_launch):
        # [FUNCTION DOC] With elastic_launch mocked, the standalone flag results in backend
        # c10d and endpoint localhost:0.
        """--standalone should still use c10d with localhost:0."""
        args = self._parse_args(["--standalone", "dummy_script.py"])
        run.run(args)
        self.assertEqual(args.rdzv_backend, "c10d")
        self.assertEqual(args.rdzv_endpoint, "localhost:0")

    def _log_line_prefix_template(self, args):
        # [FUNCTION DOC] HELPER: Build a LaunchConfig from the given arguments with both log
        # line prefix environment variables removed (via a patched environment copy) and
        # return its log_line_prefix_template.
        """config_from_args' template, with the prefix env vars under our control."""
        with patch.dict(os.environ):
            os.environ.pop("TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE", None)
            os.environ.pop("PET_LOG_LINE_PREFIX_TEMPLATE", None)
            config, _, _ = run.config_from_args(self._parse_args(args))
        return config.log_line_prefix_template

    def test_log_line_prefix_template_from_args(self):
        # [FUNCTION DOC] The log line prefix template option is passed through unchanged to
        # the config.
        """--log-line-prefix-template is passed through to LaunchConfig."""
        template = "${hostname}:${rank}: "
        self.assertEqual(
            template,
            self._log_line_prefix_template(
                [f"--log-line-prefix-template={template}", "dummy_script.py"]
            ),
        )

    def test_log_line_prefix_template_unset(self):
        # [FUNCTION DOC] Without the option or env vars, the config template is None.
        """Unset, the template stays None so the default [role_name][local_rank] prefix is kept."""
        self.assertIsNone(self._log_line_prefix_template(["dummy_script.py"]))

    def test_log_line_prefix_template_env_var(self):
        # [FUNCTION DOC] With the TORCHELASTIC prefix environment variable set, the config
        # uses it; an explicit command line option takes precedence over it.
        """TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE still works, and the flag overrides it."""
        with patch.dict(
            os.environ, {"TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE": "[${rank}]:"}
        ):
            config, _, _ = run.config_from_args(self._parse_args(["dummy_script.py"]))
            self.assertEqual("[${rank}]:", config.log_line_prefix_template)

            config, _, _ = run.config_from_args(
                self._parse_args(
                    ["--log-line-prefix-template=${hostname}: ", "dummy_script.py"]
                )
            )
            self.assertEqual("${hostname}: ", config.log_line_prefix_template)

    def _print_completion(self, shell):
        # [FUNCTION DOC] HELPER: Parse the print-completion option for a shell with stdout and
        # stderr captured, expecting SystemExit. Returns the exit code, captured stdout and
        # captured stderr.
        """Run --print-completion, returning (exit_code, stdout, stderr)."""
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            with self.assertRaises(SystemExit) as cm:
                # no training_script: the action must exit before argparse
                # complains about the missing positional
                self._parse_args([f"--print-completion={shell}"])
        return cm.exception.code, out.getvalue(), err.getvalue()

    @unittest.skipIf(find_spec("shtab") is None, "shtab is not installed")
    def test_print_completion(self):
        # [FUNCTION DOC] Skipped unless shtab is installed. For bash, zsh and tcsh, printing
        # completion exits with code 0 and the output mentions the nproc-per-node and
        # rdzv-backend options and torchrun.
        """--print-completion emits a completion script naming torchrun's options."""
        for shell in ("bash", "zsh", "tcsh"):
            with self.subTest(shell=shell):
                code, out, _ = self._print_completion(shell)
                self.assertEqual(0, code)
                # tcsh lists options without the leading dashes
                self.assertIn("nproc-per-node", out)
                self.assertIn("rdzv-backend", out)
                # completions must bind to `torchrun`, not to the module that
                # happens to be running the parser
                self.assertIn("torchrun", out)

    @unittest.skipIf(find_spec("shtab") is not None, "shtab is installed")
    def test_print_completion_without_shtab(self):
        # [FUNCTION DOC] Skipped if shtab is installed. Asking for completion without shtab
        # exits with code 1 and stderr mentions installing shtab.
        """Without shtab, the flag fails with an actionable message, not a traceback."""
        code, _, err = self._print_completion("bash")
        self.assertEqual(1, code)
        self.assertIn("pip install shtab", err)

    def test_print_completion_rejects_unknown_shell(self):
        # [FUNCTION DOC] An unknown shell name passed to the print-completion option makes
        # argument parsing exit (SystemExit) with stderr suppressed.
        """The shell choices are validated by argparse.

        Deliberately not a real shell: the choices come from
        shtab.SUPPORTED_SHELLS when shtab is installed, so naming one shtab
        could plausibly add later would make this test fail on upgrade.
        """
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                self._parse_args(["--print-completion=notashell"])


if __name__ == "__main__":
    run_tests()
