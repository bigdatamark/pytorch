#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] Unit tests for the argparse actions env and check_env in
# torch.distributed.argparse_util. These actions let a command line option fall back to a
# PET_-prefixed environment variable (name derived from the option, upper-cased) when the
# option is not given on the command line; the command line always wins. No import-time
# side effects beyond imports.
import os
import unittest
from argparse import ArgumentParser

from torch.distributed.argparse_util import check_env, env


class ArgParseUtilTest(unittest.TestCase):
    # [CLASS DOC] Test case covering the env and check_env argparse actions with and
    # without PET_ environment variables, with and without defaults, and for required
    # options. Each test builds its own parser; environment variables are set directly in
    # os.environ and cleared by setUp before the next test.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Deletes every environment variable whose name starts with
        # PET_ (mutates the process environment) so tests do not leak into one another.
        super().setUp()
        # remove any lingering environment variables
        for e in os.environ.keys():  # noqa: SIM118
            if e.startswith("PET_"):
                del os.environ[e]

    def test_env_string_arg_no_env(self):
        # [FUNCTION DOC] String option with default and no env var: the default is used
        # when absent, and short or long flags override it.
        parser = ArgumentParser()
        parser.add_argument("-f", "--foo", action=env, default="bar")

        self.assertEqual("bar", parser.parse_args([]).foo)
        self.assertEqual("baz", parser.parse_args(["-f", "baz"]).foo)
        self.assertEqual("baz", parser.parse_args(["--foo", "baz"]).foo)

    def test_env_string_arg_env(self):
        # [FUNCTION DOC] String option with default and PET_FOO set: the env value replaces
        # the default, but explicit command line values still take precedence.
        os.environ["PET_FOO"] = "env_baz"
        parser = ArgumentParser()
        parser.add_argument("-f", "--foo", action=env, default="bar")

        self.assertEqual("env_baz", parser.parse_args([]).foo)
        self.assertEqual("baz", parser.parse_args(["-f", "baz"]).foo)
        self.assertEqual("baz", parser.parse_args(["--foo", "baz"]).foo)

    def test_env_int_arg_no_env(self):
        # [FUNCTION DOC] Integer-typed option, no env var: default 1 when absent, and
        # command line strings are converted to int.
        parser = ArgumentParser()
        parser.add_argument("-f", "--foo", action=env, default=1, type=int)

        self.assertEqual(1, parser.parse_args([]).foo)
        self.assertEqual(2, parser.parse_args(["-f", "2"]).foo)
        self.assertEqual(2, parser.parse_args(["--foo", "2"]).foo)

    def test_env_int_arg_env(self):
        # [FUNCTION DOC] Integer-typed option with PET_FOO set to a numeric string: the env
        # value is converted to int (3); command line values override it.
        os.environ["PET_FOO"] = "3"
        parser = ArgumentParser()
        parser.add_argument("-f", "--foo", action=env, default=1, type=int)

        self.assertEqual(3, parser.parse_args([]).foo)
        self.assertEqual(2, parser.parse_args(["-f", "2"]).foo)
        self.assertEqual(2, parser.parse_args(["--foo", "2"]).foo)

    def test_env_no_default_no_env(self):
        # [FUNCTION DOC] No default and no env var: the parsed value is None when the
        # option is absent; command line values are parsed normally.
        parser = ArgumentParser()
        parser.add_argument("-f", "--foo", action=env)

        self.assertIsNone(parser.parse_args([]).foo)
        self.assertEqual("baz", parser.parse_args(["-f", "baz"]).foo)
        self.assertEqual("baz", parser.parse_args(["--foo", "baz"]).foo)

    def test_env_no_default_env(self):
        # [FUNCTION DOC] No default but PET_FOO set: the env value is used when the option
        # is absent; the command line still wins.
        os.environ["PET_FOO"] = "env_baz"
        parser = ArgumentParser()
        parser.add_argument("-f", "--foo", action=env)

        self.assertEqual("env_baz", parser.parse_args([]).foo)
        self.assertEqual("baz", parser.parse_args(["-f", "baz"]).foo)
        self.assertEqual("baz", parser.parse_args(["--foo", "baz"]).foo)

    def test_env_required_no_env(self):
        # [FUNCTION DOC] Required option with no env var: parsing succeeds when the flag is
        # given in short or long form (the missing case is not tested).
        parser = ArgumentParser()
        parser.add_argument("-f", "--foo", action=env, required=True)

        self.assertEqual("baz", parser.parse_args(["-f", "baz"]).foo)
        self.assertEqual("baz", parser.parse_args(["--foo", "baz"]).foo)

    def test_env_required_env(self):
        # [FUNCTION DOC] Required option with a default and PET_FOO set: the env value
        # satisfies the requirement when the flag is omitted; command line still overrides.
        os.environ["PET_FOO"] = "env_baz"
        parser = ArgumentParser()
        parser.add_argument("-f", "--foo", action=env, default="bar", required=True)

        self.assertEqual("env_baz", parser.parse_args([]).foo)
        self.assertEqual("baz", parser.parse_args(["-f", "baz"]).foo)
        self.assertEqual("baz", parser.parse_args(["--foo", "baz"]).foo)

    def test_check_env_no_env(self):
        # [FUNCTION DOC] Boolean flag via check_env without env var or default: false when
        # absent, true when the short or long flag is passed.
        parser = ArgumentParser()
        parser.add_argument("-v", "--verbose", action=check_env)

        self.assertFalse(parser.parse_args([]).verbose)
        self.assertTrue(parser.parse_args(["-v"]).verbose)
        self.assertTrue(parser.parse_args(["--verbose"]).verbose)

    def test_check_env_default_no_env(self):
        # [FUNCTION DOC] check_env flag with default true and no env var: true whether or
        # not the flag is passed.
        parser = ArgumentParser()
        parser.add_argument("-v", "--verbose", action=check_env, default=True)

        self.assertTrue(parser.parse_args([]).verbose)
        self.assertTrue(parser.parse_args(["-v"]).verbose)
        self.assertTrue(parser.parse_args(["--verbose"]).verbose)

    def test_check_env_env_zero(self):
        # [FUNCTION DOC] PET_VERBOSE set to "0": the flag is false when absent from the
        # command line, but passing the flag still makes it true.
        os.environ["PET_VERBOSE"] = "0"
        parser = ArgumentParser()
        parser.add_argument("-v", "--verbose", action=check_env)

        self.assertFalse(parser.parse_args([]).verbose)
        self.assertTrue(parser.parse_args(["--verbose"]).verbose)

    def test_check_env_env_one(self):
        # [FUNCTION DOC] PET_VERBOSE set to "1": the flag is true even when absent from the
        # command line, and true when passed.
        os.environ["PET_VERBOSE"] = "1"
        parser = ArgumentParser()
        parser.add_argument("-v", "--verbose", action=check_env)

        self.assertTrue(parser.parse_args([]).verbose)
        self.assertTrue(parser.parse_args(["--verbose"]).verbose)

    def test_check_env_default_env_zero(self):
        # [FUNCTION DOC] Default true but PET_VERBOSE "0": the env value overrides the
        # default to false; passing the flag makes it true.
        os.environ["PET_VERBOSE"] = "0"
        parser = ArgumentParser()
        parser.add_argument("-v", "--verbose", action=check_env, default=True)

        self.assertFalse(parser.parse_args([]).verbose)
        self.assertTrue(parser.parse_args(["--verbose"]).verbose)

    def test_check_env_default_env_one(self):
        # [FUNCTION DOC] Default true and PET_VERBOSE "1": true both without and with the
        # flag.
        os.environ["PET_VERBOSE"] = "1"
        parser = ArgumentParser()
        parser.add_argument("-v", "--verbose", action=check_env, default=True)

        self.assertTrue(parser.parse_args([]).verbose)
        self.assertTrue(parser.parse_args(["--verbose"]).verbose)
