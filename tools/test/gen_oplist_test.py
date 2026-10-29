#!/usr/bin/env python3
# [MODULE DOC] Unit test for the operator-list generator in tools.code_analyzer.gen_oplist.
# Verifies that throw_if_any_op_includes_overloads raises when any selected operator
# includes all overloads. Uses mock objects only; no import-time side effects.
# Copyright 2004-present Facebook. All Rights Reserved.

import unittest
from unittest.mock import MagicMock

from tools.code_analyzer.gen_oplist import throw_if_any_op_includes_overloads


class GenOplistTest(unittest.TestCase):
    # [CLASS DOC] Test case holding the single check of the overload-inclusion guard in
    # gen_oplist. Stateless.
    def setUp(self) -> None:
        # [FUNCTION DOC] HELPER: No-op fixture setup; nothing is prepared.
        pass

    def test_throw_if_any_op_includes_overloads(self) -> None:
        # [FUNCTION DOC] Builds a mock selective builder whose operators mapping contains
        # entries flagged include-all-overloads and asserts the guard raises an error; then
        # replaces the entries with none flagged and asserts the guard returns quietly.
        selective_builder = MagicMock()
        selective_builder.operators = MagicMock()
        selective_builder.operators.items.return_value = [
            ("op1", MagicMock(include_all_overloads=True)),
            ("op2", MagicMock(include_all_overloads=False)),
            ("op3", MagicMock(include_all_overloads=True)),
        ]

        self.assertRaises(
            Exception, throw_if_any_op_includes_overloads, selective_builder
        )

        selective_builder.operators.items.return_value = [
            ("op1", MagicMock(include_all_overloads=False)),
            ("op2", MagicMock(include_all_overloads=False)),
            ("op3", MagicMock(include_all_overloads=False)),
        ]

        # Here we do not expect it to throw an exception since none of the ops
        # include all overloads.
        throw_if_any_op_includes_overloads(selective_builder)
