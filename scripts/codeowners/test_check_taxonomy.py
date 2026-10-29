#!/usr/bin/env python3
# [MODULE DOC] Unit tests for check_taxonomy: pattern parsing and owner validation, tracked
# path listing in a temporary git repository (git subprocess calls), and the source-order
# specificity check in analyze. Uses PyTorch's TestCase and parametrization; run directly.

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from check_taxonomy import analyze, parse_patterns, TaxonomyPattern, tracked_paths

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestCodeownersTaxonomy(TestCase):
    # [CLASS DOC] Tests for the CODEOWNERS taxonomy checker: owner syntax handling in
    # parse_patterns, tracked_paths behavior with index-only files, and analyze's detection of
    # rule order problems. Parametrized via instantiate_parametrized_tests below.
    def parse_entry(self, entry: str):
        # [FUNCTION DOC] HELPER: Writes a minimal CODEOWNERS file in a temporary directory
        # containing the rules marker, one group named surface and the given entry line, then
        # returns the result of parse_patterns on it (patterns, groups); errors propagate.
        """Parse one taxonomy entry from a minimal CODEOWNERS fixture."""
        with TemporaryDirectory() as directory:
            codeowners = Path(directory) / "CODEOWNERS"
            codeowners.write_text(
                f"# Draft grouped review-surface definitions:\n# [surface]\n{entry}\n"
            )
            return parse_patterns(codeowners)

    @parametrize(
        "entry",
        [
            "/path/ @user",
            "/path/ @org/team",
            "/path/ owner@example.com",
            "/path/ @user # explanation",
        ],
    )
    def test_valid_active_owners(self, entry):
        # [FUNCTION DOC] Active entries with a user, team, email owner (optionally followed by
        # a trailing comment) parse into one group named surface and a path without the
        # leading slash.
        patterns, groups = self.parse_entry(entry)
        self.assertEqual(groups, ["surface"])
        self.assertEqual(patterns[0].path, "path/")

    @parametrize("entry", ["/path/ not-an-owner", "/path/ # explanation"])
    def test_invalid_active_owners(self, entry):
        # [FUNCTION DOC] An active entry with a malformed owner, or only a comment instead of
        # an owner, makes parse_patterns raise ValueError mentioning missing or invalid owners.
        with self.assertRaisesRegex(
            ValueError, "active pattern has no owners|invalid active owners"
        ):
            self.parse_entry(entry)

    def test_tracked_paths_ignore_index_only_files(self):
        # [FUNCTION DOC] Creates a temporary git repo with a commit containing a gitignore and
        # one file, then force-adds an ignored file to the index without committing it.
        # Asserts tracked_paths lists only the committed files. Runs real git commands.
        with TemporaryDirectory() as directory:
            repo = Path(directory)
            git = ["git", "-C", repo]
            subprocess.run([*git, "init"], check=True, capture_output=True)
            (repo / ".gitignore").write_text("generated.pyi\n")
            (repo / "tracked.py").touch()
            subprocess.run([*git, "add", ".gitignore", "tracked.py"], check=True)
            subprocess.run(
                [
                    *git,
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.com",
                    "commit",
                    "-m",
                    "initial",
                ],
                check=True,
                capture_output=True,
            )
            (repo / "generated.pyi").touch()
            subprocess.run([*git, "add", "--force", "generated.pyi"], check=True)
            self.assertEqual(tracked_paths(repo), [".gitignore", "tracked.py"])

    def test_source_order_must_preserve_specificity(self):
        # [FUNCTION DOC] When a broad directory rule appears after a more specific file rule,
        # analyze reports exactly one source-order mismatch message naming both groups.
        path = "torch/_higher_order_ops/flex_attention.py"
        specific = TaxonomyPattern("inductor", path, 1)
        broad = TaxonomyPattern("export", "torch/_higher_order_ops/", 2)
        report = analyze([path], [specific, broad])
        self.assertEqual(
            report.source_order_mismatches,
            [f"{path}: export:torch/_higher_order_ops/ overrides inductor:{path}"],
        )

    def test_broad_rule_before_specific_rule(self):
        # [FUNCTION DOC] When the broad rule precedes the specific rule, analyze reports no
        # source-order mismatches.
        path = "torch/_higher_order_ops/flex_attention.py"
        broad = TaxonomyPattern("export", "torch/_higher_order_ops/", 1)
        specific = TaxonomyPattern("inductor", path, 2)
        self.assertEqual(analyze([path], [broad, specific]).source_order_mismatches, [])


instantiate_parametrized_tests(TestCodeownersTaxonomy)

if __name__ == "__main__":
    run_tests()
