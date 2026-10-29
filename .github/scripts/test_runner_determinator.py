# [MODULE DOC] Unit tests for the runner_determinator script, which decides the runner fleet
# prefix (for example the default Meta fleet, the lf fleet, or the amd-sandbox fleet) for a CI
# run. Covers parsing of the rollout settings issue text (experiments and per-user opt-ins), the
# get_runner_prefix decision logic (opt-in, opt-out, rollout percentages, exception branches,
# extra experiment filters), the amd-sandbox experiment, and the no-runner-experiments label.
# Imports runner_determinator as rd; random.uniform is patched where rollouts matter. No
# import-time side effects. Runs via unittest main when executed directly.

from argparse import Namespace
from unittest import main, TestCase
from unittest.mock import Mock, patch

import runner_determinator as rd


USER_BRANCH = "somebranch"
EXCEPTION_BRANCH = "main"


class TestRunnerDeterminatorIssueParser(TestCase):
    # [CLASS DOC] Tests for parsing the settings issue text: the experiments section and the Users
    # section, including code-block wrapping, bad names and percentage handling.
    def test_parse_settings(self) -> None:
        # [FUNCTION DOC] Parses a settings text with two experiments and checks both become
        # Experiment records with the declared rollout percentage and default flag.
        settings_text = """
        experiments:
            lf:
                rollout_perc: 25
            otherExp:
                rollout_perc: 0
                default: false
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """

        settings = rd.parse_settings(settings_text)

        self.assertTupleEqual(
            rd.Experiment(rollout_perc=25),
            settings.experiments["lf"],
            "lf settings not parsed correctly",
        )
        self.assertTupleEqual(
            rd.Experiment(rollout_perc=0, default=False),
            settings.experiments["otherExp"],
            "otherExp settings not parsed correctly",
        )

    def test_parse_settings_with_invalid_experiment_name_skips_experiment(self) -> None:
        # [FUNCTION DOC] Checks an experiment whose name starts with a dash is skipped by
        # parse_settings, while valid experiments in the same text are still parsed.
        settings_text = """
        experiments:
            lf:
                rollout_perc: 25
            -badExp:
                rollout_perc: 0
                default: false
        ---

        Users:
        @User1,lf
        @User2,lf,-badExp

        """

        settings = rd.parse_settings(settings_text)

        self.assertTupleEqual(
            rd.Experiment(rollout_perc=25),
            settings.experiments["lf"],
            "lf settings not parsed correctly",
        )
        self.assertNotIn("-badExp", settings.experiments)

    def test_parse_settings_in_code_block(self) -> None:
        # [FUNCTION DOC] Checks the experiments section is parsed when wrapped in a
        # fenced code block.
        settings_text = """

        ```
        experiments:
            lf:
                rollout_perc: 25
            otherExp:
                rollout_perc: 0
                default: false
        ```

        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """

        settings = rd.parse_settings(settings_text)

        self.assertTupleEqual(
            rd.Experiment(rollout_perc=25),
            settings.experiments["lf"],
            "lf settings not parsed correctly",
        )
        self.assertTupleEqual(
            rd.Experiment(rollout_perc=0, default=False),
            settings.experiments["otherExp"],
            "otherExp settings not parsed correctly",
        )

    def test_parse_all_branches_setting(self) -> None:
        # [FUNCTION DOC] Checks the all_branches flag (lowercase or capitalized true) is parsed
        # into the Experiment record, so such experiments are allowed on exception branches.
        settings_text = """
        ```
        experiments:
            lf:
                rollout_perc: 25
                all_branches: true
            otherExp:
                all_branches: True
                rollout_perc: 0
        ```

        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """

        settings = rd.parse_settings(settings_text)

        self.assertTupleEqual(
            rd.Experiment(rollout_perc=25, all_branches=True),
            settings.experiments["lf"],
            "lf settings not parsed correctly",
        )
        self.assertTrue(settings.experiments["otherExp"].all_branches)
        self.assertTupleEqual(
            rd.Experiment(rollout_perc=0, all_branches=True),
            settings.experiments["otherExp"],
            "otherExp settings not parsed correctly",
        )

    def test_parse_users(self) -> None:
        # [FUNCTION DOC] Checks parse_users maps each user name to a list of experiment configs,
        # with a default rollout of 100 percent for plain opt-ins.
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """

        users = rd.parse_users(settings_text)
        self.assertEqual(
            [rd.UserExperimentConfig("lf", 100)],
            users["User1"],
        )
        self.assertEqual(
            [
                rd.UserExperimentConfig("lf", 100),
                rd.UserExperimentConfig("otherExp", 100),
            ],
            users["User2"],
        )

    def test_parse_users_without_settings(self) -> None:
        # [FUNCTION DOC] Checks parse_users still finds user lines when the text has
        # no settings section.
        settings_text = """

        @User1,lf
        @User2,lf,otherExp

        """

        users = rd.parse_users(settings_text)
        self.assertEqual(
            [rd.UserExperimentConfig("lf", 100)],
            users["User1"],
        )
        self.assertEqual(
            [
                rd.UserExperimentConfig("lf", 100),
                rd.UserExperimentConfig("otherExp", 100),
            ],
            users["User2"],
        )

    def test_parse_users_with_rollout_perc(self) -> None:
        # [FUNCTION DOC] Checks the name:percent syntax in user lines yields configs
        # with that percentage.
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            arc:
                rollout_perc: 0
        ---

        Users:
        @User1,lf,arc:10
        @User2,arc:50
        @User3,lf

        """

        users = rd.parse_users(settings_text)
        self.assertEqual(
            [
                rd.UserExperimentConfig("lf", 100),
                rd.UserExperimentConfig("arc", 10),
            ],
            users["User1"],
        )
        self.assertEqual(
            [rd.UserExperimentConfig("arc", 50)],
            users["User2"],
        )
        self.assertEqual(
            [rd.UserExperimentConfig("lf", 100)],
            users["User3"],
        )

    def test_parse_users_invalid_percentage_defaults_to_100(self) -> None:
        # [FUNCTION DOC] Checks a non-numeric percentage falls back to 100 percent.
        """Non-numeric percentage like arc:abc should default to 100%."""
        settings_text = """
        @User1,arc:abc
        """

        users = rd.parse_users(settings_text)
        self.assertEqual(
            [rd.UserExperimentConfig("arc", 100)],
            users["User1"],
        )

    def test_parse_users_negative_percentage_clamped_to_zero(self) -> None:
        # [FUNCTION DOC] Checks a negative percentage is clamped to 0.
        """Negative percentage like arc:-5 should be clamped to 0."""
        settings_text = """
        @User1,arc:-5
        """

        users = rd.parse_users(settings_text)
        self.assertEqual(
            [rd.UserExperimentConfig("arc", 0)],
            users["User1"],
        )

    def test_parse_users_over_100_percentage_clamped(self) -> None:
        # [FUNCTION DOC] Checks a percentage above 100 is clamped to 100.
        """Percentage over 100 like arc:200 should be clamped to 100."""
        settings_text = """
        @User1,arc:200
        """

        users = rd.parse_users(settings_text)
        self.assertEqual(
            [rd.UserExperimentConfig("arc", 100)],
            users["User1"],
        )

    def test_parse_users_opt_out_ignores_percentage(self) -> None:
        # [FUNCTION DOC] Checks an opt-out entry (leading dash) keeps its name and a 100 percent
        # value, without parsing any percentage.
        """Opt-out entries like -lf should not parse a percentage."""
        settings_text = """
        @User1,-lf
        """

        users = rd.parse_users(settings_text)
        self.assertEqual(
            [rd.UserExperimentConfig("-lf", 100)],
            users["User1"],
        )


class TestRunnerDeterminatorGetRunnerPrefix(TestCase):
    # [CLASS DOC] Tests for get_runner_prefix: which runner prefix is chosen given the settings
    # text, the requesting users, the branch, and optional experiment filters. The default fleet
    # prefix is mt- and the lf experiment yields lf-.
    def test_opted_in_user(self) -> None:
        # [FUNCTION DOC] Checks a user opted into lf gets the lf- prefix.
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """
        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for User1")

    def test_explicitly_opted_out_user(self) -> None:
        # [FUNCTION DOC] Checks a user with an explicit opt-out of lf gets the default mt- prefix
        # even when the experiment is rolled out to everyone.
        settings_text = """
        experiments:
            lf:
                rollout_perc: 100
            otherExp:
                rollout_perc: 0
        ---

        Users:
        @User1,-lf
        @User2,lf,otherExp

        """
        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("mt-", result.prefix, "Runner prefix not correct for User1")

    def test_explicitly_opted_in_and_out_user_should_opt_out(self) -> None:
        # [FUNCTION DOC] Checks that when a user both opts out and opts in to the same experiment,
        # the opt-out wins and the default prefix is used.
        settings_text = """
        experiments:
            lf:
                rollout_perc: 100
            otherExp:
                rollout_perc: 0
        ---

        Users:
        @User1,-lf,lf
        @User2,lf,otherExp

        """
        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("mt-", result.prefix, "Runner prefix not correct for User1")

    def test_opted_in_user_two_experiments(self) -> None:
        # [FUNCTION DOC] Checks a user opted into lf and another experiment still gets
        # the lf- prefix.
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """
        result = rd.get_runner_prefix(settings_text, ["User2"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for User2")

    def test_opted_in_user_two_experiments_default(self) -> None:
        # [FUNCTION DOC] Checks the lf prefix is chosen when the second experiment is non-default.
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
                default: false
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """
        result = rd.get_runner_prefix(settings_text, ["User2"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for User2")

    def test_opted_in_user_two_experiments_default_exp(self) -> None:
        # [FUNCTION DOC] Checks that with an eligible experiment filter containing both
        # experiments, the user still gets the lf- prefix.
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
                default: false
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """
        result = rd.get_runner_prefix(
            settings_text, ["User2"], USER_BRANCH, frozenset(["lf", "otherExp"])
        )
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for User2")

    def test_opted_in_user_two_experiments_default_exp_2(self) -> None:
        # [FUNCTION DOC] Checks that when the eligible experiment filter excludes lf, the user
        # falls back to the default mt- prefix.
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
                default: false
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """
        result = rd.get_runner_prefix(
            settings_text, ["User2"], USER_BRANCH, frozenset(["otherExp"])
        )
        self.assertEqual("mt-", result.prefix, "Runner prefix not correct for User2")

    @patch("random.uniform", return_value=50)
    def test_opted_out_user(self, mock_uniform: Mock) -> None:
        # [FUNCTION DOC] Checks a user not listed anywhere stays on mt- when the random draw (50)
        # is above the 25 percent rollout.
        settings_text = """
        experiments:
            lf:
                rollout_perc: 25
            otherExp:
                rollout_perc: 25
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """
        result = rd.get_runner_prefix(settings_text, ["User3"], USER_BRANCH)
        self.assertEqual("mt-", result.prefix, "Runner prefix not correct for user")

    @patch("random.uniform", return_value=10)
    def test_opted_out_user_was_pulled_in_by_rollout(self, mock_uniform: Mock) -> None:
        # [FUNCTION DOC] Checks an unlisted user is pulled into the experiment when the random
        # draw (10) is below the rollout percentage.
        settings_text = """
        experiments:
            lf:
                rollout_perc: 25
            otherExp:
                rollout_perc: 25
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """

        # User3 is opted out, but is pulled into both experiments by the 10% rollout
        result = rd.get_runner_prefix(settings_text, ["User3"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for user")

    @patch("random.uniform", return_value=10)
    def test_opted_out_user_was_pulled_in_by_rollout_excl_nondefault(
        self, mock_uniform: Mock
    ) -> None:
        # [FUNCTION DOC] Checks a rollout pulls an unlisted user into default experiments only;
        # the non-default experiment is excluded.
        settings_text = """
        experiments:
            lf:
                rollout_perc: 25
            otherExp:
                rollout_perc: 25
                default: false
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """

        # User3 is opted out, but is pulled into default experiments by the 10% rollout
        result = rd.get_runner_prefix(settings_text, ["User3"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for user")

    @patch("random.uniform", return_value=10)
    def test_opted_out_user_was_pulled_in_by_rollout_filter_exp(
        self, mock_uniform: Mock
    ) -> None:
        # [FUNCTION DOC] Checks the eligible experiment filter limits which experiments a rollout
        # can pull an unlisted user into; excluding lf leaves the default mt- prefix.
        settings_text = """
        experiments:
            lf:
                rollout_perc: 25
            otherExp:
                rollout_perc: 25
                default: false
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """

        # User3 is opted out, but is pulled into default experiments by the 10% rollout
        result = rd.get_runner_prefix(
            settings_text, ["User3"], USER_BRANCH, frozenset(["otherExp"])
        )
        self.assertEqual("mt-", result.prefix, "Runner prefix not correct for user")

    @patch("random.uniform", return_value=25)
    def test_opted_out_user_was_pulled_out_by_rollout_filter_exp(
        self, mock_uniform: Mock
    ) -> None:
        # [FUNCTION DOC] Checks an unlisted user stays on mt- when the random draw (25) is above
        # the lf rollout percentage of 10.
        settings_text = """
        experiments:
            lf:
                rollout_perc: 10
            otherExp:
                rollout_perc: 50
                default: false
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """

        # User3 is opted out, but is pulled into default experiments by the 10% rollout
        result = rd.get_runner_prefix(settings_text, ["User3"], USER_BRANCH)
        self.assertEqual("mt-", result.prefix, "Runner prefix not correct for user")

    def test_lf_prefix_always_comes_first(self) -> None:
        # [FUNCTION DOC] Checks lf wins the prefix even when it is declared after another
        # experiment or listed after it for the user.
        settings_text = """
        experiments:
            otherExp:
                rollout_perc: 0
            lf:
                rollout_perc: 0
        ---

        Users:
        @User1,lf
        @User2,otherExp,lf

        """

        result = rd.get_runner_prefix(settings_text, ["User2"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for user")

    def test_ignores_commented_users(self) -> None:
        # [FUNCTION DOC] Checks user lines starting with a hash are ignored as comments.
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
        ---

        Users:
        #@User1,lf
        @User2,lf,otherExp

        """

        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("mt-", result.prefix, "Runner prefix not correct for user")

    def test_ignores_extra_experiments(self) -> None:
        # [FUNCTION DOC] Checks user opt-ins for experiments beyond the ones that contribute a
        # prefix do not change the lf- result.
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
            foo:
                rollout_perc: 0
        ---

        Users:
        @User1,lf,otherExp,foo

        """

        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for user")

    def test_disables_experiment_on_exception_branches_when_not_explicitly_opted_in(
        self,
    ) -> None:
        # [FUNCTION DOC] Checks an experiment without all_branches is disabled on an exception
        # branch (main) unless explicitly allowed, so the prefix is mt-.
        # KNOWN ISSUE: the settings list user User while the request is for User1, so the test
        # passes without the user being opted in at all.
        settings_text = """
        experiments:
            lf:
                rollout_perc: 100
        ---

        Users:
        @User,lf,otherExp

        """

        result = rd.get_runner_prefix(settings_text, ["User1"], EXCEPTION_BRANCH)
        self.assertEqual("mt-", result.prefix, "Runner prefix not correct for user")

    def test_allows_experiment_on_exception_branches_when_explicitly_opted_in(
        self,
    ) -> None:
        # [FUNCTION DOC] Checks an experiment with all_branches enabled stays active on the
        # exception branch, yielding lf-.
        # KNOWN ISSUE: the settings list user User while the request is for User1, so the result
        # comes from the 100 percent rollout rather than the opt-in.
        settings_text = """
        experiments:
            lf:
                rollout_perc: 100
                all_branches: true
        ---

        Users:
        @User,lf,otherExp

        """

        result = rd.get_runner_prefix(settings_text, ["User1"], EXCEPTION_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for user")

    @patch("random.uniform", return_value=5)
    def test_opted_in_user_with_rollout_perc_enabled(self, mock_uniform: Mock) -> None:
        # [FUNCTION DOC] Checks a user opted in at 10 percent is enabled when the random draw (5)
        # is below that percentage.
        """User opted in with 10% rollout, random=5 -> enabled"""
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
        ---

        Users:
        @User1,lf:10

        """

        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for user")

    @patch("random.uniform", return_value=50)
    def test_opted_in_user_with_rollout_perc_disabled(self, mock_uniform: Mock) -> None:
        # [FUNCTION DOC] Checks a user opted in at 10 percent is not enabled when the random draw
        # (50) is above it, giving the mt- prefix.
        """User opted in with 10% rollout, random=50 -> disabled"""
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
        ---

        Users:
        @User1,lf:10

        """

        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("mt-", result.prefix, "Runner prefix not correct for user")

    def test_opted_in_user_without_rollout_perc_always_enabled(self) -> None:
        # [FUNCTION DOC] Checks an opt-in without a percentage is always enabled.
        """User opted in without percentage (default 100%) -> always enabled"""
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
        ---

        Users:
        @User1,lf

        """

        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for user")

    @patch("random.uniform", return_value=15)
    def test_multiple_requesters_uses_min_perc(self, mock_uniform: Mock) -> None:
        # [FUNCTION DOC] Checks that with multiple requesters the minimum opt-in percentage
        # applies: draw 15 against minimum 10 disables the experiment.
        """Two requesters with different rollout_percs, uses the minimum (10%)."""
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
        ---

        Users:
        @User1,lf:10
        @User2,lf:50

        """

        # random=15, min_perc=10 -> 15 > 10 -> disabled
        result = rd.get_runner_prefix(settings_text, ["User1", "User2"], USER_BRANCH)
        self.assertEqual("mt-", result.prefix, "Runner prefix not correct for user")

    @patch("random.uniform", return_value=5)
    def test_multiple_requesters_uses_min_perc_enabled(
        self, mock_uniform: Mock
    ) -> None:
        # [FUNCTION DOC] Checks the minimum opt-in percentage of multiple requesters enables the
        # experiment when the random draw (5) is below it.
        """Two requesters with different rollout_percs, min=10%, random=5 -> enabled."""
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
        ---

        Users:
        @User1,lf:10
        @User2,lf:50

        """

        result = rd.get_runner_prefix(settings_text, ["User1", "User2"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for user")

    def test_opt_out_overrides_rollout_perc(self) -> None:
        # [FUNCTION DOC] Checks an opt-out entry beats an opt-in with a rollout percentage.
        """Opt-out (-lf) wins over opt-in with rollout_perc (lf:50)."""
        settings_text = """
        experiments:
            lf:
                rollout_perc: 100
        ---

        Users:
        @User1,-lf,lf:50

        """

        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("mt-", result.prefix, "Runner prefix not correct for user")

    @patch("random.uniform", return_value=5)
    def test_opted_in_user_with_rollout_perc_two_experiments(
        self, mock_uniform: Mock
    ) -> None:
        # [FUNCTION DOC] Checks with a 100 percent lf opt-in and a 10 percent second experiment, a
        # draw of 5 enables both and the prefix is lf-.
        """User opted into lf at 100% and otherExp at 10%, random=5 -> both enabled"""
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
        ---

        Users:
        @User1,lf,otherExp:10

        """

        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for user")

    @patch("random.uniform", return_value=50)
    def test_opted_in_user_with_rollout_perc_partial_enable(
        self, mock_uniform: Mock
    ) -> None:
        # [FUNCTION DOC] Checks with a draw of 50 only the 100 percent experiment is enabled, so
        # the prefix is still lf-.
        """User opted into lf at 100% and otherExp at 10%, random=50 -> only lf enabled"""
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
        ---

        Users:
        @User1,lf,otherExp:10

        """

        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix, "Runner prefix not correct for user")

    def test_opted_in_user_with_zero_rollout_perc(self) -> None:
        # [FUNCTION DOC] Checks an opt-in at 0 percent is never enabled.
        """User opted in with 0% rollout -> never enabled"""
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
        ---

        Users:
        @User1,lf:0

        """

        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("mt-", result.prefix, "Runner prefix not correct for user")


class TestRunnerDeterminatorAmdSandboxExperiment(TestCase):
    # [CLASS DOC] Tests for the amd-sandbox experiment, which is reported through its own result
    # field (amd_sandbox_prefix) separate from the base prefix. The class constant
    # AMD_SANDBOX_SETTINGS is the shared settings text.
    AMD_SANDBOX_SETTINGS = """
        experiments:
            amd-sandbox:
                rollout_perc: 0
        ---

        Users:
        @User1,amd-sandbox
        @User2,lf

        """

    def test_amd_sandbox_opted_in_returns_prefix(self) -> None:
        # [FUNCTION DOC] Checks an opted-in user gets amd-sandbox- in amd_sandbox_prefix while the
        # base prefix stays the default mt-.
        result = rd.get_runner_prefix(self.AMD_SANDBOX_SETTINGS, ["User1"], USER_BRANCH)
        self.assertEqual("amd-sandbox-", result.amd_sandbox_prefix)
        # amd-sandbox is exposed via its own output; the base prefix is the default fleet
        self.assertEqual("mt-", result.prefix)

    def test_amd_sandbox_not_enabled_returns_default_fleet(self) -> None:
        # [FUNCTION DOC] Checks a user not enabled for amd-sandbox gets an empty
        # amd_sandbox_prefix and the default base prefix.
        # User2 opts into lf, but lf is not defined here, so it falls back to Meta
        result = rd.get_runner_prefix(self.AMD_SANDBOX_SETTINGS, ["User2"], USER_BRANCH)
        self.assertEqual("", result.amd_sandbox_prefix)
        self.assertEqual("mt-", result.prefix)

    def test_amd_sandbox_with_lf_keeps_both(self) -> None:
        # [FUNCTION DOC] Checks a user opted into both lf and amd-sandbox keeps lf- as the base
        # prefix and amd-sandbox- as the amd prefix.
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            amd-sandbox:
                rollout_perc: 0
        ---

        Users:
        @User1,lf,amd-sandbox

        """
        result = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix)
        self.assertEqual("amd-sandbox-", result.amd_sandbox_prefix)


class TestRunnerDeterminatorNoRunnerExperimentsLabel(TestCase):
    # [CLASS DOC] Tests that the no-runner-experiments label (opt-out of lf) keeps a run on the
    # default Meta fleet, both through get_runner_prefix and through main. The class constant
    # LF_ONLY is a settings text with only the lf experiment.
    """no-runner-experiments opts out of lf, so the run stays on the default Meta fleet."""

    LF_ONLY = """
        experiments:
            lf:
                rollout_perc: 0
        ---

        Users:
        @User1,lf

        """

    def test_opt_out_lf_returns_meta(self) -> None:
        # [FUNCTION DOC] Checks passing an opt-out set containing lf makes the prefix mt-.
        result = rd.get_runner_prefix(
            self.LF_ONLY,
            ["User1"],
            USER_BRANCH,
            opt_out_experiments=frozenset({"lf"}),
        )
        self.assertEqual("mt-", result.prefix)

    def test_without_opt_out_returns_lf(self) -> None:
        # [FUNCTION DOC] Checks without an opt-out set an lf user gets lf-.
        result = rd.get_runner_prefix(self.LF_ONLY, ["User1"], USER_BRANCH)
        self.assertEqual("lf-", result.prefix)

    def _run_main(self, *, labels: list[str], settings: str) -> dict[str, str]:
        # [FUNCTION DOC] HELPER: Runs runner_determinator main with parse_args, label lookup,
        # settings issue fetch and PR author lookup patched. Takes the PR labels and settings
        # text, and returns a mapping of the GitHub output keys written to their values (the
        # output setter is replaced by a capture into that mapping).
        args = Namespace(
            github_token="t",
            github_issue_repo="pytorch/test-infra",
            github_repo="pytorch/pytorch",
            github_issue=5132,
            github_actor="User1",
            github_issue_owner="User1",
            github_branch=USER_BRANCH,
            github_ref_type="branch",
            eligible_experiments=frozenset({"lf"}),
            opt_out_experiments=frozenset(),
            pr_number="123",
            workflow_name="pull",
        )
        captured: dict[str, str] = {}
        with (
            patch.object(rd, "parse_args", return_value=args),
            patch.object(rd, "get_labels", return_value=set(labels)),
            patch.object(rd, "get_rollout_state_from_issue", return_value=settings),
            patch.object(rd, "get_potential_pr_author", return_value="User1"),
            patch.object(rd, "set_github_output", side_effect=captured.__setitem__),
        ):
            rd.main()
        return captured

    def test_main_label_disables_lf_uses_meta(self) -> None:
        # [FUNCTION DOC] Checks main with the opt-out label present outputs the mt- prefix.
        out = self._run_main(labels=[rd.OPT_OUT_LABEL], settings=self.LF_ONLY)
        self.assertEqual("mt-", out[rd.GH_OUTPUT_KEY_LABEL_TYPE])

    def test_main_no_label_keeps_lf(self) -> None:
        # [FUNCTION DOC] Checks main without the label outputs the lf- prefix.
        out = self._run_main(labels=[], settings=self.LF_ONLY)
        self.assertEqual("lf-", out[rd.GH_OUTPUT_KEY_LABEL_TYPE])


if __name__ == "__main__":
    main()
