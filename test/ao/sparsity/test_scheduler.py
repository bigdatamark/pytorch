# Owner(s): ["module: sparse"]
# [MODULE DOC] Tests for sparsity schedulers (BaseScheduler, LambdaSL, CubicSL) driving a
# WeightNormSparsifier. Defines ImplementedScheduler, TestScheduler and TestCubicScheduler.
# Direct execution is rejected in favor of test/test_ao_sparsity.py.

import warnings

from torch import nn
from torch.ao.pruning import BaseScheduler, CubicSL, LambdaSL, WeightNormSparsifier
from torch.testing._internal.common_utils import raise_on_run_directly, TestCase


class ImplementedScheduler(BaseScheduler):
    # [CLASS DOC] HELPER: Concrete BaseScheduler halving each group's sparsity level after the
    # first epoch.
    def get_sl(self):
        # [FUNCTION DOC] HELPER: Returns a list of half of each group's sparsity_level if
        # last_epoch is positive, otherwise a copy of base_sl.
        if self.last_epoch > 0:
            return [group["sparsity_level"] * 0.5 for group in self.sparsifier.groups]
        else:
            return list(self.base_sl)


class TestScheduler(TestCase):
    # [CLASS DOC] Tests of BaseScheduler behavior and LambdaSL.
    def test_constructor(self):
        # [FUNCTION DOC] Checks the scheduler references the sparsifier, starts at step count
        # one and base_sl holds the group's sparsity level.
        model = nn.Sequential(nn.Linear(16, 16))
        sparsifier = WeightNormSparsifier()
        sparsifier.prepare(model, config=None)
        scheduler = ImplementedScheduler(sparsifier)

        if scheduler.sparsifier is not sparsifier:
            raise AssertionError("scheduler.sparsifier should be sparsifier")
        if scheduler._step_count != 1:
            raise AssertionError(f"Expected _step_count=1, got {scheduler._step_count}")
        if scheduler.base_sl != [sparsifier.groups[0]["sparsity_level"]]:
            raise AssertionError(
                f"Expected base_sl={[sparsifier.groups[0]['sparsity_level']]}, got {scheduler.base_sl}"
            )

    def test_order_of_steps(self):
        # [FUNCTION DOC] Checks stepping the scheduler before the sparsifier warns, and the
        # correct order produces no warning attributed to the base scheduler file.
        # KNOWN ISSUE: the compared warning path is an old torch/ao/sparsity location, so the
        #   no-warning check can never fail.
        """Checks if the warning is thrown if the scheduler step is called
        before the sparsifier step"""

        model = nn.Sequential(nn.Linear(16, 16))
        sparsifier = WeightNormSparsifier()
        sparsifier.prepare(model, config=None)
        scheduler = ImplementedScheduler(sparsifier)

        # Sparsifier step is not called
        with self.assertWarns(UserWarning):
            scheduler.step()

        # Correct order has no warnings
        # Note: This will trigger if other warnings are present.
        with warnings.catch_warnings(record=True) as w:
            sparsifier.step()
            scheduler.step()
            # Make sure there is no warning related to the base_scheduler
            for warning in w:
                fname = warning.filename
                fname = "/".join(fname.split("/")[-5:])
                if fname == "torch/ao/sparsity/scheduler/base_scheduler.py":
                    raise AssertionError("Unexpected warning from base_scheduler")

    def test_step(self):
        # [FUNCTION DOC] Checks the sparsity level stays 0.5 after creating the scheduler and is
        # halved to 0.25 after a sparsifier and scheduler step.
        model = nn.Sequential(nn.Linear(16, 16))
        sparsifier = WeightNormSparsifier()
        sparsifier.prepare(model, config=None)
        if sparsifier.groups[0]["sparsity_level"] != 0.5:
            raise AssertionError(
                f"Expected sparsity_level=0.5, got {sparsifier.groups[0]['sparsity_level']}"
            )
        scheduler = ImplementedScheduler(sparsifier)
        if sparsifier.groups[0]["sparsity_level"] != 0.5:
            raise AssertionError(
                f"Expected sparsity_level=0.5 after scheduler init, got {sparsifier.groups[0]['sparsity_level']}"
            )

        sparsifier.step()
        scheduler.step()
        if sparsifier.groups[0]["sparsity_level"] != 0.25:
            raise AssertionError(
                f"Expected sparsity_level=0.25 after step, got {sparsifier.groups[0]['sparsity_level']}"
            )

    def test_lambda_scheduler(self):
        # [FUNCTION DOC] Checks LambdaSL: level starts at 0.0 (function of epoch 0 times 10) and
        # becomes 5.0 after one step (base level times epoch factor).
        model = nn.Sequential(nn.Linear(16, 16))
        sparsifier = WeightNormSparsifier()
        sparsifier.prepare(model, config=None)
        if sparsifier.groups[0]["sparsity_level"] != 0.5:
            raise AssertionError(
                f"Expected sparsity_level=0.5, got {sparsifier.groups[0]['sparsity_level']}"
            )
        scheduler = LambdaSL(sparsifier, lambda epoch: epoch * 10)
        if sparsifier.groups[0]["sparsity_level"] != 0.0:  # Epoch 0
            raise AssertionError(
                f"Expected sparsity_level=0.0 at epoch 0, got {sparsifier.groups[0]['sparsity_level']}"
            )
        scheduler.step()
        if sparsifier.groups[0]["sparsity_level"] != 5.0:  # Epoch 1
            raise AssertionError(
                f"Expected sparsity_level=5.0 at epoch 1, got {sparsifier.groups[0]['sparsity_level']}"
            )


class TestCubicScheduler(TestCase):
    # [CLASS DOC] Tests of CubicSL scheduler against a two-layer sparsifier config with target
    # levels 0.8 and 0.4.
    def setUp(self):
        # [FUNCTION DOC] Stores the sparse config, its sorted levels, the initial sparsity (0.1)
        # and initial step (3).
        super().setUp()
        self.model_sparse_config = [
            {"tensor_fqn": "0.weight", "sparsity_level": 0.8},
            {"tensor_fqn": "2.weight", "sparsity_level": 0.4},
        ]
        self.sorted_sparse_levels = [
            conf["sparsity_level"] for conf in self.model_sparse_config
        ]
        self.initial_sparsity = 0.1
        self.initial_step = 3

    def _make_model(self, **kwargs):
        # [FUNCTION DOC] HELPER: Returns a Sequential of linear, dropout and linear; extra
        # keywords are ignored.
        model = nn.Sequential(
            nn.Linear(13, 17),
            nn.Dropout(0.5),
            nn.Linear(17, 3),
        )
        return model

    def _make_scheduler(self, model, **kwargs):
        # [FUNCTION DOC] HELPER: Prepares a WeightNormSparsifier on the model with the stored
        # config and builds a CubicSL with the initial level and step plus keyword overrides.
        # Returns (sparsifier, scheduler).
        sparsifier = WeightNormSparsifier()
        sparsifier.prepare(model, config=self.model_sparse_config)

        scheduler_args = {
            "init_sl": self.initial_sparsity,
            "init_t": self.initial_step,
        }
        scheduler_args.update(kwargs)

        scheduler = CubicSL(sparsifier, **scheduler_args)
        return sparsifier, scheduler

    @staticmethod
    def _get_sparsity_levels(sparsifier, precision=32):
        # [FUNCTION DOC] HELPER: Static. Returns the sparsity level of each sparsifier group
        # rounded to the given precision.
        r"""Gets the current levels of sparsity in a sparsifier."""
        return [
            round(group["sparsity_level"], precision) for group in sparsifier.groups
        ]

    def test_constructor(self):
        # [FUNCTION DOC] Checks scheduler attachment, step count one, stored target levels, and
        # that the initial sparsity is zero or the initial level depending on initially_zero.
        model = self._make_model()
        sparsifier, scheduler = self._make_scheduler(model=model, initially_zero=True)
        self.assertIs(
            scheduler.sparsifier, sparsifier, msg="Sparsifier is not properly attached"
        )
        self.assertEqual(
            scheduler._step_count,
            1,
            msg="Scheduler is initialized with incorrect step count",
        )
        self.assertEqual(
            scheduler.base_sl,
            self.sorted_sparse_levels,
            msg="Scheduler did not store the target sparsity levels correctly",
        )

        # Value before t_0 is 0
        self.assertEqual(
            self._get_sparsity_levels(sparsifier),
            scheduler._make_sure_a_list(0.0),
            msg="Sparsifier is not reset correctly after attaching to the Scheduler",
        )

        # Value before t_0 is s_0
        model = self._make_model()
        sparsifier, scheduler = self._make_scheduler(model=model, initially_zero=False)
        self.assertEqual(
            self._get_sparsity_levels(sparsifier),
            scheduler._make_sure_a_list(self.initial_sparsity),
            msg="Sparsifier is not reset correctly after attaching to the Scheduler",
        )

    def test_step(self):
        # For n=5, dt=2, there will be totally 10 steps between s_0 and s_f, starting from t_0
        # [FUNCTION DOC] Steps a CubicSL and checks levels stay zero before the initial step,
        # equal the initial level at it, follow the cubic ramp after, and reach the targets
        # after the total steps.
        model = self._make_model()
        sparsifier, scheduler = self._make_scheduler(
            model=model, initially_zero=True, init_t=3, delta_t=2, total_t=5
        )

        scheduler.step()
        scheduler.step()
        self.assertEqual(
            scheduler._step_count,
            3,
            msg="Scheduler step_count is expected to increment",
        )
        # Value before t_0 is supposed to be 0
        self.assertEqual(
            self._get_sparsity_levels(sparsifier),
            scheduler._make_sure_a_list(0.0),
            msg="Scheduler step updating the sparsity level before t_0",
        )

        scheduler.step()  # Step = 3  =>  sparsity = initial_sparsity
        self.assertEqual(
            self._get_sparsity_levels(sparsifier),
            scheduler._make_sure_a_list(self.initial_sparsity),
            msg="Sparsifier is not reset to initial sparsity at the first step",
        )

        scheduler.step()  # Step = 4  =>  sparsity ~ [0.3, 0.2]
        self.assertEqual(
            self._get_sparsity_levels(sparsifier, 1),
            [0.3, 0.2],
            msg="Sparsity level is not set correctly after the first step",
        )

        current_step = scheduler._step_count - scheduler.init_t[0] - 1
        more_steps_needed = scheduler.delta_t[0] * scheduler.total_t[0] - current_step
        for _ in range(more_steps_needed):  # More steps needed to final sparsity level
            scheduler.step()
        self.assertEqual(
            self._get_sparsity_levels(sparsifier),
            self.sorted_sparse_levels,
            msg="Sparsity level is not reaching the target level after delta_t * n steps ",
        )


if __name__ == "__main__":
    raise_on_run_directly("test/test_ao_sparsity.py")
