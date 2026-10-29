# Owner(s): ["module: sparse"]
# [MODULE DOC] Tests for BaseDataScheduler (torch.ao.pruning._experimental) driving a
# DataNormSparsifier: constructor state, warning on wrong step order, scheduled parameter
# updates and state_dict round trip. Defines ImplementedDataScheduler (halves the sparsity level
# after the first epoch) and TestBaseDataScheduler. Direct execution is rejected in favor of
# test/test_ao_sparsity.py.

import copy
import warnings

import torch
from torch import nn
from torch.ao.pruning._experimental.data_scheduler import BaseDataScheduler
from torch.ao.pruning._experimental.data_sparsifier import DataNormSparsifier
from torch.testing._internal.common_utils import raise_on_run_directly, TestCase


class ImplementedDataScheduler(BaseDataScheduler):
    # [CLASS DOC] HELPER: Concrete data scheduler for tests. After the first epoch it returns
    # half of each data group's sparsity_level; before that it returns the base parameters.
    def __init__(self, sparsifier, sparsifier_hyperparam, last_epoch=-1, verbose=False):
        # [FUNCTION DOC] HELPER: Forwards sparsifier, hyperparameter name, last_epoch and
        # verbose to the base class unchanged.
        super().__init__(sparsifier, sparsifier_hyperparam, last_epoch, verbose)

    def get_schedule_param(self):
        # [FUNCTION DOC] HELPER: Returns a mapping of data group name to half its current
        # sparsity_level when last_epoch is positive, otherwise the stored base_param mapping.
        if self.last_epoch > 0:
            return {
                name: config["sparsity_level"] * 0.5
                for name, config in self.data_sparsifier.data_groups.items()
            }
        else:
            return self.base_param


class TestBaseDataScheduler(TestCase):
    # [CLASS DOC] Tests of BaseDataScheduler behavior through ImplementedDataScheduler; the
    # _get_* methods build the data, sparsifier and scheduler fixtures.
    def _get_data(self):
        # [FUNCTION DOC] HELPER: Returns a tuple of a list of named data (a tensor, a parameter
        # and an embedding), a list with one dict holding name, data and per-data config, and a
        # defaults dict of sparsity settings.
        tensor1, param1, emb1 = (
            torch.randn(5, 5),
            nn.Parameter(torch.randn(10, 10)),
            nn.Embedding(50, 5),
        )
        data_list = [("tensor1", tensor1), ("param1", param1), ("emb1", emb1)]
        defaults = {
            "sparsity_level": 0.7,
            "sparse_block_shape": (1, 4),
            "zeros_per_block": 2,
        }
        data_with_config = [
            {
                "name": "tensor2",
                "data": torch.randn(4, 4),
                "config": {"sparsity_level": 0.3},
            }
        ]
        return data_list, data_with_config, defaults

    def _get_sparsifier(self, data_list, data_with_config, defaults):
        # [FUNCTION DOC] HELPER: Builds a DataNormSparsifier from the data list and defaults,
        # then adds each extra data entry with its own config. Returns the sparsifier.
        sparsifier = DataNormSparsifier(data_list, **defaults)
        for data_config_dict in data_with_config:
            name, data, config = (
                data_config_dict["name"],
                data_config_dict["data"],
                data_config_dict["config"],
            )
            sparsifier.add_data(name=name, data=data, **config)
        return sparsifier

    def _get_scheduler(self, sparsifier, schedule_param):
        # [FUNCTION DOC] HELPER: Returns an ImplementedDataScheduler for the sparsifier and
        # hyperparameter name.
        scheduler = ImplementedDataScheduler(sparsifier, schedule_param)
        return scheduler

    def _get_schedule_param(self):
        # [FUNCTION DOC] HELPER: Returns the scheduled hyperparameter name, sparsity_level.
        return "sparsity_level"

    def _get_name_data_config(self, some_data, defaults):
        # [FUNCTION DOC] HELPER: For a data entry (tuple or config dict) returns its name, data
        # and the effective config, i.e. a copy of the defaults updated with the entry's own
        # config if present.
        config = copy.deepcopy(defaults)
        if isinstance(some_data, tuple):
            # dealing with data_list
            name, data = some_data
        else:
            # dealing with data_with_config
            name, data, new_config = (
                some_data["name"],
                some_data["data"],
                some_data["config"],
            )
            config.update(new_config)
        return name, data, config

    def test_constructor(self):
        # [FUNCTION DOC] Checks the scheduler stores the sparsifier, starts with step count one
        # and base_param equal to each group's scheduled parameter. Raises AssertionError on
        # mismatch.
        # KNOWN ISSUE: the docstring says it checks a warning for wrong step order, but the test
        #   does not check any warning.
        """Checks if the warning is thrown if the scheduler step is called
        before the sparsifier step"""
        data_list, data_with_config, defaults = self._get_data()
        sparsifier = self._get_sparsifier(data_list, data_with_config, defaults)
        schedule_param = self._get_schedule_param()
        scheduler = self._get_scheduler(sparsifier, schedule_param)

        if scheduler.data_sparsifier != sparsifier:
            raise AssertionError("scheduler.data_sparsifier should equal sparsifier")
        if scheduler._step_count != 1:
            raise AssertionError(f"Expected _step_count=1, got {scheduler._step_count}")

        for name, config in sparsifier.data_groups.items():
            if scheduler.base_param[name] != config.get(schedule_param, None):
                raise AssertionError(
                    f"base_param[{name}] mismatch: {scheduler.base_param[name]} != {config.get(schedule_param, None)}"
                )

    def test_order_of_steps(self):
        # [FUNCTION DOC] Checks that stepping the scheduler before the sparsifier emits a
        # UserWarning, and that the correct order emits no warning from the base data scheduler
        # file.
        # KNOWN ISSUE: the filename compared in the warning check is an old torch/ao/sparsity
        #   path that does not exist for this module, so the loop can never fail.
        data_list, data_with_config, defaults = self._get_data()
        sparsifier = self._get_sparsifier(data_list, data_with_config, defaults)
        schedule_param = self._get_schedule_param()
        scheduler = self._get_scheduler(sparsifier, schedule_param)

        # Sparsifier step is not called
        with self.assertWarns(UserWarning):
            scheduler.step()

        # Correct order has no warnings
        # Note: This will trigger if other warnings are present.
        with warnings.catch_warnings(record=True) as w:
            sparsifier.step()
            scheduler.step()
            # Make sure there is no warning related to the base_data_scheduler
            for warning in w:
                fname = warning.filename
                fname = "/".join(fname.split("/")[-5:])
                if (
                    fname
                    == "torch/ao/sparsity/experimental/scheduler/data_scheduler/base_data_scheduler.py"
                ):
                    raise AssertionError("Unexpected warning from base_data_scheduler")

    def test_step(self):
        # [FUNCTION DOC] Checks that each data group starts at its configured sparsity level,
        # that one sparsifier step plus scheduler step halves it, and that the step count grows
        # to the expected total after five more steps.
        data_list, data_with_config, defaults = self._get_data()
        sparsifier = self._get_sparsifier(data_list, data_with_config, defaults)
        schedule_param = self._get_schedule_param()
        scheduler = self._get_scheduler(sparsifier, schedule_param)

        all_data = data_list + data_with_config

        for some_data in all_data:
            name, _, config = self._get_name_data_config(some_data, defaults)
            if sparsifier.data_groups[name][schedule_param] != config[schedule_param]:
                raise AssertionError(
                    f"data_groups[{name}][{schedule_param}] mismatch: "
                    f"{sparsifier.data_groups[name][schedule_param]} != {config[schedule_param]}"
                )

        sparsifier.step()
        scheduler.step()

        for some_data in all_data:
            name, _, config = self._get_name_data_config(some_data, defaults)
            expected = config[schedule_param] * 0.5
            if sparsifier.data_groups[name][schedule_param] != expected:
                raise AssertionError(
                    f"data_groups[{name}][{schedule_param}] mismatch: "
                    f"{sparsifier.data_groups[name][schedule_param]} != {expected}"
                )

        # checking step count
        step_cnt = 5
        for _ in range(step_cnt):
            sparsifier.step()
            scheduler.step()

        expected_step_count = (
            step_cnt + 2
        )  # step_cnt + step above + 1 step in constructor
        if scheduler._step_count != expected_step_count:
            raise AssertionError(
                f"Expected _step_count={expected_step_count}, got {scheduler._step_count}"
            )

    def test_state_dict(self):
        # [FUNCTION DOC] Checks that a second scheduler differs from the first after a step,
        # then that load_state_dict of the first scheduler's state makes base_param and last
        # param match.
        data_list, data_with_config, defaults = self._get_data()
        sparsifier = self._get_sparsifier(data_list, data_with_config, defaults)
        schedule_param = self._get_schedule_param()
        scheduler1 = self._get_scheduler(sparsifier, schedule_param)

        sparsifier.step()
        scheduler1.step()

        scheduler2 = self._get_scheduler(sparsifier, schedule_param)
        all_data = data_list + data_with_config
        for some_data in all_data:
            name, _, _ = self._get_name_data_config(some_data, defaults)
            if scheduler1.base_param[name] == scheduler2.base_param[name]:
                raise AssertionError(
                    f"base_param[{name}] should differ between schedulers"
                )
            if scheduler1._last_param[name] != scheduler2.base_param[name]:
                raise AssertionError(
                    f"scheduler1._last_param[{name}] should equal scheduler2.base_param[{name}]"
                )

        scheduler1_state = scheduler1.state_dict()
        scheduler2.load_state_dict(scheduler1_state)

        for some_data in all_data:
            name, _, _ = self._get_name_data_config(some_data, defaults)
            if scheduler1.base_param[name] != scheduler2.base_param[name]:
                raise AssertionError(
                    f"After load_state_dict, base_param[{name}] should match"
                )
            if scheduler1._last_param[name] != scheduler2._last_param[name]:
                raise AssertionError(
                    f"After load_state_dict, _last_param[{name}] should match"
                )


if __name__ == "__main__":
    raise_on_run_directly("test/test_ao_sparsity.py")
