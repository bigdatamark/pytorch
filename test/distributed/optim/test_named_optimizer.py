# Owner(s): ["oncall: distributed"]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] Tests for torch.distributed.optim._NamedOptimizer, a wrapper that keys
# optimizer state by parameter name rather than index. Each test builds a reference model
# with a plain optimizer and a duplicate with the named optimizer, trains both identically,
# and compares param groups and state dicts. No import-time side effects.

import unittest

import torch
import torch.nn as nn
from torch.distributed.optim import _NamedOptimizer


def _run_model_training(model_optim_lists):
    # [FUNCTION DOC] HELPER: given a list of (model, list of optimizers) pairs, runs 2 rounds; in
    # each round draws one random 5x8 input, and for every pair does a forward pass, backward on
    # the summed output and one step of each optimizer. Gradients are not zeroed between rounds.
    for _ in range(2):
        x = torch.rand(5, 8)
        for model_optim_list in model_optim_lists:
            model = model_optim_list[0]
            optim_list = model_optim_list[1]
            y = model(x)
            y.sum().backward()
            for optim in optim_list:
                optim.step()


class TestDummyModel(torch.nn.Module):
    # [CLASS DOC] HELPER: four stage network (net1 to net4: 8 to 16 to 32 to 64 to 8 with ReLUs)
    # used as the model under test. Weights are deterministic because the constructor seeds the
    # global RNG with 0 (so every instance starts identical).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: seeds the global random generator with 0 (global side effect)
        # and creates the net1..net4 submodules.
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        # [FUNCTION DOC] HELPER: applies net1, net2, net3 and net4 in sequence and returns the
        # output.
        return self.net4(self.net3(self.net2(self.net1(x))))


class NamedOptimizerTest(unittest.TestCase):
    # [CLASS DOC] Test cases for _NamedOptimizer: state dict layout, multiple param groups,
    # loading state, adding param groups, and lazy state initialization. The comparison helper
    # methods are defined on the class.
    def _compare_state_dict_group(self, group, named_group, assert_equal=True):
        # [FUNCTION DOC] HELPER: for each key other than params in group, asserts the key exists
        # in named_group, then compares the values: tensors with allclose, others with equality.
        # When assert_equal is false it asserts the values differ instead.
        for key, val in group.items():
            if key != "params":
                self.assertTrue(
                    key in named_group, f"{key} not in named optimizer state dict"
                )
                err_msg = (
                    f"{key} state not equal" if assert_equal else f"{key} state equal"
                )
                if isinstance(val, torch.Tensor):
                    fn = self.assertTrue if assert_equal else self.assertFalse
                    fn(torch.allclose(val, named_group[key]), err_msg)
                else:
                    fn = self.assertEqual if assert_equal else self.assertNotEqual
                    fn(val, named_group[key], err_msg)

    def _compare_param_groups(self, param_groups_1, param_groups_2):
        # [FUNCTION DOC] HELPER: asserts both arguments are lists and compares corresponding
        # groups pairwise (zip, so a length mismatch is not detected).
        self.assertTrue(isinstance(param_groups_1, list))
        self.assertTrue(isinstance(param_groups_2, list))
        for groups in zip(param_groups_1, param_groups_2):
            self._compare_param_group(groups[0], groups[1])

    def _compare_param_group(self, group_1, group_2):
        # [FUNCTION DOC] HELPER: asserts both are dicts, every key of the first is in the second,
        # non params values are equal, and the params tensors match pairwise with allclose.
        self.assertTrue(isinstance(group_1, dict))
        self.assertTrue(isinstance(group_2, dict))
        for key, val in group_1.items():
            self.assertTrue(key in group_2)
            if key != "params":
                self.assertEqual(val, group_2[key])
            else:
                for tensors in zip(val, group_2[key]):
                    self.assertTrue(torch.allclose(tensors[0], tensors[1]))

    def test_state_dict(self):
        # [FUNCTION DOC] Trains a SGD with momentum optimizer and a named equivalent on two
        # identical models, then compares param groups and the per-parameter state entries,
        # looking up the named state by names such as net1.0.weight where the plain state uses
        # integer indices.
        """Check that NamedOptimizer exposes the expected state dict
        interface."""
        m = TestDummyModel()
        m_dup = TestDummyModel()
        optim = torch.optim.SGD(
            m.parameters(),
            lr=1e-2,
            momentum=0.9,
        )

        named_optim = _NamedOptimizer(
            m_dup.named_parameters(),
            torch.optim.SGD,
            lr=1e-2,
            momentum=0.9,
        )
        self._compare_param_groups(optim.param_groups, named_optim.param_groups)

        _run_model_training([(m, [optim]), (m_dup, [named_optim])])
        self._compare_param_groups(optim.param_groups, named_optim.param_groups)

        sd = optim.state_dict()
        named_sd = named_optim.state_dict()

        # Compare "state" in optim state dict
        self._compare_state_dict_group(
            sd["state"][0],
            named_sd["state"]["net1.0.weight"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd["state"][3],
            named_sd["state"]["net2.0.bias"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd["state"][4],
            named_sd["state"]["net3.weight"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd["state"][7],
            named_sd["state"]["net4.1.bias"],
            assert_equal=True,
        )

    def test_state_dict_multi_param_group(self):
        # [FUNCTION DOC] Same comparison with two optimizers (SGD and Adam), each having two
        # param groups with a per-group learning rate, comparing state and param_groups entries.
        """Check that NamedOptimizer exposes the expected state dict
        interface when multiple param groups are specified."""
        m = TestDummyModel()
        m_dup = TestDummyModel()
        optim_1 = torch.optim.SGD(
            [
                {"params": m.net1.parameters()},
                {"params": m.net3.parameters(), "lr": 1e-3},
            ],
            lr=1e-2,
            momentum=0.9,
        )

        optim_2 = torch.optim.Adam(
            [
                {"params": m.net2.parameters()},
                {"params": m.net4.parameters(), "lr": 1e-5},
            ]
        )

        named_optim_1 = _NamedOptimizer(
            m_dup.named_parameters(),
            torch.optim.SGD,
            [
                {"params": m_dup.net1.parameters()},
                {"params": m_dup.net3.parameters(), "lr": 1e-3},
            ],
            lr=1e-2,
            momentum=0.9,
        )

        named_optim_2 = _NamedOptimizer(
            m_dup.named_parameters(),
            torch.optim.Adam,
            [
                {"params": m_dup.net2.parameters()},
                {"params": m_dup.net4.parameters(), "lr": 1e-5},
            ],
        )
        self._compare_param_groups(optim_1.param_groups, named_optim_1.param_groups)
        self._compare_param_groups(optim_2.param_groups, named_optim_2.param_groups)

        _run_model_training(
            [(m, [optim_1, optim_2]), (m_dup, [named_optim_1, named_optim_2])]
        )
        self._compare_param_groups(optim_1.param_groups, named_optim_1.param_groups)
        self._compare_param_groups(optim_2.param_groups, named_optim_2.param_groups)
        sd_1 = optim_1.state_dict()
        sd_2 = optim_2.state_dict()
        named_sd_1 = named_optim_1.state_dict()
        named_sd_2 = named_optim_2.state_dict()

        # Compare "state" in optim state dict
        self._compare_state_dict_group(
            sd_1["state"][0],
            named_sd_1["state"]["net1.0.weight"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd_2["state"][1],
            named_sd_2["state"]["net2.0.bias"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd_1["state"][2],
            named_sd_1["state"]["net3.weight"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd_2["state"][3],
            named_sd_2["state"]["net4.1.bias"],
            assert_equal=True,
        )

        # Compare "param_groups" in optim state dict
        self._compare_state_dict_group(
            sd_1["param_groups"][0],
            named_sd_1["param_groups"][0],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd_2["param_groups"][1], named_sd_2["param_groups"][1], assert_equal=True
        )

    def test_load_state_dict(self):
        # [FUNCTION DOC] Two named SGD optimizers with momentum 0.9 and 0.6 are trained on one
        # model. Asserts their state differs before load, load_state_dict from the first into the
        # second, then asserts state is equal afterwards.
        """Check that NamedOptimizer's load_state_dict works as expected."""
        m = TestDummyModel()
        named_optim_1 = _NamedOptimizer(
            m.named_parameters(),
            torch.optim.SGD,
            lr=1e-2,
            momentum=0.9,
        )

        _run_model_training([(m, [named_optim_1])])
        state_dict_to_load = named_optim_1.state_dict()

        named_optim_2 = _NamedOptimizer(
            m.named_parameters(),
            torch.optim.SGD,
            lr=1e-2,
            momentum=0.6,
        )

        _run_model_training([(m, [named_optim_2])])
        state_dict_before_load = named_optim_2.state_dict()

        # Compare "state" in optim state dict
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net1.0.weight"],
            state_dict_before_load["state"]["net1.0.weight"],
            assert_equal=False,
        )
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net2.0.bias"],
            state_dict_before_load["state"]["net2.0.bias"],
            assert_equal=False,
        )
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net3.weight"],
            state_dict_before_load["state"]["net3.weight"],
            assert_equal=False,
        )
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net4.1.bias"],
            state_dict_before_load["state"]["net4.1.bias"],
            assert_equal=False,
        )

        named_optim_2.load_state_dict(state_dict_to_load)
        state_dict_after_load = named_optim_2.state_dict()

        # Compare "state" in optim state dict
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net1.0.weight"],
            state_dict_after_load["state"]["net1.0.weight"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net2.0.bias"],
            state_dict_after_load["state"]["net2.0.bias"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net3.weight"],
            state_dict_after_load["state"]["net3.weight"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net4.1.bias"],
            state_dict_after_load["state"]["net4.1.bias"],
            assert_equal=True,
        )

    def test_load_state_dict_conditional_training(self):
        # [FUNCTION DOC] Loads a state dict covering only two param groups (net1 and net3) into a
        # named optimizer over all parameters and checks the state of those parameters matches.
        """Check that NamedOptimizer load_state_dict works under conditional training case."""
        m = TestDummyModel()
        named_optim_1 = _NamedOptimizer(
            m.named_parameters(),
            torch.optim.SGD,
            [
                {"params": m.net1.parameters()},
                {"params": m.net3.parameters(), "lr": 1e-3},
            ],
            lr=1e-2,
            momentum=0.9,
        )

        _run_model_training([(m, [named_optim_1])])
        state_dict_to_load = named_optim_1.state_dict()

        named_optim_2 = _NamedOptimizer(
            m.named_parameters(),
            torch.optim.SGD,
            lr=1e-2,
            momentum=0.6,
        )

        _run_model_training([(m, [named_optim_2])])
        named_optim_2.load_state_dict(state_dict_to_load)
        state_dict_after_load = named_optim_2.state_dict()

        # Compare "state" in optim state dict
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net1.0.weight"],
            state_dict_after_load["state"]["net1.0.weight"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net3.weight"],
            state_dict_after_load["state"]["net3.weight"],
            assert_equal=True,
        )

    def test_load_state_dict_error(self):
        # [FUNCTION DOC] Loading a state dict into a named optimizer whose state was never
        # initialized (no training step) must raise ValueError with a message about the optimizer
        # being uninitialized.
        m = TestDummyModel()
        named_optim_1 = _NamedOptimizer(
            m.named_parameters(),
            torch.optim.SGD,
            lr=1e-2,
            momentum=0.9,
        )

        _run_model_training([(m, [named_optim_1])])
        state_dict_to_load = named_optim_1.state_dict()

        named_optim_2 = _NamedOptimizer(
            m.named_parameters(),
            torch.optim.SGD,
            lr=1e-2,
            momentum=0.6,
        )

        err_msg = (
            "Expects the optim to be initialized before load but found not initialized"
        )
        with self.assertRaisesRegex(ValueError, err_msg):
            named_optim_2.load_state_dict(state_dict_to_load)

    def test_add_param_group(self):
        # [FUNCTION DOC] Trains, then twice adds a new param group (net2, then net4 weight) to
        # the plain and named optimizers and trains again, comparing param groups after each
        # stage.
        m = TestDummyModel()
        m_dup = TestDummyModel()
        optim = torch.optim.SGD(
            [
                {"params": m.net1.parameters()},
                {"params": m.net3.parameters(), "lr": 1e-3},
            ],
            lr=1e-2,
            momentum=0.9,
        )
        named_optim = _NamedOptimizer(
            m_dup.named_parameters(),
            torch.optim.SGD,
            [
                {"params": m_dup.net1.parameters()},
                {"params": m_dup.net3.parameters(), "lr": 1e-3},
            ],
            lr=1e-2,
            momentum=0.9,
        )

        _run_model_training([(m, [optim]), (m_dup, [named_optim])])
        self._compare_param_groups(optim.param_groups, named_optim.param_groups)

        optim.add_param_group({"params": m.net2.parameters(), "lr": 1e-5})
        named_optim.add_param_group({"params": m_dup.net2.parameters(), "lr": 1e-5})
        _run_model_training([(m, [optim]), (m_dup, [named_optim])])
        self._compare_param_groups(optim.param_groups, named_optim.param_groups)

        optim.add_param_group({"params": m.net4[1].weight, "lr": 1e-3})
        named_optim.add_param_group({"params": m_dup.net4[1].weight, "lr": 1e-3})
        _run_model_training([(m, [optim]), (m_dup, [named_optim])])
        self._compare_param_groups(optim.param_groups, named_optim.param_groups)

    def test_add_param_group_error(self):
        # [FUNCTION DOC] Adding a group holding a tensor not belonging to the module must raise
        # ValueError about parameters not being in the module.
        m = TestDummyModel()
        named_optim = _NamedOptimizer(
            m.named_parameters(),
            torch.optim.SGD,
            [
                {"params": m.net1.parameters()},
                {"params": m.net3.parameters(), "lr": 1e-3},
            ],
            lr=1e-2,
            momentum=0.9,
        )

        err_msg = "some parameters are not in the module"
        with self.assertRaisesRegex(ValueError, err_msg):
            named_optim.add_param_group({"params": [torch.ones(8, 1)], "lr": 1e-5})

    def test_init_state(self):
        # [FUNCTION DOC] Checks the state dict is initially empty and gradients are None, calls
        # init_state, then asserts gradients exist and momentum buffers for net1 and net3
        # parameters are present and zero (not all entries nonzero).
        m = TestDummyModel()
        named_optim = _NamedOptimizer(
            m.named_parameters(),
            torch.optim.SGD,
            [
                {"params": m.net1.parameters()},
                {"params": m.net3.parameters(), "lr": 1e-3},
            ],
            lr=1e-2,
            momentum=0.9,
        )
        named_sd = named_optim.state_dict()
        self.assertTrue(m.net1[0].weight.grad is None)
        self.assertTrue(len(named_sd["state"]) == 0)
        named_optim.init_state()
        named_sd = named_optim.state_dict()
        self.assertTrue(m.net1[0].weight.grad is not None)
        self.assertTrue("momentum_buffer" in named_sd["state"]["net1.0.weight"])
        self.assertFalse(
            torch.all(named_sd["state"]["net1.0.weight"]["momentum_buffer"]).item()
        )
        self.assertFalse(
            torch.all(named_sd["state"]["net1.0.bias"]["momentum_buffer"]).item()
        )
        self.assertTrue(m.net3.bias.grad is not None)
        self.assertTrue("momentum_buffer" in named_sd["state"]["net3.bias"])
        self.assertFalse(
            torch.all(named_sd["state"]["net3.bias"]["momentum_buffer"]).item()
        )
        self.assertFalse(
            torch.all(named_sd["state"]["net3.weight"]["momentum_buffer"]).item()
        )
