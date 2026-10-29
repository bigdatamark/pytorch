# Owner(s): ["oncall: distributed"]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# [MODULE DOC] Tests for the private torch.distributed.optim helpers that apply optimizers
# during the backward pass (_apply_optimizer_in_backward, _get_in_backward_optimizers). Models
# updated in backward are compared against reference models using ordinary optimizers.
# No import-time side effects.

import unittest
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributed.optim import (
    _apply_optimizer_in_backward,
    _get_in_backward_optimizers,
)


# TODO (rohan-varma): Add FSDP & DDP tests once supported


def _validate_params(params_list, fn):
    # [FUNCTION DOC] HELPER: takes the first parameter iterable as reference and applies the
    # comparison function to each pair of corresponding parameters against every other iterable.
    # KNOWN ISSUE: if the first item is a one-shot generator and more than two iterables are
    #   given, it is exhausted after the first comparison, so later lists are not compared.
    ref_params = params_list[0]
    for param_list in params_list[1:]:
        for p1, p2 in zip(ref_params, param_list):
            fn(p1, p2)


class ApplyOverlappedOptimizerTest(unittest.TestCase):
    # [CLASS DOC] Tests comparing optimizer-in-backward models with equivalent models trained
    # with regular optimizers, plus hook registration and optimizer lookup.
    def _run_training_loop_and_validate(self, inp, models, optimizers):
        # [FUNCTION DOC] HELPER: for 6 iterations runs forward and backward of every model on
        # inp, steps every given optimizer, checks all models' parameters match with assert_close
        # (as a subtest per iteration), then zeroes gradients to None.
        for i in range(6):
            for model in models:
                model(inp).sum().backward()
            for opt in optimizers:
                opt.step()

            with self.subTest(i):
                _validate_params(
                    [model.parameters() for model in models],
                    torch.testing.assert_close,
                )

            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

    def _test_apply_optimizer_in_backward(self, share_params) -> None:
        # [FUNCTION DOC] HELPER: builds a two layer model (optionally tying both layers'
        # weights), a reference copy trained with SGD (lr 1.0 for weights, 0.5 for biases) and a
        # deep copy with the same optimizers applied in backward. Checks the initial parameters
        # match, then trains both with the shared loop and checks they stay equal.
        weight_optimizer_kwargs = {"lr": 1.0}
        bias_optimizer_kwargs = {"lr": 0.5}
        model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
        if share_params:
            model[0].weight = model[1].weight

        # Use different optimizers for weights & biases.
        weights = [m.weight for m in model]
        biases = [m.bias for m in model]
        optim_weight = torch.optim.SGD(weights, **weight_optimizer_kwargs)
        optim_bias = torch.optim.SGD(biases, **bias_optimizer_kwargs)
        model_with_opt_in_bwd = deepcopy(model)

        # Apply different optimizer in backwards for weights and biases.
        _apply_optimizer_in_backward(
            torch.optim.SGD,
            [m.weight for m in model_with_opt_in_bwd],
            optimizer_kwargs=weight_optimizer_kwargs,
        )

        _apply_optimizer_in_backward(
            torch.optim.SGD,
            [m.bias for m in model_with_opt_in_bwd],
            optimizer_kwargs=bias_optimizer_kwargs,
        )

        _validate_params(
            [
                model.parameters(),
                model_with_opt_in_bwd.parameters(),
            ],
            torch.testing.assert_close,
        )

        self._run_training_loop_and_validate(
            torch.randn(4, 10),
            [model, model_with_opt_in_bwd],
            [optim_weight, optim_bias],
        )

    def test_apply_optimizer_in_backward(self) -> None:
        # [FUNCTION DOC] Runs the shared scenario without shared parameters.
        self._test_apply_optimizer_in_backward(share_params=False)

    def test_apply_optimizer_in_backward_shared_params(self) -> None:
        # [FUNCTION DOC] Runs the shared scenario with both layers sharing one weight.
        self._test_apply_optimizer_in_backward(share_params=True)

    def test_no_register_hook(self):
        # [FUNCTION DOC] Applies SGD in backward to two copies of a model, one with the default
        # hook and one with register_hook false. After one backward pass, asserts the hooked
        # model's parameters changed from the initial ones and the unhooked model's did not.
        model_with_hook = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
        initial_model = deepcopy(model_with_hook)
        model_no_hook = deepcopy(model_with_hook)
        _apply_optimizer_in_backward(
            torch.optim.SGD,
            model_with_hook.parameters(),
            optimizer_kwargs={"lr": 0.03},
        )
        _apply_optimizer_in_backward(
            torch.optim.SGD,
            model_no_hook.parameters(),
            optimizer_kwargs={"lr": 0.03},
            register_hook=False,
        )
        inp = torch.randn(4, 10)
        model_with_hook(inp).sum().backward()
        model_no_hook(inp).sum().backward()

        for p1, p2 in zip(model_with_hook.parameters(), initial_model.parameters()):
            with self.assertRaises(AssertionError):
                torch.testing.assert_close(p1, p2)

        for p1, p2 in zip(model_no_hook.parameters(), initial_model.parameters()):
            torch.testing.assert_close(p1, p2)

    def test_multiple_optim_for_params(self) -> None:
        # [FUNCTION DOC] Applies two SGD optimizers (lr 0.03 and 0.01) in backward to the same
        # parameters and checks the result matches a model stepped with both regular optimizers.
        model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
        opt_0_kwargs = {"lr": 0.03}
        opt_1_kwargs = {"lr": 0.01}
        opt_0 = torch.optim.SGD(model.parameters(), **opt_0_kwargs)
        opt_1 = torch.optim.SGD(model.parameters(), **opt_1_kwargs)
        model_with_opt_in_bwd = deepcopy(model)
        _apply_optimizer_in_backward(
            torch.optim.SGD,
            model_with_opt_in_bwd.parameters(),
            optimizer_kwargs=opt_0_kwargs,
        )
        _apply_optimizer_in_backward(
            torch.optim.SGD,
            model_with_opt_in_bwd.parameters(),
            optimizer_kwargs=opt_1_kwargs,
        )
        self._run_training_loop_and_validate(
            torch.randn(4, 10),
            [model, model_with_opt_in_bwd],
            [opt_0, opt_1],
        )

    def test_get_optimizers_in_backward(self):
        # [FUNCTION DOC] Applies SGD in backward to a small model, then asserts the number of
        # optimizers returned by _get_in_backward_optimizers equals the number of parameters and
        # the set equals the optimizers stored on each parameter.
        # Create a simple test model
        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: two linear layer module (10 to 5 to 2) with no forward method.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates the linear1 and linear2 submodules.
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 5)
                self.linear2 = torch.nn.Linear(5, 2)

        model = TestModel()

        # Apply optimizers in backward
        _apply_optimizer_in_backward(torch.optim.SGD, model.parameters(), {"lr": 0.01})
        in_backward_optims = _get_in_backward_optimizers(model)
        self.assertEqual(len(list(model.parameters())), len(in_backward_optims))
        result = set(in_backward_optims)
        expected = {
            optim for p in model.parameters() for optim in p._in_backward_optimizers
        }
        self.assertEqual(result, expected)
