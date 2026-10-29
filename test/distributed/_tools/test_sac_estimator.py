# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for SACEstimator (selective activation checkpointing estimator) under
# FakeTensorMode on CUDA: runs the estimator in both benchmark and cost-model modes on a
# Transformer, and checks that view-like, random and in-place operators are classified correctly
# on a tiny model.
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.sac_estimator import SACEstimator
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)


class TestSACEstimator(TestCase):
    # [CLASS DOC] CUDA test case for SACEstimator.
    def _sac_estimation(
        self,
        estimate_mode: str,
        model: torch.nn.Module,
        inp: torch.Tensor,
    ):
        # [FUNCTION DOC] HELPER: Runs a forward pass under a new SACEstimator in the given
        # estimate mode, then the backward pass, and asks the estimator for a piecewise-linear
        # trade-off curve with 2 segments without saving graphs.
        sace = SACEstimator()
        with sace(estimate_mode_type=estimate_mode):
            loss = model(inp).sum()
        loss.backward()
        sace.pwlf_sac_tradeoff_curve(n_segments=2, save_tradeoff_graphs=False)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_transformer_sac_estimation(self):
        # [FUNCTION DOC] Builds a 4-layer 768-wide Transformer and random token input under
        # FakeTensorMode and runs the estimation in operator-level-benchmark and
        # operator-level-cost-model modes. Only checks that it completes. Skipped without CUDA.
        """Runs a basic GPT-2 model"""
        dev = torch.cuda.current_device()
        vocab_size = 8192
        bsz, seq_len = 8, 1024
        model_args = ModelArgs(
            n_layers=4,
            n_heads=12,
            vocab_size=vocab_size,
            max_seq_len=seq_len,
            dim=768,
            dropout_p=0.1,
        )
        with FakeTensorMode():
            with torch.device(dev):
                model = Transformer(model_args)
            inp = torch.randint(
                0, model_args.vocab_size, (bsz, model_args.max_seq_len), device=dev
            )

            self._sac_estimation("operator-level-benchmark", model, inp)
            self._sac_estimation("operator-level-cost-model", model, inp)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_simple_model_sac_estimation(self):
        # [FUNCTION DOC] Runs a fully-connected plus in-place relu, cos and sin model under the
        # estimator in benchmark mode and asserts the recorded stats for the model: view-like ops
        # are operator index 0, there are no random ops, and the in-place ops are exactly the
        # three pairs of operator index and 1 (indexes 2, 3 and 4). Skipped without CUDA.
        """This test checks the correctness of view_ops, random_ops and inplace_ops"""

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Tiny model with a linear layer and in-place relu, followed by
            # in-place cos and sin in the forward pass.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates a linear layer (5 to 10) and an in-place relu.
                super().__init__()
                self.fc1 = torch.nn.Linear(5, 10)
                self.relu1 = torch.nn.ReLU(inplace=True)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer, relu, in-place cos and
                # in-place sin, and returns the result.
                x = self.fc1(x)
                x = self.relu1(x)
                x = torch.cos_(x)
                x = torch.sin_(x)
                return x

        dev = torch.cuda.current_device()
        with FakeTensorMode():
            with torch.device(dev):
                model = Foo()
            x = torch.rand((10, 5), device=dev)

            sac_estimator = SACEstimator()
            with sac_estimator(estimate_mode_type="operator-level-benchmark"):
                loss = model(x).sum()
            loss.backward()

            self.assertEqual(sac_estimator.sac_mod_stats["Foo"].view_like_ops, [0])
            self.assertEqual(sac_estimator.sac_mod_stats["Foo"].rand_ops, [])
            self.assertEqual(
                sac_estimator.sac_mod_stats["Foo"].inplace_ops, [(2, 1), (3, 1), (4, 1)]
            )


if __name__ == "__main__":
    run_tests()
