# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for the selective activation checkpointing MILP (sac_ilp): end-to-end ILP
# decisions on a Transformer using stats gathered under FakeTensorMode from MemTracker,
# RuntimeEstimator and SACEstimator with memory budgets that are binding, loose and infeasible,
# plus per-module optimal checkpointing policy on hand-made SACStats. Import-time effect: tries
# to import sac_ilp, which needs the optional pulp package; on ImportError sets the flag HAS_PULP
# to false and stubs the two functions to None so dependent tests are skipped.
import copy
import unittest

import pytest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.ilp_utils import (
    aggregate_stats,
    get_peak_memory_runtime_baseline,
    ModuleInfo,
    parse_module_info,
)
from torch.distributed._tools.mem_tracker import _ModState, MemTracker
from torch.distributed._tools.runtime_estimator import RuntimeEstimator
from torch.distributed._tools.sac_estimator import SACEstimator, SACStats
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)


# sac_ilp depends on the pulp package which may not be installed
# See: https://github.com/pytorch/pytorch/issues/162453
HAS_PULP = True
try:
    from torch.distributed._tools.sac_ilp import (
        get_optimal_checkpointing_policy_per_module,
        sac_milp,
    )
except ImportError:
    HAS_PULP = False
    get_optimal_checkpointing_policy_per_module = None  # type: ignore[assignment, misc]
    sac_milp = None  # type: ignore[assignment]


# TODO(multigpu-marker): These tests are single-process but their AC-decision
# assertions depend on RuntimeEstimator's roofline runtime estimates, which read
# the live GPU's peak FLOPS / DRAM bandwidth -- so they fail when the runner GPU
# differs (e.g. L4 vs T4). Temporarily marked multigpu to keep them on the
# original multi-GPU box until the runtime estimator is pinned to a fixed device
# spec (torch/_inductor/analysis/device_info.py) to make them hardware-independent.
@pytest.mark.multigpu
class TestSACILP(TestCase):
    # [CLASS DOC] CUDA test case (marked multigpu) that gathers module-level memory, runtime and
    # SAC statistics for a 4-layer Transformer under fake tensors, then solves the checkpointing
    # MILP for different memory budgets. The roofline device spec is pinned to an NVIDIA H100
    # string so estimates are deterministic.
    def setUp(self):
        # [FUNCTION DOC] Runs the base setup, then stores the current CUDA device index, the
        # cost-model estimate mode string, and the pinned GPU type name used for runtime
        # estimation.
        super().setUp()
        self.device = torch.cuda.current_device()
        self.estimate_mode = "operator-level-cost-model"
        # Pin the roofline device spec so runtime estimates are deterministic
        # across the physical GPU the test happens to run on.
        self.gpu_type = "NVIDIA H100"

    def _init_model_input_optimizer(
        self,
    ) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.Tensor]:
        # [FUNCTION DOC] HELPER: Builds a 4-layer 768-wide Transformer on the test device, an
        # Adam optimizer with foreach, and a random token batch of 8 sequences of length 1024.
        # Returns the tuple of model, optimizer and input.
        bsz = 8
        model_args = ModelArgs(
            n_layers=4,
            n_heads=12,
            vocab_size=8192,
            max_seq_len=1024,
            dim=768,
            dropout_p=0.1,
        )
        with torch.device(self.device):
            model = Transformer(model_args)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        inp = torch.randint(
            0, model_args.vocab_size, (bsz, model_args.max_seq_len), device=self.device
        )
        return (model, optimizer, inp)

    def _run_and_get_memTracker(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        inp: torch.Tensor,
    ) -> MemTracker:
        # [FUNCTION DOC] HELPER: Runs two training iterations of the model under a MemTracker
        # that tracks the model and optimizer externally, resetting module stats after the first
        # and capturing a snapshot of current memory during the second. Then, for each module's
        # stats lacking a post-backward snapshot (the hook is not called for modules whose inputs
        # need no gradient), adds a copy of the last snapshot. Raises AssertionError if no
        # snapshot was taken. Returns the tracker.
        mem_tracker = MemTracker()
        mem_tracker.track_external(model, optimizer)
        with mem_tracker as mt:
            for iter_idx in range(2):  # running twice to initialize optimizer
                output = model(inp)
                output.sum().backward()
                if iter_idx == 1:
                    last_snapshot = mt.get_tracker_snapshot("current")
                optimizer.step()
                optimizer.zero_grad()
                if iter_idx == 0:
                    mt.reset_mod_stats()
        if last_snapshot is None:
            raise AssertionError("Expected last_snapshot to not be None")
        for mod_stats in mem_tracker.memory_tracking.values():
            # postprocessing due to the fact that for ModTracker, the post backward hook
            # is not being called for modules whose inputs don't require gradients
            # TODO: fix this in ModTracker and ensure it does not lead to any perf regression
            if _ModState.POST_BW not in mod_stats.snapshots:
                mod_stats.snapshots.setdefault(_ModState.POST_BW, []).append(
                    copy.deepcopy(last_snapshot)
                )
        return mem_tracker

    def _run_and_get_runtime_estimator(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        inp: torch.Tensor,
    ) -> RuntimeEstimator:
        # [FUNCTION DOC] HELPER: Does one warm-up train step to initialise optimizer state, then
        # one more inside a RuntimeEstimator for the pinned GPU type and the test's estimate
        # mode. Returns the estimator.
        def _run_one_step() -> None:
            # [FUNCTION DOC] HELPER: One train step: forward, sum loss, backward, optimizer step
            # and gradient zeroing.
            output = model(inp)
            output.sum().backward()
            optimizer.step()
            optimizer.zero_grad()

        # Initializing optimizer states and warm-up
        _run_one_step()

        runtime_estimator = RuntimeEstimator(gpu_type=self.gpu_type)
        with runtime_estimator(estimate_mode_type=self.estimate_mode):
            _run_one_step()  # We use only one iteration for estimation
        return runtime_estimator

    def _run_and_get_sac_estimator(
        self,
        model: torch.nn.Module,
        inp: torch.Tensor,
    ) -> SACEstimator:
        # [FUNCTION DOC] HELPER: Runs a forward pass under a SACEstimator for the pinned GPU type
        # and estimate mode and the backward pass outside it. Returns the estimator.
        sac_estimator = SACEstimator(gpu_type=self.gpu_type)
        with sac_estimator(estimate_mode_type=self.estimate_mode):
            loss = model(inp).sum()
        loss.backward()
        return sac_estimator

    def _collect_module_info_with_fake_tensor_mode(self) -> ModuleInfo:
        # [FUNCTION DOC] HELPER: Under FakeTensorMode creates the model, optimizer and input,
        # collects the memory tracker, runtime estimator and SAC estimator results, and combines
        # them with aggregate_stats for the device. Returns the ModuleInfo.
        with FakeTensorMode():
            model, optimizer, inp = self._init_model_input_optimizer()
            mem_tracker = self._run_and_get_memTracker(model, optimizer, inp)
            runtime_estimator = self._run_and_get_runtime_estimator(
                model, optimizer, inp
            )
            sac_estimator = self._run_and_get_sac_estimator(model, inp)
            mod_info = aggregate_stats(
                model,
                mem_tracker,
                runtime_estimator,
                sac_estimator,
                torch.device(self.device),
            )
        return mod_info

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(not HAS_PULP, "pulp package not installed")
    def test_sac_ilp_case1(self):
        # [FUNCTION DOC] Budget-binding case: parses the module info into a graph, checks the
        # baseline peak memory exceeds a 1.6 GiB budget, then solves the MILP with world size 4.
        # Asserts a feasible solution (peak not minus one, positive, within the budget plus one
        # byte), positive recomputation time, non-empty decisions only for the four Transformer
        # layer modules, and each discard ratio in the interval above 0 up to 1. Skipped without
        # CUDA or pulp.
        """
        This is a case where the memory budget is either binding or too tight,
        meaning that with some AC, the model can fit into GPU memory.
        """
        mod_info = self._collect_module_info_with_fake_tensor_mode()
        g = parse_module_info(mod_info)

        memory_budget = 1.6
        budget_bytes = round(memory_budget * (2**30))

        baseline_peak_mem, _ = get_peak_memory_runtime_baseline(g)
        self.assertGreater(
            baseline_peak_mem,
            budget_bytes,
            "Test is only meaningful when the model does not fit without AC",
        )

        ac_decisions, recomputation_time, peak_mem = sac_milp(
            g, memory_budget=memory_budget, world_size=4
        )

        self.assertNotEqual(peak_mem, -1, "sac_milp failed to find a feasible solution")
        self.assertGreater(peak_mem, 0)
        self.assertLessEqual(peak_mem, budget_bytes + 1)
        self.assertGreater(recomputation_time, 0)
        self.assertGreater(len(ac_decisions), 0)

        expected_ac_candidates = {f"Transformer.layers.{i}" for i in range(4)}
        self.assertTrue(
            set(ac_decisions.keys()).issubset(expected_ac_candidates),
            lambda msg: f"{msg}\nUnexpected AC modules: "
            f"{set(ac_decisions.keys()) - expected_ac_candidates}",
        )
        for fqn, ratio in ac_decisions.items():
            self.assertGreater(
                ratio, 0, lambda msg: f"{msg}\ndiscard ratio for {fqn} should be > 0"
            )
            self.assertLessEqual(
                ratio, 1, lambda msg: f"{msg}\ndiscard ratio for {fqn} should be <= 1"
            )

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(not HAS_PULP, "pulp package not installed")
    def test_sac_ilp_case2(self):
        # [FUNCTION DOC] Loose-budget case: with a 2.4 GiB budget the MILP returns no
        # checkpointing decisions, zero recomputation time and a positive peak memory. Skipped
        # without CUDA or pulp.
        """
        This is a case where the memory budget is not binding, meaning that no
        AC is needed to fit the model into memory.
        """
        mod_info = self._collect_module_info_with_fake_tensor_mode()
        g = parse_module_info(mod_info)
        ac_decisions, recomputation_time, peak_mem = sac_milp(
            g, memory_budget=2.4, world_size=4
        )
        self.assertDictEqual(ac_decisions, {})
        self.assertEqual(recomputation_time, 0)
        self.assertGreater(peak_mem, 1)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(not HAS_PULP, "pulp package not installed")
    def test_sac_ilp_case3(self):
        # [FUNCTION DOC] Infeasible case: with a 0.8 GiB budget the MILP returns no decisions,
        # zero recomputation time and a peak of minus one marking failure. Skipped without CUDA
        # or pulp.
        """
        This is a case where the memory budget is too tight, meaning that even with
        aggressive AC, the model cannot fit into memory.
        """
        mod_info = self._collect_module_info_with_fake_tensor_mode()
        g = parse_module_info(mod_info)
        ac_decisions, recomputation_time, peak_mem = sac_milp(
            g, memory_budget=0.8, world_size=4
        )
        self.assertEqual(ac_decisions, {})
        self.assertEqual(recomputation_time, 0)
        self.assertEqual(peak_mem, -1)


class TestOptimalCheckpointingPolicy(TestCase):
    # [CLASS DOC] Test case for get_optimal_checkpointing_policy_per_module using fixed SACStats
    # adapted from xformers tests.
    # tests are adapted from tests in xformers
    # https://github.com/facebookresearch/xformers/blob/c6c0ac31f1b08542a0bc27278c6ed10f825f6963/tests/test_checkpoint.py#L222
    def setUp(self):
        # [FUNCTION DOC] Runs the base setup, then builds an eight-operator SACStats with fixed
        # function names, runtimes and memory sizes, one view-like op, one random op, two
        # in-place ops and random ops not forced to be stored, stored on the test case.
        super().setUp()
        data = [
            ("aten.copy_", 5, 0),
            ("aten.add", 5, 100),
            ("aten.div", 8, 100),
            ("aten.mm", 15, 120),
            ("aten.native_dropout", 15, 0),
            ("aten.linear", 9, 100),
            ("aten.t", 1, 0),
            ("aten.relu_", 5, 0),
        ]
        self.sac_stats = SACStats(
            func_names=[x[0] for x in data],
            runtimes=[x[1] for x in data],
            memory=[x[2] for x in data],
            view_like_ops=[6],
            rand_ops=[4],
            saved_autograd_ops=[],  # not needed for SAC decisions
            inplace_ops=[(0, 0), (7, 5)],
            force_store_random=False,
        )

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(not HAS_PULP, "pulp package not installed")
    def test_get_optimial_checkpointing_policy_per_module(self):
        # [FUNCTION DOC] For seven memory budgets given as fractions of the total memory (from 0
        # to 1), asks for the per-module optimal policy for the stats and asserts it equals a
        # fixed expected list of per-operator store or recompute flags. Skipped without CUDA or
        # pulp.
        for memory_budget, optimal_soln in [
            (0, [1, 0, 0, 0, 1, 0, 0, 0]),
            (100 / 420, [1, 0, 0, 0, 1, 1, 0, 1]),
            (120 / 420, [1, 0, 0, 1, 1, 0, 0, 0]),
            (200 / 420, [1, 0, 1, 0, 1, 1, 0, 1]),
            (220 / 420, [1, 0, 0, 1, 1, 1, 0, 1]),
            (320 / 420, [1, 0, 1, 1, 1, 1, 0, 1]),
            (420 / 420, [1, 1, 1, 1, 1, 1, 0, 1]),
        ]:
            soln = get_optimal_checkpointing_policy_per_module(
                sac_stats=self.sac_stats, memory_budget=memory_budget
            )
            self.assertEqual(optimal_soln, soln)


if __name__ == "__main__":
    run_tests()
