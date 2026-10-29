# Owner(s): ["oncall: distributed"]

# [MODULE DOC]
# Single-rank (multithreaded) tests of the module-level bookkeeping that fully_shard performs:
# retrieving the per-module state object, rejecting a second application, the dynamic class
# swap to an FSDP-prefixed subclass, rejecting container modules without forward, and
# rejecting deepcopy. No import-time side effects besides defining the test class.

import copy

import torch.nn as nn
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.testing._internal.common_fsdp import FSDPTestMultiThread, MLP
from torch.testing._internal.common_utils import run_tests


class TestFullyShardState(FSDPTestMultiThread):
    # [CLASS DOC]
    # Test case run with a single thread-simulated rank; each test applies fully_shard to a
    # small MLP-based model and checks state, class, or error behavior (no communication).
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Property telling the multithread test harness to use one rank.
        return 1

    def test_fully_shard_state(self):
        # [FUNCTION DOC] Shards three MLPs and then the parent Sequential, and checks that
        # fully_shard.state returns a non-None state for the root and that the root plus each
        # child MLP all have distinct state objects (four unique states).
        """
        Tests the ability to get the state object from a fully sharded module.
        """
        num_mlps = 3
        model = nn.Sequential(*[MLP(8) for _ in range(num_mlps)])
        for mlp in model:
            fully_shard(mlp)
        fully_shard(model)
        root_state = fully_shard.state(model)
        self.assertTrue(root_state is not None)
        all_states = [root_state] + [fully_shard.state(mlp) for mlp in model]
        # Check that each `fully_shard` call constructs a distinct state object
        self.assertEqual(len(set(all_states)), num_mlps + 1)

    def test_fully_shard_reapply(self):
        # [FUNCTION DOC] Applying fully_shard twice to the same module must raise an
        # AssertionError saying a composable distributed API can only be applied once.
        model = MLP(8)
        fully_shard(model)
        with self.assertRaisesRegex(
            AssertionError,
            "Each distinct composable distributed API can only be applied to a module once.",
        ):
            fully_shard(model)

    def test_fully_shard_cls(self):
        # [FUNCTION DOC] Checks the dynamic class swap: only the module passed to fully_shard
        # becomes an FSDPModule instance with a prefixed class name (FSDPMLP, FSDPSequential)
        # while still an instance of its original class; submodules are untouched, and
        # slicing a sharded Sequential yields a plain Sequential that is not an FSDPModule.
        # Check that we only swap class for the module passed to `fully_shard`
        model = MLP(8)
        fully_shard(model)
        self.assertTrue(isinstance(model, MLP))
        self.assertTrue(isinstance(model, FSDPModule))
        self.assertEqual(model.__class__.__name__, "FSDPMLP")
        for module in model.modules():
            if module is model:
                continue
            self.assertFalse(isinstance(module, FSDPModule))

        # Check that slicing into a `Sequential` does not preserve FSDP
        model = nn.Sequential(*[MLP(8) for _ in range(3)])
        fully_shard(model)
        self.assertTrue(isinstance(model, nn.Sequential))
        self.assertTrue(isinstance(model, FSDPModule))
        self.assertEqual(model.__class__.__name__, "FSDPSequential")
        sliced_model = model[:2]
        self.assertTrue(isinstance(sliced_model, nn.Sequential))
        self.assertFalse(isinstance(sliced_model, FSDPModule))

    def test_fully_shard_unsupported_module_cls(self):
        # [FUNCTION DOC] fully_shard on a ModuleList or ModuleDict (containers with no
        # forward) must raise a ValueError stating containers without forward are unsupported.
        regex = (
            r"fully\_shard does not support containers that do not implement forward"
        )
        model = nn.ModuleList([MLP(8) for _ in range(3)])
        with self.assertRaisesRegex(ValueError, regex):
            fully_shard(model)
        model = nn.ModuleDict({"1": MLP(8), "2": MLP(8)})
        with self.assertRaisesRegex(ValueError, regex):
            fully_shard(model)

    def test_fully_shard_deepcopy(self):
        # [FUNCTION DOC] Deep-copying a fully sharded module must raise an AssertionError
        # stating that FSDP does not support deepcopy.
        model = MLP(8)
        fully_shard(model)
        with self.assertRaisesRegex(AssertionError, "FSDP does not support deepcopy"):
            copy.deepcopy(model)


if __name__ == "__main__":
    run_tests()
