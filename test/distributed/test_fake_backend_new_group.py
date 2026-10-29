# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Regression test module for dist.new_group with backend fake: a fake subgroup of a
# real (gloo) parent must not be routed through the TorchComms split_group delegation (which
# raised NotImplementedError) and must keep the hashed group name. Runs on one CPU process with
# no GPU. Import-time side effects: none beyond imports.

"""
Regression tests for: new_group() must not route a 'fake' subgroup through the
TorchComms split_group delegation.

split_group can only split the parent's existing communicator, so it cannot
produce a child whose backend differs from the parent's. A "fake" subgroup of a
real parent is exactly that case -- which is how DeviceMesh creates disabled /
unflattened mesh dimensions (with use_local_synchronization=True for hashed PG
names). Before the fix that delegation raised NotImplementedError. The fix routes
a fake subgroup of a non-fake parent through the normal path, which builds a
FakeProcessGroup directly and still produces a hashed name.

These tests use a real (gloo) parent on a single CPU process: no GPU needed.
"""

import os

import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.testing._internal.common_utils import run_tests, TestCase


class FakeBackendNewGroupTest(TestCase):
    # [CLASS DOC] Single test case class. Each test initializes a world-size-1 gloo process
    # group as the real parent and destroys it afterward; the tests build a fake subgroup of it.
    def setUp(self):
        # [FUNCTION DOC] Sets default MASTER_ADDR (127.0.0.1) and MASTER_PORT (29517) in the
        # environment if absent and initializes a gloo default process group with rank 0 and
        # world size 1.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        # Real (non-fake) parent so a fake child differs from the parent backend.
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    def tearDown(self):
        # [FUNCTION DOC] Destroys the default process group if it is initialized.
        if dist.is_initialized():
            dist.destroy_process_group()

    def _check_fake_subgroup(self):
        # [FUNCTION DOC] HELPER: Creates a subgroup with ranks [0], backend fake and local
        # synchronization, and asserts it is not None, that its backend is fake, and that the
        # group name is a 40 character non-numeric hash string (not a counter).
        g = dist.new_group(ranks=[0], backend="fake", use_local_synchronization=True)
        self.assertIsNotNone(g)
        self.assertEqual(dist.get_backend(g), "fake")
        # Hashed-name path preserved: sha1 hexdigest (40 chars), not a counter int.
        name = g.group_name
        self.assertFalse(
            name.isdigit(), lambda msg: f"{msg}\nexpected hashed name, got int {name!r}"
        )
        self.assertEqual(
            len(name),
            40,
            lambda msg: f"{msg}\nexpected 40-char sha1 hash, got {name!r}",
        )

    def test_fake_subgroup_of_real_parent_builds_fake_pg(self):
        # [FUNCTION DOC] Temporarily replaces the c10d _use_torchcomms_enabled function with one
        # returning true (restored in a finally block), then runs the fake subgroup check.
        """With TorchComms 'enabled', new_group(backend='fake',
        use_local_synchronization=True) on a real parent builds a
        FakeProcessGroup (no NotImplementedError) and keeps the hashed name."""
        # Simulate TorchComms enabled for just the new_group call, so the test
        # needs neither the torchcomms package nor a GPU.
        orig = c10d._use_torchcomms_enabled
        c10d._use_torchcomms_enabled = lambda: True
        try:
            self._check_fake_subgroup()
        finally:
            c10d._use_torchcomms_enabled = orig

    def test_fake_subgroup_real_torchcomms_flag(self):
        # [FUNCTION DOC] Sets the real use_torchcomms config flag to true and skips the test if
        # _use_torchcomms_enabled still reports false (torchcomms missing); otherwise runs the
        # fake subgroup check. The previous flag value is restored in a finally block.
        """Same as above but driving the real dist_config.use_torchcomms flag."""
        import torch.distributed.config as dist_config

        prev = dist_config.use_torchcomms
        dist_config.use_torchcomms = True
        try:
            # With the flag set, _use_torchcomms_enabled() reduces to whether the
            # torchcomms package is available; skip when it isn't.
            if not c10d._use_torchcomms_enabled():
                self.skipTest("torchcomms not available")
            self._check_fake_subgroup()
        finally:
            dist_config.use_torchcomms = prev


if __name__ == "__main__":
    run_tests()
