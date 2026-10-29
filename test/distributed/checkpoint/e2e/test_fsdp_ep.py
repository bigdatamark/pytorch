# Owner(s): ["oncall: distributed"]

# [MODULE DOC] 8-GPU test of FSDP combined with expert-parallel style placement: some
# submodules are wrapped by FSDP on a per-expert 2-rank mesh while the rest use a mesh of
# all 8 ranks. Verifies that get_state_dict returns DTensors on the expected meshes and only
# on the owning ranks. Defines small dummy models and one test class; no import-time effects.

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import get_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir
from torch.testing._internal.distributed.common_state_dict import VerifyStateDictMixin


class Dummymodel(nn.Module):
    # [CLASS DOC] HELPER: Parameter-free placeholder for an expert layer not owned by this rank.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes an empty module.
        super().__init__()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Not usable; always raises NotImplementedError.
        raise NotImplementedError


class EPModel(nn.Module):
    # [CLASS DOC] HELPER: Expert layer with two Linear(16,16)+ReLU sequential blocks, net1
    # and net2. Only instantiated on the rank that owns the expert.
    def __init__(self, rank):
        # [FUNCTION DOC] HELPER: Builds net1 and net2; the rank argument is unused.
        super().__init__()
        self.net1 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Not usable; always raises NotImplementedError.
        raise NotImplementedError


class SecondTier(nn.Module):
    # [CLASS DOC] HELPER: Holds a list of four expert slots (a real EPModel at index equal
    # to rank modulo 4, Dummymodel elsewhere) and a shared Linear+ReLU block named net.
    def __init__(self, rank):
        # [FUNCTION DOC] HELPER: Builds ep_layers selecting the real expert by rank modulo 4,
        # and the net block.
        super().__init__()
        self.ep_layers = nn.ModuleList(
            [EPModel(rank) if rank % 4 == i else Dummymodel() for i in range(4)]
        )
        self.net = nn.Sequential(nn.Linear(16, 16), nn.ReLU())

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Not usable; always raises NotImplementedError.
        raise NotImplementedError


class TopModel(nn.Module):
    # [CLASS DOC] HELPER: Top-level model with a SecondTier child named second and a
    # Linear+ReLU block named net.
    def __init__(self, rank):
        # [FUNCTION DOC] HELPER: Seeds the global RNG with 0 (side effect) so initial weights
        # are identical across ranks, then builds second and net.
        super().__init__()
        torch.manual_seed(0)

        self.second = SecondTier(rank)
        self.net = nn.Sequential(nn.Linear(16, 16), nn.ReLU())

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Not usable; always raises NotImplementedError.
        raise NotImplementedError


class TestFSDPWithEP(DTensorTestBase, VerifyStateDictMixin):
    # [CLASS DOC] Multi-process test case for FSDP plus per-expert sharding state dicts.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Number of processes: the accelerator count capped at 8.
        return min(8, torch.accelerator.device_count())

    @with_comms
    @skip_if_lt_x_gpu(8)
    @with_temp_dir
    def test_e2e(self):
        # [FUNCTION DOC] Wraps each expert slot in FSDP over the data-parallel submesh of
        # a 2x4 mesh (with its root mesh link cleared), then wraps the second tier and the
        # whole model over an 8-rank mesh, and obtains model and Adam optimizer state dicts.
        # Asserts the always-sharded params (net and second.net) are DTensors on all 8 ranks,
        # that expert params exist only on the rank owning that expert and live on the
        # two-rank mesh containing this rank and its peer, and that optimizer state keys
        # equal model state keys. Needs 8 GPUs.
        model = TopModel(self.rank).to(self.device_type)

        mesh_fsdp_tp = init_device_mesh(
            self.device_type, (2, 4), mesh_dim_names=("dp", "tp")
        )
        # TODO: we are using an internal API atm. Change to a public API once it is ready.
        mesh_fsdp_ep = mesh_fsdp_tp["dp"]
        mesh_fsdp_ep._root_mesh = None

        mesh_fsdp = init_device_mesh(self.device_type, (8,))
        for i, l in enumerate(model.second.ep_layers):
            model.second.ep_layers[i] = FSDP(
                l, use_orig_params=True, device_mesh=mesh_fsdp_ep
            )
        model.second = FSDP(model.second, use_orig_params=True, device_mesh=mesh_fsdp)
        model = FSDP(model, use_orig_params=True, device_mesh=mesh_fsdp)
        optim = torch.optim.Adam(model.parameters(), lr=0.1)
        msd, osd = get_state_dict(model, optim)

        # FSDP only params
        for key in (
            "net.0.weight",
            "net.0.bias",
            "second.net.0.weight",
            "second.net.0.bias",
        ):
            msd_v = msd[key]
            osd_v = osd["state"][key]["exp_avg"]
            for v in (msd_v, osd_v):
                self.assertTrue(isinstance(v, DTensor))
                self.assertEqual(tuple(v.device_mesh.mesh), tuple(range(8)))

        # FSDP/EP params
        layer = self.rank % 4
        ranks = (layer, layer + 4)
        for i in range(4):
            for key in (
                f"second.ep_layers.{i}.net1.0.weight",
                f"second.ep_layers.{i}.net1.0.bias",
                f"second.ep_layers.{i}.net2.0.weight",
                f"second.ep_layers.{i}.net2.0.bias",
            ):
                if layer != i:
                    self.assertTrue(key not in msd)
                else:
                    msd_v = msd[key]
                    osd_v = osd["state"][key]["exp_avg"]
                    for v in (msd_v, osd_v):
                        self.assertTrue(isinstance(v, DTensor))
                        self.assertEqual(tuple(v.device_mesh.mesh), ranks)

        self.assertEqual(set(osd["state"].keys()), set(msd.keys()))


if __name__ == "__main__":
    run_tests()
