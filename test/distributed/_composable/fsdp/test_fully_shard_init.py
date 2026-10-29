# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for fully_shard (FSDP2) initialization. Covers moving
# parameters/buffers to the accelerator (plain and DTensor), container subclasses
# with forward, mesh argument validation, managed modules/states discovery,
# ParamModuleInfo construction, sharded parameter layout (plain and 2D DTensor),
# lazy init and root detection, meta-device init and materialization, process
# group based meshes (1D and HSDP), HSDP broadcast, custom all-reduce hooks,
# shard_placement_fn, the old import path, and mixed/non-float dtype params.
# Most classes use the multi-threaded test harness; two use real processes.
# Only sets device_type at import time.

import copy
import itertools
from typing import cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import replicate
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard._fsdp_init import (
    _get_managed_modules,
    _get_managed_states,
)
from torch.distributed.fsdp._fully_shard._fsdp_param import ParamModuleInfo
from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
    _get_param_module_infos,
)
from torch.distributed.fsdp._fully_shard._fully_shard import FSDPModule
from torch.distributed.fsdp._init_utils import (
    _init_inter_node_process_group,
    _init_intra_node_process_group,
)
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    FSDPTestMultiThread,
    get_devtype,
    MLP,
    patch_all_gather,
    patch_reduce_scatter,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)


device_type = torch.device(get_devtype())


class TestFullyShardDeviceTensor(FSDPTestMultiThread):
    # [CLASS DOC] Single-rank threaded tests that fully_shard moves plain
    # (non-DTensor) parameters and buffers to the accelerator.
    """Tests that tensor parameters are moved to the expected device."""

    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 1, the number of simulated ranks.
        return 1

    @skip_if_lt_x_gpu(1)
    def test_move_states_to_device_tensor(self):
        # [FUNCTION DOC] Builds a CPU MLP with a buffer, checks everything is on
        # CPU, applies fully_shard, and asserts every parameter and buffer is
        # now on the current accelerator device.
        model = MLP(8, torch.device("cpu"), with_buffer=True)
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            self.assertEqual(tensor.device, torch.device("cpu"))
        fully_shard(model)
        accelerator_device = torch.device(
            device_type.type, torch.get_device_module(device_type).current_device()
        )
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            self.assertEqual(tensor.device, accelerator_device)

    @skip_if_lt_x_gpu(1)
    def test_move_states_to_device_ignored_param_device(self):
        # [FUNCTION DOC] Shards a CPU MLP while ignoring the out_proj weight and
        # bias. Asserts the ignored parameters stay on CPU after fully_shard, and
        # move to the accelerator only after an explicit model.to call.
        cpu_device = torch.device("cpu")
        model = MLP(8, cpu_device, with_buffer=True)
        ignored_params = [model.out_proj.weight, model.out_proj.bias]
        fully_shard(model, ignored_params=set(ignored_params))
        for tensor in ignored_params:
            self.assertEqual(tensor.device, cpu_device)
        accelerator_device = torch.device(
            device_type.type, torch.get_device_module(device_type).current_device()
        )
        model.to(device_type)
        for tensor in ignored_params:
            self.assertEqual(tensor.device, accelerator_device)


class TestFullyShardDeviceDTensor(FSDPTestMultiThread):
    # [CLASS DOC] Four-rank threaded tests of device placement for models whose
    # parameters are DTensors from tensor parallelism, before FSDP sharding.
    """Tests that DTensor parameters are moved to the expected device."""

    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 4, the number of simulated ranks.
        return 4

    @skip_if_lt_x_gpu(1)
    def test_move_states_to_device_dtensor_valid(self):
        # [FUNCTION DOC] Applies tensor parallelism on a 2x2 (dp, tp) accelerator
        # mesh to a CPU MLP, checks DTensor params (and local shards) are already
        # on the accelerator while plain buffers stay on CPU, then applies
        # fully_shard on the dp mesh and asserts all states end up on the
        # accelerator. Raises AssertionError if world size is below 4.
        if not (self.world_size >= 4):
            raise AssertionError(f"Expected world_size >= 4, but got {self.world_size}")
        dp_size = 2
        global_mesh = init_device_mesh(
            device_type.type,
            (dp_size, self.world_size // dp_size),
            mesh_dim_names=("dp", "tp"),
        )
        dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]
        model = MLP(8, torch.device("cpu"), with_buffer=True)
        parallelize_module(
            model,
            tp_mesh,
            {"in_proj": ColwiseParallel(), "out_proj": RowwiseParallel()},
        )
        accelerator_device = torch.device(
            device_type.type, torch.get_device_module(device_type).current_device()
        )
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            if isinstance(tensor, DTensor):
                # DTensor constructor moves to the mesh's device
                self.assertEqual(tensor.device, accelerator_device)
                self.assertEqual(tensor._local_tensor.device, accelerator_device)
            else:
                self.assertEqual(tensor.device, torch.device("cpu"))
        fully_shard(model, mesh=dp_mesh)
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            self.assertEqual(tensor.device, accelerator_device)
            if isinstance(tensor, DTensor):
                self.assertEqual(tensor._local_tensor.device, accelerator_device)

    @skip_if_lt_x_gpu(1)
    def test_move_states_to_device_dtensor_invalid(self):
        # [FUNCTION DOC] Applies tensor parallelism using a CPU tp mesh but passes
        # an accelerator dp mesh to fully_shard. Asserts all states are on CPU
        # beforehand and that fully_shard raises ValueError saying the DTensor
        # mesh type (cpu) differs from the FSDP mesh type.
        if not (self.world_size >= 4):
            raise AssertionError(f"Expected world_size >= 4, but got {self.world_size}")
        dp_size = 2
        global_accelerator_mesh = init_device_mesh(
            device_type.type,
            (dp_size, self.world_size // dp_size),
            mesh_dim_names=("dp", "tp"),
        )
        global_cpu_mesh = init_device_mesh(
            "cpu", (dp_size, self.world_size // dp_size), mesh_dim_names=("dp", "tp")
        )
        dp_mesh = global_accelerator_mesh["dp"]
        tp_mesh = global_cpu_mesh["tp"]  # mismatched meshes!
        model = MLP(8, torch.device("cpu"), with_buffer=True)
        parallelize_module(
            model,
            tp_mesh,
            {"in_proj": ColwiseParallel(), "out_proj": RowwiseParallel()},
        )
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            self.assertEqual(tensor.device, torch.device("cpu"))
            if isinstance(tensor, DTensor):
                self.assertEqual(tensor._local_tensor.device, torch.device("cpu"))
        regex = (
            rf"Requires DTensor to have mesh of the same type as the FSDP mesh but got "
            rf"cpu for DTensor and {device_type.type} for FSDP"
        )
        with self.assertRaisesRegex(ValueError, regex):
            fully_shard(model, mesh=dp_mesh)


class TestFullyShardContainerSubclasses(FSDPTestMultiThread):
    # [CLASS DOC] Single-rank threaded tests that container module subclasses
    # which define forward are accepted by fully_shard. Nests two container
    # subclasses used as test models.
    """Tests that fully_shard accepts ModuleDict/ModuleList subclasses that implement forward()."""

    class DictWithForward(nn.ModuleDict):
        # [CLASS DOC] HELPER: ModuleDict subclass holding one Linear under key
        # lin and defining forward to apply it.
        def __init__(self, in_features=8, out_features=8):
            # [FUNCTION DOC] HELPER: Creates the dict with a single Linear entry.
            super().__init__({"lin": nn.Linear(in_features, out_features)})

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Returns the lin entry applied to x.
            return self["lin"](x)

    class ListWithForward(nn.ModuleList):
        # [CLASS DOC] HELPER: ModuleList subclass holding one Linear and defining
        # forward to apply each element in order.
        def __init__(self, in_features=8, out_features=8):
            # [FUNCTION DOC] HELPER: Creates the list with a single Linear.
            super().__init__([nn.Linear(in_features, out_features)])

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Applies each contained module to the running
            # value in order and returns the final output.
            out = x
            for m in self:
                out = m(out)
            return out

    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 1, the number of simulated ranks.
        return 1

    @skip_if_lt_x_gpu(1)
    def test_moduledict_subclass_with_forward(self):
        # [FUNCTION DOC] Shards a ModuleDict subclass with forward on a 1D mesh,
        # expecting no error from the container type, then runs a forward.
        model = self.DictWithForward(8, 8)
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        # Should not raise due to container type since forward() is implemented
        fsdp_model = fully_shard(model, mesh=mesh)
        x = torch.randn(2, 8, device=device_type)
        _ = fsdp_model(x)

    @skip_if_lt_x_gpu(1)
    def test_modulelist_subclass_with_forward(self):
        # [FUNCTION DOC] Same as the ModuleDict case but for a ModuleList
        # subclass with forward: shard on a 1D mesh and run one forward.
        model = self.ListWithForward(8, 8)
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        fsdp_model = fully_shard(model, mesh=mesh)
        x = torch.randn(2, 8, device=device_type)
        _ = fsdp_model(x)


class TestFullyShardMeshArg(FSDPTestMultiThread):
    # [CLASS DOC] Four-rank threaded tests of mesh argument validation in
    # fully_shard.
    """Tests the ``mesh`` argument."""

    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 4, the number of simulated ranks.
        return 4

    def test_invalid_mesh_ndim(self):
        # [FUNCTION DOC] Passes a 3D mesh and asserts ValueError stating that a 1D
        # or 2D mesh is expected.
        mesh = init_device_mesh(device_type.type, (self.world_size, 1, 1))
        model = MLP(8)
        regex = r"fully\_shard expects a 1D or 2D DeviceMesh but got DeviceMesh"
        with self.assertRaisesRegex(ValueError, regex):
            fully_shard(model, mesh=mesh)

    def test_2d_mesh_without_mesh_dim_names(self):
        # [FUNCTION DOC] Passes a 2D mesh lacking dim names and asserts an
        # AssertionError asking for mesh_dim_names to be specified for HSDP.
        mesh = init_device_mesh(device_type.type, (self.world_size // 2, 2))
        model = MLP(8)
        regex = "Please init the 2D mesh for HSDP with mesh_dim_names specified"
        with self.assertRaisesRegex(AssertionError, regex):
            fully_shard(model, mesh=mesh)


class TestFullyShardManagedModulesAndStates(FSDPTestMultiThread):
    # [CLASS DOC] Single-rank threaded tests of the private helpers that compute
    # which modules, parameters, and buffers one fully_shard call manages.
    """Tests getting the managed modules/states for a ``fully_shard`` module."""

    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 1, the number of simulated ranks.
        return 1

    def test_managed_modules_single(self):
        # [FUNCTION DOC] For one root module, expects the managed modules to be
        # all of its submodules including itself.
        model = MLP(8)
        # Assume calling `fully_shard` on `model`
        managed_modules = _get_managed_modules((model,))
        expected_managed_modules = list(model.modules())
        self._check_managed_modules(managed_modules, expected_managed_modules)

    def test_managed_modules_nested(self):
        # [FUNCTION DOC] With the first child already fully_shard-ed, expects the
        # root's managed modules to exclude that child's subtree.
        model = nn.Sequential(*[MLP(8) for _ in range(2)])
        fully_shard(model[0])
        # Assume calling `fully_shard` on `model`
        managed_modules = _get_managed_modules((model,))
        expected_managed_modules = list(model[1].modules()) + [model]
        self._check_managed_modules(managed_modules, expected_managed_modules)

    def test_managed_modules_nested_fully_shard_and_replicate(self):
        # [FUNCTION DOC] With one child under replicate and another under
        # fully_shard, expects the root's managed modules to exclude both
        # subtrees and include only the middle child and the root.
        model = nn.Sequential(*[MLP(8) for _ in range(3)])
        replicate(model[0])
        fully_shard(model[2])
        # Assume calling `fully_shard` on `model`
        managed_modules = _get_managed_modules((model,))
        expected_managed_modules = list(model[1].modules()) + [model]
        self._check_managed_modules(managed_modules, expected_managed_modules)

    def test_managed_modules_duplicate(self):
        # [FUNCTION DOC] When the same MLP appears twice in a Sequential, expects
        # the shared module to be counted once in the managed modules.
        mlp = MLP(8)
        model = nn.Sequential(mlp, mlp)  # duplicate MLP
        # Assume calling `fully_shard` on `model`
        managed_modules = _get_managed_modules((model,))
        # Check that the duplicate module is only counted once
        expected_managed_modules = list(mlp.modules()) + [model]
        self._check_managed_modules(managed_modules, expected_managed_modules)

    def test_managed_modules_list_of_mlps(self):
        # [FUNCTION DOC] Passes several sibling modules as one group and expects
        # the union of their submodules. A second group (children 1 and 3) is
        # computed but never compared.
        # KNOWN ISSUE: the second managed_modules and its expected list are
        # computed without a following check, so that case is untested.
        model = nn.Sequential(*[MLP(8) for _ in range(5)])
        # Assume calling `fully_shard` on `[model[0], model[1], model[2]]`
        managed_modules = _get_managed_modules((model[0], model[1], model[2]))
        expected_managed_modules = (
            list(model[0].modules())
            + list(model[1].modules())
            + list(model[2].modules())
        )
        self._check_managed_modules(managed_modules, expected_managed_modules)
        # Assume calling `fully_shard` on `[model[1], model[3]]`
        managed_modules = _get_managed_modules((model[1], model[3]))
        expected_managed_modules = list(model[1].modules()) + list(model[3].modules())

    def _check_managed_modules(
        self,
        managed_modules: list[nn.Module],
        expected_managed_modules: list[nn.Module],
    ):
        # [FUNCTION DOC] HELPER: Asserts the managed module list has the expected
        # length and the same members as the expected list, ignoring order.
        self.assertEqual(len(managed_modules), len(expected_managed_modules))
        # Check set comparison since we do not require anything about the order
        self.assertEqual(set(managed_modules), set(expected_managed_modules))

    def test_managed_states_shared_params_and_buffers(self):
        # [FUNCTION DOC] Ties a weight across three MLPs and a buffer across two,
        # then expects the managed states to list each shared parameter and
        # buffer only once.
        model = nn.Sequential(*[MLP(8, with_buffer=True) for _ in range(3)])
        model[0].in_proj.weight = model[1].in_proj.weight
        model[2].in_proj.weight = model[1].in_proj.weight
        model[1].buffer = model[2].buffer
        # Assume calling `fully_shard` on `model`
        managed_modules = _get_managed_modules((model,))
        params, buffers = _get_managed_states(managed_modules)
        expected_params = list(model.parameters())  # de-dups shared
        expected_buffers = list(model.buffers())  # de-dups shared
        self._check_managed_states(params, buffers, expected_params, expected_buffers)

    def test_managed_states_nested_fully_shard(self):
        # [FUNCTION DOC] With the first child already fully_shard-ed, expects the
        # root's managed parameters and buffers to be only the second child's.
        model = nn.Sequential(*[MLP(8, with_buffer=True) for _ in range(2)])
        fully_shard(model[0])
        # Assume calling `fully_shard` on `model`
        managed_modules = _get_managed_modules((model,))
        params, buffers = _get_managed_states(managed_modules)
        expected_params = list(model[1].parameters())
        expected_buffers = list(model[1].buffers())
        self._check_managed_states(params, buffers, expected_params, expected_buffers)

    def test_managed_states_list_of_mlps(self):
        # [FUNCTION DOC] For a group of three sibling MLPs, expects the managed
        # parameters and buffers to be the concatenation of theirs.
        model = nn.Sequential(*[MLP(8, with_buffer=True) for _ in range(5)])
        # Assume calling `fully_shard` on `[model[0], model[1], model[2]]`
        managed_modules = _get_managed_modules((model[0], model[1], model[2]))
        params, buffers = _get_managed_states(managed_modules)
        expected_params = (
            list(model[0].parameters())
            + list(model[1].parameters())
            + list(model[2].parameters())
        )
        expected_buffers = (
            list(model[0].buffers())
            + list(model[1].buffers())
            + list(model[2].buffers())
        )
        self._check_managed_states(params, buffers, expected_params, expected_buffers)

    def _check_managed_states(
        self,
        managed_params: list[nn.Parameter],
        managed_buffers: list[torch.Tensor],
        expected_managed_params: list[nn.Parameter],
        expected_managed_buffers: list[torch.Tensor],
    ):
        # [FUNCTION DOC] HELPER: Asserts the parameter and buffer lists match the
        # expected ones in length and membership (order ignored).
        self.assertEqual(len(managed_params), len(expected_managed_params))
        self.assertEqual(len(managed_buffers), len(expected_managed_buffers))
        self.assertEqual(set(managed_params), set(expected_managed_params))
        self.assertEqual(set(managed_buffers), set(expected_managed_buffers))


class TestFullyShardParamModuleInfos(FSDPTestMultiThread):
    # [CLASS DOC] Two-rank threaded tests of the helper that builds one
    # ParamModuleInfo per parameter (owning module, name, and shared aliases).
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 2, the number of simulated ranks.
        return 2

    def test_get_param_module_infos_shared_params(self):
        # [FUNCTION DOC] Ties a weight between two MLPs and checks the infos for
        # the de-duplicated parameters: the shared weight is recorded once, with
        # the other module and name listed as its shared alias.
        model = nn.Sequential(*[MLP(8) for _ in range(2)])
        model[0].in_proj.weight = model[1].in_proj.weight
        managed_modules = _get_managed_modules((model,))
        params, _ = _get_managed_states(managed_modules)
        param_module_infos = _get_param_module_infos(params, model)
        self.assertEqual(len(param_module_infos), len(params))
        # We expect `params` to already have de-duplicated shared parameters
        expected_param_module_infos = [
            ParamModuleInfo(model[0].in_proj, "weight", [model[1].in_proj], ["weight"]),
            ParamModuleInfo(model[0].in_proj, "bias", [], []),
            ParamModuleInfo(model[0].out_proj, "weight", [], []),
            ParamModuleInfo(model[0].out_proj, "bias", [], []),
            ParamModuleInfo(model[1].in_proj, "bias", [], []),
            ParamModuleInfo(model[1].out_proj, "weight", [], []),
            ParamModuleInfo(model[1].out_proj, "bias", [], []),
        ]
        self.assertEqual(len(param_module_infos), len(expected_param_module_infos))
        self.assertEqual(param_module_infos, expected_param_module_infos)

    def test_get_param_module_infos_duplicates(self):
        # [FUNCTION DOC] Uses parameters listed from a model with the same MLP
        # twice and checks each info lists the same module as its own alias.
        # A second scenario (a shared in_proj across two MLPs) builds an
        # expected list but does not assert anything.
        # KNOWN ISSUE: the second expected_param_module_infos is never compared
        # (and references mlp rather than the new model), so that part is
        # untested.
        mlp = MLP(8)
        model = nn.Sequential(mlp, mlp)  # shared MLP
        params = list(model.parameters())
        param_module_infos = _get_param_module_infos(params, model)
        self.assertEqual(len(param_module_infos), len(params))
        expected_param_module_infos = [
            ParamModuleInfo(mlp.in_proj, "weight", [mlp.in_proj], ["weight"]),
            ParamModuleInfo(mlp.in_proj, "bias", [mlp.in_proj], ["bias"]),
            ParamModuleInfo(mlp.out_proj, "weight", [mlp.out_proj], ["weight"]),
            ParamModuleInfo(mlp.out_proj, "bias", [mlp.out_proj], ["bias"]),
        ]
        self.assertEqual(len(param_module_infos), len(expected_param_module_infos))
        self.assertEqual(param_module_infos, expected_param_module_infos)

        model = nn.Sequential(*[MLP(8) for _ in range(2)])
        model[0].in_proj = model[1].in_proj  # shared in-projection
        params = list(model.parameters())
        param_module_infos = _get_param_module_infos(params, model)
        self.assertEqual(len(param_module_infos), len(params))
        expected_param_module_infos = [
            ParamModuleInfo(model[0].in_proj, "weight", [model[1].in_proj], ["weight"]),
            ParamModuleInfo(mlp.in_proj, "bias", [], []),
            ParamModuleInfo(mlp.out_proj, "weight", [], []),
            ParamModuleInfo(mlp.out_proj, "bias", [], []),
        ]

    def test_get_param_module_infos_list_of_mlps(self):
        # [FUNCTION DOC] For a group of two sibling MLPs, checks the infos list all
        # eight parameters in order with no shared aliases.
        model = nn.Sequential(*[MLP(8) for _ in range(2)])
        managed_modules = _get_managed_modules((model[0], model[1]))
        params, _ = _get_managed_states(managed_modules)
        param_module_infos = _get_param_module_infos(params, model)
        self.assertEqual(len(param_module_infos), len(params))
        expected_param_module_infos = [
            ParamModuleInfo(model[0].in_proj, "weight", [], []),
            ParamModuleInfo(model[0].in_proj, "bias", [], []),
            ParamModuleInfo(model[0].out_proj, "weight", [], []),
            ParamModuleInfo(model[0].out_proj, "bias", [], []),
            ParamModuleInfo(model[1].in_proj, "weight", [], []),
            ParamModuleInfo(model[1].in_proj, "bias", [], []),
            ParamModuleInfo(model[1].out_proj, "weight", [], []),
            ParamModuleInfo(model[1].out_proj, "bias", [], []),
        ]
        self.assertEqual(len(param_module_infos), len(expected_param_module_infos))
        self.assertEqual(param_module_infos, expected_param_module_infos)


class TestFullyShardShardedParameterTensor(FSDPTestMultiThread):
    # [CLASS DOC] Two-rank threaded tests of how fully_shard shards ordinary
    # (non-DTensor) parameters and which parameters it rejects.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 2, the number of simulated ranks.
        return 2

    @skip_if_lt_x_gpu(1)
    def test_shard_tensor_parameters(self):
        # [FUNCTION DOC] Shards models with odd dimension sizes (uneven shards),
        # with and without a tied in_proj, and checks each parameter via
        # _check_1d_sharded_parameters.
        # Use odd dim sizes to test uneven shards
        model = nn.Sequential(*[MLP(3, dim_multiplier=3) for _ in range(3)])
        orig_params = [param.detach().clone() for param in model.parameters()]
        fully_shard(model)
        sharded_params = list(model.parameters())
        self._check_1d_sharded_parameters(orig_params, sharded_params)

        model = nn.Sequential(*[MLP(3, dim_multiplier=3) for _ in range(3)])
        model[0].in_proj = model[1].in_proj
        orig_params = [param.detach().clone() for param in model.parameters()]
        fully_shard(model)
        sharded_params = list(model.parameters())
        self._check_1d_sharded_parameters(orig_params, sharded_params)

    def _check_1d_sharded_parameters(
        self, orig_params: list[nn.Parameter], sharded_params: list[nn.Parameter]
    ):
        # [FUNCTION DOC] HELPER: For each pair, asserts the sharded parameter is a
        # DTensor on the 1D world mesh with the original size and stride, placed
        # as a dim-0 shard, whose local tensor equals this rank's dim-0 chunk of
        # the original.
        self.assertEqual(len(orig_params), len(sharded_params))
        global_mesh = init_device_mesh(device_type.type, (self.world_size,))
        for orig_param, sharded_param in zip(orig_params, sharded_params):
            self.assertIsInstance(sharded_param, DTensor)
            self.assertEqual(sharded_param.device_mesh, global_mesh)
            self.assertEqual(sharded_param.size(), orig_param.size())
            self.assertEqual(sharded_param.stride(), orig_param.stride())
            self.assertEqual(sharded_param._spec.placements, (Shard(0),))
            chunks = torch.chunk(orig_param, self.world_size, dim=0)
            self.assertEqual(sharded_param._local_tensor, chunks[self.rank])

    def test_raise_scalar_parameter(self):
        # [FUNCTION DOC] Registers a zero-dimensional parameter and asserts
        # fully_shard raises ValueError telling the user to make it a 1D tensor
        # of one element.
        """Tests raising an exception when the model has scalar parameters."""
        model = nn.Sequential(*[MLP(3, dim_multiplier=3) for _ in range(3)])
        model.register_parameter(
            "scalar_p", nn.Parameter(torch.tensor(1.0).to(device_type))
        )
        with self.assertRaisesRegex(
            ValueError, "Change scalar_p to a 1D tensor with numel equal to 1."
        ):
            fully_shard(model)

    def test_raise_noncontiguous_parameter(self):
        # [FUNCTION DOC] Shards a channels-last Conv2d and asserts
        # NotImplementedError about non-contiguous parameters.
        """
        Tests raising an exception when the model has non-contiguous
        parameters. This is due to lack of implementation support.
        """
        conv2d = nn.Conv2d(8, 8, 3).to(memory_format=torch.channels_last)
        with self.assertRaisesRegex(
            NotImplementedError, "FSDP does not support non-contiguous parameters"
        ):
            fully_shard(conv2d)


class TestFullyShardShardedParameterDTensor(FSDPTestMultiThread):
    # [CLASS DOC] Four-rank threaded tests of FSDP sharding on top of tensor
    # parallel DTensor parameters (2D placements).
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 4, the number of simulated ranks.
        return 4

    @skip_if_lt_x_gpu(1)
    def test_shard_dtensor_parameters(self):
        # [FUNCTION DOC] Applies column/row tensor parallelism to an MLP with odd
        # sizes, then fully_shard on the dp mesh. Asserts each parameter is a
        # DTensor on the global mesh with original size and stride, and that the
        # placements are: strided dim-0 shard plus shard for in_proj, shard on
        # dim 0 and 1 for the out_proj weight, and shard plus replicate for the
        # out_proj bias.
        dp_size = 2 if self.world_size > 2 else 1
        global_mesh = init_device_mesh(
            device_type.type,
            (dp_size, self.world_size // dp_size),
            mesh_dim_names=("dp", "tp"),
        )
        dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]
        # Use odd dim sizes to test uneven shards
        model = MLP(9, dim_multiplier=3)
        orig_params = [param.detach().clone() for param in model.parameters()]
        orig_param_names = [param_name for param_name, _ in model.named_parameters()]
        parallelize_module(
            model,
            tp_mesh,
            {"in_proj": ColwiseParallel(), "out_proj": RowwiseParallel()},
        )
        fully_shard(model, mesh=dp_mesh)
        sharded_params = list(model.parameters())
        self.assertEqual(len(orig_params), len(sharded_params))
        for orig_param_name, orig_param, sharded_param in zip(
            orig_param_names, orig_params, sharded_params
        ):
            self.assertIsInstance(sharded_param, DTensor)
            self.assertEqual(sharded_param.device_mesh, global_mesh)
            self.assertEqual(sharded_param.size(), orig_param.size())
            self.assertEqual(sharded_param.stride(), orig_param.stride())
            if "in_proj" in orig_param_name:
                expected_placements = (
                    _StridedShard(0, split_factor=tp_mesh.size()),
                    Shard(0),
                )
            elif "out_proj" in orig_param_name and "weight" in orig_param_name:
                expected_placements = (Shard(0), Shard(1))
            else:
                expected_placements = (Shard(0), Replicate())
            self.assertEqual(sharded_param._spec.placements, expected_placements)


class TestFullyShardLazyInit(FSDPTestMultiThread):
    # [CLASS DOC] Two-rank threaded tests of the lazy initialization of FSDP
    # state: root detection, FQN computation, error cases, and re-registering
    # sharded parameters after meta-device materialization.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 2, the number of simulated ranks.
        return 2

    def test_fully_shard_is_root(self):
        # [FUNCTION DOC] After lazy init from the root, asserts only the root
        # state is marked root and the shared state context lists all three
        # states with the root first.
        """
        Tests that ``_is_root`` is set correctly after lazy initialization.

        FSDP(model(
            0: MLP(FSDP(in_proj), FSDP(out_proj)),
            1: MLP(in_proj, out_proj),
        ))
        """
        model = nn.Sequential(MLP(8), MLP(8))
        fully_shard(model[0].in_proj)
        fully_shard(model[0].out_proj)
        fully_shard(model)  # root gets `model[1]`
        root_state = fully_shard.state(model)
        root_state._lazy_init()

        model0_in_proj_state = fully_shard.state(model[0].in_proj)
        model0_out_proj_state = fully_shard.state(model[0].out_proj)
        self.assertTrue(root_state._is_root)
        self.assertFalse(model0_in_proj_state._is_root)
        self.assertFalse(model0_out_proj_state._is_root)

        all_states = root_state._state_ctx.all_states
        self.assertEqual(len(all_states), 3)
        self.assertEqual(
            all_states, [root_state, model0_in_proj_state, model0_out_proj_state]
        )

    def test_fully_shard_module_and_param_fqns(self):
        # [FUNCTION DOC] After lazy init, asserts each param group has the right
        # module FQN (empty for the root) and holds parameters with the right
        # fully qualified names: the root group gets only the unwrapped second
        # child's parameters; each wrapped Linear gets its own.
        """
        Tests that the module and parameter FQNs are computed correctly after
        lazy initialization.

        FSDP(model(
            0: MLP(FSDP(in_proj), FSDP(out_proj)),
            1: MLP(in_proj, out_proj),
        ))
        """
        model = nn.Sequential(MLP(8), MLP(8))
        fully_shard(model[0].in_proj)
        fully_shard(model[0].out_proj)
        fully_shard(model)  # root gets `model[1]`
        root_state = fully_shard.state(model)
        root_state._lazy_init()

        root_param_group = root_state._fsdp_param_group
        self.assertIsNotNone(root_param_group)
        self.assertEqual(root_param_group._module_fqn, "")
        root_param_fqns = {
            fsdp_param._param_fqn for fsdp_param in root_param_group.fsdp_params
        }
        self.assertEqual(
            root_param_fqns,
            {
                "1.in_proj.weight",
                "1.in_proj.bias",
                "1.out_proj.weight",
                "1.out_proj.bias",
            },
        )

        model0_in_proj_state = fully_shard.state(model[0].in_proj)
        model0_in_proj_param_group = model0_in_proj_state._fsdp_param_group
        self.assertIsNotNone(model0_in_proj_param_group)
        self.assertEqual(model0_in_proj_param_group._module_fqn, "0.in_proj")
        model0_in_proj_param_fqns = {
            fsdp_param._param_fqn
            for fsdp_param in model0_in_proj_param_group.fsdp_params
        }
        self.assertEqual(
            model0_in_proj_param_fqns, {"0.in_proj.weight", "0.in_proj.bias"}
        )

        model0_out_proj_state = fully_shard.state(model[0].out_proj)
        model0_out_proj_param_group = model0_out_proj_state._fsdp_param_group
        self.assertIsNotNone(model0_out_proj_param_group)
        self.assertEqual(model0_out_proj_param_group._module_fqn, "0.out_proj")
        model0_out_proj_param_fqns = {
            fsdp_param._param_fqn
            for fsdp_param in model0_out_proj_param_group.fsdp_params
        }
        self.assertEqual(
            model0_out_proj_param_fqns, {"0.out_proj.weight", "0.out_proj.bias"}
        )

    def test_fully_shard_double_lazy_init(self):
        # [FUNCTION DOC] Lazily initializes a non-root child state first, then the
        # root, and asserts the root call raises RuntimeError saying the child
        # was already initialized and forward must run through the root first.
        model = nn.Sequential(MLP(8), MLP(8))
        fully_shard(model[0].in_proj)
        fully_shard(model[0].out_proj)
        fully_shard(model)
        root_state = fully_shard.state(model)
        model0_in_proj_state = fully_shard.state(model[0].in_proj)
        model0_in_proj_state._lazy_init()
        regex = (
            "FSDP state has already been lazily initialized for 0.in_proj\n"
            "FSDP requires running forward through the root module first"
        )
        with self.assertRaisesRegex(RuntimeError, regex):
            root_state._lazy_init()

    def test_fully_shard_multi_module_root(self):
        # [FUNCTION DOC] Applies fully_shard to a list of two modules and asserts
        # lazy init raises RuntimeError that a single root module is required.
        model = nn.Sequential(MLP(8), MLP(8))
        fully_shard([model[0], model[1]])
        root_state = fully_shard.state(model[0])
        regex = "FSDP requires a single root module but got "
        with self.assertRaisesRegex(RuntimeError, regex):
            root_state._lazy_init()

    def test_reset_sharded_param_in_lazy_init(self):
        # [FUNCTION DOC] Builds a meta-device model, shards both layers and the
        # root, materializes the layers with to_empty, replaces the root-owned
        # weight_norm parameter, and runs forward and backward to check lazy init
        # picks up the replaced parameter without error.
        class MyModel(nn.Module):
            # [CLASS DOC] HELPER: Two bias-free Linear layers and a 3-element
            # weight_norm parameter; forward returns a scalar.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates layer1, layer2, and an
                # uninitialized weight_norm parameter.
                super().__init__()
                self.layer1 = nn.Linear(3, 3, bias=False)
                self.layer2 = nn.Linear(3, 3, bias=False)
                self.weight_norm = nn.Parameter(torch.empty(3))

            def init_weight_norm(self):
                # [FUNCTION DOC] HELPER: Under no_grad, sums the per-row norms of
                # both layer weights and assigns the result as a new weight_norm
                # parameter.
                # KNOWN ISSUE: the assignment targets the enclosing test's local
                # variable named model instead of self; it works only because
                # they are the same object.
                with torch.no_grad():
                    weight_norm = torch.linalg.norm(
                        self.layer1.weight, dim=1
                    ) + torch.linalg.norm(self.layer2.weight, dim=1)
                model.weight_norm = nn.Parameter(weight_norm)

            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Applies both layers in sequence and
                # returns the sum of the output plus the sum of weight_norm.
                out = self.layer1(inp)
                out = self.layer2(out)
                return out.sum() + self.weight_norm.sum()

        with torch.device("meta"):
            model = MyModel()
        fully_shard(model.layer1)
        fully_shard(model.layer2)
        fully_shard(model)

        model.layer1.to_empty(device=device_type.type)
        model.layer2.to_empty(device=device_type.type)
        model.init_weight_norm()

        inp = torch.randn(3, 3, device=device_type.type)
        loss = model(inp).sum()
        loss.backward()


class TestFullyShardMetaDeviceInit(FSDPTestMultiThread):
    # [CLASS DOC] Four-rank threaded tests of sharding models created on the meta
    # device and later materializing them, in 1D and 2D settings, plus the
    # error when unmaterialized meta parameters reach forward.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 4, the number of simulated ranks.
        return 4

    @skip_if_lt_x_gpu(1)
    def test_meta_device_1d_init(self):
        # [FUNCTION DOC] For several MLP widths (even, uneven, tiny) and two mixed
        # precision policies, builds meta-device models, shards them on a 1D
        # mesh, asserts parameters remain on meta, and runs
        # _test_to_empty_and_reset_parameters. Also checks fully_shard works
        # under a meta device context with the default mesh.
        default_pg = torch.distributed.distributed_c10d._get_default_group()
        mesh = init_device_mesh(device_type.type, mesh_shape=(default_pg.size(),))
        # Test both even sharding (8), uneven sharding (3), and empty local tensor (1)
        for mlp_dim in (8, 3, 1):
            # cover foreach_copy code path for bf16
            for mp_policy in (
                MixedPrecisionPolicy(),
                MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32
                ),
            ):
                with torch.device("meta"):
                    model = nn.Sequential(
                        MLP(mlp_dim, dim_multiplier=1, with_buffer=True, bias=False),
                        MLP(mlp_dim, dim_multiplier=1, bias=False),
                    )
                    for param in model.parameters():
                        self.assertEqual(param.device, torch.device("meta"))
                    fully_shard(model[0], mesh=mesh, mp_policy=mp_policy)
                    fully_shard(model[1], mesh=mesh, mp_policy=mp_policy)
                    fully_shard(model, mesh=mesh, mp_policy=mp_policy)
                for param in model.parameters():
                    self.assertEqual(param.device, torch.device("meta"))
                self._test_to_empty_and_reset_parameters(model, mesh, mlp_dim)

        # Test that we can call `fully_shard` under meta-device context and
        # that `init_device_mesh` call still works
        mlp_dim = 8
        with torch.device("meta"):
            model = nn.Sequential(MLP(mlp_dim, with_buffer=True), MLP(mlp_dim))
            for param in model.parameters():
                self.assertEqual(param.device, torch.device("meta"))
            for module in (model[0], model[1], model):
                fully_shard(module)
        for param in model.parameters():
            self.assertEqual(param.device, torch.device("meta"))
        self._test_to_empty_and_reset_parameters(model, mesh, mlp_dim)

    @skip_if_lt_x_gpu(1)
    def test_meta_device_2d_init(self):
        # [FUNCTION DOC] Builds meta-device MLPs (even and uneven width), applies
        # tensor parallelism and then fully_shard on the dp mesh, asserts the
        # parameters stay on meta, and runs the materialization checks against
        # the global 2D mesh. Raises AssertionError if world size is below 4.
        if not (self.world_size >= 4):
            raise AssertionError(f"Expected world_size >= 4, but got {self.world_size}")
        dp_size = 2
        global_mesh = init_device_mesh(
            device_type.type,
            (dp_size, self.world_size // dp_size),
            mesh_dim_names=("dp", "tp"),
        )
        dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]

        # Test both even sharding (8) and uneven sharding (3)
        for mlp_dim in (8, 3):
            with torch.device("meta"):
                model = MLP(mlp_dim, with_buffer=True)
                for param in model.parameters():
                    self.assertEqual(param.device, torch.device("meta"))
                parallelize_module(
                    model,
                    tp_mesh,
                    {"in_proj": ColwiseParallel(), "out_proj": RowwiseParallel()},
                )
                for param in model.parameters():
                    self.assertEqual(param.device, torch.device("meta"))
                fully_shard(model.in_proj, mesh=dp_mesh)
                fully_shard(model.out_proj, mesh=dp_mesh)
                fully_shard(model, mesh=dp_mesh)
            for param in model.parameters():
                self.assertEqual(param.device, torch.device("meta"))
            self._test_to_empty_and_reset_parameters(model, global_mesh, mlp_dim)

    def _test_to_empty_and_reset_parameters(
        self, model: nn.Module, mesh: DeviceMesh, mlp_dim: int
    ):
        # [FUNCTION DOC] HELPER: Materializes a meta-device sharded model on the
        # accelerator with to_empty, fills all states with a constant, calls
        # reset_parameters on modules that have it, and asserts non-empty local
        # parameter shards and all buffers no longer equal the constant. Then
        # runs one forward, backward and Adam step. The mesh argument is unused.
        # Check that we can materialize it on GPU with empty values
        device = torch.device(
            device_type.type, torch.get_device_module(device_type).current_device()
        )
        model.to_empty(device=device)
        for param in model.parameters():
            self.assertEqual(param.device, device)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Check that `reset_parameters()` on each module initializes values
        const = 1337
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            tensor.detach().fill_(const)
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        for param in model.parameters():
            local_tensor = param.to_local()
            if local_tensor.numel() > 0:
                self.assertNotEqual(local_tensor, torch.ones_like(local_tensor) * const)
        for buffer in model.buffers():
            self.assertNotEqual(buffer, torch.ones_like(buffer) * const)

        # Check that we can run an iteration without erroring
        inp = torch.randn((4, mlp_dim), device=device_type.type)
        model(inp).sum().backward()
        optim.step()

    @skip_if_lt_x_gpu(1)
    def test_invalid_meta_device_init(self):
        # [FUNCTION DOC] Shards a meta-device model and runs forward without
        # materializing it, asserting RuntimeError listing the first MLP's
        # parameters still on meta (the second child is not listed).
        default_pg = torch.distributed.distributed_c10d._get_default_group()
        mesh = init_device_mesh(device_type.type, mesh_shape=(default_pg.size(),))
        mlp_dim = 8
        with torch.device("meta"):
            model = nn.Sequential(MLP(mlp_dim, with_buffer=True), MLP(mlp_dim))
            for param in model.parameters():
                self.assertEqual(param.device, torch.device("meta"))
            fully_shard(model[0], mesh=mesh)
            fully_shard(model[1], mesh=mesh)
            fully_shard(model, mesh=mesh)
        inp = torch.randn((4, mlp_dim), device=device_type.type)
        error_regex = (
            "FSDP parameters should be materialized from meta device before training, "
            "but the following were still on meta device: "
            r"\['0.in_proj.weight', '0.in_proj.bias', '0.out_proj.weight', '0.out_proj.bias'\]"
        )
        with self.assertRaisesRegex(RuntimeError, error_regex):
            model(inp)

    @skip_if_lt_x_gpu(1)
    def test_rank0_broadcast_meta_device_init(self):
        # [FUNCTION DOC] Rank 0 builds a full CPU Transformer state dict; all
        # ranks build a sharded meta-device Transformer. Rank 0 broadcasts each
        # full tensor over the mesh group and every rank distributes it to its
        # shard, then loads the result with assign. Builds a reference sharded
        # model by broadcasting parameters, then checks parameters match and one
        # forward/backward gives equal loss and gradients.
        model_args = ModelArgs(dropout_p=0.0)
        # Assume we have a CPU full state dict on rank 0
        if self.rank == 0:
            torch.manual_seed(42)
            ref_model = Transformer(model_args)
            full_sd = ref_model.state_dict()
            for param in full_sd.values():
                self.assertEqual(param.device, torch.device("cpu"))

        # Initialize the sharded model on meta device
        fsdp_mesh = init_device_mesh(device_type.type, (self.world_size,))
        with torch.device("meta"):
            model = Transformer(model_args)
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard(module, mesh=fsdp_mesh)
        fully_shard(model, mesh=fsdp_mesh)
        for param in model.parameters():
            self.assertEqual(param.device, torch.device("meta"))

        # Construct a sharded state dict from the rank 0 full state dict by
        # broadcasting and sharding
        meta_sharded_sd = model.state_dict()
        sharded_sd = {}
        if self.rank == 0:
            self.assertEqual(len(meta_sharded_sd), len(full_sd))
            self.assertEqual(list(meta_sharded_sd.keys()), list(full_sd.keys()))
            for (param_name, full_param), sharded_meta_param in zip(
                full_sd.items(), meta_sharded_sd.values()
            ):
                full_param = full_param.detach().to(device_type)
                mesh = sharded_meta_param.device_mesh
                dist.broadcast(full_param, src=0, group=mesh.get_group(0))
                sharded_tensor = distribute_tensor(
                    full_param, mesh, sharded_meta_param.placements
                )
                sharded_sd[param_name] = nn.Parameter(sharded_tensor)
        else:
            for param_name, sharded_meta_param in meta_sharded_sd.items():
                full_tensor = torch.empty(
                    sharded_meta_param.size(),
                    device=device_type.type,
                    dtype=sharded_meta_param.dtype,
                )
                mesh = sharded_meta_param.device_mesh
                dist.broadcast(full_tensor, src=0, group=mesh.get_group(0))
                sharded_tensor = distribute_tensor(
                    full_tensor, mesh, sharded_meta_param.placements
                )
                sharded_sd[param_name] = nn.Parameter(sharded_tensor)

        model.load_state_dict(sharded_sd, assign=True)
        for param in model.parameters():
            self.assertIsInstance(param, DTensor)
            self.assertEqual(param.device.type, device_type.type)

        # Construct the reference model on nonzero ranks by broadcasting the
        # unsharded model from rank 0 and sharding on all ranks
        if self.rank != 0:
            ref_model = Transformer(model_args)
        for param in ref_model.parameters():
            torch.distributed.broadcast(param.detach(), src=0)
        for module in ref_model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard(module, mesh=fsdp_mesh)
        fully_shard(ref_model, mesh=fsdp_mesh)

        for (param_name, param), (ref_param_name, ref_param) in zip(
            model.named_parameters(), ref_model.named_parameters()
        ):
            self.assertEqual(param_name, ref_param_name)
            self.assertEqual(param, ref_param)

        # Check one forward/backward for parity
        inp = torch.randint(0, model_args.vocab_size, (2, 16), device=device_type.type)
        loss = model(inp).sum()
        loss.backward()
        ref_loss = ref_model(inp).sum()
        ref_loss.backward()
        self.assertEqual(loss, ref_loss)
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            self.assertEqual(param.grad, ref_param.grad)


class TestFullyShardProcessGroupInit(FSDPTestMultiThread):
    # [CLASS DOC] Four-rank threaded tests that device meshes built from existing
    # process groups behave like meshes from init_device_mesh when used with
    # fully_shard (1D and HSDP).
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 4, the number of simulated ranks.
        return 4

    @skip_if_lt_x_gpu(1)
    def test_1d_process_group_init(self):
        # [FUNCTION DOC] Builds a 1D dp mesh from the dp process group and checks
        # its mesh tensor, coordinates and group names equal the reference mesh.
        # Shards two copies of an MLP, one per mesh, runs forward/backward with
        # tp-consistent inputs, and asserts equal loss, local parameters,
        # gradients and mesh tensors. Raises AssertionError if world size is not 4.
        if not (self.world_size == 4):
            raise AssertionError(f"Expected world_size == 4, but got {self.world_size}")
        # For convenience, use device mesh's infra to construct the DP PG
        # (in practice, the trainer would do it manually via `new_group()`)
        dp_size = 2
        global_mesh = init_device_mesh(
            device_type.type,
            (dp_size, self.world_size // dp_size),
            mesh_dim_names=("dp", "tp"),
        )
        ref_dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]
        dp_pg = ref_dp_mesh.get_group(0)

        # Check the `from_group()` API for correctness
        dp_mesh = DeviceMesh.from_group(dp_pg, device_type.type, mesh_dim_names=("dp",))
        # Only compare the mesh tensors, not `DeviceMesh` objects themselves,
        # since the ref has a parent mesh, while the `from_group` one does not
        self.assertEqual(dp_mesh.mesh, ref_dp_mesh.mesh)
        self.assertEqual(dp_mesh._coordinate_on_dim, ref_dp_mesh._coordinate_on_dim)
        self.assertEqual(dp_mesh._dim_group_names, ref_dp_mesh._dim_group_names)

        # Check 1D FSDP forward/backward parity over the DP mesh
        # NOTE: We cannot use 2D DTensor-based training here because the DP
        # mesh from `from_group` does not respect the parent mesh.
        torch.manual_seed(42)
        mlp_dim = 8
        ref_model = MLP(mlp_dim)
        for param in ref_model.parameters():
            dist.broadcast(param.detach(), src=0)
        model = copy.deepcopy(ref_model)

        # Parallelize the test model with the ref DP mesh
        for module in (ref_model.in_proj, ref_model.out_proj, ref_model):
            fully_shard(module, mesh=ref_dp_mesh)
        # Parallelize the test model with the new DP mesh from the PG
        for module in (model.in_proj, model.out_proj, model):
            fully_shard(module, mesh=dp_mesh)

        # Ensure that TP ranks have the same input
        inp = torch.randn((4, mlp_dim), device=device_type.type)
        if self.rank in (0, 1):
            dist.broadcast(inp, src=0, group=tp_mesh.get_group(0))
        elif self.rank in (2, 3):
            dist.broadcast(inp, src=2, group=tp_mesh.get_group(0))

        ref_loss = ref_model(inp).sum()
        ref_loss.backward()
        loss = model(inp).sum()
        loss.backward()
        self.assertEqual(loss, ref_loss)
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            # Cannot compare `DTensor`s directly since their meshes are not
            # equal due to the ref parameter's mesh having a parent mesh while
            # the other's mesh does not
            self.assertEqual(param.to_local(), ref_param.to_local())
            self.assertEqual(param.device_mesh.mesh, ref_param.device_mesh.mesh)
            self.assertEqual(param.grad.to_local(), ref_param.grad.to_local())
            self.assertEqual(
                param.grad.device_mesh.mesh, ref_param.grad.device_mesh.mesh
            )

    @skip_if_lt_x_gpu(1)
    def test_2d_process_group_init(self):
        # [FUNCTION DOC] Builds an HSDP mesh from intra-node (shard) and
        # inter-node (replicate) process groups, checks it and its child meshes
        # match a reference mesh, and verifies forward/backward parity of two
        # sharded MLP copies. Then enables a separate reduce-scatter group and
        # asserts every param group has one distinct from the shard group, with
        # all modules sharing a single such group.
        shard_mesh_dim_size = 2
        if not (self.world_size % shard_mesh_dim_size == 0):
            raise AssertionError(
                f"Expects {self.world_size} to be divisible by {shard_mesh_dim_size}"
            )
        replicate_mesh_dim_size = self.world_size // shard_mesh_dim_size
        mesh_dim_names = ("replicate", "shard")
        ref_mesh = init_device_mesh(
            device_type.type,
            (replicate_mesh_dim_size, shard_mesh_dim_size),
            mesh_dim_names=mesh_dim_names,
        )

        # Use the global PG as the parent group (in practice, this could be a
        # subgroup of the global PG)
        dp_group = dist.distributed_c10d._get_default_group()
        dp_shard_group = _init_intra_node_process_group(shard_mesh_dim_size)
        dp_replicate_group = _init_inter_node_process_group(
            dp_group, replicate_mesh_dim_size
        )
        mesh_tensor = torch.tensor(
            dist.get_process_group_ranks(dp_group), dtype=torch.int
        ).view(replicate_mesh_dim_size, shard_mesh_dim_size)

        # Check the `from_group()` API for correctness
        mesh = DeviceMesh.from_group(
            [dp_replicate_group, dp_shard_group],
            device_type.type,
            mesh_dim_names=mesh_dim_names,
            mesh=mesh_tensor,
        )
        self.assertEqual(mesh.mesh, ref_mesh.mesh)
        self.assertEqual(mesh._coordinate_on_dim, ref_mesh._coordinate_on_dim)
        for mesh_dim_name in mesh_dim_names:
            child_mesh = mesh[mesh_dim_name]
            ref_child_mesh = ref_mesh[mesh_dim_name]
            self.assertEqual(child_mesh, ref_child_mesh)
            child_ranks = dist.distributed_c10d.get_process_group_ranks(
                child_mesh.get_group()
            )
            ref_child_ranks = dist.distributed_c10d.get_process_group_ranks(
                ref_child_mesh.get_group()
            )
            self.assertEqual(child_ranks, ref_child_ranks)

        # Check HSDP forward/backward parity
        torch.manual_seed(42)
        mlp_dim = 8
        ref_model = MLP(mlp_dim)
        for param in ref_model.parameters():
            dist.broadcast(param.detach(), src=0)
        model = copy.deepcopy(ref_model)

        # Parallelize the test model with the ref mesh
        for module in (ref_model.in_proj, ref_model.out_proj, ref_model):
            fully_shard(module, mesh=ref_mesh)
        # Parallelize the test model with the new mesh from the PG
        for module in (model.in_proj, model.out_proj, model):
            fully_shard(module, mesh=mesh)

        inp = torch.randn((4, mlp_dim), device=device_type.type)
        ref_loss = ref_model(inp).sum()
        ref_loss.backward()
        loss = model(inp).sum()
        loss.backward()
        self.assertEqual(loss, ref_loss)
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            self.assertEqual(param, ref_param)
            self.assertEqual(param.grad, ref_param.grad)

        # Also cover the opt-in reduce-scatter PG setup on HSDP meshes. Each
        # rank creates the communicator for its shard group, so the
        # implementation must use local synchronization internally to avoid
        # waiting for nonmember ranks in new_group's post-init barrier.
        model.set_separate_reduce_scatter_group()
        rs_process_groups = set()
        for module in model.modules():
            if isinstance(module, FSDPModule):
                for param_group in module._get_fsdp_state()._fsdp_param_groups:
                    self.assertIsNotNone(
                        param_group.mesh_info.reduce_scatter_process_group
                    )
                    self.assertIsNot(
                        param_group._reduce_scatter_process_group,
                        param_group.mesh_info.shard_process_group,
                    )
                    rs_process_groups.add(
                        id(param_group.mesh_info.reduce_scatter_process_group)
                    )
        self.assertEqual(len(rs_process_groups), 1)


class TestFullyShardHSDPBroadcast(FSDPTestMultiThread):
    # [CLASS DOC] Four-rank threaded test that replicas of an HSDP-sharded model
    # can be synchronized by broadcasting along the replicate mesh dimension.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 4, the number of simulated ranks.
        return 4

    @skip_if_lt_x_gpu(1)
    def test_hsdp_broadcast_across_replicas(self):
        # [FUNCTION DOC] Shards a Transformer (with an extra buffer) on a 2x2
        # HSDP mesh, overwrites states on non-zero replicas with a constant,
        # verifies replicas differ, broadcasts local shards from the first rank of
        # the replicate group, verifies replicas are now equal, and runs one
        # forward/backward.
        shard_size, replicate_size = 2, 2
        mesh = init_device_mesh(
            device_type.type,
            (replicate_size, shard_size),
            mesh_dim_names=("replicate", "shard"),
        )
        model_args = ModelArgs()
        model = Transformer(model_args)
        # Add a buffer to show that this flow works for buffers too
        model.buf = torch.nn.Buffer(torch.randn((model_args.dim,)))
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard(module, mesh=mesh)
        fully_shard(model, mesh=mesh)

        # Only preserve the model states on the replicate mesh's rank 0
        if mesh.get_local_rank("replicate") > 0:
            for tensor in itertools.chain(model.parameters(), model.buffers()):
                tensor.detach().fill_(1337)

        # Check that replicas are different
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            local_tensor = tensor.to_local() if isinstance(tensor, DTensor) else tensor
            local_tensor_list = [
                torch.empty_like(local_tensor) for _ in range(mesh["replicate"].size())
            ]
            dist.all_gather(
                local_tensor_list, local_tensor, group=mesh.get_group("replicate")
            )
            for other_local_tensor in local_tensor_list[1:]:
                self.assertEqual(other_local_tensor.shape, local_tensor_list[0].shape)
                self.assertNotEqual(other_local_tensor, local_tensor_list[0])

        # Broadcast from replicate mesh's rank 0
        replicate_group = mesh.get_group("replicate")
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            # E.g. for mesh [[0, 1, 2, 3], [4, 5, 6, 7]] sharding on dim-1 and
            # replicating on dim-0, broadcast with sources 0, 1, 2, 3
            src_rank = dist.get_process_group_ranks(replicate_group)[0]
            torch.distributed.broadcast(
                tensor.to_local() if isinstance(tensor, DTensor) else tensor,
                src=src_rank,
                group=replicate_group,
            )

        # Check that replicas are the same
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            local_tensor = tensor.to_local() if isinstance(tensor, DTensor) else tensor
            local_tensor_list = [
                torch.empty_like(local_tensor) for _ in range(mesh["replicate"].size())
            ]
            dist.all_gather(
                local_tensor_list, local_tensor, group=mesh.get_group("replicate")
            )
            for other_local_tensor in local_tensor_list[1:]:
                self.assertEqual(other_local_tensor, local_tensor_list[0])

        # Check that we can run an iteration without erroring
        inp = torch.randint(0, model_args.vocab_size, (2, 16), device=device_type.type)
        model(inp).sum().backward()


class TestHSDPWithCustomHook(FSDPTestMultiThread):
    # [CLASS DOC] Four-rank threaded tests of set_all_reduce_hook: stream
    # restrictions and a custom inter-replica all-reduce in place of native HSDP.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 4, the number of simulated ranks.
        return 4

    def perThreadSetUp(self) -> None:
        # [FUNCTION DOC] HELPER: Runs the base per-thread setup, then sets the
        # default tensor device to the test device for this thread.
        super().perThreadSetUp()
        torch.set_default_device(device_type)

    @skip_if_lt_x_gpu(1)
    def test_custom_hook_custom_stream(self):
        # [FUNCTION DOC] Asserts set_all_reduce_hook with a stream on a native
        # HSDP model raises ValueError "stream cannot be set when using native
        # HSDP". Then shards on a 1D intra-node mesh, installs a hook with a
        # custom stream, runs forward/backward, and asserts the hook ran on that
        # custom stream.
        hsdp_mesh = init_device_mesh(
            device_type.type, (2, 2), mesh_dim_names=("replicate", "shard")
        )
        model = MLP(10, bias=False)
        fully_shard(model, mesh=hsdp_mesh)
        model = cast(FSDPModule, model)
        custom_stream = torch.get_device_module(device_type).Stream()

        # native HSDP should reject
        with self.assertRaises(ValueError) as cm:
            model.set_all_reduce_hook(lambda output: output, stream=custom_stream)

        ex = cm.exception
        self.assertEqual(str(ex), "stream cannot be set when using native HSDP")

        # FSDP + hook in custom stream is ok
        intra_pg = _init_intra_node_process_group(2)
        fsdp_mesh = DeviceMesh.from_group(
            intra_pg,
            device_type.type,
            dist.get_process_group_ranks(intra_pg),
            mesh_dim_names=("shard",),
        )
        hook_used_stream = None

        def _hook(_output: torch.Tensor) -> None:
            # [FUNCTION DOC] HELPER: Records the current device stream into the
            # enclosing hook_used_stream variable; ignores the reduced output.
            nonlocal hook_used_stream
            hook_used_stream = torch.get_device_module(device_type).current_stream()

        model = MLP(10, bias=False)
        fully_shard(model, mesh=fsdp_mesh)
        model = cast(FSDPModule, model)
        model.set_all_reduce_hook(_hook, stream=custom_stream)

        inp = torch.arange(10, dtype=torch.float32, requires_grad=True).view(1, 10)
        out = model(inp)
        out.sum().backward()
        torch.get_device_module(device_type).synchronize()
        self.assertEqual(hook_used_stream, custom_stream)

    @skip_if_lt_x_gpu(1)
    def test_custom_hsdp_all_reduce_hook(self):
        # [FUNCTION DOC] Shards an MLP with constant per-replica-group weights on
        # an intra-node mesh and installs a hook that averages the reduced
        # gradient across the inter-node group. After forward/backward asserts
        # the hook ran and the local gradients of both projections equal the
        # expected values (halved by averaging with the zero-weight replica).
        world_pg = dist.distributed_c10d._get_default_group()
        intra_pg = _init_intra_node_process_group(2)
        inter_pg = _init_inter_node_process_group(world_pg, 2)
        mesh = DeviceMesh.from_group(
            intra_pg,
            device_type.type,
            dist.get_process_group_ranks(intra_pg),
            mesh_dim_names=("shard",),
        )
        model = MLP(10, bias=False)
        rank = dist.get_rank()
        rank_group = rank // 2

        # init the weights to be constant within each group
        # this is just to simplify the test numeric check when we do bwd
        torch.nn.init.constant_(model.in_proj.weight, 1.0 * rank_group)
        torch.nn.init.constant_(model.out_proj.weight, 2.0 * rank_group)

        model = fully_shard(model, mesh=mesh)

        hook_called: bool = False

        def _custom_hook(output: torch.Tensor) -> None:
            # [FUNCTION DOC] HELPER: All-reduces the given gradient in place with
            # an average across the inter-node group, and sets hook_called.
            nonlocal hook_called
            dist.all_reduce(output, group=inter_pg, op=dist.ReduceOp.AVG)
            hook_called = True

        model.set_all_reduce_hook(_custom_hook)

        inp = torch.arange(10, dtype=torch.float32, requires_grad=True).view(1, 10)
        out = model(inp)
        out.sum().backward()
        torch.get_device_module(device_type).synchronize()
        # custom hook was fired
        self.assertTrue(hook_called)
        # within each replica, FSDP shards the weights at dim 0
        # so half of MLP weights with 2x2 setup
        out_proj_local_grad = model.out_proj.weight.grad.to_local().cpu()
        in_proj_local_grad = model.in_proj.weight.grad.to_local().cpu()

        # grad is halved in custom bwd all reduce hook during avg
        # as replica 0 weights are 0
        self.assertEqual(
            out_proj_local_grad,
            torch.full((5, 40), 22.5, dtype=torch.float32, device="cpu"),
        )
        self.assertEqual(
            in_proj_local_grad,
            torch.arange(0, 100, 10, dtype=torch.float32, device="cpu").repeat(20, 1),
        )


class TestFullyShardShardPlacementFn(FSDPTestMultiThread):
    # [CLASS DOC] Eight-rank threaded tests of the shard_placement_fn argument,
    # which chooses the dimension each parameter is sharded on.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 8, the number of simulated ranks.
        return 8

    def _init_models(self):
        # [FUNCTION DOC] HELPER: Seeds the RNG with 42, builds a 3-layer
        # Transformer without dropout, broadcasts its parameters from rank 0, and
        # returns it with a deep copy to use as an unsharded reference.
        torch.manual_seed(42)
        model_args = ModelArgs(n_layers=3, dropout_p=0.0)
        model = Transformer(model_args)
        for param in model.parameters():
            dist.broadcast(param.detach(), src=0)
        ref_model = copy.deepcopy(model)
        return model, ref_model

    @skip_if_lt_x_gpu(1)
    def test_init_1d_transformer_shard_largest_dim(self):
        # [FUNCTION DOC] Shards each layer and the root choosing the largest
        # dimension of every parameter. Asserts every parameter has exactly one
        # Shard placement, at least one is on dim 1, and the full tensors equal
        # the reference parameters.
        model, ref_model = self._init_models()

        def shard_placement_fn(param: nn.Parameter) -> Shard | None:
            # [FUNCTION DOC] HELPER: Returns a Shard placement on the first
            # largest dimension of the parameter's shape; raises AssertionError
            # if the shape has no dimensions.
            largest_dim = largest_dim_size = -1
            for dim, dim_size in enumerate(param.shape):
                if dim_size > largest_dim_size:
                    largest_dim = dim
                    largest_dim_size = dim_size
            if not (largest_dim >= 0):
                raise AssertionError(
                    f"Expected largest_dim >= 0, but got {largest_dim} for shape {param.shape}"
                )
            return Shard(largest_dim)

        for layer in model.layers:
            fully_shard(layer, shard_placement_fn=shard_placement_fn)
        fully_shard(model, shard_placement_fn=shard_placement_fn)

        any_shard_dim1 = False
        for param in model.parameters():
            self.assertEqual(len(param.placements), 1)
            self.assertIsInstance(param.placements[0], Shard)
            any_shard_dim1 |= param.placements[0].dim == 1
        self.assertTrue(any_shard_dim1)

        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            full_param = param.full_tensor()
            self.assertEqual(full_param, ref_param)

    @skip_if_lt_x_gpu(1)
    def test_init_1d_transformer_shard_dim_neg1(self):
        # [FUNCTION DOC] Shards using a placement function that returns a negative
        # dim, checking FSDP normalizes it, and that full tensors match the
        # reference parameters.
        model, ref_model = self._init_models()

        def shard_placement_fn(param: nn.Parameter) -> Shard | None:
            # [FUNCTION DOC] HELPER: Always returns a shard on the last dimension
            # (negative index), ignoring the parameter.
            # Check that FSDP will normalize this dim to non-negative
            return Shard(-1)

        for layer in model.layers:
            fully_shard(layer, shard_placement_fn=shard_placement_fn)
        fully_shard(model, shard_placement_fn=shard_placement_fn)

        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            full_param = param.full_tensor()
            self.assertEqual(full_param, ref_param)

    @skip_if_lt_x_gpu(1)
    def test_init_2d_transformer_shard_diff_dim(self):
        # [FUNCTION DOC] Applies tensor parallelism with sequence parallel to the
        # Transformer, then fully_shard on the dp mesh with a placement function
        # choosing the dimension opposite to the TP shard. Asserts linear weights
        # have two Shard placements whose dims sum to 1 (FSDP and TP on different
        # dims), other parameters have at least one Shard, and full tensors match
        # the reference.
        model, ref_model = self._init_models()

        dp_size, tp_size = self.world_size // 2, 2
        global_mesh = init_device_mesh(
            device_type.type, (dp_size, tp_size), mesh_dim_names=("dp", "tp")
        )
        model = Transformer.parallelize(model, global_mesh["tp"], use_seq_parallel=True)

        def shard_placement_fn(param: nn.Parameter) -> Shard | None:
            # [FUNCTION DOC] HELPER: For a DTensor parameter, returns a Shard on
            # the dim mirroring its first existing Shard (last dim minus that
            # dim), raising AssertionError if negative; otherwise returns
            # Shard(0).
            if isinstance(param, DTensor):
                for placement in param.placements:
                    if isinstance(placement, Shard):
                        shard_dim = param.ndim - 1 - placement.dim
                        if not (shard_dim >= 0):
                            raise AssertionError(
                                f"Expected shard_dim >= 0, but got {shard_dim} for shape {param.shape}"
                            )
                        return Shard(shard_dim)
            return Shard(0)

        for layer in model.layers:
            fully_shard(
                layer, mesh=global_mesh["dp"], shard_placement_fn=shard_placement_fn
            )
        fully_shard(
            model, mesh=global_mesh["dp"], shard_placement_fn=shard_placement_fn
        )

        linear_weight_names = ["wq", "wk", "wv", "wo", "w1", "w2"]
        for param_name, param in model.named_parameters():
            if (
                any(n in param_name for n in linear_weight_names)
                and "weight" in param_name
            ):
                total_placement_dims = 0
                for placement in param.placements:
                    self.assertTrue(isinstance(placement, Shard))
                    total_placement_dims += placement.dim
                self.assertEqual(param.ndim, 2)
                # Check that FSDP shards on either dim-0 or dim-1, and TP
                # shards on the other
                self.assertEqual(total_placement_dims, 1)
            else:
                self.assertTrue(
                    any(isinstance(placement, Shard) for placement in param.placements)
                )

        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            full_param = param.full_tensor()
            self.assertEqual(full_param, ref_param)

    @skip_if_lt_x_gpu(1)
    def test_init_1d_uneven_shard_largest_dim(self):
        # [FUNCTION DOC] Shards a model with a 17-wide layer using the largest-dim
        # placement and asserts NotImplementedError since uneven sharding on
        # dim 1 is unsupported.
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(16, 17), nn.Linear(17, 8))

        def shard_placement_fn(param: nn.Parameter) -> Shard | None:
            # [FUNCTION DOC] HELPER: Returns a Shard on the first largest
            # dimension of the parameter; raises AssertionError for an invalid
            # dimension.
            largest_dim = -1
            largest_dim_size = -1
            for dim, dim_size in enumerate(param.shape):
                if dim_size > largest_dim_size:
                    largest_dim = dim
                    largest_dim_size = dim_size
            if not (largest_dim >= 0):
                raise AssertionError(
                    f"Expected largest_dim >= 0, but got {largest_dim} for shape {param.shape}"
                )
            if not (largest_dim < param.ndim):
                raise AssertionError(
                    f"Expected largest_dim < param.ndim, but got {largest_dim=} {param.shape}"
                )
            return Shard(largest_dim)

        with self.assertRaisesRegex(
            NotImplementedError, "FSDP does not support uneven sharding on dim 1"
        ):
            fully_shard(model, shard_placement_fn=shard_placement_fn)

    def test_invalid_shard_dim(self):
        # [FUNCTION DOC] Requests dim 1 for every parameter, including 1D biases,
        # and asserts fully_shard raises AssertionError that the dim is invalid
        # for a 1D tensor.
        model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 8))

        def shard_placement_fn(param: nn.Parameter) -> Shard | None:
            # [FUNCTION DOC] HELPER: Always returns a shard on dim 1.
            return Shard(1)

        # Shard(1) is invalid for 1D bias parameters
        with self.assertRaisesRegex(
            AssertionError, "Shard dim 1 is invalid for 1D tensor"
        ):
            fully_shard(model, shard_placement_fn=shard_placement_fn)


# TODO: Remove this test class once we remove the old import path:
# torch/distributed/_composable/fsdp
class TestFullyShardOldImport(FSDPTestMultiThread):
    # [CLASS DOC] Two-rank threaded test of training with fully_shard imported
    # through the public path; to be removed with the deprecated import path.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 2, the number of simulated ranks.
        return 2

    def test_old_import_training(self):
        # [FUNCTION DOC] Shards two Linear layers and the root with a bfloat16
        # param policy, asserts all three are FSDPModule instances, and runs one
        # forward/backward.
        # KNOWN ISSUE: despite the class comment, the imports at the top of this
        # file use the new public path, so the old path is not exercised.
        model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16))
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
        fully_shard(model[0], mp_policy=mp_policy)
        fully_shard(model[1], mp_policy=mp_policy)
        fully_shard(model, mp_policy=mp_policy)

        self.assertIsInstance(model[0], FSDPModule)
        self.assertIsInstance(model[1], FSDPModule)
        self.assertIsInstance(model, FSDPModule)

        inp = torch.randn((8, 16), device=device_type)
        model(inp).sum().backward()


class TestFullyShardMixedDtypeParam(FSDPTestMultiThread):
    # [CLASS DOC] Two-rank threaded test that frozen parameters of differing
    # dtypes can share one FSDP group.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 2, the number of simulated ranks.
        return 2

    @skip_if_lt_x_gpu(2)
    def test_mixed_dtypes_no_grad_param(self):
        # [FUNCTION DOC] Shards a model with a frozen float8 parameter and a
        # trainable float32 parameter on a 1D mesh and runs its forward, expecting
        # no error.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Holds a frozen float8 weight and a float32 weight
            # of equal shape; forward returns nothing.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the two 256x256 uninitialized
                # parameters; the float8 one has requires_grad False.
                super().__init__()
                # no grad params with different dtypes
                self.w_fp8 = torch.nn.Parameter(
                    torch.empty((256, 256), dtype=torch.float8_e4m3fn),
                    requires_grad=False,
                )
                self.w_fp32 = torch.nn.Parameter(
                    torch.empty((256, 256), dtype=torch.float32)
                )

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Ignores the input and returns None.
                return

        mesh = init_device_mesh(device_type.type, (self.world_size,))
        model = Model()
        fully_shard(model, mesh=mesh)
        model(0)


class TestFullyShardNonFloatParam(FSDPTest):
    # [CLASS DOC] Two-process test case for integer-typed (non-float) frozen
    # parameters inside an FSDP group.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 2, the number of ranks.
        return 2

    @skip_if_lt_x_gpu(2)
    def test_non_float_param(self):
        # [FUNCTION DOC] Runs the non-float parameter check over three sets of
        # integer dtypes, two mixed precision policies, and with or without an
        # extra frozen float parameter. Needs 2+ GPUs.
        """Non-float params (e.g. uint8, int64) are supported in all-gather,
        excluded from reduce-scatter, and not cast by mp_policy.param_dtype."""
        self.run_subtests(
            {
                "non_float_dtypes": [
                    [torch.uint8],
                    [torch.int64],
                    [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64],
                ],
                "mp_policy": [
                    MixedPrecisionPolicy(),
                    MixedPrecisionPolicy(
                        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
                    ),
                ],
                "frozen_float": [False, True],
            },
            self._test_non_float_param,
        )

    def _test_non_float_param(
        self,
        non_float_dtypes: list,
        mp_policy: MixedPrecisionPolicy,
        frozen_float: bool,
    ):
        # [FUNCTION DOC] HELPER: Builds a Linear model plus frozen integer
        # parameters (and optionally a frozen float one), shards it with the given
        # policy, and runs forward/backward with all-gather and reduce-scatter
        # patched to assert: all-gather input dtype is the single common dtype
        # (param_dtype for floats, original for ints) or uint8 when mixed, its
        # output byte count is the sum of per-param bytes, and the reduce-scatter
        # input numel counts only parameters requiring grad.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: Linear layer plus optional frozen float
            # parameter and one frozen integer parameter per requested dtype.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer, the optional
                # frozen float parameter, and non_float_N integer parameters
                # (random values below 127) with requires_grad False.
                super().__init__()
                self.linear = nn.Linear(16, 16)
                if frozen_float:
                    self.frozen_float = nn.Parameter(
                        torch.randn(16), requires_grad=False
                    )
                for i, dtype in enumerate(non_float_dtypes):
                    self.register_parameter(
                        f"non_float_{i}",
                        nn.Parameter(
                            torch.randint(0, 127, (16,), dtype=dtype),
                            requires_grad=False,
                        ),
                    )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies only the linear layer to x.
                return self.linear(x)

        model = Model()
        fully_shard(model, mp_policy=mp_policy)
        for p in model.parameters():
            self.assertEqual(p.size(0) % self.world_size, 0)
        ag_input_dtypes = set()
        expected_ag_output_bytes = 0
        for p in model.parameters():
            if p.dtype.is_floating_point and mp_policy.param_dtype is not None:
                ag_input_dtypes.add(mp_policy.param_dtype)
                expected_ag_output_bytes += p.numel() * mp_policy.param_dtype.itemsize
            else:
                # Non-float params keep their original dtype; param_dtype
                # only applies to floating-point params
                ag_input_dtypes.add(p.dtype)
                expected_ag_output_bytes += p.numel() * p.element_size()
        expected_ag_dtype = (
            next(iter(ag_input_dtypes)) if len(ag_input_dtypes) == 1 else torch.uint8
        )
        orig_ag = dist.all_gather_single

        def assert_all_gather(*args, **kw):
            # [FUNCTION DOC] HELPER: Wrapper for the single-tensor all-gather;
            # extracts output and input tensors from positional or keyword args,
            # asserts the input dtype and output byte size, then calls the
            # original and returns its result.
            output = kw.get("output", args[0] if len(args) > 0 else None)
            input = kw.get("input_tensor", args[1] if len(args) > 1 else None)
            self.assertEqual(input.dtype, expected_ag_dtype)
            self.assertEqual(output.nbytes, expected_ag_output_bytes)
            return orig_ag(*args, **kw)

        expected_rs_input_numel = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        orig_rs = dist.reduce_scatter_single

        def assert_reduce_scatter(*args, **kw):
            # [FUNCTION DOC] HELPER: Wrapper for the single-tensor reduce-scatter;
            # asserts the input tensor's numel equals the expected total of
            # trainable parameters, then calls the original and returns its result.
            input = kw.get("input", args[1] if len(args) > 1 else None)
            self.assertEqual(input.numel(), expected_rs_input_numel)
            return orig_rs(*args, **kw)

        x = torch.randn(4, 16, device=device_type)
        with (
            patch_all_gather(assert_all_gather),
            patch_reduce_scatter(assert_reduce_scatter),
        ):
            loss = model(x).sum()
            loss.backward()


if __name__ == "__main__":
    run_tests()
