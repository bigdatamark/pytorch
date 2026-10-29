# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for the parallelize_module API: applying ColwiseParallel, RowwiseParallel
# and the PrepareModule* styles through plans keyed by exact names, wildcard/question/digit
# patterns, the root module (empty name), under a DeviceMesh context, with src_data_rank, and
# with unmatched or empty plans (UserWarning). Parallelized models are compared numerically
# against a local copy (outputs, gradients, SGD updates). Defines DummyModule and
# TensorParallelAPITests plus a local-tensor variant skipping the src_data_rank test.
from collections import OrderedDict
from copy import deepcopy

import torch
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.parallel.api import parallelize_module
from torch.distributed.tensor.parallel.style import (
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    PrepareModuleOutput,
    RowwiseParallel,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorTestBase,
    map_local_tensor_for_rank,
    MLPModule,
    MLPStacked,
    with_comms,
)


class DummyModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Parameterless identity module used to test the input/output
    # preparation styles in isolation.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes an empty module.
        super().__init__()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the input unchanged.
        return x


class TensorParallelAPITests(DTensorTestBase):
    # [CLASS DOC] Multi-process DTensor test case for parallelize_module. Stateless; provides
    # helpers that compare a local model with its tensor-parallel copy, and many plan tests.
    @property
    def world_size(self):
        # [FUNCTION DOC] Returns the accelerator count when it is even and greater than 4,
        # otherwise 4.
        gpu_num = torch.accelerator.device_count()
        return gpu_num if gpu_num % 2 == 0 and gpu_num > 4 else 4

    def _compare_params(
        self,
        local_module,
        dist_module,
        rank0_only,
        skip_rowwise_bias=False,
        compare_grad=False,
    ):
        # [FUNCTION DOC] HELPER: For each named parameter of local_module, fetches the same
        # parameter of dist_module, optionally switching both to their gradients, redistributes
        # the distributed one to Replicate and asserts equality with the local value. Some
        # parameters (net2.bias, or any bias when skip_rowwise_bias) are only compared on rank 0
        # when rank0_only is set. Raises AssertionError on mismatch.
        replicate = [Replicate()]
        for name, param in local_module.named_parameters():
            dist_param = dist_module.get_parameter(name)
            param = param.grad if compare_grad else param
            dist_param = dist_param.grad if compare_grad else dist_param
            if (
                (not rank0_only)
                or (self.rank == 0)
                or (
                    name != "net2.bias"
                    and not skip_rowwise_bias
                    or name not in ["bias", "net2.bias"]
                )
            ):
                self.assertEqual(
                    param,
                    dist_param.redistribute(
                        device_mesh=dist_param.device_mesh, placements=replicate
                    ).to_local(),
                    lambda msg: f"{msg}\n{name} not equal between dist and non-dist",
                )

    def _compare_module(
        self, local_module, dist_module, inp_size, rank0_only=True, rowwise=False
    ):
        # [FUNCTION DOC] HELPER: Runs a full numerical check of a parallelized module against
        # its local original. Creates SGD optimizers (lr 0.25), compares parameters, runs a
        # forward on a seeded random input (chunked along the last dim per rank when rowwise),
        # compares outputs (DTensor output is redistributed to Replicate first), runs backward
        # on both, compares gradients, steps both optimizers and compares parameters again.
        LR = 0.25  # the learning rate we use for testing
        local_optim = torch.optim.SGD(local_module.parameters(), lr=LR)
        dist_optim = torch.optim.SGD(dist_module.parameters(), lr=LR)
        torch.manual_seed(0)
        inp = torch.rand(*inp_size, device=self.device_type)
        self._compare_params(local_module, dist_module, rank0_only)

        # check forward correctness
        local_output = local_module(inp)
        inp = map_local_tensor_for_rank(
            inp,
            self.rank,
            lambda inp, rank: inp.chunk(self.world_size, dim=-1)[rank]
            if rowwise
            else inp,
        )
        # inp = inp.chunk(self.world_size, dim=-1)[self.rank] if rowwise else inp
        dist_output = dist_module(inp)
        dist_output = (
            dist_output.redistribute(dist_output.device_mesh, [Replicate()]).to_local()
            if isinstance(dist_output, DTensor)
            else dist_output
        )
        self.assertEqual(local_output, dist_output)

        local_output.sum().backward()
        dist_output.sum().backward()

        # check backward and ensure gradients are same
        self._compare_params(local_module, dist_module, rank0_only, rowwise, True)

        local_optim.step()
        dist_optim.step()
        self._compare_params(local_module, dist_module, rank0_only, rowwise)

    @with_comms
    def test_parallelize_mlp_with_module_api(self):
        # [FUNCTION DOC] Parallelizes an MLP with a name-to-style plan making net1 and net2
        # column-wise with replicated output, then checks it matches the local model.
        inp_size = [12, 10]
        model = MLPModule(self.device_type)
        model_tp = deepcopy(model)

        # Parallelize module.
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        model_tp = parallelize_module(
            model_tp,
            device_mesh,
            {
                "net1": ColwiseParallel(output_layouts=Replicate()),
                "net2": ColwiseParallel(output_layouts=Replicate()),
            },
        )
        self._compare_module(model, model_tp, inp_size, rank0_only=False)

    @with_comms
    def test_parallelize_mlp_with_module_api_nested(self):
        # [FUNCTION DOC] Same as the flat MLP test but the MLP sits in a Sequential, so plan keys
        # use dotted paths through the child named dummy_encoder.
        inp_size = [12, 10]
        model = torch.nn.Sequential(
            OrderedDict([("dummy_encoder", MLPModule(self.device_type))])
        )
        model_tp = deepcopy(model)

        # Parallelize module.
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        model_tp = parallelize_module(
            model_tp,
            device_mesh,
            {
                "dummy_encoder.net1": ColwiseParallel(output_layouts=Replicate()),
                "dummy_encoder.net2": ColwiseParallel(output_layouts=Replicate()),
            },
        )
        self._compare_module(model, model_tp, inp_size, rank0_only=False)

    @with_comms
    def test_linear_row_wise_parallel(self):
        # [FUNCTION DOC] Applies RowwiseParallel directly (single style, not a plan mapping) to
        # a Linear, giving each rank a different seeded input chunk, and compares with local.
        # test RowwiseParallel
        inp_size = [9, 16]
        rowwise = RowwiseParallel()

        torch.manual_seed(5)
        model = torch.nn.Linear(16, 10, device=self.device_type)
        model_tp = deepcopy(model)

        # parallelize model_tp
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        model_tp = parallelize_module(model_tp, device_mesh, rowwise)

        # let each rank generate unique local input
        torch.manual_seed(self.rank)
        self._compare_module(model, model_tp, inp_size, rowwise=True)

    @with_comms
    def test_linear_col_wise_parallel(self):
        # [FUNCTION DOC] Applies ColwiseParallel with replicated output directly to a Linear
        # and compares its behavior with the local module.
        # test ColwiseParallel
        inp_size = [8, 10]
        colwise = ColwiseParallel(output_layouts=Replicate())

        torch.manual_seed(5)
        model = torch.nn.Linear(10, 16, device=self.device_type)
        model_tp = deepcopy(model)

        # parallelize model_tp
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        model_tp = parallelize_module(model_tp, device_mesh, colwise)

        self._compare_module(model, model_tp, inp_size)

    @with_comms
    def test_prepare_module_input(self):
        # [FUNCTION DOC] Applies PrepareModuleInput (Shard(0) to Replicate) to an identity
        # module; the output is a replicated DTensor whose Shard(0) local part equals the input.
        module = DummyModule()
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        parallelize_module(
            module,
            device_mesh,
            PrepareModuleInput(
                input_layouts=Shard(0), desired_input_layouts=Replicate()
            ),
        )
        inp = torch.rand(5, 7, device=self.device_type)
        output = module(inp).redistribute(device_mesh, [Shard(0)]).to_local()
        self.assertEqual(inp, output)

    @with_comms
    def test_prepare_module_output(self):
        # [FUNCTION DOC] Applies PrepareModuleOutput (Replicate to Shard(0)) to an identity
        # module fed a replicated DTensor; the local output equals the Shard(0) local chunk.
        module = DummyModule()
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        parallelize_module(
            module,
            device_mesh,
            PrepareModuleOutput(
                output_layouts=Replicate(), desired_output_layouts=Shard(0)
            ),
        )
        torch.manual_seed(15)
        inp = torch.rand(16, 7, device=self.device_type)
        dtensor = DTensor.from_local(inp, device_mesh, [Replicate()], run_check=False)
        output = module(dtensor)
        inp = dtensor.redistribute(device_mesh, [Shard(0)]).to_local()
        self.assertEqual(inp, output)

    @with_comms
    def test_prepare_module_input_output(self):
        # [FUNCTION DOC] Applies PrepareModuleInputOutput to an identity module: input Shard(0)
        # to Replicate, output Replicate to Shard(1). The local output equals the Shard(1)
        # chunk of the input viewed as a Shard(0) DTensor.
        module = DummyModule()
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        parallelize_module(
            module,
            device_mesh,
            PrepareModuleInputOutput(
                input_layouts=Shard(0),
                desired_input_layouts=Replicate(),
                output_layouts=Replicate(),
                desired_output_layouts=Shard(1),
            ),
        )
        inp = torch.rand(5, 7, device=self.device_type)
        output = module(inp)
        inp = (
            DTensor.from_local(inp, device_mesh, [Shard(0)], run_check=False)
            .redistribute(device_mesh, [Shard(1)])
            .to_local()
        )
        self.assertEqual(inp, output)

    @with_comms
    def test_parallelize_module_with_star(self):
        # [FUNCTION DOC] A plan key with a star wildcard (net*) matches both MLP layers; the
        # result must match the local model.
        inp_size = [12, 10]
        model = MLPModule(self.device_type)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        model_tp = deepcopy(model)
        model_tp = parallelize_module(
            model_tp,
            device_mesh,
            {
                "net*": ColwiseParallel(output_layouts=Replicate()),
            },
        )
        self._compare_module(model, model_tp, inp_size, rank0_only=False)

    @with_comms
    def test_parallelize_module_src_data_rank(self):
        # [FUNCTION DOC] With a different seed per rank, parallelize_module with src_data_rank
        # 1 must communicate and rank 1's full parameters equal its original ones. With
        # src_data_rank None (column then row style) no collectives may occur.
        # set seed different for each rank
        torch.manual_seed(self.rank)
        model = MLPModule(self.device_type)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        comm_mode = CommDebugMode()

        # test src_data_rank == 1
        with comm_mode:
            model_tp = deepcopy(model)
            model_tp = parallelize_module(
                model_tp,
                device_mesh,
                {
                    "net*": ColwiseParallel(output_layouts=Replicate()),
                },
                src_data_rank=1,
            )

        self.assertTrue(comm_mode.get_total_counts() > 0)
        tp_full_params = [param.full_tensor() for param in model_tp.parameters()]
        if self.rank == 1:
            orig_model_params = list(model.parameters())
            for idx, param in enumerate(tp_full_params):
                self.assertEqual(param, orig_model_params[idx])

        # test src_data_rank == None
        model_tp_no_comm = deepcopy(model)
        with comm_mode:
            parallelize_module(
                model_tp_no_comm,
                device_mesh,
                {
                    "net1": ColwiseParallel(),
                    "net2": RowwiseParallel(),
                },
                src_data_rank=None,
            )
            self.assertEqual(comm_mode.get_total_counts(), 0)

    @with_comms
    def test_parallelize_module_with_question(self):
        # [FUNCTION DOC] A plan key using a single-character wildcard (net?) matches both MLP
        # layers; the result must match the local model.
        inp_size = [12, 10]
        model = MLPModule(self.device_type)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        model_tp = deepcopy(model)
        model_tp = parallelize_module(
            model_tp,
            device_mesh,
            {
                "net?": ColwiseParallel(output_layouts=Replicate()),
            },
        )
        self._compare_module(model, model_tp, inp_size, rank0_only=False)

    @with_comms
    def test_parallelize_module_with_digit(self):
        # [FUNCTION DOC] A plan key with a character range (net followed by a range 1-2)
        # matches both MLP layers; the result must match the local model.
        inp_size = [12, 10]
        model = MLPModule(self.device_type)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        model_tp = deepcopy(model)
        model_tp = parallelize_module(
            model_tp,
            device_mesh,
            {
                "net[1-2]": ColwiseParallel(output_layouts=Replicate()),
            },
        )
        self._compare_module(model, model_tp, inp_size, rank0_only=False)

    @with_comms
    def test_parallelize_module_multi_wildcard(self):
        # [FUNCTION DOC] On a stacked MLP, plan keys combining a wildcard over layer names with
        # a bracketed index select net1 (column-wise) and net2 (row-wise) in every layer.
        inp_size = [12, 10]
        model = MLPStacked(self.device_type, n_layers=2)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        model_tp = deepcopy(model)
        model_tp = parallelize_module(
            model_tp,
            device_mesh,
            {
                "layers.*.net[1]": ColwiseParallel(),
                "layers.*.net[2]": RowwiseParallel(),
            },
        )
        self._compare_module(model, model_tp, inp_size, rank0_only=False)

    @with_comms
    def test_parallelize_module_with_root_module(self):
        # [FUNCTION DOC] The empty-string plan key targets the root module with
        # PrepareModuleInputOutput, alongside sequence-sharded column and row styles on the
        # children; the whole model must match the local one.
        inp_size = [16, 10]
        model = MLPModule(self.device_type)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        model_tp = deepcopy(model)
        model_tp = parallelize_module(
            model_tp,
            device_mesh,
            {
                "": PrepareModuleInputOutput(
                    input_layouts=Replicate(),
                    desired_input_layouts=Shard(0),
                    output_layouts=Shard(0),
                    desired_output_layouts=Replicate(),
                ),
                "net1": ColwiseParallel(input_layouts=Shard(0)),
                "net2": RowwiseParallel(output_layouts=Shard(0)),
            },
        )
        self._compare_module(model, model_tp, inp_size, rank0_only=False)

    @with_comms
    def test_parallelize_module_with_no_match(self):
        # [FUNCTION DOC] Plan entries naming nonexistent submodules (net0.hello.world, net3)
        # must emit a UserWarning, while matching entries are still applied correctly.
        inp_size = [16, 10]
        model = MLPModule(self.device_type)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        model_tp = deepcopy(model)
        with self.assertWarns(UserWarning):
            model_tp = parallelize_module(
                model_tp,
                device_mesh,
                {
                    "net0.hello.world": ColwiseParallel(),
                    "net1": ColwiseParallel(),
                    "net2": RowwiseParallel(),
                    "net3": ColwiseParallel(),
                },
            )
        self._compare_module(model, model_tp, inp_size, rank0_only=False)

    @with_comms
    def test_under_devicemesh_context(self):
        # [FUNCTION DOC] Calls parallelize_module without an explicit mesh inside a DeviceMesh
        # context manager; the current mesh is used and the result matches the local Linear.
        # test ColwiseParallel
        inp_size = [8, 10]
        colwise = ColwiseParallel(output_layouts=Replicate())

        torch.manual_seed(5)
        model = torch.nn.Linear(10, 16, device=self.device_type)
        model_tp = deepcopy(model)

        # Call parallelize_module under DeviceMesh context.
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        with device_mesh:
            model_tp = parallelize_module(model_tp, parallelize_plan=colwise)

        self._compare_module(model, model_tp, inp_size)

    @with_comms
    def test_empty_plan(self):
        # [FUNCTION DOC] Calling parallelize_module with no plan must only emit a UserWarning
        # and not crash.
        torch.manual_seed(5)
        model = torch.nn.Linear(10, 16, device=self.device_type)

        # Call parallelize_module with empty plan.
        # Goal is not to crash.
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        with self.assertWarns(UserWarning):
            parallelize_module(model, device_mesh)


TensorParallelAPITestsWithLocalTensor = create_local_tensor_test_class(
    TensorParallelAPITests,
    skipped_tests=[
        # Uses mesh_scatter that has local rank dependent logic
        "test_parallelize_module_src_data_rank",
    ],
)


if __name__ == "__main__":
    run_tests()
