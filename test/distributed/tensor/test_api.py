# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for the public DTensor API: distribute_tensor (sharding, src_data_rank,
# error cases, uneven sharding), distribute_module (partition/input/output functions, dtype
# casting, meta-device init, requires_grad preservation) and the checkpoint hooks that must
# reject Partial placements. Defines the MyModel helper and DTensorAPITest plus a local-tensor
# variant (which skips the checkpoint test); also the module alias c10d_ops.

import tempfile

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorTestBase,
    map_local_tensor_for_rank,
    with_comms,
)


class MyModel(nn.Module):
    # [CLASS DOC] HELPER: Stack of n_layers square Linear layers (n_features in and out) held in
    # a Sequential named seq, created directly on the given device (may be meta).
    def __init__(self, n_features, n_layers, device):
        # [FUNCTION DOC] HELPER: Builds the Sequential of n_layers Linear(n_features,
        # n_features) layers on the given device.
        super().__init__()
        self.seq = nn.Sequential(
            *[nn.Linear(n_features, n_features, device=device) for _ in range(n_layers)]
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs the input through the Linear stack and returns it.
        return self.seq(x)

    def reset_parameters(self):
        # [FUNCTION DOC] HELPER: Re-initializes every Linear layer in place; no return value.
        for m in self.seq:
            m.reset_parameters()


c10d_ops = torch.ops.c10d


class DTensorAPITest(DTensorTestBase):
    # [CLASS DOC] Multi-process DTensor test case for distribute_tensor, distribute_module and
    # checkpoint guard behavior. Stateless; world size is fixed at 4 so 2D meshes can be built.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Returns the fixed world size of 4.
        # hard code world size to 4 as we need to test
        # at least with 2d mesh
        return 4

    @with_comms
    def test_distribute_tensor_rank(self):
        # [FUNCTION DOC] Checks distribute_tensor on a Shard(0) spec issues exactly one scatter,
        # gives the expected global/local sizes and keeps leaf/requires_grad; Shard(-1) is
        # normalized to dim 1. Without local tensors, also checks src_data_rank=1 yields rank
        # 1's data in the result. With src_data_rank None no communication occurs and local
        # shapes are 3x3 for every placement.
        comm_mode = CommDebugMode()

        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]

        for requires_grad in [True, False]:
            tensor_to_shard = torch.randn(
                3 * self.world_size, 3, requires_grad=requires_grad
            )
            with comm_mode:
                dist_tensor = distribute_tensor(
                    tensor_to_shard, device_mesh, shard_spec
                )
                self.assertEqual(comm_mode.get_comm_counts()[c10d_ops.scatter_], 1)
            self.assertEqual(dist_tensor.size(), torch.Size([3 * self.world_size, 3]))
            local_tensor = dist_tensor.to_local()
            self.assertEqual(local_tensor.size(), torch.Size([3, 3]))
            if requires_grad:
                self.assertTrue(dist_tensor.requires_grad)
                self.assertTrue(dist_tensor.is_leaf)

        # test negative dim
        shard_minus_spec = [Shard(-1)]
        tensor_to_shard = torch.randn(3, 3 * self.world_size)
        dist_tensor = distribute_tensor(tensor_to_shard, device_mesh, shard_minus_spec)
        self.assertEqual(dist_tensor.placements[0].dim, 1)

        placement_combs = [
            [Shard(0)],
            [Shard(1)],
            [Replicate()],
            [Partial(reduce_op="sum")],
            [Partial(reduce_op="avg")],
        ]

        if not self.is_local_tensor_enabled:
            # test src_data_rank == 1
            # set seed differently for each rank
            self.init_manual_seed_for_rank()
            for placement in placement_combs:
                tensor_to_distribute = torch.randn(
                    3 * self.world_size, 3 * self.world_size
                )
                dtensor = distribute_tensor(
                    tensor_to_distribute, device_mesh, placement, src_data_rank=1
                )
                full_dtensor = dtensor.full_tensor()
                if self.rank == 1:
                    self.assertEqual(full_dtensor, tensor_to_distribute)

        # test src_data_rank = None, make sure it does not have communication
        with comm_mode:
            for placement in placement_combs:
                if isinstance(placement[0], Shard):
                    shard_dim = placement[0].dim
                    shape = [3, 3]
                    shape[shard_dim] *= self.world_size
                    tensor_to_distribute = torch.randn(*shape)
                else:
                    tensor_to_distribute = torch.randn(3, 3)

                dtensor = distribute_tensor(
                    tensor_to_distribute, device_mesh, placement, src_data_rank=None
                )
                self.assertEqual(dtensor.to_local().shape, (3, 3))
        self.assertEqual(comm_mode.get_total_counts(), 0)

    @with_comms
    def test_distribute_tensor_errors(self):
        # [FUNCTION DOC] On a 2x2 mesh checks distribute_tensor errors: placements length not
        # matching mesh ndim (ValueError), unsupported Partial prod conversion (ValueError),
        # non-leaf tensor requiring grad (RuntimeError), and redistributing an existing DTensor
        # to a different mesh or different placements (ValueError).
        device_mesh = DeviceMesh(
            self.device_type, torch.arange(self.world_size).reshape(2, 2)
        )
        tensor_shape = [3 * self.world_size, 3 * self.world_size]
        tensor_to_distribute = torch.randn(*tensor_shape)

        with self.assertRaisesRegex(ValueError, "must have the same length"):
            shard_spec = [Shard(0)]
            distribute_tensor(tensor_to_distribute, device_mesh, shard_spec)

        with self.assertRaisesRegex(ValueError, "conversion is not supported"):
            new_spec = [Replicate(), Partial(reduce_op="prod")]
            distribute_tensor(tensor_to_distribute, device_mesh, new_spec)

        with self.assertRaisesRegex(RuntimeError, "distribute leaf tensor"):
            shard_spec = [Shard(0)]
            global_tensor = torch.randn(*tensor_shape, requires_grad=True)
            global_tensor_to_distribute = global_tensor + 2
            distribute_tensor(global_tensor_to_distribute, device_mesh, shard_spec)

        spec = [Shard(0), Shard(1)]
        dtensor = distribute_tensor(tensor_to_distribute, device_mesh, spec)

        with self.assertRaisesRegex(ValueError, "to a different device mesh"):
            new_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
            distribute_tensor(dtensor, new_mesh, [Shard(0)])

        with self.assertRaisesRegex(ValueError, "to a different placements"):
            new_spec = [Shard(0), Replicate()]
            distribute_tensor(dtensor, device_mesh, new_spec)

    @with_comms
    def test_distribute_tensor_uneven_sharding(self):
        # [FUNCTION DOC] Shards tensors whose sharded dim size is not divisible by the world
        # size (remainders of 1 and 2, dims 0-2); the global size is preserved and each rank's
        # local shard equals its chunk from torch.chunk.
        device_mesh = self.build_device_mesh()
        input_sizes_and_shard_dims = [
            ((self.world_size * 3 + 1, 3, 3), 0),
            ((self.world_size * 3 + 2, 3, 3), 0),
            ((3, self.world_size * 3 + 1, 3), 1),
            ((3, self.world_size * 3 + 2, 3), 1),
            ((3, 3, self.world_size * 3 + 1), 2),
            ((3, 3, self.world_size * 3 + 2), 2),
        ]
        for input_size, shard_dim in input_sizes_and_shard_dims:
            shard_spec = [Shard(shard_dim)]
            tensor_to_shard = torch.randn(input_size)
            splitted_tensor_list = list(
                torch.chunk(tensor_to_shard, self.world_size, dim=shard_dim)
            )
            dist_tensor = distribute_tensor(tensor_to_shard, device_mesh, shard_spec)
            self.assertEqual(dist_tensor.size(), torch.Size(input_size))
            local_tensor = dist_tensor.to_local()
            self.assertEqual(
                local_tensor,
                map_local_tensor_for_rank(
                    splitted_tensor_list, self.rank, lambda tl, r: tl[r]
                ),
            )

    @with_comms
    def test_distribute_module(self):
        # [FUNCTION DOC] Exercises distribute_module partition functions: shard all Linear
        # params on dim 0; no partition_fn replicates everything; an explicit replicate function;
        # and sharding only seq.0 and seq.8 while all other parameters end up replicated.
        # Every parameter must become a DTensor with the expected placements.
        device_mesh = self.build_device_mesh()
        # fully shard all linear modules on dim 0
        module_to_shard = MyModel(5 * self.world_size, 20, device=self.device_type)
        shard_spec = [Shard(0)]

        def shard_fn(name, module, device_mesh):
            # [FUNCTION DOC] HELPER: Partition function replacing every Linear parameter with
            # a Parameter wrapping a dim-0 sharded DTensor, registered in place.
            if isinstance(module, nn.Linear):
                for name, param in module.named_parameters():
                    dist_param = torch.nn.Parameter(
                        distribute_tensor(param, device_mesh, shard_spec)
                    )
                    module.register_parameter(name, dist_param)

        sharded_module = distribute_module(module_to_shard, device_mesh, shard_fn)
        for param in sharded_module.parameters():
            self.assertIsInstance(param, DTensor)
            self.assertEqual(param.placements, shard_spec)

        replica_spec = [Replicate()]
        # fully replicate all modules without passing in partition_fn
        module_to_replicate = MyModel(5, 20, device=self.device_type)
        replica_module = distribute_module(module_to_replicate, device_mesh)
        for param in replica_module.parameters():
            self.assertIsInstance(param, DTensor)
            self.assertEqual(param.placements, replica_spec)

        # fully replicate all modules by passing in partition_fn
        def replicate_fn(name, module, device_mesh):
            # [FUNCTION DOC] HELPER: Partition function replacing every Linear parameter with
            # a Parameter wrapping a replicated DTensor, registered in place.
            if isinstance(module, nn.Linear):
                for name, param in module.named_parameters():
                    dist_param = torch.nn.Parameter(
                        distribute_tensor(param, device_mesh, replica_spec)
                    )
                    module.register_parameter(name, dist_param)

        module_to_replicate = MyModel(5, 20, device=self.device_type)
        replica_module = distribute_module(
            module_to_replicate, device_mesh, replicate_fn
        )
        for param in replica_module.parameters():
            self.assertIsInstance(param, DTensor)
            self.assertEqual(param.placements, replica_spec)

        # only shard part of module, and rest of module should be replicate
        def shard_fn(name, module, device_mesh):
            # [FUNCTION DOC] HELPER: Shards on dim 0 only the Linear modules named seq.0 and
            # seq.8; other modules are left for distribute_module to replicate.
            if isinstance(module, nn.Linear) and (name == "seq.0" or name == "seq.8"):
                for name, param in module.named_parameters():
                    dist_param = torch.nn.Parameter(
                        distribute_tensor(param, device_mesh, shard_spec)
                    )
                    module.register_parameter(name, dist_param)

        module_to_distribute = MyModel(5 * self.world_size, 20, device=self.device_type)
        dist_module = distribute_module(module_to_distribute, device_mesh, shard_fn)
        for name, param in dist_module.named_parameters():
            self.assertIsInstance(param, DTensor)
            if name.startswith(("seq.0", "seq.8")):
                self.assertEqual(param.placements, shard_spec)
            else:
                self.assertEqual(param.placements, replica_spec)

    @with_comms
    def test_distribute_module_input_fn_output_fn(self):
        # [FUNCTION DOC] Uses the three-argument input_fn/output_fn hooks: input is wrapped as
        # a Shard(0) DTensor and output converted back to a plain local tensor. Also checks a
        # replicated-input model yields a Replicate DTensor gradient on its parameters.
        device_mesh = self.build_device_mesh()

        # fully replicate all linear modules
        module_to_replicate = MyModel(20, 1, device=self.device_type)

        # mark input sharding on dim 0
        def input_fn(mod, inputs, device_mesh):
            # [FUNCTION DOC] HELPER: Wraps the first input as a Shard(0) DTensor.
            return DTensor.from_local(inputs[0], device_mesh, [Shard(0)])

        def output_fn(mod, outputs, device_mesh):
            # [FUNCTION DOC] HELPER: Raises AssertionError unless outputs is a DTensor;
            # returns its local tensor.
            if not isinstance(outputs, DTensor):
                raise AssertionError(f"Expected DTensor, got {type(outputs)}")
            return outputs.to_local()

        replica_module = distribute_module(
            module_to_replicate,
            device_mesh,
            input_fn=input_fn,
            output_fn=output_fn,
        )

        input_tensor = torch.randn(5, 20, device=self.device_type)
        local_out = replica_module(input_tensor)
        self.assertIsInstance(local_out, torch.Tensor)
        self.assertNotIsInstance(local_out, DTensor)

        # full replicate (even on inputs)
        model = MyModel(10, 10, device=self.device_type)

        def replicate_input_fn(mod, inputs, device_mesh):
            # [FUNCTION DOC] HELPER: Wraps the first input as a replicated DTensor.
            return DTensor.from_local(inputs[0], device_mesh, [Replicate()])

        replica_model = distribute_module(
            model,
            device_mesh,
            input_fn=replicate_input_fn,
        )
        input = torch.randn(10, 10, requires_grad=True)
        output = replica_model(input)
        output.sum().backward()
        param_grad = next(iter(replica_model.parameters())).grad
        self.assertTrue(isinstance(param_grad, DTensor))
        self.assertTrue(isinstance(param_grad.placements[0], Replicate))

    @with_comms
    def test_distribute_module_preserves_requires_grad(self):
        # [FUNCTION DOC] Distributes a module with one frozen and one trainable parameter and
        # checks the requires_grad flags survive; after backward only the trainable one has
        # a gradient.
        device_mesh = self.build_device_mesh()

        class ModelWithFrozenParam(nn.Module):
            # [CLASS DOC] HELPER: Holds two 10x10 parameters, frozen and trainable, both added
            # to the input in forward.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the frozen and trainable random parameters.
                super().__init__()
                self.frozen = nn.Parameter(torch.randn(10, 10), requires_grad=False)
                self.trainable = nn.Parameter(torch.randn(10, 10), requires_grad=True)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus both parameters.
                return x + self.frozen + self.trainable

        model = ModelWithFrozenParam().to(self.device_type)

        distributed_model = distribute_module(model, device_mesh)

        self.assertFalse(distributed_model.frozen.requires_grad)
        self.assertTrue(distributed_model.trainable.requires_grad)

        x = DTensor.from_local(
            torch.randn(10, 10, device=self.device_type),
            device_mesh,
            [Replicate()],
        )
        output = distributed_model(x)
        output.sum().backward()

        self.assertIsNone(distributed_model.frozen.grad)
        self.assertIsNotNone(distributed_model.trainable.grad)

    @with_comms
    def test_distribute_module_input_fn_output_fn_warning(self):
        # [FUNCTION DOC] Passes the deprecated two-argument input_fn/output_fn signatures to
        # distribute_module and expects a FutureWarning mentioning deprecation; the module
        # still works and returns a plain local tensor.
        device_mesh = self.build_device_mesh()

        # fully replicate all linear modules
        module_to_replicate = MyModel(20, 1, device=self.device_type)

        # mark input sharding on dim 0
        def input_fn(inputs, device_mesh):
            # [FUNCTION DOC] HELPER: Legacy-signature hook wrapping the first input as Shard(0).
            return DTensor.from_local(inputs[0], device_mesh, [Shard(0)])

        def output_fn(outputs, device_mesh):
            # [FUNCTION DOC] HELPER: Legacy-signature hook; raises AssertionError unless
            # outputs is a DTensor, returns its local tensor.
            if not isinstance(outputs, DTensor):
                raise AssertionError(f"Expected DTensor, got {type(outputs)}")
            return outputs.to_local()

        with self.assertWarnsRegex(FutureWarning, "Deprecating"):
            replica_module = distribute_module(
                module_to_replicate,
                device_mesh,
                input_fn=input_fn,
                output_fn=output_fn,
            )

        input_tensor = torch.randn(5, 20, device=self.device_type)
        local_out = replica_module(input_tensor)
        self.assertIsInstance(local_out, torch.Tensor)
        self.assertNotIsInstance(local_out, DTensor)

    @with_comms
    def test_distribute_module_casting(self):
        # [FUNCTION DOC] Checks dtype casting to bfloat16 for a DTensor, a distribute_tensor
        # result and a distributed module (both wrapper and local tensor dtypes change), and
        # that running a replicated module under autocast yields bfloat16 output.
        device_mesh = self.build_device_mesh()

        # check DTensor casting
        dt = DTensor.from_local(torch.rand(10), device_mesh, [Replicate()])
        dt = dt.to(torch.bfloat16)
        self.assertEqual(dt.dtype, torch.bfloat16)
        self.assertEqual(dt._local_tensor.dtype, torch.bfloat16)

        # check distribute_tensor casting
        dt = distribute_tensor(torch.rand(10), device_mesh, [Replicate()])
        dt = dt.to(torch.bfloat16)
        self.assertEqual(dt.dtype, torch.bfloat16)
        self.assertEqual(dt._local_tensor.dtype, torch.bfloat16)

        # check distribute_module casting
        model = MyModel(10, 10, device=self.device_type)
        replica_model = distribute_module(
            model,
            device_mesh,
        )
        replica_model = replica_model.to(torch.bfloat16)
        self.assertEqual(replica_model.seq[0].weight.dtype, torch.bfloat16)
        self.assertEqual(
            replica_model.seq[0].weight._local_tensor.dtype, torch.bfloat16
        )

        # check autocast
        # `distribute_module` is an in-place operation, so we need to create a
        # new model
        model = MyModel(10, 10, device=self.device_type)
        dt = distribute_tensor(torch.rand(10), device_mesh, [Replicate()])
        replica_model = distribute_module(
            model,
            device_mesh,
        )
        with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
            output = replica_model(dt)
        self.assertEqual(output.dtype, torch.bfloat16)

    @with_comms
    def test_distribute_module_meta(self):
        # [FUNCTION DOC] Builds the model on the meta device and uses a partition function that
        # shards each parameter then materializes it empty on the mesh device; afterwards all
        # parameters must be non-meta DTensors on the mesh device type.
        # If  the model is too big, the user may first the create entire model on the meta device and then initialize
        # it on the device in the partition function.
        device_mesh = self.build_device_mesh()

        # fully shard all parameters on dim 0
        module_to_shard = MyModel(5 * self.world_size, 20, device="meta")

        shard_spec = [Shard(0)]

        def shard_fn(name, module, device_mesh):
            # [FUNCTION DOC] HELPER: For each directly owned parameter, shards it on dim 0,
            # allocates empty storage on the mesh device and re-registers it as a Parameter.
            for param_name, param in module._parameters.items():
                dist_param = distribute_tensor(param, device_mesh, shard_spec)
                dist_param = torch.empty_like(
                    dist_param, device=device_mesh.device_type
                )
                module.register_parameter(param_name, torch.nn.Parameter(dist_param))

        sharded_module = distribute_module(module_to_shard, device_mesh, shard_fn)
        for param in sharded_module.parameters():
            self.assertIsInstance(param, DTensor)
            self.assertFalse(param.is_meta)
            self.assertTrue(param.device.type == device_mesh.device_type)

    @with_comms
    def test_checkpoint_apis_check_partial_placement(self):
        # [FUNCTION DOC] A Partial-placed DTensor must reject the checkpoint hooks (create write
        # items, create chunk list, get tensor shard) with ValueError, and dcp.save must fail
        # with CheckpointException. Writes to a fresh temp directory.
        device_mesh = self.build_device_mesh()
        tensor = torch.randn(5, 5, device=self.device_type)
        dtensor = DTensor.from_local(tensor, device_mesh, [Partial()])
        with self.assertRaisesRegex(
            ValueError, "Any checkpointing related operations are not supported for"
        ):
            dtensor.__create_write_items__("fqn", None)

        with self.assertRaisesRegex(
            ValueError, "Any checkpointing related operations are not supported for"
        ):
            dtensor.__create_chunk_list__()

        with self.assertRaisesRegex(
            ValueError, "Any checkpointing related operations are not supported for"
        ):
            dtensor.__get_tensor_shard__(0)

        # Ideally we should not allow checkpointing related operations for DTensor
        with self.assertRaisesRegex(
            dcp.api.CheckpointException,
            "Any checkpointing related operations are not supported for",
        ):
            dcp.save({"fqn": dtensor}, checkpoint_id=tempfile.mkdtemp())


DTensorAPITestWithLocalTensor = create_local_tensor_test_class(
    DTensorAPITest, skipped_tests=["test_checkpoint_apis_check_partial_placement"]
)

if __name__ == "__main__":
    run_tests()
