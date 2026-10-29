# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Numerics tests comparing torch.optim optimizers (Adam, AdamW, SGD, Adagrad,
# RMSprop, Adadelta, NAdam, RAdam, Adamax, ASGD) on a plain MLP against the same optimizer on
# a DTensor-sharded copy, plus a mixed-mesh fused AdamW test and a sharding-propagation cache
# leak test. Defines module-level shard, input and output functions for distribute_module and
# creates a local-tensor variant of the test class at import time.

from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributed.tensor import (
    distribute_module,
    distribute_tensor,
    DTensor,
    init_device_mesh,
    Replicate,
    Shard,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorTestBase,
    MLPModule,
    with_comms,
)


# shard function to do full sharding on all parameters of a module
def shard_fn(name, module, device_mesh):
    # HELPER: distribute_module partition function: for each nn.Linear module, replaces every
    # parameter with a Parameter wrapping a DTensor sharded on dim 0 over the mesh, registers a
    # gradient hook redistributing grads to Shard(0) (clearing Partial), and re-registers it.
    # Other module types are left untouched. Mutates the module in place.
    if isinstance(module, nn.Linear):
        for name, param in module.named_parameters():
            dist_param = torch.nn.Parameter(
                distribute_tensor(param, device_mesh, [Shard(0)])
            )
            # make sure partial sum get cleared after backward()
            dist_param.register_hook(
                lambda grad: grad.redistribute(placements=[Shard(0)])
            )
            module.register_parameter(name, dist_param)


# prepare input
def input_fn(mod, inputs, device_mesh):
    # HELPER: distribute_module input function: shards the first forward input on dim 0 over
    # the mesh and returns the DTensor.
    # split the input tensor to be sharded input
    dist_inp = distribute_tensor(inputs[0], device_mesh, [Shard(0)])
    return dist_inp


# prepare output to be local torch.Tensor
def output_fn(mod, outputs, device_mesh):
    # HELPER: distribute_module output function: raises AssertionError if the output is not
    # a DTensor; otherwise redistributes it to Replicate on every mesh dim and returns the
    # local tensor.
    if not isinstance(outputs, DTensor):
        raise AssertionError(f"Expected DTensor, got {type(outputs)}")
    return outputs.redistribute(placements=[Replicate()] * device_mesh.ndim).to_local()


class TestDTensorOptimizer(DTensorTestBase):
    # [CLASS DOC] Compares DTensor-sharded optimizer steps with single-device ones for several
    # optimizer types and configurations on a 1D mesh (and one mixed 1D/2D mesh case), and
    # checks the sharding-propagation cache does not grow across AdamW steps.
    def _assert_optimizer(
        self,
        mesh,
        model,
        optim,
        dist_model,
        dist_optim,
        inputs,
        *,
        rtol: float = 1.3e-6,
        atol: float = 1e-5,
    ):
        # HELPER: Runs two iterations on both the reference model and the distributed model:
        # zero_grad (set_to_none on even iterations), forward on the same input, backward on summed
        # output and optimizer step; then asserts each distributed parameter's full tensor equals
        # the reference parameter within the given rtol and atol. The mesh argument is unused.
        for iter_idx in range(2):
            # run forward/backward/optim for original model
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            out = model(inputs)
            loss = out.sum()
            loss.backward()
            optim.step()

            # run forward/backward/optim for distributed model
            dist_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            dist_out = dist_model(inputs)
            dist_loss = dist_out.sum()
            dist_loss.backward()
            dist_optim.step()

            # check that the optimizer update parameters with same numerics
            for p1, p2 in zip(model.parameters(), dist_model.parameters()):
                p2 = p2.full_tensor()
                # Default 'rtol' and 'atol' for attr:`~torch.float32` are ``1.3e-6`` and ``1e-5``
                self.assertEqual(p1, p2, atol=atol, rtol=rtol)

    def test_optimizer_foreach_supported_types_include_DTensor(self):
        # [FUNCTION DOC] Asserts DTensor is registered in the optimizer's foreach-supported types.
        from torch.optim.optimizer import _foreach_supported_types

        self.assertTrue(DTensor in _foreach_supported_types)

    @with_comms
    def test_adam_1d_sharding(self):
        # [FUNCTION DOC] For many Adam configs (foreach off/on, weight decay, amsgrad, maximize,
        # fused with float or tensor learning rates), builds an MLP and a sharded deep copy,
        # creates an optimizer for each with the same config and compares them via
        # _assert_optimizer using a ones input.
        mesh = self.build_device_mesh()

        # lr as a Tensor is not supported for capturable=False and foreach=True
        adam_float_lr_configs = [
            {"lr": 0.1, "foreach": False},
            {"lr": 0.1, "weight_decay": 0.05, "foreach": False},
            {"lr": 0.1, "weight_decay": 0.05},
            {"lr": 0.1, "weight_decay": 0.05, "amsgrad": True},
            {
                "lr": 0.1,
                "weight_decay": 0.05,
                "maximize": True,
                "amsgrad": True,
            },
        ]
        fused_adam_float_lr_configs = [
            {"lr": 0.1, "fused": True},
            {"lr": 0.1, "weight_decay": 0.05, "amsgrad": True, "fused": True},
            {
                "lr": 0.1,
                "weight_decay": 0.05,
                "maximize": True,
                "amsgrad": True,
                "fused": True,
            },
        ]
        # lr could be a Tensor or a float when fused=True for adam optimizer
        fused_adam_tensor_lr_configs = [
            {**config, "lr": torch.tensor(0.1)}
            for config in fused_adam_float_lr_configs
        ]
        fused_adam_tensor_lr_configs.extend(
            [
                {**config, "lr": torch.tensor([0.1])}
                for config in fused_adam_float_lr_configs
            ]
        )
        adam_configs = [
            *adam_float_lr_configs,
            *fused_adam_float_lr_configs,
            *fused_adam_tensor_lr_configs,
        ]

        for config in adam_configs:
            mod = MLPModule(self.device_type)
            opt = torch.optim.Adam(mod.parameters(), **config)

            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            dist_opt = torch.optim.Adam(dist_mod.parameters(), **config)

            # use ones to make sure the single machine model have the same input
            # on different ranks
            inp = torch.ones(8, 10, device=self.device_type)
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)

    @with_comms
    def test_adamw_1d_sharding(self):
        # [FUNCTION DOC] Same as the Adam test but for AdamW, with betas and eps variants in the
        # configs, including fused configs with float or tensor learning rates.
        mesh = self.build_device_mesh()

        # lr as a Tensor is not supported for capturable=False and foreach=True
        adamw_float_lr_configs = [
            {"lr": 0.1, "foreach": False},
            {"lr": 0.1, "weight_decay": 0.05, "foreach": False},
            {"lr": 0.1, "weight_decay": 0.05},
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "amsgrad": True,
            },
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "maximize": True,
                "amsgrad": True,
            },
        ]
        fused_adamw_float_lr_configs = [
            {"lr": 0.1, "weight_decay": 0.05, "fused": True},
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "amsgrad": True,
                "fused": True,
            },
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "maximize": True,
                "amsgrad": True,
                "fused": True,
            },
        ]
        # lr could be a Tensor or a float when fused=True for adamW optimizer
        fused_adamw_tensor_lr_configs = [
            {**config, "lr": torch.tensor(0.1)}
            for config in fused_adamw_float_lr_configs
        ]
        fused_adamw_tensor_lr_configs.extend(
            [
                {**config, "lr": torch.tensor([0.1])}
                for config in fused_adamw_float_lr_configs
            ]
        )
        adamw_configs = [
            *adamw_float_lr_configs,
            *fused_adamw_float_lr_configs,
            *fused_adamw_tensor_lr_configs,
        ]

        for config in adamw_configs:
            mod = MLPModule(self.device_type)
            opt = torch.optim.AdamW(mod.parameters(), **config)

            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            dist_opt = torch.optim.AdamW(dist_mod.parameters(), **config)

            # use ones to make sure the single machine model have the same input
            # on different ranks
            inp = torch.ones(8, 10, device=self.device_type)
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)

    @with_comms
    def test_sgd_1d_sharding(self):
        # [FUNCTION DOC] Compares SGD (momentum, dampening, nesterov, weight decay, maximize,
        # foreach on and off) between the plain and sharded MLP.
        mesh = self.build_device_mesh()

        sgd_configs = [
            {"lr": 0.1, "foreach": False},
            {"lr": 0.1, "momentum": 0.05, "foreach": False},
            {"lr": 0.1, "momentum": 0.05},
            {"lr": 0.1, "momentum": 0.06, "dampening": 0.07},
            {
                "lr": 0.1,
                "momentum": 0.08,
                "weight_decay": 0.05,
                "nesterov": True,
                "maximize": True,
                "foreach": False,
            },
            {
                "lr": 0.1,
                "momentum": 0.08,
                "weight_decay": 0.05,
                "nesterov": True,
                "maximize": True,
            },
        ]

        for config in sgd_configs:
            mod = MLPModule(self.device_type)
            opt = torch.optim.SGD(mod.parameters(), **config)

            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            dist_opt = torch.optim.SGD(dist_mod.parameters(), **config)

            # use ones to make sure the single machine model have the same input
            # on different ranks
            inp = torch.ones(8, 10, device=self.device_type)
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)

    @with_comms
    def test_adagrad_1d_sharding(self):
        # [FUNCTION DOC] Compares Adagrad (lr decay, weight decay, initial accumulator, eps,
        # maximize, mostly foreach disabled) between the plain and sharded MLP.
        mesh = self.build_device_mesh()

        adagrad_configs = [
            {"lr": 0.1, "foreach": False},
            {"lr": 0.1, "lr_decay": 0.05, "foreach": False},
            {"lr": 0.1, "lr_decay": 0.02, "weight_decay": 0.05, "foreach": False},
            {
                "lr": 0.1,
                "lr_decay": 0.02,
                "weight_decay": 0.05,
                "initial_accumulator_value": 0.03,
                "foreach": False,
            },
            {
                "lr": 0.1,
                "lr_decay": 0.02,
                "weight_decay": 0.05,
                "initial_accumulator_value": 0.03,
                "eps": 1e-6,
                "foreach": False,
            },
            {
                "lr": 0.1,
                "lr_decay": 0.02,
                "weight_decay": 0.05,
                "initial_accumulator_value": 0.03,
                "eps": 1e-6,
                "maximize": True,
                "foreach": False,
            },
            {
                "lr": 0.1,
                "lr_decay": 0.02,
                "weight_decay": 0.05,
                "initial_accumulator_value": 0.03,
                "eps": 1e-6,
                "maximize": True,
            },
        ]

        for config in adagrad_configs:
            mod = MLPModule(self.device_type)
            opt = torch.optim.Adagrad(mod.parameters(), **config)

            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            dist_opt = torch.optim.Adagrad(dist_mod.parameters(), **config)

            # use ones to make sure the single machine model have the same input
            # on different ranks
            inp = torch.ones(8, 10, device=self.device_type)
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)

    @with_comms
    def test_RMSprop_1d_sharding(self):
        # [FUNCTION DOC] Compares RMSprop (alpha, eps, weight decay, momentum, centered, maximize;
        # foreach disabled except for the last config) between the plain and sharded MLP.
        mesh = self.build_device_mesh()

        RMSprop_configs = [
            {"lr": 0.1, "foreach": False},
            {"lr": 0.1, "alpha": 0.85, "foreach": False},
            {"lr": 0.1, "alpha": 0.88, "eps": 1e-6, "foreach": False},
            {
                "lr": 0.1,
                "alpha": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "foreach": False,
            },
            {
                "lr": 0.1,
                "alpha": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "momentum": 0.9,
                "foreach": False,
            },
            {
                "lr": 0.1,
                "alpha": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "momentum": 0.9,
                "centered": True,
                "foreach": False,
            },
            {
                "lr": 0.1,
                "alpha": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "momentum": 0.9,
                "centered": True,
                "maximize": True,
                "foreach": False,
            },
            {
                "lr": 0.1,
                "alpha": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "momentum": 0.9,
                "centered": True,
                "maximize": True,
            },
        ]

        for config in RMSprop_configs:
            mod = MLPModule(self.device_type)
            opt = torch.optim.RMSprop(mod.parameters(), **config)

            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            dist_opt = torch.optim.RMSprop(dist_mod.parameters(), **config)

            # use ones to make sure the single machine model have the same input
            # on different ranks
            inp = torch.ones(8, 10, device=self.device_type)
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)

    @with_comms
    def test_adadelta_1d_sharding(self):
        # [FUNCTION DOC] Compares Adadelta (rho, eps, weight decay, maximize, foreach on and off)
        # between the plain and sharded MLP.
        mesh = self.build_device_mesh()

        adadelta_configs = [
            {"lr": 0.1, "foreach": False},
            {"lr": 0.1, "rho": 0.85, "foreach": False},
            {"lr": 0.1, "rho": 0.88, "eps": 1e-5, "foreach": False},
            {
                "lr": 0.1,
                "rho": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "foreach": False,
            },
            {
                "lr": 0.1,
                "rho": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
            },
            {
                "lr": 0.1,
                "rho": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "maximize": True,
            },
        ]

        for config in adadelta_configs:
            mod = MLPModule(self.device_type)
            opt = torch.optim.Adadelta(mod.parameters(), **config)

            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            dist_opt = torch.optim.Adadelta(dist_mod.parameters(), **config)

            # use ones to make sure the single machine model have the same input
            # on different ranks
            inp = torch.ones(8, 10, device=self.device_type)
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)

    @with_comms
    def test_nadam_1d_sharding(self):
        # [FUNCTION DOC] Compares NAdam (betas, eps, weight decay, decoupled weight decay)
        # between the plain and sharded MLP.
        mesh = self.build_device_mesh()

        nadam_configs = [
            {"lr": 0.1, "foreach": False},
            {"lr": 0.1, "weight_decay": 0.05, "foreach": False},
            {"lr": 0.1, "weight_decay": 0.05},
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
            },
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "decoupled_weight_decay": True,
            },
        ]

        for config in nadam_configs:
            mod = MLPModule(self.device_type)
            opt = torch.optim.NAdam(mod.parameters(), **config)

            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            dist_opt = torch.optim.NAdam(dist_mod.parameters(), **config)

            # use ones to make sure the single machine model have the same input
            # on different ranks
            inp = torch.ones(8, 10, device=self.device_type)
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)

    @with_comms
    def test_radam_1d_sharding(self):
        # [FUNCTION DOC] Compares RAdam (betas, eps, weight decay, decoupled weight decay)
        # between the plain and sharded MLP.
        mesh = self.build_device_mesh()

        radam_configs = [
            {"lr": 0.1, "foreach": False},
            {"lr": 0.1, "weight_decay": 0.05, "foreach": False},
            {
                "lr": 0.1,
                "weight_decay": 0.05,
            },
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
            },
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "decoupled_weight_decay": True,
            },
        ]

        for config in radam_configs:
            mod = MLPModule(self.device_type)
            opt = torch.optim.RAdam(mod.parameters(), **config)

            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            dist_opt = torch.optim.RAdam(dist_mod.parameters(), **config)

            # use ones to make sure the single machine model have the same input
            # on different ranks
            inp = torch.ones(8, 10, device=self.device_type)
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)

    @with_comms
    def test_adamax_1d_sharding(self):
        # [FUNCTION DOC] Compares Adamax (betas, eps, weight decay, maximize, foreach on and off)
        # between the plain and sharded MLP.
        mesh = self.build_device_mesh()

        adamax_configs = [
            {"lr": 0.1, "foreach": False},
            {"lr": 0.1, "betas": (0.6, 0.66), "foreach": False},
            {"lr": 0.1, "betas": (0.6, 0.66), "eps": 1e-6, "foreach": False},
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "foreach": False,
            },
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
            },
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "maximize": True,
            },
        ]

        for config in adamax_configs:
            mod = MLPModule(self.device_type)
            opt = torch.optim.Adamax(mod.parameters(), **config)

            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            dist_opt = torch.optim.Adamax(dist_mod.parameters(), **config)

            # use ones to make sure the single machine model have the same input
            # on different ranks
            inp = torch.ones(8, 10, device=self.device_type)
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)

    @with_comms
    def test_asgd_1d_sharding(self):
        # [FUNCTION DOC] Compares ASGD (lambd, alpha, t0, weight decay, maximize, foreach on and
        # off) between the plain and sharded MLP, using looser tolerances (atol 1.3e-5, rtol 1e-4).
        # KNOWN ISSUE: a TODO in the code notes ASGD needs higher tolerances for unknown reasons.
        mesh = self.build_device_mesh()

        asgd_configs = [
            {"lr": 0.1, "foreach": False},
            {"lr": 0.1, "lambd": 0.001, "foreach": False},
            {"lr": 0.1, "lambd": 0.001, "alpha": 0.85, "foreach": False},
            {"lr": 0.1, "lambd": 0.001, "alpha": 0.85, "t0": 1e5, "foreach": False},
            {
                "lr": 0.1,
                "lambd": 0.001,
                "alpha": 0.85,
                "t0": 1e5,
                "weight_decay": 0.05,
                "foreach": False,
            },
            {
                "lr": 0.1,
                "lambd": 0.001,
                "alpha": 0.85,
                "t0": 1e5,
                "weight_decay": 0.05,
                "foreach": True,
            },
            {
                "lr": 0.1,
                "lambd": 0.001,
                "alpha": 0.85,
                "t0": 1e5,
                "weight_decay": 0.05,
                "foreach": True,
                "maximize": True,
            },
        ]

        for config in asgd_configs:
            mod = MLPModule(self.device_type)
            opt = torch.optim.ASGD(mod.parameters(), **config)

            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            dist_opt = torch.optim.ASGD(dist_mod.parameters(), **config)

            # use ones to make sure the single machine model have the same input
            # on different ranks
            inp = torch.ones(8, 10, device=self.device_type)

            # TODO: We want to keep a unit test for ASGD optimizer for the time being, but we need to look into why
            # when using ASGD we need higher atol and rtol when comparing model parameters.
            # Default 'rtol' and 'atol' for attr:`~torch.float32` are ``1.3e-6`` and ``1e-5``
            # Pointer here: https://github.com/pytorch/pytorch/blob/main/torch/testing/_comparison.py#L65
            self._assert_optimizer(
                mesh, mod, opt, dist_mod, dist_opt, inp, atol=1.3e-5, rtol=1e-4
            )

    @with_comms
    def test_admaw_fused_across_meshes(self):
        # [FUNCTION DOC] Builds a 2D mesh and its flattened 1D mesh. For each AdamW config, shards
        # net1 of an MLP on the flattened mesh with shard_fn and net2 on the 2D mesh (replicated on
        # x, sharded on y) with local helpers, then compares the optimizer on the mixed-mesh copy
        # against the plain model via _assert_optimizer.
        # KNOWN ISSUE: the test name has a typo (admaw instead of adamw).
        mesh_shape = (2, self.world_size // 2)
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=("x", "y")
        )
        mesh_flatten = mesh_2d[("x", "y")]._flatten(mesh_dim_name="mesh_flatten")

        # lr as a Tensor is not supported for capturable=False and foreach=True
        adamw_float_lr_configs = [
            {"lr": 0.1, "foreach": False},
            {"lr": 0.1, "weight_decay": 0.05, "foreach": False},
            {"lr": 0.1, "weight_decay": 0.05},
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "amsgrad": True,
            },
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "maximize": True,
                "amsgrad": True,
            },
        ]
        fused_adamw_float_lr_configs = [
            {"lr": 0.1, "weight_decay": 0.05, "fused": True},
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "amsgrad": True,
                "fused": True,
            },
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "maximize": True,
                "amsgrad": True,
                "fused": True,
            },
        ]
        # lr could be a Tensor or a float when fused=True for adamW optimizer
        fused_adamw_tensor_lr_configs = [
            {**config, "lr": torch.tensor(0.1)}
            for config in fused_adamw_float_lr_configs
        ]
        fused_adamw_tensor_lr_configs.extend(
            [
                {**config, "lr": torch.tensor([0.1])}
                for config in fused_adamw_float_lr_configs
            ]
        )
        adamw_configs = [
            *adamw_float_lr_configs,
            *fused_adamw_float_lr_configs,
            *fused_adamw_tensor_lr_configs,
        ]

        # shard function to do full sharding on all parameters of a module
        def _shard_fn_2d(name, module, device_mesh):
            # HELPER: Like shard_fn but for a 2D mesh: shards Linear parameters as Replicate on the
            # first and Shard(0) on the second mesh dim, with a gradient hook restoring that layout.
            if isinstance(module, nn.Linear):
                for name, param in module.named_parameters():
                    dist_param = torch.nn.Parameter(
                        distribute_tensor(param, device_mesh, [Replicate(), Shard(0)])
                    )
                    # make sure partial sum get cleared after backward()
                    dist_param.register_hook(
                        lambda grad: grad.redistribute(
                            placements=[Replicate(), Shard(0)]
                        )
                    )
                    module.register_parameter(name, dist_param)

        # prepare input
        def _input_fn_2d(mod, inputs, device_mesh):
            # HELPER: Wraps the first input as a DTensor on the 2D mesh with Replicate and Shard(0)
            # placements from the local tensor, without running consistency checks.
            # split the input tensor to be sharded input on a 2d mesh
            dist_inp = DTensor.from_local(
                inputs[0], device_mesh, [Replicate(), Shard(0)], run_check=False
            )
            return dist_inp

        for config in adamw_configs:
            mod = MLPModule(self.device_type)
            opt = torch.optim.AdamW(mod.parameters(), **config)

            mod_copy = deepcopy(mod)
            # MLPModule.net1 is sharded on the flatten mesh
            distribute_module(
                mod_copy.net1, mesh_flatten, shard_fn, input_fn, output_fn
            )
            # MLPModule.net2 is sharded on the 2d mesh
            distribute_module(
                mod_copy.net2, mesh_2d, _shard_fn_2d, _input_fn_2d, output_fn
            )
            dist_opt = torch.optim.AdamW(mod_copy.parameters(), **config)

            # use ones to make sure the single machine model have the same input
            # on different ranks
            inp = torch.ones(8, 10, device=self.device_type)
            self._assert_optimizer(None, mod, opt, mod_copy, dist_opt, inp)

    @with_comms
    @parametrize("foreach", [True, False])
    def test_adamw_sharding_cache_no_leak(self, foreach):
        # [FUNCTION DOC] Parametrized by foreach. Shards an MLP's parameters on dim 0 over a 1D
        # mesh, runs 3 warmup training steps with AdamW and then 10 more, and asserts the DTensor
        # sharding propagator cache miss count does not change after warmup, i.e. step-varying
        # scalar arguments do not create new cache entries.
        """Step-varying scalar args (like AdamW's bias corrections) must not
        cause unbounded growth of the DTensor sharding propagation cache.

        foreach=True exercises the ScalarList-in-list path (_foreach_addcdiv_),
        foreach=False exercises the scalar-kwarg path (addcdiv_ value=...).
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        model = MLPModule(self.device_type)
        for name, param in model.named_parameters():
            dist_param = nn.Parameter(distribute_tensor(param, mesh, [Shard(0)]))
            dist_param.register_hook(
                lambda grad: grad.redistribute(placements=[Shard(0)])
            )
            # set the parameter on the parent module
            parts = name.split(".")
            mod = model
            for part in parts[:-1]:
                mod = getattr(mod, part)
            mod.register_parameter(parts[-1], dist_param)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=foreach)

        def run_step():
            # HELPER: One training step: builds a random sharded input, runs forward, backward on
            # the summed output, optimizer step and zero_grad.
            inp = distribute_tensor(
                torch.randn(8, 10, device=self.device_type), mesh, [Shard(0)]
            )
            out = model(inp)
            out.sum().backward()
            optimizer.step()
            optimizer.zero_grad()

        # Warmup to populate caches
        for _ in range(3):
            run_step()

        _, misses_after_warmup = torch._C._get_DTensor_sharding_propagator_cache_stats()

        for _ in range(10):
            run_step()

        # After warmup, the cache should be fully populated: no new
        # misses across subsequent training steps.
        _, misses_after_steps = torch._C._get_DTensor_sharding_propagator_cache_stats()
        self.assertEqual(
            misses_after_warmup,
            misses_after_steps,
            lambda msg: f"{msg}\nCache misses increased from {misses_after_warmup} to "
            f"{misses_after_steps} over 10 steps (expected 0 new misses "
            "after warmup)",
        )


instantiate_parametrized_tests(TestDTensorOptimizer)

TestDTensorOptimizerWithLocalTensor = create_local_tensor_test_class(
    TestDTensorOptimizer,
)

if __name__ == "__main__":
    run_tests()
