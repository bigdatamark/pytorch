# Owner(s): ["module: optimizer"]
# [MODULE DOC] Core optimizer test suite. Defines helper functions for the Rosenbrock test
# function and its gradient and a step pre-hook that seeds bfloat16 Adam state, plus one device-
# generic test class that runs against every entry of the optimizer info database. It covers
# forloop update direction, tensor learning rates, sparse and complex parameters, foreach versus
# fused versus forloop equivalence, state dict save, load and hooks, step hooks, deepcopy, and
# fused mixed precision. It also re-exports the LR scheduler, differentiable optimizer and SWA
# test classes so they are collected here. Importing instantiates device-specific classes;
# running as main calls the test runner.
import functools
import math
import tempfile
import unittest
from copy import deepcopy
from itertools import product
from typing import Any
from unittest.mock import patch

from optim.test_lrscheduler import TestLRScheduler  # noqa: F401
from optim.test_optim import TestDifferentiableOptimizer  # noqa: F401
from optim.test_swa_utils import TestSWAUtils

import torch
from torch.nn import Parameter
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import (
    register_optimizer_step_post_hook,
    register_optimizer_step_pre_hook,
)
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    largeTensorTest,
    onlyAccelerator,
    onlyCPU,
    onlyCUDA,
    onlyNativeDeviceTypes,
    skipMPS,
    TEST_WITH_ROCM,
)
from torch.testing._internal.common_dtype import floating_types_and
from torch.testing._internal.common_optimizers import (
    _get_device_type,
    _get_optim_inputs_including_global_cliquey_kwargs,
    optim_db,
    OptimizerErrorEnum,
    optims,
    TensorTracker,
)
from torch.testing._internal.common_utils import (
    getRocmVersion,
    markDynamoStrictTest,
    parametrize,
    run_tests,
    serialTest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)


FP16_REDUCED_PRECISION = {"atol": 1e-5, "rtol": 1e-4}


def rosenbrock(tensor):
    # [FUNCTION DOC] HELPER: Rosenbrock function of a two element tensor: the sum of the squared
    # distance of the first element from one and one hundred times the squared difference
    # between the second element and the square of the first. Raises AssertionError if the
    # tensor does not have shape of exactly two. Returns a scalar tensor that supports autograd.
    if tensor.size() != torch.Size([2]):
        raise AssertionError(f"Requires tensor with 2 scalars but got {tensor.size()}")
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def drosenbrock(tensor):
    # [FUNCTION DOC] HELPER: Analytic gradient of the Rosenbrock function for a two element
    # tensor, returned as a stacked two element tensor (partial with respect to the first, then
    # the second element). Raises AssertionError if the tensor shape is not two.
    if tensor.size() != torch.Size([2]):
        raise AssertionError(f"Requires tensor with 2 scalars but got {tensor.size()}")
    x, y = tensor
    return torch.stack((-400 * x * (y - x**2) - 2 * (1 - x), 200 * (y - x**2)))


def _bf16_state_init_hook(optimizer, args, kwargs):
    # [FUNCTION DOC] HELPER: Optimizer step pre-hook (arguments are the optimizer, positional
    # args and keyword args) that pre-populates Adam style state for every parameter that has a
    # gradient and no state yet: a float32 step counter (on the parameter device when capturable
    # or fused, a CPU scalar otherwise), zero bfloat16 first and second moment buffers, and a
    # bfloat16 max second moment buffer when amsgrad is set. Because the state is non-empty, the
    # optimizer skips its own float32 initialization, which makes the fused kernel take its
    # mixed precision path. Mutates optimizer state; returns nothing.
    """Step pre-hook that initializes Adam/AdamW states in bfloat16.

    Pre-populates optimizer state before Adam's lazy initialization so that
    ``_init_group`` finds non-empty state and skips its own fp32 allocation.
    The fused CUDA kernel then dispatches to its mixed-precision path.
    """
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            state = optimizer.state[p]
            if len(state) == 0:
                state["step"] = (
                    torch.zeros((), dtype=torch.float32, device=p.device)
                    if group.get("capturable") or group.get("fused")
                    else torch.tensor(0.0, dtype=torch.float32)
                )
                state["exp_avg"] = torch.zeros_like(
                    p, dtype=torch.bfloat16, memory_format=torch.preserve_format
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    p, dtype=torch.bfloat16, memory_format=torch.preserve_format
                )
                if group.get("amsgrad"):
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p,
                        dtype=torch.bfloat16,
                        memory_format=torch.preserve_format,
                    )


@markDynamoStrictTest
class TestOptimRenewed(TestCase):
    # [CLASS DOC] Main device-generic optimizer test case, run once per optimizer info and dtype
    # through the optims decorator, and marked as strict for Dynamo. Stateless apart from per-
    # test locals; each method exercises one aspect: forloop correctness, learning rate
    # handling, sparse and complex parameters, implementation equivalence, state dict and hook
    # APIs, parameter groups, and error handling. The class docstring gives the overall test
    # structure.
    """
    This test class validates the core optimizers and is structured as the correctness of:
    - The update algorithms (forloop implementation)
        * Every optimizer's algorithm is most readably implemented through a big for-loop
          over all the parameters, which is what we refer to as the forloop or single tensor
          implementation. These algorithms are manually validated by comparing to the paper
          and systematically validated by assuring that the loss goes the right direction
          when the optimizer has been applied.
        * This implementation should compose with optimizer hyperparameters well, such as
          supporting Tensor LRs, the capturable API, and sparse and complex parameters.
    - Each varying implementation
        * We then have implementations that improve upon the performance of the forloop
          implementation by leveraging fusion, namely our foreach (mult_tensor) and fused
          implementations.
        * These variations are validated numerically by comparing with the forloop version
          of the optimizer. In fact, we test most variations this way--we see the forloop
          implementation as the ground truth and expect that improvements to it in any way
          should be just as correct.
        * Both params and optimizer states should be validated numerically.
    - state_dict APIs
        * The optimizer instance should be serializable
        * Calling save and load should be deterministic
        * Moving between devices should be seamless
        * BC - load_state_dict should be able to handle older optimizer states
    - Hook APIs (everything should fire in the right order)
    - LR Scheduler integration (composing should not error + should go the right direction)
    - Parameter groups (should be equivalent to having multiple optimizers)
    - Erroring (what should error should error)

    We also cover different ways of generating parameters and grads:
    - With parameters, we either generate them randomly given specific shapes or we take
      them from a sample NN module.
        * Variety is important here because NN modules have type Parameter and randomly
          generated tensors have type Tensor.
        * Parameters can be sparse for a subset of the optimizers (check out OptimizerInfo)
        * Complex parameters should be handled using view_as_real
        * Parameters can be spread across different devices and different dtypes for any
          given optimizer
        * Parameters can be contiguous and noncontiguous
    - With grads, we follow suit from the parameters.
        * Grads can also be None, empty, or zero-valued, and this should not disrupt training.
    """

    @onlyCPU
    @optims(optim_db)
    def test_optim_infos_do_not_specify_global_cliquey_kwargs(
        self, device, dtype, optim_info
    ):
        # [FUNCTION DOC] Check that the optimizer info inputs for CPU do not set any of the
        # global flags (foreach, fused, differentiable) in their keyword arguments, since those
        # flags are varied by the test harness itself. Asserts for every optimizer input
        # generated for the device.
        global_cliquey_flags = ["foreach", "fused", "differentiable"]
        for optim_input in optim_info.optim_inputs_func(device=device):
            self.assertFalse(
                any(f for f in global_cliquey_flags if f in optim_input.kwargs)
            )

    @optims([optim for optim in optim_db if optim.optim_error_inputs_func is not None])
    def test_errors(self, device, dtype, optim_info):
        # [FUNCTION DOC] For each optimizer info that provides error inputs, build the error
        # cases for the device and dtype and check each one. Construction errors: constructing
        # the optimizer must raise the expected error type with a matching message, or emit the
        # expected warning when the type is a warning class. Step errors: construct normally,
        # then the step call must raise or warn likewise. Any other error stage raises
        # NotImplementedError.
        optim_cls = optim_info.optim_cls
        error_inputs = optim_info.optim_error_inputs_func(device=device, dtype=dtype)

        for error_input in error_inputs:
            optim_input = error_input.optimizer_error_input
            params, kwargs = optim_input.params, optim_input.kwargs
            if error_input.error_on == OptimizerErrorEnum.CONSTRUCTION_ERROR:
                if issubclass(error_input.error_type, Warning):
                    with self.assertWarnsRegex(
                        error_input.error_type, error_input.error_regex
                    ):
                        optim_cls(params, **kwargs)
                else:
                    with self.assertRaisesRegex(
                        error_input.error_type, error_input.error_regex
                    ):
                        optim_cls(params, **kwargs)
            elif error_input.error_on == OptimizerErrorEnum.STEP_ERROR:
                optim = optim_cls(params, **kwargs)
                if issubclass(error_input.error_type, Warning):
                    with self.assertWarnsRegex(
                        error_input.error_type, error_input.error_regex
                    ):
                        optim.step()
                else:
                    with self.assertRaisesRegex(
                        error_input.error_type, error_input.error_regex
                    ):
                        optim.step()
            else:
                raise NotImplementedError(f"Unknown error type {error_input.error_on}")

    @onlyCPU
    @optims(optim_db, dtypes=[torch.float32])
    def test_step_with_empty_param_group(self, device, dtype, optim_info):
        # [FUNCTION DOC] CPU only. An optimizer built with a single empty parameter group must
        # have a no-op step: with closure based optimizers the step returns the closure loss
        # value, otherwise it returns None and does not raise.
        # An empty param group means there is nothing to optimize, so step() should
        # be a no-op rather than raising. See https://github.com/pytorch/pytorch/issues/70352
        optim_cls = optim_info.optim_cls
        optimizer = optim_cls([{"params": []}])

        if optim_info.step_requires_closure:
            loss = torch.tensor(3.0, device=device, dtype=dtype)
            self.assertEqual(optimizer.step(lambda: loss), loss)
        else:
            self.assertIsNone(optimizer.step())

    @parametrize("contiguous", [True, False])
    @parametrize("with_lrsched", [True, False])
    @optims(optim_db, dtypes=[torch.float32])
    def test_forloop_goes_right_direction(
        self, device, dtype, optim_info, contiguous, with_lrsched
    ):
        # [FUNCTION DOC] Check that the single-tensor (forloop) implementation reduces the loss.
        # For each scheduler constructor set (or none) and each optimizer input, force foreach
        # off when supported, create contiguous or noncontiguous weight and bias parameters
        # (Muon only gets the weight), run twenty steps on a squared matrix-vector product loss
        # (converted to sparse grads for sparse-only optimizers), stepping schedulers (passing
        # the loss to ReduceLROnPlateau). Asserts the loss decreased, or increased when maximize
        # is set.
        optim_cls = optim_info.optim_cls
        schedulers_constructors = (
            optim_info.scheduler_inputs if with_lrsched else [None]
        )

        for schedulers_constructor in schedulers_constructors:
            # with tensor LR we need fresh inputs for each scheduler
            # or mutating it will carry across iters
            optim_inputs = optim_info.optim_inputs_func(device=device)
            for optim_input in optim_inputs:
                if "foreach" in optim_info.supported_impls:
                    optim_input.kwargs["foreach"] = False  # force forloop
                if contiguous:
                    weight = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
                    bias = Parameter(torch.randn((10), device=device, dtype=dtype))
                else:
                    weight = Parameter(
                        torch.randn((10, 5, 2), device=device, dtype=dtype)[..., 0]
                    )
                    bias = Parameter(
                        torch.randn((10, 2), device=device, dtype=dtype)[..., 0]
                    )
                input = torch.randn(5, device=device, dtype=dtype)

                params = [weight, bias] if optim_cls.__name__ != "Muon" else [weight]
                optimizer = optim_cls(params, **optim_input.kwargs)
                schedulers = [
                    s(optimizer)
                    for s in (schedulers_constructor if schedulers_constructor else [])
                ]

                def closure():
                    # [FUNCTION DOC] HELPER: Training closure. Zeroes gradients, computes the
                    # squared sum of the weight times input (plus bias unless Muon),
                    # backpropagates, converts weight and bias grads to sparse layout for
                    # sparse-only optimizers, and returns the loss.
                    optimizer.zero_grad()
                    wo = (
                        weight.mv(input)
                        if optim_cls.__name__ == "Muon"
                        else weight.mv(input) + bias
                    )
                    loss = wo.pow(2).sum()
                    loss.backward()
                    if optim_info.only_supports_sparse_grads:
                        # For this test, we naively convert the Tensor layout, which we know does
                        # NOT represent the expected use case for optims like SparseAdam!
                        weight.grad = weight.grad.to_sparse()
                        bias.grad = bias.grad.to_sparse()
                    return loss

                initial_value = closure().item()
                for _ in range(20):
                    if optim_info.step_requires_closure:
                        loss = optimizer.step(closure)
                    else:
                        loss = closure()
                        optimizer.step()

                    for scheduler in schedulers:
                        if isinstance(scheduler, ReduceLROnPlateau):
                            scheduler.step(loss)
                        else:
                            scheduler.step()

                if optim_input.kwargs.get("maximize", False):
                    self.assertGreater(closure().item(), initial_value)
                else:
                    self.assertLess(closure().item(), initial_value)

    @onlyCUDA
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @parametrize("with_lrsched", [True, False])
    @optims(optim_db, dtypes=[torch.float32])
    def test_forloop_goes_right_direction_multigpu(
        self, device, dtype, optim_info, with_lrsched
    ):
        # [FUNCTION DOC] CUDA with at least two GPUs. Like the single-device forloop direction
        # test but places the weight on the first GPU and the bias on the second, moving the
        # matrix-vector result to the second GPU in the loss. Forces foreach off where
        # supported, runs twenty closure steps with optional schedulers, and asserts the loss
        # decreased (increased under maximize).
        optim_cls = optim_info.optim_cls
        schedulers_constructors = (
            optim_info.scheduler_inputs if with_lrsched else [None]
        )
        for schedulers_constructor in schedulers_constructors:
            # We need a fresh set of inputs if we have a tensor LR
            # to not carry mutations across iterations.
            optim_inputs = optim_info.optim_inputs_func(device=device)
            for optim_input in optim_inputs:
                if "foreach" in optim_info.supported_impls:
                    optim_input.kwargs["foreach"] = False  # force forloop

                weight = Parameter(torch.randn((10, 5), device="cuda:0", dtype=dtype))
                bias = Parameter(torch.randn((10), device="cuda:1", dtype=dtype))
                inpt = torch.randn(5, device="cuda:0", dtype=dtype)

                params = [weight, bias] if optim_cls.__name__ != "Muon" else [weight]
                optimizer = optim_cls(params, **optim_input.kwargs)
                schedulers = [
                    s(optimizer)
                    for s in (schedulers_constructor if schedulers_constructor else [])
                ]

                def closure():
                    # [FUNCTION DOC] HELPER: Training closure for the two-GPU test. Zeroes
                    # gradients, computes the squared sum of the weight-vector product moved to
                    # the second GPU (plus bias unless Muon), backpropagates, makes grads sparse
                    # for sparse-only optimizers, and returns the loss.
                    optimizer.zero_grad()
                    wo = (
                        weight.mv(inpt).cuda(1)
                        if optim_cls.__name__ == "Muon"
                        else weight.mv(inpt).cuda(1) + bias
                    )
                    loss = wo.pow(2).sum()
                    loss.backward()
                    if optim_info.only_supports_sparse_grads:
                        # For this test, we naively convert the Tensor layout, which we know does
                        # NOT represent the expected use case for optims like SparseAdam!
                        weight.grad = weight.grad.to_sparse()
                        bias.grad = bias.grad.to_sparse()
                    return loss

                initial_value = closure().item()
                for _ in range(20):
                    loss = optimizer.step(closure)
                    for scheduler in schedulers:
                        if isinstance(scheduler, ReduceLROnPlateau):
                            scheduler.step(loss)
                        else:
                            scheduler.step()

                if optim_input.kwargs.get("maximize", False):
                    self.assertGreater(closure().item(), initial_value)
                else:
                    self.assertLess(closure().item(), initial_value)

    @optims(optim_db, dtypes=[torch.float32])
    def test_param_group_with_lrscheduler_goes_right_direction(
        self, device, dtype, optim_info
    ):
        # [FUNCTION DOC] Check that an optimizer with two parameter groups (the second with its
        # own learning rate, a tensor when compiling) still reduces the loss after twenty
        # closure steps with each scheduler constructor set from the optimizer info. The
        # schedulers are stepped each iteration (ReduceLROnPlateau gets the loss). Asserts the
        # final loss is less than the initial loss.
        optim_cls = optim_info.optim_cls

        for schedulers_c in optim_info.scheduler_inputs:
            weight = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
            weight2 = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
            inpt = torch.randn(5, device=device, dtype=dtype)

            # avoid endless recompiles by wrapping LR in a tensor if we're compiling
            lr = torch.tensor(0.01) if torch.compiler.is_compiling() else 0.01
            optimizer = optim_cls(
                [{"params": [weight]}, {"params": [weight2], "lr": lr}]
            )
            schedulers = [scheduler_c(optimizer) for scheduler_c in schedulers_c]

            def closure():
                # [FUNCTION DOC] HELPER: Closure for the two-group test: zeroes gradients, takes
                # the squared sum of the two weights times the input, backpropagates, converts
                # both grads to sparse for sparse-only optimizers, and returns the loss.
                optimizer.zero_grad()
                loss = (weight.mv(inpt) + weight2.mv(inpt)).pow(2).sum()
                loss.backward()
                if optim_info.only_supports_sparse_grads:
                    # For this test, we naively convert the Tensor layout, which we know does
                    # NOT represent the expected use case for optims like SparseAdam!
                    weight.grad = weight.grad.to_sparse()
                    weight2.grad = weight2.grad.to_sparse()
                return loss

            initial_value = closure().item()
            for _ in range(20):
                loss = optimizer.step(closure)
                for scheduler in schedulers:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(loss)
                    else:
                        scheduler.step()

            self.assertLess(closure().item(), initial_value)

    @parametrize("num_dim", [0, 1, 2])
    @optims(optim_db, dtypes=[torch.float32])
    def test_tensor_lr(self, device, dtype, optim_info, num_dim):
        # [FUNCTION DOC] Check that a learning rate given as a tensor (zero, one or two
        # dimensional per the parameter, on the device or on CPU) behaves identically to a float
        # learning rate. Runs for all optimizer inputs including global flags except
        # differentiable. Two optimizers on cloned parameters are stepped five times, one with a
        # float lr and one with a tensor lr, and weights (and biases, except Muon) are compared.
        # If an optimizer rejects tensor lr with a ValueError mentioning lr as a Tensor, that
        # combination is skipped.
        optim_cls = optim_info.optim_cls

        lr_devices = [device]
        if _get_device_type(device) != "cpu":
            lr_devices.append("cpu")

        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )
        for optim_input, lr_device in product(all_optim_inputs, lr_devices):
            weight = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
            weight_c = weight.detach().clone().requires_grad_(True)
            bias = Parameter(torch.randn((10), device=device, dtype=dtype))
            bias_c = bias.detach().clone().requires_grad_(True)
            inpt = torch.randn(5, device=device, dtype=dtype)

            kwargs = optim_input.kwargs
            if "lr" in kwargs:
                del kwargs["lr"]

            params = [weight, bias] if optim_cls.__name__ != "Muon" else [weight]
            kwargs["lr"] = 1.0 if optim_info.step_requires_closure else 1e-3
            optimizer_r = optim_cls(params, **kwargs)

            try:
                kwargs["lr"] = (
                    torch.tensor(kwargs["lr"]).reshape([1] * num_dim).to(lr_device)
                )
                params_c = [weight_c, bias_c]
                if optim_cls.__name__ == "Muon":
                    params_c = [weight_c]
                optimizer = optim_cls(params_c, **kwargs)
            except ValueError as e:
                self.assertRegex(str(e), ".*lr as a Tensor is not supported.*")
                continue

            def closure(optim, w, b, i):
                # [FUNCTION DOC] HELPER: Closure taking an optimizer, weight, bias and input:
                # zero grads, squared sum of the matrix-vector product (plus bias unless Muon),
                # backward, sparse grads for sparse-only optimizers; returns the loss.
                optim.zero_grad()
                wo = w.mv(i) if optim_cls.__name__ == "Muon" else w.mv(i) + b
                loss = wo.pow(2).sum()
                loss.backward()
                if optim_info.only_supports_sparse_grads:
                    # For this test, we naively convert the Tensor layout, which we know does
                    # NOT represent the expected use case for optims like SparseAdam!
                    w.grad = w.grad.to_sparse()
                    b.grad = b.grad.to_sparse()
                return loss

            for _ in range(5):
                if optim_info.step_requires_closure:
                    optimizer_r.step(
                        functools.partial(closure, optimizer_r, weight, bias, inpt)
                    )
                    optimizer.step(
                        functools.partial(closure, optimizer, weight_c, bias_c, inpt)
                    )
                else:
                    closure(optimizer_r, weight, bias, inpt)
                    optimizer_r.step()
                    closure(optimizer, weight_c, bias_c, inpt)
                    optimizer.step()

                self.assertEqual(weight, weight_c)
                if optim_cls.__name__ != "Muon":
                    self.assertEqual(bias, bias_c)

    @parametrize("with_lrsched", [True, False])
    @optims(
        [o for o in optim_db if o.supports_sparse or o.only_supports_sparse_grads],
        dtypes=[torch.float64],
    )
    def test_rosenbrock_sparse(self, device, dtype, optim_info, with_lrsched):
        # [FUNCTION DOC] Run the optimizer on the two-dimensional Rosenbrock problem with sparse
        # gradients (float64, only for optimizers supporting sparse), for 1800 cyclic coordinate
        # descent steps, each step supplying only the x or the y partial as an uncoalesced
        # sparse tensor. Skips a known slow ROCm combination. Optimizer inputs exclude
        # differentiable and fused, are adjusted by the sparse metadata of the optimizer info
        # and deduplicated. For optimizers also supporting dense grads, a second optimizer
        # receives dense gradients and parameters are compared with tolerance each step. Finally
        # asserts the distance to the solution did not grow (or the loss grew under maximize).
        # Returns early when scheduler testing is requested but none exist.
        optim_cls = optim_info.optim_cls

        if (
            TEST_WITH_ROCM
            and getRocmVersion() >= (7, 14)
            and optim_cls.__name__ == "SGD"
            and not with_lrsched
            and torch.device(device).type == "cuda"
            and dtype == torch.float64
        ):
            self.skipTest(
                "order/state-dependent sparse SGD step timeout on ROCm 7.14+ "
                "(with_lrsched=False, SGD, cuda, float64)"
            )

        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        # Fused impls do not support sparse gradients
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable", "fused")
        )
        kwarg_updates, schedulers_constructors = optim_info.metadata_for_sparse

        if with_lrsched and len(schedulers_constructors) == 0:
            return

        supported_inputs = []
        if len(kwarg_updates) != 0:
            seen = set()
            for i in all_optim_inputs:
                for k in kwarg_updates:
                    if k in i.kwargs:
                        del i.kwargs[k]
                hashable_kwargs = tuple(sorted(i.kwargs.items()))
                if len(i.kwargs) > 0 and hashable_kwargs not in seen:
                    supported_inputs.append(i)
                    seen.add(hashable_kwargs)
                    if "lr" in kwarg_updates:
                        i.kwargs["lr"] = kwarg_updates["lr"]
        else:
            supported_inputs = all_optim_inputs

        for optim_input in supported_inputs:
            kwargs = optim_input.kwargs
            multi_tensor = kwargs.get("foreach", False)

            # For rosenbrock tests, it is mandated that the param is a tensor with 2 numbers
            if multi_tensor:
                params_t = [
                    torch.tensor([1.5, 1.5]),
                    torch.tensor([1.5, 1.5], dtype=dtype),
                ]
            else:
                params_t = [torch.tensor([1.5, 1.5])]

            params = [Parameter(param_t) for param_t in params_t]
            optimizer = optim_cls(params, **kwargs)
            schedulers = [
                s(optimizer) for s in (schedulers_constructors if with_lrsched else [])
            ]

            if not optim_info.only_supports_sparse_grads:
                params_c = [Parameter(param_t.clone()) for param_t in params_t]
                optimizer_c = optim_cls(params_c, **kwargs)
                schedulers_c = [
                    s(optimizer_c)
                    for s in (schedulers_constructors if with_lrsched else [])
                ]

            solution = torch.tensor([1, 1])
            with torch.no_grad():
                initial_dist = sum(param.dist(solution) for param in params)

            def get_grad(param, sparse_grad, w):
                # [FUNCTION DOC] HELPER: Build the artificial gradient for a Rosenbrock
                # parameter. Takes the parameter, a flag for sparse output and a coordinate
                # selector. Computes the analytic gradient and returns either a sparse tensor
                # with a split, uncoalesced representation of only the x (selector true) or only
                # the y partial, or a dense tensor with the other component zeroed.
                grad = drosenbrock(param)
                # NB: We torture test the optimizer by returning an
                # uncoalesced sparse tensor

                # Depending on w, provide only the x or y gradient
                if sparse_grad:
                    if w:
                        i = torch.tensor([[0, 0]], dtype=torch.int64)
                        x = grad[0]
                        v = torch.tensor([x / 4.0, x - x / 4.0])
                    else:
                        i = torch.tensor([[1, 1]], dtype=torch.int64)
                        y = grad[1]
                        v = torch.tensor([y - y / 4.0, y / 4.0])
                    grad_out = torch.sparse_coo_tensor(i, v, (2,), dtype=v.dtype)
                else:
                    if w:
                        grad_out = torch.tensor([grad[0], 0], dtype=param.dtype)
                    else:
                        grad_out = torch.tensor([0, grad[1]], dtype=param.dtype)
                return grad_out

            def eval(params, sparse_grad, w):
                # [FUNCTION DOC] HELPER: Closure for the Rosenbrock test. Zeroes optimizer
                # gradients, computes the Rosenbrock loss (summed over parameters when multi-
                # tensor), backpropagates, then overwrites the gradients with artificial ones
                # from the gradient builder (the second converted to the test dtype) under no
                # grad and returns the loss. Its name shadows the builtin.
                optimizer.zero_grad()
                if multi_tensor:
                    loss = sum(rosenbrock(param) for param in params)
                else:
                    loss = rosenbrock(params[0])
                loss.backward()

                grads_out = [get_grad(param, sparse_grad, w) for param in params]
                with torch.no_grad():
                    params[0].grad = grads_out[0]
                    if multi_tensor:
                        params[1].grad = grads_out[1].to(dtype=dtype)
                return loss

            for i in range(1800):
                # Do cyclic coordinate descent
                w = i % 2
                optimizer.step(functools.partial(eval, params, True, w))
                for scheduler in schedulers:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(rosenbrock(params[0]))
                    else:
                        scheduler.step()
                if not optim_info.only_supports_sparse_grads:
                    optimizer_c.step(functools.partial(eval, params_c, False, w))
                    for scheduler in schedulers_c:
                        if isinstance(scheduler, ReduceLROnPlateau):
                            scheduler.step(rosenbrock(params_c[0]))
                        else:
                            scheduler.step()
                    # Tolerance is increased due to floating point error from different
                    # code path for dense case: x v.s. x - x / 4.0 + x / 4.0
                    self.assertEqual(params, params_c, atol=5e-6, rtol=5e-6)

            if not kwargs.get("maximize", False):
                self.assertLessEqual(
                    sum(param.dist(solution) for param in params), initial_dist
                )
            else:
                self.assertGreaterEqual(
                    sum(rosenbrock(param) for param in params),
                    sum(rosenbrock(param_t) for param_t in params_t),
                )

    @skipMPS
    @optims([o for o in optim_db if o.supports_complex], dtypes=[torch.complex64])
    def test_complex(self, device, dtype, optim_info):
        # [FUNCTION DOC] Check complex parameter support (complex64, not MPS). For each
        # optimizer input except differentiable and fused, run an optimizer on complex
        # parameters and another on their real views (a final real parameter is mixed in),
        # feeding both the same random gradients and losses through recorded closures for three
        # steps. Asserts final parameters and all intermediate parameter values seen in steps
        # are equal.
        optim_cls = optim_info.optim_cls
        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        # Also skip fused, since our fused kernels do not support complex
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable", "fused")
        )
        for optim_input in all_optim_inputs:
            # Last param is intentionally real to test that we can mix real and complex
            complex_params = [
                torch.randn(10, 5, device=device, dtype=dtype, requires_grad=True),
                torch.randn(10, device=device, dtype=dtype, requires_grad=True),
                torch.randn(
                    10, 5, device=device, dtype=torch.float32, requires_grad=True
                ),
            ]
            real_params = [
                (
                    torch.view_as_real(param).detach().clone().requires_grad_()
                    if param.is_complex()
                    else param.detach().clone().requires_grad_()
                )
                for param in complex_params
            ]

            complex_optimizer = optim_cls(complex_params, **optim_input.kwargs)
            real_optimizer = optim_cls(real_params, **optim_input.kwargs)
            real_steps = []
            complex_steps = []
            grads_losses = []

            def real_closure():
                # [FUNCTION DOC] HELPER: Closure for the real-valued optimizer: assigns a random
                # gradient to every real parameter, records a copy of each parameter and of each
                # gradient, records and returns a random scalar loss.
                for param in real_params:
                    grad = torch.randn_like(param)
                    param.grad = grad
                    real_steps.append(param.detach().clone())
                    grads_losses.append(grad.clone())
                loss = torch.randn(1)
                grads_losses.append(loss.clone())
                return loss

            def complex_closure():
                # [FUNCTION DOC] HELPER: Closure for the complex optimizer: replays the recorded
                # gradients, converting them to complex for complex parameters and recording
                # real views of the parameters, then assigns the gradients and returns the
                # recorded loss. Pops from the shared recorded list.
                for param in complex_params:
                    if torch.is_complex(param):
                        grad = torch.view_as_complex(grads_losses.pop(0))
                        complex_steps.append(torch.view_as_real_copy(param.detach()))
                    else:
                        grad = grads_losses.pop(0)
                        complex_steps.append(param.detach().clone())
                    param.grad = grad
                return grads_losses.pop(0)

            for _ in range(3):
                if optim_info.step_requires_closure:
                    # LBFGS, for example, requires closure and calls it internally
                    real_optimizer.step(real_closure)
                    complex_optimizer.step(complex_closure)
                else:
                    # For other optimizers, we call closure explicitly to set the gradients
                    real_closure()
                    complex_closure()
                    real_optimizer.step()
                    complex_optimizer.step()

            # Final Parameters should be the same
            complex_params_asreal = [
                torch.view_as_real(param) if param.is_complex() else param
                for param in complex_params
            ]
            self.assertEqual(real_params, complex_params_asreal)

            # All intermediate steps should also be the same
            # also checks steps taken within for example a line search
            self.assertEqual(complex_steps, real_steps)

    @skipMPS
    @optims([o for o in optim_db if o.supports_complex], dtypes=[torch.complex64])
    def test_complex_2d(self, device, dtype, optim_info):
        # [FUNCTION DOC] Check that optimizing one complex parameter matches optimizing its real
        # and imaginary parts as two real parameters (complex64, not MPS), for all optimizer
        # inputs except differentiable and fused. A seed is set for closure-based optimizers
        # because reductions across parameters are order sensitive. Closures record and compare
        # values via tensor trackers; after three steps asserts real and imaginary parts equal
        # and that all tracked values were consumed.
        optim_cls = optim_info.optim_cls
        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        # Also skip fused, since our fused kernels do not support complex
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable", "fused")
        )
        for optim_input in all_optim_inputs:
            if optim_info.step_requires_closure:
                # Why? The way we implement complex is by turning complex params into view_as_real
                # alternatives. For example, an size (M,N) tensor will become (M,N,2). In this test,
                # we break apart a tensor into its real and imaginary parts, which would be 2x(M,N).
                # For other pointwise optimizers, this distinction is trivial, but for LBFGS where
                # there are reductions across all parameters (and all the grads get flattened into
                # one long Tensor), this ordering matters. Why? Reductions are not deterministic
                # because addition between floating point numbers is not associative, i.e.,
                # a + b + c != a + c + b. Thus, we add a seed here to control the discrepancy that
                # will happen with LBFGS. Note that in test_complex above, there is no need for a seed
                # nor for increased tolerance, because results should be bitwise equivalent.
                torch.manual_seed(2024)

            a1 = torch.randn(2, device=device, dtype=dtype, requires_grad=True)
            a1_real = a1.real.detach().clone()
            a1_imag = a1.imag.detach().clone()
            a1_real.requires_grad_()
            a1_imag.requires_grad_()
            optim1 = optim_cls([a1], **optim_input.kwargs)
            optim2 = optim_cls([a1_real, a1_imag], **optim_input.kwargs)

            a1_reals = TensorTracker()
            a1_imags = TensorTracker()
            a1_grad_reals = TensorTracker()
            a1_grad_imags = TensorTracker()
            losses = TensorTracker()

            def closure1():
                # [FUNCTION DOC] HELPER: Closure for the complex parameter: zero grads, loss is
                # the absolute value of the Rosenbrock function of the complex parameter,
                # backward, and records the real and imaginary parts of the value and gradient
                # and the loss in trackers. Returns the loss.
                optim1.zero_grad()
                loss = rosenbrock(a1).abs()
                loss.backward()

                # Track clones to best test accuracy
                a1_reals.add(a1.real)
                a1_imags.add(a1.imag)
                a1_grad_reals.add(a1.grad.real)
                a1_grad_imags.add(a1.grad.imag)

                losses.add(loss)

                return loss

            def closure2():
                # [FUNCTION DOC] HELPER: Closure for the two real parameters: zero grads, checks
                # and sets each real parameter from the tracked values, forms a complex number,
                # computes the same loss, compares it with the tracked loss, backward, and
                # checks the gradients against the tracked ones. Returns the loss.
                optim2.zero_grad()
                a1_reals.pop_check_set(a1_real, self)
                a1_imags.pop_check_set(a1_imag, self)
                a2 = torch.complex(a1_real, a1_imag)
                loss = rosenbrock(a2).abs()
                losses.pop_check_set(loss, self)
                loss.backward()
                a1_grad_reals.pop_check_set(a1_real.grad, self)
                a1_grad_imags.pop_check_set(a1_imag.grad, self)
                return loss

            for _ in range(3):
                if optim_info.step_requires_closure:
                    # LBFGS, for example, requires closure and calls it internally
                    optim1.step(closure1)
                    optim2.step(closure2)
                else:
                    closure1()
                    closure2()
                    optim1.step()
                    optim2.step()

                self.assertEqual(a1.real, a1_real)
                self.assertEqual(a1.imag, a1_imag)

            self.assertTrue(a1_reals.all_popped())
            self.assertTrue(a1_imags.all_popped())
            self.assertTrue(a1_grad_reals.all_popped())
            self.assertTrue(a1_grad_imags.all_popped())
            self.assertTrue(losses.all_popped())

    def test_adamw_serialization(self, device):
        # [FUNCTION DOC] Check AdamW and Adam state dict handling of the decoupled weight decay
        # group option. A state dict from AdamW loaded into Adam sets the option to true on
        # Adam; an Adam state dict has it false but loading it into AdamW forces it to true; an
        # old style AdamW state dict with the key removed also loads as true in a new AdamW.
        model = torch.nn.Linear(5, 5).to(device)
        optim = torch.optim.AdamW(model.parameters())

        loaded_dict = optim.state_dict()

        # Test that Adam respects the decoupled_weight_decay key
        new_optim = torch.optim.Adam(model.parameters())
        new_optim.load_state_dict(loaded_dict)
        self.assertTrue(new_optim.param_groups[0]["decoupled_weight_decay"])

        # Test that decoupled_weight_decay is always True for AdamW
        adam_optim = torch.optim.Adam(model.parameters())
        adam_state_dict = adam_optim.state_dict()
        self.assertFalse(adam_state_dict["param_groups"][0]["decoupled_weight_decay"])

        new_optim = torch.optim.AdamW(model.parameters())
        new_optim.load_state_dict(adam_state_dict)
        self.assertTrue(new_optim.param_groups[0]["decoupled_weight_decay"])

        # Test that state_dicts from the old AdamW (with no decoupled_weight_decay key)
        # will have decoupled_weight_decay=True in new AdamW:
        old_adamw_dict = deepcopy(loaded_dict)
        del old_adamw_dict["param_groups"][0]["decoupled_weight_decay"]
        self.assertFalse("decoupled_weight_decay" in old_adamw_dict["param_groups"][0])

        new_optim = torch.optim.AdamW(model.parameters())
        new_optim.load_state_dict(old_adamw_dict)
        self.assertTrue(new_optim.param_groups[0]["decoupled_weight_decay"])

    def _compare_between(
        self, inputs, models, optimizers, assert_eq_kwargs=None, assert_step_dtype=None
    ):
        # [FUNCTION DOC] HELPER: Train two models with two optimizers in lockstep for seven
        # iterations and compare. Takes a single input or list of inputs (a single input is
        # duplicated), models, optimizers, tolerances for comparison and an optional expected
        # dtype of the step state. Each iteration zeroes grads, freezes the third layer of each
        # model on iteration three and unfreezes it on iteration five, skips forward and
        # backward on iteration two so the step sees no gradients, then steps. After each
        # iteration, parameters and every optimizer state entry of the first optimizer are added
        # to a tracker and compared against those of the second; step dtype is asserted when
        # requested. Asserts that all tracked values are consumed.
        # why 7? iteration 7 is where we start to see differences for RAdam
        # params interacting with the small eps value, because that's right
        # after rho_t becomes greater than 5 in step 6.
        if assert_eq_kwargs is None:
            assert_eq_kwargs = {}
        kIterations = 7
        tracker = TensorTracker(assert_eq_kwargs)
        for i in range(kIterations):
            state, updated_params = [], []
            if not isinstance(inputs, list):
                inputs = [inputs, inputs]
            for input, model, optimizer in zip(inputs, models, optimizers):
                optimizer.zero_grad()

                if i == 3:
                    # Freeze a layer to test if the step of this layer in 'fused' or 'foreach'
                    # is same as the step in 'forloop'.
                    model[2].requires_grad_(False)
                if i == 5:
                    # Unfreeze the layer after 2 iters.
                    model[2].requires_grad_(True)

                # Test that step behaves as expected (a no-op) when grads are set to None
                if i != 2:
                    output = model(input)
                    loss = output.sum()
                    loss.backward()

                optimizer.step()
                state.append(optimizer.state)
                updated_params.append(model.parameters())

            og_state, new_state = state
            for og_p, new_p in zip(updated_params[0], updated_params[1]):
                tracker.add(og_p)
                tracker.pop_check_set(new_p, self)

                # check that optimizer states are the same
                og_p_state = og_state[og_p]
                new_p_state = new_state[new_p]
                if assert_step_dtype is not None:
                    if torch.is_tensor(og_p_state.get("step", None)):
                        self.assertEqual(og_p_state["step"].dtype, assert_step_dtype)
                    if torch.is_tensor(new_p_state.get("step", None)):
                        self.assertEqual(new_p_state["step"].dtype, assert_step_dtype)
                for k in og_p_state:
                    tracker.add(og_p_state[k])
                    tracker.pop_check_set(new_p_state[k], self)

            self.assertTrue(tracker.all_popped())

    def _test_derived_optimizers(
        self,
        device,
        dtype,
        optim_info,
        flag,
        reduced_precision=False,
        assert_step_dtype=None,
    ):
        # [FUNCTION DOC] HELPER: Parity test for foreach or fused implementations against the
        # forloop one. Takes device, dtype, optimizer info, the flag name, a reduced precision
        # switch and an optional step dtype. Raises AssertionError for any flag other than
        # foreach or fused. For each optimizer input (skipping capturable on CPU), builds two
        # identical seeded small models (an extra empty zero-dimensional parameter with a
        # gradient is appended to exercise zero-size handling) with the flag off and on, then
        # runs the lockstep comparison on a fixed input with loosened tolerances if reduced
        # precision.
        """
        Given a flag 'fused' or 'foreach', test for parity of optimizer state
        and updated parameters between when the flag is set to True and False
        for provided optimizer configurations.
        """
        if flag not in ("foreach", "fused"):
            raise AssertionError(f"flag must be 'foreach' or 'fused', got {flag!r}")
        assert_eq_kwargs = {} if not reduced_precision else FP16_REDUCED_PRECISION

        optim_inputs = optim_info.optim_inputs_func(device=device, dtype=dtype)
        optim_cls = optim_info.optim_cls
        for optim_input in optim_inputs:
            models, optimizers = [], []
            kwargs = deepcopy(optim_input.kwargs)
            if kwargs.get("capturable", False) and _get_device_type(device) == "cpu":
                # capturable is not supported on CPU
                continue
            for flag_value in (False, True):
                kwargs[flag] = flag_value
                input = torch.tensor(
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=dtype, device=device
                ).reshape(3, 2)

                torch.manual_seed(1)
                model = torch.nn.Sequential(
                    torch.nn.Linear(2, 3),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(3, 1),
                    torch.nn.Sigmoid(),
                )
                model.to(dtype=dtype, device=device)

                # foreach/fused optimizers should be tested with a
                # zero_size tensor as its last param.
                # ref: https://github.com/pytorch/pytorch/issues/100701
                empty_param = torch.empty(
                    (), device=device, dtype=dtype, requires_grad=True
                )
                empty_param.grad = torch.rand_like(empty_param)
                params = list(model.parameters()) + [empty_param]

                optimizer = optim_cls(params, **kwargs)
                models.append(model)
                optimizers.append(optimizer)

            self._compare_between(
                input, models, optimizers, assert_eq_kwargs, assert_step_dtype
            )

    @skipMPS  # MPS doesn't support torch.float64, see https://github.com/pytorch/pytorch/issues/115350
    @optims(
        [optim for optim in optim_db if "foreach" in optim.supported_impls],
        dtypes=[torch.float64],
    )
    def test_foreach_matches_forloop(self, device, dtype, optim_info):
        # [FUNCTION DOC] Check that the foreach implementation matches the forloop one in
        # float64 for every optimizer supporting foreach, using the shared derived optimizer
        # helper (skipped on MPS).
        self._test_derived_optimizers(device, dtype, optim_info, "foreach")

    @onlyCUDA
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @parametrize("impl", ["foreach", "fused"])
    @optims(
        [
            optim
            for optim in optim_db
            if "foreach" in optim.supported_impls or "fused" in optim.supported_impls
        ]
    )
    def test_mixed_device_dtype(self, device, dtype, optim_info, impl):
        # [FUNCTION DOC] CUDA with multiple GPUs. Check foreach or fused against forloop with
        # parameters of mixed dtypes (float64, float32, float16, bfloat16) on two GPUs, plus an
        # integer tensor without grad that is not given to the optimizer. Gradients are random;
        # the foreach flag runs seven steps and fused one step. Parameters and every state entry
        # must match within five times the single op tolerances. Raises AssertionError for an
        # unknown impl.
        # KNOWN ISSUE: when the implementation is unsupported the function returns a skip
        # decorator object instead of calling the test skip method, so the test silently passes
        # without running.
        """
        Similar in essence to _test_derived_optimizers above. The main difference is that
        _test_derived_optimizers uses model parameters whereas we randomly pass in
        parameters of different dtypes and devices here. We need multiple GPUs (vs just a
        CPU and GPU) because fused adam only works on GPUs. (Thus we only run the tests
        that call into this helper when TEST_MULTIGPU.)
        """
        if impl not in ("foreach", "fused"):
            raise AssertionError(f"impl must be 'foreach' or 'fused', got {impl!r}")
        if impl == "foreach" and "foreach" not in optim_info.supported_impls:
            return unittest.skip(
                f"foreach not supported for {optim_info.optim_cls.__name__}"
            )
        elif impl == "fused" and "cuda" not in optim_info.supports_fused_on:
            return unittest.skip(
                f"fused not supported for {optim_info.optim_cls.__name__} on cuda"
            )

        params = [
            torch.rand(2, 3, dtype=torch.float64, device="cuda:0", requires_grad=True),
            torch.rand(2, 3, dtype=torch.float32, device="cuda:0", requires_grad=True),
            torch.rand(2, 3, dtype=torch.float16, device="cuda:0", requires_grad=True),
            torch.rand(2, 3, dtype=torch.bfloat16, device="cuda:0", requires_grad=True),
            torch.rand(2, 3, dtype=torch.float64, device="cuda:1", requires_grad=True),
            torch.rand(2, 3, dtype=torch.float32, device="cuda:1", requires_grad=True),
            torch.rand(2, 3, dtype=torch.float16, device="cuda:1", requires_grad=True),
            torch.rand(2, 3, dtype=torch.bfloat16, device="cuda:1", requires_grad=True),
            torch.randint(
                1024, (2, 3), dtype=torch.int64, device="cuda:1", requires_grad=False
            ),
        ]

        for p in params:
            if p.requires_grad:
                p.grad = torch.rand_like(p, device=p.device, dtype=p.dtype)

        kIterations = 7 if impl == "foreach" else 1
        optim_inputs = optim_info.optim_inputs_func(device=device)
        optim_cls = optim_info.optim_cls
        for optim_input in optim_inputs:
            updated_params, state = [], []
            kwargs = deepcopy(optim_input.kwargs)
            if kwargs.get("capturable", False) and _get_device_type(device) == "cpu":
                # capturable is not supported on CPU
                continue
            for use_impl in (False, True):
                kwargs[impl] = use_impl
                params_clone = []
                for p in params:
                    p_clone = p.detach().clone()
                    if p.requires_grad:
                        p_clone.requires_grad = True
                        p_clone.grad = p.grad.detach().clone()
                        params_clone.append(p_clone)

                optimizer = optim_cls(params_clone, **kwargs)
                for _ in range(kIterations):
                    optimizer.step()

                state.append(optimizer.state)
                updated_params.append(params_clone)

            og_state, new_state = state
            for og_p, new_p in zip(updated_params[0], updated_params[1]):
                # Increasing the tolerance as we are collating lots of ops together for optimizers and
                # the designated tolerances are for single op only.
                single_rtol, single_atol = torch.testing._comparison.get_tolerances(
                    new_p.dtype, rtol=None, atol=None
                )
                rtol = 5 * single_rtol
                atol = 5 * single_atol

                self.assertEqual(og_p, new_p, rtol=rtol, atol=atol)

                # check that optimizer states are the same
                og_p_state = og_state[og_p]
                new_p_state = new_state[new_p]

                for k in og_p_state:
                    actual = new_p_state[k]
                    self.assertEqual(og_p_state[k], actual, rtol=rtol, atol=atol)

    @skipMPS  # MPS does not support float64
    @onlyAccelerator
    @optims(
        [optim for optim in optim_db if "foreach" in optim.supported_impls],
        dtypes=[torch.float64],
    )
    def test_set_default_dtype_works_with_foreach(self, device, dtype, optim_info):
        # [FUNCTION DOC] Check that foreach works when the global default dtype is float64 or
        # float16 (restored afterwards in a finally block). Runs the derived optimizer parity
        # helper, with loose tolerances for float16, expecting the step state to be float64 when
        # the default is float64 and float32 otherwise. Accelerator only, skipped on MPS.
        # https://github.com/pytorch/pytorch/issues/110940
        # We coerce step to always be float32 unless the
        # default dtype is higher prec float64
        old_default_dtype = torch.get_default_dtype()
        for default_dtype in [torch.float64, torch.float16]:
            try:
                torch.set_default_dtype(default_dtype)
                self._test_derived_optimizers(
                    device,
                    dtype,
                    optim_info,
                    "foreach",
                    reduced_precision=default_dtype == torch.float16,
                    assert_step_dtype=(
                        torch.float64
                        if default_dtype == torch.float64
                        else torch.float32
                    ),
                )
            finally:
                torch.set_default_dtype(old_default_dtype)

    @onlyCUDA
    @largeTensorTest("72GB", "cuda")
    @serialTest()
    @optims(
        [optim for optim in optim_db if "foreach" in optim.supported_impls],
        dtypes=[torch.float16],
    )
    def test_foreach_large_tensor(self, device, dtype, optim_info):
        # [FUNCTION DOC] CUDA, large memory, serial. For each optimizer input, create a float16
        # parameter of 2 to the 32 elements with zero gradient and take one foreach step to
        # check that very large tensors do not fail.
        optim_cls = optim_info.optim_cls
        optim_inputs = optim_info.optim_inputs_func(device=device)
        for optim_input in optim_inputs:
            params = [torch.ones(2**32, device=device, dtype=dtype)]
            params[0].grad = torch.zeros_like(params[0])
            optimizer = optim_cls(params, foreach=True, **optim_input.kwargs)
            optimizer.step()

    @onlyCUDA
    @optims(
        [optim for optim in optim_db if "foreach" in optim.supported_impls],
        dtypes=[torch.float32],
    )
    def test_peak_memory_foreach(self, device, dtype, optim_info):
        # [FUNCTION DOC] CUDA only. Measure peak memory for one optimizer step with foreach off
        # and on across ten small parameters sized to fill a whole caching allocator block, and
        # assert the foreach peak is at most the single tensor peak plus an optimizer specific
        # number of full-size intermediates (one by default, more for capturable, Adadelta,
        # ASGD, RAdam, NAdam, and two for Adagrad, RMSprop, Adafactor), with extra slack for
        # Dynamo and ROCm.
        nparams = 10
        optim_inputs = optim_info.optim_inputs_func(device=device)
        optim_cls = optim_info.optim_cls
        for optim_input in optim_inputs:
            kwargs = deepcopy(optim_input.kwargs)
            max_mems = []
            for flag_value in (False, True):
                kwargs["foreach"] = flag_value
                # The 16 * 8 = 128 is critical here! Our CUDACachingAllocator allocates in blocks
                # of 512, meaning any tensor that occupies <512 bytes of memory will allocate a
                # whole 512 bytes anyway. We use 128 (cuz datasize would be 4 bytes) so that param
                # is size 512 exactly, making our later calculations for intermediate_size easy.
                param = torch.rand(16, 8, device=device, dtype=dtype)
                params = [torch.rand_like(param) for _ in range(nparams)]

                optimizer = optim_cls(params, **kwargs)

                for p in params:
                    p.grad = torch.rand_like(p)

                optimizer.step()
                import gc

                gc.collect()
                torch.cuda.reset_peak_memory_stats()
                optimizer.step()
                gc.collect()
                max_mems.append(torch.cuda.max_memory_allocated())

            st_max_mem, mt_max_mem = max_mems
            intermediate_size = nparams * param.nelement() * param.element_size()
            nintermediates = 1  # we expect a budget of 1 intermediate most of the time

            # Check the param group directly to handle if the compiler set capturable
            if optimizer.param_groups[0].get(
                "capturable", False
            ) or optim_cls.__name__ in ["Adadelta", "ASGD", "RAdam"]:
                # with capturable in Adam(W), we have 2 extra intermediates for the bias_corrections
                # with Adadelta, we have 2 extra for (acc_delta + eps) and (square_avg + eps)
                # ASGD allocates axs, 2x mus, 2x etas, and grads at the same time
                nintermediates = 3
                if optim_cls.__name__ == "NAdam":
                    # with capturable in NAdam, we have 3 extra intermediates for the
                    # bias_correction, mus, and mu_nexts
                    if TEST_WITH_TORCHDYNAMO:
                        # With dynamo, the eager/FX backend appears to hold memory longer than
                        # vanilla eager: https://github.com/pytorch/pytorch/issues/125511
                        nintermediates = 8
                    else:
                        nintermediates = 5

                if optim_cls.__name__ == "RAdam":
                    # RAdam has four intermediates with capturable
                    # num, unrect_step_size, buffer, grouped_grads
                    if TEST_WITH_TORCHDYNAMO:
                        # With dynamo, the eager/FX backend appears to hold memory than
                        # vanilla eager: https://github.com/pytorch/pytorch/issues/125511
                        nintermediates = 6
                    else:
                        nintermediates = 4

            elif optim_cls.__name__ in ["NAdam", "Adagrad", "RMSprop", "Adafactor"]:
                # NAdam uses two intermediates at the same time (grads & exp_avg_sq_sqrt)
                # Adagrad uses std and grads at the same time
                # RMSprop uses avg and grads
                # Adafactor uses row/col var and its mean
                nintermediates = 2

                if optim_cls.__name__ == "Adafactor" and kwargs.get("maximize", False):
                    # When maximize is True, Adafactor also tracks device_grad
                    nintermediates = 3

            # Dynamo ST uses less mem than eager in the case of Adam/Adagrad/Nadam/RAdam
            # which makes the foreach memory check fail
            if TEST_WITH_TORCHDYNAMO:
                st_max_mem += 6000

            expected_max_mem = st_max_mem + intermediate_size * nintermediates
            # hipcc currently can't generate efficient code for the small buffer optimization
            # code path (see Note [small buffer optimization] for details), thus we always
            # dynamically allocate the tensor metadata for ROCM. Adjusting the expected max
            # memory usage to account for this.
            if TEST_WITH_ROCM:
                expected_max_mem *= 1.02

            self.assertLessEqual(mt_max_mem, expected_max_mem)

    @optims(
        [optim for optim in optim_db if "fused" in optim.supported_impls],
        dtypes=floating_types_and(
            torch.bfloat16,
            torch.float16,
        ),
    )
    def test_fused_matches_forloop(self, device, dtype, optim_info):
        # [FUNCTION DOC] Check that the fused implementation matches the forloop one for
        # floating dtypes including bfloat16 and float16, via the derived optimizer parity
        # helper. Skipped if the device type does not support fused for the optimizer, or on MPS
        # for unsupported dtypes.
        if _get_device_type(device) not in optim_info.supports_fused_on:
            self.skipTest(
                f"{device} is not supported for fused on {optim_info.optim_cls.__name__}"
            )
        if _get_device_type(device) == "mps" and dtype not in (
            torch.float16,
            torch.float32,
            torch.bfloat16,
        ):
            self.skipTest(
                "MPS supports only torch.float16, torch.float32 and torch.bfloat16"
            )
        self._test_derived_optimizers(device, dtype, optim_info, "fused")

    # Test vectorization path splits for fused kernels. Ensure the scalar path
    # and vectorized path compute the same thing.
    # See https://github.com/pytorch/pytorch/issues/191763.
    # fp64 is left out so this runs on MPS; it shares the same code w/ fp32
    @optims(
        [optim for optim in optim_db if "fused" in optim.supported_impls],
        dtypes=(torch.float32, torch.float16, torch.bfloat16),
    )
    # Vectorized::size() is at most 32 elements (AVX512, bf16/fp16), so 64 tests
    # two full blocks with no remainder, and 65 adds a scalar remainder.
    @parametrize("numel", [64, 65])
    def test_fused_vectorized_block_matches_scalar(
        self, device, dtype, optim_info, numel
    ):
        # [FUNCTION DOC] Check that the fused vectorized code path and the scalar path agree.
        # Two fused optimizers are run for three steps, one on a parameter whose element count
        # is 64 or 65 (full vector blocks, or blocks with a scalar remainder) and another on a
        # one element parameter, both filled with identical values and gradients. Asserts the
        # large parameter equals the broadcast single parameter and that same shaped state
        # tensors also match. Skips capturable on CPU and unsupported devices.
        if _get_device_type(device) not in optim_info.supports_fused_on:
            self.skipTest(
                f"{device} is not supported for fused on {optim_info.optim_cls.__name__}"
            )
        optim_cls = optim_info.optim_cls
        for optim_input in optim_info.optim_inputs_func(device=device, dtype=dtype):
            kwargs = deepcopy(optim_input.kwargs)
            if kwargs.get("capturable", False) and _get_device_type(device) == "cpu":
                # capturable is not supported on CPU
                continue
            param_numel, param_single = (
                torch.nn.Parameter(torch.full((n,), 0.25, device=device, dtype=dtype))
                for n in (numel, 1)
            )
            opt_numel = optim_cls([param_numel], fused=True, **kwargs)
            opt_single = optim_cls([param_single], fused=True, **kwargs)
            for step in range(3):
                grad_value = 0.125 * (step + 1)
                param_numel.grad = torch.full_like(param_numel, grad_value)
                param_single.grad = torch.full_like(param_single, grad_value)
                opt_numel.step()
                opt_single.step()
                self.assertEqual(param_numel, param_single.expand(numel))
                state = opt_numel.state[param_numel]
                ref_state = opt_single.state[param_single]
                self.assertEqual(state.keys(), ref_state.keys())
                for k, v in state.items():
                    # the shape guard skips step and SGD's None momentum_buffer
                    if torch.is_tensor(v) and v.shape == param_numel.shape:
                        self.assertEqual(v, ref_state[k].expand(numel))

    @optims(
        [optim for optim in optim_db if "fused" in optim.supported_impls],
        dtypes=(torch.float32,),
    )
    def test_fused_error_on_params_on_meta(self, device, dtype, optim_info):
        # [FUNCTION DOC] Check that fused optimizers raise a RuntimeError about requiring
        # floating point tensors when parameters are on the meta device, and that after zeroing
        # grads, materializing the model on the real device, and assigning new gradients, a step
        # succeeds. Skipped if fused is not supported on the device type.
        if _get_device_type(device) not in optim_info.supports_fused_on:
            self.skipTest(
                f"{device} is not supported for fused on {optim_info.optim_cls.__name__}"
            )

        with torch.device("meta"):
            model = torch.nn.Sequential(
                torch.nn.Linear(2, 3),
                torch.nn.Sigmoid(),
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid(),
            ).to(dtype)

        optimizer = optim_info.optim_cls(model.parameters(), fused=True)
        with torch.device("meta"):
            for p in model.parameters():
                p.grad = torch.rand_like(p)

        with self.assertRaisesRegex(
            RuntimeError,
            "`fused=True` requires all the params to be floating point Tensors",
        ):
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        model.to_empty(device=device)
        for p in model.parameters():
            p.grad = torch.rand_like(p)
        optimizer.step()

    @onlyNativeDeviceTypes
    @largeTensorTest("64GB")
    @serialTest()
    @optims(
        [optim for optim in optim_db if "fused" in optim.supported_impls],
        dtypes=[torch.float16],
    )
    def test_fused_large_tensor(self, device, dtype, optim_info):
        # [FUNCTION DOC] Large memory fused test: for each optimizer input create a float16
        # parameter of 2 to the 32 elements with zero grad and take one fused step to ensure
        # large tensors work. Skipped if the device is not in the fused supported list of the
        # optimizer.
        if device not in optim_info.supports_fused_on:
            self.skipTest(
                f"{device} is not supported for fused on {optim_info.optim_cls.__name__}"
            )
        optim_cls = optim_info.optim_cls
        optim_inputs = optim_info.optim_inputs_func(device=device)
        for optim_input in optim_inputs:
            params = [torch.ones(2**32, device=device, dtype=dtype)]
            params[0].grad = torch.zeros_like(params[0])
            optimizer = optim_cls(params, fused=True, **optim_input.kwargs)
            optimizer.step()

    @skipMPS  # MPS fused optimizer does not properly handle found_inf
    @onlyAccelerator
    @optims(
        [optim for optim in optim_db if "fused" in optim.supported_impls],
        dtypes=[torch.float32],
    )
    def test_fused_does_not_step_if_foundinf(self, device, dtype, optim_info):
        # [FUNCTION DOC] Check that a fused optimizer skips the update when a non-zero found-inf
        # tensor is attached to it. For each optimizer input, with and without a grad scale
        # tensor, five parameters with unit gradients get one fused step; asserts any tensor
        # step counter stays zero and the parameters equal their pre-step copies. Accelerator
        # only, not MPS, skipped when fused is unsupported for the device type.
        if _get_device_type(device) not in optim_info.supports_fused_on:
            self.skipTest(
                f"{device} is not supported for fused on {optim_info.optim_cls.__name__}"
            )
        optim_cls = optim_info.optim_cls
        optim_inputs = optim_info.optim_inputs_func(device=device)
        num_params = 5
        for optim_input in optim_inputs:
            for no_grad_scale in (False, True):
                params = [
                    torch.ones((1,), device=device, dtype=dtype)
                    for _ in range(num_params)
                ]
                params_c = [param.detach().clone() for param in params]
                for p in params:
                    p.grad = torch.ones_like(p)
                optimizer = optim_cls(params, fused=True, **optim_input.kwargs)
                optimizer.grad_scale = (
                    None
                    if no_grad_scale
                    else torch.ones((1,), dtype=dtype, device=device)
                )
                optimizer.found_inf = torch.ones((), dtype=dtype, device=device)
                optimizer.step()
                for p in params:
                    if "step" in optimizer.state[p]:
                        self.assertEqual(
                            torch.zeros((), dtype=dtype, device=device),
                            optimizer.state[p]["step"],
                        )
                self.assertEqual(params, params_c)

    @parametrize("impl", ["fused", "capturable"])
    @optims(
        [optim for optim in optim_db if "fused" in optim.supported_impls],
        dtypes=[torch.float32],
    )
    def test_cpu_load_state_dict(self, device, dtype, impl, optim_info):
        # [FUNCTION DOC] Simulate loading a CPU checkpoint of a fused or capturable optimizer
        # onto an accelerator. For each CPU optimizer input, run a step on a CPU parameter, deep
        # copy its state dict, set the fused or capturable group option to true in the copy,
        # then build a device optimizer with that option, load the copied state, set a new
        # random gradient and step. The test completes if no exception occurs. Skipped on CPU,
        # for SGD or Adagrad capturable, for fused where unsupported, and for capturable on MPS.
        # NOTE: This SIMULATES a fused/capturable optimizer with state moved to CPU, issue 103256
        # How do we get there? Users typically create CUDA models on fused optimizers and then
        # store checkpoints on CPU as CUDA memory is limited with torch.load(...map_location="cpu").
        # Since this is a unit test, it is more expedient to simulate what the state_dict
        # would look like, which is basically CPU tensors with fused/capturable flag = True.
        optim_cls = optim_info.optim_cls
        opt_name = optim_cls.__name__
        if opt_name in ("SGD", "Adagrad") and impl == "capturable":
            # Capturable SGD/Adagrad does not exist
            self.skipTest(f"{opt_name} does not currently support capturable")
        if _get_device_type(device) == "cpu":
            self.skipTest("Test is only for non-cpu devices")
        elif (
            impl == "fused"
            and _get_device_type(device) not in optim_info.supports_fused_on
        ):
            self.skipTest(f"{device} is not supported for fused on {opt_name}")
        elif impl == "capturable" and _get_device_type(device) == "mps":
            self.skipTest("MPS does not support capturable")

        cpu_optim_inputs = optim_info.optim_inputs_func(device="cpu")
        for optim_input in cpu_optim_inputs:
            param = torch.tensor([0.1, 0.2], dtype=dtype, device="cpu")
            optimizer = optim_cls([param], **optim_input.kwargs)
            param.grad = torch.rand_like(param)
            optimizer.step()
            optim_state_dict_cpu = deepcopy(optimizer.state_dict())
            optim_state_dict_cpu["param_groups"][0][impl] = True

            # load
            optim_input.kwargs[impl] = True
            param_device = param.detach().clone().to(device=device)
            optimizer_device = optim_cls([param_device], **optim_input.kwargs)
            optimizer_device.load_state_dict(optim_state_dict_cpu)
            optimizer_device.zero_grad()
            param_device.grad = torch.rand_like(param_device)
            optimizer_device.step()

    @optims(optim_db, dtypes=[torch.float32])
    def test_param_groups_weight_decay(self, device, dtype, optim_info):
        # [FUNCTION DOC] Check parameter groups with different weight decay. Two parameters are
        # put in separate groups with identical kwargs except the second has zero weight decay;
        # after twenty steps (sparse grads for sparse-only optimizers) asserts the loss
        # decreased, or increased under maximize. Runs over all inputs except differentiable.
        optim_cls = optim_info.optim_cls
        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )
        for optim_input in all_optim_inputs:
            weight_kwargs = optim_input.kwargs
            weight2_kwargs = deepcopy(optim_input.kwargs)
            weight2_kwargs["weight_decay"] = 0.0

            weight = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
            weight2 = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
            input = torch.randn(5, device=device, dtype=dtype)

            optimizer = optim_cls(
                [
                    dict(params=[weight], **weight_kwargs),
                    dict(params=[weight2], **weight2_kwargs),
                ]
            )

            loss = (weight.mv(input) + weight2.mv(input)).pow(2).sum()
            initial_value = loss.item()
            for _ in range(20):
                optimizer.zero_grad()
                loss = (weight.mv(input) + weight2.mv(input)).pow(2).sum()
                loss.backward()
                if optim_info.only_supports_sparse_grads:
                    # For this test, we naively convert the Tensor layout, which we know does
                    # NOT represent the expected use case for optims like SparseAdam!
                    weight.grad = weight.grad.to_sparse()
                    weight2.grad = weight2.grad.to_sparse()
                optimizer.step()

            # Test that the direction of loss moved appropriately
            if optim_input.kwargs.get("maximize", False):
                self.assertGreater(loss.item(), initial_value)
            else:
                self.assertLess(loss.item(), initial_value)

    @optims(optim_db, dtypes=[torch.float32])
    def test_param_groups_lr(self, device, dtype, optim_info):
        # [FUNCTION DOC] Check per-group learning rates override the constructor default. A
        # group holds the real parameters with a positive lr and a second group holds an
        # unrelated parameter that uses a tiny outer lr (Rprop gets a zero minimum step). Twenty
        # steps with random gradient for the unrelated parameter; asserts the main loss moved in
        # the right direction and that the unrelated parameter is unchanged.
        optim_cls = optim_info.optim_cls
        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )
        for optim_input in all_optim_inputs:
            # optim_input.kwargs will be the param group kwargs, which should have >0 lr
            if "lr" not in optim_input.kwargs or optim_input.kwargs["lr"] == 0:
                optim_input.kwargs["lr"] = 1e-3
            outer_kwargs = {"lr": 1e-28}
            if optim_cls.__name__ == "Rprop":
                # Allow min step size to be 0
                outer_kwargs["step_sizes"] = (0, 50)

            weight = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
            bias = Parameter(torch.randn((10), device=device, dtype=dtype))
            irrelevant = Parameter(torch.randn((2, 2), device=device, dtype=dtype))
            irrelevant_clone = irrelevant.clone()
            input = torch.randn(5, device=device, dtype=dtype)
            params = [weight, bias] if optim_cls.__name__ != "Muon" else [weight]
            optimizer = optim_cls(
                [
                    dict(params=params, **optim_input.kwargs),
                    dict(params=[irrelevant]),
                ],
                **outer_kwargs,
            )

            wo = (
                weight.mv(input)
                if optim_cls.__name__ == "Muon"
                else weight.mv(input) + bias
            )
            loss = wo.pow(2).sum()
            initial_value = loss.item()
            for _ in range(20):
                optimizer.zero_grad()
                wo = (
                    weight.mv(input)
                    if optim_cls.__name__ == "Muon"
                    else weight.mv(input) + bias
                )
                loss = wo.pow(2).sum()
                loss.backward()
                irrelevant.grad = torch.rand_like(irrelevant)
                if optim_info.only_supports_sparse_grads:
                    # For this test, we naively convert the Tensor layout, which we know does
                    # NOT represent the expected use case for optims like SparseAdam!
                    weight.grad = weight.grad.to_sparse()
                    bias.grad = bias.grad.to_sparse()
                    irrelevant.grad = irrelevant.grad.to_sparse()
                optimizer.step()

            # Test that the direction of loss moved appropriately
            if optim_input.kwargs.get("maximize", False):
                self.assertGreater(loss.item(), initial_value)
            else:
                self.assertLess(loss.item(), initial_value)

            # Test that irrelevant parameters were not updated since lr was almost 0
            self.assertEqual(irrelevant, irrelevant_clone)

    @optims(optim_db, dtypes=[torch.float32])
    def test_step_is_noop_when_params_have_no_grad(self, device, dtype, optim_info):
        # [FUNCTION DOC] Check that stepping an optimizer whose parameters do not require grad
        # and have none runs without error, for all optimizer inputs; the closure returns a
        # constant tensor.
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        params = [
            torch.randn(2, 3, requires_grad=False, device=device, dtype=dtype)
            for _ in range(2)
        ]

        def closure():
            # [FUNCTION DOC] HELPER: Closure returning a constant one element tensor of the test
            # dtype and device without touching parameters.
            return torch.tensor([1], device=device, dtype=dtype)

        for optim_input in all_optim_inputs:
            optimizer = optim_cls(params, **optim_input.kwargs)
            optimizer.step(closure)

    @optims(optim_db, dtypes=[torch.float32])
    def test_step_is_noop_for_zero_grads(self, device, dtype, optim_info):
        # [FUNCTION DOC] Check that a step with all-zero (or empty sparse) gradients leaves the
        # parameter unchanged. Inputs with nonzero weight decay are skipped, and AdamW and Muon
        # get weight decay forced to zero since their default would decay parameters. Sparse-
        # only optimizers get a sparse gradient with a multi-dimensional empty values tensor
        # (regression for an issue about empty sparse grads). Differentiable mode uses a
        # detached parameter.
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )

        def closure():
            # [FUNCTION DOC] HELPER: Closure returning a constant one element tensor of the test
            # dtype and device.
            return torch.tensor([1], device=device, dtype=dtype)

        for optim_input in all_optim_inputs:
            kwargs = optim_input.kwargs

            # params will decay even if grads are empty if weight_decay != 0,
            # and capturable doesn't work for CPU tensors
            if kwargs.get("weight_decay", 0) != 0:
                continue

            # AdamW and Muon have a nonzero default weight_decay, which decays
            # params even with zero grads; override it so the step is a true no-op.
            if optim_cls.__name__ in ("AdamW", "Muon"):
                kwargs["weight_decay"] = 0

            param = torch.randn((5, 1), device=device, dtype=dtype, requires_grad=True)
            old_param = param.detach().clone()

            if kwargs.get("differentiable", False):
                params = [param.detach()]
            else:
                params = [param]

            optimizer = optim_cls(params, **kwargs)
            if optim_info.only_supports_sparse_grads:
                # Intentionally construct a multidimensional empty v for the sparse grad
                # Single dim v passes the test while multidim correctly repros the issue
                # https://github.com/pytorch/pytorch/issues/82486
                i = torch.empty((1, 0), device=device, dtype=dtype)
                v = torch.empty((0, 1), device=device, dtype=dtype)
                params[0].grad = torch.sparse_coo_tensor(
                    i, v, (5, 1), device=device, dtype=dtype
                )
            else:
                params[0].grad = torch.zeros_like(params[0])
            optimizer.step(closure)
            self.assertEqual(old_param, params[0])

    @optims(optim_db, dtypes=[torch.float32])
    def test_grads_are_never_inplaced_into(self, device, dtype, optim_info):
        # [FUNCTION DOC] Check that optimizers never modify the gradient tensor in place: for
        # each optimizer input, assign a random (or empty sparse) gradient, record its version
        # counter, take five steps, and assert the counter is unchanged each time.
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        param = torch.randn((5, 1), device=device, dtype=dtype, requires_grad=True)

        def closure():
            # [FUNCTION DOC] HELPER: Closure returning a constant one element tensor of the test
            # dtype and device.
            return torch.tensor([1], device=device, dtype=dtype)

        for optim_input in all_optim_inputs:
            kwargs = optim_input.kwargs

            if kwargs.get("differentiable", False):
                params = [param.detach()]
            else:
                params = [param]

            optimizer = optim_cls(params, **kwargs)
            if optim_info.only_supports_sparse_grads:
                # Intentionally construct a multidimensional empty v for the sparse grad
                # Single dim v passes the test while multidim correctly repros the issue
                # https://github.com/pytorch/pytorch/issues/82486
                i = torch.empty((1, 0), device=device, dtype=dtype)
                v = torch.empty((0, 1), device=device, dtype=dtype)
                params[0].grad = torch.sparse_coo_tensor(
                    i, v, (5, 1), device=device, dtype=dtype
                )
            else:
                params[0].grad = torch.rand_like(params[0])

            old_version = params[0].grad._version

            for _ in range(5):
                optimizer.step(closure)
                self.assertEqual(params[0].grad._version, old_version)

    @optims(optim_db, dtypes=[torch.float32])
    def test_optimizer_can_be_printed(self, device, dtype, optim_info):
        # [FUNCTION DOC] Check that the optimizer representation method can be called without
        # error for every optimizer input on two parameters.
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        params = [
            Parameter(torch.randn(2, 3, requires_grad=True, device=device, dtype=dtype))
            for _ in range(2)
        ]
        for optim_input in all_optim_inputs:
            optimizer = optim_cls(params, **optim_input.kwargs)
            optimizer.__repr__()

    @parametrize("is_named_optim0", [True, False])
    @parametrize("is_named_optim1", [True, False])
    @optims(optim_db, dtypes=[torch.float32])
    def test_state_dict_deterministic(
        self, device, dtype, optim_info, is_named_optim0, is_named_optim1
    ):
        # [FUNCTION DOC] Check that state dicts are deterministic and equal for optimizers with
        # equal parameters and warm state. Parametrized on whether each of two optimizers is
        # built with named parameters. Trains one optimizer for ten steps, clones the parameters
        # into a second optimizer, loads the deep copied state dict, trains both ten steps in
        # lockstep and asserts weights stay equal. Then compares the two state dicts ignoring
        # parameter names, and checks that appending a repeated param group gives the same
        # representation for the last group.
        optim_cls = optim_info.optim_cls

        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )
        weight = Parameter(
            torch.randn(2, 3, requires_grad=True, device=device, dtype=dtype)
        )
        bias = Parameter(torch.randn(2, requires_grad=True, device=device, dtype=dtype))
        input = torch.randn(3, requires_grad=True, device=device, dtype=dtype)
        params = [weight, bias]
        if optim_cls.__name__ == "Muon":
            params = [weight]

        def make_named_param(param, is_named):
            # [FUNCTION DOC] HELPER: Return the parameters unchanged, or when named is
            # requested, a list of name and parameter pairs with names generated from the
            # position.
            if not is_named:
                return param
            return [(f"name{i}", p) for i, p in enumerate(param)]

        def without_param_names(state_dict):
            # [FUNCTION DOC] HELPER: Return a deep copy of the state dict with any parameter
            # names entry removed from every param group.
            new_state_dict = deepcopy(state_dict)
            for pg in new_state_dict["param_groups"]:
                pg.pop("param_names", None)
            return new_state_dict

        def fwd_bwd(optim, w, b, i):
            # [FUNCTION DOC] HELPER: Zero grads, compute the squared sum of the matrix-vector
            # product (plus bias unless Muon), backward, convert existing grads to sparse for
            # sparse-only optimizers, and return the loss.
            optim.zero_grad()
            wo = w.mv(i) if optim_cls.__name__ == "Muon" else w.mv(i) + b
            loss = wo.pow(2).sum()
            loss.backward()
            if optim_info.only_supports_sparse_grads:
                if w.grad is not None:
                    w.grad = w.grad.to_sparse()
                if b.grad is not None:
                    b.grad = b.grad.to_sparse()
            return loss

        for optim_input in all_optim_inputs:
            params_in = make_named_param(params, is_named=is_named_optim0)
            optimizer = optim_cls(params_in, **optim_input.kwargs)
            closure = functools.partial(fwd_bwd, optimizer, weight, bias, input)

            # Prime the optimizer
            for _ in range(10):
                if optim_info.step_requires_closure:
                    optimizer.step(closure)
                else:
                    closure()
                    optimizer.step()

            # Clone the weights and construct a new optimizer for them
            with torch.no_grad():
                weight_c = Parameter(weight.clone())
                bias_c = Parameter(bias.clone())
            params_c_list = (
                [weight_c, bias_c] if optim_cls.__name__ != "Muon" else [weight_c]
            )
            params_c = make_named_param(params_c_list, is_named=is_named_optim1)
            optimizer_c = optim_cls(params_c, **optim_input.kwargs)
            closure_c = functools.partial(fwd_bwd, optimizer_c, weight_c, bias_c, input)

            # Load the state dict from the original optimizer into the new one
            optimizer_c.load_state_dict(deepcopy(optimizer.state_dict()))

            # Run both optimizers in parallel
            for _ in range(10):
                if optim_info.step_requires_closure:
                    optimizer.step(closure)
                    optimizer_c.step(closure_c)
                else:
                    closure()
                    closure_c()
                    optimizer.step()
                    optimizer_c.step()

                self.assertEqual(weight, weight_c)
                if optim_cls.__name__ != "Muon":
                    self.assertEqual(bias, bias_c)

            # Make sure state dict is deterministic with equal (not identical) parameters
            # Param names are optional and not needed to be the consistent.
            self.assertEqual(
                without_param_names(optimizer.state_dict()),
                without_param_names(optimizer_c.state_dict()),
            )

            # Make sure repeated parameters have identical representation (see #36831)
            optimizer_c.param_groups.extend(optimizer_c.param_groups)
            self.assertEqual(
                without_param_names(optimizer.state_dict())["param_groups"][-1],
                without_param_names(optimizer_c.state_dict())["param_groups"][-1],
            )

    @optims(optim_db, dtypes=[torch.float32])
    def test_can_load_older_state_dict(self, device, dtype, optim_info):
        # [FUNCTION DOC] Check that an optimizer can load a state dict from an older version
        # lacking newer flags. After three steps on a small convolution model (or a linear layer
        # for Muon), deep copy the state dict, delete the flags that the optimizer info lists as
        # not originally supported from every group, load it and verify another step succeeds.
        optim_cls = optim_info.optim_cls

        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )

        def _get_model_and_input_tensor(device, dtype, optim_cls):
            # [FUNCTION DOC] HELPER: Build a model and matching input on the device and dtype: a
            # bias-free linear layer with a vector input for Muon (which accepts only 2D
            # parameters), otherwise a convolution plus batch norm model with an image input.
            if optim_cls.__name__ == "Muon":
                # Muon only accepts 2D parameter.
                model = torch.nn.Linear(10, 4, bias=False)
                input = torch.rand(10, device=device, dtype=dtype)
            else:
                model = torch.nn.Sequential(
                    torch.nn.Conv2d(4, 2, 1, stride=2),
                    torch.nn.BatchNorm2d(2, eps=1e-05, momentum=0.1),
                )
                input = torch.rand(1, 4, 16, 16, device=device, dtype=dtype)
            model.to(dtype=dtype, device=device)
            return model, input

        for optim_input in all_optim_inputs:
            torch.manual_seed(1)
            model, input = _get_model_and_input_tensor(device, dtype, optim_cls)
            optimizer = optim_cls(model.parameters(), **optim_input.kwargs)

            def fwd_bwd(optim, mod, i):
                # [FUNCTION DOC] HELPER: Zero grads, sum the model output on the input,
                # backward, and return the loss.
                optim.zero_grad()
                loss = mod(i).sum()
                loss.backward()
                return loss

            for _ in range(3):
                if optim_info.step_requires_closure:
                    optimizer.step(functools.partial(fwd_bwd, optimizer, model, input))
                else:
                    fwd_bwd(optimizer, model, input)
                    optimizer.step()

            # old_state_dict has all new flags del'd
            old_state_dict = deepcopy(optimizer.state_dict())
            old_state_dict_pg = old_state_dict["param_groups"]
            for group in old_state_dict_pg:
                for flag in optim_info.not_og_supported_flags:
                    if flag in group:
                        del group[flag]

            optimizer.load_state_dict(old_state_dict)

            # Make sure we can still step
            if optim_info.step_requires_closure:
                optimizer.step(functools.partial(fwd_bwd, optimizer, model, input))
            else:
                fwd_bwd(optimizer, model, input)
                optimizer.step()

    @parametrize("is_named_optim0", [True, False])
    @parametrize("is_named_optim1", [True, False])
    @optims(
        [o for o in optim_db if not o.only_supports_sparse_grads],
        dtypes=[torch.float32],
    )
    def test_can_load_from_to_named_state_dict(
        self, device, dtype, optim_info, is_named_optim0, is_named_optim1
    ):
        # [FUNCTION DOC] Check loading state dicts between optimizers built from plain or named
        # parameters (parametrized on both). Trains one optimizer for three steps, loads its
        # state dict into a second optimizer over the same model, steps it, and if either used
        # names, asserts the second optimizer state dict keeps the model parameter names.
        # Excludes sparse-only optimizers.
        optim_cls = optim_info.optim_cls

        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )

        def _get_model_and_input_tensor(device, dtype, optim_cls):
            # [FUNCTION DOC] HELPER: Build a model and matching input on the device and dtype: a
            # bias-free linear layer with a vector input for Muon, otherwise a convolution plus
            # batch norm model with an image input.
            if optim_cls.__name__ == "Muon":
                # Muon only accepts 2D parameter.
                model = torch.nn.Linear(10, 4, bias=False)
                input = torch.rand(10, device=device, dtype=dtype)
            else:
                model = torch.nn.Sequential(
                    torch.nn.Conv2d(4, 2, 1, stride=2),
                    torch.nn.BatchNorm2d(2, eps=1e-05, momentum=0.1),
                )
                input = torch.rand(1, 4, 16, 16, device=device, dtype=dtype)
            model.to(dtype=dtype, device=device)
            return model, input

        for optim_input in all_optim_inputs:
            torch.manual_seed(1)
            model, input = _get_model_and_input_tensor(device, dtype, optim_cls)

            def fwd_bwd(optim, mod, i):
                # [FUNCTION DOC] HELPER: Zero grads, sum the model output on the input,
                # backward, and return the loss.
                optim.zero_grad()
                loss = mod(i).sum()
                loss.backward()
                return loss

            # test for parameters, named_parameters, and 2 groups:
            params_to_optimizer = (
                model.named_parameters() if is_named_optim0 else model.parameters()
            )
            optimizer = optim_cls(params_to_optimizer, **optim_input.kwargs)

            for _ in range(3):
                if optim_info.step_requires_closure:
                    optimizer.step(functools.partial(fwd_bwd, optimizer, model, input))
                else:
                    fwd_bwd(optimizer, model, input)
                    optimizer.step()

            # old_state_dict has all new flags del'd
            old_state_dict = deepcopy(optimizer.state_dict())

            params_to_optimizer2 = (
                model.named_parameters() if is_named_optim1 else model.parameters()
            )
            optimizer2 = optim_cls(params_to_optimizer2, **optim_input.kwargs)
            optimizer2.load_state_dict(old_state_dict)

            # Make sure we can still step
            if optim_info.step_requires_closure:
                optimizer2.step(functools.partial(fwd_bwd, optimizer2, model, input))
            else:
                fwd_bwd(optimizer2, model, input)
                optimizer2.step()

            ref_names = [p[0] for p in model.named_parameters()]
            # Make sure that param_names are preserved when provided to at least one of the optimizers
            if is_named_optim0 or is_named_optim1:
                self.assertEqual(
                    optimizer2.state_dict()["param_groups"][0]["param_names"],
                    ref_names,
                )

    @parametrize("is_named_optim", [True, False])
    @optims(optim_db, dtypes=[torch.float32])
    def test_save_load_equality_with_weights_only(
        self, device, dtype, optim_info, is_named_optim
    ):
        # [FUNCTION DOC] Check that a state dict saved to a temporary file and loaded back
        # equals the original, both with the default load and with the weights-only restricted
        # load. Parametrized on named parameters; the optimizer is stepped three times with a
        # closure first.
        optim_cls = optim_info.optim_cls

        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )
        weight = Parameter(
            torch.randn(2, 3, requires_grad=True, device=device, dtype=dtype)
        )
        bias = Parameter(torch.randn(2, requires_grad=True, device=device, dtype=dtype))
        input = torch.randn(3, requires_grad=True, device=device, dtype=dtype)
        params = [weight, bias] if optim_cls.__name__ != "Muon" else [weight]

        def make_named_param(param, is_named):
            # [FUNCTION DOC] HELPER: Return the parameters unchanged, or a list of name and
            # parameter pairs when named is requested.
            if not is_named:
                return param
            return [(f"name{i}", p) for i, p in enumerate(param)]

        def fwd_bwd(optim, w, b, i):
            # [FUNCTION DOC] HELPER: Zero grads, compute the squared sum of the matrix-vector
            # product (plus bias unless Muon), backward, convert the outer weight and bias grads
            # to sparse for sparse-only optimizers, return the loss.
            optim.zero_grad()
            wo = w.mv(i) if optim_cls.__name__ == "Muon" else w.mv(i) + b
            loss = wo.pow(2).sum()
            loss.backward()
            if optim_info.only_supports_sparse_grads:
                weight.grad = weight.grad.to_sparse()
                bias.grad = bias.grad.to_sparse()
            return loss

        for optim_input in all_optim_inputs:
            params_in = make_named_param(params, is_named=is_named_optim)
            optimizer = optim_cls(params_in, **optim_input.kwargs)
            closure = functools.partial(fwd_bwd, optimizer, weight, bias, input)

            # Prime the optimizer
            for _ in range(3):
                optimizer.step(closure)

            sd = optimizer.state_dict()

            # === Check saved/loaded state_dict are the same (including weights_only load). ===
            with tempfile.TemporaryFile() as f:
                torch.save(sd, f)
                f.seek(0)
                sd_copy = torch.load(f)
                self.assertEqual(sd_copy, sd)
                del sd_copy
                f.seek(0)
                sd_copy_wo = torch.load(f, weights_only=True)
                self.assertEqual(sd_copy_wo, sd)

    @optims(optim_db, dtypes=[torch.float32])
    def test_load_nontensor_step(self, device, dtype, optim_info):
        # [FUNCTION DOC] Check that a state dict whose per-parameter step entries were converted
        # from tensors to plain numbers can still be loaded and stepped. Three steps are run,
        # steps are converted with item, the state is loaded back, and one more step is taken.
        optim_cls = optim_info.optim_cls

        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )
        params = [
            Parameter(torch.randn(2, 3, device=device, dtype=dtype)) for _ in range(2)
        ]
        for p in params:
            p.grad = torch.rand_like(p)
            if optim_info.only_supports_sparse_grads:
                # For this test, we naively convert the Tensor layout, which we know does
                # NOT represent the expected use case for optims like SparseAdam!
                p.grad = p.grad.to_sparse()

        # Needed for second order optims like LBFGS
        closure_loss = torch.rand(1, device=device, dtype=dtype)

        def closure():
            # [FUNCTION DOC] HELPER: Return the preset random loss for closure based optimizers,
            # otherwise None.
            return closure_loss if optim_info.step_requires_closure else None

        for optim_input in all_optim_inputs:
            optimizer = optim_cls(params, **optim_input.kwargs)
            for _ in range(3):
                optimizer.step(closure)
            state_dict = deepcopy(optimizer.state_dict())
            for p_state in state_dict["state"].values():
                if "step" in p_state and torch.is_tensor(p_state["step"]):
                    p_state["step"] = p_state["step"].item()
            optimizer.load_state_dict(state_dict)
            optimizer.step(closure)

    @skipMPS  # Muon optimizer creates float64 tensors internally, unsupported on MPS
    @onlyAccelerator
    @optims(optim_db, dtypes=[torch.float32])
    def test_state_dict_cross_device(self, device, dtype, optim_info):
        # [FUNCTION DOC] Check moving an optimizer state from CPU to an accelerator. A CPU
        # optimizer is stepped three times, its state dict is loaded into an optimizer over
        # device copies of the parameters; the loaded dict must be unchanged by loading, step
        # tensors must be on CPU unless capturable or fused is set (then on the device), and
        # five further lockstep steps keep parameters and state dicts equal. Skipped for fused
        # when unsupported on the device; not on MPS.
        optim_cls = optim_info.optim_cls

        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        # We limit our configs to CPU only, because we will be moving them to the target device later
        cpu_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            "cpu", dtype, optim_info, skip=("differentiable",)
        )

        # Needed for second order optims like LBFGS
        closure_loss = torch.rand(1, device=device, dtype=dtype)

        def closure():
            return closure_loss if optim_info.step_requires_closure else None
            # [FUNCTION DOC] HELPER: Return the preset random loss for closure based optimizers,
            # otherwise None.

        for optim_input in cpu_optim_inputs:
            if (
                "fused" in optim_input.kwargs
                and _get_device_type(device) not in optim_info.supports_fused_on
            ):
                self.skipTest(
                    f"{_get_device_type(device)} is not supported for fused on {optim_cls.__name__}"
                )
            params = [
                Parameter(torch.randn(2, 3, device="cpu", dtype=dtype))
                for _ in range(2)
            ]
            for p in params:
                p.grad = torch.randn_like(p)
                if optim_info.only_supports_sparse_grads:
                    # For this test, we naively convert the Tensor layout, which we know does
                    # NOT represent the expected use case for optims like SparseAdam!
                    p.grad = p.grad.to_sparse()

            optimizer = optim_cls(params, **optim_input.kwargs)

            for _ in range(3):
                optimizer.step(closure)

            with torch.no_grad():
                params_device = [p.to(device=device) for p in params]
                for i, p in enumerate(params_device):
                    p.grad = params[i].grad.to(device=device)
            optimizer_device = optim_cls(params_device, **optim_input.kwargs)

            state_dict_cpu = deepcopy(optimizer.state_dict())
            state_dict_device = deepcopy(optimizer.state_dict())
            optimizer_device.load_state_dict(state_dict_device)

            # Make sure state_dict_device isn't modified by merely calling load_state_dict
            self.assertEqual(state_dict_cpu, state_dict_device)

            # Make sure that device of state['step'] is still CPU _unless_ torch.compile() added a capturable!
            capturable = state_dict_cpu["param_groups"][0].get("capturable", False)
            fused = state_dict_cpu["param_groups"][0].get("fused", False)
            new_state_dict = optimizer_device.state_dict()
            for state_cpu, state_device in zip(
                state_dict_cpu["state"].values(), new_state_dict["state"].values()
            ):
                if "step" in state_cpu and torch.is_tensor(state_cpu["step"]):
                    self.assertEqual(
                        state_device["step"].device.type,
                        _get_device_type(device) if capturable or fused else "cpu",
                    )

            for _ in range(5):
                optimizer.step(closure)
                optimizer_device.step(closure)
                self.assertEqual(params, params_device)
                self.assertEqual(optimizer.state_dict(), optimizer_device.state_dict())

    @staticmethod
    def _state_dict_pre_hook(optimizer: Optimizer) -> None:
        # [FUNCTION DOC] HELPER: Static state dict pre-hook that stores a marker entry in the
        # optimizer state mapping, so tests can observe that the hook ran. Returns nothing.
        optimizer.state["test"] = 1

    @staticmethod
    def _state_dict_post_hook(
        optimizer: Optimizer, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Static state dict post-hook. If the marker entry written by the
        # pre-hook is present in the state portion of the state dict, remove it and set a
        # boolean flag entry in the dict to true; otherwise set the flag to false. Returns the
        # (mutated) state dict.
        if "test" in state_dict["state"]:
            state_dict["state"].pop("test")
            state_dict["ran_state_dict_pre_hook"] = True
        else:
            state_dict["ran_state_dict_pre_hook"] = False
        return state_dict

    @optims(optim_db, dtypes=[torch.float32])
    def test_state_dict_pre_hook(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        for optim_input in all_optim_inputs:
            param = torch.rand(2, 3, device=device, dtype=dtype, requires_grad=True)
            optim = optim_cls([param], **optim_input.kwargs)
            optim.register_state_dict_pre_hook(self.__class__._state_dict_pre_hook)
            state_dict = optim.state_dict()
            self.assertEqual(state_dict["state"]["test"], 1)

    @optims(optim_db, dtypes=[torch.float32])
    def test_state_dict_post_hook(self, device, dtype, optim_info):
        # [FUNCTION DOC] Check the state dict post-hook alone: with only the post-hook
        # registered, the marker from the pre-hook is absent, so the produced state dict reports
        # the pre-hook flag as false.
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        for optim_input in all_optim_inputs:
            param = torch.rand(2, 3, device=device, dtype=dtype, requires_grad=True)
            optim = optim_cls([param], **optim_input.kwargs)
            optim.register_state_dict_post_hook(self.__class__._state_dict_post_hook)
            state_dict = optim.state_dict()
            self.assertFalse(state_dict["ran_state_dict_pre_hook"])

    @optims(optim_db, dtypes=[torch.float32])
    def test_state_dict_pre_post_hook(self, device, dtype, optim_info):
        # [FUNCTION DOC] Check state dict hooks together: with the pre-hook adding a marker to
        # the state and the post-hook removing it, the state dict has no marker entry and the
        # flag set by the post-hook is true.
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        for optim_input in all_optim_inputs:
            param = torch.rand(2, 3, device=device, dtype=dtype, requires_grad=True)
            optim = optim_cls([param], **optim_input.kwargs)
            optim.register_state_dict_pre_hook(self.__class__._state_dict_pre_hook)
            optim.register_state_dict_post_hook(self.__class__._state_dict_post_hook)
            state_dict = optim.state_dict()
            self.assertFalse("test" in state_dict["state"])
            self.assertTrue(state_dict["ran_state_dict_pre_hook"])

    @staticmethod
    def _load_state_dict_pre_hook1(
        optimizer: Optimizer, state_dict: dict[str, Any]
    ) -> None:
        # [FUNCTION DOC] HELPER: Static load-state-dict pre-hook that mutates the incoming state
        # dict in place, setting the learning rate of the first param group to a fixed value of
        # 0.002. Returns nothing.
        state_dict["param_groups"][0]["lr"] = 0.002

    @staticmethod
    def _load_state_dict_pre_hook2(
        optimizer: Optimizer, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Static load-state-dict pre-hook that deep copies the incoming
        # state dict, sets the first group learning rate to 0.003 in the copy, and returns the
        # copy, showing a returned dict replaces the one to be loaded.
        # The typical use case for returning a state dict is to drastically modify the state dict.
        # I will simulate by simply making a deep copy and ensuring that my_state_dict still gets used
        my_state_dict = deepcopy(state_dict)
        my_state_dict["param_groups"][0]["lr"] = 0.003
        return my_state_dict

    @staticmethod
    def _load_state_dict_post_hook(optimizer: Optimizer) -> None:
        # [FUNCTION DOC] HELPER: Static load-state-dict post-hook that records in the optimizer
        # state mapping whether the first group learning rate equals 0.003 (evidence that the
        # second pre-hook ran) and that the post-hook ran. Mutates the optimizer state; returns
        # nothing.
        optimizer.state["ran_load_state_dict_pre_hook2"] = (
            optimizer.param_groups[0]["lr"] == 0.003
        )
        optimizer.state["ran_load_state_dict_post_hook"] = True

    @optims(optim_db, dtypes=[torch.float32])
    def test_load_state_dict_pre_hook_and_prepend(self, device, dtype, optim_info):
        # [FUNCTION DOC] Check load-state-dict pre-hooks and the prepend option. Loading with
        # the first hook gives a learning rate of 0.002; registering the second hook (which
        # returns a replacement dict with 0.003) with prepend then loading again still gives
        # 0.002, since the prepended hook runs first and the first hook overrides it.
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        for optim_input in all_optim_inputs:
            param = torch.rand(2, 3, device=device, dtype=dtype, requires_grad=True)
            optim = optim_cls([param], **optim_input.kwargs)
            state_dict = optim.state_dict()

            # usually one would have a new optim instance here, but it's all the same here
            optim.register_load_state_dict_pre_hook(
                self.__class__._load_state_dict_pre_hook1
            )
            optim.load_state_dict(state_dict)
            self.assertEqual(optim.param_groups[0]["lr"], 0.002)

            optim.register_load_state_dict_pre_hook(
                self.__class__._load_state_dict_pre_hook2, prepend=True
            )
            optim.load_state_dict(state_dict)
            # If prepend were False would be 0.003 but since prepend is True, the other hook overrides
            self.assertEqual(optim.param_groups[0]["lr"], 0.002)

    @optims(optim_db, dtypes=[torch.float32])
    def test_load_state_dict_post_hook(self, device, dtype, optim_info):
        # [FUNCTION DOC] Check a load-state-dict post-hook runs after loading: the state records
        # the post-hook flag as true, while the flag that depends on the second pre-hook is
        # false as that hook is not registered.
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        for optim_input in all_optim_inputs:
            param = torch.rand(2, 3, device=device, dtype=dtype, requires_grad=True)
            optim = optim_cls([param], **optim_input.kwargs)

            optim.register_load_state_dict_post_hook(
                self.__class__._load_state_dict_post_hook
            )
            optim.load_state_dict(optim.state_dict())
            self.assertFalse(optim.state["ran_load_state_dict_pre_hook2"])
            self.assertTrue(optim.state["ran_load_state_dict_post_hook"])

    @optims(optim_db, dtypes=[torch.float32])
    def test_load_state_dict_pre_post_hook(self, device, dtype, optim_info):
        # [FUNCTION DOC] Check a load-state-dict pre-hook that returns a replacement state dict
        # together with a post-hook: after loading, the post-hook sees the replaced learning
        # rate (flag true) and the post-hook flag is true.
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        for optim_input in all_optim_inputs:
            param = torch.rand(2, 3, device=device, dtype=dtype, requires_grad=True)
            optim = optim_cls([param], **optim_input.kwargs)

            optim.register_load_state_dict_pre_hook(
                self.__class__._load_state_dict_pre_hook2
            )
            optim.register_load_state_dict_post_hook(
                self.__class__._load_state_dict_post_hook
            )
            optim.load_state_dict(optim.state_dict())
            self.assertTrue(optim.state["ran_load_state_dict_pre_hook2"])
            self.assertTrue(optim.state["ran_load_state_dict_post_hook"])

    @optims(optim_db, dtypes=[torch.float32])
    @parametrize("hook_type", ["pre", "post"])
    def test_step_local_hook(self, device, dtype, optim_info, hook_type):
        # [FUNCTION DOC] Check local step hooks (pre or post, per hook type parametrization)
        # registered on a single optimizer. The hook asserts whether the step body (the closure)
        # already ran. Over all optimizer inputs: two steps call the hook twice, a step on a
        # different optimizer does not call it, and after removing the handle another step
        # leaves the count unchanged.
        # total number of times the hook has been called
        count = 0
        # marks that the optimizer's step body has executed
        body_ran = False

        def closure():
            # [FUNCTION DOC] HELPER: Step closure that sets a flag in the enclosing scope
            # showing the step body ran, and returns the integer one.
            nonlocal body_ran
            body_ran = True
            return 1

        def hook(opt: Optimizer, args: tuple[Any, ...], kwargs: dict[Any, Any]):
            # [FUNCTION DOC] HELPER: Step hook taking the optimizer, positional and keyword
            # arguments: asserts the closure has not run (pre hook) or has run (post hook) and
            # increments the shared call counter.
            nonlocal count
            if hook_type == "pre":
                self.assertFalse(body_ran)
            else:
                self.assertTrue(body_ran)
            count += 1

        # Create a random 2D tensor for compatibility with Muon.
        params = [torch.tensor([[1, 1]], device=device, dtype=dtype)]

        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        for optim_input in all_optim_inputs:
            optim = optim_info.optim_cls(params, **optim_input.kwargs)
            # a second optimizer to prove a local hook stays isolated to the
            # optimizer it is registered on
            optim2 = SGD(params)
            count = 0
            register = (
                optim.register_step_pre_hook
                if hook_type == "pre"
                else optim.register_step_post_hook
            )
            hook_handle = register(hook)

            # reset the marker before each step
            body_ran = False
            optim.step(closure)
            body_ran = False
            optim.step(closure)
            # check if hooks were triggered accordingly
            self.assertEqual(count, 2)

            # a local hook does not fire for a different optimizer
            body_ran = False
            optim2.step(closure)
            self.assertEqual(count, 2)

            # remove handles, take step and verify that hook is no longer registered
            hook_handle.remove()
            body_ran = False
            optim.step(closure)
            self.assertEqual(count, 2)

    @optims(optim_db, dtypes=[torch.float32])
    @parametrize("hook_type", ["pre", "post"])
    def test_step_global_hook(self, device, dtype, optim_info, hook_type):
        # [FUNCTION DOC] Check global step hooks (pre or post) registered for all optimizers.
        # Same flow as the local hook test, except a step on a second plain SGD optimizer also
        # triggers the hook (count becomes three), and after removal neither optimizer triggers
        # it. The handle is removed in a finally block because global hooks live in module level
        # state.
        # total number of times the hook has been called
        count = 0
        # marks that the optimizer's step body has executed
        body_ran = False

        def closure():
            # [FUNCTION DOC] HELPER: Step closure that sets a flag in the enclosing scope
            # showing the step body ran, and returns the integer one.
            nonlocal body_ran
            body_ran = True
            return 1

        def hook(opt: Optimizer, args: tuple[Any, ...], kwargs: dict[Any, Any]):
            # [FUNCTION DOC] HELPER: Global step hook that asserts the closure has not run (pre
            # hook) or has run (post hook) and increments the shared call counter.
            nonlocal count
            if hook_type == "pre":
                self.assertFalse(body_ran)
            else:
                self.assertTrue(body_ran)
            count += 1

        # Create an arbitrary 2D tensor for compatibility with Muon.
        params = [torch.tensor([[1, 1]], device=device, dtype=dtype)]

        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        for optim_input in all_optim_inputs:
            optim = optim_info.optim_cls(params, **optim_input.kwargs)
            optim2 = SGD(params)
            count = 0
            register = (
                register_optimizer_step_pre_hook
                if hook_type == "pre"
                else register_optimizer_step_post_hook
            )
            hook_handle = register(hook)
            try:
                # reset the marker before each step
                body_ran = False
                optim.step(closure)
                body_ran = False
                optim.step(closure)
                # check if hooks were triggered accordingly
                self.assertEqual(count, 2)

                # a global hook also fires for a different optimizer
                body_ran = False
                optim2.step(closure)
                self.assertEqual(count, 3)

                # remove handles, take step and verify that hook is no longer registered
                hook_handle.remove()
                body_ran = False
                optim.step(closure)
                body_ran = False
                optim2.step(closure)
                self.assertEqual(count, 3)
            finally:
                # global hooks persist in module-level state; always clean up
                hook_handle.remove()

    @optims(optim_db, dtypes=[torch.float32])
    def test_step_all_hooks(self, device, dtype, optim_info):
        # [FUNCTION DOC] Check ordering of all four hook kinds across two optimizers. Global pre
        # and post hooks and local pre and post hooks (on both optimizers) append distinct
        # markers to a shared list; each step must append global pre, local pre, local post,
        # global post in that order, for either optimizer. After removing all handles, further
        # steps add nothing.
        def global_pre_hook(
            opt: Optimizer, args: tuple[Any, ...], kwargs: dict[Any, Any]
        ):
            # [FUNCTION DOC] HELPER: Global pre hook that appends the marker zero to the shared
            # list.
            nonlocal data
            data.append(0)

        def global_post_hook(
            opt: Optimizer, args: tuple[Any, ...], kwargs: dict[Any, Any]
        ):
            # [FUNCTION DOC] HELPER: Global post hook that appends the marker five to the shared
            # list.
            nonlocal data
            data.append(5)

        def local_pre_hook(
            opt: Optimizer, args: tuple[Any, ...], kwargs: dict[Any, Any]
        ):
            # [FUNCTION DOC] HELPER: Local pre hook that appends the marker one to the shared
            # list.
            nonlocal data
            data.append(1)

        def local_post_hook(
            opt: Optimizer, args: tuple[Any, ...], kwargs: dict[Any, Any]
        ):
            # [FUNCTION DOC] HELPER: Local post hook that appends the marker two to the shared
            # list.
            nonlocal data
            data.append(2)

        params = [torch.tensor([[1, 1]], device=device, dtype=dtype)]

        def dummy_closure():
            # [FUNCTION DOC] HELPER: Closure returning the integer one; used when the optimizer
            # requires a closure.
            return 1

        closure = dummy_closure if optim_info.step_requires_closure else None

        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        for optim_input in all_optim_inputs:
            optim = optim_info.optim_cls(params, **optim_input.kwargs)
            optim2 = SGD(params)
            data = []

            # register global hooks to both optimizers
            global_pre_handle = register_optimizer_step_pre_hook(global_pre_hook)
            global_post_handle = register_optimizer_step_post_hook(global_post_hook)

            # register local hooks
            first_pre_handle = optim.register_step_pre_hook(local_pre_hook)
            first_post_handle = optim.register_step_post_hook(local_post_hook)
            second_pre_handle = optim2.register_step_pre_hook(local_pre_hook)
            second_post_handle = optim2.register_step_post_hook(local_post_hook)

            optim.step(closure)
            self.assertListEqual(data, [0, 1, 2, 5])
            optim2.step(closure)
            self.assertListEqual(data, [0, 1, 2, 5, 0, 1, 2, 5])
            optim.step(closure)
            self.assertListEqual(data, [0, 1, 2, 5, 0, 1, 2, 5, 0, 1, 2, 5])

            # remove all hooks
            global_pre_handle.remove()
            global_post_handle.remove()
            first_pre_handle.remove()
            first_post_handle.remove()
            second_pre_handle.remove()
            second_post_handle.remove()

            optim.step(closure)
            optim2.step(closure)
            self.assertListEqual(data, [0, 1, 2, 5, 0, 1, 2, 5, 0, 1, 2, 5])

    @optims(optim_db, dtypes=[torch.float32])
    def test_deepcopy_copies_all_public_attrs(self, device, dtype, optim_info):
        # [FUNCTION DOC] Check that a deep copy of a stepped optimizer has the same set of
        # public instance attributes (names not starting with an underscore) as the original,
        # for every optimizer input except differentiable.
        optim_cls = optim_info.optim_cls

        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )

        params = [
            Parameter(torch.randn(2, 3, device=device, dtype=dtype)) for _ in range(2)
        ]
        for p in params:
            p.grad = torch.rand_like(p)
            if optim_info.only_supports_sparse_grads:
                # For this test, we naively convert the Tensor layout, which we know does
                # NOT represent the expected use case for optims like SparseAdam!
                p.grad = p.grad.to_sparse()

        # Needed for second order optims like LBFGS
        def closure():
            # [FUNCTION DOC] HELPER: Closure returning one for closure based optimizers,
            # otherwise None.
            return 1 if optim_info.step_requires_closure else None

        def getPublicAttrs(obj):
            # [FUNCTION DOC] HELPER: Return the set of attribute names in the object's instance
            # dictionary that do not start with an underscore.
            return {k for k in obj.__dict__ if not k.startswith("_")}

        for optim_input in all_optim_inputs:
            optimizer = optim_cls(params, **optim_input.kwargs)

            # Make some state
            for _ in range(3):
                if optim_info.step_requires_closure:
                    optimizer.step(closure)
                else:
                    closure()
                    optimizer.step()

            self.assertEqual(
                getPublicAttrs(optimizer), getPublicAttrs(deepcopy(optimizer))
            )

    @optims(
        [optim for optim in optim_db if optim.step_requires_closure],
        dtypes=[torch.float32],
    )
    def test_second_order_optims_return_consistent_types(
        self, device, dtype, optim_info
    ):
        # [FUNCTION DOC] Check that a closure based (second order) optimizer returns the same
        # result type whether the gradient tolerance is positive or negative infinity, i.e.
        # whether it exits early or not. Uses two optimizers over the same parameters and a
        # constant tensor closure.
        # Motivated by #7586
        optim_cls = optim_info.optim_cls
        params = [
            torch.randn(10, 5, device=device, dtype=dtype),
            torch.randn(10, device=device, dtype=dtype),
        ]

        def closure():
        # [FUNCTION DOC] HELPER: Closure returning a constant one element tensor of the test
        # dtype and device.
            return torch.tensor([10], device=device, dtype=dtype)

        for optim_input in optim_info.optim_inputs_func(device=device):
            # Currently, the only second order optim is LBFGS, so we just go ahead and modify
            # "tolerance_grad", but this may not scale if we add second order optims in the future
            kwargs = optim_input.kwargs
            kwargs["tolerance_grad"] = math.inf
            optim_inf = optim_cls(params, **kwargs)
            kwargs["tolerance_grad"] = -math.inf
            optim_neg_inf = optim_cls(params, **kwargs)

            res1 = optim_inf.step(closure)
            res2 = optim_neg_inf.step(closure)
            self.assertEqual(type(res1), type(res2))

    @skipMPS  # MPS does not support float64
    @onlyAccelerator
    @optims(
        [optim for optim in optim_db if "cpu" in optim.supports_fused_on],
        dtypes=floating_types_and(
            torch.bfloat16,
            torch.float16,
        ),
    )
    def test_fused_cpu_matches_device(self, device, dtype, optim_info):
        # [FUNCTION DOC] Compare the fused optimizer on CPU against the accelerator for floating
        # dtypes (not MPS). For each CPU optimizer input, build the same seeded model with an
        # extra zero-dimensional parameter for each of the two devices with fused on, then run
        # the lockstep comparison helper. Skipped if the device type does not support fused.
        # KNOWN ISSUE: the call to the comparison helper is placed after the loop over optimizer
        # inputs, so only the models of the last optimizer input are compared; the earlier ones
        # are built but never checked.
        if _get_device_type(device) not in optim_info.supports_fused_on:
            self.skipTest(
                f"{_get_device_type(device)} is not supported for fused on {optim_info.optim_cls.__name__}"
            )
        optim_cls = optim_info.optim_cls
        optim_inputs = optim_info.optim_inputs_func(device="cpu")
        for optim_input in optim_inputs:
            inpts, models, optimizers = [], [], []
            for dev in ("cpu", _get_device_type(device)):
                kwargs = optim_input.kwargs
                kwargs["fused"] = True
                inpt = torch.tensor(
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=dtype, device=dev
                ).reshape(3, 2)

                torch.manual_seed(1)
                model = torch.nn.Sequential(
                    torch.nn.Linear(2, 3),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(3, 1),
                    torch.nn.Sigmoid(),
                )
                model.to(dtype=dtype, device=dev)

                # foreach/fused optimizers should be tested with a
                # zero_size tensor as its last param.
                # ref: https://github.com/pytorch/pytorch/issues/100701
                empty_param = torch.empty(
                    (), device=dev, dtype=dtype, requires_grad=True
                )
                empty_param.grad = torch.rand_like(empty_param)
                params = list(model.parameters()) + [empty_param]

                optimizer = optim_cls(params, **kwargs)
                inpts.append(inpt)
                models.append(model)
                optimizers.append(optimizer)
        self._compare_between(inpts, models, optimizers)

    @onlyCUDA
    @optims(
        [
            o
            for o in optim_db
            if ("foreach" in o.supported_impls and o.optim_cls.__name__ != "Adafactor")
        ],
        dtypes=[torch.float32],
    )
    def test_defaults_changed_to_foreach(self, device, dtype, optim_info):
        # [FUNCTION DOC] CUDA only. For each optimizer supporting foreach (except Adafactor,
        # which defaults to single tensor), check that a default constructed optimizer step
        # calls the multi tensor implementation function in the optimizer module (AdamW uses the
        # one from Adam), by patching that function with a mock and asserting it was called.
        # Test that the default implementations for optimizers are changed to foreach
        # except Adafactor, which defaults to the single tensor impl for memory efficiency.
        from torch.optim import Adam, AdamW

        optim_cls = optim_info.optim_cls
        model = torch.nn.Linear(5, 5)
        model.to(dtype=dtype, device=device)
        inpt = torch.rand(2, 5, dtype=dtype, device=device)

        import inspect

        # AdamW dispatches to superclass' adam
        if optim_cls is AdamW:
            module = inspect.getmodule(Adam)
            module_name = "_multi_tensor_adam"
        else:
            module = inspect.getmodule(optim_cls)
            module_name = f"_multi_tensor_{optim_cls.__name__.lower()}"

        for optim_input in optim_info.optim_inputs_func(device=device):
            optim = optim_cls(model.parameters(), **optim_input.kwargs)
            optim.zero_grad()
            output = model(inpt)
            loss = output.sum()
            loss.backward()
            with patch.object(module, module_name) as mocked_foreach_impl:
                optim.step()
                self.assertTrue(mocked_foreach_impl.called)

    @optims(optim_db, dtypes=[torch.float32])
    def test_non_empty_state(self, device, dtype, optim_info):
        # [FUNCTION DOC] Check that after one step every per-parameter state entry is non-empty
        # (a bias-free linear layer is used since Muon needs 2D parameters). Sparse-only
        # optimizers get sparse grads and closure based ones a constant closure.
        # There are internal tests that check that the state is not empty
        optim_cls = optim_info.optim_cls
        # Muon only accepts 2D parameter.
        model = torch.nn.Linear(5, 5, bias=False)
        model.to(dtype=dtype, device=device)
        inpt = torch.rand(2, 5, dtype=dtype, device=device)

        for optim_input in optim_info.optim_inputs_func(device=device):
            optim = optim_cls(model.parameters(), **optim_input.kwargs)
            optim.zero_grad()
            output = model(inpt)
            loss = output.sum()
            loss.backward()

            if optim_info.only_supports_sparse_grads:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = param.grad.to_sparse()

            if optim_info.step_requires_closure:
                optim.step(lambda: 1.0)
            else:
                optim.step()

            for state in optim.state.values():
                self.assertGreater(len(state), 0)

    @onlyCUDA
    @parametrize("amsgrad", [False, True])
    @optims(
        [o for o in optim_db if o.optim_cls.__name__ in ["Adam", "AdamW"]],
        dtypes=[torch.float32],
    )
    def test_fused_mixed_precision_state_init(self, device, dtype, optim_info, amsgrad):
        # [FUNCTION DOC] CUDA only. Check Adam and AdamW fused with state pre-populated in
        # bfloat16 by the step pre-hook: after one step parameters stay float32, step is float32
        # and moment buffers (and max buffer with amsgrad) are bfloat16; after a second step
        # with new gradients the dtypes are unchanged. Parametrized on amsgrad.
        optim_cls = optim_info.optim_cls
        params = [torch.rand(20, 7, device=device, dtype=dtype) for _ in range(5)]
        for p in params:
            p.grad = torch.rand_like(p)

        optim = optim_cls(params, lr=1e-3, fused=True, amsgrad=amsgrad)
        optim.register_step_pre_hook(_bf16_state_init_hook)

        optim.step()

        for p in params:
            self.assertEqual(p.dtype, torch.float32)
            state = optim.state[p]
            self.assertEqual(state["step"].dtype, torch.float32)
            self.assertEqual(state["exp_avg"].dtype, torch.bfloat16)
            self.assertEqual(state["exp_avg_sq"].dtype, torch.bfloat16)
            if amsgrad:
                self.assertEqual(state["max_exp_avg_sq"].dtype, torch.bfloat16)

        # Second step: hook should be idempotent (skips already-populated state)
        for p in params:
            p.grad = torch.rand_like(p)
        optim.step()

        for p in params:
            state = optim.state[p]
            self.assertEqual(state["step"].dtype, torch.float32)
            self.assertEqual(state["exp_avg"].dtype, torch.bfloat16)
            self.assertEqual(state["exp_avg_sq"].dtype, torch.bfloat16)
            if amsgrad:
                self.assertEqual(state["max_exp_avg_sq"].dtype, torch.bfloat16)

    @onlyCUDA
    @parametrize("amsgrad", [False, True])
    @optims(
        [o for o in optim_db if o.optim_cls.__name__ in ["Adam", "AdamW"]],
        dtypes=[torch.float32],
    )
    def test_fused_mixed_precision_hook_skips_existing_state(
        self, device, dtype, optim_info, amsgrad
    ):
        # [FUNCTION DOC] CUDA only. With two param groups, the first has float32 state pre-
        # filled and the second has none; after registering the bfloat16 init hook and one fused
        # step, asserts the hook skipped the first group (state stays float32) and initialized
        # the second (moments bfloat16, step float32). Parametrized on amsgrad.
        optim_cls = optim_info.optim_cls

        # Two param groups: group 1 gets f32 state pre-populated (hook should
        # skip it), group 2 has no state (hook should initialize it in bf16).
        # This exercises the fused kernel handling two groups whose states have
        # different dtypes within the same optimizer.step() call.
        g1_params = [torch.rand(10, 5, device=device, dtype=dtype) for _ in range(2)]
        g2_params = [torch.rand(10, 5, device=device, dtype=dtype) for _ in range(2)]
        for p in g1_params + g2_params:
            p.grad = torch.rand_like(p)

        optim = optim_cls(
            [{"params": g1_params}, {"params": g2_params}],
            lr=1e-3,
            fused=True,
            amsgrad=amsgrad,
        )

        for p in g1_params:
            optim.state[p]["step"] = torch.zeros(
                (), dtype=torch.float32, device=p.device
            )
            optim.state[p]["exp_avg"] = torch.zeros_like(p)
            optim.state[p]["exp_avg_sq"] = torch.zeros_like(p)
            if amsgrad:
                optim.state[p]["max_exp_avg_sq"] = torch.zeros_like(p)

        optim.register_step_pre_hook(_bf16_state_init_hook)
        optim.step()

        # Group 1: hook skipped (state was non-empty), dtypes stay f32.
        for p in g1_params:
            state = optim.state[p]
            self.assertEqual(state["step"].dtype, torch.float32)
            self.assertEqual(state["exp_avg"].dtype, torch.float32)
            self.assertEqual(state["exp_avg_sq"].dtype, torch.float32)
            if amsgrad:
                self.assertEqual(state["max_exp_avg_sq"].dtype, torch.float32)

        # Group 2: hook initialized state in bf16.
        for p in g2_params:
            state = optim.state[p]
            self.assertEqual(state["step"].dtype, torch.float32)
            self.assertEqual(state["exp_avg"].dtype, torch.bfloat16)
            self.assertEqual(state["exp_avg_sq"].dtype, torch.bfloat16)
            if amsgrad:
                self.assertEqual(state["max_exp_avg_sq"].dtype, torch.bfloat16)

    @onlyCUDA
    @optims(
        [o for o in optim_db if o.optim_cls.__name__ in ["Adam", "AdamW"]],
        dtypes=[torch.float32],
    )
    def test_fused_mixed_precision_numerics(self, device, dtype, optim_info):
        # [FUNCTION DOC] CUDA only. Compare fused Adam and AdamW with bfloat16 state
        # (initialized by the pre-hook) against a reference optimizer whose states are rounded
        # to bfloat16 and back after every step. For each optimizer input, run seven steps on
        # ten random parameters and assert parameters match each step, and gradients, parameters
        # and moment buffers are tracked and compared via the tensor tracker which must be fully
        # consumed.
        optim_inputs = optim_info.optim_inputs_func(device=device, dtype=dtype)
        optim_cls = optim_info.optim_cls
        for optim_input in optim_inputs:
            kwargs = {**optim_input.kwargs, "fused": True}

            params = [torch.rand(20, 7, device=device, dtype=dtype) for _ in range(10)]
            for p in params:
                p.grad = torch.rand_like(p)

            params_c = [p.clone() for p in params]
            for p, pc in zip(params, params_c):
                pc.grad = p.grad.clone()

            ref_optim = optim_cls(params, **kwargs)
            bf16_optim = optim_cls(params_c, **kwargs)
            bf16_optim.register_step_pre_hook(_bf16_state_init_hook)

            # Simulate bf16 storage: after each ref step, quantize states to
            # bf16 and back so the reference matches the mixed-precision kernel.
            tracker = TensorTracker()
            for i in range(7):
                ref_optim.step()
                bf16_optim.step()
                for p in params:
                    tracker.add(p)
                    tracker.add(p.grad)
                for d in ref_optim.state.values():
                    exp_avg_bf16 = d["exp_avg"].to(torch.bfloat16)
                    tracker.add(exp_avg_bf16)
                    d["exp_avg"] = exp_avg_bf16.to(torch.float32)
                    exp_avg_sq_bf16 = d["exp_avg_sq"].to(torch.bfloat16)
                    tracker.add(exp_avg_sq_bf16)
                    d["exp_avg_sq"] = exp_avg_sq_bf16.to(torch.float32)
                    if "max_exp_avg_sq" in d:
                        max_exp_avg_sq_bf16 = d["max_exp_avg_sq"].to(torch.bfloat16)
                        tracker.add(max_exp_avg_sq_bf16)
                        d["max_exp_avg_sq"] = max_exp_avg_sq_bf16.to(torch.float32)

                for e, pc in enumerate(params_c):
                    tracker.pop_check_set(pc, self)
                    tracker.pop_check_set(pc.grad, self)

                for p, pc in zip(params, params_c):
                    self.assertEqual(p, pc)

                for dc in bf16_optim.state.values():
                    tracker.pop_check_set(dc["exp_avg"], self)
                    tracker.pop_check_set(dc["exp_avg_sq"], self)
                    if "max_exp_avg_sq" in dc:
                        tracker.pop_check_set(dc["max_exp_avg_sq"], self)
                self.assertTrue(tracker.all_popped())

    @parametrize("dtype", [torch.float32])
    def test_step_iteration(self, device, dtype):
        # [FUNCTION DOC] Check the step iteration count of LBFGS with a strong Wolfe line
        # search: with a maximum of one iteration and five evaluations, one step must call the
        # closure exactly six times. The closure counter is incremented in the nested helper.
        def _get_model_and_input_tensor(device, dtype):
            # [FUNCTION DOC] HELPER: Build a convolution plus batch norm model and a random
            # image input on the device and dtype and return them.
            model = torch.nn.Sequential(
                torch.nn.Conv2d(4, 2, 1, stride=2),
                torch.nn.BatchNorm2d(2, eps=1e-05, momentum=0.1),
            )
            input = torch.rand(1, 4, 16, 16, device=device, dtype=dtype)
            model.to(dtype=dtype, device=device)
            return model, input

        counter = 0

        def fwd_bwd(optim, mod, i):
            # [FUNCTION DOC] HELPER: Closure that increments the enclosing call counter, zeroes
            # gradients, sums the model output on the input, runs backward, and returns the
            # loss.
            nonlocal counter
            counter += 1
            optim.zero_grad()
            loss = mod(i).sum()
            loss.backward()
            return loss

        model, input = _get_model_and_input_tensor(device, dtype)
        optimizer = torch.optim.LBFGS(
            model.parameters(), max_iter=1, max_eval=5, line_search_fn="strong_wolfe"
        )
        optimizer.step(functools.partial(fwd_bwd, optimizer, model, input))
        self.assertEqual(counter, 6)


instantiate_device_type_tests(TestOptimRenewed, globals(), allow_mps=True)
instantiate_device_type_tests(TestSWAUtils, globals(), allow_mps=True)


if __name__ == "__main__":
    run_tests()
