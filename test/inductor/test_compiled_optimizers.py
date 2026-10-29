# Owner(s): ["module: inductor"]

# [MODULE DOC] Tests that torch.compile of optimizer steps (all optimizers in the shared
# optimizer database that have a known kernel count) matches eager results and generates the
# expected number of inductor kernels, with and without learning-rate schedulers, tensor learning
# rates and closures. Also holds recompile tests, foreach codegen tests, a foreach_map Adam test,
# and bitwise-equality tests that run with eager-numerics configs. At import time it builds a
# database of generated test configurations, attaches generated test methods to
# CompiledOptimizerTests (including bitwise variants for GPU capturable cases), instantiates
# device-type tests for the parity class, and tries to import check helpers from
# test_torchinductor, exiting quietly on skip or import errors when run as a script.
import random
import sys
import types
import unittest
import weakref
from contextlib import ExitStack
from copy import deepcopy
from typing import NamedTuple

from expecttest import assert_expected_inline

import torch
import torch._inductor
import torch._inductor.cudagraph_trees
import torch.optim.lr_scheduler
from torch._higher_order_ops import foreach_map
from torch._inductor import config
from torch._inductor.test_case import TestCase
from torch.optim import (
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    ASGD,
    NAdam,
    RAdam,
    RMSprop,
    Rprop,
    SGD,
    SparseAdam,
)
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialLR,
    LambdaLR,
    LinearLR,
    MultiplicativeLR,
    MultiStepLR,
    OneCycleLR,
    PolynomialLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    skipCUDAIf,
    skipXPUIf,
)
from torch.testing._internal.common_optimizers import (
    _get_optim_inputs_including_global_cliquey_kwargs,
    optim_db,
    optims,
)
from torch.testing._internal.common_utils import (
    parametrize,
    skipIfRocm,
    skipIfWindows,
    skipIfXpu,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU,
    has_triton,
)
from torch.testing._internal.triton_utils import requires_gpu, requires_gpu_and_triton


def get_inputs(optim):
    # [FUNCTION DOC] HELPER: Walks the optimizer parameter groups and collects, per parameter,
    # the step counter, the parameter, and the exp_avg and exp_avg_sq state tensors. Gradients
    # are gathered into a local list but not returned. Returns the four lists (steps, params,
    # exp_avgs, exp_avg_sqs). Raises KeyError if the optimizer state lacks those entries.
    steps = []
    params = []
    grads = []
    exp_avgs = []
    exp_avg_sqs = []
    for group in optim.param_groups:
        for p in group["params"]:
            params.append(p)
            grads.append(p.grad)
            state = optim.state[p]
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            steps.append(state["step"])

    return steps, params, exp_avgs, exp_avg_sqs


def update_exp_avg_sq(exp_avg_sq, grad, beta2):
    # [FUNCTION DOC] HELPER: Returns a new tensor equal to the second-moment average scaled by
    # beta2 plus the squared gradient scaled by one minus beta2 (Adam second moment update). Does
    # not mutate inputs.
    return exp_avg_sq.mul(beta2).addcmul(grad, grad, value=1 - beta2)


def update_param(param, step, exp_avg, exp_avg_sq, beta1, beta2, lr, eps):
    # [FUNCTION DOC] HELPER: Computes the Adam-style updated parameter from step, first and
    # second moment tensors, beta1, beta2, lr and eps: bias corrections from the step, a negative
    # step size, and a denominator built from the square root of the second moment. Returns the
    # new parameter tensor without mutating inputs.
    bias_correction1 = 1 - torch.pow(beta1, step)
    bias_correction2 = (1 - torch.pow(beta2, step)).sqrt()
    step_size = (lr / bias_correction1).neg()
    denom = (exp_avg_sq.sqrt() / (bias_correction2 * step_size)).add(eps / step_size)
    return torch.add(param, torch.div(exp_avg, denom))


def foreach_map_adam(
    steps,
    params,
    exp_avgs,
    exp_avg_sqs,
    weight_decay=0,
    beta1=0.9,
    beta2=0.999,
    lr=1e-3,
    eps=1e-8,
):
    # [FUNCTION DOC] HELPER: Adam step written with the foreach_map higher-order op, under
    # no_grad. Takes lists of steps, params, exp_avgs, exp_avg_sqs and hyperparameters.
    # Increments the steps through a mapped add and copies them back in place, computes updated
    # first and second moments and the updated params with one mapped call each (the op cannot
    # return several outputs), then copies the results into the moment and param lists in place.
    # The mapped add for weight decay (when non-zero) has its result discarded, so it has no
    # effect on the update. Returns None.
    with torch.no_grad():
        grads = [param.grad for param in params]
        # update step
        updated_steps = foreach_map(lambda x: x + 1, steps)
        torch._foreach_copy_(steps, updated_steps)

        if weight_decay != 0:
            foreach_map(torch.add, (grads,), alpha=weight_decay)

        # HOPS cannot have multiple outputs at the moment
        # need to call foreach_map once for each output
        exp_avgs_updated = foreach_map(torch.lerp, exp_avgs, grads, 1 - beta1)
        exp_avgs_sq_updated = foreach_map(update_exp_avg_sq, exp_avg_sqs, grads, beta2)
        params_updated = foreach_map(
            update_param,
            params,
            steps,
            exp_avgs_updated,
            exp_avgs_sq_updated,
            beta1,
            beta2,
            lr,
            eps,
        )
        # No input mutation for HOPS
        torch._foreach_copy_(exp_avgs, exp_avgs_updated)
        torch._foreach_copy_(exp_avg_sqs, exp_avgs_sq_updated)
        torch._foreach_copy_(params, params_updated)
    return


# Note: we use atypical values to amplify error
LR_SCHEDULER_TO_KWARGS = {
    LambdaLR: {"lr_lambda": lambda x: 10},
    MultiplicativeLR: {"lr_lambda": lambda x: 10},
    StepLR: {"step_size": 1, "gamma": 100},
    MultiStepLR: {"milestones": [1, 2], "gamma": 100},
    ExponentialLR: {"gamma": 100},
    SequentialLR: {"schedulers": None, "milestones": [1, 2]},
    CosineAnnealingLR: {"T_max": 7},
    ChainedScheduler: {"schedulers": None},
    CyclicLR: {"base_lr": 0.001, "max_lr": 0.02, "cycle_momentum": False},
    CosineAnnealingWarmRestarts: {"T_0": 1},
    OneCycleLR: {
        "max_lr": 0.02,
        "cycle_momentum": False,
        "steps_per_epoch": 1,
        "epochs": 10,
    },
    ConstantLR: {"factor": 0.001},
    LinearLR: {},
    ReduceLROnPlateau: {"factor": 0.99, "patience": 1},
    PolynomialLR: {},
}


def create_scheduler(scheduler, optim):
    # [FUNCTION DOC] HELPER: Builds a learning-rate scheduler of the given class for the
    # optimizer from a deep copy of the stored default kwargs. If the kwargs list sub-schedulers,
    # they are replaced by two ConstantLR and one LambdaLR schedulers made recursively.
    # ChainedScheduler is constructed from kwargs only; every other class receives the optimizer
    # first. Does not mutate the shared kwargs mapping.
    kwargs = deepcopy(LR_SCHEDULER_TO_KWARGS[scheduler])
    if "schedulers" in kwargs:
        kwargs["schedulers"] = [
            create_scheduler(torch.optim.lr_scheduler.ConstantLR, optim)
            for _ in range(2)
        ] + [create_scheduler(torch.optim.lr_scheduler.LambdaLR, optim)]

    if scheduler == ChainedScheduler:
        return scheduler(**kwargs)
    else:
        return scheduler(optim, **kwargs)


class KernelCounts(NamedTuple):
    # [CLASS DOC] Named tuple holding the expected generated kernel counts for one optimizer
    # class: one for the multi-tensor (foreach) path and one for the single-tensor path.
    multitensor: int
    singletensor: int


# With different settings for certain
# tests you can get different kernel counts
# This maps the test name to the
# expected kernel count

# fmt: off
# expecttest got error after PYFMT add line break for the triple quotes
KERNEL_COUNT_OVERRIDES = {
    "test_rmsprop_foreach_weight_decay_cpu": lambda x: assert_expected_inline(x, """12""") ,
    "test_nadam_foreach_weight_decay_momentum_decay_cpu": lambda x: assert_expected_inline(x, """20"""),
    "test_adam_capturable_foreach_xpu": lambda x: assert_expected_inline(x, """2"""),
    "test_adam_foreach_xpu": lambda x: assert_expected_inline(x, """2"""),
    "test_adam_weight_decay_amsgrad_capturable_foreach_xpu": lambda x: assert_expected_inline(x, """2"""),
    "test_adam_weight_decay_amsgrad_foreach_xpu": lambda x: assert_expected_inline(x, """2"""),
    "test_adam_weight_decay_foreach_xpu": lambda x: assert_expected_inline(x, """2"""),
    "test_adam_weight_decay_maximize_foreach_xpu": lambda x: assert_expected_inline(x, """2"""),
    "test_adamw_amsgrad_capturable_foreach_cuda": lambda x: assert_expected_inline(x, """3"""),
    "test_adamw_amsgrad_capturable_foreach_xpu": lambda x: assert_expected_inline(x, """3"""),
    "test_adamw_amsgrad_capturable_cuda": lambda x: assert_expected_inline(x, """6"""),
    "test_adamw_amsgrad_capturable_xpu": lambda x: assert_expected_inline(x, """6"""),
    "test_adamw_tensor_lr_tensor_betas_amsgrad_capturable_cuda": lambda x: assert_expected_inline(x, """6"""),
    "test_adamw_tensor_lr_tensor_betas_amsgrad_capturable_xpu": lambda x: assert_expected_inline(x, """6"""),
    "test_adamw_tensor_lr_tensor_betas_capturable_cuda": lambda x: assert_expected_inline(x, """6"""),
    "test_adamw_tensor_lr_tensor_betas_capturable_xpu": lambda x: assert_expected_inline(x, """6"""),
    "test_adamw_tensor_lr_amsgrad_capturable_cuda": lambda x: assert_expected_inline(x, """6"""),
    "test_adamw_tensor_lr_amsgrad_capturable_xpu": lambda x: assert_expected_inline(x, """6"""),
    "test_adam_tensor_lr_amsgrad_capturable_cuda": lambda x: assert_expected_inline(x, """6"""),
    "test_adam_tensor_lr_amsgrad_capturable_xpu": lambda x: assert_expected_inline(x, """6"""),
    "test_adam_tensor_lr_tensor_betas_amsgrad_capturable_cuda": lambda x: assert_expected_inline(x, """6"""),
    "test_adam_tensor_lr_tensor_betas_amsgrad_capturable_xpu": lambda x: assert_expected_inline(x, """6"""),
    "test_adam_tensor_lr_tensor_betas_capturable_cuda": lambda x: assert_expected_inline(x, """6"""),
    "test_adam_tensor_lr_tensor_betas_capturable_xpu": lambda x: assert_expected_inline(x, """6"""),
    "test_adam_amsgrad_capturable_cuda": lambda x: assert_expected_inline(x, """6"""),
    "test_adam_amsgrad_capturable_xpu": lambda x: assert_expected_inline(x, """6"""),
    "test_adadelta_tensor_lr_capturable_cuda": lambda x: assert_expected_inline(x, """6"""),
    "test_adadelta_tensor_lr_capturable_xpu": lambda x: assert_expected_inline(x, """6"""),
    "test_rmsprop_tensor_lr_capturable_cuda": lambda x: assert_expected_inline(x, """6"""),
    "test_rmsprop_tensor_lr_capturable_xpu": lambda x: assert_expected_inline(x, """6"""),
    "test_adadelta_foreach_weight_decay_maximize_cpu": lambda x: assert_expected_inline(x, """12"""),
    "test_adadelta_foreach_rho_weight_decay_cpu": lambda x: assert_expected_inline(x, """12"""),
    "test_adadelta_foreach_weight_decay_cpu": lambda x: assert_expected_inline(x, """12"""),
    "test_sgd_foreach_momentum_weight_decay_cpu": lambda x: assert_expected_inline(x, """16"""),
    "test_sgd_foreach_momentum_nesterov_weight_decay_cpu": lambda x: assert_expected_inline(x, """16"""),
    "test_sgd_momentum_dampening_foreach_cuda": lambda x: assert_expected_inline(x, """5"""),
    "test_sgd_momentum_dampening_foreach_xpu": lambda x: assert_expected_inline(x, """5"""),
    "test_sgd_momentum_foreach_cuda": lambda x: assert_expected_inline(x, """5"""),
    "test_sgd_momentum_foreach_xpu": lambda x: assert_expected_inline(x, """5"""),
    "test_sgd_weight_decay_maximize_cuda": lambda x: assert_expected_inline(x, """4"""),
    "test_sgd_weight_decay_maximize_xpu": lambda x: assert_expected_inline(x, """4"""),
    "test_sgd_weight_decay_maximize_cpu": lambda x: assert_expected_inline(x, """4"""),
    "test_sgd_weight_decay_cpu": lambda x: assert_expected_inline(x, """4"""),
    "test_sgd_weight_decay_cuda": lambda x: assert_expected_inline(x, """4"""),
    "test_sgd_weight_decay_xpu": lambda x: assert_expected_inline(x, """4"""),
    "test_sgd_momentum_weight_decay_foreach_cuda": lambda x: assert_expected_inline(x, """2"""),
    "test_sgd_momentum_weight_decay_foreach_xpu": lambda x: assert_expected_inline(x, """2"""),
    "test_sgd_momentum_nesterov_weight_decay_foreach_cuda": lambda x: assert_expected_inline(x, """2"""),
    "test_sgd_momentum_nesterov_weight_decay_foreach_xpu": lambda x: assert_expected_inline(x, """2"""),
    "test_sgd_cuda": lambda x: assert_expected_inline(x, """4"""),
    "test_sgd_cpu": lambda x: assert_expected_inline(x, """4"""),
    "test_sgd_xpu": lambda x: assert_expected_inline(x, """4"""),
    "test_adagrad_tensor_lr_cpu": lambda x: assert_expected_inline(x, """6"""),
    "test_adagrad_tensor_lr_cuda": lambda x: assert_expected_inline(x, """6"""),
    "test_adagrad_tensor_lr_xpu": lambda x: assert_expected_inline(x, """6"""),
    "test_adamax_tensor_lr_weight_decay_capturable_cuda": lambda x: assert_expected_inline(x, """6"""),
    "test_adamax_tensor_lr_weight_decay_capturable_xpu": lambda x: assert_expected_inline(x, """6"""),
    "test_asgd_tensor_lr_weight_decay_maximize_capturable_cuda": lambda x: assert_expected_inline(x, """5"""),
    "test_asgd_tensor_lr_weight_decay_maximize_capturable_xpu": lambda x: assert_expected_inline(x, """5"""),
    "test_nadam_tensor_lr_weight_decay_momentum_decay_decoupled_weight_decay_capturable_cuda": lambda x: assert_expected_inline(x, """6"""),
    "test_nadam_tensor_lr_weight_decay_momentum_decay_decoupled_weight_decay_capturable_xpu": lambda x: assert_expected_inline(x, """6"""),
    "test_radam_tensor_lr_capturable_weight_decay_decoupled_weight_decay_cuda": lambda x: assert_expected_inline(x, """6"""),
    "test_radam_tensor_lr_capturable_weight_decay_decoupled_weight_decay_xpu": lambda x: assert_expected_inline(x, """6"""),
    "test_sgd_tensor_lr_cpu": lambda x: assert_expected_inline(x, """2"""),
    "test_sgd_tensor_lr_cuda": lambda x: assert_expected_inline(x, """2"""),
    "test_sgd_tensor_lr_xpu": lambda x: assert_expected_inline(x, """2"""),
}
# fmt: on

# also tracks currently supported optimizers
KERNEL_COUNTS = {
    Adam: KernelCounts(multitensor=2, singletensor=8),
    AdamW: KernelCounts(multitensor=2, singletensor=8),
    NAdam: KernelCounts(multitensor=2, singletensor=8),
    Rprop: KernelCounts(multitensor=2, singletensor=8),
    RMSprop: KernelCounts(multitensor=2, singletensor=8),
    Adadelta: KernelCounts(multitensor=2, singletensor=8),
    Adagrad: KernelCounts(multitensor=2, singletensor=8),
    SGD: KernelCounts(multitensor=1, singletensor=8),
    ASGD: KernelCounts(multitensor=2, singletensor=8),
    RAdam: KernelCounts(multitensor=2, singletensor=8),
    Adamax: KernelCounts(multitensor=2, singletensor=8),
}


def build_opt_kwarg_db():
    # [FUNCTION DOC] HELPER: Builds the list of generated test configurations. For every
    # optimizer database entry whose class has a known kernel count, for the cpu device and the
    # GPU device, and for each optimizer input set (skipping differentiable and fused variants),
    # it derives a test name from the class name, the non-default kwargs, tensor lr or betas
    # markers and the device, adds the device and a kernel count (an override by name, else the
    # multi-tensor count when foreach is on and the device is the GPU, else the single-tensor
    # count), and drops entries with no kernel count or with fused set. Entries with a tensor lr
    # are repeated for every scheduler class with the scheduler name appended; others use no
    # scheduler. Returns a list of tuples of optimizer class, test name, kwargs and scheduler
    # class.
    compiled_opt_db = []
    for optim_info in optim_db:
        if optim_info.optim_cls not in KERNEL_COUNTS:
            continue

        for device in ["cpu", GPU_TYPE]:
            for optim_inputs in _get_optim_inputs_including_global_cliquey_kwargs(
                device, None, optim_info, skip=("differentiable", "fused")
            ):
                kwargs = dict(optim_inputs.kwargs)
                name = f"test_{optim_info.optim_cls.__name__.lower()}"

                has_tensor_lr = False
                for key, val in kwargs.items():
                    if (key != "lr" and key != "betas") and (
                        not isinstance(val, bool) or (isinstance(val, bool) and val)
                    ):
                        name += "_" + key

                    if key == "lr" and isinstance(kwargs["lr"], torch.Tensor):
                        has_tensor_lr = True
                        name += "_tensor_lr"

                    if key == "betas" and isinstance(kwargs["betas"][0], torch.Tensor):
                        name += "_tensor_betas"

                name += f"_{device}"

                kwargs["device"] = device
                if name in KERNEL_COUNT_OVERRIDES:
                    kwargs["kernel_count"] = KERNEL_COUNT_OVERRIDES[name]
                else:
                    kwargs["kernel_count"] = (
                        KERNEL_COUNTS[optim_info.optim_cls].multitensor
                        if kwargs.get("foreach", False) and device == GPU_TYPE
                        else KERNEL_COUNTS[optim_info.optim_cls].singletensor
                    )

                if kwargs["kernel_count"] is None or kwargs.get("fused", False):
                    continue

                if has_tensor_lr:
                    for scheduler_cls in LR_SCHEDULER_TO_KWARGS:
                        name_w_scheduler = name + f"_{scheduler_cls.__name__.lower()}"
                        compiled_opt_db.append(
                            (
                                optim_info.optim_cls,
                                name_w_scheduler,
                                kwargs,
                                scheduler_cls,
                            )
                        )
                else:
                    compiled_opt_db.append((optim_info.optim_cls, name, kwargs, None))

    return compiled_opt_db


COMPILED_OPT_KWARG_DB = build_opt_kwarg_db()

aten = torch.ops.aten


try:
    try:
        from .test_torchinductor import check_model, check_model_gpu
    except ImportError:
        from test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
            check_model,
            check_model_gpu,
        )
except (unittest.SkipTest, ImportError) as e:
    sys.stderr.write(f"{type(e)}: {e}\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise


def call_scheduler(scheduler):
    # [FUNCTION DOC] HELPER: Advances a scheduler by one step; ReduceLROnPlateau needs a metric
    # so it is stepped with a constant value, others are stepped with no arguments.
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(1.0)  # we won't reduce the metric over two iters anyway
    else:
        scheduler.step()


def compile_opt(opt_compiled, closure=None, fullgraph=True):
    # [FUNCTION DOC] HELPER: Prepares an optimizer for compiled stepping. Applies the dynamo
    # torch patcher, unwraps the step method twice to skip the differentiable-grad wrapper
    # (avoiding a graph break), marks the optimizer as already called to silence scheduler
    # warnings, and returns an inductor-compiled zero-argument function that calls the raw step
    # (with the closure when given). Fullgraph mode is controlled by the argument. Mutates the
    # optimizer's called flag and global patch state.
    # run the patcher so that step has the expected structure
    torch._dynamo.eval_frame.TorchPatcher.patch()

    # unwrap step TWICE to avoid a deliberate graph break due to
    # a limitation of functionalization/no_grad detection
    # see the [Note on graph break] in optimizer.py
    # This ignores the outer _use_grad_if_differentiable wrapper
    # and instead manually disables grad before calling step, which is fine
    # for now as dynamo does not support differentiable optimizers anyway
    step_fn = opt_compiled.step.__wrapped__.__wrapped__

    # This ensures we don't receive spam of warnings from LR Scheduler
    opt_compiled._opt_called = True

    if closure is not None:

        def fn():
            # [FUNCTION DOC] HELPER: calls the unwrapped step function with the optimizer and the
            # closure.
            step_fn(opt_compiled, closure)

    else:

        def fn():
            # [FUNCTION DOC] HELPER: calls the unwrapped step function with the optimizer only.
            step_fn(opt_compiled)

    return torch.compile(fn, backend="inductor", fullgraph=fullgraph)


def check_optim(
    self,
    optim_cls,
    params_eager,
    params_compiled,
    state_eager,
    state_compiled,
    atol=None,
    rtol=None,
):
    # [FUNCTION DOC] HELPER: Asserts the eager and compiled parameter lists are equal and that
    # the per-parameter optimizer state matches, using the test case's assertEqual (self is the
    # test case, not a method receiver). Uses looser tolerances for Adadelta and NAdam. KNOWN
    # ISSUE: the atol and rtol arguments are overwritten with None at the start, so caller-
    # supplied tolerances are ignored.
    params_eager = list(params_eager)
    params_compiled = list(params_compiled)
    # Note on tolerances:
    # test_correctness_Adadelta_cuda_float32
    # Mismatched elements: 10 / 100 (10.0%)
    # Greatest absolute difference: 4.838220775127411e-05 at index (7, 4) (up to 1e-05 allowed)
    # Greatest relative difference: 0.007270356640219688 at index (7, 2) (up to 1e-05 allowed)
    # This is due to floating point ordering error + usage of sqrt
    rtol = None
    atol = None
    if optim_cls is Adadelta:
        rtol = 5.5e-4
        atol = 5e-5

    # inductor/test_compiled_optimizers.py::CompiledOptimizerTests::test_nadam_tensor_lr_weight_decay_momentum_decay_decoupled_weight_decay_capturable_foreach_cuda_lambdalr
    # Mismatched elements: 100 / 100 (100.0%)
    # Greatest absolute difference: 1.4960765838623047e-05 at index (2, 0) (up to 1e-05 allowed)
    # Greatest relative difference: 1.686977884673979e-05 at index (2, 0) (up to 1.3e-06 allowed)
    if optim_cls is NAdam:
        atol = 1.5e-5
        rtol = 1.7e-5

    self.assertEqual(list(params_eager), list(params_compiled), atol=atol, rtol=rtol)

    for p_eager, p_compiled in zip(params_eager, params_compiled):
        self.assertEqual(
            state_eager[p_eager],
            state_compiled[p_compiled],
            atol=atol,
            rtol=rtol,
        )


def make_test(
    optim_cls,
    closure=None,
    scheduler_cls=None,
    kernel_count=2,
    device=GPU_TYPE,
    **kwargs,
):
    # [FUNCTION DOC] HELPER: Factory returning a test function for one optimizer configuration.
    # The returned test (config patched for a low fusion memory threshold, and wrapped to require
    # a GPU when the device is the GPU) is described in its own block. Arguments: optimizer
    # class, optional closure, optional scheduler class, expected kernel count (an int or a
    # checker function), device and optimizer kwargs.
    @config.patch("score_fusion_memory_threshold", 1)
    def test_fn(self):
        # [FUNCTION DOC] HELPER: Generated test body. Optionally enables cudagraphs (cuda device,
        # not Adagrad or SGD), moves tensor lr and betas to the device, builds two identical two-
        # layer models with gradients, an eager and a compiled optimizer (plus schedulers whose
        # last epoch is forced to one), runs two steps of each under no_grad, compares results
        # with check_optim, verifies cudagraphs ran if enabled, and checks the generated kernel
        # count (exact match, or via the checker function) when the class flag allows it. Resets
        # dynamo and inductor metrics first and closes its context stack at the end.
        stack = ExitStack()
        try:
            # https://github.com/pytorch/pytorch/issues/118715 for capturable Adagrad support
            # https://github.com/pytorch/pytorch/issues/118018 for capturable SGD support
            run_cudagraphs = device == "cuda" and optim_cls not in (Adagrad, SGD)
            if run_cudagraphs:
                stack.enter_context(config.patch({"triton.cudagraphs": True}))

            kwargs_compiled = deepcopy(kwargs)
            if isinstance(kwargs.get("lr"), torch.Tensor):
                kwargs["lr"] = kwargs["lr"].to(device)
                kwargs_compiled["lr"] = kwargs_compiled["lr"].to(device)

            if "betas" in kwargs and isinstance(kwargs["betas"][0], torch.Tensor):
                kwargs["betas"] = (
                    kwargs["betas"][0].to(device),
                    kwargs["betas"][1].to(device),
                )
                kwargs_compiled["betas"] = (
                    kwargs_compiled["betas"][0].to(device),
                    kwargs_compiled["betas"][1].to(device),
                )

            torch._dynamo.reset()
            torch._inductor.metrics.reset()
            input = torch.ones([10, 10], device=device)
            model_eager = torch.nn.Sequential(
                *[torch.nn.Linear(10, 10, device=device) for _ in range(2)]
            )
            model_eager(input).sum().backward()

            input = torch.ones([10, 10], device=device)
            model_compiled = deepcopy(model_eager)
            model_compiled(input).sum().backward()

            opt_eager = optim_cls(model_eager.parameters(), **kwargs)
            opt_compiled = optim_cls(model_compiled.parameters(), **kwargs_compiled)
            compiled_step = compile_opt(opt_compiled, closure=closure)

            if scheduler_cls:
                scheduler_compiled = create_scheduler(scheduler_cls, opt_compiled)
                scheduler_eager = create_scheduler(scheduler_cls, opt_eager)
                # some schedulers only change after at least an epoch has passed
                scheduler_compiled.last_epoch = 1
                scheduler_eager.last_epoch = 1

            with torch.set_grad_enabled(False):
                for _ in range(2):
                    compiled_step()
                    opt_eager.step()
                    if scheduler_cls:
                        call_scheduler(scheduler_eager)
                        call_scheduler(scheduler_compiled)

            check_optim(
                self,
                optim_cls,
                model_eager.parameters(),
                model_compiled.parameters(),
                opt_eager.state,
                opt_compiled.state,
            )

            if run_cudagraphs:
                self.check_cudagraphs_ran()

            if self.check_kernel_count:
                # currently, we compile the step and the rest of the computation
                # separately because the step is a single element tensor
                # hence, the usual kernel count is 2
                if isinstance(kernel_count, types.LambdaType):
                    kernel_count(str(torch._inductor.metrics.generated_kernel_count))
                else:
                    self.assertEqual(
                        torch._inductor.metrics.generated_kernel_count, kernel_count
                    )
        finally:
            stack.close()

    if device == GPU_TYPE:
        test_fn = requires_gpu(test_fn)

    return test_fn


def make_recompile_test(optim_cls, closure=None, kernel_count=2, **kwargs):
    # [FUNCTION DOC] HELPER: Factory returning a GPU-only test function that checks recompilation
    # behavior of an optimizer step. Arguments: optimizer class, optional closure (unused),
    # expected kernel count (may be None) and optimizer kwargs.
    @config.patch("score_fusion_memory_threshold", 1)
    @requires_gpu
    def test_fn(self):
        # [FUNCTION DOC] HELPER: Generated test body. Compiles the step, runs four steps
        # expecting no recompile, then perturbs state (change lr for SGD, replace the Adagrad sum
        # tensor, replace the Adam exp_avg tensor, else clear all state) and steps again to force
        # a recompile. If kernel counting is on and a count is given, asserts the generated
        # kernel count equals twice the count.
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        input = torch.ones([10, 10], device=GPU_TYPE)
        model = torch.nn.Sequential(
            *[torch.nn.Linear(10, 10, device=GPU_TYPE) for _ in range(2)]
        )
        model(input).sum().backward()

        opt_compiled = optim_cls(model.parameters(), **kwargs)
        compiled_step = compile_opt(opt_compiled)

        # check no recompile here
        with torch.set_grad_enabled(False):
            for _ in range(4):
                compiled_step()

            # perturb state to force recompile
            # SGD has an empty state
            if optim_cls is SGD:
                opt_compiled.param_groups[0]["lr"] = 0.02
            elif optim_cls is Adagrad:
                state_tensor = opt_compiled.state[
                    opt_compiled.param_groups[0]["params"][0]
                ]["sum"]
                opt_compiled.state[opt_compiled.param_groups[0]["params"][0]]["sum"] = (
                    torch.zeros_like(state_tensor)
                )
            elif optim_cls is Adam:  # ensure we are guarding on the data_ptr of states
                state_tensor = opt_compiled.state[
                    opt_compiled.param_groups[0]["params"][0]
                ]["exp_avg"]
                opt_compiled.state[opt_compiled.param_groups[0]["params"][0]][
                    "exp_avg"
                ] = torch.zeros_like(state_tensor)
            else:
                opt_compiled.state.clear()

            compiled_step()

        if self.check_kernel_count and kernel_count is not None:
            # currently, we compile the step and the rest of the computation
            # separately because the step is a single element tensor
            # hence, the usual kernel count is 2
            # multiply by 2 to account for the recompile
            multiplier = 2

            self.assertEqual(
                torch._inductor.metrics.generated_kernel_count,
                multiplier * kernel_count,
            )

    return test_fn


class CompiledOptimizerParityTests(TestCase):
    # [CLASS DOC] Test case that compares compiled and eager optimizer behavior across the
    # optimizer database. Its device-generic test is instantiated per device (excluding cpu) by
    # the module-level instantiation call.
    @skipCUDAIf(not has_triton(), "torch.compile with cuda requires triton")
    @skipXPUIf(not has_triton(), "torch.compile with xpu requires triton")
    @optims(optim_db, dtypes=[torch.float32])
    @parametrize("use_closure", [True, False])
    def test_correctness(self, device, dtype, optim_info, use_closure):
        # [FUNCTION DOC] Device-generic test: for each optimizer in the database (float32), with
        # and without a closure, builds identical eager and compiled models and optimizers for
        # every optimizer input set (skipping differentiable), and for each scheduler class when
        # the lr is a tensor. Seeds all RNGs, converts gradients to sparse for SparseAdam, runs
        # two eager steps and two compiled steps, and compares params and state with check_optim.
        # Returns early when the optimizer requires a closure but none is used. Skipped without
        # triton.
        torch.get_device_module(device).manual_seed_all(0)
        torch.manual_seed(0)
        random.seed(0)
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )

        if optim_info.step_requires_closure and not use_closure:
            return

        for optim_input in all_optim_inputs:
            kwargs = optim_input.kwargs

            use_scheduler = isinstance(kwargs.get("lr", None), torch.Tensor)
            scheduler_classes = (
                list(LR_SCHEDULER_TO_KWARGS.keys()) if use_scheduler else [None]
            )

            for scheduler_cls in scheduler_classes:
                torch._dynamo.reset()
                torch._inductor.metrics.reset()
                input = torch.ones([10, 10], device=device)
                model_eager = torch.nn.Sequential(
                    *[
                        torch.nn.Linear(10, 10, device=device, bias=False)
                        for _ in range(2)
                    ]
                )
                model_eager(input).sum().backward()
                model_compiled = deepcopy(model_eager)
                model_compiled(input).sum().backward()

                if optim_cls is SparseAdam:
                    for param in model_eager.parameters():
                        param.grad = param.grad.to_sparse()
                    for param in model_compiled.parameters():
                        param.grad = param.grad.to_sparse()

                opt_compiled = optim_cls(
                    model_compiled.parameters(), **deepcopy(kwargs)
                )
                opt_eager = optim_cls(model_eager.parameters(), **deepcopy(kwargs))
                if scheduler_cls:
                    scheduler_compiled = create_scheduler(scheduler_cls, opt_compiled)
                    scheduler_eager = create_scheduler(scheduler_cls, opt_eager)
                    # some schedulers only change after at least an epoch has passed
                    scheduler_compiled.last_epoch = 1
                    scheduler_eager.last_epoch = 1

                num_steps = 2
                if use_closure:

                    @torch.compile()
                    def fn():
                        # [FUNCTION DOC] HELPER: Compiled function that defines a closure computing
                        # the loss, backpropagating (and sparsifying grads if required), steps the
                        # compiled optimizer with it and then advances the scheduler if any.
                        def closure():
                            # [FUNCTION DOC] HELPER: Closure for the compiled optimizer: forward,
                            # sum, backward on the compiled model; converts grads to sparse
                            # when the optimizer only supports sparse grads; returns the loss.
                            loss = model_compiled(input).sum()
                            loss.backward()
                            if optim_info.only_supports_sparse_grads:
                                for param in model_compiled.parameters():
                                    param.grad = param.grad.to_sparse()
                            return loss

                        opt_compiled.step(closure)
                        if scheduler_cls:
                            call_scheduler(scheduler_compiled)

                    def closure_eager():
                        # [FUNCTION DOC] HELPER: Closure for the eager optimizer: forward, sum and
                        # backward on the eager model, converting grads to sparse when needed,
                        # returning the loss.
                        loss = model_eager(input).sum()
                        loss.backward()
                        if optim_info.only_supports_sparse_grads:
                            for param in model_eager.parameters():
                                param.grad = param.grad.to_sparse()

                        return loss

                    for _ in range(num_steps):
                        opt_eager.step(closure_eager)
                        if scheduler_cls:
                            call_scheduler(scheduler_eager)
                else:

                    @torch.compile()
                    def fn():
                        # [FUNCTION DOC] HELPER: Compiled function that steps the compiled
                        # optimizer and then the scheduler if any.
                        opt_compiled.step()
                        if scheduler_cls:
                            call_scheduler(scheduler_compiled)

                    for _ in range(num_steps):
                        opt_eager.step()
                        if scheduler_cls:
                            call_scheduler(scheduler_eager)

                for _ in range(num_steps):
                    fn()

                check_optim(
                    self,
                    optim_cls,
                    model_eager.parameters(),
                    model_compiled.parameters(),
                    opt_eager.state,
                    opt_compiled.state,
                )


class CompiledOptimizerTests(TestCase):
    # [CLASS DOC] Test case holding generated per-optimizer compile tests (attached after class
    # creation by module-level loops) plus handwritten tests. Class attributes alias the model-
    # check helpers and a flag that enables kernel count assertions. Setup and teardown reset
    # dynamo and inductor metrics.
    check_model_gpu = check_model_gpu
    check_model_cpu = check_model
    check_kernel_count = True

    def setUp(self):
        # [FUNCTION DOC] Runs the base setup, then resets dynamo and inductor metrics.
        super().setUp()
        torch._dynamo.reset()
        torch._inductor.metrics.reset()

    def tearDown(self):
        # [FUNCTION DOC] Runs the base teardown, then resets dynamo and inductor metrics.
        super().tearDown()
        torch._dynamo.reset()
        torch._inductor.metrics.reset()

    def check_cudagraphs_ran(self):
        # [FUNCTION DOC] HELPER: Asserts that cudagraph trees ran for device zero: the tree
        # manager exists and asking it for a new graph id yields one.
        # We run the zeroth device currently
        manager = torch._inductor.cudagraph_trees.get_container(0).tree_manager
        self.assertIsNotNone(manager)
        self.assertEqual(manager.new_graph_id().id, 1)

    def test_create_scheduler_does_not_mutate_kwargs(self):
        # [FUNCTION DOC] Checks that creating ChainedScheduler and SequentialLR schedulers does
        # not modify the shared default kwargs mapping.
        for scheduler_cls in (ChainedScheduler, SequentialLR):
            expected_kwargs = deepcopy(LR_SCHEDULER_TO_KWARGS[scheduler_cls])
            model = torch.nn.Linear(1, 1)
            opt = SGD(model.parameters(), lr=0.1)

            create_scheduler(scheduler_cls, opt)

            self.assertEqual(LR_SCHEDULER_TO_KWARGS[scheduler_cls], expected_kwargs)

    test_adam_recompile = make_recompile_test(Adam, lr=0.01)
    test_adamw_recompile = make_recompile_test(AdamW, lr=0.01)
    test_adamax_recompile = make_recompile_test(Adamax, lr=0.01)
    test_nadam_recompile = make_recompile_test(NAdam, lr=0.01)
    test_rprop_recompile = make_recompile_test(Rprop, lr=0.01, kernel_count=2)
    test_rmsprop_recompile = make_recompile_test(RMSprop, lr=0.01)
    test_adadelta_recompile = make_recompile_test(Adadelta, lr=0.01)
    test_adagrad_recompile = make_recompile_test(Adagrad, lr=0.01, kernel_count=None)
    test_asgd_recompile_default = make_recompile_test(ASGD, lr=0.01)
    test_asgd_recompile_single = make_recompile_test(
        ASGD, kernel_count=8, lr=0.01, foreach=False
    )
    test_asgd_recompile_foreach = make_recompile_test(ASGD, lr=0.01, foreach=True)
    test_sgd_recompile_single = make_recompile_test(
        SGD, kernel_count=4, lr=0.01, foreach=False
    )
    test_sgd_recompile_foreach = make_recompile_test(
        SGD, kernel_count=1, lr=0.01, foreach=True
    )

    @skipIfWindows
    @requires_gpu
    def test_static_address_finalizer(self):
        # [FUNCTION DOC] Skipped on Windows and without a GPU. Checks that compiling an Adam step
        # does not keep a parameter alive: the garbage collector is disabled, a parameter
        # reference is taken inside a nested function, and after that function returns the weak
        # reference must be dead. Re-enables gc at the end.
        import gc

        gc.disable()
        p_ref = None

        def fn():
            # [FUNCTION DOC] HELPER: Builds a linear module with gradients, an Adam optimizer,
            # compiles a step function under no_grad and runs it, then stores a weak reference to
            # the last parameter in the enclosing scope and checks it is still alive.
            nonlocal p_ref
            mod = torch.nn.Linear(10, 10, device=GPU_TYPE, bias=False)
            for p in mod.parameters():
                p.grad = torch.rand_like(p)

            opt = torch.optim.Adam(mod.parameters(), lr=0.1)

            def fn():
                # [FUNCTION DOC] HELPER: Runs one optimizer step.
                opt.step()

            with torch.set_grad_enabled(False):
                step_fn_compiled = torch.compile(fn)
                step_fn_compiled()
            p_ref = weakref.ref(p)
            self.assertTrue(p_ref() is not None)

        fn()

        self.assertTrue(p_ref() is None)
        gc.enable()

    def test_guard_on_none_grads(self):
        # [FUNCTION DOC] Checks guards on None gradients with the eager backend: a training loop
        # with Adam where gradients are set to None on one iteration (so step does nothing) must
        # give the same list of recorded step values compiled and uncompiled.
        def training_loop():
            # [FUNCTION DOC] HELPER: Runs six iterations of zero grad, optional forward and
            # backward (skipped on iteration three), and Adam step on a small network, recording
            # the first parameter's step state each time. Returns the list.
            input = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).reshape(3, 2)

            model = torch.nn.Sequential(
                torch.nn.Linear(2, 3),
                torch.nn.Sigmoid(),
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid(),
            )

            params = list(model.parameters())
            optimizer = torch.optim.Adam(params)
            step_list = []

            for i in range(6):
                optimizer.zero_grad()
                # Test that step behaves as expected (a no-op) when grads are set to None
                if i != 3:
                    output = model(input)
                    loss = output.sum()
                    loss.backward()

                optimizer.step()
                step_list.append(optimizer.state[params[0]]["step"])

            return step_list

        compiled_training_loop = torch.compile(training_loop, backend="eager")
        actual_steps = compiled_training_loop()
        expected_steps = training_loop()
        self.assertEqual(actual_steps, expected_steps)

    # Basic shampoo test to verify we support compiling the various ops without error
    @requires_gpu
    def test_basic_shampoo(self):
        # [FUNCTION DOC] Smoke test (GPU) that a simplified Shampoo-style update built from
        # foreach ops compiles and gives the same result as eager on transposed slices of a
        # buffer with random gradients.
        param_buf = torch.rand((1024, 128))
        param_buf_c = param_buf.detach().clone()

        params_c = [param_buf_c[0:512, :].t(), param_buf_c[512:, :].t()]
        params = [param_buf[0:512, :].t(), param_buf[512:, :].t()]

        for p, p_c in zip(params, params_c):
            p.grad = torch.rand_like(p)
            p_c.grad = p.grad.detach().clone()

        # note this skips the root inverse because this has a lot of internal dependencies
        # we also don't compile it regardless
        @torch.no_grad()
        def shampoo_functional_basic(params):
            # [FUNCTION DOC] HELPER: Functional Shampoo-like step under no_grad: weight decay,
            # preconditioner update, momentum, bias corrections, preconditioning, a decay term
            # and the parameter update, all with foreach ops, mutating the given params. Returns
            # params, preconditioners and grads.
            step = 1
            weight_decay = 0.1
            grads = [p.grad for p in params]
            beta1 = 0.9
            beta2 = 1.0
            epsilon = 1e-10
            preconditioners = [torch.zeros_like(p) for p in params]
            lr = 0.01

            # pt2 region 1
            # weight decay
            torch._foreach_add_(grads, params, alpha=weight_decay)

            # update preconditioners
            torch._foreach_addcmul_(preconditioners, grads, grads, value=1.0)

            torch._foreach_mul_(grads, beta1)
            torch._foreach_add_(
                grads,
                grads,
                alpha=1 - beta1,
            )
            bias_correction1 = 1.0 - beta1**step
            grad_list = torch._foreach_div(grads, bias_correction1)

            # pt2 region 2
            # precondition (with shampoo branch), with no grafting
            bias_correction2 = 1.0 - beta2**step
            bias_corrected_preconditioner_list = torch._foreach_div(
                preconditioners, bias_correction2
            )
            torch._foreach_sqrt_(bias_corrected_preconditioner_list)
            torch._foreach_add_(bias_corrected_preconditioner_list, epsilon)
            search_directions = torch._foreach_div(
                grad_list, bias_corrected_preconditioner_list
            )

            torch._foreach_add_(
                search_directions,
                params,
                alpha=weight_decay,
            )

            torch._foreach_mul_(search_directions, -lr)
            # pt2 region 3 update params
            torch._foreach_add_(params, search_directions)

            return params, preconditioners, grads

        compiled_fn = torch.compile(shampoo_functional_basic)

        self.assertEqual(compiled_fn(params_c), shampoo_functional_basic(params))

    @requires_gpu
    def test_closure_graph_break(self):
        # [FUNCTION DOC] Checks (GPU) that a closure which sets gradients causes a graph break
        # but still yields the same parameter as eager when an AdamW step is run through a
        # function compiled with the eager backend.
        param = torch.rand(
            2, 3, dtype=torch.float32, device=GPU_TYPE, requires_grad=True
        )
        param_c = param.detach().clone().requires_grad_(True)

        def closure():
            # [FUNCTION DOC] HELPER: Sets the gradient of the first parameter to twos and returns
            # it.
            param.grad = torch.ones_like(param) * 2
            return param.grad

        def closure_c():
            # [FUNCTION DOC] HELPER: Sets the gradient of the cloned parameter to twos and
            # returns it.
            param_c.grad = torch.ones_like(param_c) * 2
            return param_c.grad

        optimizer = torch.optim.AdamW([param])
        optimizer_c = torch.optim.AdamW([param_c])

        def loop(opt, c):
            # [FUNCTION DOC] HELPER: Calls the optimizer step with the closure.
            opt.step(c)

        compiled_loop = torch.compile(loop, backend="eager")

        compiled_loop(optimizer, closure)
        loop(optimizer_c, closure_c)

        self.assertEqual(param, param_c)

    def test_get_value_on_static_address(self):
        # [FUNCTION DOC] Checks that the optimizer _get_value helper compiles on a tensor marked
        # as a static address (with guard) and returns an equal tensor.
        from torch._dynamo.decorators import mark_static_address
        from torch.optim.optimizer import _get_value

        compiled = torch.compile(_get_value)

        x = torch.ones(2, 2)
        mark_static_address(x, guard=True)

        ret_val = compiled(x)

        self.assertEqual(ret_val, x)

    # compile a large foreach op and verify
    # that the time taken is within an expected range
    @requires_gpu
    def test_compile_time_smoketest(self):
        # [FUNCTION DOC] Smoke test (GPU): compiles a foreach add of two hundred small tensors
        # and asserts the wall time of the first call is under ninety seconds.
        import time

        xs = [torch.ones(2, 2, device=GPU_TYPE) for _ in range(100)]
        ys = [torch.ones(2, 2, device=GPU_TYPE) for _ in range(100)]

        @torch.compile
        def fn(xs, ys):
            # [FUNCTION DOC] HELPER: Returns the foreach add of two tensor lists.
            return torch._foreach_add(xs, ys)

        start = time.perf_counter()
        fn(xs, ys)
        end = time.perf_counter()

        self.assertLess(end - start, 90)

    @requires_gpu_and_triton
    @skipIfRocm(msg="ROCm Triton compile time regresses on joined foreach bodies")
    def test_foreach_shared_body_codegen(self):
        # [FUNCTION DOC] Checks triton code generation for a mixed-size foreach add (GPU, triton,
        # not ROCm). For two size sets, with a fresh cache and mixed-size combo kernels enabled,
        # the compiled result must match eager and generate exactly one foreach kernel with two
        # loads and one store, with a shared pointer variable assigned from both the first and
        # third input pointers, and with the last load of it appearing after the last branch
        # condition for that size set.
        from torch._inductor.utils import fresh_cache, run_and_get_code

        def fn(xs, ys):
            # [FUNCTION DOC] HELPER: Returns the foreach add of two tensor lists.
            return torch._foreach_add(xs, ys)

        for sizes, last_branch in (
            ((1536, 2048, 2560), "elif pid < num_xblocks_2"),
            ((1536, 2048, 2304), "elif pid % 3 == 2"),
        ):
            with (
                self.subTest(sizes=sizes),
                fresh_cache(),
                config.patch(combo_kernel_allow_mixed_sizes=2),
            ):
                xs = [torch.randn(n, device=GPU_TYPE) for n in sizes]
                ys = [torch.randn_like(x) for x in xs]
                expected = torch._foreach_add(xs, ys)
                actual, codes = run_and_get_code(
                    torch.compile(fn, fullgraph=True), xs, ys
                )

            self.assertEqual(actual, expected)
            foreach_codes = [
                code for code in codes if "@triton_heuristics.foreach" in code
            ]
            self.assertEqual(len(foreach_codes), 1)
            code = foreach_codes[0]
            self.assertEqual(code.count("tl.load("), 2)
            self.assertEqual(code.count("tl.store("), 1)
            self.assertIn("foreach_arg0 = in_ptr0", code)
            self.assertIn("foreach_arg0 = in_ptr2", code)
            self.assertGreater(
                code.rfind("tmp0 = tl.load(foreach_arg0"),
                code.rfind(last_branch),
            )

    @requires_gpu_and_triton
    def test_foreach_optimizer_shared_body_correctness(self):
        # [FUNCTION DOC] Checks correctness of a compiled momentum-style foreach update on three
        # tensors of different sizes (GPU, triton): params and momentum match the eager
        # reference, and the generated foreach kernels use the shared argument variable except on
        # ROCm, where none may.
        from torch._inductor.utils import fresh_cache, run_and_get_code

        def opt_step(params, grads, momentum):
            # [FUNCTION DOC] HELPER: Scales momentum by 0.9, adds gradients scaled by 0.1, then
            # adds momentum scaled by minus 0.01 to params, all in place with foreach ops.
            # Returns params and momentum.
            torch._foreach_mul_(momentum, 0.9)
            torch._foreach_add_(momentum, grads, alpha=0.1)
            torch._foreach_add_(params, momentum, alpha=-0.01)
            return params, momentum

        params = [torch.randn(n, device=GPU_TYPE) for n in (1536, 2048, 2560)]
        grads = [torch.randn_like(p) for p in params]
        momentum = [torch.zeros_like(p) for p in params]
        params_ref = [p.clone() for p in params]
        momentum_ref = [m.clone() for m in momentum]

        with fresh_cache():
            _, codes = run_and_get_code(
                torch.compile(opt_step, fullgraph=True), params, grads, momentum
            )
        opt_step(params_ref, grads, momentum_ref)

        self.assertEqual(params, params_ref)
        self.assertEqual(momentum, momentum_ref)
        foreach_codes = [code for code in codes if "@triton_heuristics.foreach" in code]
        if torch.version.hip is not None:
            self.assertFalse(any("foreach_arg0" in code for code in foreach_codes))
        else:
            self.assertTrue(any("foreach_arg0" in code for code in foreach_codes))

    @requires_gpu_and_triton
    def test_S429861(self):
        # [FUNCTION DOC] Only checks that a function from an external repro module compiles
        # without error using inputs parsed from its saved graph, under a fresh cache (GPU,
        # triton). Imports the repro module relatively or absolutely.
        # Just verify we can compile this function without error
        try:
            from . import s429861_repro
        except ImportError:
            import s429861_repro  # @manual

        forward = s429861_repro.forward

        import torch._dynamo
        import torch._inductor
        from torch._dynamo.debug_utils import aot_graph_input_parser
        from torch._inductor.utils import fresh_cache

        with fresh_cache():
            kwargs = aot_graph_input_parser(forward, device=GPU_TYPE)
            torch.compile(forward)(**kwargs)

    @requires_gpu_and_triton
    def test_foreach_map_adam(self):
        # [FUNCTION DOC] Checks (GPU, triton) that an Adam step expressed with the foreach_map
        # helper, compiled, produces the same parameters as the real capturable foreach Adam
        # optimizer after a warm-up step, and that exactly two kernels are generated. Uses ten
        # large parameters with random gradients.
        params = [
            torch.rand(
                1000, 1000, dtype=torch.float32, device=GPU_TYPE, requires_grad=True
            )
            for _ in range(10)
        ]

        for param in params:
            param.grad = torch.rand_like(param)

        params_ref = [p.detach().clone().requires_grad_(True) for p in params]
        for param, param_ref in zip(params, params_ref):
            param_ref.grad = param.grad.detach().clone()

        optimizer = torch.optim.Adam(params, capturable=True, foreach=True)
        optimizer_ref = torch.optim.Adam(params_ref, capturable=True, foreach=True)

        # warm up the optimizer state
        optimizer.step()
        optimizer_ref.step()

        inps = get_inputs(optimizer)

        @torch.compile()
        def foreach_map_adam_step():
            # [FUNCTION DOC] HELPER: Compiled function that runs the foreach_map Adam step on the
            # captured state lists.
            foreach_map_adam(*inps)

        def loop():
            # [FUNCTION DOC] HELPER: Runs the compiled foreach_map step and then the reference
            # optimizer step.
            foreach_map_adam_step()
            optimizer_ref.step()

        loop()

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

        for param, param_ref in zip(params, params_ref):
            self.assertEqual(param, param_ref)


@skipIfRocm(msg="ROCm may have different numerical behavior")
@requires_gpu_and_triton
class CompiledOptimizerBitwiseTests(TestCase):
    # [CLASS DOC] Test case (skipped on ROCm, needs GPU and triton) asserting compiled optimizer
    # results are bitwise identical to eager when precision-related config options are enabled. A
    # static helper does the comparison; generated tests are attached at module level.
    """
    Tests that compiled optimizers produce bitwise identical results to eager
    when precision configs are enabled.

    These tests verify that with the following config options:
    - eager_numerics.division_rounding = True
    - eager_numerics.use_pytorch_libdevice = True
    - emulate_precision_casts = True

    The compiled optimizer step produces results that are bitwise identical
    to the eager optimizer step.
    """

    @config.patch(
        {
            "score_fusion_memory_threshold": 1,
            "eager_numerics.division_rounding": True,
            "eager_numerics.use_pytorch_libdevice": True,
            "emulate_precision_casts": True,
        }
    )
    def test_foreach_lerp_scalar_high_weight_bitwise(self):
        # [FUNCTION DOC] Checks (with the precision configs on) that compiled foreach lerp with a
        # scalar weight is bitwise equal (zero tolerance) to eager for several weights and dtype
        # pairs, including mixed float32 and float64 and half precision.
        cases = [
            (0.9, (torch.float32, torch.float32)),
            (0.1, (torch.float32, torch.float32)),
            (0.9, (torch.float32, torch.float64)),
            (0.49999999, (torch.float32, torch.float64)),
            (0.9, (torch.float16, torch.float16)),
        ]

        for weight, dtypes in cases:
            with self.subTest(weight=weight, dtypes=dtypes):

                def fn(start, end):
                    # [FUNCTION DOC] HELPER: Returns the foreach lerp of the two tensor lists
                    # with the loop's scalar weight.
                    return torch._foreach_lerp(start, end, weight)

                torch.manual_seed(42)
                start = [
                    torch.randn(32, device=GPU_TYPE, dtype=dtypes[0]),
                    torch.randn(16, device=GPU_TYPE, dtype=dtypes[1]),
                ]
                end = [
                    torch.randn(32, device=GPU_TYPE, dtype=dtypes[0]),
                    torch.randn(16, device=GPU_TYPE, dtype=dtypes[1]),
                ]
                expected = fn(start, end)
                actual = torch.compile(fn)(start, end)
                self.assertEqual(actual, expected, atol=0, rtol=0)

    @staticmethod
    def _test_optimizer_bitwise(
        test_case,
        optim_cls,
        kernel_count=None,
        num_steps=10,
        **optim_kwargs,
    ):
        # [FUNCTION DOC] HELPER: Static helper. Resets dynamo and metrics, seeds, builds two
        # identical models with gradients, an eager and a compiled optimizer, runs the given
        # number of steps of each under no_grad, and after each step asserts all parameters are
        # bitwise equal (zero tolerances). Afterwards asserts every tensor entry of optimizer
        # state is bitwise equal. Finally, if a kernel count is given and the test case enables
        # counting, checks the generated kernel count (via a checker function or equality).
        """Helper to test optimizer bitwise equality."""
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        torch.manual_seed(42)

        input = torch.ones([10, 10], device=GPU_TYPE)
        model_eager = torch.nn.Sequential(
            *[torch.nn.Linear(10, 10, device=GPU_TYPE) for _ in range(2)]
        )
        model_eager(input).sum().backward()

        model_compiled = deepcopy(model_eager)
        model_compiled(input).sum().backward()

        opt_eager = optim_cls(model_eager.parameters(), **optim_kwargs)
        opt_compiled = optim_cls(model_compiled.parameters(), **optim_kwargs)
        compiled_step = compile_opt(opt_compiled)

        with torch.set_grad_enabled(False):
            for step in range(num_steps):
                compiled_step()
                opt_eager.step()

                # Check bitwise equality
                for i, (p_eager, p_compiled) in enumerate(
                    zip(model_eager.parameters(), model_compiled.parameters())
                ):
                    test_case.assertEqual(
                        p_eager,
                        p_compiled,
                        atol=0,
                        rtol=0,
                        msg=lambda msg: f"{msg}\nStep {step + 1}, param {i}: params differ",
                    )

        # Also check optimizer state
        for p_eager, p_compiled in zip(
            model_eager.parameters(), model_compiled.parameters()
        ):
            for key in opt_eager.state[p_eager]:
                eager_val = opt_eager.state[p_eager][key]
                compiled_val = opt_compiled.state[p_compiled][key]
                if isinstance(eager_val, torch.Tensor):
                    test_case.assertEqual(
                        eager_val,
                        compiled_val,
                        atol=0,
                        rtol=0,
                        msg=lambda msg: f"{msg}\nState '{key}' differs",
                    )

        if kernel_count is not None and test_case.check_kernel_count:
            if isinstance(kernel_count, types.LambdaType):
                kernel_count(str(torch._inductor.metrics.generated_kernel_count))
            else:
                test_case.assertEqual(
                    torch._inductor.metrics.generated_kernel_count, kernel_count
                )


for optim_cls, name, kwargs, scheduler_cls in COMPILED_OPT_KWARG_DB:
    setattr(
        CompiledOptimizerTests,
        name,
        make_test(optim_cls, scheduler_cls=scheduler_cls, **kwargs),
    )


def _make_bitwise_test(optim_cls, kernel_count=None, **optim_kwargs):
    # [FUNCTION DOC] HELPER: Factory returning a bitwise test for an optimizer class and kwargs.
    # The returned test is skipped on ROCm and XPU, needs GPU and triton, and runs with config
    # patched for precision parity.
    @skipIfRocm(msg="ROCm may have different numerical behavior")
    @skipIfXpu(msg="AttributeError, torch-xpu-ops: #2999")
    @requires_gpu_and_triton
    @config.patch(
        {
            "score_fusion_memory_threshold": 1,
            "eager_numerics.division_rounding": True,
            "eager_numerics.use_pytorch_libdevice": True,
            "emulate_precision_casts": True,
        }
    )
    def test_fn(self):
        # [FUNCTION DOC] HELPER: Generated test body; delegates to the static bitwise helper of
        # CompiledOptimizerBitwiseTests with the test case, optimizer class, kernel count and
        # optimizer kwargs.
        CompiledOptimizerBitwiseTests._test_optimizer_bitwise(
            self, optim_cls, kernel_count=kernel_count, **optim_kwargs
        )

    return test_fn


_BITWISE_CAPTURABLE_OPTIMS = (
    Adam,
    AdamW,
    Adadelta,
    Adamax,
    ASGD,
    NAdam,
    RAdam,
    RMSprop,
    Rprop,
)
# SGD doesn't support capturable but has no item() calls
# so it compiles without graph breaks and can be tested bitwise.
_BITWISE_NON_CAPTURABLE_OPTIMS = (SGD,)

for optim_cls, name, kwargs, scheduler_cls in COMPILED_OPT_KWARG_DB:
    if (
        kwargs.get("device") == GPU_TYPE
        and "tensor_lr" not in name
        and scheduler_cls is None
        and (
            (
                optim_cls in _BITWISE_CAPTURABLE_OPTIMS
                and kwargs.get("capturable", False)
            )
            or optim_cls in _BITWISE_NON_CAPTURABLE_OPTIMS
        )
    ):
        bitwise_name = name.replace("test_", "test_bitwise_")
        # Use the same kernel count as the non-bitwise test, including
        # any overrides for specific test configurations.
        if name in KERNEL_COUNT_OVERRIDES:
            kernel_count = KERNEL_COUNT_OVERRIDES[name]
        else:
            kernel_count = (
                KERNEL_COUNTS[optim_cls].multitensor
                if kwargs.get("foreach", False)
                else KERNEL_COUNTS[optim_cls].singletensor
            )
        optim_kwargs = {
            k: v for k, v in kwargs.items() if k not in ("device", "kernel_count")
        }
        setattr(
            CompiledOptimizerTests,
            bitwise_name,
            _make_bitwise_test(optim_cls, kernel_count=kernel_count, **optim_kwargs),
        )


instantiate_device_type_tests(
    CompiledOptimizerParityTests, globals(), allow_xpu=True, except_for="cpu"
)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests(needs="filelock")
