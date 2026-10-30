# Owner(s): ["module: optimizer"]

from __future__ import annotations
# [MODULE DOC] Gradient-check tests for differentiable optimizer steps. It defines two helper
# functions that run optimizer steps on cloned inputs so that gradcheck can track state tensors
# and hyperparameters, and one test class that checks first-order gradients through a single
# step of many optimizers and multi-step meta-gradients with respect to learning rate, weight
# decay and betas for Adam, AdamW and SGD. Import-time side effect: rebinds the sharding
# test-filter hook. Running the file directly only prints a note to use the main optimizer
# test file.

from typing import Any

import torch
from torch import nn, Tensor
from torch.optim import (
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    ASGD,
    NAdam,
    Optimizer,
    RAdam,
    RMSprop,
    Rprop,
    SGD,
)
from torch.testing._internal.common_utils import (
    gradcheck,
    HardwareClassification,
    load_tests,
    skipIfTorchDynamo,
    TestCase,
)


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127


def _diff_fn(p, grad, opt_differentiable_state, opt_class, kwargs, *ignored):
    # [FUNCTION DOC] HELPER: Function under test for gradcheck. Clones the parameter, attaches
    # the given gradient, clones tensor entries of the state mapping, builds an optimizer of the
    # given class with the keyword arguments, seeds its state for the parameter with the cloned
    # state, and runs one step. Returns a tuple of the updated parameter followed by every state
    # tensor that requires grad. The extra positional arguments are ignored; they exist only so
    # that gradcheck sees the state tensors as inputs.
    # Ignored is the list of values in `opt_differentiable_state`, we do this
    # for `gradcheck` to correctly track the state tensors as function inputs
    # because otherwise it can't unpack the values in the `opt_differentiable_state`
    # dict
    p = p.clone()
    p.grad = grad
    opt_differentiable_state = {
        k: v.clone() if isinstance(v, torch.Tensor) else v
        for k, v in opt_differentiable_state.items()
    }
    opt = opt_class([p], **kwargs)
    opt.state[p].update(opt_differentiable_state)
    opt.step()
    return (p,) + tuple(
        v
        for v in opt.state[p].values()
        if isinstance(v, torch.Tensor) and v.requires_grad
    )


def _multistep_backprop_diff_hyperparams_fn(
    params: Tensor,
    grad: Tensor,
    opt_differentiable_state: dict[str, Any],
    opt_class: type[Optimizer],
    kwargs: dict[str, Any],
    *ignored: Any,
) -> tuple[Tensor, ...]:
    # [FUNCTION DOC] HELPER: Function under test for gradcheck that differentiates through two
    # optimizer steps with respect to hyperparameters. Requires the differentiable flag in
    # kwargs, else raises AssertionError. Clones params, state and kwargs tensors; if separate
    # beta1 and beta2 entries are given (both must be present, else AssertionError) they are
    # merged into a betas pair. Collects the hyperparameter tensors that require grad, runs two
    # steps with a squared error loss on a fixed scalar pair (keeping the graph), backpropagates
    # the final loss to those hyperparameters and raises AssertionError if any got no gradient.
    # Returns the final loss, the state tensors requiring grad and the hyperparameter tensors.
    # KNOWN ISSUE: an existing comment inside this function cites a source line number that
    # is stale.
    if kwargs["differentiable"] is not True:
        raise AssertionError("Only call this test function when differentiable=True")

    params = params.clone()
    params.grad = grad

    opt_differentiable_state = {
        k: v.clone() if isinstance(v, torch.Tensor) else v
        for k, v in opt_differentiable_state.items()
    }

    # This copy is necessary so the update on line 78 doesn't overwrite the original kwargs values
    kwargs = kwargs.copy()

    # Have to pass in beta1 and beta2 separately
    # so they're passed in as Tensors (not a tuple) and recognized by gradcheck
    if "beta1" in kwargs or "beta2" in kwargs:
        # Prevent just one beta kwarg from being passed in
        if not ("beta1" in kwargs and "beta2" in kwargs):
            raise AssertionError("Both betas should be defined in kwargs")
        kwargs.update({"betas": (kwargs.pop("beta1"), kwargs.pop("beta2"))})

    kwargs.update(
        {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    )
    differentiable_kwargs = [
        v for v in kwargs.values() if isinstance(v, torch.Tensor) and v.requires_grad
    ] + (list(kwargs["betas"]) if "betas" in kwargs else [])

    criterion = nn.MSELoss()

    optimizer = opt_class([params], **kwargs)
    optimizer.state[params].update(opt_differentiable_state)

    # Simple x, y pair
    x = torch.tensor([1.0], dtype=torch.float64)
    y = torch.tensor([2.0], dtype=torch.float64)

    for _ in range(2):
        loss = criterion(x * torch.sum(params), y)
        loss.backward(
            inputs=(params,),
            create_graph=True,
        )
        optimizer.step()
        optimizer.zero_grad()

    meta_loss = loss
    meta_loss.backward(inputs=(*differentiable_kwargs,), create_graph=True)

    # Extra check to make sure the test properly computed a gradient for all kwargs
    for kwarg in differentiable_kwargs:
        if kwarg.grad is None:
            raise AssertionError("Expected gradient for kwarg but got None")

    return (
        (meta_loss,)
        + tuple(
            v
            for v in optimizer.state[params].values()
            if isinstance(v, torch.Tensor) and v.requires_grad
        )
        + tuple(differentiable_kwargs)
    )


@skipIfTorchDynamo("Differentiable optimizers not supported")
class TestDifferentiableOptimizer(TestCase):
    # [CLASS DOC] Gradcheck tests of differentiable optimizer steps in double precision. Skipped
    # under Dynamo. The class attribute marks the tests as generic hardware. Single-step tests
    # use the first helper; hyperparameter tests use the multi-step helper.
    hw_classification = HardwareClassification.GENERIC

    def test_sgd(self):
        # [FUNCTION DOC] Gradchecks one differentiable SGD step with a random momentum buffer
        # as state.
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        mbuff = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state = {"momentum_buffer": mbuff}
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                SGD,
                {"lr": 0.9, "differentiable": True},
                *state.values(),
            ),
        )

    def test_adam(self):
        # [FUNCTION DOC] Gradchecks one differentiable Adam step with the amsgrad option, using
        # random first and second moment states and a fixed step count that is kept out
        # of differentiation.
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` is not a continuous variable (even though we define it as a float)
        # and so it shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["max_exp_avg_sq"] = torch.rand(
            10, requires_grad=True, dtype=torch.float64
        )

        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                Adam,
                {"lr": 0.9, "differentiable": True, "amsgrad": True},
                *state.values(),
            ),
        )

    def test_rmsprop(self):
        # [FUNCTION DOC] Gradchecks one differentiable RMSprop step with maximize, momentum,
        # centered and weight decay options, using random square average, momentum and (scaled
        # down) gradient average states and a zero step.
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["step"] = torch.zeros((), dtype=torch.float64)
        state["square_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["momentum_buffer"] = torch.rand(
            10, requires_grad=True, dtype=torch.float64
        )
        # This can cause issues with large values and nan due to sqrt ops
        state["grad_avg"] = 1e-2 * torch.rand(
            10, requires_grad=True, dtype=torch.float64
        )
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                RMSprop,
                {
                    "lr": 0.9,
                    "maximize": True,
                    "momentum": 0.9,
                    "differentiable": True,
                    "centered": True,
                    "weight_decay": 0.1,
                },
                *state.values(),
            ),
        )

    def test_adadelta(self):
        # [FUNCTION DOC] Gradchecks one differentiable Adadelta step with weight decay, using
        # random square average and accumulated delta states.
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` is not a continuous variable (even though we define it as a float)
        # and so it shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["square_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["acc_delta"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                Adadelta,
                {"lr": 0.9, "weight_decay": 0.1, "differentiable": True},
                *state.values(),
            ),
        )

    def test_adagrad(self):
        # [FUNCTION DOC] Gradchecks one differentiable Adagrad step with weight decay, using a
        # random sum state.
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` is not a continuous variable (even though we define it as a float)
        # and so it shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["sum"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                Adagrad,
                {"lr": 0.9, "weight_decay": 0.1, "differentiable": True},
                *state.values(),
            ),
        )

    def test_adamax(self):
        # [FUNCTION DOC] Gradchecks one differentiable Adamax step with weight decay, using
        # random first-moment and infinity-norm states.
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` is not a continuous variable (even though we define it as a float)
        # and so it shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["exp_inf"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                Adamax,
                {"lr": 0.9, "weight_decay": 0.1, "differentiable": True},
                *state.values(),
            ),
        )

    @skipIfTorchDynamo(
        "The inplace mu update fails with dynamo, "
        "since this is only happening when differentiable is enabled, skipping for now"
    )
    def test_asgd(self):
        # [FUNCTION DOC] Gradchecks one differentiable ASGD step with an averaged-parameter
        # state; the step, eta and mu entries are non-differentiated constants. Skipped under
        # Dynamo because of an in-place update of mu.
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` `eta` & `mu` are not continuous variables (even though we define them as floats)
        # and so they shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["eta"] = torch.tensor(0.9, requires_grad=False, dtype=torch.float64)
        state["mu"] = torch.tensor(1.0, requires_grad=False, dtype=torch.float64)
        state["ax"] = torch.rand(10, requires_grad=True, dtype=torch.float64)

        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                ASGD,
                {"lr": 0.9, "differentiable": True},
                *state.values(),
            ),
        )

    def test_rprop(self):
        # [FUNCTION DOC] Gradchecks one differentiable Rprop step using random previous-gradient
        # and step-size states.
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` is not a continuous variable (even though we define it as a float)
        # and so it shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["prev"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["step_size"] = torch.rand(10, requires_grad=True, dtype=torch.float64)

        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                Rprop,
                {"lr": 0.9, "differentiable": True},
                *state.values(),
            ),
        )

    def test_adamw(self):
        # [FUNCTION DOC] Gradchecks one differentiable AdamW step with the amsgrad option, using
        # random moment states.
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` is not a continuous variable (even though we define it as a float)
        # and so it shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["max_exp_avg_sq"] = torch.rand(
            10, requires_grad=True, dtype=torch.float64
        )

        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                AdamW,
                {"lr": 0.9, "differentiable": True, "amsgrad": True},
                *state.values(),
            ),
        )

    def test_nadam(self):
        # [FUNCTION DOC] Gradchecks one differentiable NAdam step twice: with default weight
        # decay handling and with decoupled weight decay enabled; the state includes a momentum
        # product tensor.
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` is not a continuous variable (even though we define it as a float)
        # and so it shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["mu_product"] = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)

        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                NAdam,
                {"lr": 0.9, "differentiable": True},
                *state.values(),
            ),
        )

        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                NAdam,
                {"lr": 0.9, "decoupled_weight_decay": True, "differentiable": True},
                *state.values(),
            ),
        )

    def test_radam(self):
        # [FUNCTION DOC] Gradchecks one differentiable RAdam step twice: with default options
        # and with weight decay in decoupled form.
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` is not a continuous variable (even though we define it as a float)
        # and so it shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)

        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                RAdam,
                {"lr": 0.9, "differentiable": True},
                *state.values(),
            ),
        )
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                RAdam,
                {
                    "lr": 0.9,
                    "weight_decay": 0.1,
                    "decoupled_weight_decay": True,
                    "differentiable": True,
                },
                *state.values(),
            ),
        )

    def test_adam_differentiable_lr(self):
        # [FUNCTION DOC] Gradchecks a two-step differentiable Adam run with respect to a
        # learning rate tensor that requires grad.
        params = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand_like(params, requires_grad=True, dtype=torch.float64)
        lr = torch.tensor(0.001, requires_grad=True, dtype=torch.float64)

        state = {}
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["max_exp_avg_sq"] = torch.rand(
            10, requires_grad=True, dtype=torch.float64
        )
        kwargs: dict[str, Any] = {"lr": lr, "differentiable": True}

        gradcheck(
            _multistep_backprop_diff_hyperparams_fn,
            (
                params,
                grad,
                state,
                Adam,
                kwargs,  # includes lr
                *state.values(),
                *kwargs.values(),
            ),
        )

    def test_adam_differentiable_weight_decay(self):
        # [FUNCTION DOC] Gradchecks a two-step differentiable Adam run with respect to a weight
        # decay tensor that requires grad.
        params = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand_like(params, requires_grad=True, dtype=torch.float64)
        weight_decay = torch.tensor(0.999, requires_grad=True, dtype=torch.float64)

        state = {}
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["max_exp_avg_sq"] = torch.rand(
            10, requires_grad=True, dtype=torch.float64
        )
        kwargs: dict[str, Any] = {"weight_decay": weight_decay, "differentiable": True}

        gradcheck(
            _multistep_backprop_diff_hyperparams_fn,
            (
                params,
                grad,
                state,
                Adam,
                kwargs,  # includes weight_decay
                *state.values(),
                *kwargs.values(),
            ),
        )

    def test_adam_differentiable_betas(self):
        # [FUNCTION DOC] Gradchecks a two-step differentiable Adam run with respect to both beta
        # tensors and a one-element learning rate tensor, passing the betas as separate entries
        # that the helper merges into a pair.
        params = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand_like(params, requires_grad=True, dtype=torch.float64)

        lr = torch.tensor([0.001], requires_grad=True, dtype=torch.float64)
        betas = (
            torch.tensor(0.9, requires_grad=True, dtype=torch.float64),
            torch.tensor(0.999, requires_grad=True, dtype=torch.float64),
        )
        state = {}
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["max_exp_avg_sq"] = torch.rand(
            10, requires_grad=True, dtype=torch.float64
        )

        # Have to pass in beta1 and beta2 separately
        # so they're passed in as Tensors (not a tuple) and recognized by gradcheck.
        # In the test, this is called: kwargs.update({betas: (beta1, beta2)})
        kwargs: dict[str, Any] = {
            "beta1": betas[0],
            "beta2": betas[1],
            "lr": lr,
            "differentiable": True,
        }

        gradcheck(
            _multistep_backprop_diff_hyperparams_fn,
            (
                params,
                grad,
                state,
                Adam,
                kwargs,  # includes betas
                *state.values(),
                *kwargs.values(),
            ),
        )

    def test_adam_differentiable_all_hyperparams(self):
        # [FUNCTION DOC] Gradchecks a two-step differentiable Adam run with respect to learning
        # rate, weight decay and both betas together.
        params = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand_like(params, requires_grad=True, dtype=torch.float64)

        lr = torch.tensor(0.001, requires_grad=True, dtype=torch.float64)
        weight_decay = torch.tensor(0.999, requires_grad=True, dtype=torch.float64)
        betas = (
            torch.tensor(0.9, requires_grad=True, dtype=torch.float64),
            torch.tensor(0.999, requires_grad=True, dtype=torch.float64),
        )
        state = {}
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["max_exp_avg_sq"] = torch.rand(
            10, requires_grad=True, dtype=torch.float64
        )

        # Have to pass in beta1 and beta2 separately
        # so they're passed in as Tensors (not a tuple) and recognized by gradcheck.
        # In the test, this is called: kwargs.update({betas: (beta1, beta2)})
        kwargs: dict[str, Any] = {
            "lr": lr,
            "weight_decay": weight_decay,
            "beta1": betas[0],
            "beta2": betas[1],
            "differentiable": True,
        }

        gradcheck(
            _multistep_backprop_diff_hyperparams_fn,
            (
                params,
                grad,
                state,
                Adam,
                kwargs,  # includes betas
                *state.values(),
                *kwargs.values(),
            ),
        )

    def test_adamw_differentiable_lr(self):
        # [FUNCTION DOC] Gradchecks a two-step differentiable AdamW run with respect to a
        # learning rate tensor that requires grad.
        params = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand_like(params, requires_grad=True, dtype=torch.float64)
        lr = torch.tensor(0.001, requires_grad=True, dtype=torch.float64)

        state = {}
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["max_exp_avg_sq"] = torch.rand(
            10, requires_grad=True, dtype=torch.float64
        )
        kwargs: dict[str, Any] = {"lr": lr, "differentiable": True}

        gradcheck(
            _multistep_backprop_diff_hyperparams_fn,
            (
                params,
                grad,
                state,
                AdamW,
                kwargs,  # includes lr
                *state.values(),
                *kwargs.values(),
            ),
        )

    def test_adamw_differentiable_weight_decay(self):
        # [FUNCTION DOC] Gradchecks a two-step differentiable AdamW run with respect to a weight
        # decay tensor that requires grad.
        params = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand_like(params, requires_grad=True, dtype=torch.float64)
        weight_decay = torch.tensor(0.999, requires_grad=True, dtype=torch.float64)

        state = {}
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["max_exp_avg_sq"] = torch.rand(
            10, requires_grad=True, dtype=torch.float64
        )
        kwargs: dict[str, Any] = {"weight_decay": weight_decay, "differentiable": True}

        gradcheck(
            _multistep_backprop_diff_hyperparams_fn,
            (
                params,
                grad,
                state,
                AdamW,
                kwargs,  # includes weight_decay
                *state.values(),
                *kwargs.values(),
            ),
        )

    def test_adamw_differentiable_betas(self):
        # [FUNCTION DOC] Gradchecks a two-step differentiable AdamW run with respect to both
        # beta tensors, passed as separate entries.
        params = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand_like(params, requires_grad=True, dtype=torch.float64)

        betas = (
            torch.tensor(0.9, requires_grad=True, dtype=torch.float64),
            torch.tensor(0.999, requires_grad=True, dtype=torch.float64),
        )
        state = {}
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["max_exp_avg_sq"] = torch.rand(
            10, requires_grad=True, dtype=torch.float64
        )

        # Have to pass in beta1 and beta2 separately
        # so they're passed in as Tensors (not a tuple) and recognized by gradcheck.
        # In the test, this is called: kwargs.update({betas: (beta1, beta2)})
        kwargs: dict[str, Any] = {
            "beta1": betas[0],
            "beta2": betas[1],
            "differentiable": True,
        }

        gradcheck(
            _multistep_backprop_diff_hyperparams_fn,
            (
                params,
                grad,
                state,
                AdamW,
                kwargs,  # includes betas
                *state.values(),
                *kwargs.values(),
            ),
        )

    def test_adamw_differentiable_all_hyperparams(self):
        # [FUNCTION DOC] Gradchecks a two-step differentiable AdamW run with respect to learning
        # rate, weight decay and both betas together.
        params = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand_like(params, requires_grad=True, dtype=torch.float64)

        lr = torch.tensor(0.001, requires_grad=True, dtype=torch.float64)
        weight_decay = torch.tensor(0.999, requires_grad=True, dtype=torch.float64)
        betas = (
            torch.tensor(0.9, requires_grad=True, dtype=torch.float64),
            torch.tensor(0.999, requires_grad=True, dtype=torch.float64),
        )
        state = {}
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["max_exp_avg_sq"] = torch.rand(
            10, requires_grad=True, dtype=torch.float64
        )

        # Have to pass in beta1 and beta2 separately
        # so they're passed in as Tensors (not a tuple) and recognized by gradcheck.
        # In the test, this is called: kwargs.update({betas: (beta1, beta2)})
        kwargs: dict[str, Any] = {
            "lr": lr,
            "weight_decay": weight_decay,
            "beta1": betas[0],
            "beta2": betas[1],
            "differentiable": True,
        }

        gradcheck(
            _multistep_backprop_diff_hyperparams_fn,
            (
                params,
                grad,
                state,
                AdamW,
                kwargs,  # includes betas
                *state.values(),
                *kwargs.values(),
            ),
        )

    def test_differentiable_lr(self):
        # [FUNCTION DOC] Gradchecks a two-step differentiable SGD run with a momentum buffer
        # state, with respect to a learning rate tensor.
        params = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand_like(params, requires_grad=True, dtype=torch.float64)
        lr = torch.tensor(0.001, requires_grad=True, dtype=torch.float64)

        mbuff = torch.rand_like(params, requires_grad=True, dtype=torch.float64)
        state = {"momentum_buffer": mbuff}
        kwargs: dict[str, Any] = {"lr": lr, "differentiable": True}

        gradcheck(
            _multistep_backprop_diff_hyperparams_fn,
            (
                params,
                grad,
                state,
                SGD,
                kwargs,  # includes lr
                *state.values(),
                *kwargs.values(),
            ),
        )

    def test_differentiable_weight_decay(self):
        # [FUNCTION DOC] Gradchecks a two-step differentiable SGD run with a momentum buffer
        # state, with respect to a weight decay tensor.
        params = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand_like(params, requires_grad=True, dtype=torch.float64)
        weight_decay = torch.tensor(0.9, requires_grad=True, dtype=torch.float64)

        mbuff = torch.rand_like(params, requires_grad=True, dtype=torch.float64)
        state = {"momentum_buffer": mbuff}
        kwargs: dict[str, Any] = {"weight_decay": weight_decay, "differentiable": True}

        gradcheck(
            _multistep_backprop_diff_hyperparams_fn,
            (
                params,
                grad,
                state,
                SGD,
                kwargs,  # includes weight_decay
                *state.values(),
                *kwargs.values(),
            ),
        )

    def test_differentiable_weight_decay_and_lr(self):
        # [FUNCTION DOC] Gradchecks a two-step differentiable SGD run with a momentum buffer
        # state, with respect to both weight decay and learning rate tensors.
        params = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand_like(params, requires_grad=True, dtype=torch.float64)

        weight_decay = torch.tensor(0.9, requires_grad=True, dtype=torch.float64)
        lr = torch.tensor(0.001, requires_grad=True, dtype=torch.float64)

        mbuff = torch.rand_like(params, requires_grad=True, dtype=torch.float64)
        state = {"momentum_buffer": mbuff}

        kwargs: dict[str, Any] = {
            "weight_decay": weight_decay,
            "lr": lr,
            "differentiable": True,
        }

        gradcheck(
            _multistep_backprop_diff_hyperparams_fn,
            (
                params,
                grad,
                state,
                SGD,
                kwargs,  # includes lr & weight_decay
                *state.values(),
                *kwargs.values(),
            ),
        )


if __name__ == "__main__":
    print("These tests should be run through test/test_optim.py instead")
