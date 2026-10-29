# mypy: allow-untyped-defs
# [MODULE DOC] HELPER: Defines _FunctionalRMSprop, a functional RMSprop optimizer for the distributed optimizer; no exports (empty __all__), no import-time side effects.

import torch
import torch.optim._functional as F
from torch import Tensor
from torch.distributed.optim._deprecation_warning import (
    _scripted_functional_optimizer_deprecation_warning,
)


__all__: list[str] = []


# Define a Functional RMSprop Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly allow the distributed optimizer pass gradients to
# the `step` function. In this way, we could separate the gradients
# and parameters and allow multithreaded trainer to update the
# parameters without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.
class _FunctionalRMSprop:
    # [CLASS DOC] HELPER: Functional (stateless-API) RMSprop optimizer used only by the distributed optimizer internals.
    # Holds a single param_group (a mapping with the parameter list), a defaults mapping of hyperparameters,
    # and a per-parameter state mapping created lazily on first step. Gradients are passed explicitly to step
    # instead of being read from the parameter grad attribute, so multithreaded trainers avoid racing on it.
    # Supports centered, momentum, foreach and maximize options.
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
        foreach: bool = False,
        maximize: bool = False,
        _allow_empty_param_list: bool = False,
    ):
        # [FUNCTION DOC] HELPER: Builds the functional RMSprop optimizer from a parameter list and hyperparameters.
        # Emits the scripted-functional-optimizer deprecation warning. No hyperparameter range validation is performed.
        # Raises ValueError if the parameter list is empty (unless _allow_empty_param_list is true).
        # Stores defaults, a single param_group, and an empty state mapping.
        _scripted_functional_optimizer_deprecation_warning(stacklevel=2)
        self.defaults = {
            "lr": lr,
            "alpha": alpha,
            "eps": eps,
            "weight_decay": weight_decay,
            "momentum": momentum,
        }
        self.centered = centered
        self.foreach = foreach
        self.maximize = maximize

        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # NOTE: we only have one param_group and don't allow user to add additional
        # param group as it's not a common use case.
        self.param_group = {"params": params}

        self.state = torch.jit.annotate(dict[torch.Tensor, dict[str, torch.Tensor]], {})

    def step(self, gradients: list[Tensor | None]):
        # [FUNCTION DOC] HELPER: Performs one RMSprop update given a gradient list aligned with the parameters.
        # Raises ValueError if the gradient count differs from the parameter count. Parameters with a None gradient
        # are skipped. State (step counter, square_avg, plus momentum_buffer when momentum is positive and grad_avg when centered) is lazily created per parameter,
        # then the shared functional RMSprop routine is called under no-grad, mutating parameters and state in place. Returns None.
        params = self.param_group["params"]
        params_with_grad = []
        grads = []
        square_avgs = []
        grad_avgs = []
        momentum_buffer_list = []
        state_steps = []
        lr = self.defaults["lr"]
        alpha = self.defaults["alpha"]
        eps = self.defaults["eps"]
        momentum = self.defaults["momentum"]
        weight_decay = self.defaults["weight_decay"]

        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        has_complex = False
        for param, gradient in zip(params, gradients):
            if gradient is not None:
                has_complex |= torch.is_complex(param)
                params_with_grad.append(param)
                grads.append(gradient)
                # Lazy state initialization
                if param not in self.state:
                    self.state[param] = {}
                    state = self.state[param]
                    state["step"] = torch.tensor(0.0)
                    state["square_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    if momentum > 0:
                        state["momentum_buffer"] = torch.zeros_like(
                            param, memory_format=torch.preserve_format
                        )
                    if self.centered:
                        state["grad_avg"] = torch.zeros_like(
                            param, memory_format=torch.preserve_format
                        )

                state = self.state[param]
                square_avgs.append(state["square_avg"])
                if momentum > 0:
                    momentum_buffer_list.append(state["momentum_buffer"])
                if self.centered:
                    grad_avgs.append(state["grad_avg"])

                state_steps.append(state["step"])

        with torch.no_grad():
            F.rmsprop(
                params_with_grad,
                grads,
                square_avgs,
                grad_avgs,
                momentum_buffer_list,
                state_steps,
                lr=lr,
                alpha=alpha,
                eps=eps,
                weight_decay=weight_decay,
                momentum=momentum,
                centered=self.centered,
                foreach=self.foreach,
                maximize=self.maximize,
                has_complex=has_complex,
            )
