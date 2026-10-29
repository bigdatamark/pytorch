# mypy: allow-untyped-defs
# [MODULE DOC] NAdam optimizer (Adam with Nesterov momentum): the NAdam class, the
# functional entry point nadam, and single-tensor and multi-tensor (foreach) implementations
# of the update.
# Import-time side effects: after the class is defined its __doc__ is assigned a long string
# containing the algorithm in math form and shared argument documentation. Exports NAdam and
# nadam.
r"""Implementation for the NAdam algorithm."""

from typing import cast

import torch
from torch import Tensor

from .optimizer import (
    _capturable_doc,
    _default_to_fused_or_foreach,
    _differentiable_doc,
    _disable_dynamo_if_unsupported,
    _foreach_doc,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value,
    _maximize_doc,
    _params_doc,
    _stack_if_compiling,
    _to_scalar,
    _use_grad_for_differentiable,
    _view_as_real,
    Optimizer,
    ParamsT,
)


__all__ = ["NAdam", "nadam"]


class NAdam(Optimizer):
    # [CLASS DOC] Optimizer implementing NAdam. Per-parameter state: step and mu_product
    # (scalar tensors, on the parameter device only when capturable, else CPU), exp_avg and
    # exp_avg_sq (same shape as the parameter). Group options: lr, betas, eps, weight_decay,
    # momentum_decay, decoupled_weight_decay, maximize, foreach, capturable, differentiable.
    # Key methods: step gathers tensors per group and calls the functional nadam; the class
    # docstring is set after the class body.
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 2e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum_decay: float = 4e-3,
        decoupled_weight_decay: bool = False,
        *,
        foreach: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
    ) -> None:
        # [FUNCTION DOC] Validate hyperparameters and register the param groups. Raises
        # ValueError for a multi-element tensor lr, or a negative lr, eps, weight_decay or
        # momentum_decay, or a beta outside the range zero to one (upper bound exclusive).
        # Builds the defaults mapping and calls the base optimizer constructor. Args:
        # params, lr, betas, eps, weight_decay, momentum_decay, decoupled_weight_decay, and
        # keyword-only foreach, maximize, capturable, differentiable.
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= momentum_decay:
            raise ValueError(f"Invalid momentum_decay value: {momentum_decay}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "momentum_decay": momentum_decay,
            "decoupled_weight_decay": decoupled_weight_decay,
            "maximize": maximize,
            "foreach": foreach,
            "capturable": capturable,
            "differentiable": differentiable,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        # [FUNCTION DOC] Restore after unpickling or loading state. Fills in defaults for
        # options missing from older checkpoints in every group, and converts any non-tensor
        # step and mu_product entries in the per-parameter state to scalar tensors (on the
        # parameter device if capturable, else CPU). Mutates self.
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("decoupled_weight_decay", False)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0:
                    if not torch.is_tensor(p_state["step"]):
                        step_val = float(p_state["step"])
                        p_state["step"] = (
                            torch.tensor(
                                step_val, dtype=_get_scalar_dtype(), device=p.device
                            )
                            if group["capturable"]
                            else torch.tensor(step_val, dtype=_get_scalar_dtype())
                        )
                    if not torch.is_tensor(p_state["mu_product"]):
                        mu_prod_val = p_state["mu_product"]
                        p_state["mu_product"] = (
                            torch.tensor(
                                mu_prod_val, dtype=_get_scalar_dtype(), device=p.device
                            )
                            if group["capturable"]
                            else torch.tensor(mu_prod_val, dtype=_get_scalar_dtype())
                        )

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        mu_products,
        state_steps,
    ):
        # [FUNCTION DOC] HELPER: Collect the tensors of one param group that have gradients
        # into the given output lists (params, grads, exp_avgs, exp_avg_sqs, mu_products,
        # state_steps), lazily creating state on first use. Raises RuntimeError for sparse
        # gradients. Returns whether any parameter with a gradient is complex. Mutates the
        # lists and the optimizer state.
        has_complex = False
        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError("NAdam does not support sparse gradients")
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` and `mu_product` on CPU if capturable is False.
                    # This is because kernel launches are costly on CUDA and XLA.
                    state["step"] = (
                        torch.zeros((), dtype=_get_scalar_dtype(), device=p.device)
                        if group["capturable"]
                        else torch.tensor(0.0, dtype=_get_scalar_dtype())
                    )
                    state["mu_product"] = (
                        torch.ones((), dtype=_get_scalar_dtype(), device=p.device)
                        if group["capturable"]
                        else torch.tensor(1.0, dtype=_get_scalar_dtype())
                    )
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                mu_products.append(state["mu_product"])
                state_steps.append(state["step"])
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        # [FUNCTION DOC] Perform one optimization step. Args: closure (optional callable re-
        # evaluating the model; run with gradients enabled). Runs a graph-capture health
        # check, then for each group gathers tensors and invokes the functional nadam with
        # the group's options. Updates parameters and state in place. Returns the closure's
        # loss or None.
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._accelerator_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            mu_products: list[Tensor] = []
            state_steps: list[Tensor] = []
            beta1, beta2 = cast(tuple[float, float], group["betas"])

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                mu_products,
                state_steps,
            )

            nadam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                mu_products,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                momentum_decay=group["momentum_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                decoupled_weight_decay=group["decoupled_weight_decay"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                has_complex=has_complex,
            )

        return loss


NAdam.__doc__ = (
    r"""Implements NAdam algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma_t \text{ (lr)}, \: \beta_1,\beta_2 \text{ (betas)},
                \: \theta_0 \text{ (params)}, \: f(\theta) \text{ (objective)}                   \\
            &\hspace{13mm} \: \lambda \text{ (weight decay)}, \:\psi \text{ (momentum decay)}    \\
            &\hspace{13mm} \: \textit{decoupled\_weight\_decay}, \:\textit{maximize}             \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0 \leftarrow 0 \text{ ( second moment)}                                 \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm} \theta_t \leftarrow \theta_{t-1}                                       \\
            &\hspace{5mm} \textbf{if} \: \lambda \neq 0                                          \\
            &\hspace{10mm}\textbf{if} \: \textit{decoupled\_weight\_decay}                       \\
            &\hspace{15mm} \theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}                    \\
            &\hspace{10mm}\textbf{else}                                                          \\
            &\hspace{15mm} g_t \leftarrow g_t + \lambda \theta_{t-1}                             \\
            &\hspace{5mm} \mu_t \leftarrow \beta_1 \big(1 - \frac{1}{2}  0.96^{t \psi} \big)     \\
            &\hspace{5mm} \mu_{t+1} \leftarrow \beta_1 \big(1 - \frac{1}{2} 0.96^{(t+1)\psi}\big)\\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow \mu_{t+1} m_t/(1-\prod_{i=1}^{t+1}\mu_i)\\[-1.ex]
            & \hspace{11mm} + (1-\mu_t) g_t /(1-\prod_{i=1}^{t} \mu_{i})                         \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Incorporating Nesterov Momentum into Adam`_.
    """
    + rf"""
    Args:
        {_params_doc}
        lr (float, Tensor, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        momentum_decay (float, optional): momentum decay (default: 4e-3)
        decoupled_weight_decay (bool, optional): whether to decouple the weight
            decay as in AdamW to obtain NAdamW. If True, the algorithm does not
            accumulate weight decay in the momentum nor variance. (default: False)
        {_foreach_doc}
        {_maximize_doc}
        {_capturable_doc}
        {_differentiable_doc}

    .. _Incorporating Nesterov Momentum into Adam:
        https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101

    """
)


def _single_tensor_nadam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    mu_products: list[Tensor],
    state_steps: list[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: float | Tensor,
    weight_decay: float,
    momentum_decay: float,
    eps: float,
    decoupled_weight_decay: bool,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    has_complex: bool,
) -> None:
    # [FUNCTION DOC] HELPER: Per-tensor NAdam update. For each parameter: negates the
    # gradient if maximize, views complex tensors as real, increments the step tensor,
    # applies weight decay (decoupled: scales the parameter; otherwise adds to the
    # gradient), computes the momentum cache values for this and the next step, multiplies
    # mu_product, updates the first and second moment averages and applies the two Nesterov-
    # style updates to the parameter. Uses tensor arithmetic when capturable or
    # differentiable so autograd or graph capture can follow, otherwise scalar values.
    # Raises AssertionError if capturable is set but tensors are not on a supported device.
    # Mutates params, exp_avgs, exp_avg_sqs, mu_products and state_steps in place. Returns
    # nothing.
    # KNOWN ISSUE: the has_complex argument is unused here; complexness is checked per
    #     parameter.
    if not torch.jit.is_scripting():
        lr = _to_scalar(lr)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        mu_product = mu_products[i]
        step_t = state_steps[i]

        if torch.is_complex(param):
            param = torch.view_as_real(param)
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch.compiler.is_compiling() and capturable:
            capturable_supported_devices = _get_capturable_supported_devices()
            if not (
                param.device.type == mu_product.device.type == step_t.device.type
                and param.device.type in capturable_supported_devices
            ):
                raise AssertionError(
                    f"If capturable=True, params, mu_products and state_steps must be "
                    f"on supported devices: {capturable_supported_devices}."
                )

        # update step
        step_t += 1

        if capturable:
            step = step_t
        else:
            step = _get_value(step_t)

        bias_correction2 = 1 - beta2**step

        if weight_decay != 0:
            if decoupled_weight_decay:
                # Perform step weight decay
                param.mul_(1 - lr * weight_decay)
            else:
                grad = grad.add(param, alpha=weight_decay)

        # calculate the momentum cache \mu^{t} and \mu^{t+1}
        mu = beta1 * (1.0 - 0.5 * (0.96 ** (step * momentum_decay)))
        mu_next = beta1 * (1.0 - 0.5 * (0.96 ** ((step + 1) * momentum_decay)))

        # update mu_product
        mu_product *= mu

        # decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        denom = exp_avg_sq.div(bias_correction2).sqrt()

        if differentiable or capturable:
            denom = denom.add(eps)
            # Make autograd track the operations
            # by updating the grad and exp_avg directly and not using the
            # scalar "value" argument of addcdiv.
            mu_product_next = mu_product * mu_next
            grad = grad * (-lr * (1.0 - mu) / (1.0 - mu_product))
            exp_avg = exp_avg * (-lr * mu_next / (1.0 - mu_product_next))
            param.addcdiv_(grad, denom)
            param.addcdiv_(exp_avg, denom)
        else:
            mu_product_next = _get_value(mu_product) * mu_next
            denom.add_(eps)
            param.addcdiv_(
                grad,
                denom,
                value=(-lr * (1.0 - mu) / (1.0 - _get_value(mu_product))),  # type: ignore[arg-type]
            )
            param.addcdiv_(
                exp_avg,
                denom,
                value=cast(float, (-lr * mu_next) / (1.0 - mu_product_next)),
            )


def _multi_tensor_nadam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    mu_products: list[Tensor],
    state_steps: list[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: float | Tensor,
    weight_decay: float,
    momentum_decay: float,
    eps: float,
    decoupled_weight_decay: bool,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    has_complex: bool,
) -> None:
    # [FUNCTION DOC] HELPER: Batched (foreach) NAdam update with the same math as the
    # single-tensor version. Returns immediately for an empty list. Raises AssertionError if
    # differentiable is set, or if capturable is set and tensors are on unsupported devices.
    # Groups tensors by device and dtype, optionally views complex tensors as real, updates
    # steps, applies weight decay, updates moment averages, computes the bias corrections,
    # mu values and step sizes (as tensors when capturable, scalars otherwise) and applies
    # the two addcdiv-style parameter updates. Mutates inputs in place; returns nothing.
    if len(params) == 0:
        return

    if differentiable:
        raise AssertionError("_foreach ops don't support autograd")

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch.compiler.is_compiling() and capturable:
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )
        if not all(
            p.device.type == mp.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, mp, step in zip(params, mu_products, state_steps, strict=True)
        ):
            raise AssertionError(
                "If capturable=True, "
                "params, mu_products, and state_steps must be on supported devices: "
                f"{capturable_supported_devices}."
            )

    lr = _to_scalar(lr)

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, mu_products, state_steps]  # type: ignore[list-item]
    )
    for (
        grouped_params_,
        grouped_grads_,
        grouped_exp_avgs_,
        grouped_exp_avg_sqs_,
        grouped_mu_products_,
        grouped_state_steps_,
    ), _ in grouped_tensors.values():
        grouped_params = cast(list[Tensor], grouped_params_)
        grouped_grads = cast(list[Tensor], grouped_grads_)
        grouped_exp_avgs = cast(list[Tensor], grouped_exp_avgs_)
        grouped_exp_avg_sqs = cast(list[Tensor], grouped_exp_avg_sqs_)
        grouped_mu_products = cast(list[Tensor], grouped_mu_products_)
        grouped_state_steps = cast(list[Tensor], grouped_state_steps_)

        # handle complex
        if has_complex:
            _view_as_real(
                grouped_params, grouped_grads, grouped_exp_avgs, grouped_exp_avg_sqs
            )

        if maximize:
            grouped_grads = torch._foreach_neg(grouped_grads)  # type: ignore[assignment]

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if not torch.compiler.is_compiling() and grouped_state_steps[0].is_cpu:
            torch._foreach_add_(
                grouped_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
            )
        else:
            torch._foreach_add_(grouped_state_steps, 1)

        if weight_decay != 0:
            if decoupled_weight_decay:
                # Perform step weight decay
                torch._foreach_mul_(grouped_params, 1 - lr * weight_decay)
            else:
                # Reuse the intermediate memory (grouped_grads) already allocated for maximize
                if maximize:
                    torch._foreach_add_(
                        grouped_grads, grouped_params, alpha=weight_decay
                    )
                else:
                    grouped_grads = torch._foreach_add(  # type: ignore[assignment]
                        grouped_grads, grouped_params, alpha=weight_decay
                    )

        # Decay the first and second moment running average coefficient
        torch._foreach_lerp_(grouped_exp_avgs, grouped_grads, 1 - beta1)

        torch._foreach_mul_(grouped_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(
            grouped_exp_avg_sqs, grouped_grads, grouped_grads, 1 - beta2
        )

        exp_avg_sq_sqrt = torch._foreach_sqrt(grouped_exp_avg_sqs)

        bias_correction_sqrt: tuple[Tensor, ...] | list[Tensor]
        mus: tuple[Tensor, ...] | list[Tensor]
        mu_nexts: tuple[Tensor, ...] | list[Tensor]
        if capturable:
            # mus will be beta1 * (1 - 0.5 * 0.96 ** (step * momentum_decay))
            exponent = torch._foreach_mul(grouped_state_steps, momentum_decay)
            mus = torch._foreach_pow(0.96, exponent)
            torch._foreach_mul_(mus, -0.5)
            torch._foreach_add_(mus, 1.0)
            torch._foreach_mul_(mus, beta1)

            # mu_nexts will be beta1 * (1 - 0.5 * 0.96 ** ((step + 1) * momentum_decay))
            torch._foreach_add_(exponent, momentum_decay)
            mu_nexts = torch._foreach_pow(0.96, exponent)
            torch._foreach_mul_(mu_nexts, -0.5)
            torch._foreach_add_(mu_nexts, 1.0)
            torch._foreach_mul_(mu_nexts, beta1)

            # save peak memory as we don't need exponent anymore
            del exponent

            bias_correction_sqrt = torch._foreach_pow(beta2, grouped_state_steps)
            # foreach_sub doesn't allow a scalar as the first arg
            torch._foreach_sub_(bias_correction_sqrt, 1.0)
            torch._foreach_neg_(bias_correction_sqrt)
            torch._foreach_sqrt_(bias_correction_sqrt)
        else:
            bias_correction_sqrt = [
                (1 - beta2 ** _get_value(step)) ** 0.5 for step in grouped_state_steps
            ]
            mus = [
                beta1 * (1.0 - 0.5 * (0.96 ** (_get_value(step) * momentum_decay)))
                for step in grouped_state_steps
            ]
            mu_nexts = [
                beta1
                * (1.0 - 0.5 * (0.96 ** ((_get_value(step) + 1) * momentum_decay)))
                for step in grouped_state_steps
            ]

        # update mu_products
        torch._foreach_mul_(grouped_mu_products, mus)

        torch._foreach_div_(exp_avg_sq_sqrt, bias_correction_sqrt)
        torch._foreach_add_(exp_avg_sq_sqrt, eps)

        # explicitly delete bias_correction refs to save memory
        del bias_correction_sqrt

        if capturable:
            # Build up the step_size multiplier for grad, reusing mus' memory
            torch._foreach_sub_(mus, 1.0)
            torch._foreach_mul_(mus, lr)
            # foreach_sub doesn't allow a scalar as the first arg
            denom = torch._foreach_sub(grouped_mu_products, 1.0)
            torch._foreach_neg_(denom)
            torch._foreach_div_(mus, denom)
            # - lr * (1 - mu) / (1 - mu_product)
            step_size_grads = mus
            # explicitly delete denom to save memory
            del denom

            # Build up the step_size multiplier for exp_avg, reusing mu_nexts' memory
            denom = torch._foreach_mul(grouped_mu_products, mu_nexts)
            torch._foreach_mul_(mu_nexts, lr)
            # foreach_sub doesn't allow a scalar as the first arg, but it's okay because
            # we need a negative here anyway
            torch._foreach_sub_(denom, 1.0)
            torch._foreach_div_(mu_nexts, denom)
            # - lr * mu_next / (1 - mu_product * mu_next)
            step_size_expavg = mu_nexts
            # explicitly delete denom to save memory
            del denom

            # we cannot inplace into step_size_grads cuz it is a list of ScalarTensors
            # and mul'ing with grouped_grads will result in a list of bigger Tensors
            numerator = torch._foreach_mul(step_size_grads, grouped_grads)
            torch._foreach_addcmul_(numerator, step_size_expavg, grouped_exp_avgs)

            # finally, update params
            torch._foreach_addcdiv_(grouped_params, numerator, exp_avg_sq_sqrt)
        else:
            step_size_grads = _stack_if_compiling(
                [
                    (_get_value(lr) * (1.0 - mu) / (1.0 - _get_value(mu_product))) * -1
                    for mu_product, mu in zip(grouped_mu_products, mus, strict=True)
                ]
            )
            step_size_expavg = _stack_if_compiling(
                [
                    (
                        _get_value(lr)
                        * mu_next
                        / (1.0 - _get_value(mu_product) * mu_next)
                    )
                    * -1
                    for mu_product, mu_next in zip(
                        grouped_mu_products, mu_nexts, strict=True
                    )
                ]
            )

            torch._foreach_addcdiv_(  # type: ignore[arg-type]
                grouped_params,
                grouped_grads,
                exp_avg_sq_sqrt,
                step_size_grads,
            )
            torch._foreach_addcdiv_(  # type: ignore[arg-type]
                grouped_params,
                grouped_exp_avgs,
                exp_avg_sq_sqrt,
                step_size_expavg,
            )


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_nadam)
def nadam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    mu_products: list[Tensor],
    state_steps: list[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    decoupled_weight_decay: bool = False,
    foreach: bool | None = None,
    capturable: bool = False,
    differentiable: bool = False,
    has_complex: bool = False,
    maximize: bool = False,
    *,
    beta1: float,
    beta2: float,
    lr: float | Tensor,
    weight_decay: float,
    momentum_decay: float,
    eps: float,
) -> None:
    # [FUNCTION DOC] Functional NAdam entry point. Raises RuntimeError if state_steps or
    # mu_products contain non-tensors, or if foreach is requested under scripting. If
    # foreach is None it is chosen automatically. Dispatches to the multi-tensor or the
    # single-tensor implementation (scripting forces single-tensor) with the given options
    # and updates the tensors in place. Returns nothing.
    # Note the unusual argument order: several options with defaults precede the keyword-
    # only beta1, beta2, lr, weight_decay, momentum_decay and eps.
    r"""Functional API that performs NAdam algorithm computation.

    See :class:`~torch.optim.NAdam` for details.
    """
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if not all(isinstance(t, torch.Tensor) for t in mu_products):
        raise RuntimeError(
            "API has changed, `mu_products` argument must contain a list of singleton tensors"
        )

    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_nadam
    else:
        func = _single_tensor_nadam

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        mu_products,
        state_steps,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        momentum_decay=momentum_decay,
        maximize=maximize,
        decoupled_weight_decay=decoupled_weight_decay,
        eps=eps,
        capturable=capturable,
        differentiable=differentiable,
        has_complex=has_complex,
    )
