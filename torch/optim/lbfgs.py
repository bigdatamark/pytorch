# mypy: allow-untyped-defs
# [MODULE DOC]
# L-BFGS optimizer. Contains the LBFGS class (limited-memory BFGS over all parameters
# flattened into one vector, single param group only, optional strong Wolfe line search),
# plus two private numeric helpers: cubic interpolation of a step length and a strong Wolfe
# line search ported from the Lua torch optim library. Import-time side effect: assigns
# LBFGS.__doc__.

import torch
from torch import Tensor

from .optimizer import _maximize_doc, _to_scalar, Optimizer, ParamsT


__all__ = ["LBFGS"]


def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # [FUNCTION DOC] HELPER: Given two points with function values (f) and derivatives (g),
    # returns the minimizer of the cubic through them, clamped to the bounds (default: the
    # interval between the two points). If the discriminant is negative (no real minimizer),
    # returns the midpoint of the bounds. The inputs are expected to be scalar tensors or
    # numbers; no side effects.
    # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.0


def _strong_wolfe(
    obj_func, x, t, d, f, g, gtd, c1=1e-4, c2=0.9, tolerance_change=1e-9, max_ls=25
):
    # [FUNCTION DOC] HELPER: Strong Wolfe line search along direction d from the point x with
    # initial step t, current loss f, gradient g and directional derivative gtd. obj_func
    # evaluates loss and gradient at a trial step. Phase one expands the step until it
    # brackets a point satisfying the sufficient decrease (c1) and curvature (c2) conditions;
    # phase two zooms in by cubic interpolation until the conditions hold, the bracket is
    # smaller than tolerance_change, or max_ls iterations are used. Returns the new loss,
    # new gradient, chosen step and number of function evaluations. No parameter mutation
    # besides what obj_func does.
    # KNOWN ISSUE: if the first loop exhausts max_ls, the bracket directional derivatives are
    # never defined, so the zoom phase relies on the iteration guard to skip using them.
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev, f_prev, gtd_prev, t, f_new, gtd_new, bounds=(min_step, max_step)
        )

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)  # type: ignore[possibly-undefined]
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:  # type: ignore[possibly-undefined]
            break

        # compute new trial value
        t = _cubic_interpolate(
            # pyrefly: ignore [unbound-name]
            bracket[0],
            # pyrefly: ignore [unbound-name]
            bracket_f[0],
            bracket_gtd[0],  # type: ignore[possibly-undefined]
            # pyrefly: ignore [unbound-name]
            bracket[1],
            # pyrefly: ignore [unbound-name]
            bracket_f[1],
            # pyrefly: ignore [unbound-name]
            bracket_gtd[1],
        )

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        # pyrefly: ignore [unbound-name]
        eps = 0.1 * (max(bracket) - min(bracket))
        # pyrefly: ignore [unbound-name]
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            # pyrefly: ignore [unbound-name]
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                # pyrefly: ignore [unbound-name]
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    # pyrefly: ignore [unbound-name]
                    t = max(bracket) - eps
                else:
                    # pyrefly: ignore [unbound-name]
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        # pyrefly: ignore [unbound-name]
        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            # pyrefly: ignore [unbound-name]
            bracket[high_pos] = t
            # pyrefly: ignore [unbound-name]
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)  # type: ignore[possibly-undefined]
            # pyrefly: ignore [unbound-name]
            bracket_gtd[high_pos] = gtd_new
            # pyrefly: ignore [unbound-name]
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            # pyrefly: ignore [unbound-name]
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                # pyrefly: ignore [unbound-name]
                bracket[high_pos] = bracket[low_pos]
                # pyrefly: ignore [unbound-name]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]  # type: ignore[possibly-undefined]
                # pyrefly: ignore [unbound-name]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            # pyrefly: ignore [unbound-name]
            bracket[low_pos] = t
            # pyrefly: ignore [unbound-name]
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)  # type: ignore[possibly-undefined]
            # pyrefly: ignore [unbound-name]
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]  # type: ignore[possibly-undefined]
    # pyrefly: ignore [unbound-name]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]  # type: ignore[possibly-undefined]
    return f_new, g_new, t, ls_func_evals


class LBFGS(Optimizer):
    # [CLASS DOC]
    # L-BFGS optimizer operating on the concatenation of all parameters (complex parameters
    # contribute their real and imaginary parts). Requires exactly one param group. Options:
    # lr, max_iter, max_eval, tolerance_grad, tolerance_change, history_size, line_search_fn
    # (None or the strong Wolfe search), maximize. Holds a cached total element count. Its
    # only state is global and is stored under the first parameter: counters (func_evals,
    # n_iter) and the history needed to resume (direction, step, curvature pairs, scaling,
    # previous gradient and loss). Key methods: step (requires a closure), flat gradient helpers.
    # KNOWN ISSUE: the argument docs say parameters must be real, but the code handles complex.
    r"""Implements L-BFGS algorithm."""

    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 1,
        max_iter: int = 20,
        max_eval: int | None = None,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 100,
        line_search_fn: str | None = None,
        *,
        maximize: bool = False,
    ) -> None:
        # [FUNCTION DOC]
        # Validates lr (single-element if a tensor, non-negative) raising ValueError, defaults
        # max_eval to five quarters of max_iter, builds the defaults mapping and calls the base
        # constructor. Raises ValueError if more than one param group results. Stores the
        # parameter list and initializes the element count cache to empty.
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = {
            "lr": lr,
            "max_iter": max_iter,
            "max_eval": max_eval,
            "tolerance_grad": tolerance_grad,
            "tolerance_change": tolerance_change,
            "history_size": history_size,
            "line_search_fn": line_search_fn,
            "maximize": maximize,
        }
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "LBFGS doesn't support per-parameter options (parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self._numel_cache = None

    def _numel(self):
        # [FUNCTION DOC] HELPER: Returns the total number of scalar elements over all
        # parameters (complex counts double), computing it once and caching it.
        if self._numel_cache is None:
            self._numel_cache = sum(
                2 * p.numel() if torch.is_complex(p) else p.numel()
                for p in self._params
            )

        return self._numel_cache

    def _gather_flat_grad(self):
        # [FUNCTION DOC] HELPER: Concatenates all parameter gradients into one flat vector.
        # Missing gradients become zeros, sparse gradients are densified, complex gradients
        # are viewed as real pairs. Negates the result in place when maximizing. Returns the
        # new flat tensor.
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            if torch.is_complex(view):
                view = torch.view_as_real(view).view(-1)
            views.append(view)
        flat_grad = torch.cat(views, 0)
        if self.param_groups[0].get("maximize", False):
            flat_grad.neg_()
        return flat_grad

    def _add_grad(self, step_size, update) -> None:
        # [FUNCTION DOC] HELPER: Adds step_size times the corresponding slice of the flat
        # update vector to each parameter in place (complex parameters via a real view).
        # Raises AssertionError if the consumed element count differs from the total count.
        offset = 0
        for p in self._params:
            if torch.is_complex(p):
                p = torch.view_as_real(p)
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset : offset + numel].view_as(p), alpha=step_size)
            offset += numel
        if offset != self._numel():
            raise AssertionError(f"Expected offset {offset} to equal {self._numel()}")

    def _clone_param(self):
        # [FUNCTION DOC] HELPER: Returns a list of contiguous copies of all parameters, used
        # to restore the starting point during line search.
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data) -> None:
        # [FUNCTION DOC] HELPER: Copies each tensor of the given list into the matching
        # parameter in place (lists must have equal length, else an error is raised).
        for p, pdata in zip(self._params, params_data, strict=True):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        # [FUNCTION DOC] HELPER: Evaluates the objective at the point reached by moving
        # parameters along direction d by step t. Calls the closure, converts the loss to a
        # float (negated when maximizing), gathers the flat gradient, then restores the
        # parameters to the saved values x. Returns the loss and flat gradient.
        self._add_grad(t, d)
        loss = float(closure())
        if self.param_groups[0].get("maximize", False):
            loss = -loss
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @torch.no_grad()
    def step(self, closure):  # type: ignore[override]
        # [FUNCTION DOC]
        # Runs up to max_iter L-BFGS iterations (no gradient tracking, closure always run with
        # gradients). Raises AssertionError if there is not exactly one group, and
        # RuntimeError for an unsupported line search name. With no parameters it just returns
        # the closure result. Returns early with the first loss if the gradient is already
        # within tolerance. Each iteration builds a descent direction (negative gradient first,
        # then the two-loop recursion over stored curvature pairs, keeping only pairs with
        # positive curvature and at most history_size), picks a step (scaled initially, else
        # lr, or from the strong Wolfe search), moves the parameters, re-evaluates the loss
        # unless on the last iteration, and stops on evaluation budget, tolerance, flat
        # progress or loss change. Persists history in the global state. Returns the loss
        # of the first closure evaluation.
        """Perform a single optimization step.

        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the loss.
        """
        if len(self.param_groups) != 1:
            raise AssertionError(
                f"Expected exactly one param_group, but got {len(self.param_groups)}"
            )

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        # There is nothing to optimize, so skip the step like the other optimizers
        # do, but still evaluate the closure so the caller gets a loss back. The
        # rest of the step cannot run anyway: the global state below is keyed on
        # the first parameter, and the flat gradient would be empty.
        if len(self._params) == 0:
            return closure()

        group = self.param_groups[0]
        lr = _to_scalar(group["lr"])
        max_iter = group["max_iter"]
        max_eval = group["max_eval"]
        tolerance_grad = group["tolerance_grad"]
        tolerance_change = group["tolerance_change"]
        line_search_fn = group["line_search_fn"]
        history_size = group["history_size"]
        maximize = group.get("maximize", False)

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault("func_evals", 0)
        state.setdefault("n_iter", 0)

        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        loss = -float(orig_loss) if maximize else float(orig_loss)
        current_evals = 1
        state["func_evals"] += 1

        flat_grad = self._gather_flat_grad()
        opt_cond = flat_grad.abs().max() <= tolerance_grad

        # optimal condition
        if opt_cond:
            return orig_loss

        # tensors cached in state (for tracing)
        d = state.get("d")
        t = state.get("t")
        old_dirs = state.get("old_dirs")
        old_stps = state.get("old_stps")
        ro = state.get("ro")
        H_diag = state.get("H_diag")
        prev_flat_grad = state.get("prev_flat_grad")
        prev_loss = state.get("prev_loss")

        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            n_iter += 1
            state["n_iter"] += 1

            ############################################################
            # compute gradient descent direction
            ############################################################
            if state["n_iter"] == 1:
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1
            else:
                # do lbfgs update (update memory)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)  # y*s
                if ys > 1e-10:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)

                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1.0 / ys)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)

                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                num_old = len(old_dirs)

                if "al" not in state:
                    state["al"] = [None] * history_size
                al = state["al"]

                # iteration in L-BFGS loop collapsed to use just one buffer
                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_stps[i].dot(q) * ro[i]
                    q.add_(old_dirs[i], alpha=-al[i])

                # multiply by initial Hessian
                # r/d is the final direction
                d = r = torch.mul(q, H_diag)
                for i in range(num_old):
                    be_i = old_dirs[i].dot(r) * ro[i]
                    r.add_(old_stps[i], alpha=al[i] - be_i)

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
            if state["n_iter"] == 1:
                t = min(1.0, 1.0 / flat_grad.abs().sum()) * lr
            else:
                t = lr

            # directional derivative
            gtd = flat_grad.dot(d)  # g * d

            # directional derivative is below tolerance
            if gtd > -tolerance_change:
                break

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                # perform line search, using user function
                if line_search_fn != "strong_wolfe":
                    raise RuntimeError("only 'strong_wolfe' is supported")
                else:
                    x_init = self._clone_param()

                    def obj_func(x, t, d):
                        # [FUNCTION DOC] HELPER: Callback for the line search; evaluates loss
                        # and flat gradient at step t along d from the saved point x via the
                        # directional evaluate method, then restores the parameters.
                        return self._directional_evaluate(closure, x, t, d)

                    loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                        obj_func,
                        x_init,
                        t,
                        d,
                        loss,
                        flat_grad,
                        gtd,
                        max_ls=max_eval - current_evals,
                    )
                self._add_grad(t, d)
                opt_cond = flat_grad.abs().max() <= tolerance_grad
            else:
                # no line search, simply move with fixed-step
                self._add_grad(t, d)
                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    with torch.enable_grad():
                        loss = closure()
                    loss = -float(loss) if maximize else float(loss)
                    flat_grad = self._gather_flat_grad()
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state["func_evals"] += ls_func_evals

            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                break

            if current_evals >= max_eval:
                break

            # optimal condition
            if opt_cond:
                break

            # lack of progress
            if d.mul(t).abs().max() <= tolerance_change:
                break

            if abs(loss - prev_loss) < tolerance_change:
                break

        state["d"] = d
        state["t"] = t
        state["old_dirs"] = old_dirs
        state["old_stps"] = old_stps
        state["ro"] = ro
        state["H_diag"] = H_diag
        state["prev_flat_grad"] = prev_flat_grad
        state["prev_loss"] = prev_loss

        return orig_loss


LBFGS.__doc__ = (
    r"""Implements L-BFGS algorithm.

    Heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`_.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.
    """
    + rf"""
    Args:
        params (iterable): iterable of parameters to optimize. Parameters must be real.
        lr (float, optional): learning rate (default: 1)
        max_iter (int, optional): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int, optional): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float, optional): termination tolerance on first order optimality
            (default: 1e-7).
        tolerance_change (float, optional): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int, optional): update history size (default: 100).
        line_search_fn (str, optional): either 'strong_wolfe' or None (default: None).
        {_maximize_doc}
    """
)
