# mypy: allow-untyped-defs
# [MODULE DOC] Defines PostLocalSGDOptimizer, an optimizer wrapper that applies a local optimizer then runs a model averager (post-local SGD).
# No import-time side effects beyond imports.
import warnings

import torch
import torch.distributed.algorithms.model_averaging.averagers as averagers


class PostLocalSGDOptimizer(torch.optim.Optimizer):
    # [CLASS DOC] Wraps a torch optimizer plus a ModelAverager. Delegates state, param_groups, repr, zero_grad and add_param_group to the wrapped optimizer;
    # step runs the local step then averages parameters across workers via the averager. The averager's step counter is saved in and restored from state dicts.
    # Does not call the base optimizer constructor; it only mirrors the wrapped optimizer's param_groups.
    r"""
    Wraps an arbitrary :class:`torch.optim.Optimizer` and runs `post-local SGD <https://arxiv.org/abs/1808.07217>`_,
    This optimizer runs local optimizer at every step.
    After the warm-up stage, it averages parameters periodically after the local optimizer is applied.

    Args:
        optim: The local optimizer.
        averager: A model averager instance to run post-localSGD algorithm.

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> import torch
        >>> import torch.distributed as dist
        >>> import torch.distributed.algorithms.model_averaging.averagers as averagers
        >>> import torch.nn as nn
        >>> from torch.distributed.optim import PostLocalSGDOptimizer
        >>> from torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook import (
        >>>   PostLocalSGDState,
        >>>   post_localSGD_hook,
        >>> )
        >>>
        >>> model = nn.parallel.DistributedDataParallel(
        >>>    module, device_ids=[rank], output_device=rank
        >>> )
        >>>
        >>> # Register a post-localSGD communication hook.
        >>> state = PostLocalSGDState(process_group=None, subgroup=None, start_localSGD_iter=100)
        >>> model.register_comm_hook(state, post_localSGD_hook)
        >>>
        >>> # Create a post-localSGD optimizer that wraps a local optimizer.
        >>> # Note that ``warmup_steps`` used in ``PostLocalSGDOptimizer`` must be the same as
        >>> # ``start_localSGD_iter`` used in ``PostLocalSGDState``.
        >>> local_optim = torch.optim.SGD(params=model.parameters(), lr=0.01)
        >>> opt = PostLocalSGDOptimizer(
        >>>     optim=local_optim,
        >>>     averager=averagers.PeriodicModelAverager(period=4, warmup_steps=100)
        >>> )
        >>>
        >>> # In the first 100 steps, DDP runs global gradient averaging at every step.
        >>> # After 100 steps, DDP runs gradient averaging within each subgroup (intra-node by default),
        >>> # and post-localSGD optimizer runs global model averaging every 4 steps after applying the local optimizer.
        >>> for step in range(0, 200):
        >>>    opt.zero_grad()
        >>>    loss = loss_fn(output, labels)
        >>>    loss.backward()
        >>>    opt.step()
    """

    def __init__(self, optim: torch.optim.Optimizer, averager: averagers.ModelAverager):
        # [FUNCTION DOC] Stores the wrapped optimizer and averager and shares the optimizer's param_groups list.
        self.optim = optim
        self.param_groups = self.optim.param_groups
        self.averager = averager

    @property
    def state(self):  # type: ignore[override]
        # [FUNCTION DOC] Property returning the wrapped optimizer's state mapping.
        return self.optim.state

    def __repr__(self):
        # [FUNCTION DOC] Returns the repr of the wrapped optimizer.
        return self.optim.__repr__()

    def state_dict(self):
        # [FUNCTION DOC] Returns the wrapped optimizer's state dict with an added step entry holding the averager's current step.
        r"""
        This is the same as :class:`torch.optim.Optimizer` :meth:`state_dict`,
        but adds an extra entry to record model averager's step to the checkpoint
        to ensure reload does not cause unnecessary warm up again.
        """
        optim_state_dict = self.optim.state_dict()
        optim_state_dict["step"] = self.averager.step
        return optim_state_dict

    def load_state_dict(self, state_dict):
        # [FUNCTION DOC] Loads the state dict into the wrapped optimizer, then sets the averager step from the step entry.
        # If the entry is missing, warns and resets the averager step to 0.
        r"""
        This is the same as :class:`torch.optim.Optimizer` :meth:`load_state_dict`,
        but also restores model averager's step value to the one
        saved in the provided ``state_dict``.

        If there is no ``"step"`` entry in ``state_dict``,
        it will raise a warning and initialize the model averager's step to 0.
        """
        self.optim.load_state_dict(state_dict)
        if "step" in state_dict:
            self.averager.step = state_dict["step"]
        else:
            warnings.warn(
                "Loaded state dict does not contain a step counter for an averager. "
                "Setting step counter to 0.",
                stacklevel=2,
            )
            self.averager.step = 0

    def step(self):  # type: ignore[override]
        # [FUNCTION DOC] Runs the wrapped optimizer step, then asks the averager to average parameters over the param_groups (a collective operation). Returns None.
        r"""
        Performs a single optimization step (parameter update).
        """
        self.optim.step()
        self.averager.average_parameters(params=self.param_groups)

    def zero_grad(self, set_to_none: bool = True):  # type: ignore[override]
        # [FUNCTION DOC] Delegates zero_grad to the wrapped optimizer, passing set_to_none (default true).
        self.optim.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group):
        # [FUNCTION DOC] Delegates adding a param group to the wrapped optimizer.
        self.optim.add_param_group(param_group)
