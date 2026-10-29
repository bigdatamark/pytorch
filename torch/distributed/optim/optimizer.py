# mypy: allow-untyped-defs
# [MODULE DOC] Implements DistributedOptimizer (the public export), which creates a local optimizer on each worker owning parameters
# and drives steps over RPC using distributed autograd gradients. Also holds private local/functional optimizer wrappers
# and RPC helper functions. Creates a module logger at import time.
import logging
from collections import defaultdict
from threading import Lock

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.jit as jit
from torch import Tensor
from torch.distributed.rpc import RRef

from .utils import functional_optim_map


__all__ = ["DistributedOptimizer"]

logger = logging.getLogger(__name__)


class _FunctionalLocalOptimizer:
    # [CLASS DOC] HELPER: Worker-side wrapper that builds a functional optimizer over local parameters resolved from RRefs
    # and applies it using gradients fetched from a distributed autograd context without touching the grad attribute.
    def __init__(self, optim_cls, local_params_rref, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Resolves each RRef to its local tensor, stores the list, and instantiates the optimizer class on it with the extra arguments.
        self._local_params = [rref.local_value() for rref in local_params_rref]
        self.optim = optim_cls(self._local_params, *args, **kwargs)

    def step(self, autograd_ctx_id: int):
        # [FUNCTION DOC] HELPER: Fetches the gradients of the given autograd context, builds a list aligned with the local parameters
        # (None where a parameter has no gradient), and calls the functional optimizer step with it.
        all_local_grads = dist_autograd.get_gradients(autograd_ctx_id)
        # apply functional optimizer step with a list of gradients
        grads: list[Tensor | None] = [
            all_local_grads[p] if p in all_local_grads else None  # noqa: SIM401
            for p in self._local_params
        ]

        self.optim.step(grads)


# TODO (wanchaol): remove/merge this with _FunctionalLocalOptimizer once
# we have converted all to functional optimizer in distributed.optim
class _LocalOptimizer:
    # [CLASS DOC] HELPER: Worker-side wrapper around a regular optimizer over local parameters resolved from RRefs.
    # A class-level global lock serializes steps across instances on a worker (assumes they optimize the same parameters).
    # Ideally we would only need to share a lock for instances of
    # _LocalOptimizer that deal with the same parameters. We are
    # making a simplifying assumption here that if there is more
    # than one instance of _LocalOptimizer per worker, they will
    # be optimizing the same parameters (e.g. each data parallel
    # trainer will create its own instance of _LocalOptimizer but
    # they will all optimize the same parameters on each worker)
    global_lock = Lock()

    def __init__(self, optim_cls, local_params_rref, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Resolves each RRef to its local tensor, stores the list, and instantiates the optimizer class on it with the extra arguments.
        self._local_params = [rref.local_value() for rref in local_params_rref]
        self.optim = optim_cls(self._local_params, *args, **kwargs)

    def step(self, autograd_ctx_id):
        # [FUNCTION DOC] HELPER: Under the global lock, copies the context's gradients onto the parameters' grad attributes and calls the optimizer step.
        all_local_grads = dist_autograd.get_gradients(autograd_ctx_id)

        with _LocalOptimizer.global_lock:
            for param, grad in all_local_grads.items():
                param.grad = grad
            self.optim.step()


def _new_local_optimizer(optim_cls, local_params_rref, *args, **kwargs):
# [FUNCTION DOC] HELPER: RPC target that creates a _LocalOptimizer on the worker and returns an RRef to it.
    return rpc.RRef(_LocalOptimizer(optim_cls, local_params_rref, *args, **kwargs))


def _local_optimizer_step(local_optim_rref, autograd_ctx_id):
# [FUNCTION DOC] HELPER: RPC target that runs step on the local optimizer behind the given RRef for the autograd context id.
    local_optim = local_optim_rref.local_value()
    local_optim.step(autograd_ctx_id)


def _new_functional_local_optimizer(optim_cls, local_params_rref, *args, **kwargs):
# [FUNCTION DOC] HELPER: RPC target that creates a _FunctionalLocalOptimizer on the worker and returns an RRef to it.
    optim = _FunctionalLocalOptimizer(optim_cls, local_params_rref, *args, **kwargs)
    return rpc.RRef(optim)


def _functional_local_optimizer_step(
    local_optim_rref: RRef, autograd_ctx_id: int
) -> None:
    # [FUNCTION DOC] HELPER: RPC target that runs step on the functional local optimizer behind the given RRef for the autograd context id.
    local_optim = local_optim_rref.local_value()
    local_optim.step(autograd_ctx_id)


def _wait_for_all(rpc_futs):
    # [FUNCTION DOC] HELPER: Waits on every future in order, collecting results (exceptions are collected as results too).
    # If any failed, raises the last exception seen after all have been waited on; otherwise returns the list of results.
    # TODO: improve error propagation
    exception = None
    results = []
    for fut in rpc_futs:
        try:
            results.append(fut.wait())
        except Exception as e:
            results.append(e)
            exception = e
    if exception is not None:
        raise exception
    return results


class DistributedOptimizer:
    # [CLASS DOC] Public distributed optimizer: groups parameter RRefs by owning worker and creates one optimizer per worker over RPC.
    # State: is_functional_optim (whether a functional optimizer was substituted) and remote_optimizers (RRefs to per-worker optimizers).
    # step triggers a step on each worker using a distributed autograd context.
    """
    DistributedOptimizer takes remote references to parameters scattered
    across workers and applies the given optimizer locally for each parameter.

    This class uses :meth:`~torch.distributed.autograd.get_gradients` in order
    to retrieve the gradients for specific parameters.

    Concurrent calls to
    :meth:`~torch.distributed.optim.DistributedOptimizer.step`,
    either from the same or different clients, will
    be serialized on each worker -- as each worker's optimizer can only work
    on one set of gradients at a time. However, there is no guarantee that
    the full forward-backward-optimizer sequence will execute for one client
    at a time. This means that the gradients being applied may not correspond
    to the latest forward pass executed on a given worker. Also, there is no
    guaranteed ordering across workers.

    `DistributedOptimizer` uses a functional optimizer internally when one is
    available for the given ``optimizer_class``, so that optimizer updates are
    not blocked by the Python Global Interpreter Lock (GIL) in the case of
    multithreaded training (e.g. Distributed Model Parallel). This feature is
    currently enabled for most optimizers.

    Args:
        optimizer_class (optim.Optimizer): the class of optimizer to
            instantiate on each worker.
        params_rref (list[RRef]): list of RRefs to local or remote parameters
            to optimize.
        args: arguments to pass to the optimizer constructor on each worker.
        kwargs: arguments to pass to the optimizer constructor on each worker.

    Example::
        >>> # xdoctest: +SKIP("distributed")
        >>> import torch.distributed.autograd as dist_autograd
        >>> import torch.distributed.rpc as rpc
        >>> from torch import optim
        >>> from torch.distributed.optim import DistributedOptimizer
        >>>
        >>> with dist_autograd.context() as context_id:
        >>>   # Forward pass.
        >>>   rref1 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 3))
        >>>   rref2 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 1))
        >>>   loss = rref1.to_here() + rref2.to_here()
        >>>
        >>>   # Backward pass.
        >>>   dist_autograd.backward(context_id, [loss.sum()])
        >>>
        >>>   # Optimizer.
        >>>   dist_optim = DistributedOptimizer(
        >>>      optim.SGD,
        >>>      [rref1, rref2],
        >>>      lr=0.05,
        >>>   )
        >>>   dist_optim.step(context_id)
    """

    def __init__(self, optimizer_class, params_rref, *args, **kwargs):
        # [FUNCTION DOC] Groups params_rref by owner, picks a functional optimizer from functional_optim_map when the class is mapped
        # and JIT is enabled (otherwise logs a slow-path warning and uses the class as is), then asynchronously creates the optimizer on each worker
        # and waits for all creations, storing the RRefs. Raises any exception from a failed worker.
        torch._C._log_api_usage_once("torch.distributed.optim.DistributedOptimizer")
        per_worker_params_rref = defaultdict(list)
        for param in params_rref:
            per_worker_params_rref[param.owner()].append(param)

        if optimizer_class in functional_optim_map and jit._state._enabled:
            optim_ctor = functional_optim_map.get(optimizer_class)
        else:
            optim_ctor = optimizer_class
        self.is_functional_optim = optim_ctor != optimizer_class

        if self.is_functional_optim:
            optimizer_new_func = _new_functional_local_optimizer
        else:
            logger.warning(
                "Creating the optimizer %s without a functional optimizer, "
                "this might result in slow computation time in multithreading environment"
                "(i.e. Distributed Model Parallel training on CPU) due to the Python's "
                "Global Interpreter Lock (GIL). Please file an issue if you need a "
                "functional optimizer for this optimizer. ",
                optimizer_class,
            )
            optimizer_new_func = _new_local_optimizer

        remote_optim_futs = []
        for worker, param_rrefs in per_worker_params_rref.items():
            remote_optim_rref_fut = rpc.rpc_async(
                worker,
                optimizer_new_func,
                args=(optim_ctor, param_rrefs) + args,
                kwargs=kwargs,
            )
            remote_optim_futs.append(remote_optim_rref_fut)

        self.remote_optimizers = _wait_for_all(remote_optim_futs)

    def step(self, context_id):
        # [FUNCTION DOC] Validates the autograd context id, asynchronously invokes the matching step function on every worker's optimizer,
        # and blocks until all finish, raising if any failed. Returns None.
        """
        Performs a single optimization step.

        This will call :meth:`torch.optim.Optimizer.step` on each worker
        containing parameters to be optimized, and will block until all workers
        return. The provided ``context_id`` will be used to retrieve the
        corresponding :class:`~torch.distributed.autograd.context` that
        contains the gradients that should be applied to the parameters.

        Args:
            context_id: the autograd context id for which we should run the
                optimizer step.
        """
        dist_autograd._is_valid_context(context_id)

        optimizer_step_func = (
            _functional_local_optimizer_step
            if self.is_functional_optim
            else _local_optimizer_step
        )

        rpc_futs = [
            rpc.rpc_async(
                optimizer.owner(),
                optimizer_step_func,
                args=(optimizer, context_id),
            )
            for optimizer in self.remote_optimizers
        ]
        _wait_for_all(rpc_futs)
