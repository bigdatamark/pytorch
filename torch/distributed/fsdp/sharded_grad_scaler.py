# mypy: allow-untyped-defs
# [MODULE DOC] Gradient scaler for sharded training (FSDP/DDP). Defines ShardedGradScaler, a
# GradScaler subclass that supports CPU and other non-CUDA devices, keeps the scaled loss in its
# original (possibly low precision) dtype, and all-reduces inf/nan flags across a process group.
# Also defines small private helpers. Only creates a module logger at import time.
import logging
from collections import abc, defaultdict
from collections.abc import Iterable
from typing import Any, overload

import torch
import torch.distributed as dist
from torch.amp.grad_scaler import _MultiDeviceReplicator, GradScaler, OptState
from torch.distributed.distributed_c10d import ProcessGroup


logger = logging.getLogger(__name__)


def _refresh_per_optimizer_state() -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Factory for fresh per-optimizer scaler state: stage READY and an empty
    # mapping from device to found-inf tensor. Used as the defaultdict default factory.
    return {"stage": OptState.READY, "found_inf_per_device": {}}


def _is_supported_device(tensor: torch.Tensor) -> bool:
    # [FUNCTION DOC] HELPER: True if the tensor lives on CUDA or on one of xla, cpu, hpu, mtia, xpu
    # or the registered privateuse1 backend; false otherwise.
    return tensor.is_cuda or tensor.device.type in (
        "xla",
        "cpu",
        "hpu",
        "mtia",
        "xpu",
        torch._C._get_privateuse1_backend_name(),
    )


class _GeneralMultiDeviceReplicator(_MultiDeviceReplicator):
    # [CLASS DOC] HELPER: Multi-device tensor replicator that also accepts CPU and other supported
    # devices. Holds a master tensor and a lazily filled mapping of device to copy (the base class
    # get method serves per-device copies).
    """
    Lazily serves tensor to request device. This class extends
    _MultiDeviceReplicator to allow support for "cpu" as a device.
    """

    def __init__(self, master_tensor: torch.Tensor) -> None:
        # [FUNCTION DOC] HELPER: Stores master_tensor and an empty per-device cache. Raises
        # AssertionError if the tensor's device is unsupported. Does not call the base constructor.
        if not _is_supported_device(master_tensor):
            raise AssertionError(
                f"Expected supported device, got {master_tensor.device}"
            )
        self.master = master_tensor
        self._per_device_tensors: dict[torch.device, torch.Tensor] = {}


class ShardedGradScaler(GradScaler):
    # [CLASS DOC] GradScaler for sharded/offloaded training. Adds a process_group (set only when
    # enabled) used to all-reduce found-inf flags in unscale_, supports CPU/non-CUDA devices, keeps
    # scaled outputs in their input dtype, and tracks per-optimizer state keyed by optimizer id.
    # Key methods: scale, unscale_, update. When disabled, all methods are no-ops/pass-through.
    """
    ShardedGradScaler helps perform gradient scaling in a shard aware manner. It extends
    functionality from GradScaler:
    * Supports PyTorch DDP and FSDP implementations
    * Support CPU offloaded tensors (as used in fully sharded data parallel[FSDP])
    * Supports the custom Mixed Precision loss dtype (fp16, bf16) that FSDP returns
    * Sync inf/nan for scaled gradient tensors on any torch.device (where tensors are placed) across
    nodes

    Example::

        # Creates a ShardedGradScaler once at the beginning of training.
        scaler = ShardedGradScaler()

        for epoch in epochs:
            for input, target in data:
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)

                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                scaler.scale(loss).backward()

                # scaler.step() first unscales gradients of the optimizer's params.
                # If gradients don't contain infs/NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()

    See :class:`GradScaler` for explanation of scaling/unscaling and more use cases.

    Args:
        init_scale (float, optional, default=2.**16):  Initial scale factor.
        growth_factor (float, optional, default=2.0):  Factor by which the scale is multiplied during
            :meth:`update` if no inf/NaN gradients occur for ``growth_interval`` consecutive iterations.
        backoff_factor (float, optional, default=0.5):  Factor by which the scale is multiplied during
            :meth:`update` if inf/NaN gradients occur in an iteration.
        growth_interval (int, optional, default=2000):  Number of consecutive iterations without inf/NaN gradients
            that must occur for the scale to be multiplied by ``growth_factor``.
        enabled (bool, optional):  If ``False``, disables gradient scaling. :meth:`step` simply
            invokes the underlying ``optimizer.step()``, and other methods become no-ops.
            Default: ``True``
        process_group (ProcessGroup, optional, default=torch.distributed.group.WORLD):
            process group for sharding
    """

    def __init__(
        self,
        device: str = "cuda",
        init_scale: float = 2.0**16,
        backoff_factor: float = 0.5,
        growth_factor: float = 2.0,
        growth_interval: int = 2000,
        enabled: bool = True,
        process_group: ProcessGroup | None = dist.group.WORLD,
    ) -> None:
        # [FUNCTION DOC] Initializes the base GradScaler with the given device and scaling
        # hyperparameters. If enabled, also stores process_group (default world group) and creates
        # the per-optimizer state mapping with fresh READY entries as default.
        super().__init__(
            device,
            init_scale=init_scale,
            backoff_factor=backoff_factor,
            growth_factor=growth_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
        if self._enabled:
            self.process_group = process_group
            self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)

    @overload
    def scale(self, outputs: torch.Tensor) -> torch.Tensor: ...

    @overload
    def scale(self, outputs: list[torch.Tensor]) -> list[torch.Tensor]: ...

    @overload
    def scale(self, outputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]: ...

    @overload
    def scale(self, outputs: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]: ...

    def scale(
        self, outputs: torch.Tensor | Iterable[torch.Tensor]
    ) -> torch.Tensor | Iterable[torch.Tensor]:
        # [FUNCTION DOC] Multiplies a tensor, or recursively each tensor in nested lists/tuples/
        # iterables, by the current scale; the result keeps each input's dtype. Lazily initializes
        # the scale on the first tensor's device. Returns outputs unchanged if disabled. Raises
        # AssertionError for unsupported devices and ValueError for non-tensor, non-iterable input.
        # Non-list/tuple iterables are returned as lazy iterators.
        if not self._enabled:
            return outputs

        if isinstance(outputs, torch.Tensor):
            if not _is_supported_device(outputs):
                raise AssertionError(f"Expected supported device, got {outputs.device}")
            if self._scale is None:
                self._lazy_init_scale_growth_tracker(outputs.device)
            if self._scale is None:
                raise AssertionError("Expected _scale to be initialized, got None")
            scaled_output = outputs * self._scale.to(
                device=outputs.device, non_blocking=True
            )
            # Here we ensure the return dtype is the same as the outputs dtype.
            # For the FSDP + Mixed Precision use case, the loss output is in the Mixed Precision
            # format (fp16, bf16) and so the scaled loss should be of the same dtype.
            return scaled_output.type(outputs.dtype)

        stash: list[_GeneralMultiDeviceReplicator] = []

        def apply_scale(val: torch.Tensor | Iterable[torch.Tensor]):
            # [FUNCTION DOC] HELPER: Recursive worker for scale. For a tensor, scales it using a shared
            # replicator (created once into the stash) and restores its dtype. For an iterable, maps
            # itself over the elements, rebuilding lists/tuples. Raises ValueError otherwise.
            if isinstance(val, torch.Tensor):
                if not _is_supported_device(val):
                    raise AssertionError(f"Expected supported device, got {val.device}")
                if len(stash) == 0:
                    if self._scale is None:
                        self._lazy_init_scale_growth_tracker(val.device)
                    if self._scale is None:
                        raise AssertionError(
                            "Expected _scale to be initialized, got None"
                        )
                    stash.append(_GeneralMultiDeviceReplicator(self._scale))
                scaled_val = val * stash[0].get(val.device)
                # Here we ensure the return dtype is the same as the outputs dtype.
                # For the FSDP + Mixed Precision use case, the loss output is in the Mixed Precision
                # format (fp16, bf16) and so the scaled loss should be of the same dtype.
                return scaled_val.type(val.dtype)
            if isinstance(val, abc.Iterable):
                iterator = map(apply_scale, val)
                if isinstance(val, (list, tuple)):
                    return type(val)(iterator)
                return iterator
            raise ValueError("outputs must be a Tensor or an iterable of Tensors")

        return apply_scale(outputs)

    def _unscale_grads_(
        self,
        optimizer: torch.optim.Optimizer,
        inv_scale: torch.Tensor,
        found_inf: torch.Tensor,
        allow_fp16: bool = True,
    ) -> dict[torch.device, torch.Tensor]:
        # [FUNCTION DOC] HELPER: Groups the optimizer's gradients (sparse ones via their values) by
        # device and dtype, then unscales each group in place with the fused non-finite check,
        # recording inf/nan into a per-device found_inf tensor. Raises ValueError for fp16 grads when
        # allow_fp16 is false; fp16 sparse grads are coalesced via fp32 (mutating param.grad).
        # Ensures the result is non-empty by touching the scale's device (ranks may own no grads).
        # Returns the mapping of device to found_inf tensor.
        per_device_inv_scale = _GeneralMultiDeviceReplicator(inv_scale)
        per_device_found_inf = _GeneralMultiDeviceReplicator(found_inf)

        # To set up _amp_foreach_non_finite_check_and_unscale_, split grads by device and dtype.
        # There could be thousands of grads, so we'd like to iterate through them just once.
        # However, we don't know their devices or dtypes in advance.

        # https://stackoverflow.com/questions/5029934/defaultdict-of-defaultdict
        # Google says mypy struggles with defaultdicts type annotations.
        per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))  # type: ignore[var-annotated]
        with torch.no_grad():
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    if (not allow_fp16) and param.grad.dtype == torch.float16:
                        raise ValueError("Attempting to unscale FP16 gradients.")
                    if param.grad.is_sparse:
                        # is_coalesced() == False means the sparse grad has values with duplicate indices.
                        # coalesce() deduplicates indices and adds all values that have the same index.
                        # For scaled fp16 values, there's a good chance coalescing will cause overflow,
                        # so we should check the coalesced _values().
                        if param.grad.dtype is torch.float16:
                            # coalesce is not supported in torch.float16
                            param_grad_fp32 = param.grad.type(torch.float32).coalesce()
                            param.grad = param_grad_fp32.type(torch.float16)
                        to_unscale = param.grad._values()
                    else:
                        to_unscale = param.grad

                    per_device_and_dtype_grads[to_unscale.device][
                        to_unscale.dtype
                    ].append(to_unscale)

            for device, per_dtype_grads in per_device_and_dtype_grads.items():
                for grads in per_dtype_grads.values():
                    torch._amp_foreach_non_finite_check_and_unscale_(
                        grads,
                        per_device_found_inf.get(device),
                        per_device_inv_scale.get(device),
                    )
        # There exist contexts (e.g. w/ `use_orig_params=True`) wherein some
        # ranks may have no (non-zero sized) parameter shards, necessitating the
        # initialization of `per_device_found_inf._per_device_tensors` here
        if not per_device_found_inf._per_device_tensors:
            if self._scale is None:
                raise AssertionError("Expected _scale to be initialized, got None")
            per_device_found_inf.get(self._scale.device)
        return per_device_found_inf._per_device_tensors

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        # [FUNCTION DOC] Divides the optimizer's gradients by the scale (reciprocal computed in fp64),
        # records per-device found_inf flags, marks the optimizer UNSCALED, then asynchronously
        # all-reduces the flags over process_group (CPU flags are moved to the scaler device for the
        # reduce and copied back). No-op if disabled. Raises RuntimeError if already unscaled since
        # the last update or if called after step.
        if not self._enabled:
            return

        self._check_scale_growth_tracker("unscale_")

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state["stage"] is OptState.UNSCALED:
            raise RuntimeError(
                "unscale_() has already been called on this optimizer since the last update()."
            )
        elif optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("unscale_() is being called after step().")

        # FP32 division can be imprecise for certain compile options, so we carry out the reciprocal in FP64.
        if self._scale is None:
            raise AssertionError("Expected _scale to be initialized, got None")
        inv_scale = self._scale.double().reciprocal().float()
        found_inf = torch.full(
            (1,), 0.0, dtype=torch.float32, device=self._scale.device
        )

        optimizer_state["found_inf_per_device"] = self._unscale_grads_(
            optimizer, inv_scale, found_inf, True
        )
        optimizer_state["stage"] = OptState.UNSCALED

        # Synchronize the detected inf across the ranks
        optimizer_state = self._per_optimizer_states[id(optimizer)]
        works = []
        found_inf_on_cpus = []
        found_inf_on_devices = []

        for found_inf in optimizer_state["found_inf_per_device"].values():
            if self._device != "cpu" and found_inf.device.type == "cpu":
                found_inf_on_cpus.append(found_inf)
                found_inf_on_device = found_inf.to(self._device)
                found_inf_on_devices.append(found_inf_on_device)
                works.append(
                    dist.all_reduce(
                        found_inf_on_device, async_op=True, group=self.process_group
                    )
                )
            else:
                works.append(
                    dist.all_reduce(found_inf, async_op=True, group=self.process_group)
                )
        for work in works:
            work.wait()
        if found_inf_on_cpus:
            torch._foreach_copy_(found_inf_on_cpus, found_inf_on_devices)

    def _amp_update_scale_cpu_(self, found_inf: torch.Tensor) -> None:
        # [FUNCTION DOC] HELPER: CPU implementation of the scale update. If found_inf is at least 1,
        # multiplies the scale by the backoff factor and zeroes the growth tracker; otherwise
        # increments the tracker and, on reaching the growth interval, multiplies the scale by the
        # growth factor and resets the tracker. Mutates scale/tracker; raises AssertionError if uninitialized.
        """
        If found_inf is 1.0 (True), then scale is multiplied by backoff_factor and growth_tracker is set to zero.
        Otherwise, scale is multiplied by the growth factor when the growth interval is reached.
        """
        if self._scale is None or self._growth_tracker is None:
            raise AssertionError(
                "Expected _scale and _growth_tracker to be initialized, got None"
            )

        if found_inf.item() >= 1.0:
            self._scale *= self._backoff_factor
            self._growth_tracker.fill_(0)
        else:
            successful = self._growth_tracker + 1
            if successful == self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker.fill_(0)
            else:
                self._growth_tracker = successful

    def update(self, new_scale: float | torch.Tensor | None = None) -> None:
        # [FUNCTION DOC] Ends the iteration: sets the scale to new_scale if given (a float, or a
        # one-element non-grad tensor on the scaler device; AssertionError otherwise), else sums the
        # recorded found_inf flags from all optimizers and applies backoff/growth (CPU path or the
        # fused update op). Raises AssertionError if no inf checks were recorded. Always resets the
        # per-optimizer states afterwards. No-op if disabled.
        """
        Updates the scale factor.
        If any optimizer steps were skipped the scale is multiplied by ``backoff_factor``
        to reduce it. If ``growth_interval`` unskipped iterations occurred consecutively,
        the scale is multiplied by ``growth_factor`` to increase it.
        Passing ``new_scale`` sets the new scale value manually. (``new_scale`` is not
        used directly, it's used to fill GradScaler's internal scale tensor. So if
        ``new_scale`` was a tensor, later in-place changes to that tensor will not further
        affect the scale GradScaler uses internally.)
        Args:
            new_scale (float or :class:`torch.Tensor`, optional, default=None):  New scale factor.
        .. warning::
            :meth:`update` should only be called at the end of the iteration, after ``scaler.step(optimizer)`` has
            been invoked for all optimizers used this iteration.
        """

        if not self._enabled:
            return

        _scale, _growth_tracker = self._check_scale_growth_tracker("update")  # type: ignore[var-annotated]

        if new_scale is not None:
            # Accept a new user-defined scale.
            if isinstance(new_scale, float):
                self._scale.fill_(new_scale)  # type: ignore[union-attr]
            else:
                reason = (
                    "new_scale should be a float or a 1-element torch.cuda.FloatTensor or "
                    "torch.FloatTensor with requires_grad=False."
                )
                if new_scale.device.type != self._device:
                    raise AssertionError(reason)
                if new_scale.numel() != 1:
                    raise AssertionError(reason)
                if new_scale.requires_grad is not False:
                    raise AssertionError(reason)
                self._scale.copy_(new_scale)  # type: ignore[union-attr]
        else:
            # Consume shared inf/nan data collected from optimizers to update the scale.
            # If all found_inf tensors are on the same device as self._scale, this operation is asynchronous.
            found_infs = [
                found_inf.to(device=_scale.device, non_blocking=True)
                for state in self._per_optimizer_states.values()
                for found_inf in state["found_inf_per_device"].values()
            ]

            if len(found_infs) == 0:
                raise AssertionError("No inf checks were recorded prior to update.")

            found_inf_combined = found_infs[0]
            if len(found_infs) > 1:
                for i in range(1, len(found_infs)):
                    found_inf_combined += found_infs[i]

            if _scale.device.type == "cpu":
                self._amp_update_scale_cpu_(found_inf_combined)
            else:
                torch._amp_update_scale_(
                    self._scale,  # type: ignore[arg-type]
                    self._growth_tracker,  # type: ignore[arg-type]
                    found_inf_combined,
                    self._growth_factor,  # type: ignore[arg-type]
                    self._backoff_factor,  # type: ignore[arg-type]
                    self._growth_interval,  # type: ignore[arg-type]
                )

        # To prepare for next iteration, clear the data collected from optimizers this iteration.
        self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)
