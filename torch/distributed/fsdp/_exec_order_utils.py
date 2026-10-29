# mypy: allow-untyped-defs
# [MODULE DOC] HELPER: Tracks FSDP1 execution order for forward/backward prefetching and, at
# DETAIL debug level, validates that the forward order matches across ranks (first iteration)
# and across iterations (warnings). Contains _ExecOrderWarnStatus and _ExecOrderData.
import itertools
import warnings
from enum import auto, Enum

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.fsdp._common_utils import _FSDPState, _get_param_to_fqns
from torch.distributed.fsdp._flat_param import FlatParamHandle


class _ExecOrderWarnStatus(Enum):
    # [CLASS DOC] HELPER: Enum of the order-deviation warning state: NONE (no deviation),
    # WARNING (deviating in the current iteration), WARNED (deviated in an earlier iteration).
    """Used internally for execution order validation."""

    NONE = auto()  # no deviation yet
    WARNING = auto()  # deviated this iteration; currently issuing warnings
    WARNED = auto()  # deviated in a previous iteration


class _ExecOrderData:
    # [CLASS DOC] HELPER: Per-root-FSDP record of handle execution order. State: the static
    # pre-forward order (first iteration only), the post-forward order (reset every
    # iteration), prefetch limits, an iteration counter, and (when checking at DETAIL debug
    # level) the process group, world size, ordered list of all handles, parameter to FQN
    # mapping, current order index and warning status. Key methods: init, record_pre_forward,
    # record_post_forward, get_handle_to_forward/backward_prefetch, next_iter.
    """
    This contains the data structures to track the execution order. We track
    the pre-forward order on the *first* iteration for forward prefetching
    (which thus assumes static graph) and the post-forward order on *every*
    iteration for backward prefetching (which thus does not assume static
    graph but may be provide an incorrect order).
    """

    def __init__(
        self,
        debug_level: dist.DebugLevel,
        backward_prefetch_limit: int,
        forward_prefetch_limit: int,
    ) -> None:
        # [FUNCTION DOC] HELPER: Creates empty order lists, stores the backward and forward
        # prefetch limits, and enables order checking only if debug_level is DETAIL. The
        # process group, world size and handle metadata are filled later by init.
        # Tracks the (static) pre-forward order for execution order validation
        # and forward prefetching
        self.handles_pre_forward_order: list[FlatParamHandle] = []
        # Tracks the post-forward order for pre-backward prefetching
        self.handles_post_forward_order: list[FlatParamHandle | None] = []
        self._iter = 0

        # Gives the max number of backward/forward prefetched all-gathers by a
        # single module
        self._backward_prefetch_limit = backward_prefetch_limit
        self._forward_prefetch_limit = forward_prefetch_limit

        # Data structures for execution order validation
        self._checking_order: bool = debug_level == dist.DebugLevel.DETAIL
        self.process_group: dist.ProcessGroup | None = None
        self.world_size: int | None = None
        self.all_handles: list[FlatParamHandle] = []
        # Names are prefixed from the root module
        self.param_to_fqn: dict[nn.Parameter, list[str]] = {}
        # Current index in the pre-forward execution order
        self.current_order_index = 0
        self.warn_status = _ExecOrderWarnStatus.NONE

    def init(
        self,
        state: _FSDPState,
        root_module: nn.Module,
        process_group: dist.ProcessGroup,
    ) -> None:
        # [FUNCTION DOC] HELPER: Called once after the root FSDP is set during lazy init.
        # Stores the process group, rank and world size, assigns each handle found under
        # root_module a stable index (setting its handle index attribute) and builds the
        # parameter to FQN mapping. The state argument is unused.
        """
        Initializes the data structures needed for checking the forward order.
        This should be called after a root FSDP instance has been set during
        lazy initialization.
        """
        self.process_group = process_group
        self.rank = process_group.rank()
        self.world_size = process_group.size()
        # Fix an order over the handles, which should be the same across ranks
        for handle in traversal_utils._get_fsdp_handles(root_module):
            index = len(self.all_handles)
            self.all_handles.append(handle)
            handle._handle_index = index
        self.param_to_fqn = _get_param_to_fqns(root_module)
        # TODO (awgu): We can broadcast the metadata of rank 0's `all_handles`
        # to check that all ranks have the same handles in the same order.
        # https://github.com/pytorch/pytorch/issues/79620

    @property
    def is_first_iter(self) -> bool:
        # [FUNCTION DOC] HELPER: True while the iteration counter is zero.
        return self._iter == 0

    def get_handle_to_backward_prefetch(
        self,
        current_handle: FlatParamHandle,
    ) -> FlatParamHandle | None:
        # [FUNCTION DOC] HELPER: Returns the single handle to prefetch for backward: walking
        # back from the current handle's post-forward index by up to the backward prefetch
        # limit, the last handle reached. Returns None if the current handle has no recorded
        # index (or the limit/index leaves nothing to fetch).
        # KNOWN ISSUE: docstring says it returns a list; it returns one handle or None.
        """
        Returns a :class:`list` of the handles keys of the handles to backward
        prefetch given the current handles key. If there are no valid handles
        keys to prefetch, then this returns an empty :class:`list`.
        """
        current_index = current_handle._post_forward_index
        if current_index is None:
            return None
        target_index = current_index - 1
        target_handle: FlatParamHandle | None = None
        for _ in range(self._backward_prefetch_limit):
            if target_index < 0:
                break
            target_handle = self.handles_post_forward_order[target_index]
            target_index -= 1
        return target_handle

    def get_handle_to_forward_prefetch(
        self,
        current_handle: FlatParamHandle,
    ) -> FlatParamHandle | None:
        # [FUNCTION DOC] HELPER: Returns the single handle to prefetch for forward: walking
        # ahead from the current handle's pre-forward order index by up to the forward
        # prefetch limit (stopping at the end of the recorded order), the last handle reached.
        # Returns None if the current handle has no recorded index or nothing is available.
        # KNOWN ISSUE: docstring says it returns a list; it returns one handle or None.
        """
        Returns a :class:`list` of the handles keys of the handles to forward
        prefetch given the current handles key. If there are no valid handles
        keys to prefetch, then this returns an empty :class:`list`.
        """
        current_index = current_handle._pre_forward_order_index
        if current_index is None:
            return None
        target_index = current_index + 1
        target_handle: FlatParamHandle | None = None
        for _ in range(self._forward_prefetch_limit):
            if target_index >= len(self.handles_pre_forward_order):
                break
            target_handle = self.handles_pre_forward_order[target_index]
            target_index += 1
        return target_handle

    def record_post_forward(self, handle: FlatParamHandle | None) -> None:
        # [FUNCTION DOC] HELPER: Appends the handle to the post-forward order (does nothing
        # for a falsy handle). Sets the handle's post-forward index to its position on first
        # use only; later uses are appended without changing the index.
        # KNOWN ISSUE: the first-use test is truthiness of the index, so a handle whose index
        # is 0 is treated as already recorded (harmless: index stays 0).
        """
        Records ``handles`` in the post-forward order, where ``handles`` should
        be a group of handles used in the same module's forward. If ``handles``
        is empty, then it is omitted.

        Unlike :meth:`record_pre_forward`, this records the order *every*
        iteration with the expectation that the recorded order is reset in
        :meth:`next_iter`.
        """
        if not handle:
            return
        # Only record the first usage of a handles key
        if handle._post_forward_index:
            self.handles_post_forward_order.append(handle)
            return
        index = len(self.handles_post_forward_order)
        handle._post_forward_index = index
        self.handles_post_forward_order.append(handle)

    def record_pre_forward(
        self, handle: FlatParamHandle | None, is_training: bool
    ) -> None:
        # [FUNCTION DOC] HELPER: Ignores a falsy handle. Otherwise runs the order check, then
        # on the first iteration only, records the handle's index and appends it to the static
        # pre-forward order (once per handle). Can raise RuntimeError from the check.
        """
        Records ``handles`` in the pre-forward order, where ``handles`` should
        be a group of handles used in the same module's forward. If ``handles``
        is empty, then it is omitted.

        On the first iteration, this checks the execution order across ranks.
        See :meth:`_check_order` for details.
        """
        if not handle:
            return
        self._check_order(handle, is_training)
        # Fix the order after the first iteration and only record the first
        # usage of a handles key
        if not self.is_first_iter or handle._pre_forward_order_index is not None:
            return
        index = len(self.handles_pre_forward_order)
        handle._pre_forward_order_index = index
        self.handles_pre_forward_order.append(handle)

    def _check_order(self, handle: FlatParamHandle, is_training: bool) -> None:
        # [FUNCTION DOC] HELPER: No-op unless training and checking is enabled. First
        # iteration: all-gathers each rank's handle count, then handle indices, and raises
        # RuntimeError if ranks disagree (comparisons skipped under Dynamo compilation).
        # Later iterations: compares the handle to the expected one at the current order
        # index and warns once per deviating run (extra or different all-gather), updating the
        # warning status; advances the order index. Side effects: collectives, warnings.
        """
        Checks the forward execution order as long as ``is_training`` is
        ``True`` since checking in eval mode is not supported. This only checks
        if the distributed debug level is DETAIL.

        - On the first iteration, this uses all-gathers to check that all ranks
        are all-gathering the same handles and hence ``FlatParameter`` s,
        raising an error if not.
        - On subsequent iterations, this checks that each rank is locally
        consistent with its own forward order from the first iteration, issuing
        a warning if not. This issues a warning on the first deviating
        iteration and stops warning thereafter.
        """
        # Do not check order in eval mode since the post-backward callback does
        # not run so it cannot be used to mark the end of an iteration
        if not is_training or not self._checking_order:
            return
        if self.is_first_iter:
            msg_prefix = "Forward order differs across ranks:"
            optional_local_indices: tuple[int | None, ...] = self._get_handle_indices(
                handle
            )
            device = handle.device  # guaranteed to be non-CPU
            num_valid_indices = sum(
                (index is not None) for index in optional_local_indices
            )
            tensor_kwargs: dict[str, torch.dtype | torch.device] = {
                "dtype": torch.int32,
                "device": device,
            }
            world_num_valid_indices = torch.zeros(self.world_size, **tensor_kwargs)  # type: ignore[arg-type, call-overload]
            local_num_valid_indices = torch.tensor([num_valid_indices], **tensor_kwargs)  # type: ignore[arg-type, call-overload]
            dist.all_gather_single(
                world_num_valid_indices,
                local_num_valid_indices,
                group=self.process_group,
            )
            # Copy entire tensor from D2H once to avoid per element D2H copies
            world_num_valid_indices = world_num_valid_indices.cpu()
            # Check that all ranks plan to all-gather the same number of
            # parameters
            # TODO (awgu): Since every module has at most one handle in the
            # current implementation, this should never raise the error.
            if self.world_size is None:
                raise AssertionError("Expected world_size to not be None")
            if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
                # TODO(voz): Don't graph break on this - dynamo hates the n1 != n2
                # tensor comparison control flow.
                # https://github.com/pytorch/pytorch/issues/107055
                for (r1, n1), (r2, n2) in itertools.combinations(
                    (
                        (rank, world_num_valid_indices[rank])
                        for rank in range(self.world_size)
                    ),
                    2,
                ):
                    if n1 != n2:
                        raise RuntimeError(
                            f"{msg_prefix} rank {r1} is all-gathering {n1} parameters "
                            f"while rank {r2} is all-gathering {n2} parameters"
                        )
            world_indices = torch.zeros(  # type: ignore[call-overload]
                self.world_size * num_valid_indices, **tensor_kwargs
            )
            local_indices = torch.tensor(optional_local_indices, **tensor_kwargs)  # type: ignore[arg-type]
            dist.all_gather_single(
                world_indices, local_indices, group=self.process_group
            )
            # Copy entire tensor from D2H once to avoid per element D2H copies
            world_indices = world_indices.cpu()
            # Check that all ranks plan to all-gather the same index parameters
            if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
                # TODO(voz): Don't graph break on this - dynamo hates the i1 != i2
                # tensor comparison control flow.
                # https://github.com/pytorch/pytorch/issues/107055
                for (r1, i1), (r2, i2) in itertools.combinations(
                    (
                        (
                            rank,
                            world_indices[
                                rank * num_valid_indices : (rank + 1)
                                * num_valid_indices
                            ],
                        )
                        for rank in range(self.world_size)
                    ),
                    2,
                ):
                    if i1 != i2:
                        r1_param_names = self._get_names_from_handle_indices(i1)
                        r2_param_names = self._get_names_from_handle_indices(i2)
                        raise RuntimeError(
                            f"{msg_prefix} rank {r1} is all-gathering parameters "
                            f"for {r1_param_names} while rank {r2} is all-gathering "
                            f"parameters for {r2_param_names}"
                        )
        else:
            # Only issue warnings on the first deviating iteration and stop
            # checking thereafter to avoid flooding the console
            if self.warn_status == _ExecOrderWarnStatus.WARNED:
                return
            msg_prefix = None  # non-`None` means we should warn
            if self.current_order_index >= len(self.handles_pre_forward_order):
                # This iteration sees extra all-gather(s) compared to the first
                msg_prefix = (
                    "Expected to not all-gather any more parameters in the "
                    "forward but trying to all-gather parameters for "
                )
            else:
                expected_handle = self.handles_pre_forward_order[
                    self.current_order_index
                ]
                if expected_handle != handle:
                    expected_param_names = self._get_names_from_handles(expected_handle)
                    msg_prefix = (
                        f"Expected to all-gather for {expected_param_names} "
                        "but trying to all-gather parameters for "
                    )
            if msg_prefix is not None:
                param_names = self._get_names_from_handles(handle)
                msg_suffix = (
                    f"{param_names}"
                    if param_names
                    else "a newly-added parameter since construction time"
                )
                warnings.warn(
                    "Forward order differs from that of the first iteration "
                    f"on rank {self.rank}. Collectives are unchecked and may "
                    f"give incorrect results or hang.\n{msg_prefix}{msg_suffix}",
                    stacklevel=2,
                )
                self.warn_status = _ExecOrderWarnStatus.WARNING
            self.current_order_index += 1

    def _get_handle_indices(
        self,
        handle: FlatParamHandle,
    ) -> tuple[int | None, ...]:
        # [FUNCTION DOC] HELPER: Returns a tuple holding the handle's assigned index, or an
        # empty tuple if the handle is falsy.
        """
        Returns the handle indices (i.e. indices into ``self.all_handles``)
        corresponding to the handles in ``handle``. An entry in the
        returned tuple is ``None`` if the handle is invalid.
        """
        indices: list[int | None] = []
        if handle:
            indices.append(handle._handle_index)
        return tuple(indices)

    def _get_names_from_handle_indices(
        self,
        handle_indices: tuple[int, ...],
    ) -> list[list[str]]:
        # [FUNCTION DOC] HELPER: Maps each valid index into the ordered handle list to the
        # FQN list of that handle's flat parameter; invalid or None indices are skipped.
        """
        Returns a list of FQNs for each handle in ``handle_indices``. If a
        handle index is invalid, then its FQNs are omitted from the returned
        list.
        """
        fqns: list[list[str]] = []
        for index in handle_indices:
            if index is None or index < 0 or index >= len(self.all_handles):
                continue
            handle = self.all_handles[index]
            flat_param = handle.flat_param
            fqns.append(self.param_to_fqn[flat_param])
        return fqns

    def _get_names_from_handles(
        self,
        handle: FlatParamHandle,
    ) -> list[list[str]]:
        # [FUNCTION DOC] HELPER: Returns a list with the FQN list of the handle's flat
        # parameter, or an empty list if the handle is falsy or its parameter is unknown.
        """
        Returns a list of FQNs for each handle in ``handles_key``. If a handle
        is invalid, then its FQNs are omitted from the returned list.
        """
        fqns: list[list[str]] = []
        if handle:
            flat_param = handle.flat_param
            if flat_param in self.param_to_fqn:
                fqns.append(self.param_to_fqn[flat_param])
        return fqns

    def next_iter(self):
        # [FUNCTION DOC] HELPER: Ends an iteration: increments the counter, clears the
        # post-forward order and, when checking, resets the order index and promotes the
        # warning status from WARNING to WARNED.
        """
        Advances the internal data structures per iteration. This should be
        called in the post-backward callback since that marks the true end of
        an iteration.
        """
        self._iter += 1
        self.handles_post_forward_order.clear()
        if self._checking_order:
            self.current_order_index = 0
            if self.warn_status == _ExecOrderWarnStatus.WARNING:
                self.warn_status = _ExecOrderWarnStatus.WARNED
