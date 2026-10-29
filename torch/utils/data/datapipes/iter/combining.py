# mypy: allow-untyped-defs
# [MODULE DOC] Iterable datapipes that combine or split streams: Concater (concat), Forker
# (fork) and Demultiplexer (demux) with their private container and child datapipes that
# share a bounded buffer, Multiplexer (mux) and Zipper (zip). Import registers these
# functional names; no other side effects.
import copy as copymodule
import warnings
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterator, Sized
from typing import Any, Literal, TypeVar

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import _check_unpickable_fn, StreamWrapper


__all__ = [
    "ConcaterIterDataPipe",
    "DemultiplexerIterDataPipe",
    "ForkerIterDataPipe",
    "MultiplexerIterDataPipe",
    "ZipperIterDataPipe",
]


_T_co = TypeVar("_T_co", covariant=True)


@functional_datapipe("concat")
class ConcaterIterDataPipe(IterDataPipe):
    # [CLASS DOC] Yields all items of the first input datapipe, then the next, and so on.
    # State: datapipes tuple. Sized only if every input is sized.
    r"""
    Concatenates multiple Iterable DataPipes (functional name: ``concat``).

    The resulting DataPipe will yield all the elements from the first input DataPipe, before yielding from the subsequent ones.

    Args:
        datapipes: Iterable DataPipes being concatenated

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> import random
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1 = IterableWrapper(range(3))
        >>> dp2 = IterableWrapper(range(5))
        >>> list(dp1.concat(dp2))
        [0, 1, 2, 0, 1, 2, 3, 4]
    """

    datapipes: tuple[IterDataPipe]

    def __init__(self, *datapipes: IterDataPipe) -> None:
        # [FUNCTION DOC] Stores the input datapipes. Raises ValueError if none are given
        # and TypeError if any input is not an IterDataPipe. Does not call the base
        # constructor.
        if len(datapipes) == 0:
            raise ValueError("Expected at least one DataPipe, but got nothing")
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError("Expected all inputs to be `IterDataPipe`")
        self.datapipes = datapipes  # type: ignore[assignment]

    def __iter__(self) -> Iterator:
        # [FUNCTION DOC] Yields every item of each input datapipe in order.
        for dp in self.datapipes:
            yield from dp

    def __len__(self) -> int:
        # [FUNCTION DOC] Returns the sum of input lengths; raises TypeError if any input
        # has no length.
        # pyrefly: ignore [unsafe-overlap]
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            # pyrefly: ignore [bad-argument-type]
            return sum(len(dp) for dp in self.datapipes)
        else:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")


@functional_datapipe("fork")
class ForkerIterDataPipe(IterDataPipe):
    # [CLASS DOC] Factory-style datapipe (functional name fork): constructing it returns
    # not an instance of itself but either the input unchanged (one instance) or a list
    # of child datapipes sharing one buffered container.
    r"""
    Creates multiple instances of the same Iterable DataPipe (functional name: ``fork``).

    Args:
        datapipe: Iterable DataPipe being copied
        num_instances: number of instances of the datapipe to create
        buffer_size: this restricts how far ahead the leading child DataPipe
           can read relative to the slowest child DataPipe.
           Defaults to ``1000``. Use ``-1`` for the unlimited buffer.
        copy: copy strategy to use for items yielded by each branch. Supported
            options are ``None`` for no copying, ``"shallow"`` for shallow object
            copies, and ``"deep"`` for deep object copies. Defaults to ``None``.

    Note:
        All branches of the forked pipeline return the identical object unless
        the copy parameter is supplied. If the object is mutable or contains
        mutable objects, changing them in one branch will affect all others.

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper(range(5))
        >>> dp1, dp2 = source_dp.fork(num_instances=2)
        >>> list(dp1)
        [0, 1, 2, 3, 4]
        >>> list(dp2)
        [0, 1, 2, 3, 4]
    """

    def __new__(
        cls,
        datapipe: IterDataPipe,
        num_instances: int,
        buffer_size: int = 1000,
        copy: Literal["shallow", "deep"] | None = None,
    ):
        # [FUNCTION DOC] Raises ValueError if num_instances is below 1. Returns the input
        # datapipe itself when num_instances is 1; otherwise builds the shared container
        # (buffer_size limits how far the leading child may run ahead, negative means
        # unlimited; copy picks no copy, shallow or deep copy of yielded items) and
        # returns a list of num_instances child datapipes.
        if num_instances < 1:
            raise ValueError(
                f"Expected `num_instances` larger than 0, but {num_instances} is found"
            )
        if num_instances == 1:
            return datapipe
        container = _ForkerIterDataPipe(datapipe, num_instances, buffer_size, copy)  # type: ignore[abstract]
        return [_ChildDataPipe(container, i) for i in range(num_instances)]


class _ContainerTemplate(ABC):
    # [CLASS DOC] HELPER: Abstract interface that a container datapipe must implement for
    # its child datapipes: fetch the next element for an instance, report whether all
    # instances are exhausted, reset, and report a per-instance length.
    r"""Abstract class for container ``DataPipes``. The following are three required methods."""

    # [FUNCTION DOC] HELPER: Abstract; returns an iterator over the elements for the child
    # with the given instance_id. (Comment placed above the one-line definition.)
    @abstractmethod
    def get_next_element_by_instance(self, instance_id: int): ...

    # [FUNCTION DOC] HELPER: Abstract; returns True when all children finished iterating.
    @abstractmethod
    def is_every_instance_exhausted(self) -> bool: ...

    # [FUNCTION DOC] HELPER: Abstract; restores the container to its initial state.
    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def get_length_by_instance(self, instance_id: int):
        # [FUNCTION DOC] HELPER: Abstract; returns the length for the given child or raises
        # TypeError when no length is supported.
        r"""Raise TypeError if it's not supposed to be implemented to support `list(datapipe)`."""


def _no_op(x):
    # [FUNCTION DOC] HELPER: Identity function used as the default copy function.
    return x


class _ForkerIterDataPipe(IterDataPipe, _ContainerTemplate):
    # [CLASS DOC] HELPER: Container behind fork. Holds the source iterator and a shared
    # buffer deque of items not yet read by the slowest child. State: child_pointers
    # (per-child count of items consumed), slowest_ptr, leading_ptr, end_ptr (total length
    # once the source ends), _child_stop flags, buffer_size limit, copy_fn.
    r"""
    Container to hold instance-specific information on behalf of ForkerIterDataPipe.

    It tracks the state of its child DataPipes, maintains the buffer, and yields the next value
    as requested by the child DataPipes.
    """

    def __init__(
        self,
        datapipe: IterDataPipe,
        num_instances: int,
        buffer_size: int = 1000,
        copy: Literal["shallow", "deep"] | None = None,
    ) -> None:
        # [FUNCTION DOC] Initializes buffer and pointer state for num_instances children.
        # Warns (UserWarning) if buffer_size is negative (unlimited). Picks copy_fn: no-op
        # for None, shallow or deep copy; raises ValueError for any other copy value.
        # Does not call the base constructor.
        self.main_datapipe = datapipe
        self._datapipe_iterator: Iterator[Any] | None = None
        self.num_instances = num_instances
        self.buffer: deque = deque()
        self.buffer_size = buffer_size
        if self.buffer_size < 0:
            warnings.warn(
                "Unlimited buffer size is set for `fork`, "
                "please be aware of OOM at random places",
                UserWarning,
                stacklevel=2,
            )
        if copy is None:
            self.copy_fn = _no_op
        elif copy == "shallow":
            self.copy_fn = copymodule.copy
        elif copy == "deep":
            self.copy_fn = copymodule.deepcopy
        else:
            raise ValueError(
                f"Unknown copy method `{copy}` requested, choose one of None, `shallow` or `deep`."
            )

        self.child_pointers: list[int] = [
            0
        ] * num_instances  # Indicate the indices of the next element to get
        self.slowest_ptr = 0  # The index to read by the slowest child
        self.leading_ptr = 0  # The index to read by the fastest child
        self.end_ptr: int | None = None  # The index to stop child
        self._child_stop: list[bool] = [True for _ in range(num_instances)]

    def __len__(self) -> int:
        # [FUNCTION DOC] Returns the length of the source datapipe (TypeError if unsized).
        # pyrefly: ignore [bad-argument-type]
        return len(self.main_datapipe)

    def get_next_element_by_instance(self, instance_id: int):
        # [FUNCTION DOC] Generator for one child. If no iteration is active, starts a new
        # source iterator and marks all children as running. Each step advances that child's
        # pointer, serves the item from the shared buffer if a faster child already read
        # it, otherwise pulls from the source and buffers it; drops buffer entries once the
        # slowest child passes them. Yields copy_fn of each item. Raises BufferError when
        # the leading child gets more than buffer_size items ahead of the slowest one. On
        # exit it marks the child stopped and, when all stopped, clears the iterator and
        # closes streams still in the buffer.
        if self._datapipe_iterator is None and self._child_stop[instance_id]:
            self._datapipe_iterator = iter(self.main_datapipe)
            self._snapshot_state = _SnapshotState.Iterating
            for i in range(self.num_instances):
                self._child_stop[i] = False
        try:
            while not self._child_stop[instance_id]:
                self.child_pointers[instance_id] += 1
                if (
                    self.end_ptr is not None
                    and self.child_pointers[instance_id] == self.end_ptr
                ):
                    self._child_stop[instance_id] = True
                    break
                # Use buffer
                if self.buffer and self.child_pointers[instance_id] <= self.leading_ptr:
                    idx = self.child_pointers[instance_id] - self.slowest_ptr - 1
                    return_val = self.buffer[idx]
                else:  # Retrieve one element from main datapipe
                    self.leading_ptr = self.child_pointers[instance_id]
                    try:
                        return_val = next(self._datapipe_iterator)  # type: ignore[arg-type]
                        self.buffer.append(return_val)
                    except StopIteration:
                        self._child_stop[instance_id] = True
                        self._datapipe_iterator = None
                        self.end_ptr = self.leading_ptr
                        continue
                if self.child_pointers[instance_id] == self.slowest_ptr + 1:
                    new_min = min(
                        self.child_pointers
                    )  # Can optimize by avoiding the call to min()
                    if self.slowest_ptr < new_min:
                        self.slowest_ptr = new_min
                        self.buffer.popleft()
                if (
                    self.buffer_size >= 0
                    and self.leading_ptr > self.buffer_size + self.slowest_ptr
                ):
                    raise BufferError(
                        "ForkerIterDataPipe buffer overflow,"
                        + f"buffer size {self.buffer_size} is insufficient."
                    )

                yield self.copy_fn(return_val)  # type: ignore[possibly-undefined]
        finally:
            self._child_stop[instance_id] = True
            # Cleanup _datapipe_iterator for the case that fork exits earlier
            if all(self._child_stop):
                self._datapipe_iterator = None
                self._cleanup()

    def is_every_instance_exhausted(self) -> bool:
        # [FUNCTION DOC] True once the source has ended and every child has stopped.
        return self.end_ptr is not None and all(self._child_stop)

    def get_length_by_instance(self, instance_id: int) -> int:
        # [FUNCTION DOC] Returns the source length for any child (TypeError if unsized).
        # pyrefly: ignore [bad-argument-type]
        return len(self.main_datapipe)

    def reset(self) -> None:
        # [FUNCTION DOC] Drops the source iterator, empties the buffer (without closing
        # streams) and zeros all pointers and stop flags.
        self._datapipe_iterator = None
        self.buffer = deque()
        self.child_pointers = [0] * self.num_instances
        self.slowest_ptr = 0
        self.leading_ptr = 0
        self.end_ptr = None
        self._child_stop = [True for _ in range(self.num_instances)]

    def __getstate__(self):
        # [FUNCTION DOC] Pickle support: returns a tuple of source, instance count, buffer
        # size, copy function, iterator id and sample counter (buffer is not saved), passed
        # through IterDataPipe.getstate_hook when set.
        state = (
            self.main_datapipe,
            self.num_instances,
            self.buffer_size,
            self.copy_fn,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        # [FUNCTION DOC] Restores saved fields and resets buffer, pointers and stop flags.
        (
            self.main_datapipe,
            self.num_instances,
            self.buffer_size,
            self.copy_fn,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        self._datapipe_iterator = None
        self.buffer = deque()
        self.child_pointers = [0] * self.num_instances
        self.slowest_ptr = 0
        self.leading_ptr = 0
        self.end_ptr = None
        self._child_stop = [True for _ in range(self.num_instances)]

    def _cleanup(self) -> None:
        # [FUNCTION DOC] HELPER: Empties the buffer, closing any stream objects in items.
        while self.buffer:
            d = self.buffer.popleft()
            StreamWrapper.close_streams(d)

    def __del__(self) -> None:
        # [FUNCTION DOC] Finalizer: cleans up (closes) buffered streams.
        self._cleanup()


class _ChildDataPipe(IterDataPipe):
    # [CLASS DOC] HELPER: One branch of a fork or demux. Delegates iteration and length to
    # its container by instance_id, and coordinates the shared "valid iterator" generation:
    # starting a new iterator on one child invalidates older iterators on all children and
    # resets the container. State: main_datapipe (container), instance_id.
    r"""
    Iterable Datapipe that is a child of a main DataPipe.

    The instance of this class will pass its instance_id to get the next value from its main DataPipe.

    Note:
        ChildDataPipe, like all other IterDataPipe, follows the single iterator per IterDataPipe constraint.
        Since ChildDataPipes share a common buffer, when an iterator is created for one of the ChildDataPipes,
        the previous iterators  for all ChildDataPipes must be invalidated, with the exception when a ChildDataPipe
        hasn't had an iterator created from it since the last invalidation. See the example below.

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> # Single Iterator per IterDataPipe Invalidation
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper(range(10))
        >>> cdp1, cdp2 = source_dp.fork(num_instances=2)
        >>> it1, it2 = iter(cdp1), iter(cdp2)
        >>> it3 = iter(cdp1)
        >>> # The line above invalidates `it1` and `it2`, and resets `ForkerIterDataPipe`.
        >>> it4 = iter(cdp2)
        >>> # The line above doesn't invalidate `it3`, because an iterator for `cdp2` hasn't been created since
        >>> # the last invalidation.

    Args:
        main_datapipe: Main DataPipe with a method 'get_next_element_by_instance(instance_id)'
        instance_id: integer identifier of this instance
    """

    _is_child_datapipe: bool = True

    def __init__(self, main_datapipe: IterDataPipe, instance_id: int) -> None:
        # [FUNCTION DOC] Stores the container and this child's id. Raises AssertionError
        # if the container does not implement _ContainerTemplate. Does not call the base
        # constructor.
        if not isinstance(main_datapipe, _ContainerTemplate):
            raise AssertionError("main_datapipe must implement _ContainerTemplate")

        self.main_datapipe: IterDataPipe = main_datapipe
        self.instance_id = instance_id

    def __iter__(self):
        # [FUNCTION DOC] Returns the container's generator for this child's id.
        # Note that the logic behind setting iterator ID and `reset` are handled within `hook_iterator`
        # We want to separate the code for reset and yield, so that 'reset' executes before __next__ is called
        return self.main_datapipe.get_next_element_by_instance(self.instance_id)

    def __len__(self) -> int:
        # [FUNCTION DOC] Returns the container's length for this child (may raise TypeError).
        return self.main_datapipe.get_length_by_instance(self.instance_id)

    # This method is called by `hook_iterator` in `_typing.py`.
    def _set_main_datapipe_valid_iterator_id(self) -> int:
        # [FUNCTION DOC] HELPER: Called when a new iterator is created. If the container has
        # no id yet, sets it to 0. If this child is in the same generation as the
        # container, increments the container id, warns (UserWarning) if not all children
        # were exhausted, and resets the container. Finally adopts the container id and
        # returns it.
        r"""
        Update the valid iterator ID for both this DataPipe object and `main_datapipe`.

        `main_datapipe.reset()` is called when the ID is incremented to a new generation.
        """
        # 1. First time any child iterator is created
        if self.main_datapipe._valid_iterator_id is None:
            self.main_datapipe._valid_iterator_id = 0  # type: ignore[attr-defined]
        # 2. This instance was already in the same generation as `main_datapipe`,
        #    we need to increment the ID further by 1
        elif self.main_datapipe._valid_iterator_id == self._valid_iterator_id:  # type: ignore[has-type]
            self.main_datapipe._valid_iterator_id += 1  # type: ignore[attr-defined]
            # Whenever a new generation of iterator is created, the `main_datapipe` must reset
            if not self.main_datapipe.is_every_instance_exhausted():
                warnings.warn(
                    "Some child DataPipes are not exhausted when __iter__ is called. We are resetting "
                    "the buffer and each child DataPipe will read from the start again.",
                    UserWarning,
                    stacklevel=2,
                )
            self.main_datapipe.reset()
        # 3. Otherwise, the iterator is behind the others, so it will just need to catch up by setting
        #    the instance's iterator to match that of `main_datapipe`
        self._valid_iterator_id = self.main_datapipe._valid_iterator_id
        return self._valid_iterator_id

    # This method is called by `hook_iterator` in `_typing.py`.
    def _check_valid_iterator_id(self, iterator_id) -> bool:
        # [FUNCTION DOC] HELPER: True only if iterator_id equals both this child's and the
        # container's current valid id.
        r"""Check the valid iterator ID against that of DataPipe object and that of `main_datapipe`."""
        return (
            iterator_id == self._valid_iterator_id
            and iterator_id == self.main_datapipe._valid_iterator_id
        )


@functional_datapipe("demux")
class DemultiplexerIterDataPipe(IterDataPipe):
    # [CLASS DOC] Factory-style datapipe (functional name demux): constructing it returns a
    # list of child datapipes, each yielding the items the classifier assigns to it.
    r"""
    Splits the input DataPipe into multiple child DataPipes, using the given classification function (functional name: ``demux``).

    A list of the child DataPipes is returned from this operation.

    Args:
        datapipe: Iterable DataPipe being filtered
        num_instances: number of instances of the DataPipe to create
        classifier_fn: a function that maps values to an integer within the range ``[0, num_instances - 1]`` or ``None``
        drop_none: defaults to ``False``, if ``True``, the function will skip over elements classified as ``None``
        buffer_size: this defines the maximum number of inputs that the buffer can hold across all child
            DataPipes while waiting for their values to be yielded.
            Defaults to ``1000``. Use ``-1`` for the unlimited buffer.

    Examples:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> def odd_or_even(n):
        ...     return n % 2
        >>> source_dp = IterableWrapper(range(5))
        >>> dp1, dp2 = source_dp.demux(num_instances=2, classifier_fn=odd_or_even)
        >>> list(dp1)
        [0, 2, 4]
        >>> list(dp2)
        [1, 3]
        >>> # It can also filter out any element that gets `None` from the `classifier_fn`
        >>> def odd_or_even_no_zero(n):
        ...     return n % 2 if n != 0 else None
        >>> dp1, dp2 = source_dp.demux(
        ...     num_instances=2, classifier_fn=odd_or_even_no_zero, drop_none=True
        ... )
        >>> list(dp1)
        [2, 4]
        >>> list(dp2)
        [1, 3]
    """

    def __new__(
        cls,
        datapipe: IterDataPipe,
        num_instances: int,
        classifier_fn: Callable[[_T_co], int | None],
        drop_none: bool = False,
        buffer_size: int = 1000,
    ):
        # [FUNCTION DOC] Raises ValueError if num_instances is below 1; checks that the
        # classifier is picklable (via a shared utility). Builds the shared container and
        # returns a list of num_instances child datapipes. Items classified as None are
        # dropped only when drop_none is set.
        if num_instances < 1:
            raise ValueError(
                f"Expected `num_instances` larger than 0, but {num_instances} is found"
            )

        _check_unpickable_fn(classifier_fn)

        # When num_instances == 1, demux can be replaced by filter,
        # but keep it as Demultiplexer for the sake of consistency
        # like throwing Error when classification result is out of range
        container = _DemultiplexerIterDataPipe(
            datapipe, num_instances, classifier_fn, drop_none, buffer_size
        )  # type: ignore[abstract]
        return [_ChildDataPipe(container, i) for i in range(num_instances)]


class _DemultiplexerIterDataPipe(IterDataPipe, _ContainerTemplate):
    # [CLASS DOC] HELPER: Container behind demux. Holds the source iterator, one buffer
    # deque per child for items read on behalf of other children, current_buffer_usage
    # across all buffers, classifier_fn, drop_none, buffer_size, and stop/exhausted flags.
    r"""
    Container to hold instance-specific information on behalf of DemultiplexerIterDataPipe.

    It tracks the state of its child DataPipes, maintains the buffer, classifies and yields the next correct value
    as requested by the child DataPipes.
    """

    def __init__(
        self,
        datapipe: IterDataPipe[_T_co],
        num_instances: int,
        classifier_fn: Callable[[_T_co], int | None],
        drop_none: bool,
        buffer_size: int,
    ) -> None:
        # [FUNCTION DOC] Stores configuration and creates empty per-child buffers. Warns
        # (UserWarning) when buffer_size is negative (unlimited). Does not call the base
        # constructor.
        # pyrefly: ignore [invalid-type-var]
        self.main_datapipe = datapipe
        self._datapipe_iterator: Iterator[Any] | None = None
        self.num_instances = num_instances
        self.buffer_size = buffer_size
        if self.buffer_size < 0:
            warnings.warn(
                "Unlimited buffer size is set for `demux`, "
                "please be aware of OOM at random places",
                UserWarning,
                stacklevel=2,
            )
        self.current_buffer_usage = 0
        # pyrefly: ignore [invalid-type-var]
        self.child_buffers: list[deque[_T_co]] = [deque() for _ in range(num_instances)]
        # pyrefly: ignore [invalid-type-var]
        self.classifier_fn = classifier_fn
        self.drop_none = drop_none
        self.main_datapipe_exhausted = False
        self._child_stop: list[bool] = [True for _ in range(num_instances)]

    def _find_next(self, instance_id: int) -> _T_co:  # type: ignore[type-var]
        # [FUNCTION DOC] HELPER: Reads source items until one is classified for
        # instance_id and returns it; items for other children are appended to their
        # buffers. Raises StopIteration when the source is exhausted or the child stopped,
        # ValueError if the iterator is unset or the classification is None (without
        # drop_none) or outside the valid range, BufferError if total buffered items exceed
        # buffer_size. Dropped None items have their streams closed.
        while True:
            if self.main_datapipe_exhausted or self._child_stop[instance_id]:
                raise StopIteration
            if self._datapipe_iterator is None:
                raise ValueError(
                    "_datapipe_iterator has not been set, likely because this private method is called directly "
                    "without invoking get_next_element_by_instance() first."
                )
            value = next(self._datapipe_iterator)
            classification = self.classifier_fn(value)
            if classification is None and self.drop_none:
                StreamWrapper.close_streams(value)
                continue
            if (
                classification is None
                or classification >= self.num_instances
                or classification < 0
            ):
                raise ValueError(
                    f"Output of the classification fn should be between 0 and {self.num_instances - 1}. "
                    + f"{classification} is returned."
                )
            if classification == instance_id:
                return value
            self.child_buffers[classification].append(value)
            self.current_buffer_usage += 1
            if self.buffer_size >= 0 and self.current_buffer_usage > self.buffer_size:
                raise BufferError(
                    f"DemultiplexerIterDataPipe buffer overflow, buffer size {self.buffer_size} is insufficient."
                )

    def get_next_element_by_instance(self, instance_id: int):
        # [FUNCTION DOC] Generator for one child. Starts a source iterator if none is active
        # and marks children as running. Yields buffered items for this child first, then
        # items found by _find_next. When the source is exhausted marks the child stopped.
        # On exit marks the child stopped, clears the iterator when all stopped, and
        # discards (closing streams) this child's remaining buffer.
        if self._datapipe_iterator is None and self._child_stop[instance_id]:
            self._datapipe_iterator = iter(self.main_datapipe)
            self._snapshot_state = (
                _SnapshotState.Iterating
            )  # This is necessary for the DataPipe to reset properly.
            self.main_datapipe_exhausted = False
            for i in range(self.num_instances):
                self._child_stop[i] = False

        try:
            while not self._child_stop[instance_id]:
                if self.child_buffers[instance_id]:
                    self.current_buffer_usage -= 1
                    yield self.child_buffers[instance_id].popleft()
                else:
                    try:
                        yield self._find_next(instance_id)
                    except StopIteration:
                        self._child_stop[instance_id] = True
                        self.main_datapipe_exhausted = True
                        self._datapipe_iterator = None
        finally:
            self._child_stop[instance_id] = True
            # Cleanup _datapipe_iterator for the case that demux exits earlier
            if all(self._child_stop):
                self._datapipe_iterator = None
            if self.child_buffers[instance_id]:
                self._cleanup(instance_id)

    def is_every_instance_exhausted(self) -> bool:
        # [FUNCTION DOC] True once the source is exhausted and every child has stopped.
        return self.main_datapipe_exhausted and all(self._child_stop)

    def get_length_by_instance(self, instance_id: int) -> int:
        # [FUNCTION DOC] Always raises TypeError; child lengths are unknown up front.
        raise TypeError

    def reset(self) -> None:
        # [FUNCTION DOC] Drops the source iterator, zeros buffer usage, replaces buffers
        # with empty ones (no stream closing), sets all children stopped and clears the
        # exhausted flag.
        self._datapipe_iterator = None
        self.current_buffer_usage = 0
        self.child_buffers = [deque() for _ in range(self.num_instances)]
        self._child_stop = [True for _ in range(self.num_instances)]
        self.main_datapipe_exhausted = False

    def __getstate__(self):
        # [FUNCTION DOC] Pickle support: returns a tuple of source, instance count, buffer
        # size, classifier, drop_none, iterator id and sample counter (buffers not saved),
        # passed through IterDataPipe.getstate_hook when set.
        state = (
            self.main_datapipe,
            self.num_instances,
            self.buffer_size,
            self.classifier_fn,
            self.drop_none,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        # [FUNCTION DOC] Restores saved fields and resets iterator, buffers and flags.
        (
            self.main_datapipe,
            self.num_instances,
            self.buffer_size,
            self.classifier_fn,
            self.drop_none,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        self._datapipe_iterator = None
        self.current_buffer_usage = 0
        self.child_buffers = [deque() for _ in range(self.num_instances)]
        self._child_stop = [True for _ in range(self.num_instances)]
        self.main_datapipe_exhausted = False

    def _cleanup(self, instance_id: int | None = None) -> None:
        # [FUNCTION DOC] HELPER: Empties the buffer of one child (or of all when
        # instance_id is None), closing any stream objects in discarded items.
        ids = (
            range(self.num_instances)
            if instance_id is None
            else [
                instance_id,
            ]
        )
        for i in ids:
            q = self.child_buffers[i]
            while q:
                d = q.popleft()
                StreamWrapper.close_streams(d)

    def __del__(self) -> None:
        # [FUNCTION DOC] Finalizer: cleans up (closes) all buffered streams.
        self._cleanup()


@functional_datapipe("mux")
class MultiplexerIterDataPipe(IterDataPipe):
    # [CLASS DOC] Round-robin interleaving of several datapipes, one item from each in
    # turn, ending when the shortest is exhausted. State: datapipes tuple and a small
    # buffer holding the items of the current round.
    r"""
    Yields one element at a time from each of the input Iterable DataPipes (functional name: ``mux``).

    As in, one element from the 1st input DataPipe, then one element from the 2nd DataPipe in the next iteration,
    and so on. It ends when the shortest input DataPipe is exhausted.

    Args:
        datapipes: Iterable DataPipes that will take turn to yield their elements, until the shortest DataPipe is exhausted

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1, dp2, dp3 = (
        ...     IterableWrapper(range(3)),
        ...     IterableWrapper(range(10, 15)),
        ...     IterableWrapper(range(20, 25)),
        ... )
        >>> list(dp1.mux(dp2, dp3))
        [0, 10, 20, 1, 11, 21, 2, 12, 22]
    """

    def __init__(self, *datapipes) -> None:
        # [FUNCTION DOC] Stores the input datapipes and an empty round buffer. No type
        # validation is done and the base constructor is not called.
        self.datapipes = datapipes
        self.buffer: list = []  # Store values to be yielded only when every iterator provides one

    def __iter__(self):
        # [FUNCTION DOC] Each round takes one item from every input iterator into the
        # buffer and, if all succeeded, yields them in input order. If any input is
        # exhausted mid-round the partial round is discarded and iteration ends.
        iterators = [iter(x) for x in self.datapipes]
        while iterators:
            for it in iterators:
                try:
                    value = next(it)
                    self.buffer.append(value)
                except StopIteration:
                    self.buffer.clear()
                    return
            yield from self.buffer
            self.buffer.clear()

    def __len__(self) -> int:
        # [FUNCTION DOC] Returns the shortest input length times the number of inputs;
        # raises TypeError if any input has no length.
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            return min(len(dp) for dp in self.datapipes) * len(self.datapipes)
        else:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")

    def reset(self) -> None:
        # [FUNCTION DOC] Empties the round buffer.
        self.buffer = []

    def __getstate__(self):
        # [FUNCTION DOC] Pickle support: returns a tuple of inputs, iterator id and sample
        # counter, passed through IterDataPipe.getstate_hook when set.
        state = (
            self.datapipes,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        # [FUNCTION DOC] Restores saved fields and starts with an empty round buffer.
        (
            self.datapipes,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        self.buffer = []

    def __del__(self) -> None:
        # [FUNCTION DOC] Finalizer: empties the round buffer.
        self.buffer.clear()


@functional_datapipe("zip")
class ZipperIterDataPipe(IterDataPipe[tuple[_T_co]]):
    # [CLASS DOC] Yields tuples of one item from each input datapipe, stopping when the
    # shortest input ends. State: datapipes tuple. Sized only if every input is sized.
    r"""
    Aggregates elements into a tuple from each of the input DataPipes (functional name: ``zip``).

    The output is stopped as soon as the shortest input DataPipe is exhausted.

    Args:
        *datapipes: Iterable DataPipes being aggregated

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1, dp2, dp3 = (
        ...     IterableWrapper(range(5)),
        ...     IterableWrapper(range(10, 15)),
        ...     IterableWrapper(range(20, 25)),
        ... )
        >>> list(dp1.zip(dp2, dp3))
        [(0, 10, 20), (1, 11, 21), (2, 12, 22), (3, 13, 23), (4, 14, 24)]
    """

    datapipes: tuple[IterDataPipe]

    def __init__(self, *datapipes: IterDataPipe) -> None:
        # [FUNCTION DOC] Stores the input datapipes. Raises TypeError if any is not an
        # IterDataPipe. Zero inputs is accepted (and yields nothing).
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError(
                "All inputs are required to be `IterDataPipe` for `ZipIterDataPipe`."
            )
        super().__init__()
        self.datapipes = datapipes  # type: ignore[assignment]

    def __iter__(self) -> Iterator[tuple[_T_co]]:
        # [FUNCTION DOC] Creates an iterator per input and yields tuples of their items
        # until the shortest is exhausted (non-strict zip).
        iterators = [iter(datapipe) for datapipe in self.datapipes]
        yield from zip(*iterators, strict=False)

    def __len__(self) -> int:
        # [FUNCTION DOC] Returns the shortest input length; raises TypeError if any input
        # has no length (and ValueError from min for zero inputs).
        # pyrefly: ignore [unsafe-overlap]
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            # pyrefly: ignore [bad-argument-type]
            return min(len(dp) for dp in self.datapipes)
        else:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
