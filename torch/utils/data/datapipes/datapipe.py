# [MODULE DOC] Core datapipe base classes: IterDataPipe (iterable style) and MapDataPipe
# (map style) with a registry of functional-form methods and serialization/representation
# hooks, DataChunk (list-like batch), DFIterDataPipe (marker base for DataFrame pipes), and
# internal serialization wrappers that fall back to dill for unpicklable pipes. Import
# attempts to import dill and records its availability in HAS_DILL.
import functools
import pickle
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

from torch.utils._import_utils import import_dill
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes._typing import _DataPipeMeta, _IterDataPipeMeta
from torch.utils.data.datapipes.utils.common import (
    _deprecation_warning,
    _iter_deprecated_functional_names,
    _map_deprecated_functional_names,
)
from torch.utils.data.dataset import Dataset, IterableDataset


dill = import_dill()
HAS_DILL = dill is not None

__all__ = [
    "DataChunk",
    "DFIterDataPipe",
    "IterDataPipe",
    "MapDataPipe",
]


_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)

UNTRACABLE_DATAFRAME_PIPES = [
    "batch",  # As it returns DataChunks
    "groupby",  # As it returns DataChunks
    "_dataframes_as_tuples",  # As it unpacks DF
    "trace_as_dataframe",  # As it used to mark DF for tracing
]


class DataChunk(list[_T]):
    # [CLASS DOC] A list subclass used as a batch of samples. The same items are kept in the
    # list itself and in the items attribute, so subclasses can change how iteration works
    # (see DataChunkDF) while raw_iterator still gives the stored items.
    def __init__(self, items: Iterable[_T]) -> None:
        # [FUNCTION DOC] Copies the given iterable into a list, uses it as the list content,
        # and keeps a reference to that list copy as the items attribute.
        items = list(items)
        super().__init__(items)
        self.items = items

    def as_str(self, indent: str = "") -> str:
        # [FUNCTION DOC] Returns the string form of the chunk: items joined by commas inside
        # square brackets, with indent prepended.
        return indent + "[" + ", ".join(str(i) for i in iter(self)) + "]"

    def __iter__(self) -> Iterator[_T]:
        # [FUNCTION DOC] Yields the list elements in order (overridable by subclasses).
        yield from super().__iter__()

    def raw_iterator(self) -> Iterator[_T]:
        # [FUNCTION DOC] Yields the elements stored in the items attribute, bypassing any
        # subclass iteration override.
        yield from self.items


class IterDataPipe(IterableDataset[_T_co], metaclass=_IterDataPipeMeta):
    # [CLASS DOC] Base class of iterable-style datapipes. Class-level state: functions (name
    # to callable registry backing the functional form, looked up in __getattr__),
    # reduce_ex_hook, getstate_hook, str_hook, repr_hook (global overrides), and per-instance
    # iteration bookkeeping (_valid_iterator_id, _number_of_samples_yielded,
    # _snapshot_state, _fast_forward_iterator) managed by the metaclass hooks. Subclasses
    # override the iteration method and optionally reset.
    r"""
    Iterable-style DataPipe.

    All DataPipes that represent an iterable of data samples should subclass this.
    This style of DataPipes is particularly useful when data come from a stream, or
    when the number of samples is too large to fit them all in memory. ``IterDataPipe`` is lazily initialized and its
    elements are computed only when ``next()`` is called on the iterator of an ``IterDataPipe``.

    All subclasses should overwrite :meth:`__iter__`, which would return an
    iterator of samples in this DataPipe. Calling ``__iter__`` of an ``IterDataPipe`` automatically invokes its
    method ``reset()``, which by default performs no operation. When writing a custom ``IterDataPipe``, users should
    override ``reset()`` if necessary. The common usages include resetting buffers, pointers,
    and various state variables within the custom ``IterDataPipe``.

    Note:
        Only `one` iterator can be valid for each ``IterDataPipe`` at a time,
        and the creation a second iterator will invalidate the first one. This constraint is necessary because
        some ``IterDataPipe`` have internal buffers, whose states can become invalid if there are multiple iterators.
        The code example below presents details on how this constraint looks in practice.
        If you have any feedback related to this constraint, please see `GitHub IterDataPipe Single Iterator Issue`_.

    These DataPipes can be invoked in two ways, using the class constructor or applying their
    functional form onto an existing ``IterDataPipe`` (recommended, available to most but not all DataPipes).
    You can chain multiple `IterDataPipe` together to form a pipeline that will perform multiple
    operations in succession.

    .. _GitHub IterDataPipe Single Iterator Issue:
        https://github.com/pytorch/data/issues/45

    Note:
        When a subclass is used with :class:`~torch.utils.data.DataLoader`, each
        item in the DataPipe will be yielded from the :class:`~torch.utils.data.DataLoader`
        iterator. When :attr:`num_workers > 0`, each worker process will have a
        different copy of the DataPipe object, so it is often desired to configure
        each copy independently to avoid having duplicate data returned from the
        workers. :func:`~torch.utils.data.get_worker_info`, when called in a worker
        process, returns information about the worker. It can be used in either the
        dataset's :meth:`__iter__` method or the :class:`~torch.utils.data.DataLoader` 's
        :attr:`worker_init_fn` option to modify each copy's behavior.

    Examples:
        General Usage:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import IterableWrapper, Mapper
            >>> dp = IterableWrapper(range(10))
            >>> map_dp_1 = Mapper(dp, lambda x: x + 1)  # Using class constructor
            >>> map_dp_2 = dp.map(
            ...     lambda x: x + 1
            ... )  # Using functional form (recommended)
            >>> list(map_dp_1)
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            >>> list(map_dp_2)
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            >>> filter_dp = map_dp_1.filter(lambda x: x % 2 == 0)
            >>> list(filter_dp)
            [2, 4, 6, 8, 10]
        Single Iterator Constraint Example:
            >>> from torchdata.datapipes.iter import IterableWrapper, Mapper
            >>> source_dp = IterableWrapper(range(10))
            >>> it1 = iter(source_dp)
            >>> list(it1)
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            >>> it1 = iter(source_dp)
            >>> it2 = iter(
            ...     source_dp
            ... )  # The creation of a new iterator invalidates `it1`
            >>> next(it2)
            0
            >>> next(it1)  # Further usage of `it1` will raise a `RunTimeError`
    """

    functions: dict[str, Callable] = {}
    reduce_ex_hook: Callable | None = None
    getstate_hook: Callable | None = None
    str_hook: Callable | None = None
    repr_hook: Callable | None = None
    _valid_iterator_id: int | None = None
    _number_of_samples_yielded: int = 0
    _snapshot_state: _SnapshotState = _SnapshotState.NotStarted
    _fast_forward_iterator: Iterator | None = None

    def __iter__(self) -> Iterator[_T_co]:
        # [FUNCTION DOC] Default iteration: returns the datapipe itself as the iterator.
        # Subclasses are expected to override this.
        # pyrefly: ignore [bad-return]
        return self

    def __getattr__(self, attribute_name):
        # [FUNCTION DOC] Resolves functional-form names. If attribute_name is registered,
        # emits a deprecation warning when it is on the deprecated list, and returns the
        # registered function bound to this instance as its first argument (keeping the
        # docstring). Otherwise raises AttributeError.
        # KNOWN ISSUE: the error message is missing the closing quote after the attribute
        # name.
        if attribute_name in IterDataPipe.functions:
            if attribute_name in _iter_deprecated_functional_names:
                kwargs = _iter_deprecated_functional_names[attribute_name]
                _deprecation_warning(**kwargs)
            f = IterDataPipe.functions[attribute_name]
            function = functools.partial(f, self)
            functools.update_wrapper(wrapper=function, wrapped=f, assigned=("__doc__",))
            return function
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{attribute_name}"
            )

    @classmethod
    def register_function(cls, function_name, function) -> None:
        # [FUNCTION DOC] Adds or overwrites a callable in the functional-form registry under
        # function_name (global side effect on the class registry).
        cls.functions[function_name] = function

    @classmethod
    def register_datapipe_as_function(
        cls, function_name, cls_to_register, enable_df_api_tracing=False
    ) -> None:
        # [FUNCTION DOC] Registers a datapipe class under a functional name. Raises a generic
        # Exception if the name is already taken. The registered callable constructs the
        # class from the source datapipe plus extra arguments; if the result is an
        # IterDataPipe and tracing is enabled (or the source is a DataFrame pipe) and the
        # name is not in the untraceable list, the result is wrapped by trace_as_dataframe.
        # The callable copies the class docstring.
        if function_name in cls.functions:
            raise Exception(  # noqa: TRY002
                f"Unable to add DataPipe function name {function_name} as it is already taken"
            )

        def class_function(cls, enable_df_api_tracing, source_dp, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Builds cls(source_dp, ...) and optionally wraps an
            # IterDataPipe result for DataFrame tracing as described in the registrar.
            result_pipe = cls(source_dp, *args, **kwargs)
            if isinstance(result_pipe, IterDataPipe):
                if enable_df_api_tracing or isinstance(source_dp, DFIterDataPipe):
                    if function_name not in UNTRACABLE_DATAFRAME_PIPES:
                        result_pipe = result_pipe.trace_as_dataframe()

            return result_pipe

        function = functools.partial(
            class_function, cls_to_register, enable_df_api_tracing
        )
        functools.update_wrapper(
            wrapper=function, wrapped=cls_to_register, assigned=("__doc__",)
        )
        cls.functions[function_name] = function

    def __getstate__(self):
        # [FUNCTION DOC] Pickle support: returns the instance attribute mapping itself (not
        # a copy), passed through the class-wide getstate_hook if one is set.
        """
        Serialize `lambda` functions when `dill` is available.

        If this doesn't cover your custom DataPipe's use case, consider writing custom methods for
        `__getstate__` and `__setstate__`, or use `pickle.dumps` for serialization.
        """
        state = self.__dict__
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __reduce_ex__(self, *args, **kwargs):
        # [FUNCTION DOC] Pickle support: uses reduce_ex_hook(self) if set, falling back to the
        # default behavior if the hook raises NotImplementedError.
        if IterDataPipe.reduce_ex_hook is not None:
            try:
                return IterDataPipe.reduce_ex_hook(self)
            except NotImplementedError:
                pass
        return super().__reduce_ex__(*args, **kwargs)

    @classmethod
    def set_getstate_hook(cls, hook_fn) -> None:
        # [FUNCTION DOC] Installs the global getstate hook; raises RuntimeError when setting
        # a hook while one is already installed (pass None to clear).
        if IterDataPipe.getstate_hook is not None and hook_fn is not None:
            raise RuntimeError("Attempt to override existing getstate_hook")
        IterDataPipe.getstate_hook = hook_fn

    @classmethod
    def set_reduce_ex_hook(cls, hook_fn) -> None:
        # [FUNCTION DOC] Installs the global reduce-ex hook; raises RuntimeError when setting
        # a hook while one is already installed (pass None to clear).
        if IterDataPipe.reduce_ex_hook is not None and hook_fn is not None:
            raise RuntimeError("Attempt to override existing reduce_ex_hook")
        IterDataPipe.reduce_ex_hook = hook_fn

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns repr_hook(self) if a hook is set, else the qualified class
        # name.
        if self.repr_hook is not None:
            return self.repr_hook(self)
        # Instead of showing <torch. ... .MapperIterDataPipe object at 0x.....>, return the class name
        return str(self.__class__.__qualname__)

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns str_hook(self) if a hook is set, else the qualified class
        # name.
        if self.str_hook is not None:
            return self.str_hook(self)
        # Instead of showing <torch. ... .MapperIterDataPipe object at 0x.....>, return the class name
        return str(self.__class__.__qualname__)

    def __dir__(self):
        # [FUNCTION DOC] Returns the normal attribute names plus all registered functional
        # names, to support autocompletion.
        # for auto-completion in a REPL (e.g. Jupyter notebook)
        return list(super().__dir__()) + list(self.functions.keys())

    def reset(self) -> None:
        # [FUNCTION DOC] Hook to restore the datapipe to its initial state; the default does
        # nothing. Called automatically on each new iteration by the iterator hook.
        r"""
        Reset the `IterDataPipe` to the initial state.

        By default, no-op. For subclasses of `IterDataPipe`, depending on their functionalities,
        they may want to override this method with implementations that
        may clear the buffers and reset pointers of the DataPipe.
        The `reset` method is always called when `__iter__` is called as part of `hook_iterator`.
        """


class DFIterDataPipe(IterDataPipe):
    # [CLASS DOC] Marker base class for iterable datapipes that carry DataFrames; used to
    # decide whether functional results get traced.
    def _is_dfpipe(self) -> bool:
        # [FUNCTION DOC] HELPER: Always returns True, identifying a DataFrame pipe.
        return True


class MapDataPipe(Dataset[_T_co], metaclass=_DataPipeMeta):
    # [CLASS DOC] Base class of map-style datapipes (key to sample). Mirrors IterDataPipe's
    # functional-form registry (functions) and the global hooks (reduce_ex_hook,
    # getstate_hook, str_hook, repr_hook), but has no iteration bookkeeping. Subclasses
    # override the item access method and optionally length.
    r"""
    Map-style DataPipe.

    All datasets that represent a map from keys to data samples should subclass this.
    Subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given, unique key. Subclasses can also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.

    These DataPipes can be invoked in two ways, using the class constructor or applying their
    functional form onto an existing `MapDataPipe` (recommend, available to most but not all DataPipes).

    Note:
        :class:`~torch.utils.data.DataLoader` by default constructs an index
        sampler that yields integral indices. To make it work with a map-style
        DataPipe with non-integral indices/keys, a custom sampler must be provided.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper, Mapper
        >>> dp = SequenceWrapper(range(10))
        >>> map_dp_1 = dp.map(lambda x: x + 1)  # Using functional form (recommended)
        >>> list(map_dp_1)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> map_dp_2 = Mapper(dp, lambda x: x + 1)  # Using class constructor
        >>> list(map_dp_2)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> batch_dp = map_dp_1.batch(batch_size=2)
        >>> list(batch_dp)
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    """

    functions: dict[str, Callable] = {}
    reduce_ex_hook: Callable | None = None
    getstate_hook: Callable | None = None
    str_hook: Callable | None = None
    repr_hook: Callable | None = None

    def __getattr__(self, attribute_name):
        # [FUNCTION DOC] Resolves functional-form names from the map registry: warns if the
        # name is deprecated, and returns the registered function bound to this instance.
        # Otherwise raises AttributeError.
        # KNOWN ISSUE: the error message is missing the closing quote after the attribute
        # name.
        if attribute_name in MapDataPipe.functions:
            if attribute_name in _map_deprecated_functional_names:
                kwargs = _map_deprecated_functional_names[attribute_name]
                _deprecation_warning(**kwargs)
            f = MapDataPipe.functions[attribute_name]
            function = functools.partial(f, self)
            functools.update_wrapper(wrapper=function, wrapped=f, assigned=("__doc__",))
            return function
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{attribute_name}"
            )

    @classmethod
    def register_function(cls, function_name, function) -> None:
        # [FUNCTION DOC] Adds or overwrites a callable in the map registry under function_name
        # (global side effect).
        cls.functions[function_name] = function

    @classmethod
    def register_datapipe_as_function(cls, function_name, cls_to_register) -> None:
        # [FUNCTION DOC] Registers a datapipe class under a functional name; raises a generic
        # Exception if the name is taken. The registered callable constructs the class from
        # the source datapipe plus extra arguments and copies the class docstring.
        if function_name in cls.functions:
            raise Exception(  # noqa: TRY002
                f"Unable to add DataPipe function name {function_name} as it is already taken"
            )

        def class_function(cls, source_dp, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Returns cls(source_dp, ...).
            result_pipe = cls(source_dp, *args, **kwargs)
            return result_pipe

        function = functools.partial(class_function, cls_to_register)
        functools.update_wrapper(
            wrapper=function, wrapped=cls_to_register, assigned=("__doc__",)
        )
        cls.functions[function_name] = function

    def __getstate__(self):
        # [FUNCTION DOC] Pickle support: returns the instance attribute mapping itself (not
        # a copy), passed through the class-wide getstate_hook if one is set.
        """
        Serialize `lambda` functions when `dill` is available.

        If this doesn't cover your custom DataPipe's use case, consider writing custom methods for
        `__getstate__` and `__setstate__`, or use `pickle.dumps` for serialization.
        """
        state = self.__dict__
        if MapDataPipe.getstate_hook is not None:
            return MapDataPipe.getstate_hook(state)
        return state

    def __reduce_ex__(self, *args, **kwargs):
        # [FUNCTION DOC] Pickle support: uses reduce_ex_hook(self) if set, falling back to the
        # default behavior if the hook raises NotImplementedError.
        if MapDataPipe.reduce_ex_hook is not None:
            try:
                return MapDataPipe.reduce_ex_hook(self)
            except NotImplementedError:
                pass
        return super().__reduce_ex__(*args, **kwargs)

    @classmethod
    def set_getstate_hook(cls, hook_fn) -> None:
        # [FUNCTION DOC] Installs the global getstate hook; raises RuntimeError when setting
        # a hook while one is already installed (pass None to clear).
        if MapDataPipe.getstate_hook is not None and hook_fn is not None:
            raise RuntimeError("Attempt to override existing getstate_hook")
        MapDataPipe.getstate_hook = hook_fn

    @classmethod
    def set_reduce_ex_hook(cls, hook_fn) -> None:
        # [FUNCTION DOC] Installs the global reduce-ex hook; raises RuntimeError when setting
        # a hook while one is already installed (pass None to clear).
        if MapDataPipe.reduce_ex_hook is not None and hook_fn is not None:
            raise RuntimeError("Attempt to override existing reduce_ex_hook")
        MapDataPipe.reduce_ex_hook = hook_fn

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns repr_hook(self) if a hook is set, else the qualified class
        # name.
        if self.repr_hook is not None:
            return self.repr_hook(self)
        # Instead of showing <torch. ... .MapperMapDataPipe object at 0x.....>, return the class name
        return str(self.__class__.__qualname__)

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns str_hook(self) if a hook is set, else the qualified class
        # name.
        if self.str_hook is not None:
            return self.str_hook(self)
        # Instead of showing <torch. ... .MapperMapDataPipe object at 0x.....>, return the class name
        return str(self.__class__.__qualname__)

    def __dir__(self):
        # [FUNCTION DOC] Returns the normal attribute names plus all registered functional
        # names, to support autocompletion.
        # for auto-completion in a REPL (e.g. Jupyter notebook)
        return list(super().__dir__()) + list(self.functions.keys())


class _DataPipeSerializationWrapper:
    # [CLASS DOC] HELPER: Mixin wrapping a datapipe so it can be pickled even when plain
    # pickle fails, by falling back to dill. State: _datapipe.
    def __init__(self, datapipe) -> None:
        # [FUNCTION DOC] Stores the wrapped datapipe.
        self._datapipe = datapipe

    def __getstate__(self):
        # [FUNCTION DOC] Serializes the wrapped datapipe with pickle; if that fails and dill
        # is available, uses dill instead (otherwise re-raises). Returns a pair of the
        # serialized bytes and a flag telling whether dill was used.
        use_dill = False
        try:
            value = pickle.dumps(self._datapipe)
        except Exception:
            if HAS_DILL:
                # pyrefly: ignore [missing-attribute]
                value = dill.dumps(self._datapipe)
                use_dill = True
            else:
                raise
        return (value, use_dill)

    def __setstate__(self, state):
        # [FUNCTION DOC] Restores the wrapped datapipe from the (bytes, used-dill flag) pair,
        # using the matching loader.
        value, use_dill = state
        if use_dill:
            # pyrefly: ignore [missing-attribute]
            self._datapipe = dill.loads(value)
        else:
            self._datapipe = pickle.loads(value)

    def __len__(self) -> int:
        # [FUNCTION DOC] Returns the wrapped datapipe's length; any failure is converted to a
        # TypeError saying the instance has no valid length.
        try:
            return len(self._datapipe)
        except Exception as e:
            raise TypeError(
                f"{type(self).__name__} instance doesn't have valid length"
            ) from e


class _IterDataPipeSerializationWrapper(_DataPipeSerializationWrapper, IterDataPipe):
    # [CLASS DOC] HELPER: Serialization wrapper for iterable datapipes; acts as its own
    # iterator over the wrapped pipe. State: _datapipe, _datapipe_iter.
    def __init__(self, datapipe: IterDataPipe[_T_co]) -> None:
        # [FUNCTION DOC] Wraps the datapipe and starts with no active iterator.
        super().__init__(datapipe)
        # pyrefly: ignore [invalid-type-var]
        self._datapipe_iter: Iterator[_T_co] | None = None

    def __iter__(self) -> "_IterDataPipeSerializationWrapper":
        # [FUNCTION DOC] Creates a fresh iterator over the wrapped pipe, stores it, and
        # returns self.
        self._datapipe_iter = iter(self._datapipe)
        return self

    def __next__(self) -> _T_co:  # type: ignore[type-var]
        # [FUNCTION DOC] Returns the next item from the stored iterator; raises
        # AssertionError if the iteration method was not called first.
        if self._datapipe_iter is None:
            raise AssertionError(
                "Iterator has not been initialized; call __iter__() before __next__()"
            )
        return next(self._datapipe_iter)


class _MapDataPipeSerializationWrapper(_DataPipeSerializationWrapper, MapDataPipe):
    # [CLASS DOC] HELPER: Serialization wrapper for map-style datapipes; forwards item access
    # to the wrapped pipe.
    def __getitem__(self, idx):
        # [FUNCTION DOC] Returns the wrapped pipe's item for idx.
        return self._datapipe[idx]
