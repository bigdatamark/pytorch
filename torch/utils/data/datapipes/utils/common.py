# mypy: allow-untyped-defs
# [MODULE DOC]
# Shared helpers for DataPipes: validation of callables and pathname/stream data, glob-style
# file listing and opening, the deprecation-warning builder, and StreamWrapper which wraps
# file objects so they get closed deterministically. Import-time side effect: records
# whether the optional dill package is available (DILL_AVAILABLE) and creates the registries
# of deprecated functional names.
import fnmatch
import functools
import inspect
import os
import warnings
from collections.abc import Callable, Iterable
from io import IOBase
from typing import Any, NoReturn

from torch.utils._import_utils import dill_available


__all__ = [
    "validate_input_col",
    "StreamWrapper",
    "get_file_binaries_from_pathnames",
    "get_file_pathnames_from_root",
    "match_masks",
    "validate_pathname_binary_tuple",
]


# BC for torchdata
DILL_AVAILABLE = dill_available()


def validate_input_col(fn: Callable, input_col: int | tuple | list | None) -> None:
    # [FUNCTION DOC] Inspects the callable's signature and checks it can accept the number
    # of arguments implied by input_col (the list/tuple length, or one for a single index or
    # key). Returns nothing and skips the check when the signature cannot be inspected.
    # Raises ValueError for non-default keyword-only parameters, too few parameters, extra
    # required positional parameters, or too few positional ones without a variadic.
    # KNOWN ISSUE: the docstring examples assert on the return value, but it is None.
    """
    Check that function used in a callable datapipe works with the input column.

    This simply ensures that the number of positional arguments matches the size
    of the input column. The function must not contain any non-default
    keyword-only arguments.

    Examples:
        >>> # xdoctest: +SKIP("Failing on some CI machines")
        >>> def f(a, b, *, c=1):
        >>>     return a + b + c
        >>> def f_def(a, b=1, *, c=1):
        >>>     return a + b + c
        >>> assert validate_input_col(f, [1, 2])
        >>> assert validate_input_col(f_def, 1)
        >>> assert validate_input_col(f_def, [1, 2])

    Notes:
        If the function contains variable positional (``inspect.VAR_POSITIONAL``) arguments,
        for example, ``f(a, *args)``, the validator will accept any size of input column
        greater than or equal to the number of positional arguments.
        (in this case, 1).

    Args:
        fn: The function to check.
        input_col: The input column to check.

    Raises:
        ValueError: If the function is not compatible with the input column.
    """
    try:
        sig = inspect.signature(fn)
    except (
        ValueError
    ):  # Signature cannot be inspected, likely it is a built-in fn or written in C
        return
    if isinstance(input_col, (list, tuple)):
        input_col_size = len(input_col)
    else:
        input_col_size = 1

    pos = []
    var_positional = False
    non_default_kw_only = []

    for p in sig.parameters.values():
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            pos.append(p)
        elif p.kind is inspect.Parameter.VAR_POSITIONAL:
            var_positional = True
        elif p.kind is inspect.Parameter.KEYWORD_ONLY:
            if p.default is p.empty:
                non_default_kw_only.append(p)
        else:
            continue

    if isinstance(fn, functools.partial):
        fn_name = getattr(fn.func, "__name__", repr(fn.func))
    else:
        fn_name = getattr(fn, "__name__", repr(fn))

    if len(non_default_kw_only) > 0:
        raise ValueError(
            f"The function {fn_name} takes {len(non_default_kw_only)} "
            f"non-default keyword-only parameters, which is not allowed."
        )

    if len(sig.parameters) < input_col_size:
        if not var_positional:
            raise ValueError(
                f"The function {fn_name} takes {len(sig.parameters)} "
                f"parameters, but {input_col_size} are required."
            )
    else:
        if len(pos) > input_col_size:
            if any(p.default is p.empty for p in pos[input_col_size:]):
                raise ValueError(
                    f"The function {fn_name} takes {len(pos)} "
                    f"positional parameters, but {input_col_size} are required."
                )
        elif len(pos) < input_col_size:
            if not var_positional:
                raise ValueError(
                    f"The function {fn_name} takes {len(pos)} "
                    f"positional parameters, but {input_col_size} are required."
                )


def _is_local_fn(fn):
    # [FUNCTION DOC] HELPER: Reports whether a callable is defined inside another function.
    # Uses the nested flag of the code object for functions and methods, else looks for the
    # local-scope marker in the qualified name of the object or its type. Returns a truthy
    # value (a flag integer for functions, a bool otherwise), or False if undetermined.
    # Functions or Methods
    if hasattr(fn, "__code__"):
        return fn.__code__.co_flags & inspect.CO_NESTED
    # Callable Objects
    else:
        if hasattr(fn, "__qualname__"):
            return "<locals>" in fn.__qualname__
        fn_type = type(fn)
        if hasattr(fn_type, "__qualname__"):
            return "<locals>" in fn_type.__qualname__
    return False


def _check_unpickable_fn(fn: Callable) -> None:
    # [FUNCTION DOC] HELPER: Validates a function passed to a DataPipe. Raises TypeError if it
    # is not callable. For a partial object, checks the underlying function. Emits a warning
    # (not an error) when it is a local function or a lambda and dill is unavailable.
    # KNOWN ISSUE: the docstring says a UserWarning is raised; it is a plain warning issued
    # only when dill is missing.
    """
    Check function is pickable or not.

    If it is a lambda or local function, a UserWarning will be raised. If it's not a callable function, a TypeError will be raised.
    """
    if not callable(fn):
        raise TypeError(f"A callable function is expected, but {type(fn)} is provided.")

    # Extract function from partial object
    # Nested partial function is automatically expanded as a single partial object
    if isinstance(fn, functools.partial):
        fn = fn.func

    # Local function
    if _is_local_fn(fn) and not dill_available():
        warnings.warn(
            "Local function is not supported by pickle, please use "
            "regular python function or functools.partial instead.",
            stacklevel=2,
        )
        return

    # Lambda function
    if hasattr(fn, "__name__") and fn.__name__ == "<lambda>" and not dill_available():
        warnings.warn(
            "Lambda function is not supported by pickle, please use "
            "regular python function or functools.partial instead.",
            stacklevel=2,
        )
        return


def match_masks(name: str, masks: str | list[str]) -> bool:
    # [FUNCTION DOC] Returns True if the name matches the glob pattern, or any pattern in the
    # list. An empty mask (empty string or list) matches every name.
    # empty mask matches any input name
    if not masks:
        return True

    if isinstance(masks, str):
        return fnmatch.fnmatch(name, masks)

    for mask in masks:
        if fnmatch.fnmatch(name, mask):
            return True
    return False


def get_file_pathnames_from_root(
    root: str,
    masks: str | list[str],
    recursive: bool = False,
    abspath: bool = False,
    non_deterministic: bool = False,
) -> Iterable[str]:
    # [FUNCTION DOC] Generator of file paths under root whose base names match the masks. If
    # root is a file, yields it (when it matches). Otherwise walks the directory tree, only
    # the top level unless recursive, and yields joined paths (made absolute if abspath).
    # Unless non_deterministic, files and directories are sorted for a stable order. Walk
    # errors emit a warning and are re-raised as the original OSError.
    # print out an error message and raise the error out
    def onerror(err: OSError) -> NoReturn:
        # [FUNCTION DOC] HELPER: Directory-walk error handler; warns with the file name and
        # reason, then re-raises the given OSError.
        warnings.warn(err.filename + " : " + err.strerror, stacklevel=2)
        raise err

    if os.path.isfile(root):
        path = root
        if abspath:
            path = os.path.abspath(path)
        fname = os.path.basename(path)
        if match_masks(fname, masks):
            yield path
    else:
        for path, dirs, files in os.walk(root, onerror=onerror):
            if abspath:
                path = os.path.abspath(path)
            if not non_deterministic:
                files.sort()
            for f in files:
                if match_masks(f, masks):
                    yield os.path.join(path, f)
            if not recursive:
                break
            if not non_deterministic:
                # Note that this is in-place modifying the internal list from `os.walk`
                # This only works because `os.walk` doesn't shallow copy before turn
                # https://github.com/python/cpython/blob/f4c03484da59049eb62a9bf7777b963e2267d187/Lib/os.py#L407
                dirs.sort()


def get_file_binaries_from_pathnames(
    pathnames: Iterable, mode: str, encoding: str | None = None
):
    # [FUNCTION DOC] Generator that opens each pathname and yields (pathname, StreamWrapper
    # around the open file). Mode "b" or "t" is read mode with that suffix. Side effect:
    # opens files, which the consumer must close. Raises TypeError for a non-string pathname.
    # KNOWN ISSUE: a plain string is itself iterable, so the single-pathname wrapping never
    # applies to strings and each character is treated as a pathname.
    if not isinstance(pathnames, Iterable):
        pathnames = [
            pathnames,
        ]

    if mode in ("b", "t"):
        mode = "r" + mode

    for pathname in pathnames:
        if not isinstance(pathname, str):
            raise TypeError(
                f"Expected string type for pathname, but got {type(pathname)}"
            )
        yield pathname, StreamWrapper(open(pathname, mode, encoding=encoding))  # noqa:SIM115


def validate_pathname_binary_tuple(data: tuple[str, IOBase]) -> None:
    # [FUNCTION DOC] Checks that data is a 2-tuple of a string pathname and an IOBase or
    # StreamWrapper stream; otherwise raises TypeError describing the first violation.
    # KNOWN ISSUE: the last error message is missing a space between its two sentence parts.
    if not isinstance(data, tuple):
        raise TypeError(
            f"pathname binary data should be tuple type, but it is type {type(data)}"
        )
    if len(data) != 2:
        raise TypeError(
            f"pathname binary stream tuple length should be 2, but got {len(data)}"
        )
    if not isinstance(data[0], str):
        raise TypeError(
            f"pathname within the tuple should have string type pathname, but it is type {type(data[0])}"
        )
    if not isinstance(data[1], IOBase) and not isinstance(data[1], StreamWrapper):
        raise TypeError(
            f"binary stream within the tuple should have IOBase or"
            f"its subclasses as type, but it is type {type(data[1])}"
        )


# Deprecated function names and its corresponding DataPipe type and kwargs for the `_deprecation_warning` function
_iter_deprecated_functional_names: dict[str, dict] = {}
_map_deprecated_functional_names: dict[str, dict] = {}


def _deprecation_warning(
    old_class_name: str,
    *,
    deprecation_version: str,
    removal_version: str,
    old_functional_name: str = "",
    old_argument_name: str = "",
    new_class_name: str = "",
    new_functional_name: str = "",
    new_argument_name: str = "",
    deprecate_functional_name_only: bool = False,
) -> None:
    # [FUNCTION DOC] HELPER: Builds and emits a FutureWarning announcing that a DataPipe
    # class, its functional name, or one of its arguments is deprecated since one version
    # and removed in another, with optional replacement names. Raises ValueError if a new
    # functional or argument name is given without the old one, or if both an old
    # functional name and an old argument name are given.
    if new_functional_name and not old_functional_name:
        raise ValueError(
            "Old functional API needs to be specified for the deprecation warning."
        )
    if new_argument_name and not old_argument_name:
        raise ValueError(
            "Old argument name needs to be specified for the deprecation warning."
        )

    if old_functional_name and old_argument_name:
        raise ValueError(
            "Deprecating warning for functional API and argument should be separated."
        )

    msg = f"`{old_class_name}()`"
    if deprecate_functional_name_only and old_functional_name:
        msg = f"{msg}'s functional API `.{old_functional_name}()` is"
    elif old_functional_name:
        msg = f"{msg} and its functional API `.{old_functional_name}()` are"
    elif old_argument_name:
        msg = f"The argument `{old_argument_name}` of {msg} is"
    else:
        msg = f"{msg} is"
    msg = (
        f"{msg} deprecated since {deprecation_version} and will be removed in {removal_version}."
        f"\nSee https://github.com/pytorch/data/issues/163 for details."
    )

    if new_class_name or new_functional_name:
        msg = f"{msg}\nPlease use"
        if new_class_name:
            msg = f"{msg} `{new_class_name}()`"
        if new_class_name and new_functional_name:
            msg = f"{msg} or"
        if new_functional_name:
            msg = f"{msg} `.{new_functional_name}()`"
        msg = f"{msg} instead."

    if new_argument_name:
        msg = f"{msg}\nPlease use `{old_class_name}({new_argument_name}=)` instead."

    warnings.warn(msg, FutureWarning, stacklevel=2)


class StreamWrapper:
    # [CLASS DOC]
    # Proxy around a file-like object that forwards unknown attributes to it and guarantees
    # closing. State: the wrapped object, an optional parent StreamWrapper with a child
    # counter (a parent set to auto-close is closed with its last child), a closed flag, an
    # optional name, and class-level debug tracking of unclosed streams. Closed on deletion.
    """
    StreamWrapper is introduced to wrap file handler generated by DataPipe operation like `FileOpener`.

    StreamWrapper would guarantee the wrapped file handler is closed when it's out of scope.
    """

    session_streams: dict[Any, int] = {}
    debug_unclosed_streams: bool = False

    def __init__(self, file_obj, parent_stream=None, name=None) -> None:
        # [FUNCTION DOC] Wraps file_obj. If a parent stream is given it must be a
        # StreamWrapper (otherwise raises RuntimeError) and its child counter is incremented.
        # When debug tracking is on, registers this stream in the class-level session map.
        self.file_obj = file_obj
        self.child_counter = 0
        self.parent_stream = parent_stream
        self.close_on_last_child = False
        self.name = name
        self.closed = False
        if parent_stream is not None:
            if not isinstance(parent_stream, StreamWrapper):
                raise RuntimeError(
                    f"Parent stream should be StreamWrapper, {type(parent_stream)} was given"
                )
            parent_stream.child_counter += 1
            self.parent_stream = parent_stream
        if StreamWrapper.debug_unclosed_streams:
            StreamWrapper.session_streams[self] = 1

    @classmethod
    def close_streams(cls, v, depth=0) -> None:
        # [FUNCTION DOC] Closes v if it is a StreamWrapper; otherwise recurses through the
        # values of a dict or the items of a list or tuple, down to nesting depth 10.
        """Traverse structure and attempts to close all found StreamWrappers on best effort basis."""
        if depth > 10:
            return
        if isinstance(v, StreamWrapper):
            v.close()
        else:
            # Traverse only simple structures
            if isinstance(v, dict):
                for vv in v.values():
                    cls.close_streams(vv, depth=depth + 1)
            elif isinstance(v, (list, tuple)):
                for vv in v:
                    cls.close_streams(vv, depth=depth + 1)

    def __getattr__(self, name):
        # [FUNCTION DOC] Forwards lookup of any attribute not found on the wrapper to the
        # wrapped object (read directly from the instance dictionary to avoid recursion).
        file_obj = self.__dict__["file_obj"]
        return getattr(file_obj, name)

    def close(self, *args, **kwargs) -> None:
        # [FUNCTION DOC] Idempotent close: unregisters from debug tracking, decrements the
        # parent's child counter (closing the parent if it is set to auto-close and this was
        # its last child), closes the wrapped object with the given arguments (ignoring
        # AttributeError if it has no close), and marks this wrapper closed.
        # KNOWN ISSUE: with debug tracking on, a stream created before tracking was enabled
        # is missing from the session map and raises KeyError here.
        if self.closed:
            return
        if StreamWrapper.debug_unclosed_streams:
            del StreamWrapper.session_streams[self]
        if hasattr(self, "parent_stream") and self.parent_stream is not None:
            self.parent_stream.child_counter -= 1
            if (
                not self.parent_stream.child_counter
                and self.parent_stream.close_on_last_child
            ):
                self.parent_stream.close()
        try:
            self.file_obj.close(*args, **kwargs)
        except AttributeError:
            pass
        self.closed = True

    def autoclose(self) -> None:
        # [FUNCTION DOC] Marks the stream to close when its last child closes; closes it right
        # away if it has no children.
        """Automatically close stream when all child streams are closed or if there are none."""
        self.close_on_last_child = True
        if self.child_counter == 0:
            self.close()

    def __dir__(self):
        # [FUNCTION DOC] Returns the unique attribute names of this wrapper, its class and the
        # wrapped object.
        attrs = list(self.__dict__.keys()) + list(StreamWrapper.__dict__.keys())
        attrs += dir(self.file_obj)
        return list(set(attrs))

    def __del__(self) -> None:
        # [FUNCTION DOC] Finalizer: closes the stream if it has not been closed yet.
        if not self.closed:
            self.close()

    def __iter__(self):
        # [FUNCTION DOC] Iterates over the wrapped object.
        yield from self.file_obj

    def __next__(self):
        # [FUNCTION DOC] Returns the next item of the wrapped object.
        return next(self.file_obj)

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text form showing the optional name and the wrapped object.
        if self.name is None:
            return f"StreamWrapper<{self.file_obj!r}>"
        else:
            return f"StreamWrapper<{self.name},{self.file_obj!r}>"

    def __getstate__(self):
        # [FUNCTION DOC] Pickling support: the state is just the wrapped object.
        return self.file_obj

    def __setstate__(self, obj):
        # [FUNCTION DOC] Restores only the wrapped object.
        # KNOWN ISSUE: the other fields (closed, parent, counters, name) are not restored, so
        # attribute lookups for them fall through to the wrapped object.
        self.file_obj = obj
