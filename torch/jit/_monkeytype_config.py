# mypy: allow-untyped-defs
# [MODULE DOC]
# Integration with MonkeyType for collecting argument types of calls so
# TorchScript can infer annotations. Provides a trace store, trace logger,
# config and code filter, plus type-to-string conversion helpers. If MonkeyType
# is not installed, stub placeholder classes are defined instead. Import-time
# side effects: attempts to import monkeytype and records availability in a
# module flag.
import inspect
import sys
import typing
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from types import CodeType

import torch


_IS_MONKEYTYPE_INSTALLED = True
try:
    import monkeytype  # type: ignore[import]

    # pyrefly: ignore [import-error, missing-import]
    from monkeytype import trace as monkeytype_trace
    from monkeytype.config import _startswith, LIB_PATHS  # type: ignore[import]
    from monkeytype.db.base import (  # type: ignore[import]
        CallTraceStore,
        CallTraceStoreLogger,
        CallTraceThunk,
    )
    from monkeytype.tracing import CallTrace, CodeFilter  # type: ignore[import]
except ImportError:
    _IS_MONKEYTYPE_INSTALLED = False


# Checks whether a class is defined in `torch.*` modules
def is_torch_native_class(cls):
    # [FUNCTION DOC] Returns True if cls has a module attribute whose top-level
    # package is the torch package, else False.
    if not hasattr(cls, "__module__"):
        return False

    parent_modules = cls.__module__.split(".")
    if not parent_modules:
        return False

    root_module = sys.modules.get(parent_modules[0])
    return root_module is torch


def get_type(type):
    # [FUNCTION DOC] Converts a type to a string TorchScript accepts: strings
    # pass through; typing constructs have the module prefix removed from their
    # text; torch classes become module-qualified names; all others use the plain
    # type name.
    """Convert the given type to a torchScript acceptable format."""
    if isinstance(type, str):
        return type
    elif inspect.getmodule(type) == typing:
        # If the type is a type imported from typing
        # like Tuple, List, Dict then replace `typing.`
        # with a null string. This needs to be done since
        # typing.List is not accepted by TorchScript.
        type_to_string = str(type)
        return type_to_string.replace(type.__module__ + ".", "")
    elif is_torch_native_class(type):
        # If the type is a subtype of torch module, then TorchScript expects a fully qualified name
        # for the type which is obtained by combining the module name and type name.
        return type.__module__ + "." + type.__name__
    else:
        # For all other types use the name for the type.
        return type.__name__


def get_optional_of_element_type(types):
    # [FUNCTION DOC] Given a two-element list of types where one is NoneType,
    # picks the other as the element type and returns the string for an optional
    # of that type. Temporary workaround until Union support exists.
    """Extract element type, return as `Optional[element type]` from consolidated types.

    Helper function to extract the type of the element to be annotated to Optional
    from the list of consolidated types and returns `Optional[element type]`.
    TODO: To remove this check once Union support lands.
    """
    elem_type = types[1] if type(None) is types[0] else types[0]
    elem_type = get_type(elem_type)

    # Optional type is internally converted to Union[type, NoneType], which
    # is not supported yet in TorchScript. Hence, representing the optional type as string.
    return "Optional[" + elem_type + "]"


def get_qualified_name(func):
    # [FUNCTION DOC] Returns the qualified name attribute of func.
    # KNOWN ISSUE: only the qualified name (no module) is used as key, so
    # same-named functions in different modules collide.
    return func.__qualname__


if _IS_MONKEYTYPE_INSTALLED:

    class JitTypeTraceStoreLogger(CallTraceStoreLogger):
        """A JitTypeCallTraceLogger that stores logged traces in a CallTraceStore."""

        def log(self, trace: CallTrace) -> None:
            # [FUNCTION DOC] Appends the given call trace to the traces list of
            # this logger (the list is provided by the base class).
            # pyrefly: ignore [missing-attribute]
            self.traces.append(trace)

    class JitTypeTraceStore(CallTraceStore):
        # [CLASS DOC] In-memory MonkeyType call trace store. State: trace_records,
        # a mapping from the qualified function name to a list of CallTrace
        # objects. Methods add, filter and analyze/consolidate_types/get_args_types
        # derive per-argument type strings from the collected traces.
        def __init__(self) -> None:
            # [FUNCTION DOC] Initializes the base store and an empty mapping from
            # qualified function name to list of traces.
            super().__init__()
            # A dictionary keeping all collected CallTrace
            # key is fully qualified name of called function
            # value is list of all CallTrace
            self.trace_records: dict[str, list] = defaultdict(list)

        def add(self, traces: Iterable[CallTrace]) -> None:
            # [FUNCTION DOC] Adds each trace to the list kept under its function's
            # qualified name. Mutates trace_records.
            for t in traces:
                qualified_name = get_qualified_name(t.func)
                self.trace_records[qualified_name].append(t)

        def filter(
            self,
            qualified_name: str,
            qualname_prefix: str | None = None,
            limit: int = 2000,
        ) -> list[CallTraceThunk]:
            # [FUNCTION DOC] Returns the list of traces recorded for the qualified
            # name; the prefix and limit arguments are ignored. Creates an empty
            # entry if the name is unknown.
            return self.trace_records[qualified_name]

        def analyze(self, qualified_name: str) -> dict:
            # [FUNCTION DOC] Collects, for the traces of the named function, a
            # mapping from argument name to the set of observed types.
            # Analyze the types for the given module
            # and create a dictionary of all the types
            # for arguments.
            records = self.trace_records[qualified_name]
            all_args = defaultdict(set)
            for record in records:
                for arg, arg_type in record.arg_types.items():
                    all_args[arg].add(arg_type)
            return all_args

        def consolidate_types(self, qualified_name: str) -> dict:
            # [FUNCTION DOC] Reduces each argument's observed type set to one
            # string: a single type becomes its converted name; exactly two types
            # including NoneType become an optional of the other type; more than
            # one type otherwise becomes Any. Returns the mapping of argument name
            # to string (the mapping from analyze, modified in place).
            all_args = self.analyze(qualified_name)
            # If there are more types for an argument,
            # then consolidate the type to `Any` and replace the entry
            # by type `Any`.
            for arg, types in all_args.items():
                types = list(types)
                type_length = len(types)
                if type_length == 2 and type(None) in types:
                    # TODO: To remove this check once Union support in TorchScript lands.
                    all_args[arg] = get_optional_of_element_type(types)
                elif type_length > 1:
                    all_args[arg] = "Any"
                elif type_length == 1:
                    all_args[arg] = get_type(types[0])
            return all_args

        def get_args_types(self, qualified_name: str) -> dict:
            # [FUNCTION DOC] Returns the consolidated argument type strings for
            # the named function; thin wrapper over consolidate_types.
            return self.consolidate_types(qualified_name)

    class JitTypeTraceConfig(monkeytype.config.Config):
        # [CLASS DOC] MonkeyType config that logs to a given JitTypeTraceStore and
        # uses the custom jit_code_filter. State: the store, kept as attribute s.
        def __init__(self, s: JitTypeTraceStore) -> None:
            # [FUNCTION DOC] Initializes the base config and stores the trace
            # store s used for all logging.
            super().__init__()
            self.s = s

        def trace_logger(self) -> JitTypeTraceStoreLogger:
            # [FUNCTION DOC] Returns a new trace logger bound to the configured
            # trace store.
            """Return a JitCallTraceStoreLogger that logs to the configured trace store."""
            # pyrefly: ignore [bad-argument-count]
            return JitTypeTraceStoreLogger(self.trace_store())

        def trace_store(self) -> CallTraceStore:
            # [FUNCTION DOC] Returns the trace store given at construction.
            return self.s

        def code_filter(self) -> CodeFilter | None:
            # [FUNCTION DOC] Returns the custom jit_code_filter function.
            return jit_code_filter

else:
    # When MonkeyType is not installed, we provide dummy class definitions
    # for the below classes.
    class JitTypeTraceStoreLogger:  # type:  ignore[no-redef]
        # [CLASS DOC] HELPER: Placeholder used when MonkeyType is not installed.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Does nothing.
            pass

    class JitTypeTraceStore:  # type:  ignore[no-redef]
        # [CLASS DOC] HELPER: Placeholder used when MonkeyType is not installed;
        # its trace_records attribute is None.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Sets trace_records to None.
            self.trace_records = None

    class JitTypeTraceConfig:  # type:  ignore[no-redef]
        # [CLASS DOC] HELPER: Placeholder used when MonkeyType is not installed.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Does nothing.
            pass

    monkeytype_trace = None  # type: ignore[assignment]


def jit_code_filter(code: CodeType) -> bool:
    # [FUNCTION DOC] MonkeyType code filter deciding whether to trace a code
    # object. Rejects code with no file name or a file name starting with an
    # angle bracket unless the function is named forward (so FX-generated forward
    # calls are traced); otherwise resolves the file path and returns True only if
    # it is not under any standard-library or site-packages path.
    """Codefilter for Torchscript to trace forward calls.

    The custom CodeFilter is required while scripting a FX Traced forward calls.
    FX Traced forward calls have `code.co_filename` start with '<' which is used
    to exclude tracing of stdlib and site-packages in the default code filter.
    Since we need all forward calls to be traced, this custom code filter
    checks for code.co_name to be 'forward' and enables tracing for all such calls.
    The code filter is similar to default code filter for monkeytype and
    excludes tracing of stdlib and site-packages.
    """
    # Filter code without a source file and exclude this check for 'forward' calls.
    if code.co_name != "forward" and (
        not code.co_filename or code.co_filename[0] == "<"
    ):
        return False

    filename = Path(code.co_filename).resolve()
    return not any(_startswith(filename, lib_path) for lib_path in LIB_PATHS)
