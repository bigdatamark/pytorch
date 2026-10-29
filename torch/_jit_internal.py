# mypy: allow-untyped-defs
# [MODULE DOC] Internal TorchScript support utilities kept outside torch/jit to avoid circular imports (used by torch.nn
# and others). Section 1 covers: type helpers (is_final, BroadcastingList1..6), qualified-name computation,
# source/closure based name-resolution callbacks, and type-hint capture. Later parts hold scripting decorators, fork/
# wait helpers, optional/Future/Await type predicates and class/module annotation handling. Import-time side effects:
# imports torch.distributed.rpc, defines LockType, boolean_dispatched weak mapping, the module-level source loader, and
# registers BroadcastingList2..6 as module globals aliasing BroadcastingList1.
"""
The weak_script annotation needs to be here instead of inside torch/jit/ so it
can be used in other places in torch/ (namely torch.nn) without running into
circular dependency problems
"""

import ast
import builtins
import collections
import contextlib
import enum
import functools
import inspect
import io
import pickle
import sys
import textwrap
import threading
import types
import typing
import warnings
import weakref
from typing import (  # noqa: UP035, F401  # (Dict, List, Tuple) imported by torch.jit.annotations
    Any,
    Callable,
    Dict,
    Final,
    ForwardRef,
    get_args,
    get_origin,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)
from typing_extensions import ParamSpec

import torch

# This is needed. `torch._jit_internal` is imported before `torch.distributed.__init__`.
# Explicitly ask to import `torch.distributed.__init__` first.
# Otherwise, "AttributeError: module 'torch' has no attribute 'distributed'" is raised.
import torch.distributed.rpc
import torch.package._mangling as package_mangling
from torch._awaits import _Await
from torch._C import _Await as CAwait, Future as CFuture
from torch._sources import fake_range, get_source_lines_and_file, parse_def
from torch.futures import Future


class HasGetattr(Protocol):
    # [CLASS DOC] HELPER: Protocol for objects that support attribute lookup by name (declared via a one-line stub).
    def __getattr__(self, key: str) -> Any: ...


_P = ParamSpec("_P")
_R = TypeVar("_R")

BuiltinUnionType: type | tuple[type, ...] = types.UnionType

LockType: type
try:
    import _thread

    LockType = _thread.LockType
except ImportError:
    import _dummy_thread  # type: ignore[import-not-found]

    LockType = _dummy_thread.LockType

# Wrapper functions that can call either of 2 functions depending on a boolean
# argument
boolean_dispatched: "weakref.WeakKeyDictionary[Callable, dict[str, Callable]]" = (
    weakref.WeakKeyDictionary()
)  # noqa: T484


FAKE_FILENAME_PREFIX = "__torch_jit_dataclass"


def is_final(ann) -> bool:
    # [FUNCTION DOC] Returns true if the annotation is the typing Final marker (bare or subscripted), recognized by its
    # module being typing or typing_extensions.
    return (
        hasattr(ann, "__module__")
        and ann.__module__ in {"typing", "typing_extensions"}
        and (get_origin(ann) is Final or isinstance(ann, type(Final)))
    )


# allows BroadcastingList instance to be subscriptable
class BroadcastingListCls:
    # [CLASS DOC] HELPER: Placeholder type whose instances support subscripting; used so BroadcastingListN[T] annotations
    # are valid expressions. Subscripting always returns None.
    def __getitem__(self, types):
        # [FUNCTION DOC] HELPER: Accepts any subscript and returns None.
        return


# mypy doesn't support parameters on types, so we have to explicitly type each
# list size
BroadcastingList1 = BroadcastingListCls()
for i in range(2, 7):
    globals()[f"BroadcastingList{i}"] = BroadcastingList1


def is_scripting() -> bool:
    # [FUNCTION DOC] Returns false in eager Python; the TorchScript compiler treats a call to it as true, so code can
    # branch on scripting. Takes no arguments.
    r"""
    Function that returns True when in compilation and False otherwise. This
    is useful especially with the @unused decorator to leave code in your
    model that is not yet TorchScript compatible.
    .. testcode::

        import torch

        @torch.jit.unused
        def unsupported_linear_op(x):
            return x

        def linear(x):
            if torch.jit.is_scripting():
                return torch.linear(x)
            else:
                return unsupported_linear_op(x)
    """
    return False


# Retrieves a fully-qualified name (module hierarchy + classname) for a given obj.
def _qualified_name(obj, mangle_name=True) -> str:
    # [FUNCTION DOC] HELPER: Returns the fully qualified TorchScript name (module plus class/function name) for an
    # object. Uses the override attribute if present, the qualified name of script functions and torchbind classes
    # directly; otherwise builds it from the name (lambda becomes a valid identifier), normalizing torch.package mangling,
    # and, if mangle_name, prefixing the module with the torch namespace (main becomes the torch namespace itself).
    # Raises RuntimeError if no name can be found, the module is None, or the name contains a dot.
    # This special case allows us to override the qualified name on a type.
    # It's currently used in conjunction with tracing, where we create a
    # fake module to filter only supported attributes. However, since this
    # new type is defined as a local class, we need a mechanism to override
    # its qualname so it appears correctly in the TorchScript system. This,
    # we set '_jit_override_qualname' with the original traced module's
    # qualified name, which is picked up here
    if hasattr(obj, "_jit_override_qualname"):
        return obj._jit_override_qualname
    # short-circuit in cases where the object already has a known qualified name
    if isinstance(obj, torch._C.ScriptFunction):
        return obj.qualified_name

    if getattr(obj, "__name__", None):
        name = obj.__name__
    # Enum classes do not have `__name__` attr, instead they have `name`.
    elif isinstance(obj, enum.Enum):
        name = obj.name
    else:
        raise RuntimeError("Could not get name of python class object")

    if name == "<lambda>":
        name = "_lambda"  # make name a valid identifier

    module_name = obj.__module__

    # If the module is actually a torchbind module, then we should short circuit
    if module_name == "torch._classes":
        return obj.qualified_name

    # The Python docs are very clear that `__module__` can be None, but I can't
    # figure out when it actually would be.
    if module_name is None:
        raise RuntimeError(
            f"Could not get qualified name for class '{name}': "
            "__module__ can't be None."
        )

    # if getattr(sys.modules[module_name], name) is not obj:
    #     raise RuntimeError(f"Could not get qualified name for class '{name}': "
    #                        f"the attr {name} on module {module_name} is not the class")

    # torch.package and TorchScript have separate mangling schemes to avoid
    # name collisions from multiple packages. To avoid them interfering with
    # each other, normalize the package managing here.
    if package_mangling.is_mangled(module_name):
        module_name = module_name.replace("<", "_")
        module_name = module_name.replace(">", "_")

    # The PythonExceptionValue C++ class in torch/csrc/jit/python/python_sugared_value.h
    # does not need mangle the python class name.
    if mangle_name:
        # __main__ is a builtin module, so rewrite it to "__torch__".
        if module_name == "__main__":
            module_name = "__torch__"
        else:
            # Everything else gets a "__torch__" prefix to avoid name collisions
            # with the names of user values.
            module_name = "__torch__." + module_name

    if "." in name:
        raise RuntimeError(
            f"Could not get qualified name for class '{name}': "
            f"'{name}' is not a valid identifier"
        )

    return module_name + "." + name


class SourceLoader:
    # [CLASS DOC] Cache of function-to-source-text overrides used when source cannot be read from files. State: content,
    # mapping from function to source. A module-level instance named loader is created.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Initializes the empty content mapping.
        self.content = {}

    def cache(self, fn, source):
        # [FUNCTION DOC] HELPER: Stores the source text for a function.
        self.content[fn] = source

    def get_source(self, fn):
        # [FUNCTION DOC] HELPER: Returns the stored source for a function, or None if absent.
        return self.content.get(fn)


loader = SourceLoader()


def createResolutionCallbackFromEnv(lookup_base: HasGetattr) -> Callable[[str], Any]:
    # [FUNCTION DOC] Builds a callback mapping an expression string (possibly dotted and subscripted like a type) to a
    # Python object by resolving names via attribute lookup on lookup_base. Returns None from the callback when the Python
    # resolver fails so the C++ resolver can take over. Only for use by the other createResolutionCallbackFrom functions.
    """
    Creates a resolution callback that will look up qualified names in an
    environment, starting with `lookup_base` for the base of any qualified
    names, then proceeding down the lookup chain with the resolved object.

    You should not use this directly, it should only be used from the other
    createResolutionCallbackFrom* functions.
    """

    def lookupInModule(qualified_name: str, module: Any) -> Any:
        # [FUNCTION DOC] HELPER: Resolves a dotted name by successive attribute lookup starting at the given module/object.
        if "." in qualified_name:
            base, remaining_pieces = qualified_name.split(".", maxsplit=1)
            module_value = getattr(module, base)
            return lookupInModule(remaining_pieces, module_value)
        else:
            return getattr(module, qualified_name)

    def parseNestedExpr(expr: str, module: Any) -> tuple[Any, int]:
        # [FUNCTION DOC] HELPER: Parses a type-like expression with nested bracketed, comma-separated parts, returning the
        # resolved object and the number of characters consumed. Handles the empty tuple specially; raises AssertionError for
        # an unresolvable name.
        # KNOWN ISSUE: the second check that the current character is an open bracket can never fail, since the earlier
        # return already covers every other case.
        i = 0
        while i < len(expr) and expr[i] not in (",", "[", "]"):
            i += 1

        # Special case logic for the empty Tuple as a subscript (used
        # in the type annotation `Tuple[()]`)
        if expr[:i] == "()":
            return (), i

        base = lookupInModule(expr[:i].strip(), module)
        if base is None:
            raise AssertionError(f"Unresolvable type {expr[:i]}")
        if i == len(expr) or expr[i] != "[":
            return base, i

        if expr[i] != "[":
            raise AssertionError(f"expected '[' at position {i}, got {expr[i]!r}")
        parts = []
        while expr[i] != "]":
            part_len = 0
            i += 1
            part, part_len = parseNestedExpr(expr[i:], module)
            parts.append(part)
            i += part_len
        if len(parts) > 1:
            return base[tuple(parts)], i + 1
        else:
            return base[parts[0]], i + 1

    def parseExpr(expr: str, module: Any) -> Any:
        # [FUNCTION DOC] HELPER: Resolves the whole expression string; returns None on any failure or if not fully consumed.
        try:
            value, len_parsed = parseNestedExpr(expr, module)
            if len_parsed != len(expr):
                raise AssertionError(
                    "whole expression was not parsed, falling back to c++ parser"
                )
            return value
        except Exception:
            """
            The python resolver fails in several cases in known unit tests, and is intended
            to fall back gracefully to the c++ resolver in general.  For example, python 2 style
            annotations which are frequent in our unit tests often fail with types e.g. int not
            resolvable from the calling frame.
            """
            return None

    return lambda expr: parseExpr(expr, lookup_base)


def createResolutionCallbackFromFrame(frames_up: int = 0) -> Callable[[str], Any]:
    # [FUNCTION DOC] Returns a resolution callback over the local variables, globals and builtins of a caller frame
    # (frames_up extra frames above the immediate caller). Raises AssertionError if the stack is too shallow.
    """
    Creates a function which, given a string variable name,
    returns the value of the variable in the scope of the caller of
    the function which called createResolutionCallbackFromFrame (by default).

    This is used to enable access in-scope Python variables inside
    TorchScript fragments.

    frames_up is number of additional frames to go up on the stack.
    The default value is 0, which correspond to the frame of the caller
    of createResolutionCallbackFromFrame. Also for example, if frames_up is set
    to 1, then the frame of the caller's caller of createResolutionCallbackFromFrame
    will be taken.

    For example, the following program prints 2::

        def bar():
            cb = createResolutionCallbackFromFrame(1)
            print(cb("foo"))


        def baz():
            foo = 2
            bar()


        baz()
    """
    frame = inspect.currentframe()
    i = 0
    while i < frames_up + 1:
        if frame is None:
            raise AssertionError(f"frame is None at iteration {i}")
        frame = frame.f_back
        i += 1

    if frame is None:
        raise AssertionError("frame is None after traversing frames_up")
    f_locals = frame.f_locals
    f_globals = frame.f_globals

    class env:
        # [CLASS DOC] HELPER: Environment object whose attribute lookup consults the captured frame locals, then globals, then
        # builtins.
        def __getattr__(self, key: str) -> Any:
            # [FUNCTION DOC] HELPER: Returns the named value from locals, globals or builtins; returns None implicitly if absent.
            if key in f_locals:
                return f_locals[key]
            elif key in f_globals:
                return f_globals[key]
            elif key in dir(builtins):
                return getattr(builtins, key)

    return createResolutionCallbackFromEnv(env())


def get_closure(fn):
    """
    # [FUNCTION DOC] Returns a mapping of names visible to a function: all its globals plus its closed-over (free)
    # variables with their cell contents.
    Get a dictionary of closed over variables from a function
    """
    captures = {}
    captures.update(fn.__globals__)

    for index, captured_name in enumerate(fn.__code__.co_freevars):
        captures[captured_name] = fn.__closure__[index].cell_contents

    return captures


# [local resolution in python]
# Depending on where a variable is defined, and where it is used, we may
# or may not be able to recover its value when recursively compiling a
# script function. Remember in the general case, a module or function is
# first defined and then later scripted. This means we do not have a
# chance to capture the active frames when the function is defined. Hence any
# name resolution has to happen later on the created closure. The way
# python captures type annotations restricts what we can recover. The
# follow example illustrates the different cases:
#
#         class MyGlobalClass:
#         ...
#         def my_local_scope():
#             @torch.jit.script
#             class MyClass:
#                 ...
#             @torch.jit.script
#             class MyClassUsedAsVar:
#                 ...
#             def eg(x: MyClass, y: MyGlobalClass):
#                 a_local_capture : Foo
#                 return MyClassUsedAsVar(x)
#
# MyGlobalClass is defined in the __globals__ dictionary of function
# 'eg', so it is always recoverable. my_local_scope introduces a new local
# variable scope in the function. Classes defined here are only visible as
# local variables. For the case of MyClassUsedAsVar, it is captured
# because it is used as a variable inside the body of the function, and we
# can resolve it using the captures returned from `get_closure`. However,
# the type annotations are not captured by the closure. In Python
# 3.0--3.9, the _value_ of MyClass and MyGlobalClass will be available as
# annotations on `eg``, but starting in Python 4.0, they will represented as
# strings and no longer present. Furthermore, since the body of `eg` does
# not reference those names, they do not appear in the list of closed over
# variables. We anticipate
# that most users will not run into this issue because their modules and
# functions will be defined at a global scope like MyGlobalClass. In cases
# where they are not, it is possible to work around issues by declaring the
# values global in the function.
# In Python 3.9 declaring class as global will make it invisible to
# `inspect.getsource`, see https://bugs.python.org/issue42666 .
# This could be worked around by manually adding it to `global()` dictionary.


def createResolutionCallbackFromClosure(fn) -> Callable[[str], Any]:
    # [FUNCTION DOC] Returns a resolution callback built from a function's closure and globals rather than the call
    # stack; lookups also fall back to typing and builtins names, and yield None if nothing matches.
    """
    Create a resolutionCallback by introspecting the function instead of
    looking up the stack for the enclosing scope
    """
    closure = get_closure(fn)

    class closure_lookup:
        # [CLASS DOC] HELPER: Wraps the captured closure mapping so lookups work through attribute access.
        # This is a class since `closure` is a dict and it's easier in
        # `env_helper` if everything just works with `getattr` calls
        def __getattr__(self, key: str) -> Any:
            # [FUNCTION DOC] HELPER: Looks the name up in the closure, then typing, then builtins; returns None if not found.
            if key in closure:
                return closure[key]
            elif hasattr(typing, key):
                return getattr(typing, key)
            elif hasattr(builtins, key):
                return getattr(builtins, key)
            return None

    return createResolutionCallbackFromEnv(closure_lookup())


@functools.cache
def can_compile_class(cls) -> bool:
    # [FUNCTION DOC] Cached check for whether a class can be compiled by TorchScript: false for ignored functions/
    # classes, subclasses of Module, tuple, list or Exception, and any class where a routine attribute lacks a code
    # object (builtins or C-bound types); true otherwise.
    # If any of the functions on a type don't have a code object, this type can't
    # be compiled and is probably a builtin / bound from C
    if is_ignored_fn(cls):
        return False

    # Ignore the following list of built-in classes.
    ignored_builtin_classes = (torch.nn.Module, tuple, list, Exception)
    if issubclass(cls, ignored_builtin_classes):
        return False

    names = cls.__dict__
    fns = [
        getattr(cls, name)
        for name in names
        if inspect.isroutine(getattr(cls, name, None))
    ]
    has_code = [hasattr(fn, "__code__") for fn in fns]
    return all(has_code)


def get_callable_argument_names(fn) -> list[str]:
    # [FUNCTION DOC] Returns the names of the callable's parameters if all are positional-or-keyword kind, else empty
    # list. Also returns an empty list when the signature cannot be obtained.
    # KNOWN ISSUE: the docstring says an empty list is returned when other kinds are present, but the code only skips
    # those parameters and still returns the remaining positional-or-keyword names.
    """
    Gets names of all POSITIONAL_OR_KEYWORD arguments for callable `fn`.
    Returns an empty list when other types of arguments are present.

    This is used by `torch.jit.trace` to assign meaningful argument names to
    traced functions and modules.

    Args:
        fn: A callable.
    Returns:
        Argument names: List[str]
    """
    # inspect.signature may fail, give up in that case.
    try:
        callable_signature = inspect.signature(fn)
    except Exception:
        return []

    argument_names = []
    for name, param in callable_signature.parameters.items():
        # All four other types of arguments do not map to individual values
        # with a keyword as name.
        if param.kind != param.POSITIONAL_OR_KEYWORD:
            continue

        argument_names.append(name)

    return argument_names


def get_annotation_str(annotation):
    # [FUNCTION DOC] Converts an AST annotation node (name, attribute, subscript, tuple, constant) into its source text
    # string. Returns None for unsupported node kinds.
    """
    Convert an AST node containing a type annotation to the string present in the source
    that represents the same annotation.
    """
    if isinstance(annotation, ast.Name):
        return annotation.id
    elif isinstance(annotation, ast.Attribute):
        return ".".join([get_annotation_str(annotation.value), annotation.attr])
    elif isinstance(annotation, ast.Subscript):
        # In Python3.9+ subscript indices are not wrapped in ast.Index
        subscript_slice = annotation.slice
        return f"{get_annotation_str(annotation.value)}[{get_annotation_str(subscript_slice)}]"
    elif isinstance(annotation, ast.Tuple):
        return ",".join([get_annotation_str(elt) for elt in annotation.elts])
    elif isinstance(annotation, ast.Constant):
        return f"{annotation.value}"

    # If an AST node is not handled here, it's probably handled in ScriptTypeParser.
    return None


def get_type_hint_captures(fn):
    # [FUNCTION DOC] Returns a mapping from literal annotation text on a function's parameters (and return value, if
    # present and not a string) to the type objects they refer to. Gets source from the loader or inspect (raises OSError
    # if unavailable), parses it with the ast module (raises RuntimeError if not a single function), skips string
    # annotations and unsupported nodes.
    """
    Get a dictionary containing type resolution mappings necessary to resolve types
    for the literal annotations on 'fn'. These are not considered to be closed-over by fn
    and must be obtained separately (e.g. using this function).

    Args:
        fn: A callable.
    Returns:
        A Dict[str, Any] containing a mapping from the literal annotations used on
        fn to the Python objects they refer to.
    """
    # First, try to get the source of the function. We'll need to parse it to find the actual string names
    # that were used to annotate the types, since inspect.signature() will only return the class object that
    # the annotation refers to, not the string name. If we can't get the source, simply return an empty dict.
    # This may happen in cases where the function is synthesized dynamically at runtime.
    src = loader.get_source(fn)
    if src is None:
        try:
            src = inspect.getsource(fn)
        except OSError as e:
            raise OSError(
                f"Failed to get source for {fn} using inspect.getsource"
            ) from e

    # Gather a dictionary of parameter name -> type, skipping any parameters whose annotated
    # types are strings. These are only understood by TorchScript in the context of a type annotation
    # that refers to a class in its own definition, but trying to include a mapping for this in the result
    # function would cause infinite recursion because the class is currently being compiled.
    # In addition, there is logic in ScriptTypeParser to handle this.
    signature = inspect.signature(fn)
    name_to_type = {
        name: parameter.annotation
        for name, parameter in signature.parameters.items()
        if parameter.annotation is not inspect.Parameter.empty
        and not isinstance(parameter.annotation, str)
    }

    # Then, get the literal type annotations from the function declaration
    # by source inspection. This accounts for the case in which aliases are used
    # to annotate the arguments (e.g device_t = torch.device, and then d: device_t).
    # frontend.py cannot be used here because it includes _jit_internal, so use ast instead.
    a = ast.parse(textwrap.dedent(src))
    if len(a.body) != 1 or not isinstance(a.body[0], ast.FunctionDef):
        raise RuntimeError(f"Expected {fn} to be a function")
    f = a.body[0]

    # Prepare a dictionary of source annotation -> type, which will be the final result of this function,
    # by using the parsed AST (f) to reconstruct source annotations as strings for each parameter and mapping
    # them to the type object corresponding to the annotation via name_to_type using the parameter name.
    annotation_to_type = {}

    for arg in f.args.args:
        # Get the source type annotation string for this argument if possible.
        arg_annotation_str = (
            get_annotation_str(arg.annotation) if arg.annotation else None
        )

        # If the argument has no annotation or get_annotation_str cannot convert it to a string,
        # arg_annotation_str will be None. Skip this arg; ScriptTypeParser will probably handle
        # this in the latter case.
        if arg_annotation_str is None:
            continue

        # Insert {arg_annotation_str: type} into annotation_to_type if possible. One reason arg_name may not
        # be present in name_to_type is that the annotation itself is a string and not a type object
        # (common for self-referential annotations in classes). Once again, let ScriptTypeParser handle this.
        arg_name = arg.arg
        if arg_name in name_to_type:
            annotation_to_type[arg_annotation_str] = name_to_type[arg_name]

    # If there is a valid return annotation, include it in annotation_to_type. As with argument annotations,
    # the literal annotation has to be convertible to a string by get_annotation_str, and the actual type
    # of the annotation cannot be a string.
    literal_return_annotation = get_annotation_str(f.returns)
    valid_literal_annotation = literal_return_annotation is not None
    return_annotation = signature.return_annotation
    valid_return_annotation_type = (
        return_annotation is not inspect.Parameter.empty
        and not isinstance(return_annotation, str)
    )
    if valid_literal_annotation and valid_return_annotation_type:
        annotation_to_type[literal_return_annotation] = return_annotation

    return annotation_to_type


def createResolutionCallbackForClassMethods(cls: type) -> Callable[[str], Any]:
    # [FUNCTION DOC] Builds a resolution callback for a class: gathers closures and type-hint captures of all its
    # non-builtin routines (skipping the deferred-annotation helper) into one mapping. The callback returns the captured
    # value for a name, else a builtins value, else None.
    """
    This looks at all the methods defined in a class and pulls their closed-over
    variables into a dictionary and uses that to resolve variables.
    """
    # cls is a type here, so `ismethod` is false since the methods on the type
    # aren't bound to anything, so Python treats them as regular functions
    fns = [
        getattr(cls, name)
        for name in cls.__dict__
        if inspect.isroutine(getattr(cls, name))
    ]
    # Skip built-ins, as they do not have global scope nor type hints
    # Needed to support `enum.Enum` derived classes in Python-3.11
    # That adds `_new_member_` property which is an alias to `__new__`
    # Skip __annotate__ added by PEP 649 for deferred annotation evaluation
    fns = [
        fn
        for fn in fns
        if not inspect.isbuiltin(fn)
        and hasattr(fn, "__globals__")
        and fn.__name__ != "__annotate__"
    ]
    captures = {}

    for fn in fns:
        captures.update(get_closure(fn))
        captures.update(get_type_hint_captures(fn))

    def lookup_in_class(key: str) -> Any:
        # [FUNCTION DOC] HELPER: Looks the key up in the captured mapping, falling back to builtins (None if missing).
        if key in captures:
            return captures[key]
        else:
            return getattr(builtins, key, None)

    return lookup_in_class


def boolean_dispatch(
    arg_name,
    arg_index,
    default,
    if_true,
    if_false,
    module_name,
    func_name,
):
    # [FUNCTION DOC] Creates a wrapper function that dispatches to if_true or if_false depending on the value of a boolean
    # argument (found by keyword name, else by position, else default). Copies the single available docstring to both
    # functions and the wrapper, sets the wrapper's module and name if given, and registers the wrapper in the
    # boolean_dispatched mapping with its metadata. Raises RuntimeError if both functions have docstrings.
    # Side effects: mutates docstrings of the inputs.
    """
    Dispatches to either of 2 script functions based on a boolean argument.
    In TorchScript, the boolean argument must be constant so that the correct
    function to use can be determined at compile time.
    """

    def fn(*args, **kwargs):
        # [FUNCTION DOC] HELPER: The wrapper; picks the flag from keyword, position or default and calls the matching function.
        dispatch_flag = default
        if arg_name in kwargs:
            dispatch_flag = kwargs[arg_name]
        elif arg_index < len(args):
            dispatch_flag = args[arg_index]

        if dispatch_flag:
            return if_true(*args, **kwargs)
        else:
            return if_false(*args, **kwargs)

    if if_true.__doc__ is None and if_false.__doc__ is not None:
        doc = if_false.__doc__
        if_true.__doc__ = doc
    elif if_false.__doc__ is None and if_true.__doc__ is not None:
        doc = if_true.__doc__
        if_false.__doc__ = doc
    elif if_false.__doc__ is None and if_true.__doc__ is None:
        # neither function has a docstring
        doc = None
    else:
        raise RuntimeError("only one function can have a docstring")
    fn.__doc__ = doc

    if module_name is not None:
        fn.__module__ = module_name
    if func_name is not None:
        fn.__name__ = func_name

    boolean_dispatched[fn] = {
        "if_true": if_true,
        "if_false": if_false,
        "index": arg_index,
        "default": default,
        "arg_name": arg_name,
    }
    return fn


class FunctionModifiers:
    # [CLASS DOC] Namespace of string constants marking how TorchScript treats a function: UNUSED, IGNORE, EXPORT,
    # DEFAULT, COPY_TO_SCRIPT_WRAPPER and the private _DROP. Compared by identity.
    """
    Used to denote the behavior of a function in TorchScript. See export() and
    ignore() for details.
    """

    UNUSED = "unused (ignored and replaced with raising of an exception)"
    IGNORE = "ignore (leave as a call to Python, cannot be torch.jit.save'd)"
    EXPORT = "export (compile this function even if nothing calls it)"
    DEFAULT = "default (compile if called from an exported function / forward)"
    COPY_TO_SCRIPT_WRAPPER = (
        "if this method is not scripted, copy the python method onto the scripted model"
    )
    _DROP = "_drop (function is fully ignored, declaration can be unscriptable)"


def export(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    # [FUNCTION DOC] Decorator marking a method as a TorchScript entry point by setting its modifier attribute to EXPORT.
    # Mutates and returns the same function.
    """
    This decorator indicates that a method on an ``nn.Module`` is used as an entry point into a
    :class:`ScriptModule` and should be compiled.

    .. deprecated:: 2.5
        Please use :func:`torch.compile` instead.

    ``forward`` implicitly is assumed to be an entry point, so it does not need this decorator.
    Functions and methods called from ``forward`` are compiled as they are seen
    by the compiler, so they do not need this decorator either.

    Example (using ``@torch.jit.export`` on a method):

    .. testcode::

        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            def implicitly_compiled_method(self, x):
                return x + 99

            # `forward` is implicitly decorated with `@torch.jit.export`,
            # so adding it here would have no effect
            def forward(self, x):
                return x + 10

            @torch.jit.export
            def another_forward(self, x):
                # When the compiler sees this call, it will compile
                # `implicitly_compiled_method`
                return self.implicitly_compiled_method(x)

            def unused_method(self, x):
                return x - 20

        # `m` will contain compiled methods:
        #     `forward`
        #     `another_forward`
        #     `implicitly_compiled_method`
        # `unused_method` will not be compiled since it was not called from
        # any compiled methods and wasn't decorated with `@torch.jit.export`
        m = torch.jit.script(MyModule())
    """
    fn._torchscript_modifier = FunctionModifiers.EXPORT  # type:ignore[attr-defined]
    return fn


def unused(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    # [FUNCTION DOC] Decorator marking a function to be replaced by raising an exception when scripted: sets the
    # modifier to UNUSED. For a property, marks the getter (and the setter if present) and returns the property.
    """
    This decorator indicates to the compiler that a function or method should
    be ignored and replaced with the raising of an exception. This allows you
    to leave code in your model that is not yet TorchScript compatible and still
    export your model.

        Example (using ``@torch.jit.unused`` on a method)::

            import torch
            import torch.nn as nn


            class MyModule(nn.Module):
                def __init__(self, use_memory_efficient):
                    super().__init__()
                    self.use_memory_efficient = use_memory_efficient

                @torch.jit.unused
                def memory_efficient(self, x):
                    import pdb

                    pdb.set_trace()
                    return x + 10

                def forward(self, x):
                    # Use not-yet-scriptable memory efficient mode
                    if self.use_memory_efficient:
                        return self.memory_efficient(x)
                    else:
                        return x + 10


            m = torch.jit.script(MyModule(use_memory_efficient=False))
            m.save("m.pt")

            m = torch.jit.script(MyModule(use_memory_efficient=True))
            # exception raised
            m(torch.rand(100))
    """
    if isinstance(fn, property):
        prop = fn
        setattr(  # noqa: B010
            prop.fget, "_torchscript_modifier", FunctionModifiers.UNUSED
        )

        if prop.fset:
            setattr(  # noqa: B010
                prop.fset, "_torchscript_modifier", FunctionModifiers.UNUSED
            )

        return prop

    fn._torchscript_modifier = FunctionModifiers.UNUSED  # type: ignore[attr-defined]
    return fn


# No op context manager from python side
class _IgnoreContextManager(contextlib.AbstractContextManager):
    # [CLASS DOC] HELPER: Context manager that does nothing and swallows nothing; used as a Python-side stand-in for the
    # ignore context manager.
    def __init__(self, **kwargs):
        # [FUNCTION DOC] HELPER: Accepts and ignores any keyword arguments.
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        # [FUNCTION DOC] HELPER: Does nothing; does not suppress exceptions.
        pass


def ignore(drop=False, **kwargs):
    # [FUNCTION DOC] Decorator (usable bare or with arguments) leaving a function as a Python call in TorchScript. If
    # given a callable, marks it IGNORE and returns it. Otherwise drop must be a bool (RuntimeError if not) and returns a
    # decorator that marks the function UNUSED when drop is true, else IGNORE. The deprecated drop_on_export keyword
    # overrides drop. Emits a FutureWarning for drop_on_export or drop being true.
    # KNOWN ISSUE: the warning texts end with a stray placeholder pair of braces that is never filled in.
    """
    This decorator indicates to the compiler that a function or method should
    be ignored and left as a Python function. This allows you to leave code in
    your model that is not yet TorchScript compatible. If called from TorchScript,
    ignored functions will dispatch the call to the Python interpreter. Models with ignored
    functions cannot be exported; use :func:`@torch.jit.unused <torch.jit.unused>` instead.

    .. deprecated:: 2.5
        Please use :func:`torch.compile` instead.

    Example (using ``@torch.jit.ignore`` on a method)::

        import torch
        import torch.nn as nn


        class MyModule(nn.Module):
            @torch.jit.ignore
            def debugger(self, x):
                import pdb

                pdb.set_trace()

            def forward(self, x):
                x += 10
                # The compiler would normally try to compile `debugger`,
                # but since it is `@ignore`d, it will be left as a call
                # to Python
                self.debugger(x)
                return x


        m = torch.jit.script(MyModule())

        # Error! The call `debugger` cannot be saved since it calls into Python
        m.save("m.pt")

    Example (using ``@torch.jit.ignore(drop=True)`` on a method):

    .. testcode::

        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            @torch.jit.ignore(drop=True)
            def training_method(self, x):
                import pdb
                pdb.set_trace()

            def forward(self, x):
                if self.training:
                    self.training_method(x)
                return x

        m = torch.jit.script(MyModule())

        # This is OK since `training_method` is not saved, the call is replaced
        # with a `raise`.
        m.save("m.pt")

    .. testcleanup::

        import os
        os.remove('m.pt')
    """

    if callable(drop):
        # used without any args, so drop is actually a function
        #   @torch.jit.ignore
        #   def fn(...):
        fn = drop
        # pyrefly: ignore [missing-attribute]
        fn._torchscript_modifier = FunctionModifiers.IGNORE
        return fn

    if not isinstance(drop, bool):
        raise RuntimeError(
            f"Argument to @torch.jit.ignore must be a bool or a function but got {drop}"
        )

    # for backwards compat
    drop_on_export = kwargs.pop("drop_on_export", None)
    if drop_on_export:
        warnings.warn(
            "ignore(drop_on_export=True) has been deprecated. TorchScript will now drop the function "
            "call on compilation. Use torch.jit.unused now. {}",
            stacklevel=2,
            category=FutureWarning,
        )

        drop = drop_on_export
    elif drop:
        warnings.warn(
            "ignore(True) has been deprecated. TorchScript will now drop the function "
            "call on compilation. Use torch.jit.unused now. {}",
            stacklevel=2,
            category=FutureWarning,
        )

    def decorator(fn):
        # [FUNCTION DOC] HELPER: Sets the modifier (UNUSED if drop, else IGNORE) on the function and returns it.
        if drop:
            fn._torchscript_modifier = FunctionModifiers.UNUSED
        else:
            fn._torchscript_modifier = FunctionModifiers.IGNORE
        return fn

    return decorator


def _drop(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    # [FUNCTION DOC] HELPER: Marks the function as fully dropped (private _DROP modifier) and returns it.
    fn._torchscript_modifier = FunctionModifiers._DROP  # type: ignore[attr-defined]
    return fn


def _copy_to_script_wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    # [FUNCTION DOC] HELPER: Marks the function to be copied onto the scripted module if not itself scripted; returns it.
    fn._torchscript_modifier = FunctionModifiers.COPY_TO_SCRIPT_WRAPPER  # type: ignore[attr-defined]
    return fn


def module_has_exports(mod):
    # [FUNCTION DOC] Returns true if any callable attribute of the module has the EXPORT modifier.
    for name in dir(mod):
        if hasattr(mod, name):
            item = getattr(mod, name)
            if callable(item):
                if get_torchscript_modifier(item) is FunctionModifiers.EXPORT:
                    return True
    return False


# WARNING: should_drop is currently being used by our JIT code coverage plug-in to mark JIT'd code as covered. If you
# rename this function, please update references in tools/coverage_plugins_package/src/coverage_plugins/jit_plugin.py to
# allow JIT'd code to still be covered.
def should_drop(fn) -> bool:
    # [FUNCTION DOC] Returns true if the function's modifier is UNUSED or _DROP (its body should be dropped); false for
    # non-callables. Name is referenced by an external coverage plugin and must not be renamed.
    attr = get_torchscript_modifier(fn)
    if attr is None:
        return False
    return attr is FunctionModifiers.UNUSED or attr is FunctionModifiers._DROP


def is_ignored_fn(fn) -> bool:
    # [FUNCTION DOC] Returns true if the function's modifier is UNUSED, IGNORE or _DROP.
    mod = get_torchscript_modifier(fn)
    return (
        mod is FunctionModifiers.UNUSED
        or mod is FunctionModifiers.IGNORE
        or mod is FunctionModifiers._DROP
    )


def _is_drop_fn(fn) -> bool:
    # [FUNCTION DOC] HELPER: Returns true if the function's modifier is _DROP.
    mod = get_torchscript_modifier(fn)
    return mod is FunctionModifiers._DROP


def is_static_fn(cls, fn) -> bool:
    # [FUNCTION DOC] Returns true if the named attribute of the class is a static method (found without triggering
    # descriptors).
    return isinstance(inspect.getattr_static(cls, fn, default=None), staticmethod)


def get_static_fn(cls, fn):
    # [FUNCTION DOC] Returns the underlying plain function of the named static method of the class.
    return inspect.getattr_static(cls, fn).__func__


def get_torchscript_modifier(fn):
    # [FUNCTION DOC] Returns the TorchScript modifier of a callable (unwrapping bound methods), DEFAULT if unmarked, or
    # None if the object is not callable.
    if not callable(fn):
        return None
    if hasattr(fn, "__func__"):
        fn = fn.__func__
    return getattr(fn, "_torchscript_modifier", FunctionModifiers.DEFAULT)


def copy_torchscript_modifier(orig, new) -> None:
    # [FUNCTION DOC] Copies the TorchScript modifier from one function to another unless the source is not callable.
    attr = get_torchscript_modifier(orig)
    if attr is None:
        return
    new._torchscript_modifier = attr


# overloading registration
# overloads get registered in this file, and compiled in torch/jit/__init__.py
# so that they can be imported in nn/functional.py without an import cycle

# qualified_name => list[overload_functions]
_overloaded_fns: dict[str, list[Callable]] = {}  # noqa: T484


_OVERLOAD_EXAMPLE = """
Example usage of overload function:
@torch.jit._overload
def my_function(x: type0) -> type0: # decl 1
    pass

@torch.jit._overload
def my_function(x: type1) -> type1: # decl 2
    pass

def my_function(x):                 # implementation
    if isinstance(x, type0):
        return x
    elif isinstance(x, type1):
        return x
"""


def get_overload_no_implementation_error_message(kind, obj):
    # [FUNCTION DOC] Builds the error message text for an overload declaration lacking an implementation, including the
    # source location and lines of the object and the overload usage example.
    sourcelines, file_lineno, filename = get_source_lines_and_file(obj)
    return (
        f'Implementation for the {kind} "{_qualified_name(obj)}" is missing. Please make '
        f"sure a definition is provided and defined after all overload declarations.\n"
        f'File "{filename}", line {file_lineno}:\n'
        + "".join(sourcelines)
        + "\n"
        + _OVERLOAD_EXAMPLE
    )


def _check_overload_body(func):
    # [FUNCTION DOC] HELPER: Checks that an overload declaration's body is just pass or an ellipsis; raises RuntimeError
    # with source context otherwise. If the source cannot be read it only emits a warning and returns.
    try:
        parsed_def = parse_def(func)
    except OSError:
        # Parsing the function definition can raise an OSError if source is unavailable.
        # Since this is just an initial check, just raise a warning if this is the case.
        warnings.warn(
            f"Unable to retrieve source for @torch.jit._overload function: {func}.",
            stacklevel=2,
        )
        return

    body = parsed_def.ast.body[0].body

    def is_pass(x):
        # [FUNCTION DOC] HELPER: Returns true if the AST node is a pass statement.
        return isinstance(x, ast.Pass)

    def is_ellipsis(x):
        # [FUNCTION DOC] HELPER: Returns true if the AST node is an expression statement consisting of an ellipsis constant.
        return (
            isinstance(x, ast.Expr)
            and isinstance(x.value, ast.Constant)
            and x.value.value is Ellipsis
        )

    if len(body) != 1 or not (is_pass(body[0]) or is_ellipsis(body[0])):
        msg = (
            "Only `pass` statement or `...` can be the body of overload declaration:\n"
        )
        msg += "\n".join(parsed_def.source.split("\n")[:3])
        msg += " <- Expecting `pass` or `...` here!\n" + _OVERLOAD_EXAMPLE
        raise RuntimeError(msg)


def _overload(func: Callable[_P, _R]) -> Callable[_P, _R]:
    # [FUNCTION DOC] HELPER: Decorator registering a function as an overload declaration under its qualified name in the
    # global overload registry after validating its body; returns the function unchanged.
    _check_overload_body(func)
    qual_name = _qualified_name(func)
    global _overloaded_fns
    fn_overload_list = _overloaded_fns.get(qual_name)
    if fn_overload_list is None:
        fn_overload_list = []
        _overloaded_fns[qual_name] = fn_overload_list
    fn_overload_list.append(func)
    return func


def _get_fn_overloads(qual_name):
    # [FUNCTION DOC] HELPER: Returns the registered list of overload declarations for the qualified name, or None.
    return _overloaded_fns.get(qual_name)


def _clear_fn_overloads(qual_name) -> None:
    # [FUNCTION DOC] HELPER: Removes the overload list for the qualified name; raises KeyError if absent.
    del _overloaded_fns[qual_name]


def get_class_name_lineno(method) -> tuple[str, int]:
    # [FUNCTION DOC] Walks two frames up the stack and returns the name and first line number of the code object found
    # there (the class body that is declaring an overloaded method). The method argument is unused. Raises AssertionError
    # if the stack is too shallow.
    current_frame = inspect.currentframe()

    # one for the get_class_name call, one for _overload_method call
    for i in range(2):
        if current_frame is None:
            raise AssertionError(f"current_frame is None at iteration {i}")
        current_frame = current_frame.f_back

    if current_frame is None:
        raise AssertionError("current_frame is None after traversing frames")
    class_name = current_frame.f_code.co_name
    line_no = current_frame.f_code.co_firstlineno
    return class_name, line_no


# At the point the decorator is applied to class methods the method
# has no reference to its owning class. _qualified_name would not include
# the class it is defined in, so any methods with the same name in the same file
# would have the same _qualified_name, even if they were defined in different
# classes. This problem only exists in python 2.
# We get around this problem by looking at the stack frame and identifying
# the class name, and throwing an error whenever overloads are used
# when modules of the same name are in the same file

# qualified_name => class name => list[overload_functions]
_overloaded_methods: dict[str, dict[str, list[Callable]]] = {}  # noqa: T484


# (qualified_name, class name) => class_fileno
_overloaded_method_class_fileno: dict[tuple[str, str], int] = {}


def _overload_method(func: Callable[_P, _R]) -> Callable[_P, _R]:
    # [FUNCTION DOC] HELPER: Decorator registering a method as an overload declaration, grouped by qualified name and then
    # by the name of the defining class (found from the call stack). Validates the body (the check is skipped with a
    # warning on CPython 3.13.8 when an indentation error occurs from a source bug; other versions re-raise). Raises
    # RuntimeError if the same method name is overloaded in two different same-named classes at different lines.
    # Returns the function unchanged; mutates module-level registries.
    try:
        _check_overload_body(func)
    except IndentationError:
        # CPython 3.13.8 has a bug (https://github.com/python/cpython/issues/139783)
        # where inspect.getsourcelines() returns truncated source when a decorator
        # is followed by a comment, causing ast.parse() to fail with IndentationError.
        # Fixed in 3.13.9. Swallow the error on affected versions; re-raise otherwise.
        if sys.version_info[:3] == (3, 13, 8):
            import warnings

            warnings.warn(
                "Skipping overload body check due to a known CPython 3.13.8 bug "
                "(https://github.com/python/cpython/issues/139783). "
                "Consider upgrading to Python 3.13.9+.",
                stacklevel=2,
            )
        else:
            raise
    qual_name = _qualified_name(func)
    global _overloaded_methods
    class_name_map = _overloaded_methods.get(qual_name)
    if class_name_map is None:
        class_name_map = {}
        _overloaded_methods[qual_name] = class_name_map

    class_name, line_no = get_class_name_lineno(func)
    method_overloads = class_name_map.get(class_name)
    if method_overloads is None:
        method_overloads = []
        class_name_map[class_name] = method_overloads
        _overloaded_method_class_fileno[(qual_name, class_name)] = line_no
    else:
        existing_lineno = _overloaded_method_class_fileno[(qual_name, class_name)]
        if existing_lineno != line_no:
            raise RuntimeError(
                "Cannot currently overload the same method name in two different"
                " classes with the same name in the same module"
            )

    method_overloads.append(func)
    return func


def _get_overloaded_methods(method, mod_class):
    # [FUNCTION DOC] HELPER: Returns the registered overload list for a method of the given module class, or None if the
    # method has no name or no overloads. Raises AssertionError if the method's source line lies outside the class's
    # source range (module redeclared in the same file).
    # TODO: __name__ not set for submodules in recursive script
    if not hasattr(method, "__name__"):
        return None
    qual_name = _qualified_name(method)
    class_name_map = _overloaded_methods.get(qual_name)
    if class_name_map is None:
        return None
    overloads = class_name_map.get(mod_class.__name__, None)
    if overloads is None:
        return None

    method_line_no = get_source_lines_and_file(method)[1]
    mod_class_fileno = get_source_lines_and_file(mod_class)[1]
    mod_end_fileno = mod_class_fileno + len(get_source_lines_and_file(mod_class)[0])
    if not (method_line_no >= mod_class_fileno and method_line_no <= mod_end_fileno):
        raise AssertionError(
            "Overloads are not usable when a module is redeclared within the same file: "
            + str(method)
        )
    return overloads


def is_tuple(ann) -> bool:
    # [FUNCTION DOC] Returns true if the annotation is a tuple type (builtin or typing origin). Raises RuntimeError if the
    # bare typing Tuple is used without element types.
    # Check for typing.Tuple missing args (but `tuple` is fine)
    if ann is typing.Tuple:  # noqa: UP006
        raise_error_container_parameter_missing("Tuple")

    # For some reason Python 3.7 violates the Type[A, B].__origin__ == Type rule
    if not hasattr(ann, "__module__"):
        return False

    ann_origin = get_origin(ann)
    return ann.__module__ in ("builtins", "typing") and ann_origin is tuple


def is_list(ann) -> bool:
    # [FUNCTION DOC] Returns true if the annotation is a list type (builtin or typing origin). Raises RuntimeError for the
    # bare typing List.
    # Check for typing.List missing args (but `list` is fine)
    if ann is typing.List:  # noqa: UP006
        raise_error_container_parameter_missing("List")

    if not hasattr(ann, "__module__"):
        return False

    ann_origin = get_origin(ann)
    return ann.__module__ in ("builtins", "typing") and ann_origin is list


def is_dict(ann) -> bool:
    # [FUNCTION DOC] Returns true if the annotation is a dict type (builtin or typing origin). Raises RuntimeError for the
    # bare typing Dict.
    # Check for typing.Dict missing args (but `dict` is fine)
    if ann is typing.Dict:  # noqa: UP006
        raise_error_container_parameter_missing("Dict")

    if not hasattr(ann, "__module__"):
        return False

    ann_origin = get_origin(ann)
    return ann.__module__ in ("builtins", "typing") and ann_origin is dict


def is_union(ann):
    # [FUNCTION DOC] Returns true if the annotation is a Union (typing form or the builtin union type). Raises
    # RuntimeError for the bare Union.
    if ann is Union:
        raise_error_container_parameter_missing("Union")

    return isinstance(ann, BuiltinUnionType) or (
        hasattr(ann, "__module__")
        and ann.__module__ == "typing"
        and (get_origin(ann) is Union)
    )


def is_optional(ann):
    # [FUNCTION DOC] Returns true if the annotation is Optional, either directly or as a two-member union containing None.
    # Raises RuntimeError for the bare Optional.
    # KNOWN ISSUE: the direct Optional check compares the origin to Optional, which typing never reports, so only the
    # union form actually matches.
    if ann is Optional:
        raise_error_container_parameter_missing("Optional")

    def is_optional_as_optional(ann):
        # [FUNCTION DOC] HELPER: True if the annotation's typing origin is Optional.
        return (
            hasattr(ann, "__module__")
            and ann.__module__ == "typing"
            and (get_origin(ann) is Optional)
        )

    def is_union_as_optional(ann):
        # [FUNCTION DOC] HELPER: True if the annotation has exactly two arguments, one of which is None.
        ann_args = get_args(ann)
        return len(ann_args) == 2 and (None in ann_args or type(None) in ann_args)

    return is_optional_as_optional(ann) or (is_union(ann) and is_union_as_optional(ann))


def is_future(ann) -> bool:
    # [FUNCTION DOC] Returns true if the annotation is a subscripted Future type. Raises RuntimeError for the bare Future.
    if ann is Future:
        raise RuntimeError(
            "Attempted to use Future without a "
            "contained type. Please add a contained type, e.g. "
            "Future[int]"
        )
    return get_origin(ann) is Future


def is_await(ann) -> bool:
    # [FUNCTION DOC] Returns true if the annotation is the Await type or a subscripted Await.
    if ann is _Await:
        return True
    return get_origin(ann) is _Await


if torch.distributed.rpc.is_available():
    from torch._C._distributed_rpc import PyRRef
    from torch.distributed.rpc import RRef

    def is_rref(ann) -> bool:
        # [FUNCTION DOC] Returns true if the annotation is a subscripted RRef. Raises RuntimeError for the bare RRef. Defined
        # only when distributed RPC is available.
        if ann is RRef:
            raise RuntimeError(
                "Attempted to use RRef without a "
                "contained type. Please add a contained type, e.g. "
                "RRef[int]"
            )
        return get_origin(ann) is RRef

    def is_rref_instance(obj) -> bool:
        # [FUNCTION DOC] Returns true if the object is a remote reference instance.
        return isinstance(obj, PyRRef)

else:

    def is_rref_instance(obj) -> bool:
        # [FUNCTION DOC] Fallback when RPC is unavailable: always returns false.
        # If the RPC module doesn't exist then RRefs don't exist either.
        return False


def _try_get_dispatched_fn(fn):
    # [FUNCTION DOC] HELPER: Returns the boolean-dispatch metadata registered for a callable, or None if not callable or
    # unregistered.
    if not callable(fn):
        return None
    return boolean_dispatched.get(fn)


def _get_named_tuple_properties(
    obj,
    loc: torch._C._jit_tree_views.SourceRange | None = None,
    rcb=None,
):
    # [FUNCTION DOC] HELPER: Extracts namedtuple info for TorchScript: returns the class name, field names, a list of
    # TorchScript types per field (inferred tensor type when unannotated; string forward references resolved through rcb),
    # and the default values. Annotations are taken from the class or its base. Raises AssertionError if the object is not
    # a namedtuple class; raises ValueError for a forward reference that cannot be resolved.
    if loc is None:
        loc = fake_range()

    if not issubclass(obj, tuple) or not hasattr(obj, "_fields"):
        raise AssertionError(
            f"expected namedtuple (tuple subclass with _fields), got {obj}"
        )
    if hasattr(obj, "_field_defaults"):
        defaults = [
            obj._field_defaults[field]
            for field in obj._fields
            if field in obj._field_defaults
        ]
    else:
        defaults = []

    obj_annotations = inspect.get_annotations(obj)
    if len(obj_annotations) == 0 and hasattr(obj, "__base__"):
        obj_annotations = inspect.get_annotations(
            # pyrefly: ignore [bad-argument-type]
            obj.__base__
        )

    annotations = []
    for field in obj._fields:
        if field in obj_annotations:
            field_type = obj_annotations[field]
            # [Note: ForwardRef annotations in NamedTuple attributes]
            # NamedTuple types are slightly different from normal types.
            #
            # Normally, annotations are evaluated like this (during jit.script):
            # 1. Load strings of python code into c++ and parse.
            # 2. Get annotations as strings
            # 3. Use the PythonResolver's resolution callback (rcb) to convert
            #    the string into a python object
            # 4. We call into annotations.py:ann_to_type to convert python obj
            #    from step 3 into a type that torchscript understands.
            #
            # NamedTuples are more complicated, because it has sub-types.
            # Normally, once we have the NamedTuple type object from #3,
            # we can just look at the annotation literal values and use
            # ann_to_type directly on them.
            #
            # But sometimes, users will annotate with string literals, e.g.
            #    x: 'int'
            # This also happens with PEP563 (from __forward__ import annotations)
            #
            # These annotations appear in the annotation dict as ForwardRef('int').
            #
            # Then, we need to convert the string into a python object. This
            # requires having local context for custom objects or imported types.
            # rcb() is what gives us this. So, we plumb rcb through the stack so
            # it can be used in this context for the if block below.
            #
            # FAQ:
            # - Why do we need this special handling for NamedTuple but string
            #   annotations work fine for normal types? Normally, we parse the
            #   string directly and then call rcb() directly from C++.
            # - Why not use ForwardRef._evaluate? For that, we need globals()
            #   and locals() for the local context where the NamedTuple was defined.
            #   rcb is what lets us look up into these. So, basically rcb does the
            #   hard work for us.
            if isinstance(field_type, ForwardRef) and rcb is not None:
                rcb_type = rcb(field_type.__forward_arg__)
                # rcb returns None if it can't find anything.
                if rcb_type is None:
                    raise ValueError(
                        f"Unknown type annotation: '{field_type}' in NamedTuple {obj.__name__}."
                        f" Likely due to partial support for ForwardRef parameters in NamedTuples, see #95858."
                        f" Issue occurred at {loc.highlight()}"
                    )
                field_type = rcb_type
            the_type = torch.jit.annotations.ann_to_type(field_type, loc, rcb)
            annotations.append(the_type)
        else:
            annotations.append(torch._C.TensorType.getInferred())
    return type(obj).__name__, obj._fields, annotations, defaults


def _create_named_tuple(
    t,
    unqual_name: str,
    field_names: list[str],
    defaults: tuple[Any, ...],
):
    # [FUNCTION DOC] HELPER: Creates a namedtuple class with the given name, fields and defaults and instantiates it from
    # the sequence t. The first argument is the values; no caching of the class.
    TupleType = collections.namedtuple(unqual_name, field_names, defaults=defaults)  # type: ignore[call-arg, no-redef, misc]
    return TupleType(*t)


@contextlib.contextmanager
def _disable_emit_hooks():
    # [FUNCTION DOC] HELPER: Context manager that temporarily clears the JIT emit hooks and restores the previous hooks on
    # exit, even on error.
    hooks = torch._C._jit_get_emit_hooks()
    torch._C._jit_set_emit_hooks(None, None)
    try:
        yield
    finally:
        torch._C._jit_set_emit_hooks(hooks[0], hooks[1])


def _disable_emit_hooks_decorator(_DecoratorContextManager) -> None:
    def __enter__(self) -> None:
        # [FUNCTION DOC] HELPER: Intended enter method; saves current emit hooks on self and clears them.
        # KNOWN ISSUE: the enclosing function only defines these two nested functions and returns None, so they are dead code.
        self.hooks = torch._C._jit_get_emit_hooks()
        torch._C._jit_set_emit_hooks(None, None)

    def __exit__(self, *args) -> None:
        # [FUNCTION DOC] HELPER: Intended exit method; restores the saved emit hooks.
        torch._C._jit_set_emit_hooks(self.hooks[0], self.hooks[1])


def _is_exception(obj) -> bool:
    # [FUNCTION DOC] HELPER: Returns true if the object is a class deriving from Exception.
    if not inspect.isclass(obj):
        return False
    return issubclass(obj, Exception)


def raise_error_container_parameter_missing(target_type) -> None:
    # [FUNCTION DOC] Always raises RuntimeError telling the user a container type needs contained types; the message
    # differs for dictionary-like names (ending in "ict", which takes two types) versus others.
    if target_type.endswith("ict"):
        raise RuntimeError(
            f"Attempted to use {target_type} without "
            "contained types. Please add contained type, e.g. "
            f"{target_type}[int, int]"
        )
    raise RuntimeError(
        f"Attempted to use {target_type} without a "
        "contained type. Please add a contained type, e.g. "
        f"{target_type}[int]"
    )


_RAW_TYPE_NAME_MAPPING = {
    dict: "dict",
    list: "list",
    tuple: "tuple",
    typing.Dict: "Dict",  # noqa: UP006
    typing.List: "List",  # noqa: UP006
    typing.Optional: "Optional",
    typing.Tuple: "Tuple",  # noqa: UP006
}


def check_args_exist(target_type) -> None:
    # [FUNCTION DOC] Raises RuntimeError (via the helper) if the target is one of the raw container types listed in the
    # name mapping used without arguments; otherwise does nothing.
    if name := _RAW_TYPE_NAME_MAPPING.get(target_type):
        raise_error_container_parameter_missing(name)


def check_empty_containers(obj) -> None:
    # [FUNCTION DOC] Emits a warning if the object is an empty list, dict or tuple, because the inner type information is
    # lost in eager mode.
    if obj == [] or obj == {} or obj == ():
        warnings.warn(
            "The inner type of a container is lost when "
            "calling torch.jit.isinstance in eager mode. For "
            "example, List[int] would become list and "
            "therefore falsely return True for List[float] or"
            " List[str].",
            stacklevel=2,
        )


# supports List/Dict/Tuple and Optional types
# TODO support future
def container_checker(obj, target_type) -> bool:
    # [FUNCTION DOC] Checks an object against a parameterized List, Dict, Tuple, Union or Optional type, recursing into
    # nested containers; returns false for unsupported origins. Empty-container warnings may be emitted. Raises
    # RuntimeError for raw container types.
    # KNOWN ISSUE: in the Union case it returns the result of the first inner type that is itself a container without
    # trying remaining members.
    origin_type = get_origin(target_type)
    check_args_exist(target_type)
    if origin_type is None:
        return False
    elif origin_type is list or origin_type is typing.List:  # noqa: UP006
        check_empty_containers(obj)
        if not isinstance(obj, list):
            return False
        arg_type = get_args(target_type)[0]
        arg_origin = get_origin(arg_type)
        for el in obj:
            # check if nested container, ex: List[List[str]]
            if arg_origin:  # processes nested container, ex: List[List[str]]
                if not container_checker(el, arg_type):
                    return False
            elif not isinstance(el, arg_type):
                return False
        return True
    elif origin_type is typing.Dict or origin_type is dict:  # noqa: UP006
        check_empty_containers(obj)
        if not isinstance(obj, dict):
            return False
        key_type = get_args(target_type)[0]
        val_type = get_args(target_type)[1]
        for key, val in obj.items():
            # check if keys are of right type
            if not isinstance(key, key_type):
                return False
            val_origin = get_origin(val_type)
            if val_origin:
                if not container_checker(val, val_type):
                    return False
            elif not isinstance(val, val_type):
                return False
        return True
    elif origin_type is typing.Tuple or origin_type is tuple:  # noqa: UP006
        check_empty_containers(obj)
        if not isinstance(obj, tuple):
            return False
        arg_types = get_args(target_type)
        if len(obj) != len(arg_types):
            return False
        for el, el_type in zip(obj, arg_types):
            el_origin = get_origin(el_type)
            if el_origin:
                if not container_checker(el, el_type):
                    return False
            elif not isinstance(el, el_type):
                return False
        return True
    elif origin_type is Union or issubclass(
        origin_type,
        BuiltinUnionType,
    ):  # also handles Optional
        if obj is None:  # check before recursion because None is always fine
            return True
        inner_types = get_args(target_type)
        for t in inner_types:
            t_origin = get_origin(t)
            if t_origin:
                return container_checker(obj, t)
            elif isinstance(obj, t):
                return True
    return False


def _isinstance(obj, target_type) -> bool:
    # [FUNCTION DOC] HELPER: Eager-mode implementation of the jit isinstance check. Accepts a type or tuple of types (any
    # other container raises RuntimeError); parameterized types go through container_checker; raw container types raise
    # via check_args_exist; otherwise uses the built-in instance check.
    if isinstance(target_type, collections.abc.Container):
        if not isinstance(target_type, tuple):
            raise RuntimeError(
                "The second argument to "
                "`torch.jit.isinstance` must be a type "
                "or a tuple of types"
            )
        for t_type in target_type:
            if _isinstance(obj, t_type):
                return True
        return False

    origin_type = get_origin(target_type)
    if origin_type:
        return container_checker(obj, target_type)

    # Check to handle non-typed optional origin returns as none instead
    #    of as optional in 3.7-3.8
    check_args_exist(target_type)

    # handle non-containers
    return isinstance(obj, target_type)


class _TensorExtractor(pickle.Pickler):
    # [CLASS DOC] HELPER: Pickler subclass that collects every tensor in an object graph into a caller-supplied list. State:
    # tensors. Tensors, locks, futures, remote references, awaits, CUDA events and threads are replaced by an empty
    # persistent id so they are not serialized.
    def __init__(self, *args, tensors: list[torch.Tensor], **kwargs):
        # [FUNCTION DOC] HELPER: Passes arguments to the base pickler and stores the list that will receive tensors.
        super().__init__(*args, **kwargs)
        self.tensors = tensors

    def persistent_id(self, obj):
        # [FUNCTION DOC] HELPER: Pickle hook: appends tensors to the list and emits an empty id for tensors and the listed
        # unpicklable-but-tensor-free types; returns None (normal pickling) for everything else.
        if isinstance(obj, torch.Tensor):
            self.tensors.append(obj)
            return ""
        # Since we just want to extract tensors, we don't mind if an object is
        # unpicklable if it doesn't contain tensors, as we can just ignore/skip
        # it. To play it safe, we only do so for common objects that we're sure
        # don't contain tensors. Feel free to add new types here. Note also that
        # even if a type isn't listed here this won't block users, since they
        # can just add a __getstate__ or __reduce__ method to their class.
        if isinstance(obj, LockType):
            return ""
        # Futures and RRefs don't technically contain a value, they just offer
        # the means to access a value.
        if isinstance(obj, CFuture) or is_rref_instance(obj):
            return ""
        if isinstance(obj, CAwait):
            return ""
        if isinstance(obj, torch.cuda.Event):
            return ""
        if isinstance(obj, threading.Thread):
            return ""
        return None


def _extract_tensors(obj):
    # [FUNCTION DOC] HELPER: Called from C++. Returns the list of tensors found by pickling the object with the tensor
    # extractor to an in-memory buffer.
    r"""
    This function is exclusively called from C++.
    See ``torch/csrc/jit/python/python_ivalue.h``.

    It extracts the tensors contained in the given object, through pickling.
    """
    tensors: list[torch.Tensor] = []
    extractor = _TensorExtractor(io.BytesIO(), protocol=-1, tensors=tensors)
    extractor.dump(obj)
    return tensors


def _get_model_id(obj) -> str | None:
    # [FUNCTION DOC] HELPER: Returns an id string for a script module (its C++ type string) or script function (its
    # qualified name), else None.
    if isinstance(obj, torch.jit.ScriptModule):
        return str(obj._c._type())
    elif isinstance(obj, torch.jit.ScriptFunction):
        return obj.qualified_name
    else:
        return None


# In Python-3.11+ typed enums (i.e. IntEnum for example) retain number of base class methods in subclass
# that were previously dropped. To preserve the behavior, explicitly drop them there

if sys.version_info >= (3, 11):
    _drop(enum.Enum.__new__)
    _drop(enum.Enum.__format__)
    _drop(enum.Enum.__repr__)
    _drop(enum.Enum.__str__)
