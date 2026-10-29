# mypy: allow-untyped-defs
# [MODULE DOC] Decorators and context managers for DataPipes: a functional-name registration
# decorator, global determinism guard and non_deterministic marker, argument type validation,
# and runtime output-type validation. Holds two module-level flags (determinism, default off;
# runtime validation, default on).
import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any, get_type_hints

from torch.utils.data.datapipes._typing import _DataPipeMeta
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe


######################################################
# Functional API
######################################################
class functional_datapipe:
    # [CLASS DOC] Class decorator that registers a DataPipe class as a method on IterDataPipe or
    # MapDataPipe under the given name. Stores name and the dataframe-tracing flag.
    name: str

    def __init__(self, name: str, enable_df_api_tracing=False) -> None:
        # [FUNCTION DOC] Stores the functional name and the flag enabling DataFrame API tracing
        # for registered iterable DataPipes.
        """
        Define a functional datapipe.

        Args:
            enable_df_api_tracing - if set, any returned DataPipe would accept
            DataFrames API in tracing mode.
        """
        self.name = name
        self.enable_df_api_tracing = enable_df_api_tracing

    def __call__(self, cls):
        # [FUNCTION DOC] Registers cls under the stored name on IterDataPipe or MapDataPipe
        # (mutating that class) and returns cls unchanged. For iterable pipes the target must
        # be a class using the DataPipe metaclass or a non_deterministic-wrapped object, else
        # raises TypeError. Other types are silently not registered.
        if issubclass(cls, IterDataPipe):
            if isinstance(cls, type):  # type: ignore[arg-type]
                if not isinstance(cls, _DataPipeMeta):
                    raise TypeError(
                        "`functional_datapipe` can only decorate IterDataPipe"
                    )
            # with non_deterministic decorator
            else:
                if not isinstance(cls, non_deterministic) and not (
                    hasattr(cls, "__self__")
                    and isinstance(cls.__self__, non_deterministic)
                ):
                    raise TypeError(
                        "`functional_datapipe` can only decorate IterDataPipe"
                    )
            IterDataPipe.register_datapipe_as_function(
                self.name, cls, enable_df_api_tracing=self.enable_df_api_tracing
            )
        elif issubclass(cls, MapDataPipe):
            MapDataPipe.register_datapipe_as_function(self.name, cls)

        return cls


######################################################
# Determinism
######################################################
_determinism: bool = False


class guaranteed_datapipes_determinism:
    # [CLASS DOC] Context manager that sets the global determinism flag on construction (not on
    # entering) and restores the previous value on exit. While set, non_deterministic
    # DataPipes raise an error when invoked.
    prev: bool

    def __init__(self) -> None:
        # [FUNCTION DOC] Saves the current global determinism flag and sets it to true.
        global _determinism
        self.prev = _determinism
        _determinism = True

    def __enter__(self) -> None:
        # [FUNCTION DOC] No-op; the flag was already set in the constructor.
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        # [FUNCTION DOC] Restores the saved determinism flag; does not suppress exceptions.
        global _determinism
        _determinism = self.prev


class non_deterministic:
    # [CLASS DOC] Decorator marking an IterDataPipe class as non-deterministic. Either wraps a
    # class directly, or wraps a predicate function (given as argument) that decides per
    # call, from the constructor arguments, whether the instance is non-deterministic.
    cls: type[IterDataPipe] | None = None
    # TODO: Lambda for picking
    deterministic_fn: Callable[..., bool]

    def __init__(self, arg: type[IterDataPipe] | Callable[..., bool]) -> None:
        # [FUNCTION DOC] Accepts an IterDataPipe subclass (stored as cls) or a callable
        # predicate (stored as deterministic_fn). Raises TypeError for a non-IterDataPipe
        # class or for any other argument type.
        # 1. Decorator doesn't have any argument
        if isinstance(arg, type):  # type: ignore[arg-type]
            if not issubclass(arg, IterDataPipe):  # type: ignore[arg-type]
                raise TypeError(
                    "Only `IterDataPipe` can be decorated with `non_deterministic`"
                    f", but {arg.__name__} is found"
                )
            self.cls = arg  # type: ignore[assignment]
        # 2. Decorator has an argument of a function
        #    This class should behave differently given different inputs. Use this
        #    function to verify the determinism for each instance.
        #    When the function returns True, the instance is non-deterministic. Otherwise,
        #    the instance is a deterministic DataPipe.
        elif isinstance(arg, Callable):  # type:ignore[arg-type]
            self.deterministic_fn = arg
        else:
            raise TypeError(f"{arg} can not be decorated by non_deterministic")

    def __call__(self, *args, **kwargs):
        # [FUNCTION DOC] When wrapping a class, acts as its constructor: raises TypeError if
        # global determinism is enabled, else returns a new instance. When wrapping a
        # predicate, receives the class to decorate (first positional argument, must be an
        # IterDataPipe subclass else TypeError), stores it and returns the wrapper function.
        global _determinism
        #  Decorate IterDataPipe
        if self.cls is not None:
            if _determinism:
                raise TypeError(
                    f"{self.cls.__name__} is non-deterministic, but you set 'guaranteed_datapipes_determinism'. "
                    "You can turn off determinism for this DataPipe if that is acceptable "
                    "for your application"
                )
            return self.cls(*args, **kwargs)  # type: ignore[call-arg]

        # Decorate with a functional argument
        if not (
            isinstance(args[0], type) and issubclass(args[0], IterDataPipe)  # type: ignore[arg-type]
        ):
            raise TypeError(
                f"Only `IterDataPipe` can be decorated, but {args[0].__name__} is found"
            )
        self.cls = args[0]
        return self.deterministic_wrapper_fn

    def deterministic_wrapper_fn(self, *args, **kwargs) -> IterDataPipe:
        # [FUNCTION DOC] Calls the predicate with the given arguments; raises TypeError if it does
        # not return a bool, or if it returns true while global determinism is enabled.
        # Otherwise constructs and returns an instance of the wrapped class.
        res = self.deterministic_fn(*args, **kwargs)
        if not isinstance(res, bool):
            raise TypeError(
                "deterministic_fn of `non_deterministic` decorator is required "
                f"to return a boolean value, but {type(res)} is found"
            )
        global _determinism
        if _determinism and res:
            raise TypeError(
                f"{self.cls.__name__} is non-deterministic with the inputs, but you set "  # type: ignore[union-attr]
                "'guaranteed_datapipes_determinism'. You can turn off determinism "
                "for this DataPipe if that is acceptable for your application"
            )
        return self.cls(*args, **kwargs)  # type: ignore[call-arg, misc]


######################################################
# Type validation
######################################################
# Validate each argument of DataPipe with hint as a subtype of the hint.
def argument_validation(f):
    # [FUNCTION DOC] Decorator for DataPipe methods that checks arguments annotated with a
    # DataPipe type. Returns a wrapper that binds the call arguments and, for each hinted
    # argument, raises TypeError if the value is not an IterDataPipe or its element type is
    # not a subtype of the hint, then calls the original function.
    signature = inspect.signature(f)
    hints = get_type_hints(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Validation wrapper described in argument_validation; returns
        # the wrapped function's result.
        bound = signature.bind(*args, **kwargs)
        for argument_name, value in bound.arguments.items():
            if argument_name in hints and isinstance(
                hints[argument_name], _DataPipeMeta
            ):
                hint = hints[argument_name]
                if not isinstance(value, IterDataPipe):
                    raise TypeError(
                        f"Expected argument '{argument_name}' as an IterDataPipe, but found {type(value)}"
                    )
                if not value.type.issubtype(hint.type):
                    raise TypeError(
                        f"Expected type of argument '{argument_name}' as a subtype of "
                        f"hint {hint.type}, but found {value.type}"
                    )

        return f(*args, **kwargs)

    return wrapper


# Default value is True
_runtime_validation_enabled: bool = True


class runtime_validation_disabled:
    # [CLASS DOC] Context manager that disables runtime output type validation (sets the global
    # flag to false on construction) and restores the previous value on exit.
    prev: bool

    def __init__(self) -> None:
        # [FUNCTION DOC] Saves the current runtime-validation flag and sets it to false.
        global _runtime_validation_enabled
        self.prev = _runtime_validation_enabled
        _runtime_validation_enabled = False

    def __enter__(self) -> None:
        # [FUNCTION DOC] No-op; the flag was already changed in the constructor.
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        # [FUNCTION DOC] Restores the saved runtime-validation flag; does not suppress
        # exceptions.
        global _runtime_validation_enabled
        _runtime_validation_enabled = self.prev


# Runtime checking
# Validate output data is subtype of return hint
def runtime_validation(f):
    # [FUNCTION DOC] Decorator for an __iter__ method that validates each yielded item against
    # the DataPipe's declared element type. Raises TypeError at decoration time if the
    # function is not named __iter__. Returns a generator wrapper; at iteration it raises
    # RuntimeError for an item that is not an instance of the declared type, unless
    # validation is globally disabled, in which case items pass through.
    # TODO:
    # Can be extended to validate '__getitem__' and nonblocking
    if f.__name__ != "__iter__":
        raise TypeError(
            f"Can not decorate function {f.__name__} with 'runtime_validation'"
        )

    @wraps(f)
    def wrapper(self):
        # [FUNCTION DOC] HELPER: Generator wrapper described in runtime_validation.
        global _runtime_validation_enabled
        if not _runtime_validation_enabled:
            yield from f(self)
        else:
            it = f(self)
            for d in it:
                if not self.type.issubtype_of_instance(d):
                    raise RuntimeError(
                        f"Expected an instance as subtype of {self.type}, but found {d}({type(d)})"
                    )
                yield d

    return wrapper
