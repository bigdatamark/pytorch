# [MODULE DOC]
# Typing protocols describing the parts of the timeit Timer, the compiled timeit template module
# and the torch C valgrind bindings that the benchmark utilities rely on. Only type definitions;
# no behavior and no import-time side effects.
from typing import Any
from collections.abc import Callable
from typing_extensions import Protocol, runtime_checkable


class TimerClass(Protocol):
    # [CLASS DOC] Structural type for a timer: constructed from a statement, setup, clock
    # function and globals mapping, and able to run the statement a number of times.
    """This is the portion of the `timeit.Timer` API used by benchmark utils."""
    def __init__(
        self,
        stmt: str,
        setup: str,
        timer: Callable[[], float],
        globals: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] HELPER: Protocol constructor signature only; no body.
        ...

    def timeit(self, number: int) -> float:
        # [FUNCTION DOC] HELPER: Protocol signature; runs the statement the given number of
        # times and returns the elapsed seconds.
        ...


@runtime_checkable
class TimeitModuleType(Protocol):
    # [CLASS DOC] Runtime checkable structural type for a compiled module built from the timeit
    # C++ template; it must provide the timeit method.
    """Modules generated from `timeit_template.cpp`."""
    def timeit(self, number: int) -> float:
        # [FUNCTION DOC] HELPER: Protocol signature; runs the statement the given number of
        # times and returns the elapsed seconds.
        ...


class CallgrindModuleType(Protocol):
    # [CLASS DOC] Structural type for a module exposing the valgrind support check and toggle
    # bindings, with file and name attributes.
    """Replicates the valgrind endpoints in `torch._C`.

    These bindings are used to collect Callgrind profiles on earlier versions
    of PyTorch and will eventually be removed.
    """
    __file__: str
    __name__: str

    def _valgrind_supported_platform(self) -> bool:
        # [FUNCTION DOC] HELPER: Protocol signature; tells whether valgrind profiling works on
        # this platform.
        ...

    def _valgrind_toggle(self) -> None:
        # [FUNCTION DOC] HELPER: Protocol signature; toggles callgrind collection on or off.
        ...
