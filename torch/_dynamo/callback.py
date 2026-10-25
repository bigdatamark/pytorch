# [MODULE DOC]
# Compilation start/end callback registry for Dynamo. Defines CallbackTrigger (enum of
# what kicked off a compile), CallbackArgs (payload given to callbacks),
# CompilationCallbackHandler (registry plus nesting-aware runner), the singleton
# callback_handler, and the on_compile_start / on_compile_end registration decorators.
# Import-time side effect: creates the singleton handler.
"""
This module provides callback management functionality for TorchDynamo's compilation process.

It implements a thread-safe system for registering, managing and executing callbacks that run
at the start and end of TorchDynamo compilations. Key features include:

- Registration and deregistration of compilation callbacks
- Thread-safe callback handling with proper locking mechanisms
- Prevention of duplicate callback execution when configured
- Decorator utilities for easy callback registration
- Context manager for controlled callback lifecycle

The module centers around the CompilationCallbackHandler class which maintains separate
lists for start and end callbacks, manages their execution order, and ensures thread-safety.
Utility decorators @on_compile_start and @on_compile_end provide a convenient way to
register compilation hooks.

Example usage:
    @on_compile_start
    def my_start_callback():
        print("Starting compilation")

    @on_compile_end
    def my_end_callback():
        print("Compilation complete")
"""

import enum
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


class CallbackTrigger(enum.Enum):
    # [CLASS DOC]
    # Enumeration of the reasons a compilation event happens: DYNAMO (frame trace),
    # LAZY_BACKWARD (deferred backward compile), TRITON_AUTOTUNING (runtime autotune,
    # noted as temporarily disabled), CUDAGRAPH_RECORDING.
    # most common case, dynamo attempts to trace a new frame
    DYNAMO = 1
    # backward compilation can be deferred to runtime
    LAZY_BACKWARD = 2
    # some backends autotune at runtime
    TRITON_AUTOTUNING = 3  # Temporarily disabled due to spam
    # cudagraphs record at runtime
    CUDAGRAPH_RECORDING = 4


@dataclass
class CallbackArgs:
    # [CLASS DOC]
    # Immutable-style payload passed to every callback: the trigger kind and the
    # compile id string identifying the compilation.
    callback_trigger: CallbackTrigger
    compile_id: str


@dataclass
class CompilationCallbackHandler:
    # [CLASS DOC]
    # Registry of start and end callbacks plus a lock-protected counter of nested
    # in-flight compilations. Start callbacks run only when the outermost compile
    # begins and end callbacks only when the outermost compile finishes.
    # State: start_callbacks, end_callbacks (ordered lists), and private counter/lock.
    start_callbacks: list[Callable[[CallbackArgs], None]] = field(default_factory=list)
    end_callbacks: list[Callable[[CallbackArgs], None]] = field(default_factory=list)

    __pending_callbacks_counter: int = field(default=0, init=False, repr=False)
    __pending_callbacks_counter_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def register_start_callback(
        self, callback: Callable[[CallbackArgs], None]
    ) -> Callable[[CallbackArgs], None]:
        # [FUNCTION DOC]
        # Appends the callback to the start list (duplicates allowed) and returns it
        # unchanged so it can be used as a decorator.
        """
        Register a callback function to be called when the compilation starts.

        Args:
        - callback (Callable): The callback function to register.
        """
        self.start_callbacks.append(callback)
        return callback

    def register_end_callback(
        self, callback: Callable[[CallbackArgs], None]
    ) -> Callable[[CallbackArgs], None]:
        # [FUNCTION DOC]
        # Appends the callback to the end list (duplicates allowed) and returns it
        # unchanged so it can be used as a decorator.
        """
        Register a callback function to be called when the compilation ends.

        Args:
        - callback (Callable): The callback function to register.
        """
        self.end_callbacks.append(callback)
        return callback

    def remove_start_callback(self, callback: Callable[[CallbackArgs], None]) -> None:
        # [FUNCTION DOC]
        # Removes the first matching start callback; raises an error if not registered.
        """
        Remove a registered start callback function.

        Args:
        - callback (Callable): The callback function to remove.
        """
        self.start_callbacks.remove(callback)

    def remove_end_callback(self, callback: Callable[[CallbackArgs], None]) -> None:
        # [FUNCTION DOC]
        # Removes the first matching end callback; raises an error if not registered.
        """
        Remove a registered end callback function.

        Args:
        - callback (Callable): The callback function to remove.
        """
        self.end_callbacks.remove(callback)

    def run_start_callbacks(self, args: CallbackArgs) -> None:
        # [FUNCTION DOC]
        # Calls each start callback in registration order with args. Exceptions from a
        # callback propagate and stop later callbacks.
        """
        Execute all registered start callbacks.
        """
        for callback in self.start_callbacks:
            callback(args)

    def run_end_callbacks(self, args: CallbackArgs) -> None:
        # [FUNCTION DOC]
        # Calls each end callback in registration order with args. Exceptions from a
        # callback propagate and stop later callbacks.
        """
        Execute all registered end callbacks.
        """
        for callback in self.end_callbacks:
            callback(args)

    @contextmanager
    def install_callbacks(
        self, trigger: CallbackTrigger, compile_id: str
    ) -> Generator[None, Any, Any]:
        # [FUNCTION DOC]
        # Context manager wrapping one compilation. On entry increments the pending
        # counter under the lock and runs start callbacks if this is the outermost
        # entry. On exit (even on error) runs end callbacks if outermost, then
        # decrements the counter. Raises an error if the counter is not positive on exit.
        # Note: start/end callbacks run while holding the lock, and a failing start
        # callback leaves the counter incremented, which the finally block then undoes.
        """
        Context manager to install the callbacks and run them when the context is exited.
        """
        args = CallbackArgs(trigger, compile_id)
        try:
            with self.__pending_callbacks_counter_lock:
                self.__pending_callbacks_counter += 1
                if self.__pending_callbacks_counter == 1:
                    self.run_start_callbacks(args)
            yield
        finally:
            with self.__pending_callbacks_counter_lock:
                if self.__pending_callbacks_counter <= 0:
                    raise AssertionError(
                        "Pending callbacks counter cannot become negative."
                    )
                if self.__pending_callbacks_counter == 1:
                    self.run_end_callbacks(args)
                self.__pending_callbacks_counter -= 1

    def clear(self) -> None:
        # [FUNCTION DOC]
        # Empties both callback lists, then raises an error if any compilations are
        # still pending (the lists are already cleared by then).
        """
        Clear all registered callbacks.
        """
        self.start_callbacks.clear()
        self.end_callbacks.clear()
        if self.__pending_callbacks_counter != 0:
            raise AssertionError(
                f"Cannot clear callbacks while {self.__pending_callbacks_counter} are pending"
            )


callback_handler = CompilationCallbackHandler()


def on_compile_start(
    callback: Callable[[CallbackArgs], None],
) -> Callable[[CallbackArgs], None]:
    # [FUNCTION DOC]
    # Registers the callback on the global handler as a start callback; returns it.
    """
    Decorator to register a callback function for the start of the compilation.
    """
    callback_handler.register_start_callback(callback)
    return callback


def on_compile_end(
    callback: Callable[[CallbackArgs], None],
) -> Callable[[CallbackArgs], None]:
    # [FUNCTION DOC]
    # Registers the callback on the global handler as an end callback; returns it.
    """
    Decorator to register a callback function for the end of the compilation.
    """
    callback_handler.register_end_callback(callback)
    return callback
