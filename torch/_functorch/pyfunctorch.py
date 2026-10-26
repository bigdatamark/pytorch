# [MODULE DOC] Python re-implementation of functorch's dynamic layer stack dispatch for
# PyDispatcher. Defines Python interpreter wrappers (vmap, grad, jvp, functionalize) around
# the C++ interpreter objects, helpers to temporarily pop/clear/restore the interpreter stack,
# a function to validate recorded stack state, and dispatch_functorch which runs an operator
# through the top interpreter. No import-time side effects.
from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch._C._functorch import (
    CFunctionalizeInterpreterPtr,
    CGradInterpreterPtr,
    CInterpreter,
    CJvpInterpreterPtr,
    CVmapInterpreterPtr,
    pop_dynamic_layer_stack,
    push_dynamic_layer_stack,
    RandomnessType,
    TransformType,
)
from torch.autograd.forward_ad import _set_fwd_grad_enabled


if TYPE_CHECKING:
    from collections.abc import Generator


"""
This file contains the functorch integration with PyDispatcher.

PyDispatcher does not understand functorch's DynamicLayerStack dispatching
logic because it is entirely implemented in C++ in the fallbacks for two
dispatch keys, FuncTorchDynamicLayer{Front, Back}Mode (PyDispatcher is unable
to directly reuse C++ boxed fallbacks).

Instead of trying to hammer PyDispatcher into understanding those fallbacks,
we re-implement the logic of peeking the top of the stack for an interpreter,
selecting the interpreter to dispatch on, etc, in Python. This leads to a
simpler design.

The main difference between C++ functorch and PyDispatcher's functorch logic
is that:
- C++ functorch needs to manually tweak dispatch keys to ping-pong between
  DynamicLayerFrontMode and DynamicLayerBackMode.
- PyDispatcher's functorch logic pops an Interpreter from the top of the stack
  and asks it to execute the rule associated with the Interpreter.

In C++ we do the ping-pong because e.g. vmap rules are associated with the
batched DispatchKey, but in PyDispatcher we are able to avoid this by asking
the user to register a batching rule directly to a transform that an
interpreter then invokes.
"""


# FuncTorchInterpreter is the Python version of Interpreter (recall that
# the DynamicLayerStack is a stack of interpreters).
# It is a wrapper around the actual C++ Interpreter object.
#
# Keep the methods in sync with aten/src/ATen/functorch/Interpreter.h
class FuncTorchInterpreter(ABC):
    # [CLASS DOC] Abstract Python wrapper around a C++ functorch Interpreter. Holds the C++
    # pointer object in _cptr (subclasses define it lazily from the generic cdata). Subclasses
    # implement process (run an op's rule for the transform) and get_state (hashable summary
    # used to validate that a recorded graph matches the current stack).
    def __init__(self, cptr: Any) -> None:
        # [FUNCTION DOC] Stores the C++ interpreter pointer object in _cptr.
        self._cptr = cptr

    # Process an operation. eg for vmap, this is invoking a batching rule.
    # Conceptually this is analogous to Interpreter::process in C++
    @abstractmethod
    def process(self, op: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # [FUNCTION DOC] Abstract: executes op with args and kwargs under this transform's
        # rule and returns the result. Base implementation does nothing.
        pass

    # lower an operation from this Interpreter to the next Interpreter on the stack.
    # Concretely, this involves temporarily popping the current Interpreter.
    # Conceptually this is analogous to Interpreter::sendToNextInterpreter in C++
    def lower(self) -> contextlib.AbstractContextManager[Any]:
        # [FUNCTION DOC] Returns a context manager that pops the top interpreter while active,
        # so ops run against the next interpreter down; the layer is pushed back on exit.
        return temporarily_pop_interpreter_stack()

    def level(self) -> int:
        # [FUNCTION DOC] Returns the nesting level of this interpreter.
        return self._cptr.level()

    def key(self) -> TransformType:
        # [FUNCTION DOC] Returns the TransformType (grad, vmap, jvp, functionalize) of this interpreter.
        return self._cptr.key()

    def get_state(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] Returns a tuple describing this interpreter for state comparison.
        # Base class raises NotImplementedError; subclasses override.
        raise NotImplementedError

    def check_state(self, state: tuple[Any, ...]) -> bool:
        # [FUNCTION DOC] Returns True if the given recorded state equals this interpreter's current state.
        return state == self.get_state()

    def __getstate__(self) -> dict[str, Any]:
        # [FUNCTION DOC] Pickle support: returns a copy of the instance dict without the
        # unpicklable C++ pointer entry _cptr.
        state = self.__dict__.copy()
        state.pop("_cptr", None)
        return state


@contextlib.contextmanager
def temporarily_pop_interpreter_stack() -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager: pops the top dynamic layer (interpreter) from the global
    # stack, runs the body, then pushes the saved layer back, even if the body raises.
    # Side effect: mutates the global functorch dynamic layer stack.
    try:
        saved = pop_dynamic_layer_stack()
        yield
    finally:
        push_dynamic_layer_stack(saved)


@contextlib.contextmanager
def temporarily_clear_interpreter_stack() -> Generator[list[Any], None, None]:
    # [FUNCTION DOC] Context manager: pops every layer off the dynamic layer stack (top first),
    # yields a copy of the popped list (top-most first), and on exit pushes the layers back in
    # their original order. Side effect: mutates the global dynamic layer stack.
    stack: list[Any] = []
    try:
        while torch._C._functorch.peek_interpreter_stack() is not None:
            stack.append(pop_dynamic_layer_stack())
        yield list(stack)
    finally:
        while stack:
            push_dynamic_layer_stack(stack.pop())


@contextlib.contextmanager
def temporarily_restore_interpreter_stack(
    stack: list[Any] | None,
) -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager: pushes the given saved layers (list given top-first, so
    # pushed in reverse) onto the dynamic layer stack, yields, then pops exactly as many layers
    # on exit. A None stack makes it return without yielding (note: using it in a with
    # statement then fails, since a generator context manager must yield).
    # KNOWN ISSUE: the early return before the yield makes the None case raise RuntimeError
    # ("generator didn't yield") when entered.
    pushed: list[Any] = []
    if stack is None:
        return
    try:
        for s in reversed(stack):
            push_dynamic_layer_stack(s)
            pushed.append(s)
        yield
    finally:
        for _ in reversed(pushed):
            # TODO: would be nice to assert that the layers are the same, but
            # Python object identity is not preserved
            pop_dynamic_layer_stack()


class VmapInterpreter(FuncTorchInterpreter):
    # [CLASS DOC] Interpreter for vmap layers. Stores the generic C++ interpreter in _cdata and
    # lazily wraps it in a vmap-specific pointer (_cptr). process runs the op's registered vmap
    # rule; exposes batch size and randomness mode.
    def __init__(self, cdata: CInterpreter) -> None:
        # [FUNCTION DOC] Stores the C++ interpreter cdata; raises AssertionError if its key is not Vmap.
        if cdata.key() != TransformType.Vmap:
            raise AssertionError(f"expected TransformType.Vmap, got {cdata.key()}")
        # NOTE: [Interpreter cdata vs cptr]
        # cdata is a generic CInterpreter. We wrap it in a CVmapInterpreterPtr
        # so that we can access methods specific to the vmap interpreter
        self._cdata = cdata

    @cached_property
    # pyrefly: ignore [bad-override]
    def _cptr(self) -> CVmapInterpreterPtr:
        # [FUNCTION DOC] HELPER: Cached property building the vmap-specific C++ pointer wrapper from _cdata.
        return CVmapInterpreterPtr(self._cdata)

    def process(self, op: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # [FUNCTION DOC] Looks up the op's vmap rule in its functorch table and calls it with
        # this interpreter, args and kwargs; returns its result. Raises KeyError if no rule is registered.
        kernel = op.functorch_table[TransformType.Vmap]
        return kernel(self, *args, **kwargs)

    def batch_size(self) -> int:
        # [FUNCTION DOC] Returns the batch size of this vmap layer.
        return self._cptr.batchSize()

    def randomness(self) -> str:
        # [FUNCTION DOC] Returns the randomness mode as a string: "error", "same" or "different".
        # Raises RuntimeError for an unknown randomness type.
        typ = self._cptr.randomness()
        if typ == RandomnessType.Error:
            return "error"
        elif typ == RandomnessType.Same:
            return "same"
        elif typ == RandomnessType.Different:
            return "different"
        raise RuntimeError(f"Unknown RandomnessType: {typ}")

    def get_state(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] Returns (transform name, level, randomness string).
        return (self.key().name, self.level(), self.randomness())


@contextlib.contextmanager
def nested(
    *contexts: contextlib.AbstractContextManager[Any],
) -> Generator[tuple[contextlib.AbstractContextManager[Any], ...], None, None]:
    # [FUNCTION DOC] HELPER: Context manager entering all given context managers in order
    # (via an exit stack, so they exit in reverse order) and yielding the tuple of contexts.
    with contextlib.ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx)
        yield contexts


class GradInterpreter(FuncTorchInterpreter):
    # [CLASS DOC] Interpreter for grad/vjp layers. Stores _cdata and lazily wraps it as _cptr.
    # process lifts tensor arguments into this level then runs the op's grad rule; lower also
    # disables grad mode when the grad mode before this transform was off.
    def __init__(self, cdata: CInterpreter) -> None:
        # [FUNCTION DOC] Stores the C++ interpreter cdata; raises AssertionError if its key is not Grad.
        if cdata.key() != TransformType.Grad:
            raise AssertionError(f"expected TransformType.Grad, got {cdata.key()}")
        # See NOTE: [Interpreter cdata vs cptr]
        self._cdata = cdata

    @cached_property
    # pyrefly: ignore [bad-override]
    def _cptr(self) -> CGradInterpreterPtr:
        # [FUNCTION DOC] HELPER: Cached property building the grad-specific C++ pointer wrapper from _cdata.
        return CGradInterpreterPtr(self._cdata)

    def lift(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        # [FUNCTION DOC] Maps every tensor found in args and kwargs through the C++ lift (wrapping
        # it for this level); returns the new (args, kwargs). Non-tensors are unchanged.
        args, kwargs = pytree.tree_map_only(
            torch.Tensor, self._cptr.lift, [args, kwargs]
        )
        return args, kwargs

    def process(self, op: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # [FUNCTION DOC] Lifts args/kwargs to this level, then calls the op's grad rule with this
        # interpreter and the lifted arguments; returns its result. Raises KeyError if no rule exists.
        kernel = op.functorch_table[TransformType.Grad]
        args, kwargs = self.lift(args, kwargs)
        return kernel(self, *args, **kwargs)

    # GradInterpreter has custom lower because of the no_grad interaction
    # See NOTE [grad and vjp interaction with no_grad]
    # This logic is mirrored from C++ GradInterpreterPtr::sendToNextInterpreter
    def lower(self) -> contextlib.AbstractContextManager[Any]:
        # [FUNCTION DOC] Returns the context for lowering to the next interpreter; if grad mode was
        # disabled before this transform, additionally enters no_grad.
        prev_grad_mode = self.prev_grad_mode()
        if not prev_grad_mode:
            return nested(torch.no_grad(), super().lower())
        return super().lower()

    def prev_grad_mode(self) -> bool:
        # [FUNCTION DOC] Returns whether grad mode was enabled when this layer was created.
        return self._cptr.prevGradMode()

    def get_state(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] Returns (transform name, level, previous grad mode).
        return (self.key().name, self.level(), self.prev_grad_mode())


class JvpInterpreter(FuncTorchInterpreter):
    # [CLASS DOC] Interpreter for forward-mode AD (jvp) layers. Stores _cdata and lazily wraps it
    # as _cptr. process lifts tensor arguments then runs the op's jvp rule; lower also disables
    # forward grad when it was off before this transform.
    def __init__(self, cdata: CInterpreter) -> None:
        # [FUNCTION DOC] Stores the C++ interpreter cdata; raises AssertionError if its key is not Jvp.
        if cdata.key() != TransformType.Jvp:
            raise AssertionError(f"expected TransformType.Jvp, got {cdata.key()}")
        # See NOTE: [Interpreter cdata vs cptr]
        self._cdata = cdata

    @cached_property
    # pyrefly: ignore [bad-override]
    def _cptr(self) -> CJvpInterpreterPtr:
        # [FUNCTION DOC] HELPER: Cached property building the jvp-specific C++ pointer wrapper from _cdata.
        return CJvpInterpreterPtr(self._cdata)

    def lift(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        # [FUNCTION DOC] Maps every tensor found in args and kwargs through the C++ lift (wrapping
        # it for this level); returns the new (args, kwargs). Non-tensors are unchanged.
        args, kwargs = pytree.tree_map_only(
            torch.Tensor, self._cptr.lift, [args, kwargs]
        )
        return args, kwargs

    def process(self, op: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # [FUNCTION DOC] Lifts args/kwargs to this level, then calls the op's jvp rule with this
        # interpreter and the lifted arguments; returns its result. Raises KeyError if no rule exists.
        kernel = op.functorch_table[TransformType.Jvp]
        args, kwargs = self.lift(args, kwargs)
        return kernel(self, *args, **kwargs)

    # Jvp has custom lower because of the no_fwd_grad interaction
    # See NOTE [grad and vjp interaction with no_grad] for related info.
    # This logic is mirrored from C++ JvpInterpreterPtr::sendToNextInterpreter
    def lower(self) -> contextlib.AbstractContextManager[Any]:
        # [FUNCTION DOC] Returns the context for lowering to the next interpreter; if forward grad
        # was disabled before this transform, additionally disables forward grad while active.
        prev_fwd_grad_mode = self.prev_fwd_grad_mode()
        if not prev_fwd_grad_mode:
            return nested(_set_fwd_grad_enabled(False), super().lower())
        return super().lower()

    def prev_fwd_grad_mode(self) -> bool:
        # [FUNCTION DOC] Returns whether forward grad was enabled when this layer was created.
        return self._cptr.prevFwdGradMode()

    def get_state(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] Returns (transform name, level, previous forward grad mode).
        return (self.key().name, self.level(), self.prev_fwd_grad_mode())


class FunctionalizeInterpreter(FuncTorchInterpreter):
    # [CLASS DOC] Interpreter for functionalize layers. Stores _cdata and lazily wraps it as
    # _cptr. process runs the op's functionalize rule; exposes the add-back-views option.
    def __init__(self, cdata: CInterpreter) -> None:
        # [FUNCTION DOC] Stores the C++ interpreter cdata; raises AssertionError if its key is not Functionalize.
        if cdata.key() != TransformType.Functionalize:
            raise AssertionError(
                f"expected TransformType.Functionalize, got {cdata.key()}"
            )
        self._cdata = cdata

    @cached_property
    # pyrefly: ignore [bad-override]
    def _cptr(self) -> CFunctionalizeInterpreterPtr:
        # [FUNCTION DOC] HELPER: Cached property building the functionalize-specific C++ pointer wrapper from _cdata.
        return CFunctionalizeInterpreterPtr(self._cdata)

    def process(self, op: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # [FUNCTION DOC] Looks up the op's functionalize rule and calls it with this interpreter,
        # args and kwargs; returns its result. Raises KeyError if no rule is registered.
        kernel = op.functorch_table[TransformType.Functionalize]
        return kernel(self, *args, **kwargs)

    def functionalize_add_back_views(self) -> bool:
        # [FUNCTION DOC] Returns whether this functionalize layer adds back views.
        return self._cptr.functionalizeAddBackViews()

    def get_state(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] Returns (transform name, level).
        return (self.key().name, self.level())


def coerce_cinterpreter(cinterpreter: CInterpreter) -> FuncTorchInterpreter:
    # [FUNCTION DOC] Wraps a C++ interpreter in the Python subclass matching its transform key
    # (Grad, Vmap, Jvp, Functionalize). Raises RuntimeError for any other transform type.
    key = cinterpreter.key()
    if key == TransformType.Grad:
        return GradInterpreter(cinterpreter)
    if key == TransformType.Vmap:
        return VmapInterpreter(cinterpreter)
    if key == TransformType.Jvp:
        return JvpInterpreter(cinterpreter)
    if key == TransformType.Functionalize:
        return FunctionalizeInterpreter(cinterpreter)
    raise RuntimeError(f"NYI: PyDispatcher has not implemented support for {key}")


def retrieve_current_functorch_interpreter() -> FuncTorchInterpreter:
    # [FUNCTION DOC] Returns the Python-wrapped interpreter at the top of the dynamic layer
    # stack. Raises AssertionError if the stack is empty.
    interpreter = torch._C._functorch.peek_interpreter_stack()
    if interpreter is None:
        raise AssertionError("interpreter must not be None")
    return coerce_cinterpreter(interpreter)


def retrieve_all_functorch_interpreters() -> list[FuncTorchInterpreter]:
    # [FUNCTION DOC] Returns Python-wrapped interpreters for the whole stack in stack order, or an
    # empty list if the stack is unavailable (None).
    cis = torch._C._functorch.get_interpreter_stack()
    if cis is None:
        return []
    return [coerce_cinterpreter(ci) for ci in cis]


def compare_functorch_state(states: list[tuple[Any, ...]]) -> bool:
    # [FUNCTION DOC] Returns True if the current interpreter stack matches the recorded list of
    # interpreter states: both empty, or same length with every interpreter's check_state true.
    # Returns False when exactly one of the two is empty or the states differ.
    # There are four possible cases covered here:
    # 1. Current stack empty AND stack when generated not empty -> Invalidate
    # 2. Current stack not empty AND stack when generated empty -> Invalidate
    # 3. Current stack and generated stack empty -> Valid FX graph
    # 4. Current stack and generated stack not empty -> Valid if both states match
    peek = torch._C._functorch.peek_interpreter_stack()
    if (peek is None and len(states) != 0) or (peek is not None and len(states) == 0):
        return False

    cis = retrieve_all_functorch_interpreters()
    return len(cis) == len(states) and all(
        ci.check_state(state) for ci, state in zip(cis, states)
    )


def dispatch_functorch(op: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    # [FUNCTION DOC] Runs op through the interpreter on top of the stack: first unwraps dead
    # tensor wrappers in args and kwargs, then calls the interpreter's process. Returns its result.
    # Raises AssertionError if the interpreter stack is empty.
    interpreter = retrieve_current_functorch_interpreter()
    # In traditional PyTorch operators, DispatchKey::FuncTorchTensorWrapper's
    # unwrap_dead_tensors fallback handles unwrapping dead tensor wrappers.
    # PyDispatcher sidesteps the PyTorch dispatcher when dealing with functorch
    # transforms, so we manually unwrap the dead tensors here.
    # This logic won't need to exist when we have mode-only functorch.
    args, kwargs = pytree.tree_map_only(
        torch.Tensor, torch._C._functorch.unwrap_if_dead, (args, kwargs)
    )
    return interpreter.process(op, args, kwargs)
