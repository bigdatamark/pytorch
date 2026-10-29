# mypy: allow-untyped-defs
# [MODULE DOC] Defines the thread-local, dynamically scoped "virtualized" globals used throughout Inductor (ops, graph,
# kernel, fake_mode, debug, interpreter, current_node, choices, etc.). Main contents: Virtualized (a descriptor-like
# holder over a thread-local key), NullHandler/NullKernelHandler sentinels, OpsValue and OpsWrapper (operator-overloaded
# wrapper around ops calls), the _V accessor class and its singleton V, and the top-level ops handler. Import-time side
# effects: creates the thread-local storage and the module-level Virtualized instances; nothing is installed until a
# caller uses a set_* context manager.
"""
This file provides a number of "global" variables/handlers that are actually
thread local and dynamically scoped, with Inductor patching them to various
implementations depending on the situation.

These handlers are interacted with in a fairly stylized way.  Typically,
we will import V from this module::

    from .virtualized import V

Various handlers are accessible as attributes on this module; for example,
you might access ``V.graph.sizevars.optimization_hint`` to resolve a hint
associated with a symbolic expression.

There are a few distinct usage patterns for virtualized global variables:

1. Implicit argument passing.  Examples: ``V.current_node``, ``V.aot_compilation``.
   Use ``V.set_current_node`` to change what the current node is while we're
   executing some region of code, so code inside that region can query ``V.current_node``
   to find out what it is.  This is often more convenient than manually threading
   the current node as an argument through all call stacks.

2. Per-compilation global state.  Examples: ``V.fake_mode``, ``V.graph``.  For a
   given ``compile_fx`` invocation, these typically don't change, but they are
   associated with some internal state so they cannot just be global functions.
   We install these objects at the beginning of compilation and then you can
   conveniently access them without having to pass them around.

3. Alternate define-by-run interpretations.  Examples: ``V.ops``, ``V.kernel``.
   A commonly used IR in Inductor is define-by-run: instead of maintaining
   explicit syntax data structures, we instead represent loop bodies as
   callable functions, which internally invoke operations defined on
   ``V.ops``.  To perform semantic analysis, print or code generate these
   operations, we dynamically patch ``V.ops`` with an alternate handler with
   the intended semantics and then run the callable function.  For example, to
   extract out a traditional (FX) graph representation of the define-by-run
   IR, simply install a handler that records each ``ops`` call to a graph.

   TODO: Define a parent class / protocol that defines all of the operations
   V.ops is expected to support.

It is typically an error to access a virtualized global without having installed
an appropriate handler (you will get a NullHandler), although in some cases we
provide a default implementation.

One last thing: although most virtualized globals are accessed via ``V``, ``ops`` is
ubiquitous enough to have its own top level variable, so you will typically see
``ops.constant(...)`` rather than ``V.ops.constant(...)``.  In fact, these are not
equivalent; the former interface supports arithmetic overloads like ``x + y``
instead of forcing ``ops.add(x, y)``, so it should be preferred.

Some operators are seemingly unused, but they are implicitly used by ops_wrapper.
In particular, we typically have an operator for every basic pointwise PyTorch operation
supported.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from threading import local
from typing import Any, cast, Generic, TYPE_CHECKING, TypeVar

from torch.utils._ordered_set import OrderedSet

from .ops_handler import (  # noqa: F401
    DefaultHandler,
    KernelFormatterHandler,
    MockHandler,
    OpsHandler,
    ReductionType,
    StoreMode,
    WrapperHandler,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Protocol

    import torch
    from torch._inductor.choices import InductorChoices
    from torch._inductor.codegen.common import CSE, CSEVariable, KernelArgs
    from torch._inductor.codegen.cpp_utils import LocalBufferContext
    from torch._inductor.debug import DebugContext
    from torch._inductor.graph import GraphLowering
    from torch._inductor.ir import ExternKernelNode
    from torch._inductor.loop_body import InterpreterShim
    from torch._inductor.scheduler import SchedulerNode
    from torch._subclasses import FakeTensorMode

    from .distributed_autotune import _DistributedAutotuneState
    from .utils import IndentedBuffer, InputType

    class KernelLike(Protocol):
        # [CLASS DOC] HELPER: Type-checking-only Protocol describing the common surface of V.kernel (removed_buffers,
        # inplaced_to_remove, index_dtype, cse, compute, current_node, args, create_cse_var, plus attribute fallbacks).
        # The one-line stubs create_cse_var, __getattr__ and __setattr__ declare the interface only.
        """
        Structural type for ``V.kernel``.  Its backing handler ranges from the
        ``NullKernelHandler`` sentinel (installed while codegening the wrapper,
        before any kernel exists) to a concrete ``Kernel`` subclass during
        kernel codegen, so callers duck-type against this common surface rather
        than a single class.  The members declared explicitly below are the
        common surface relied on regardless of backend; the ``__getattr__``
        fallback keeps backend-specific members (e.g. Triton/CPP tiling state)
        accessible as ``Any`` without forcing every kernel class to enumerate
        them here.
        """

        removed_buffers: OrderedSet[str]
        inplaced_to_remove: OrderedSet[str]
        index_dtype: str
        cse: CSE[Any, Any]
        compute: IndentedBuffer
        current_node: SchedulerNode | None
        args: KernelArgs

        def create_cse_var(self, *args: Any, **kwargs: Any) -> CSEVariable: ...
        def __getattr__(self, name: str) -> Any: ...
        def __setattr__(self, name: str, value: Any) -> None: ...


threadlocal = local()

T = TypeVar("T")


class NullHandler:
    # [CLASS DOC] Sentinel type used as the default for a virtualized variable that was never set; it has no behavior,
    # so errors surface only when an attribute is accessed on an instance.
    """
    Sentinel indicating that a global variable is unset ala None.  Typically,
    attempting to access the global variable before it's set is an error, but with
    NullHandler it won't fail until you try to access an attribute on it.
    """


# If a virtualized value is set to _PoisonedVirtual then any attempt to get the
# value will result in an exception being raised. This is useful if we want to
# trap uninitialized reads of virtualized globals - for example when compiling
# in a subprocess we don't want the child reading globals that weren't copied
# from the parent.
_PoisonedVirtual = object()


class Virtualized(Generic[T]):
    # [CLASS DOC] A global variable whose value lives in thread-local storage under a key derived from its name.
    # State: variable name, thread-local key, and a default factory. Reads resolve via _get_handler; any attribute access
    # on the object is forwarded to the current handler. Not a singleton: one instance per virtualized variable. May hold
    # non-handler values such as booleans.
    """
    Implements a global variable that redirects via thread local variable
    (NB: construct this class to create the global variable; this is not
    a singleton class!)

    This allows us to swap in different op implementations in codegen.

    NB: Despite the fact that we typically call these "handlers" (e.g., NullHandler is
    the default value of the variable), we sometimes use these variables to
    store other things, like booleans.
    """

    def __init__(self, vname: str, default: Callable[[], T] | type[NullHandler]):
        # [FUNCTION DOC] Stores the variable name, builds the thread-local key as a prefixed name, and records the default
        # factory (a callable or the NullHandler type) used when nothing is set.
        self._vname = vname
        self._key: str = f"__torchinductor_{vname}"
        self._default = default

    def _set_handler(self, value: T) -> AbstractContextManager[None]:
        # [FUNCTION DOC] Installs a new value in the thread-local slot and returns a context manager. On exit of the context
        # the previously observed value (taken without the poison check, possibly the default) is reinstalled. Side effect:
        # the value is set immediately when called, not when the context is entered.
        prior = self._get_handler(False)
        setattr(threadlocal, self._key, value)

        @contextmanager
        def ctx():
            # [FUNCTION DOC] HELPER: Context manager body that yields and then restores the prior value.
            try:
                yield
            finally:
                self._set_handler(prior)

        return ctx()

    def _get_handler(self, check_poisoned: bool = True) -> T:
        # [FUNCTION DOC] Returns the thread-local value. If check_poisoned is true and the value is the poison marker, raises
        # RuntimeError. If nothing was set in this thread, returns the result of calling the default factory.
        try:
            value = getattr(threadlocal, self._key)
            if check_poisoned and value is _PoisonedVirtual:
                raise RuntimeError(
                    f"Attempt to use poisoned virtualized value '{self._vname}'."
                )
            return value
        except AttributeError:
            # TODO: To be honest, I feel we probably should just error in this
            # case, instead of making a null handler that will probably error
            # when you getattr on it
            return self._default()  # type: ignore[return-value]

    def __getattr__(self, name: str) -> Any:
        # [FUNCTION DOC] Forwards any unknown attribute access to the current handler (raising if poisoned).
        return getattr(self._get_handler(), name)


class NullKernelHandler(NullHandler):
    # [CLASS DOC] HELPER: Default handler for V.kernel when no kernel exists (e.g. while generating the wrapper).
    # Provides empty removed_buffers and inplaced_to_remove sets and an index_dtype of the 64-bit Triton integer string.
    """
    We need access `V.kernel.removed_buffers` in DeferredLine class when there
    is no kernel in the context. This happens when codegening the wrapper.
    Initialize `removed_buffers` and `inplaced_to_remove` explicitly so we don't
    need call 'getattr' with default value which is error prone to typo in
    attribute name.
    """

    def __init__(self):
        # [FUNCTION DOC] HELPER: Initializes empty removed_buffers and inplaced_to_remove ordered sets and sets index_dtype
        # to the Triton 64-bit integer name.
        super().__init__()
        self.removed_buffers = OrderedSet[Any]()
        self.inplaced_to_remove = OrderedSet[Any]()
        self.index_dtype = "tl.int64"

    def get_index_dtype_as_torch_dtype(self):
        # [FUNCTION DOC] HELPER: Maps index_dtype to a torch dtype: the 64-bit name gives int64, the 32-bit name gives int32;
        # any other value raises ValueError.
        import torch

        if self.index_dtype == "tl.int64":
            return torch.int64
        elif self.index_dtype == "tl.int32":
            return torch.int32
        else:
            raise ValueError(f"Unknown dtype: {self.index_dtype}")


_ops: Virtualized[OpsHandler[Any]] = Virtualized(
    "ops", cast(type[OpsHandler[Any]], MockHandler)
)
_graph: Virtualized[GraphLowering] = Virtualized("graph", NullHandler)
_extern_kernel_nodes: Virtualized[list[ExternKernelNode]] = Virtualized(
    "extern_kernel_nodes", NullHandler
)
_real_inputs: Virtualized[Sequence[InputType]] = Virtualized("real_inputs", NullHandler)
_fake_mode: Virtualized[FakeTensorMode] = Virtualized("fake_mode", NullHandler)
_kernel: Virtualized[KernelLike] = Virtualized("kernel", NullKernelHandler)
_debug: Virtualized[DebugContext] = Virtualized("debug", NullHandler)
_interpreter: Virtualized[InterpreterShim] = Virtualized("interpreter", NullHandler)
_aot_compilation: Virtualized[bool] = Virtualized("aot_compilation", NullHandler)
_current_node: Virtualized[torch.fx.Node] = Virtualized("current_node", NullHandler)
_local_buffer_context: Virtualized[LocalBufferContext] = Virtualized(
    "local_buffer_context", NullHandler
)
_distributed_autotune_state: Virtualized[_DistributedAutotuneState] = Virtualized(
    "distributed_autotune_state", NullHandler
)


def _active_user_lowering_ops_default() -> OrderedSet[Any]:
    # [FUNCTION DOC] HELPER: Default factory for active_user_lowering_ops: creates an empty ordered set, stores it in the
    # thread-local slot so later reads return the same set, and returns it.
    """Default factory for active_user_lowering_ops - returns persisted empty set."""
    rv: OrderedSet[Any] = OrderedSet()
    setattr(threadlocal, _active_user_lowering_ops._key, rv)
    return rv


_active_user_lowering_ops: Virtualized[OrderedSet[Any]] = Virtualized(
    "active_user_lowering_ops", _active_user_lowering_ops_default
)


def _choices_default():
    # [FUNCTION DOC] HELPER: Lazy default factory for the choices handler: instantiates config.inductor_choices_class if
    # set, otherwise InductorChoices, stores the instance in the thread-local slot and returns it.
    """
    Lazy init the global choices handler

    We virtualize InductorChoices to allow changing inductor heuristics from out of tree.
    """
    from torch._inductor import config
    from torch._inductor.choices import InductorChoices

    if config.inductor_choices_class is not None:
        rv = config.inductor_choices_class()
    else:
        rv = InductorChoices()
    setattr(threadlocal, _choices._key, rv)
    return rv


_choices: Virtualized[InductorChoices] = Virtualized("choices", _choices_default)


class OpsValue:
    # [CLASS DOC] Wrapper for the result of an ops call that overloads arithmetic, comparison and bitwise operators so
    # expressions can be written naturally; each operator calls the matching ops function on the wrapped values.
    # State: value, the underlying IR value.
    """The return type of most ops calls.

    This exists so we can overload magic methods, and write mathematical
    expressions much more fluently. So instead of

        ops.add(ops.mul(ops.mul(ops.sub(ops.mul(_Ap2, x), _Ap3), x), x), _1)

    we can write

        (_Ap2 * x - _Ap3) * x * x + _1

    """

    value: Any

    def __init__(self, value):
        # [FUNCTION DOC] Stores the wrapped IR value.
        self.value = value

    def __str__(self):
        # [FUNCTION DOC] Returns the string form of the wrapped value.
        return str(self.value)

    def __repr__(self):
        # [FUNCTION DOC] Returns a debug string showing the wrapped value.
        return f"OpsValue({self.value!r})"

    def __add__(self, other):
        # [FUNCTION DOC] Addition via the add op.
        return ops.add(self, other)

    def __mul__(self, other):
        # [FUNCTION DOC] Multiplication via the mul op.
        return ops.mul(self, other)

    def __sub__(self, other):
        # [FUNCTION DOC] Subtraction via the sub op.
        return ops.sub(self, other)

    def __neg__(self):
        # [FUNCTION DOC] Negation via the neg op.
        return ops.neg(self)

    def __truediv__(self, other):
        # [FUNCTION DOC] True division via the truediv op.
        return ops.truediv(self, other)

    def __floordiv__(self, other):
        # [FUNCTION DOC] Floor division via the floordiv op.
        return ops.floordiv(self, other)

    def __mod__(self, other):
        # [FUNCTION DOC] Modulo via the mod op.
        return ops.mod(self, other)

    def __pow__(self, other):
        # [FUNCTION DOC] Power via the pow op.
        return ops.pow(self, other)

    def __lt__(self, other):
        # [FUNCTION DOC] Less-than via the lt op (returns an IR value, not a bool).
        return ops.lt(self, other)

    def __le__(self, other):
        # [FUNCTION DOC] Less-or-equal via the le op (returns an IR value, not a bool).
        return ops.le(self, other)

    def __eq__(self, other):
        # [FUNCTION DOC] Equality via the eq op (returns an IR value, not a bool).
        # KNOWN ISSUE: defining equality without a hash method makes instances unhashable.
        return ops.eq(self, other)

    def __ne__(self, other):
        # [FUNCTION DOC] Inequality via the ne op (returns an IR value, not a bool).
        return ops.ne(self, other)

    def __gt__(self, other):
        # [FUNCTION DOC] Greater-than via the gt op (returns an IR value, not a bool).
        return ops.gt(self, other)

    def __ge__(self, other):
        # [FUNCTION DOC] Greater-or-equal via the ge op (returns an IR value, not a bool).
        return ops.ge(self, other)

    def __and__(self, other):
        # [FUNCTION DOC] Bitwise and via the bitwise_and op.
        return ops.bitwise_and(self, other)

    def __or__(self, other):
        # [FUNCTION DOC] Bitwise or via the bitwise_or op.
        return ops.bitwise_or(self, other)

    def __xor__(self, other):
        # [FUNCTION DOC] Bitwise xor via the bitwise_xor op.
        return ops.bitwise_xor(self, other)

    def __invert__(self):
        # [FUNCTION DOC] Bitwise not via the bitwise_not op.
        return ops.bitwise_not(self)

    def __rshfit__(self, n):
        # [FUNCTION DOC] Right shift via the bitwise_right_shift op.
        # KNOWN ISSUE: the method name is misspelled (rshfit), so Python never uses it for the right-shift operator.
        return ops.bitwise_right_shift(self, n)

    def __lshift__(self, n):
        # [FUNCTION DOC] Left shift via the bitwise_left_shift op.
        return ops.bitwise_left_shift(self, n)


class OpsWrapper(DefaultHandler):
    # [CLASS DOC] Default-handler subclass used as the global ops object. Every op call unwraps OpsValue arguments,
    # dispatches to the currently installed ops handler, and wraps results back into OpsValue.
    """This wraps any returned IR values into an `OpsValue` instance, so that we
    can overload the magic methods for writing mathematical expressions fluently.
    """

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # [FUNCTION DOC] HELPER: Generic op dispatch: unwraps args and kwargs, calls the op of that name on the current ops
        # handler, and wraps the result (tuples are wrapped element-wise).
        new_args = [OpsWrapper._unwrap(a) for a in args]
        new_kwargs = {k: OpsWrapper._unwrap(v) for k, v in kwargs.items()}
        return OpsWrapper._wrap(getattr(_ops, name)(*new_args, **new_kwargs))

    @staticmethod
    def _unwrap(x):
        # [FUNCTION DOC] HELPER: Recursively converts lists/tuples to tuples of unwrapped values; OpsValue gives its value;
        # anything else is returned unchanged.
        if isinstance(x, (list, tuple)):
            return tuple(OpsWrapper._unwrap(v) for v in x)
        if isinstance(x, OpsValue):
            return x.value
        return x

    @staticmethod
    def _wrap(x):
        # [FUNCTION DOC] HELPER: Wraps a list/tuple element-wise into a tuple of OpsValue, or a single value into one OpsValue.
        if isinstance(x, (list, tuple)):
            return tuple(OpsValue(v) for v in x)
        return OpsValue(x)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def indirect_indexing(index, size, check=True, wrap_neg=True):
        # [FUNCTION DOC] Unwraps the index and delegates to the current ops handler. Returns a sympy value rather than an
        # OpsValue (not wrapped). Arguments: index, size, check, wrap_neg flags passed through.
        # Returns a sympy value, not IR value
        index = OpsWrapper._unwrap(index)
        return _ops.indirect_indexing(index, size, check, wrap_neg)


ops: OpsHandler[Any] = OpsWrapper()


class _V:
    # [CLASS DOC] Namespace singleton V: class attributes expose the handler classes and set_*/get_* callables bound to
    # each Virtualized variable (set_* return context managers that restore the previous value on exit), and properties
    # read the current value of each variable. Created once as V.
    MockHandler = MockHandler
    KernelFormatterHandler = KernelFormatterHandler
    WrapperHandler = WrapperHandler

    set_ops_handler: Callable[[OpsHandler[Any]], AbstractContextManager[None]] = (
        _ops._set_handler
    )
    get_ops_handler: Callable[[], OpsHandler[Any]] = _ops._get_handler
    set_graph_handler: Callable[[GraphLowering], AbstractContextManager[None]] = (
        _graph._set_handler
    )
    set_extern_kernel_nodes: Callable[
        [list[ExternKernelNode]], AbstractContextManager[None]
    ] = _extern_kernel_nodes._set_handler
    set_real_inputs: Callable[[Sequence[InputType]], AbstractContextManager[None]] = (
        _real_inputs._set_handler
    )
    get_real_inputs: Callable[[], Sequence[InputType]] = _real_inputs._get_handler
    # Broad param: fake_mode may be installed as None (see compile_fx); the
    # getter still narrows to FakeTensorMode for the common set-then-read path.
    set_fake_mode: Callable[[Any], AbstractContextManager[None]] = (
        _fake_mode._set_handler
    )
    get_fake_mode: Callable[[], FakeTensorMode] = _fake_mode._get_handler
    set_kernel_handler: Callable[[Any], AbstractContextManager[None]] = (
        _kernel._set_handler
    )
    set_debug_handler: Callable[[DebugContext], AbstractContextManager[None]] = (
        _debug._set_handler
    )
    set_interpreter_handler: Callable[
        [InterpreterShim], AbstractContextManager[None]
    ] = _interpreter._set_handler
    set_aot_compilation: Callable[[bool], AbstractContextManager[None]] = (
        _aot_compilation._set_handler
    )
    get_aot_compilation: Callable[[], bool] = _aot_compilation._get_handler
    set_current_node: Callable[[torch.fx.Node], AbstractContextManager[None]] = (
        _current_node._set_handler
    )
    get_current_node: Callable[[], torch.fx.Node] = _current_node._get_handler
    set_local_buffer_context: Callable[
        [LocalBufferContext], AbstractContextManager[None]
    ] = _local_buffer_context._set_handler
    get_local_buffer_context: Callable[[], LocalBufferContext] = (
        _local_buffer_context._get_handler
    )
    set_choices_handler: Callable[[InductorChoices], AbstractContextManager[None]] = (
        _choices._set_handler
    )
    # Broad param: the state is reset by installing a NullHandler (see
    # distributed_autotune.graph_context); the getter narrows on the read path.
    set_distributed_autotune_state: Callable[[Any], AbstractContextManager[None]] = (
        _distributed_autotune_state._set_handler
    )
    get_distributed_autotune_state: Callable[[], _DistributedAutotuneState] = (
        _distributed_autotune_state._get_handler
    )
    set_active_user_lowering_ops: Callable[
        [OrderedSet[Any]], AbstractContextManager[None]
    ] = _active_user_lowering_ops._set_handler
    get_active_user_lowering_ops: Callable[[], OrderedSet[Any]] = (
        _active_user_lowering_ops._get_handler
    )

    @property
    def ops(self) -> OpsHandler[Any]:
        # [FUNCTION DOC] Returns the current ops handler.
        """The operator handler specific to the current codegen task"""
        return _ops._get_handler()

    @property
    def graph(self) -> GraphLowering:
        """The graph currently being generated"""
        # [FUNCTION DOC] Returns the graph currently being lowered.
        return _graph._get_handler()

    @property
    def extern_kernel_nodes(self) -> list[ExternKernelNode]:
        # [FUNCTION DOC] Returns the list of extern kernel nodes for the whole graph including subgraphs.
        """
        The extern_kernel_nodes needed for the entire graph, including the
        subgraphs.
        See `ProxyExecutor Design Note` in ir.py for more details
        """
        return _extern_kernel_nodes._get_handler()

    @property
    def real_inputs(self) -> Sequence[InputType]:
        # [FUNCTION DOC] Returns the non-fake example inputs.
        """non-fake example inputs"""
        return _real_inputs._get_handler()

    @property
    def fake_mode(self) -> FakeTensorMode:
        """The graph currently being generated"""
        # [FUNCTION DOC] Returns the current fake tensor mode.
        # KNOWN ISSUE: the docstring says "graph currently being generated", copied from the graph property.
        return _fake_mode._get_handler()

    @property
    def kernel(self) -> KernelLike:
        """The kernel currently being generated"""
        # [FUNCTION DOC] Returns the current kernel handler (the null kernel handler when none is active).
        return _kernel._get_handler()

    # debug and interpreter intentionally keep an inferred (dynamic) return type.
    # DebugContext dispatches via __getattr__ and InterpreterShim carries an
    # Optional current_node, so pinning the concrete class here surfaces spurious
    # callable/None errors at unrelated call sites.
    @property
    def debug(self):
        # [FUNCTION DOC] Returns the current debug context (dynamically typed).
        return _debug._get_handler()

    @property
    def interpreter(self):
        # [FUNCTION DOC] Returns the current interpreter shim (dynamically typed).
        return _interpreter._get_handler()

    @property
    def aot_compilation(self) -> bool:
        # [FUNCTION DOC] Returns true only if the aot_compilation flag was explicitly set to true; unset gives false.
        return _aot_compilation._get_handler() is True

    @property
    def current_node(self) -> torch.fx.Node:
        # [FUNCTION DOC] Returns the current FX node (a NullHandler instance if unset).
        return _current_node._get_handler()

    @property
    def local_buffer_context(self) -> LocalBufferContext:
        # [FUNCTION DOC] Returns the current local buffer context.
        return _local_buffer_context._get_handler()

    @property
    def choices(self) -> InductorChoices:
        # [FUNCTION DOC] Returns the choices handler, lazily creating it on first use in this thread.
        return _choices._get_handler()

    @property
    def distributed_autotune_state(self) -> _DistributedAutotuneState:
        # [FUNCTION DOC] Returns the current distributed autotune state.
        return _distributed_autotune_state._get_handler()

    @property
    def active_user_lowering_ops(self) -> OrderedSet[Any]:
        # [FUNCTION DOC] Returns the set of ops currently being lowered via user lowerings (a recursion guard), created lazily.
        """Set of ops currently being lowered via user_lowerings (for recursion guard)."""
        return _active_user_lowering_ops._get_handler()


V = _V()
