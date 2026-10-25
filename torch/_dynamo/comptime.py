# [MODULE DOC] Public "comptime" API for TorchDynamo. Defines ComptimeVar (a read-only view of a
# symbolically traced value), ComptimeContext (a facade over the active instruction translator)
# and the module-level singleton comptime, whose callable/static methods run user callbacks
# during Dynamo tracing and fall back to no-ops (or plain print) in eager execution.
# No import-time side effects other than creating the comptime singleton.
"""
This module provides the public comptime interface to TorchDynamo, enabling users to execute
arbitrary Python code during symbolic evaluation of their programs.

The comptime interface allows inspection and modification of TorchDynamo's compilation
process while it is running. This can be useful for:

- Debugging compilation issues
- Inspecting intermediate state
- Adding custom guards or graph breaks
- Analyzing symbolic shapes and values

Example usage:

    import torch
    from torch._dynamo.comptime import comptime

    def my_model(x):
        # Print the compile-time known information about x
        comptime.print(x)

        # Print the current FX graph being constructed
        comptime.print_graph()

        # Force a value to be treated as static
        if comptime(lambda ctx: ctx.get_local("x").is_dynamic()):
            comptime.force_static(x)

        # Add a manual graph break
        comptime.graph_break()

Note: While this API provides significant flexibility, it intentionally avoids
exposing internal implementation details of TorchDynamo to maintain compatibility
across versions.
"""

import builtins
import dis
import time
import traceback
from collections.abc import Callable
from typing import Any, TextIO

import torch
from torch._dynamo.symbolic_convert import InstructionTranslatorBase
from torch._dynamo.variables.base import VariableTracker
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.symbolic_shapes import free_symbols

from .exc import unimplemented
from .variables import CellVariable
from .variables.tensor import SymNodeVariable


class ComptimeVar:
    """
    A ComptimeVar represents a Python value, at some particular point
    in time, in the Python code we are symbolically evaluating with
    torchdynamo.  This must be distinguished from a runtime value, as
    at compile-time there are some properties of the variable we
    do not know (for example, if the ComptimeVar represents a Tensor,
    we only know metadata about the tensor; we do NOT know what the
    actual data in the Tensor is.)
    """
    # [CLASS DOC] Public wrapper around one Dynamo VariableTracker (held in a name-mangled private
    # attribute). Exposes compile-time queries (proxy, fake value, size, python type, constant,
    # dynamism) and a force_static operation, without exposing Dynamo internals.

    def __init__(self, v: VariableTracker) -> None:
        # [FUNCTION DOC] Store the VariableTracker v being wrapped. No other effects.
        self.__variable = v

    def as_proxy(self) -> VariableTracker | list[VariableTracker]:
        # [FUNCTION DOC] Delegate to the tracker's as_proxy: returns the fx proxy (or list/tuple of
        # proxies) for this value. Raises whatever the tracker raises if it is not graph-tracked.
        # KNOWN ISSUE: return annotation says VariableTracker but the value is an fx Proxy.
        """
        Returns an fx.Proxy (or tuple/list of fx.Proxy) representing
        this variable in the FX graph we are assembling to pass
        to the user compiler.

        This method only works for variables we actually track in
        the FX graph, aka Tensors (and ints, if you are compiling
        with dynamic shapes).  In particular, if you have a list
        or tuple of tensors, you will get a list/tuple of proxies
        (not a single proxy representing the entire list/tuple).
        """
        return self.__variable.as_proxy()

    def is_proxy(self) -> bool:
        # [FUNCTION DOC] Return whether the wrapped tracker can produce a proxy.
        """
        Returns True if as_proxy() would succeed.
        """
        return self.__variable.is_proxy()

    def as_fake(self) -> FakeTensor | torch.SymInt:
        # [FUNCTION DOC] Return the example value (FakeTensor or SymInt) stored in the metadata of
        # the proxy's graph node. Fails if the value has no proxy or example value. Read-only use.
        """
        Returns a "fake" value (either a FakeTensor or a SymInt)
        representing the variable in question.  This only works
        for variables that denote Tensor or int.  You can use
        this to query metadata; e.g., v.as_fake().size(0) will
        tell you the compile-time known size of the tensor.

        WARNING: Do NOT mutate the returned tensor.
        """
        return self.__variable.as_proxy().node.meta["example_value"]

    def size(self, dim: int | None = None) -> int | torch.SymInt:
        # [FUNCTION DOC] Return the full size or the size at dimension dim of the fake value; may be
        # symbolic. Raises if the value is not a tensor.
        """
        Returns the size of the tensor (if dim is None) or the size
        at the dimension dim.  The returned size may be a SymInt.
        """
        return self.as_fake().size(dim)  # type: ignore[union-attr, return-value]

    def python_type(self) -> type:
        # [FUNCTION DOC] Return the Python type the traced value would have at runtime.
        """
        Returns what type(v) would have returned for the variable
        at compile time.
        """
        return self.__variable.python_type()

    def as_python_constant(self) -> Any:
        # [FUNCTION DOC] Return the compile-time constant value of the tracker; raises if the value
        # is not fully known at compile time. The result must not be mutated.
        """
        Returns the Python value this variable would have, but only if it is
        completely known at compile-time (e.g., it is constant).

        WARNING: Do NOT mutate the returned constant.  The returned constant
        may or may not correspond to the actual value this variable may take
        on at runtime; for example, if the variable in question is a constant
        list, we may return a copy of that list.
        """
        return self.__variable.as_python_constant()

    def is_python_constant(self) -> bool:
        # [FUNCTION DOC] Return whether the value is a compile-time constant.
        """
        Returns True if as_python_constant would succeed.
        """
        return self.__variable.is_python_constant()

    def is_dynamic(self) -> bool:
        # [FUNCTION DOC] Return true only if the value is a symbolic-number tracker whose expression
        # has free symbols (i.e. is dynamic); all other trackers, including tensors, give false.
        if isinstance(self.__variable, SymNodeVariable):
            fs = free_symbols(self.__variable.sym_num)
            return bool(fs)
        return False

    def force_static(self) -> None:
        # [FUNCTION DOC] For a symbolic number, evaluate its expression, which installs a guard on
        # the specific value. For other python constants do nothing. Raises AssertionError otherwise.
        """
        Forces that a value is static, inducing a guard on its specific value
        """
        if isinstance(self.__variable, SymNodeVariable):
            self.__variable.evaluate_expr()
        elif self.__variable.is_python_constant():
            # TODO: Maybe complain if this isn't a int/bool/float variable
            pass
        else:
            raise AssertionError(
                f"cannot force {self.__variable} ({type(self.__variable)}) static"
            )

    def _i_will_not_complain_if_bc_breaks_VariableTracker(self) -> VariableTracker:
        # [FUNCTION DOC] Escape hatch returning the wrapped VariableTracker; no compatibility promise.
        """
        Returns the internal data structure VariableTracker that Dynamo uses
        to represent variables at compile time.  There are no BC guarantees on
        this API and WE RESERVE THE RIGHT TO BREAK YOUR CODE if you rely on
        it.
        """
        return self.__variable

    def __repr__(self) -> str:
        # [FUNCTION DOC] Return the tracker's debug representation.
        return self.__variable.debug_repr()

    # TODO: API for adding a custom guard


class ComptimeContext:
    """
    This context class provides access to a public API for Dynamo's internals.
    If there is something here you would find useful that is missing, please
    file a feature request at https://github.com/pytorch/pytorch/
    """
    # [CLASS DOC] Public facade passed to comptime callbacks. Holds the active instruction
    # translator (private attribute) and offers inspection/printing helpers, graph break,
    # static assertions and an escape hatch to the raw translator. Stack level arguments walk
    # up the chain of parent translators (inlined callers).

    def __init__(self, tx: InstructionTranslatorBase) -> None:
        # [FUNCTION DOC] Store the instruction translator tx.
        self.__tx = tx

    def get_local(self, name: str, *, stacklevel: int = 0) -> ComptimeVar:
        # [FUNCTION DOC] Look up local variable name in the symbolic locals of the translator
        # stacklevel frames up; return it wrapped as ComptimeVar, dereferencing cell variables.
        # Raises a lookup error if the name is not a local.
        """
        Retrieve the compile-time known information about a local.
        """
        tx = self.__get_tx(stacklevel)
        var = tx.symbolic_locals[name]

        # Auto-dereference when accessing cell locals in python.
        if isinstance(var, CellVariable):
            return ComptimeVar(tx.output.side_effects.load_cell(var))

        return ComptimeVar(var)

    def graph_break(self, msg: str = "ComptimeContext.graph_break") -> None:
        # [FUNCTION DOC] Always raises Dynamo's Unsupported error (via unimplemented) with msg as
        # context, forcing a graph break at this point.
        """
        Manually trigger a graph break
        """
        unimplemented(
            gb_type="ComptimeContext graph break",
            context=msg,
            explanation=f"Manually triggered ComptimeContext graph break with message {msg}.",
            hints=[],
        )

    def graph(self) -> torch.fx.Graph:
        # [FUNCTION DOC] Return the in-progress FX graph of the output graph (live, not a copy).
        """
        Retrieve the partially constructed FX graph that would be
        passed to the user compiler after compilation.
        """
        return self.__tx.output.graph

    def assert_static(self, val: ComptimeVar) -> None:
        # [FUNCTION DOC] Raise AssertionError if val is dynamic; otherwise do nothing.
        """
        Asserts that the int is static (and not dynamic, per dynamic shapes)
        """
        if val.is_dynamic():
            raise AssertionError(
                "expected static but got dynamic (run with TORCH_LOGS=dynamic for more info)"
            )

    def print_graph(self, *, verbose: bool = True, file: TextIO | None = None) -> None:
        # [FUNCTION DOC] Print the generated python source of the current FX graph to file
        # (stdout by default); verbose controls the code generation verbosity.
        """
        Print the partially constructed FX graph that would be passed
        to the user compiler after compilation.
        """
        print(
            self.__tx.output.graph.python_code("self", verbose=verbose).src, file=file
        )

    def parent(self) -> "ComptimeContext":
        # [FUNCTION DOC] Return a new context wrapping the parent (caller) translator.
        # Fails with an attribute error if there is no parent.
        return ComptimeContext(self.__tx.parent)  # type: ignore[arg-type]

    def __get_tx(self, stacklevel: int) -> Any:
        # [FUNCTION DOC] HELPER: Return the translator reached by following the parent link
        # stacklevel times from the stored one.
        tx = self.__tx
        # pyrefly: ignore [bad-assignment, non-convergent-recursion]
        for _ in range(stacklevel):
            tx = tx.parent  # type: ignore[assignment]
        return tx

    def print(self, val: Any, *, file: TextIO | None = None) -> None:
        # [FUNCTION DOC] Print the repr of val to file (stdout by default).
        print(repr(val), file=file)

    def print_disas(self, *, file: TextIO | None = None, stacklevel: int = 0) -> None:
        # [FUNCTION DOC] Print the disassembly of the code object of the translator at stacklevel,
        # marking the current instruction offset, to file (stdout by default).
        """
        Print the current series of opcodes being executed (not including
        parent frames), including where you are in the particular opcode
        stream.
        """
        tx = self.__get_tx(stacklevel)
        print(
            dis.Bytecode(
                tx.f_code,
                current_offset=tx.instructions[tx.instruction_pointer].offset,
            ).dis(),
            file=file,
        )

    def print_value_stack(
        self, *, file: TextIO | None = None, stacklevel: int = 0
    ) -> None:
        # [FUNCTION DOC] Print one line per entry of the symbolic value stack (debug repr) of the
        # translator at stacklevel, to file (stdout by default).
        """
        Print the current Python value stack.  Note that this is NOT the same
        as the traceback; use print_bt() to print that.  Note that at
        stacklevel=0, this will typically be empty, as comptime cannot
        currently be used in an expression context where there would be
        intermediates on the stack.  If you would find this useful, please
        file a bug at https://github.com/pytorch/pytorch/

        NB: Stack grows downwards in our print
        """
        tx = self.__get_tx(stacklevel)
        for s in tx.stack:
            print(f"- {s.debug_repr()}", file=file)

    def print_locals(self, *, file: TextIO | None = None, stacklevel: int = 0) -> None:
        # [FUNCTION DOC] Print "name = debug repr" for every symbolic local of the translator at
        # stacklevel, to file (stdout by default).
        """
        Print all of the locals available in the current context.
        By default this view is very limited; you can get more information
        about any individual local using get_local().
        """
        tx = self.__get_tx(stacklevel)
        for k, v in tx.symbolic_locals.items():
            print(f"{k} = {v.debug_repr()}", file=file)

    def print_bt(self, *, file: TextIO | None = None, stacklevel: int = 0) -> None:
        # [FUNCTION DOC] Collect a frame summary from the translator at stacklevel up through all
        # parents, then print the formatted stack (outermost first) to file.
        """
        Print the user code backtrace, starting at the beginning of the
        frame Dynamo started evaluating.  Note that this MAY NOT go all
        the way to the torch.compile invocation, as we may have done
        a graph break and are compiling an intermediate frame as the
        starting point.  If you think the other behavior would be better,
        file a bug at https://github.com/pytorch/pytorch/
        """
        stack = []
        tx = self.__get_tx(stacklevel)
        while tx is not None:
            stack.append(tx.frame_summary())
            tx = getattr(tx, "parent", None)
        print(
            "".join(traceback.StackSummary.from_list(reversed(stack)).format()),
            file=file,
        )

    def print_guards(self, *, file: TextIO | None = None) -> None:
        # [FUNCTION DOC] Print the repr of each currently installed guard, sorted, one per line.
        # Ignores the stack level: always uses this context's translator.
        """
        Print the currently installed guards for the Dynamo context.
        This does NOT include guards associated with variables that
        may or may not be installed in the future if those variables
        are used.
        """
        # TODO: improve print format, current guard format is extremely
        # verbose
        print(
            "\n".join(f"{repr(guard)}" for guard in sorted(self.__tx.output.guards)),
            file=file,
        )

    def _i_will_not_complain_if_bc_breaks_InstructionTranslator(
        self,
    ) -> InstructionTranslatorBase:
        # [FUNCTION DOC] Escape hatch returning the raw translator; no compatibility promise.
        """
        Returns the internal data structure InstructionTranslator that Dynamo
        uses to track state of symbolic evaluation.  There are no BC
        guarantees on this API and WE RESERVE THE RIGHT TO BREAK YOUR CODE if
        you rely on it.
        """
        return self.__tx

    def sleep(self, sec: int | float) -> None:
        # [FUNCTION DOC] Block the compiling thread for sec seconds.
        time.sleep(sec)


class _Comptime:
    # [CLASS DOC] Stateless namespace behind the comptime singleton. Each method is meant to be
    # intercepted by Dynamo (fn is invoked at trace time with a ComptimeContext); when run in
    # plain Python the methods are mostly no-ops or call their fallback.
    @staticmethod
    def __call__(
        fn: Callable[[ComptimeContext], Any],
        fallback_fn: Callable[[], Any] = lambda: None,
    ) -> Any:
        # [FUNCTION DOC] Under Dynamo, fn runs at compile time with the context. In eager mode,
        # call fallback_fn (default does nothing) and discard its result.
        # KNOWN ISSUE: declared to return Any but returns None always.
        """fn gets called at compile time in TorchDynamo, calls fallback_fn otherwise"""
        fallback_fn()

    # Convenience wrappers that are more compact to use

    @staticmethod
    def graph_break() -> None:
        # [FUNCTION DOC] Compile-time graph break via the context; no-op in eager.
        comptime(lambda ctx: ctx.graph_break())

    @staticmethod
    def print(e: Any) -> None:
        # [FUNCTION DOC] Under Dynamo print the compile-time info of e; in eager print e normally.
        comptime(lambda ctx: ctx.print(ctx.get_local("e")), lambda: print(e))

    @staticmethod
    def print_graph() -> None:
        # [FUNCTION DOC] Compile-time print of the partial FX graph; no-op in eager.
        comptime(lambda ctx: ctx.print_graph())

    @staticmethod
    def print_disas(*, stacklevel: int = 0) -> None:
        # [FUNCTION DOC] Compile-time disassembly print; stacklevel is shifted by one to skip the
        # wrapper frame. No-op in eager.
        comptime(
            lambda ctx: ctx.print_disas(
                stacklevel=ctx.get_local("stacklevel").as_python_constant() + 1
            )
        )

    @staticmethod
    def print_value_stack(*, stacklevel: int = 0) -> None:
        # [FUNCTION DOC] Compile-time value stack print; stacklevel shifted by one. No-op in eager.
        comptime(
            lambda ctx: ctx.print_value_stack(
                stacklevel=ctx.get_local("stacklevel").as_python_constant() + 1
            )
        )

    # This is a more useful variant of print_value_stack that can be used
    # in an expression context; e.g., x + print_value_stack_and_return(y + z),
    # you will see x on the stack prior to the addition operation
    @staticmethod
    def print_value_stack_and_return(e: Any, *, stacklevel: int = 0) -> Any:
        # [FUNCTION DOC] Like print_value_stack but usable inside an expression: prints the stack at
        # compile time (stacklevel shifted by one) and returns e unchanged.
        comptime(
            lambda ctx: ctx.print_value_stack(
                stacklevel=ctx.get_local("stacklevel").as_python_constant() + 1
            )
        )
        return e

    @staticmethod
    def print_locals(*, stacklevel: int = 0) -> None:
        # [FUNCTION DOC] Compile-time locals print; stacklevel shifted by one. No-op in eager.
        comptime(
            lambda ctx: ctx.print_locals(
                stacklevel=ctx.get_local("stacklevel").as_python_constant() + 1
            )
        )

    @staticmethod
    def print_bt(*, stacklevel: int = 0) -> None:
        # [FUNCTION DOC] Compile-time backtrace print; stacklevel shifted by one. No-op in eager.
        comptime(
            lambda ctx: ctx.print_bt(
                stacklevel=ctx.get_local("stacklevel").as_python_constant() + 1
            )
        )

    @staticmethod
    def print_guards() -> None:
        # [FUNCTION DOC] Compile-time print of installed guards. No-op in eager.
        comptime(lambda ctx: ctx.print_guards())

    @staticmethod
    def assert_static(val: Any) -> None:
        # [FUNCTION DOC] At compile time raise if val is dynamic. No-op in eager.
        comptime(lambda ctx: ctx.assert_static(ctx.get_local("val")))

    @staticmethod
    def force_static(val: Any) -> None:
        # [FUNCTION DOC] At compile time force val static (guard on its value). No-op in eager.
        comptime(lambda ctx: ctx.get_local("val").force_static())

    @staticmethod
    def breakpoint() -> None:
        # [FUNCTION DOC] At compile time open the builtin debugger with ctx bound to the caller's
        # context. No-op in eager.
        """
        Like pdb breakpoint(), but drop into pdb whenever this line
        of code is compiled by dynamo.  Use it by putting
        this in your model code::

            from torch._dynamo.comptime import comptime

            comptime.breakpoint()

        And then, inside pdb, you can access 'ctx' to query things
        about the compilation context::

            (Pdb) !ctx.print_bt()
            (Pdb) !ctx.print_locals()
            (Pdb) p ctx.get_local("attention").as_fake()
        """

        def inner(inner_ctx: ComptimeContext) -> None:
            # [FUNCTION DOC] HELPER: Compile-time callback; derive the parent context as ctx and
            # drop into the debugger.
            ctx = inner_ctx.parent()  # noqa: F841
            builtins.breakpoint()

        comptime(inner)

    @staticmethod
    def sleep(sec: int | float) -> None:
        # [FUNCTION DOC] At compile time sleep sec seconds. No-op in eager.
        comptime(lambda ctx: ctx.sleep(ctx.get_local("sec").as_python_constant()))


comptime = _Comptime()
