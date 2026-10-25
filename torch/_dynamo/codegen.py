# [MODULE DOC]
# Bytecode emission for Dynamo output code. Defines GraphOutputEntry (a graph output
# slot plus its variable) and PyCodegen, which accumulates Instruction objects (and
# optionally equivalent Python source lines) that reconstruct variables/sources onto the
# stack, load graph outputs, call the generated graph function, and perform helper
# loads/stores. Handles Python-version differences (3.10 to 3.13+). No import-time
# side effects beyond imports.
"""
This module provides utilities for generating Python bytecode in PyTorch's Dynamo system.
It includes functionality for:
- Constructing bytecode sequences for Python operations
- Managing stack operations and variable tracking
- Handling graph outputs and their conversions
- Supporting different Python versions (3.11+, 3.12+, 3.13+)
- Converting high-level operations to low-level bytecode instructions
- Managing constant loading and attribute access
- Supporting function creation and closure handling
"""

import collections
import dataclasses
import re
import sys
import types
from collections import Counter, deque
from collections.abc import Callable, Iterable
from typing import Any, TYPE_CHECKING, Union

import torch.nn
from torch._library.opaque_object import is_opaque_constant_type
from torch.utils._ordered_set import OrderedSet

from . import config, graph_break_hints, utils
from .bytecode_transformation import (
    add_push_null,
    add_push_null_call_function_ex,
    create_binary_subscr,
    create_build_tuple,
    create_call_function,
    create_call_function_ex,
    create_call_method,
    create_dup_top,
    create_instruction,
    create_load_const,
    create_load_method,
    create_rot_n,
    Instruction,
)
from .exc import unimplemented
from .source import AttrSource, ChainedSource, DictGetItemSource, Source
from .utils import is_safe_constant, rot_n_helper
from .variables.base import ValueMutationExisting, VariableTracker
from .variables.functions import (
    ContextlibContextManagerLocalGeneratorObjectVariable,
    LocalGeneratorObjectVariable,
)
from .variables.lazy import ComputedLazyConstantVariable
from .variables.nn_module import NNModuleVariable
from .variables.script_object import CustomClassObjectVariable
from .variables.tensor import (
    NumpyNdarrayVariable,
    SymNodeVariable,
    TensorVariable,
    UnspecializedPythonVariable,
)
from .variables.torch_function import TensorWithTFOverrideVariable


if TYPE_CHECKING:
    from torch._dynamo.variables.builder import GraphArg

    from .symbolic_convert import InstructionTranslatorBase


@dataclasses.dataclass
class GraphOutputEntry:
    # [CLASS DOC]
    # Record for one value returned by the compiled graph: its position (index) in the
    # graph output list and the VariableTracker that produced it.
    index: int
    variable: VariableTracker


class PyCodegen:
    # [CLASS DOC]
    # Builder of output bytecode for a translator. State: the instruction list, a
    # top_of_stack tracker (last value pushed, enabling a duplicate instead of
    # re-reconstruction), a use counter, graph_outputs (mapping from proxy id to
    # GraphOutputEntry), tempvars (values/sources cached in local temps), source
    # overrides, and optional generated Python source lines. Key methods: __call__
    # (push a value), make_call_generated_code, load/store helpers. Invariant: any
    # append that changes the stack unpredictably must clear top_of_stack.
    """
    Helper class uses for constructing Python bytecode
    """

    def __init__(
        self,
        tx: "InstructionTranslatorBase",
        root: torch.nn.Module | None = None,
        graph_output_var: str | None = None,
        tempvars: dict[VariableTracker | Source, Any] | None = None,
        overridden_sources: dict[Source, Source] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Initializes codegen for a translator: stores the root module, the local name
        # holding graph outputs, shared tempvars and source overrides, and captures
        # code options, cell/freevar accessor and new-variable factory from the
        # translator's output graph. Starts with empty output and no top of stack.
        self.root = root
        self.top_of_stack: VariableTracker | Source | None = None
        self.uses: Counter[VariableTracker | Source] = collections.Counter()
        self.graph_outputs: dict[int, GraphOutputEntry] = {}
        self._output: list[Instruction] = []
        # This determines which VariableTracker/Source should be stored as
        # locals, and maps the VariableTracker/Source to the local variable
        # name. Note that it could map to None initially, in which case we'll
        # overwrite it to map to real temporary names via `add_cache`.
        self.tempvars: dict[VariableTracker | Source, Any] = tempvars or {}
        self.tx = tx
        self.graph_output_var = graph_output_var
        self.code_options = self.tx.output.code_options
        self.cell_and_freevars = self.tx.cell_and_freevars
        self.new_var = self.tx.output.new_var
        self.value_from_source: bool = True
        # This serves as a way for codegen to use a different source; we need
        # this because sometimes we can't easily modify the original source
        # without affecting other components, e.g., guards.
        self.overridden_sources: dict[Source, Source] = overridden_sources or {}
        self.pycodes = []

    def add_pycode(self, pycode: str, *args):
        # [FUNCTION DOC]
        # When config.generate_pycode is on, realizes any lazy variable arguments,
        # formats the template with each argument's Python-source reconstruction, and
        # appends the line to the pycode list. No-op otherwise.
        if not config.generate_pycode:
            return
        for a in args:
            if isinstance(a, VariableTracker):
                a.realize()
        self.pycodes.append(pycode.format(*[a.reconstruct_pycode(self) for a in args]))

    def restore_stack(
        self, stack_values: list[Any], *, value_from_source: bool = True
    ) -> None:
        # [FUNCTION DOC]
        # Pushes each stack value in order, optionally forcing reconstruction instead
        # of loading from source (value_from_source False). Resets the "stack" pycode
        # variable counter so repeated passes give the same names, and records pycode
        # assignments for each value. Restores the previous value_from_source flag even
        # on error.
        prev = self.value_from_source
        self.value_from_source &= value_from_source
        try:
            self.foreach(stack_values)
            # `restore_stack` is called once per codegen pass (e.g. pass1 for
            # use tracking, pass2 for actual emission). Reset the stack counter
            # so each pass produces the same `__stackN` names.
            self.tx.reset_pycode_varname_counter("stack")
            for v in stack_values:
                self.add_pycode(f"{self.tx.new_pycode_varname('stack')} = {{}}", v)
        finally:
            self.value_from_source = prev

    def graph_output_vars(self) -> list[VariableTracker]:
        # [FUNCTION DOC]
        # Returns the variables registered as graph outputs, in registration order.
        return [x.variable for x in self.graph_outputs.values()]

    def call_reconstruct(
        self, value: Union[VariableTracker, Source, "GraphArg"]
    ) -> None:
        # [FUNCTION DOC]
        # Asks a variable, source or graph arg to emit its reconstruction bytecode into
        # this codegen. Raises an error if reconstruct returns anything other than
        # nothing; may raise NotImplementedError from the callee.
        res = value.reconstruct(self)
        if res is not None:
            raise AssertionError(f"reconstruct!=None {value}")

    def add_push_null(
        self, gen_fn: Callable[[], None], call_function_ex: bool = False
    ) -> None:
        # [FUNCTION DOC]
        # Runs gen_fn (which must push one callable), then rewrites the emitted
        # instructions so a NULL is pushed in the version-appropriate way (using the
        # NULL bit of load instructions when possible; a variant for call-function-ex).
        # On Python before 3.13 the top-of-stack is cleared first; on 3.13+ afterwards.
        """
        `gen_fn` generates instructions via PyCodegen methods
        that push a single callable to the stack.

        `add_push_null` pushes a NULL to the stack before or after the
        instructions generated by `gen_fn`, depending on Python version.

        Will attempt to use the NULL push bit for instructions
        with such bits (LOAD_GLOBAL 3.11+, LOAD_ATTR 3.12+, LOAD_SUPER_ATTR).
        """
        old_len = len(self._output)
        if sys.version_info < (3, 13):
            # gen_fn may DUP_TOP instead if TOS is not cleared.
            # Will cause problems since NULL will be pushed right
            # before the generated instructions in <= 3.12
            self.clear_tos()
        gen_fn()
        # inplace modify self._output
        added_insts = self._output[old_len:]
        del self._output[old_len:]
        if call_function_ex:
            self._output.extend(add_push_null_call_function_ex(added_insts))
        else:
            self._output.extend(add_push_null(added_insts))
        if sys.version_info >= (3, 13):
            # NULL will be at top of stack
            self.clear_tos()

    def __call__(
        self, value: VariableTracker | Source | None, allow_cache: bool = True
    ) -> None:
        # [FUNCTION DOC]
        # Emits bytecode leaving the given value (VariableTracker or Source) on top of
        # the stack. Order of strategies: duplicate top-of-stack if the same value was
        # just pushed; load a cached temp; reconstruct from the value's source (to keep
        # aliasing) when allowed; otherwise special-case constants, tensor-subclass
        # overrides, float symbolic nodes, graph outputs (tensors, numpy, opaque
        # objects), nn modules (via root and attribute path), and finally call the
        # variable's reconstruct. Caches into a temp when the value is marked a temp.
        # Raises an error on None input, a Source with allow_cache False, or non
        # variable input; triggers a graph break (unimplemented) when reconstruction is
        # unsupported or a contextmanager generator object is returned. Updates uses,
        # top_of_stack and graph_outputs.
        """
        Generate code such that top-of-stack (TOS) is set to value.

        `allow_cache` controls the behavior in the following manner. `value` can
        either be a VariableTracker or a Source.

        If `value` is a `Source`, `allow_cache` must be True (invariant asserted
        below). If the source was reconstructed earlier, we will reuse the
        generated code by loading from top of stack or tempvars.

        If `value` is a `VariableTracker`, we have the following cases:

        1) `allow_cache=True`
            a) If the value.source is not None, we will emit the code based on
            `value.source` to handle aliasing.
            b) If value.source is None (example reconstructing a local list
            returned by the compiled function), we will reconstruct the variable
            tracker (w/o any source) to emit bytecode that generates a new
            python object.

            In both cases of value.source being None or not, if the value was
            reconstructed earlier, we will reuse the generated code by loading from
            top of stack or tempvars.

        2) `allow_cache=False` - This is a special case (allow_cache defaults to
        True).
            a) If the value.source is not None, we reconstruct the variable
            tracker and emit a new python object. You might wonder what about
            aliasing? The place where we use this config also has the followup
            code where the original python object is assigned to this new python
            value to handle aliasing (check side_effects.py and search for
            allow_cache=False).

            b) If value.source is None, this is not allowed

        Notable effects:
        1. `self.top_of_stack` will be set to `value`, if we don't codegen
           `value` based on source.
        2. `self.uses[value]` will increment, unless (a). we codegen via
            `top_of_stack` or cached `tempvars`, or (b). `value` has special VT
            types like `NNModuleVariable`, etc.
        """
        if value is None:
            raise AssertionError("value must not be None")
        if isinstance(value, Source):
            # If the source needs to be overridden, use the new one.
            source = self.overridden_sources.get(value, value)
            if allow_cache is not True:
                raise AssertionError("allow_cache must be True for Source")
            if self.top_of_stack is value:
                self._output.append(create_dup_top())
                return

            if self.tempvars.get(source) is not None:
                self._output.append(self.create_load(self.tempvars[source]))
                self.top_of_stack = source
                return

            self.uses[source] += 1
            try:
                self.call_reconstruct(source)
            except NotImplementedError:
                unimplemented(
                    gb_type="Reconstruction failure: source.reconstruct not implemented",
                    context=str(source),
                    explanation=f"Dynamo has no bytecode reconstruction implemented for {type(source)} variable {source}.",
                    hints=[*graph_break_hints.DYNAMO_BUG],
                )
            if source in self.tempvars:
                self._output.append(create_dup_top())
                self.add_cache(source)
            self.top_of_stack = source

            return

        if not isinstance(value, VariableTracker):
            raise AssertionError(f"expected VariableTracker, got {type(value)}")
        output = self._output
        graph_outputs = self.graph_outputs

        if allow_cache:
            if self.top_of_stack is value:
                output.append(create_dup_top())
                return

            if self.tempvars.get(value) is not None:
                output.append(self.create_load(self.tempvars[value]))
                self.top_of_stack = value
                return

        if value.is_realized() and isinstance(
            value, ContextlibContextManagerLocalGeneratorObjectVariable
        ):
            unimplemented(
                gb_type="reconstructing @contextmanager object",
                context=f"object: {value}",
                explanation="Returning a @contextmanager object from a compiled function is not supported.",
                hints=[
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        # Dynamo normally prefers codegen from source to account for aliasing.
        if (
            value.source is not None
            and allow_cache
            and not (
                value.is_realized() and isinstance(value, LocalGeneratorObjectVariable)
            )
        ):
            # There's a corner case for export: for instance, if the computation
            # graph is just identity on an input tensor, Dynamo would just emit
            # a `LOAD_FAST` from the input source, rather than generating an
            # identity FX graph.
            #
            # However, export wants to maximize graph capture; in the case
            # above, export _wants to_ obtain an identity FX graph (despite it
            # appears unnecessarily expensive for `torch.compile`), so we have
            # the following option to override Dynamo's preference for codegen
            # from source. Moreover, this option applies recursively, for cases
            # like input tensor being returned in a new dictionary.
            #
            # And why the `ValueMutationExisting` check? Not sure, so leaving it
            # to keep the old behavior, as when `value_from_source` was
            # introduced. TODO sort out the invariants among side effect,
            # codegen and export.
            if (
                isinstance(value.mutation_type, ValueMutationExisting)
                or self.value_from_source
            ):
                return self(value.source)

        if isinstance(value, ComputedLazyConstantVariable) and not value.is_realized():
            # Recompute from the operands at runtime instead of burning in the value
            self.uses[value] += 1
            self.call_reconstruct(value)
            if allow_cache and value in self.tempvars:
                self._output.append(create_dup_top())
                self.add_cache(value)
        elif value.is_python_constant() and is_safe_constant(
            value.as_python_constant()
        ):
            output.append(self.create_load_const(value.as_python_constant()))
        elif isinstance(value, TensorWithTFOverrideVariable):
            graph_outputs_key = self.add_graph_output(value)

            self.add_push_null(
                lambda: self.load_import_from(utils.__name__, "to_subclass")
            )
            self.load_graph_output(graph_outputs[graph_outputs_key].index)
            output.append(
                self.create_load_global(
                    value.global_mangled_class_name(self.tx),  # type: ignore[arg-type]
                    add=True,
                )
            )
            output.extend(create_call_function(2, False))
        elif (
            isinstance(value, SymNodeVariable)
            and value.python_type() is float
            and not self.tx.export
        ):
            # This is a little unusual; force the output convention to be a
            # Tensor here.  Don't do this for export because this is
            # apparently load bearing for export tests (but I am a bit
            # doubtful it actually works in the real world)
            # NB: It works to add_graph_output on a computed expression
            # as_tensor here, because we memoize as_tensor calls on
            # SymNodeVariable!
            graph_outputs_key = self.add_graph_output(
                value.as_tensor(self.tx, torch.float64)
            )

            def gen_fn() -> None:
                self.load_graph_output(graph_outputs[graph_outputs_key].index)
                output.append(self.create_load_attr("item"))

            self.add_push_null(gen_fn)
            output.extend(create_call_function(0, False))
        elif isinstance(
            value,
            (
                TensorVariable,
                SymNodeVariable,
                UnspecializedPythonVariable,
                NumpyNdarrayVariable,
                CustomClassObjectVariable,
            ),
        ):
            graph_outputs_key = self.add_graph_output(value)

            if isinstance(value, NumpyNdarrayVariable):
                self.add_push_null(
                    lambda: self.load_import_from(utils.__name__, "to_numpy_helper")
                )
                self.load_graph_output(graph_outputs[graph_outputs_key].index)
                output.extend(create_call_function(1, False))
            elif isinstance(value, UnspecializedPythonVariable) and value.need_unwrap:

                def gen_fn() -> None:
                    self.load_graph_output(graph_outputs[graph_outputs_key].index)
                    output.append(self.create_load_attr("item"))

                self.add_push_null(gen_fn)
                output.extend(create_call_function(0, False))
            elif isinstance(value, CustomClassObjectVariable):
                try:
                    py_value = value.as_python_constant()
                except NotImplementedError:
                    py_value = None

                if is_opaque_constant_type(type(py_value)):
                    self.add_push_null(
                        lambda: self.load_import_from(
                            "torch._library.fake_class_registry",
                            "maybe_unwrap_fake_script_object",
                        )
                    )
                    self.load_graph_output(graph_outputs[graph_outputs_key].index)
                    output.extend(create_call_function(1, False))
                else:
                    self.load_graph_output(graph_outputs[graph_outputs_key].index)
            else:
                self.load_graph_output(graph_outputs[graph_outputs_key].index)
        elif isinstance(value, NNModuleVariable):
            parts = value.module_key.split(".")
            if parts[0] in self.code_options["co_varnames"]:
                output.append(self.create_load(parts[0]))
                parts = parts[1:]
            else:
                if self.root is None:
                    raise AssertionError("self.root must not be None")
                output.append(self.create_load_const_unchecked(self.root))
            for part in parts:
                output.append(self.create_load_attr(part))
        else:
            self.uses[value] += 1
            try:
                self.call_reconstruct(value)
            except NotImplementedError as e:
                unimplemented(
                    gb_type="Reconstruction failure",
                    context=str(value),
                    explanation=f"Dynamo has no bytecode reconstruction implemented for sourceless variable {value}.",
                    hints=[
                        "If Dynamo is attempting to trace a return statement and your code is attempting to return a variable "
                        "that Dynamo cannot reconstruct, then remove it from the return statement.",
                        *graph_break_hints.CAUSED_BY_EARLIER_GRAPH_BREAK,
                        "Report an issue to PyTorch if you need reconstrtuction support. Note that objects that don't have "
                        "reconstruction rules may be fundamentally unreconstructable.",
                    ],
                    from_exc=e,
                )
            if allow_cache and value in self.tempvars:
                self._output.append(create_dup_top())
                self.add_cache(value)

        self.top_of_stack = value

    def add_graph_output(self, value: VariableTracker) -> int:
        # [FUNCTION DOC]
        # Registers the variable's proxy as a graph output, keyed by proxy identity, if
        # not already present (assigning the next index). Returns the key.
        graph_outputs_key = id(value.as_proxy())
        if graph_outputs_key not in self.graph_outputs:
            self.graph_outputs[graph_outputs_key] = GraphOutputEntry(
                len(self.graph_outputs), value
            )
        return graph_outputs_key

    def load_graph_output(self, index: int) -> None:
        # [FUNCTION DOC]
        # Emits code to load the graph-output container local and subscript it with the
        # given index. Raises an error if graph_output_var is unset.
        output = self._output
        if self.graph_output_var is None:
            raise AssertionError("graph_output_var must not be None")
        output.append(self.create_load(self.graph_output_var))
        output.append(self.create_load_const(index))
        output.append(self.create_binary_subscr())

    def add_cache(self, value: VariableTracker | Source) -> None:
        # [FUNCTION DOC]
        # Allocates a fresh local variable, records it as the temp for the value, and
        # emits a store of the top of stack into it (the caller must have duplicated).
        var = self.new_var()
        self.tempvars[value] = var
        self._output.append(self.create_store(var))

    def foreach(self, items: Iterable[VariableTracker | Source]) -> None:
        # [FUNCTION DOC]
        # Pushes each item onto the stack in order via the main push method.
        for i in items:
            self(i)

    def create_binary_subscr(self) -> Instruction:
        # [FUNCTION DOC]
        # Returns a subscript instruction appropriate for the running Python version.
        return create_binary_subscr()

    def setup_globally_cached(self, name: str, value: Any) -> list[Instruction]:
        # [FUNCTION DOC]
        # Sanitizes the name (non-identifier characters become underscores), installs
        # the value in the translator's globals under it, and returns a one-element
        # list with a global load. Raises an error if the name already maps to a
        # different object. Side effect: mutates f_globals.
        """Store value in a new global"""
        name = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
        f_globals = self.tx.f_globals
        if name in f_globals:
            if id(f_globals[name]) != id(value):
                raise AssertionError(
                    f"f_globals[{name!r}] already exists with a different identity"
                )
        else:
            f_globals[name] = value
        return [self.create_load_global(name, add=True)]

    def clear_tos(self) -> None:
        # [FUNCTION DOC]
        # Forgets the tracked top-of-stack value so it will not be duplicated.
        self.top_of_stack = None

    def append_output(self, inst: Instruction) -> None:
        # [FUNCTION DOC]
        # Appends one instruction to the output and clears top-of-stack tracking.
        # Raises an error if the argument is not an Instruction.
        if not isinstance(inst, Instruction):
            raise AssertionError(f"expected Instruction, got {type(inst)}")
        self._output.append(inst)
        self.clear_tos()

    def extend_output(self, insts: list[Instruction]) -> None:
        # [FUNCTION DOC]
        # Appends all instructions to the output and clears top-of-stack tracking.
        # Raises an error if any element is not an Instruction.
        if not all(isinstance(x, Instruction) for x in insts):
            raise AssertionError("all elements of insts must be Instruction instances")
        self._output.extend(insts)
        self.clear_tos()

    def get_instructions(self) -> list[Instruction]:
        # [FUNCTION DOC]
        # Returns the live (not copied) list of emitted instructions.
        return self._output

    def get_pycode(self) -> list[str] | None:
        # [FUNCTION DOC]
        # Returns the accumulated Python source lines, or None when generate_pycode is
        # disabled in config.
        if not config.generate_pycode:
            return None
        return self.pycodes

    def create_load(self, name: str) -> Instruction:
        # [FUNCTION DOC]
        # Returns a fast-local load for the named variable. Raises an error if the name
        # is not among the code's local variable names.
        if name not in self.code_options["co_varnames"]:
            raise AssertionError(f"{name} missing")
        return create_instruction("LOAD_FAST", argval=name)

    def create_load_closure(self, name: str) -> Instruction:
        # [FUNCTION DOC]
        # Returns an instruction loading the cell object itself (closure load; a fast
        # load on 3.13+). Raises an error if the name is not a cell or free variable.
        if name not in self.cell_and_freevars():
            raise AssertionError(f"{name!r} not in cell_and_freevars")
        inst_name = "LOAD_FAST" if sys.version_info >= (3, 13) else "LOAD_CLOSURE"
        return create_instruction(inst_name, argval=name)

    def create_load_deref(self, name: str) -> Instruction:
        # [FUNCTION DOC]
        # Returns a load of the contents of the named cell/free variable. Raises an
        # error if the name is not a cell or free variable.
        if name not in self.cell_and_freevars():
            raise AssertionError(f"{name!r} not in cell_and_freevars")
        return create_instruction("LOAD_DEREF", argval=name)

    def create_store(self, name: str) -> Instruction:
        # [FUNCTION DOC]
        # Returns a fast-local store for the named variable. Raises an error if the name
        # is not among the code's local variable names.
        if name not in self.code_options["co_varnames"]:
            raise AssertionError(f"{name} missing")
        return create_instruction("STORE_FAST", argval=name)

    def create_store_deref(self, name: str) -> Instruction:
        # [FUNCTION DOC]
        # Returns a store into the named cell/free variable. Raises an error if the name
        # is not a cell or free variable.
        if name not in self.cell_and_freevars():
            raise AssertionError(f"{name!r} not in cell_and_freevars")
        return create_instruction("STORE_DEREF", argval=name)

    def create_load_global(self, name: str, add: bool = False) -> Instruction:
        # [FUNCTION DOC]
        # Returns a global load for the name. If add is true, first registers the name
        # in the code's names table (side effect on the output graph). Raises an error
        # if the name is still absent from the names table.
        if add:
            self.tx.output.update_co_names(name)
        if name not in self.code_options["co_names"]:
            raise AssertionError(f"{name} not in co_names")
        return create_instruction("LOAD_GLOBAL", argval=name)

    def create_load_const(self, value: Any) -> Instruction:
        # [FUNCTION DOC]
        # Returns a constant-load instruction; the shared helper checks the constant is
        # of a safe type.
        return create_load_const(value)

    def create_load_const_unchecked(self, value: Any) -> Instruction:
        # [FUNCTION DOC]
        # Returns a constant-load instruction without the safe-constant type check, for
        # arbitrary objects such as modules or helper functions.
        return create_load_const(value, checked=False)

    def load_method(self, name: str) -> None:
        # [FUNCTION DOC]
        # Registers the name and appends a method-load for it to the output, clearing
        # top-of-stack tracking.
        self.tx.output.update_co_names(name)
        self.append_output(create_load_method(name))

    def call_method(self, nargs: int) -> None:
        # [FUNCTION DOC]
        # Appends instructions calling a previously loaded method with nargs arguments.
        self.extend_output(create_call_method(nargs))

    def create_list_append(self) -> list[Instruction]:
        # [FUNCTION DOC]
        # Returns instructions appending the top of stack to the list beneath it,
        # leaving the list on the stack. Implemented as build-one-element-list plus
        # extend because the plain append opcode is unsafe on free-threaded builds.
        # Append TOS to the list at TOS-1, leaving the list on the stack
        # (same stack effect as LIST_APPEND with arg=1).
        #
        # The bare LIST_APPEND opcode does not lock the list and so requires
        # the target be uniquely owned (refcnt == 1) on free-threaded builds.
        # Dynamo can't enforce this, so instead use LIST_EXTEND, which does
        # lock
        return [
            create_instruction("BUILD_LIST", arg=1),
            create_instruction("LIST_EXTEND", arg=1),
        ]

    def create_load_attr(self, name: str) -> Instruction:
        # [FUNCTION DOC]
        # Returns an attribute-load instruction, first appending the name to the code
        # options names tuple if missing (mutates code options).
        if name not in self.code_options["co_names"]:
            self.code_options["co_names"] += (name,)
        return create_instruction("LOAD_ATTR", argval=name)

    def load_attr(self, name: str) -> None:
        # [FUNCTION DOC]
        # Appends an attribute load for the name to the output; clears top-of-stack.
        self.append_output(self.create_load_attr(name))

    def create_load_attrs(self, names: str) -> list[Instruction]:
        # [FUNCTION DOC]
        # Splits a dotted path and returns a chain of attribute-load instructions, one
        # per component (does not append to the output).
        return [self.create_load_attr(name) for name in names.split(".")]

    def create_store_attr(self, name: str) -> Instruction:
        # [FUNCTION DOC]
        # Returns an attribute-store instruction, registering the name in the code
        # options names tuple if missing (mutates code options).
        if name not in self.code_options["co_names"]:
            self.code_options["co_names"] += (name,)
        return create_instruction("STORE_ATTR", argval=name)

    def store_attr(self, name: str) -> None:
        # [FUNCTION DOC]
        # Appends an attribute store for the name to the output; clears top-of-stack.
        self.append_output(self.create_store_attr(name))

    def load_function_name(
        self, fn_name: str, push_null: bool, num_on_stack: int = 0
    ) -> list[Instruction]:
        # [FUNCTION DOC]
        # Returns instructions that load a global function and place it (with a NULL
        # beneath/above it on 3.11+ when push_null) below num_on_stack existing stack
        # items, using rotations. Returned, not appended. Registers the global name.
        """Load the global fn_name on the stack num_on_stack down"""
        output = []
        if push_null and sys.version_info >= (3, 11):
            output.extend(add_push_null(self.create_load_global(fn_name, add=True)))
            if num_on_stack > 0:
                output.extend(
                    [
                        *self.rot_n(num_on_stack + 2),
                        *self.rot_n(num_on_stack + 2),
                    ]
                )
        else:
            output.extend(
                [
                    self.create_load_global(fn_name, add=True),
                    *self.rot_n(num_on_stack + 1),
                ]
            )
        return output

    def rot_n(self, n: int) -> list[Instruction]:
        # [FUNCTION DOC]
        # Returns instructions rotating the top n stack items. If the interpreter lacks
        # a native rotate for n, falls back to packing a tuple, calling a generated
        # helper function on it and unpacking the result.
        try:
            return create_rot_n(n)
        except AttributeError:
            # desired rotate bytecode doesn't exist, generate equivalent bytecode
            return [
                create_build_tuple(n),
                self.create_load_const_unchecked(rot_n_helper(n)),
                *create_rot_n(2),
                *create_call_function_ex(False, False),
                create_instruction("UNPACK_SEQUENCE", arg=n),
            ]

    def pop_null(self) -> list[Instruction]:
        # [FUNCTION DOC]
        # Returns instructions discarding a NULL on the stack by calling a no-op
        # function with it and popping the result (a plain pop cannot remove NULL).
        # Raises an error on Python before 3.11.
        # POP_TOP doesn't work for null, so we pop nulls by pushing in a
        # nop function, calling it (which consumes the null), and popping the result.
        if sys.version_info < (3, 11):
            raise AssertionError("pop_null requires Python 3.11+")
        return [
            self.create_load_const_unchecked(lambda: None),
            # 3.13 swapped NULL and callable
            *(
                (create_instruction("SWAP", arg=2),)
                if sys.version_info >= (3, 13)
                else ()
            ),
            *create_call_function(0, False),
            create_instruction("POP_TOP"),
        ]

    def pop_top(self) -> None:
        # [FUNCTION DOC]
        # Appends an instruction discarding the top of stack; clears top-of-stack.
        self.append_output(create_instruction("POP_TOP"))

    def call_function(self, nargs: int, push_null: bool) -> None:
        # [FUNCTION DOC]
        # Appends version-appropriate instructions calling a stack callable with nargs
        # positional arguments, optionally pushing the NULL as part of the call setup.
        self.extend_output(create_call_function(nargs, push_null=push_null))

    def dup_top(self) -> None:
        # [FUNCTION DOC]
        # Appends an instruction duplicating the top of stack; clears top-of-stack.
        self.append_output(create_dup_top())

    def store(self, varname: str) -> None:
        # [FUNCTION DOC]
        # Appends a fast-local store of the top of stack into varname (must be a known
        # local name, else an error is raised); clears top-of-stack.
        self.append_output(self.create_store(varname))

    def load_deref(self, varname: str) -> None:
        # [FUNCTION DOC]
        # Appends a load of the named cell/free variable's contents; clears
        # top-of-stack. Raises an error for an unknown cell name.
        self.append_output(self.create_load_deref(varname))

    def make_function_with_closure(
        self,
        fn_name: str,
        code: types.CodeType,
    ) -> None:
        # [FUNCTION DOC]
        # Emits a code-constant load (plus qualified name before 3.11) and a make-function
        # with the closure flag, consuming the tuple of cells at top of stack and
        # leaving the new function. On 3.13+ uses make-function then set-attribute.
        # Clears top-of-stack.
        """Creates a closure with code object `code`.

        Expects the TOS to be the tuple of cells to use for this closure.
        TOS will be popped to create the closure.
        Args:
            - fn_name: name of the function
            - code: code object of the function
                (does not include the tuple of cells on the TOS)
        """
        output = self._output

        output.append(self.create_load_const(code))
        if sys.version_info < (3, 11):
            output.append(self.create_load_const(fn_name))
        if sys.version_info >= (3, 13):
            output.extend(
                [
                    create_instruction("MAKE_FUNCTION"),
                    create_instruction("SET_FUNCTION_ATTRIBUTE", arg=0x08),
                ]
            )
        else:
            output.append(create_instruction("MAKE_FUNCTION", arg=0x08))

        self.clear_tos()

    def create_load_python_module(self, mod: types.ModuleType) -> Instruction:
        # [FUNCTION DOC]
        # Returns a global load for the module: uses its short name if the global scope
        # already binds that name to the same module, otherwise installs it under a
        # generated unique global name (side effect on output graph globals).
        """
        Generate a LOAD_GLOBAL instruction to fetch a given python module.
        """
        output = self.tx.output
        global_scope = output.global_scope
        name = re.sub(r"^.*[.]", "", mod.__name__)
        if global_scope.get(name, None) is mod:
            return self.create_load_global(name, add=True)
        prefix = f"___module_{name}"
        global_name = self.tx.output.install_global_by_id(prefix, mod)
        return self.create_load_global(global_name, add=True)

    def mark_source_temp(self, source: Source) -> None:
        # [FUNCTION DOC]
        # Flags the source for caching in a temp local by adding a placeholder entry to
        # tempvars if absent; the real variable name is assigned on first use.
        """
        Mark a source as a temp variable, so that it can be reused.
        """
        if source not in self.tempvars:
            self.tempvars[source] = None

    def make_call_generated_code(self, fn_name: str) -> None:
        # [FUNCTION DOC]
        # Emits the call to the compiled graph function named fn_name: loads it, finds
        # sources used more than once among graph args and marks them temps (a form of
        # common subexpression elimination), reconstructs every graph arg (wrapping
        # via as-tensor when requested), optionally brackets this with runtime-overhead
        # recording calls, then calls with len(graphargs) arguments. Also records
        # equivalent pycode lines. Mutates output, tempvars and pycode state.
        """Call the generated code function stored in fn_name"""
        self.extend_output(self.load_function_name(fn_name, True))

        graphargs = self.tx.output.graphargs

        def extract_nested_sources(source: Source) -> list[Source]:
            # [FUNCTION DOC] HELPER: Returns the immediate sub-sources of a source: the
            # base of a chained source, plus the index of a dict-item source when that
            # index is itself a source.
            nested_sources: list[Source] = []
            if isinstance(source, ChainedSource):
                nested_sources.append(source.base)
            if isinstance(source, DictGetItemSource) and isinstance(
                source.index, Source
            ):
                nested_sources.append(source.index)
            return nested_sources

        def collect_temp_sources(sources: deque[Source], codegen: PyCodegen) -> None:
            # [FUNCTION DOC] HELPER: Breadth-first walk over sources and their nested
            # sources; any source seen a second time is marked as a temp on the codegen
            # and not traversed further, to limit how many nodes become temps.
            seen_sources: OrderedSet[Source] = OrderedSet()
            while sources:
                current_source = sources.popleft()
                if current_source in seen_sources:
                    # This source is used at least twice, so it can be reused
                    codegen.mark_source_temp(current_source)
                    # Don't trace source further. This prevents us from marking too
                    # many nodes as temp sources.
                    continue
                seen_sources.add(current_source)
                sources.extend(extract_nested_sources(current_source))

        # Collect all the sources that are used more than once, so that we can
        # generate tmp variables in the generated pre-graph bytecode. This
        # essentially implements CSE.
        collect_temp_sources(
            deque([arg.source for arg in graphargs if arg.source is not None]), self
        )

        cm_var = None
        if config.record_runtime_overhead:
            # Record the pregraph bytecode start
            self.add_push_null(
                lambda: self.load_import_from(
                    utils.__name__, "record_pregraph_bytecode_enter"
                )
            )
            self.extend_output(create_call_function(0, False))
            cm_var = self.new_var()
            self.store(cm_var)

        arg_varnames = []
        for i, arg in enumerate(graphargs):
            arg_varname = self.tx.new_pycode_varname("arg")
            arg_varnames.append(arg_varname)
            if arg.pass_arg_as_tensor:
                self.add_push_null(
                    lambda: self.extend_output(
                        [
                            self.create_load_python_module(torch),
                            self.create_load_attr("_as_tensor_fullprec"),
                        ]
                    )
                )
                self.call_reconstruct(arg)
                self.extend_output(create_call_function(1, False))
                self.add_pycode(f"{arg_varname} = torch._as_tensor_fullprec({{}})", arg)
            else:
                self.call_reconstruct(arg)
                self.add_pycode(f"{arg_varname} = {{}}", arg)

        if config.record_runtime_overhead:
            # Record the pregraph bytecode end
            self.add_push_null(
                lambda: self.load_import_from(
                    utils.__name__, "record_pregraph_bytecode_exit"
                )
            )
            if cm_var is None:
                raise AssertionError("cm_var must not be None")
            self.extend_output([self.create_load(cm_var)])
            self.extend_output(create_call_function(1, False))
            self.pop_top()

        self.extend_output(create_call_function(len(graphargs), False))
        self.add_pycode(
            f"__graph_out = {fn_name}({', '.join(arg_varnames)})",
        )

    def create_import_name(self, module_name: str) -> Instruction:
        # [FUNCTION DOC]
        # Returns an import-name instruction for the module name.
        return create_instruction("IMPORT_NAME", argval=module_name)

    def load_import_from(self, module_name: str, object_name: str) -> None:
        # [FUNCTION DOC]
        # Pushes an attribute of an imported module (an attribute source on the
        # translator's import source) onto the stack, always marking it as a temp so
        # repeated loads are cached.
        source = AttrSource(self.tx.import_source(module_name), object_name)
        # Note: This approach is somewhat aggressive because typically, a source is marked
        # as a tempvar only when it is used more than once. In this case, we're marking it
        # as a tempvar without performing that analysis. However, this is a simple solution,
        # and in many cases, load imports are reused multiple times.
        self.mark_source_temp(source)
        self(source)

    def create_call_function_kw(
        self, nargs: int, kw_names: Iterable[str], push_null: bool
    ) -> list[Instruction]:
        # [FUNCTION DOC]
        # Returns instructions calling a stack callable with nargs arguments where the
        # last len(kw_names) are keyword arguments named by kw_names. Uses the
        # keyword-call opcode on 3.13+, a keyword-names instruction before the call on
        # 3.11/3.12, and a load-const plus keyword-call opcode before 3.11. Raises an
        # error if the base call sequence lacks the expected call instruction.
        if sys.version_info >= (3, 13):
            output = create_call_function(nargs, push_null)
            if output[-1].opname != "CALL":
                raise AssertionError(
                    f"expected last instruction to be CALL, got {output[-1].opname}"
                )
            output.insert(-1, self.create_load_const(kw_names))
            output[-1] = create_instruction("CALL_KW", arg=nargs)
            return output
        elif sys.version_info >= (3, 11):
            output = create_call_function(nargs, push_null)
            if sys.version_info >= (3, 12):
                idx = -1
                expected_inst = "CALL"
            else:
                idx = -2
                expected_inst = "PRECALL"
            if output[idx].opname != expected_inst:
                raise AssertionError(
                    f"expected instruction at index {idx} to be {expected_inst}, "
                    f"got {output[idx].opname}"
                )
            kw_names_inst = create_instruction("KW_NAMES", argval=kw_names)
            output.insert(idx, kw_names_inst)
            return output
        return [
            self.create_load_const(kw_names),
            create_instruction("CALL_FUNCTION_KW", arg=nargs),
        ]

    def create_delete(self, value: object) -> Instruction:
        # [FUNCTION DOC]
        # Returns a fast-local delete instruction for the given variable name.
        return create_instruction("DELETE_FAST", argval=value)
