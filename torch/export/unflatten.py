# mypy: allow-untyped-defs
# [MODULE DOC]
# Implements torch.export.unflatten: turns a flat ExportedProgram graph back into
# a module hierarchy matching the original eager module. Public contents:
# unflatten, UnflattenedModule (root module), InterpreterModule and
# InterpreterModuleDispatcher (per-call submodule types), FlatArgsAdapter
# (abstract input adapter). Private contents: _ModuleFrame and helpers that
# outline graph nodes into submodules, sink parameters/buffers/constants to the
# modules that use them, deduplicate repeated module calls, and inline or
# reorder modules. Import-time side effects: creates the module logger and the
# module-level flag RUN_WITH_INTERPRETER (default True).
import abc
import copy
import logging
import operator
import re
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import is_custom_class_obj
from torch.export import ExportedProgram
from torch.export._tree_utils import reorder_kwargs
from torch.export.exported_program import (
    ConstantArgument,
    ExportGraphSignature,
    InputKind,
    ModuleCallSignature,
    SymBoolArgument,
    SymFloatArgument,
    SymIntArgument,
    TensorArgument,
)
from torch.fx._symbolic_trace import is_fx_symbolic_tracing
from torch.fx.graph_module import _get_attr, _get_attr_via_attr_list, _print_readable
from torch.utils._pytree import GetAttrKey, SequenceKey

from ._remove_effect_tokens_pass import _remove_effect_tokens


log = logging.getLogger(__name__)


__all__ = [
    "FlatArgsAdapter",
    "InterpreterModule",
    "InterpreterModuleDispatcher",
    "UnflattenedModule",
    "unflatten",
]


class _AttrKind(Enum):
    # [CLASS DOC] HELPER: Enumeration of kinds of attributes installed on
    # modules during unflattening: parameter, buffer, constant, or module.
    PARAMETER = "parameter"
    BUFFER = "buffer"
    CONSTANT = "constant"
    MODULE = "module"


@dataclass(frozen=True)
class _TensorID:
    # [CLASS DOC] HELPER: Frozen (hashable) identity key for a tensor made of its
    # untyped storage, stride, size and storage offset; used to detect aliased
    # tensors that share the same memory view.
    """Custom tensor identifier containing storage, stride, and size information."""

    untyped_storage: torch.UntypedStorage
    stride: tuple
    size: tuple
    storage_offset: int


RUN_WITH_INTERPRETER = True


@contextmanager
def _disable_interpreter():
    # [FUNCTION DOC] HELPER: Context manager that temporarily sets the module
    # global RUN_WITH_INTERPRETER to False, so modules constructed inside the
    # block run via generated code instead of the interpreter. The previous value
    # is restored on exit, even on error.
    global RUN_WITH_INTERPRETER
    old_flag = RUN_WITH_INTERPRETER
    RUN_WITH_INTERPRETER = False
    try:
        yield
    finally:
        RUN_WITH_INTERPRETER = old_flag


# Assign attribute 'from_obj' to the qualified name 'target' on 'to_module
# This installs empty Modules where none exist yet if they are subpaths of target
def _assign_attr(
    from_obj: Any,
    to_module: torch.nn.Module,
    target: str,
    attr_kind: _AttrKind,
    persistent: bool = True,
):
    # [FUNCTION DOC] HELPER: Installs from_obj at the dotted path target under
    # to_module, creating empty intermediate modules as needed. The attribute is
    # assigned on every call-name variant of each intermediate module so all
    # calls share it. By kind: registers a parameter, registers a buffer (with the
    # persistent flag), sets a constant, or sets a submodule. Mutates modules.
    # Raises AssertionError if from_obj has the wrong type for the kind
    # (including a FakeScriptObject constant).
    *prefix, field = target.split(".")
    # We need to generate all submodules of `to_module` that are at `prefix` and
    # variants of `prefix` that differ only by call name. All of these submodules
    # will then be assigned `from_obj` at `field` so that they can share this attribute.
    # For example, if target is foo.bar.f, foo has another call name foo@1,
    # and bar has other call names bar@1, bar@2, then we will assign f to
    # foo.bar, foo.bar@1, foo.bar@2, foo@1.bar, foo@1.bar@1, foo@1.bar@2.
    to_modules = {to_module}
    for item in prefix:
        ts: set[torch.nn.Module] = set()
        for to_module in to_modules:
            if not hasattr(to_module, item):
                setattr(to_module, item, torch.nn.Module())
            # Collect `item` and every call-name variant `item@N` present in
            # _modules, regardless of contiguity. Export passes (e.g.
            # replace_set_grad_with_hop_pass) can relocate intermediate calls
            # into HOP subgraphs, leaving non-contiguous indices (e.g. base + @3
            # with @1/@2 absent); scanning contiguously from @1 would stop at the
            # gap and never populate the surviving higher-index copies.
            ts.update(
                t_call  # type: ignore[misc]
                for k, t_call in to_module._modules.items()
                if _is_call_name(k, item)
            )
        to_modules = ts

    for to_module in to_modules:
        if attr_kind == _AttrKind.PARAMETER:
            if not isinstance(from_obj, torch.nn.Parameter):
                raise AssertionError(
                    f"expected torch.nn.Parameter for PARAMETER attr_kind, got {type(from_obj)}"
                )
            to_module.register_parameter(field, from_obj)
        elif attr_kind == _AttrKind.BUFFER:
            if not isinstance(from_obj, torch.Tensor):
                raise AssertionError(
                    f"expected torch.Tensor for BUFFER attr_kind, got {type(from_obj)}"
                )
            to_module.register_buffer(field, from_obj, persistent=persistent)
        elif attr_kind == _AttrKind.CONSTANT:
            if isinstance(from_obj, FakeScriptObject):
                raise AssertionError(
                    "FakeScriptObject should only exist during tracing."
                )
            if not isinstance(
                from_obj,
                (
                    torch.Tensor,
                    torch.ScriptObject,
                ),
            ) and not is_custom_class_obj(from_obj):
                raise AssertionError(
                    f"expected torch.Tensor, torch.ScriptObject, or opaque type for CONSTANT attr_kind, got {type(from_obj)}"
                )
            setattr(to_module, field, from_obj)
        elif attr_kind == _AttrKind.MODULE:
            if not isinstance(from_obj, torch.nn.Module):
                raise AssertionError(
                    f"expected torch.nn.Module for MODULE attr_kind, got {type(from_obj)}"
                )
            setattr(to_module, field, from_obj)


class _SubmoduleBase:
    # [CLASS DOC] HELPER: Mixin for unflattened module types holding _ty, the
    # original eager module's type name (or None), exposed through type_name.
    _ty: str | None

    def type_name(self) -> str | None:
        # [FUNCTION DOC] Returns the recorded original module type name, or None if unknown.
        """
        Subclass of this class - InterpreterModule, InterpreterModuleDispatcher, represents
        corresponding model in eager model. To get this type information for those modules
        in eager model we need to use this method.
        """
        return self._ty


class InterpreterModule(_SubmoduleBase, torch.nn.Module):
    # [CLASS DOC] Module backed by an fx graph that executes by interpreting the
    # graph node by node (better stack traces) instead of generated code.
    # State: graph, _ty (eager type name), _run_with_interpreter (captured from
    # the global flag at construction), graph_module (generated-code version,
    # present only after finalize) and arg_names (placeholder names, set by
    # finalize). Must be finalized before use.
    """A module that uses torch.fx.Interpreter to execute instead of the usual
    codegen that GraphModule uses. This provides better stack trace information
    and makes it easier to debug execution.
    """

    graph_module: torch.fx.GraphModule | None

    def __init__(
        self,
        graph: torch.fx.Graph,
        ty: str | None = None,
    ):
        # [FUNCTION DOC] Stores the graph and eager type name, makes this module the
        # graph's owning module, and captures the current RUN_WITH_INTERPRETER flag.
        super().__init__()
        self.graph = graph
        self._ty = ty
        self.graph.owning_module = self  # type: ignore[assignment]
        self._run_with_interpreter = RUN_WITH_INTERPRETER

    def forward(self, *args, **kwargs):
        # [FUNCTION DOC] Runs the graph. Raises AssertionError if finalize was not
        # called (or, with keyword arguments, if their names do not exactly match
        # the remaining placeholder names). When dynamo is compiling or the
        # interpreter is disabled, runs the generated code of graph_module bound to
        # this module; otherwise maps keyword arguments onto positional ones by
        # placeholder name and runs the fx interpreter. Returns the graph output.
        # KNOWN ISSUE: the graph_module check is an attribute read that raises
        # AttributeError (not the intended AssertionError) when never finalized,
        # because only a class annotation exists.
        if self.graph_module is None:
            raise AssertionError("Didn't finalize this InterpreterModule")
        if not is_fx_symbolic_tracing() and (
            torch.compiler.is_dynamo_compiling() or not self._run_with_interpreter
        ):
            # Dynamo cannot trace through torch.fx.Interpreter, so fall back to
            # GraphModule codegen in this instance.
            # Patch the codegened forward to run with this InterpreterModule,
            # so attribute accesses, etc. are on this module instead.
            return type(self.graph_module).forward(self, *args, **kwargs)
        else:
            if kwargs:
                # Handle **kwargs. FX only natively supports positional
                # arguments (through placeholders). So in order to pass in
                # kwargs, we must correspond the names of the placeholders with
                # the keys in the kwarg dict.
                arg_list = list(args)
                kwarg_names = self.arg_names[len(arg_list) :]
                arg_list.extend(
                    kwargs[kwarg_name]
                    for kwarg_name in kwarg_names
                    if kwarg_name in kwargs
                )

                # Assert that the kwargs passed in exactly match the positional
                # arguments specified by the GraphModule. This should be
                # guaranteed by the unflattening process.
                if len(kwarg_names) != len(kwargs):
                    raise AssertionError(
                        f"kwarg_names length {len(kwarg_names)} does not match kwargs length {len(kwargs)}"
                    )
                if len(arg_list) != len(self.arg_names):
                    raise AssertionError(
                        f"arg_list length {len(arg_list)} does not match arg_names length {len(self.arg_names)}"
                    )
                args = tuple(arg_list)

            return torch.fx.Interpreter(self, graph=self.graph).run(
                *args, enable_io_processing=False
            )

    def finalize(self):
        # [FUNCTION DOC] Builds the generated-code GraphModule from the completed
        # graph (stored directly in the instance dict so it is not registered as a
        # submodule), lints the graph, and caches the placeholder names in
        # arg_names for keyword handling. Call once after the graph is fully built.
        # We need to "finalize" because GraphModule populates its own state_dict
        # based on the get_attrs observed in the graph. So we need to fully
        # construct the graph and call _sink_params before generating this
        # GraphModule.

        # need to set `graph_module` directly on the dict to avoid it getting
        # registered as a submodule.
        self.__dict__["graph_module"] = torch.fx.GraphModule(self, self.graph)
        self.graph.lint()

        # Cache arg names for kwarg handling (see forward())
        self.arg_names = []
        for node in self.graph.nodes:
            if node.op == "placeholder":
                self.arg_names.append(node.target)

    def print_readable(
        self,
        print_output=True,
        include_stride=False,
        include_device=False,
        colored=False,
    ):
        # [FUNCTION DOC] Pretty-prints (and returns) the readable form of this module
        # under the label InterpreterModule; flags control printing, strides, devices and color.
        return _print_readable(
            self,
            "InterpreterModule",
            print_output,
            include_stride,
            include_device,
            colored,
        )


class InterpreterModuleDispatcher(_SubmoduleBase, torch.nn.Module):
    # [CLASS DOC] Stand-in for a module called several times with different
    # per-call graphs. State: _call_modules (one InterpreterModule per call, in
    # call order), _num_calls (index of the next call, wraps around), shared
    # submodule table and selected attributes copied from the first call module.
    # Each forward runs the next call module.
    """
    A module that carries a sequence of InterpreterModules corresponding to
    a sequence of calls of that module. Each call to the module dispatches
    to the next InterpreterModule, and wraps back around after the last.
    """

    def __init__(self, attrs: set[str], call_modules: list[InterpreterModule]):
        # [FUNCTION DOC] Takes the attribute names to expose and the ordered call
        # modules. Shares the first call module's submodule table, copies the named
        # attributes and type name from it, and resets the call counter.
        # Raises AssertionError if call_modules is empty.
        super().__init__()
        if not call_modules:
            raise AssertionError("call_modules must not be empty")
        self._modules = call_modules[0]._modules
        for accessor in attrs:
            setattr(self, accessor, getattr(call_modules[0], accessor))
        self._ty = call_modules[0]._ty
        self._call_modules = call_modules
        self._num_calls = 0

    def forward(self, *args, **kwargs):
        # [FUNCTION DOC] Calls the current call module with the arguments, then
        # advances the counter modulo the number of call modules. If the call raises,
        # the counter is reset to zero and the error is re-raised. Returns the call's output.
        call_module = self._call_modules[self._num_calls]
        self._num_calls = (self._num_calls + 1) % len(self._call_modules)
        try:
            return call_module(*args, **kwargs)
        except Exception:
            self._num_calls = 0
            raise

    def call_modules(self):
        # [FUNCTION DOC] Returns the ordered list of per-call modules.
        return self._call_modules

    def print_readable(
        self,
        print_output=True,
        include_stride=False,
        include_device=False,
        colored=False,
    ):
        # [FUNCTION DOC] Returns the readable forms of all call modules joined by newlines.
        outputs = [
            mod.print_readable(
                print_output,
                include_stride,
                include_device,
                colored,
            )
            for mod in self._call_modules
        ]
        return "\n".join(outputs)


class FlatArgsAdapter(abc.ABC):
    # [CLASS DOC] Abstract interface for adapting flat input arguments whose tree
    # structure differs from the exported module's expected structure. Subclasses
    # implement adapt; get_flat_arg_paths optionally names argument locations.
    """
    Adapts input arguments with ``input_spec`` to align ``target_spec``.
    """

    @abc.abstractmethod
    def adapt(
        self,
        target_spec: pytree.TreeSpec,
        input_spec: pytree.TreeSpec,
        input_args: list[Any],
        metadata: dict[str, Any] | None = None,
        obj: Any | None = None,
    ) -> list[Any]:
        # [FUNCTION DOC] Abstract. Converts flat input_args laid out per input_spec into a
        # flat list laid out per target_spec, using optional metadata and the original
        # input object. Returns the adapted list; may mutate input_args.
        """NOTE: This adapter may mutate given ``input_args_with_path``."""
        ...

    def get_flat_arg_paths(self) -> list[str]:
        # [FUNCTION DOC] Returns names describing where each flat adapted argument came
        # from (used in constraint-check error paths); default is an empty list.
        """Returns a list of paths that are used to access the flat args."""
        return []


class UnflattenedModule(_SubmoduleBase, torch.nn.Module):
    # [CLASS DOC] Root module produced by unflattening an ExportedProgram: a module
    # tree mirroring the original eager hierarchy. State: graph and graph_signature
    # (copies), module_call_graph, flat_args_adapter, adapted flag, meta (shared with
    # the program, holds a back reference), ivals (intermediate-value mutation
    # tracker), range_constraints, input_placeholders, check_input_constraints, and
    # per-module parameters, buffers and constants. Calling it flattens the inputs,
    # runs the root graph and rebuilds the output structure.
    def __init__(
        self,
        export_module: ExportedProgram,
        flat_args_adapter: FlatArgsAdapter | None = None,
    ):
        # [FUNCTION DOC] Builds the module tree from the exported program. Steps: copy
        # graph, signature and call graph; convert functionalized buffer/input
        # mutations to in-place copies; fix inconsistent module stacks; outline graph
        # nodes into submodules; install parameters, buffers (persistent or not) and
        # constants as attributes with aliasing preserved by tensor identity (tensors
        # are cloned); sink them to the modules that use them; deduplicate repeated
        # module calls; replace multi-call modules with dispatchers; reorder
        # submodules by call-graph order; lint and finalize. Raises ValueError for
        # joint (backward) programs; AssertionError for malformed inputs. Mutates
        # export_module.graph_module.meta by adding a self reference.
        # KNOWN ISSUE: in the loop adding aliased constants, added_params_buffers adds
        # the stale loop variable s.target (leaked from the earlier loop) rather than the
        # constant name; harmless today because that set is only consulted for state-dict names.
        super().__init__()
        if export_module.graph_signature.backward_signature is not None:
            raise ValueError("Unflattening on JointExportModule NYI")

        def _id(obj):
            # [FUNCTION DOC] HELPER: Returns a tensor identity key (storage, stride, size,
            # offset) for tensors, or the Python object id for anything else.
            """Returns _TensorID dataclass for tensors, otherwise id()."""
            if isinstance(obj, torch.Tensor):
                return _TensorID(
                    untyped_storage=obj.untyped_storage(),
                    stride=obj.stride(),
                    size=obj.size(),
                    storage_offset=obj.storage_offset(),  # type: ignore[arg-type]
                )
            return id(obj)

        fqn_list = [entry.fqn for entry in export_module.module_call_graph]
        if fqn_list[0] != "":
            raise AssertionError(
                f"expected first fqn to be empty string, got {fqn_list[0]!r}"
            )
        export_graph = deepcopy(export_module.graph)
        self.graph_signature = deepcopy(export_module.graph_signature)
        self.graph = torch.fx.Graph()
        self.graph.owning_module = self  # type: ignore[assignment]
        self.module_call_graph = deepcopy(export_module.module_call_graph)
        self.flat_args_adapter = flat_args_adapter

        self.meta = export_module.graph_module.meta
        self.meta["unflattened_module"] = self

        # Flag to indicate whether args have been adapted.
        self.adapted = False
        self._run_with_interpreter = RUN_WITH_INTERPRETER

        _inplace_buffer_and_input_mutations(export_graph, self.graph_signature)
        _fix_nn_module_stacks(export_graph)
        self._ty = _root_module_type(export_graph)

        self.ivals = _IVals()
        # for any intermediate value of a mutation that is read, track the mutation
        seen_modules, seen_attrs = _outline_submodules(export_graph, self)
        # for each read intermediate value of a mutation, find where it was created,
        # and perform the mutation
        self.ivals.update(seen_modules.values())
        # move attributes that correspond to graph arguments for HOPs
        # from exported program to unflattened submodules
        _copy_graph_attrs(export_module._graph_module, self, seen_attrs)

        self.range_constraints = export_module.range_constraints
        self.equality_constraints: list = []

        # aliasing/unused param or buffer issues:
        # in strict-mode export, dynamo export will deduplicate aliased tensors,
        # and ignore unused tensors. For aliasing, this causes issues when some aliases
        # are unused, and we're unable to match the placeholder node to the correct FQN.
        # This leads to the graph signature potentially having the wrong target FQN,
        # and downstream issues where parameters are assigned to the wrong target attribute,
        # mismatching the relevant placeholder node in the unflattened module.
        # To resolve this we restore (_assign_attr) all aliased/unused tensors in
        # the state_dict as module attributes, but only keep the used tensors in the
        # graph's forward pass (_sink_params).
        state_dict = export_module.state_dict
        assigned_params: set[str] = set()  # tracking unused params
        id_to_param: dict[
            int | _TensorID, torch.nn.Parameter
        ] = {}  # handling weight-sharing
        for name in self.graph_signature.parameters:  # this loop adds used params
            param = state_dict[name]
            if _id(param) not in id_to_param:
                id_to_param[_id(param)] = torch.nn.Parameter(
                    param.clone(), requires_grad=param.requires_grad
                )

            _assign_attr(
                id_to_param[_id(param)],
                self,
                name,
                attr_kind=_AttrKind.PARAMETER,
            )
            assigned_params.add(name)

        non_persistent_buffers = set(self.graph_signature.non_persistent_buffers)
        assigned_buffers: set[str] = set()  # tracking unused buffers
        id_to_buffer: dict[int | _TensorID, tuple[torch.nn.Parameter, bool]] = {}
        for name in self.graph_signature.buffers:  # this loop adds used buffers
            if name in non_persistent_buffers:
                persistent = False
                buffer = export_module.constants[name]
            else:
                persistent = True
                buffer = state_dict[name]

            if _id(buffer) not in id_to_buffer:
                id_to_buffer[_id(buffer)] = (buffer.clone(), persistent)

            _assign_attr(
                id_to_buffer[_id(buffer)][0],
                self,
                name,
                attr_kind=_AttrKind.BUFFER,
                persistent=persistent,
            )
            assigned_buffers.add(name)

        # restore aliased/unused params and buffers
        # these appear in state dict but not graph signature
        for name, tensor in state_dict.items():
            if name in assigned_params or name in assigned_buffers:  # already assigned
                continue

            is_buffer = False
            if _id(tensor) in id_to_buffer or not isinstance(
                tensor, torch.nn.Parameter
            ):  # aliased buffer
                is_buffer = True

            if is_buffer:
                if (
                    _id(tensor) not in id_to_buffer
                ):  # this is completely unused (not weight-sharing)
                    id_to_buffer[_id(tensor)] = (
                        tensor,
                        True,
                    )  # assign to respect original model
                _assign_attr(
                    id_to_buffer[_id(tensor)][0],
                    self,
                    name,
                    attr_kind=_AttrKind.BUFFER,
                    persistent=True,
                )
            else:
                if _id(tensor) not in id_to_param:  # this is unused
                    id_to_param[_id(tensor)] = tensor
                _assign_attr(
                    id_to_param[_id(tensor)],
                    self,
                    name,
                    attr_kind=_AttrKind.PARAMETER,
                )

        # use id map so we don't double-clone aliased constants
        id_to_const: dict[int | _TensorID, torch.Tensor | torch._C.ScriptObject] = {}
        for fqn, constant in export_module.constants.items():
            if _id(constant) not in id_to_const:
                if isinstance(constant, torch.Tensor):
                    constant = constant.clone()
                id_to_const[_id(constant)] = constant
            _constant = id_to_const[_id(constant)]
            _assign_attr(
                _constant,
                self,
                fqn,
                attr_kind=_AttrKind.CONSTANT,
            )

        # This is to handle parameters/buffers that point to the same tensor
        # object id -> list of (node_name, target_name)
        consts_map: dict[int | _TensorID, list[tuple[str, str]]] = defaultdict(list)
        consts_targets: set[str] = set()

        def add_to_consts_map(obj_id, node_name, target_name):
            # [FUNCTION DOC] HELPER: Appends a (placeholder node name, target attribute
            # name) pair to the list of aliases recorded for the object identity key.
            name_list = consts_map[obj_id]
            name_list.append((node_name, target_name))

        # track aliased/unused params, buffers
        # prefer using untyped_storage() over id() when it's available
        added_params_buffers: set[str] = set()
        for s in self.graph_signature.input_specs:
            if s.kind == InputKind.PARAMETER or (
                s.kind == InputKind.BUFFER and s.persistent
            ):
                if not hasattr(s.arg, "name"):
                    raise AssertionError(
                        f"expected s.arg to have 'name' attribute, got {type(s.arg)}"
                    )
                if not isinstance(s.target, str):
                    raise AssertionError(
                        f"expected s.target to be str, got {type(s.target)}"
                    )
                add_to_consts_map(
                    _id(export_module.state_dict[s.target]),
                    s.arg.name,
                    s.target,
                )
                consts_targets.add(s.target)
                added_params_buffers.add(s.target)
            elif (
                s.kind == InputKind.BUFFER
                and not s.persistent
                or s.kind == InputKind.CONSTANT_TENSOR
                or s.kind == InputKind.CUSTOM_OBJ
            ):
                if not hasattr(s.arg, "name"):
                    raise AssertionError(
                        f"expected s.arg to have 'name' attribute for kind {s.kind}, got {type(s.arg)}"
                    )
                if not isinstance(s.target, str):
                    raise AssertionError(
                        f"expected s.target to be str for kind {s.kind}, got {type(s.target)}"
                    )
                add_to_consts_map(
                    _id(export_module.constants[s.target]),
                    s.arg.name,
                    s.target,
                )
                consts_targets.add(s.target)

        # add constants that are aliased and don't appear in graph signature
        for const_name, const in export_module.constants.items():
            if const_name not in consts_targets:
                const_id = _id(const)
                if const_id not in consts_map:
                    raise AssertionError(
                        f"constant {const_name!r} id not found in consts_map"
                    )
                ph_name, _ = consts_map[const_id][0]
                add_to_consts_map(const_id, ph_name, const_name)
                added_params_buffers.add(s.target)

        # add aliased/unused params and buffers that don't appear in graph signature
        for fqn, tensor in export_module.state_dict.items():
            if fqn not in added_params_buffers:
                tensor_id = _id(tensor)
                if tensor_id not in consts_map:
                    # completely unused (no weight-sharing), ignore.
                    # this weight doesn't appear in graph module,
                    # so won't cause FQN assignment issues
                    continue
                ph_name, _ = consts_map[tensor_id][0]
                add_to_consts_map(tensor_id, ph_name, fqn)

        # node name -> list of possible targets
        inputs_to_state: dict[str, list[str]] = {}
        for node_target in consts_map.values():
            targets = [t[1] for t in node_target]
            for n, _ in node_target:
                inputs_to_state[n] = targets

        _sink_params(self, inputs_to_state, [])

        redirected_call_indices = _deduplicate_modules(seen_modules.values())
        fqn_list = [fqn for fqn in fqn_list if fqn not in redirected_call_indices]

        self._dispatch_modules(redirected_call_indices, consts_targets)
        fqn_list = [fqn for fqn in fqn_list if "@" not in fqn]

        # Cache so we don't have to compute this every time.
        # NOTE: this needs to be kept in sync with the placeholders in
        # self.graph, but currently we have no way to guarantee that.
        self.input_placeholders = [
            node for node in self.graph.nodes if node.op == "placeholder"
        ]
        self.check_input_constraints = True
        # TODO(zhxchen17) We can register modules ahead of time instead of reorder later.
        fqn_order = {fqn: i for i, fqn in enumerate(fqn_list)}
        # In the case of legacy IR, we might be missing some modules from metadata.
        for name, _ in self.named_modules(remove_duplicate=False):
            if name not in fqn_order:
                fqn_order[name] = len(fqn_order)
        _reorder_submodules(self, fqn_order)
        self.graph.lint()
        self.finalize()

    def _print_graph(self):
        # [FUNCTION DOC] HELPER: Debug aid that prints, for every named submodule that
        # has an fx graph, the module name and its graph to standard output.
        for fqn, mod in self.named_modules():
            print(fqn + ":")
            if hasattr(mod, "graph") and isinstance(mod.graph, torch.fx.Graph):
                print(mod.graph)

    def _adapt_flat_args(self, flat_args, in_spec, input):
        # [FUNCTION DOC] HELPER: Returns flat_args unchanged when the input tree spec equals
        # the exported signature's input spec. Otherwise runs the flat args adapter
        # (target spec, input spec, args, metadata, original input) and checks the result
        # length. Raises TypeError if no adapter is configured or the adapted count mismatches.
        signature = self.module_call_graph[0].signature
        if in_spec == signature.in_spec:
            return flat_args

        if self.flat_args_adapter is None:
            raise TypeError(
                "There is no flat args adapter specified. "
                "Are you sure you are calling this with the right arguments? "
            )
        else:
            flat_args = self.flat_args_adapter.adapt(
                target_spec=signature.in_spec,
                input_spec=in_spec,
                input_args=flat_args,
                metadata=self.meta,
                obj=input,
            )

            if len(flat_args) != signature.in_spec.num_leaves:
                raise TypeError(
                    f"Flat args adaption failed, number of args mismatch "
                    f"Adapted: {len(flat_args)} \n"
                    f"Exported module: {signature.in_spec.num_leaves}"
                )
            return flat_args

    def process_forward_inputs(self, *args, **kwargs):
        # [FUNCTION DOC] Turns call arguments into the flat list the graph expects:
        # reorders keyword arguments to the signature, flattens (args, kwargs) with
        # paths, and, if the tree spec differs, prints a notice (first time only) and
        # adapts via the adapter, setting the adapted flag. During fx symbolic tracing
        # returns the flat args directly. When enabled, checks input constraints against
        # range constraints. Returns the flat args. Raises TypeError from adaptation or
        # constraint violations from the check; side effect: prints to standard output.
        signature = self.module_call_graph[0].signature

        reordered_kwargs = kwargs
        if kwargs:
            reordered_kwargs = reorder_kwargs(kwargs, signature.in_spec)

        flat_args_with_path, in_spec = pytree.tree_flatten_with_path(
            (args, reordered_kwargs)
        )
        flat_args = [x[1] for x in flat_args_with_path]

        if is_fx_symbolic_tracing():
            return flat_args

        if in_spec != signature.in_spec:
            if not self.adapted:
                print(
                    "Input treespec does not match with exported module's: \n"
                    f"Input treespec: {in_spec}. ",
                    f"Exported module treespec: {signature.in_spec}",
                )
                print("Adapting flat arg to match exported module's treespec")
            flat_args = self._adapt_flat_args(flat_args, in_spec, args)
            self.adapted = True

        if self.check_input_constraints:
            # Import here to avoid an unfortunate circular dependency.
            # TODO(suo): untangle this.
            from torch._export.utils import _check_input_constraints_for_graph

            if self.adapted is True:
                flat_arg_paths = (
                    self.flat_args_adapter.get_flat_arg_paths()
                    if self.flat_args_adapter
                    else []
                )
                if flat_arg_paths and len(flat_arg_paths) != len(flat_args):
                    raise AssertionError(
                        f"flat_arg_paths length {len(flat_arg_paths)} does not match flat_args length {len(flat_args)}"
                    )
                new_flat_args_with_path = [  # type: ignore[var-annotated]
                    (
                        (
                            SequenceKey(idx=idx),
                            GetAttrKey(
                                name=flat_arg_paths[idx]
                                if flat_arg_paths
                                else "<unknown location>"
                            ),
                        ),
                        arg,
                    )
                    for idx, arg in enumerate(flat_args)
                ]
            else:
                new_flat_args_with_path = flat_args_with_path  # type: ignore[assignment]

            _check_input_constraints_for_graph(
                self.input_placeholders, new_flat_args_with_path, self.range_constraints
            )

        return flat_args

    def forward(self, *args, **kwargs):
        # [FUNCTION DOC] Runs the unflattened program: processes inputs to a flat list,
        # runs the root graph (interpreter, or generated code when dynamo-compiling or the
        # interpreter is disabled), and rebuilds the output structure from the signature's
        # output spec. Under fx symbolic tracing returns the raw result, unwrapping a
        # one-element tuple. Returns the structured output.
        flat_args = self.process_forward_inputs(*args, **kwargs)
        signature = self.module_call_graph[0].signature

        if is_fx_symbolic_tracing():
            return_val = torch.fx.Interpreter(self, graph=self.graph).run(
                *flat_args, enable_io_processing=False
            )
            # For scalar return value, fx.Graph wraps in a tuple
            if isinstance(return_val, tuple) and len(return_val) == 1:
                return return_val[0]
            return return_val

        if torch.compiler.is_dynamo_compiling() or not self._run_with_interpreter:
            tree_out = type(self.graph_module).forward(self, *flat_args)  # type: ignore[union-attr]
        else:
            tree_out = torch.fx.Interpreter(self, graph=self.graph).run(
                *flat_args, enable_io_processing=False
            )
        return pytree.tree_unflatten(tree_out, signature.out_spec)

    def finalize(self):
        # [FUNCTION DOC] Builds the generated-code GraphModule for the root graph (kept out
        # of the registered submodules) and lints the graph. Call after the graph is complete.
        self.__dict__["graph_module"] = torch.fx.GraphModule(self, self.graph)
        self.graph.lint()

    def _dispatch_modules(self, redirected_call_indices, consts_targets):
        # [FUNCTION DOC] HELPER: For modules whose call signatures are preserved and that are
        # called more than once, removes the per-call submodules (names with an @ suffix) and
        # installs one InterpreterModuleDispatcher at the base name, exposing the constant
        # attribute names found for it. Then rewrites call_module targets in all graphs to drop
        # the call index. Inputs: mapping of redirected call names to kept names, and the set
        # of constant target names. Mutates the module tree and graphs.
        """For a module whose call signatures are preserved, replace
        multiple modules corresponding to multiple calls to that module
        with a single dispatcher module that tracks which module to call.
        """

        # for each fqn whose module call signature is preserved,
        # map that fqn to a list of called modules
        called_modules = defaultdict(list)
        for entry in self.module_call_graph:
            if entry.fqn and entry.signature:
                # some modules were removed and their fqns redirected to other
                # fqns during deduplication
                fqn = entry.fqn
                mod = _get_attr(self, redirected_call_indices.get(fqn, fqn))
                base, idx = fqn.split("@") if "@" in fqn else [fqn, "0"]
                called_modules[base].append((int(idx), mod))

        attrs_map = defaultdict(set)
        for target in consts_targets:
            if "." in target:
                orig_fqn, name = target.rsplit(".", 1)
                attrs_map[orig_fqn].add(name)
            else:
                attrs_map[""].add(target)

        # replace multiple call modules with a single dispatcher module
        for orig_fqn, indexed_call_modules in called_modules.items():
            call_modules = [mod for _, mod in sorted(indexed_call_modules)]
            if len(call_modules) > 1:
                for i in range(len(call_modules)):
                    fqn = _call_name(orig_fqn, i + 1)
                    if fqn not in redirected_call_indices:
                        *prefix, name = fqn.split(".")
                        _get_attr_via_attr_list(self, prefix)._modules.pop(name)
                self.set_submodule(
                    orig_fqn,
                    InterpreterModuleDispatcher(attrs_map[orig_fqn], call_modules),
                )

        # elide call indices in call modules because they are
        # tracked automatically inside the dispatcher module
        def elide_call_indices(prefix, graph):
            # [FUNCTION DOC] HELPER: In the graph, strips the call index suffix from each
            # call_module target whose full path (prefix plus base name) is a dispatched module.
            for node in graph.nodes:
                if node.op == "call_module":
                    fqn = node.target.split("@")[0]
                    path = f"{prefix}.{fqn}" if prefix else fqn
                    if path in called_modules:
                        node.target = fqn

        for fqn, mod in self.named_modules(remove_duplicate=False):
            if hasattr(mod, "graph"):
                elide_call_indices(fqn, mod.graph)
            elif hasattr(mod, "_call_modules"):
                for mod_ in mod._call_modules:
                    if not hasattr(mod_, "graph"):
                        raise AssertionError(
                            f"expected mod_ to have 'graph' attribute, got {type(mod_)}"
                        )
                    elide_call_indices(fqn, mod_.graph)

    def print_readable(
        self,
        print_output=True,
        include_stride=False,
        include_device=False,
        colored=False,
    ):
        # [FUNCTION DOC] Pretty-prints (and returns) the readable form under the label
        # UnflattenedModule; flags control printing, strides, devices and color.
        return _print_readable(
            self,
            "UnflattenedModule",
            print_output,
            include_stride,
            include_device,
            colored,
        )


def unflatten(
    module: ExportedProgram, flat_args_adapter: FlatArgsAdapter | None = None
) -> UnflattenedModule:
    # [FUNCTION DOC] Public entry point. Removes effect tokens from the exported program,
    # builds an UnflattenedModule with the optional input adapter, and wraps its input
    # processing method so dynamo does not trace into it. Returns the unflattened module
    # (module hierarchy matching the eager module). Errors from construction propagate.
    """Unflatten an ExportedProgram, producing a module with the same module
    hierarchy as the original eager module. This can be useful if you are trying
    to use :mod:`torch.export` with another system that expects a module
    hierarchy instead of the flat graph that :mod:`torch.export` usually produces.

    .. note:: The args/kwargs of unflattened modules will not necessarily match
        the eager module, so doing a module swap (e.g. :code:`self.submod =
        new_mod`) will not necessarily work. If you need to swap a module out, you
        need to set the :code:`preserve_module_call_signature` parameter of
        :func:`torch.export.export`.

    Args:
        module (ExportedProgram): The ExportedProgram to unflatten.
        flat_args_adapter (Optional[FlatArgsAdapter]): Adapt flat args if input TreeSpec does not match with exported module's.

    Returns:
        An instance of :class:`UnflattenedModule`, which has the same module
        hierarchy as the original eager module pre-export.
    """
    module = _remove_effect_tokens(module)
    m = UnflattenedModule(module, flat_args_adapter)

    # Disable process_forward_inputs as the adapter has many
    # non-dynamo-traceable behavior.
    m.process_forward_inputs = torch._dynamo.disable(  # type: ignore[method-assign]
        m.process_forward_inputs,
        reason="do not trace into preprocessing the inputs",
        recursive=True,
    )

    return m


def _inplace_buffer_and_input_mutations(
    graph: torch.fx.Graph,
    graph_signature: ExportGraphSignature,
) -> None:
    # [FUNCTION DOC] HELPER: Rewrites the graph in place so each mutated buffer or user
    # input, which functionalization returns as leading outputs, is instead updated by an
    # in-place copy node inserted after the mutation value; uses of the value are redirected
    # and the mutation outputs are dropped from the output node. Metadata is copied to the
    # new nodes. Raises AssertionError if the last node is not a single-argument output node.
    """Transform buffer and input mutations from their functionalized form
    into copy_ nodes in the graph.

    Functionalization represents a buffer mutation by passing the buffer as
    an input and output. For example, consider the eager code:
        def forward(self, x):
            self.buffer += x
            return x * x

    This corresponds to a graph that looks like:
        def forward(self, buffer, x):
            mutated_buffer = aten.add(buffer, x)
            mul = aten.mul(x, x)
            return (mutated_buffer, mul)

    We want to inplace this into something that looks like the original
    eager code:
        def forward(self, buffer, x):
            mutated_buffer = aten.add(buffer, x)
            buffer.copy_(mutated_buffer)
            mul = aten.mul(x, x)
            return (mul,)

    Input mutations are handled similarly.
    """
    output_node = next(iter(reversed(graph.nodes)))
    if output_node.op != "output" or len(output_node.args) != 1:
        raise AssertionError(
            f"expected output node with op='output' and 1 arg, got op={output_node.op!r} with {len(output_node.args)} args"
        )
    return_args = output_node.args[0]

    input_name_to_node = {
        node.name: node for node in graph.nodes if node.op == "placeholder"
    }
    mutation_name_to_input_name = {}

    # Collect mutated buffers.
    buffer_fqn_to_input_name = {
        buffer_fqn: k for k, buffer_fqn in graph_signature.inputs_to_buffers.items()
    }
    mutation_name_to_input_name = {
        k: buffer_fqn_to_input_name[buffer_fqn]
        for k, buffer_fqn in graph_signature.buffers_to_mutate.items()
    }
    # Collect mutated user inputs.
    mutation_name_to_input_name.update(graph_signature.user_inputs_to_mutate)

    num_mutations = len(mutation_name_to_input_name)

    for mutation in return_args[:num_mutations]:
        input_name = mutation_name_to_input_name[mutation.name]
        input_node = input_name_to_node[input_name]

        with graph.inserting_after(mutation):
            # Create a copy_ node that inplaces the mutation.
            new_node = graph.create_node(
                "call_function", torch.ops.aten.copy_.default, (input_node, mutation)
            )
            for k, v in mutation.meta.items():
                new_node.meta[k] = v
        # Replace all uses of the previously functional mutation with
        # our copy_ node.
        mutation.replace_all_uses_with(new_node, lambda x: x is not new_node)

    # Remove the mutated buffer / input from the graph outputs, since we don't
    # need to thread it through anymore.
    user_outputs = tuple(return_args[num_mutations:])
    output_node.args = ((user_outputs),)


def _root_module_type(graph: torch.fx.Graph) -> str | None:
    # [FUNCTION DOC] HELPER: Returns the type name recorded for the root module (empty path)
    # in the first node whose module stack contains it, or None if none does.
    for node in graph.nodes:
        if "nn_module_stack" not in node.meta:
            continue

        for path, ty in node.meta["nn_module_stack"].values():
            if not path:
                return ty
    return None


def _fix_nn_module_stacks(graph):
    # [FUNCTION DOC] HELPER: For every node with a module stack, strips call suffixes from
    # the names and keeps only the leading entries where each name is a strict prefix of the
    # next; the remaining entries are dropped with a logged warning. Mutates node metadata.
    # For each nn module stack in the graph, check if the fqns in it represent a stack:
    # 1. Each fqn must be a prefix of the next fqn.
    # 2. If not, remove the entries starting from the next fqn, emitting a warning.
    for node in graph.nodes:
        if "nn_module_stack" not in node.meta:
            continue

        nn_module_stack = node.meta["nn_module_stack"]
        fqns = [
            fqn.split("@")[0] if "@" in fqn else fqn
            for fqn, _t in nn_module_stack.values()
        ]

        # Check if each FQN is a prefix of the next one
        prev_fqn, *next_fqns = fqns
        num_valid_indices = 1  # root FQN
        for curr_fqn in next_fqns:
            # Check if the previous FQN is a prefix of the current one
            if _is_prefix(prev_fqn, curr_fqn):
                num_valid_indices += 1
                prev_fqn = curr_fqn
            else:
                # Found a non-prefix FQN, stop here
                break

        # If we need to remove entries, create a new stack with only valid entries
        if num_valid_indices < len(nn_module_stack):
            log.warning(
                "nn_module_stack fqns %s at node %s do not form a stack! dropping last %d entries",
                fqns,
                node,
                len(nn_module_stack) - num_valid_indices,
            )
            node.meta["nn_module_stack"] = dict(
                list(nn_module_stack.items())[:num_valid_indices]
            )


def _is_prefix(candidate, target):
    # [FUNCTION DOC] HELPER: True if candidate is a strictly shorter string-prefix of target.
    # KNOWN ISSUE: plain string prefix, so "a.b" counts as a prefix of "a.bc".
    """Check whether `candidate` is a prefix of `target`."""
    return len(candidate) < len(target) and target[: len(candidate)] == candidate


def _compute_accessor(parent_fqn: str, child_fqn: str) -> str:
    # [FUNCTION DOC] HELPER: Returns the dotted attribute path of the child relative to the
    # parent (the child name unchanged for the root parent). Raises RuntimeError if the child
    # is not a descendant of the parent.
    if parent_fqn == "":
        # Handle the root module correctly.
        return child_fqn

    parent_split = parent_fqn.split(".")
    child_split = child_fqn.split(".")

    # TODO: support skip connection by inlining the child module.
    if child_split[: len(parent_split)] != parent_split:
        raise RuntimeError(
            f"Child module '{child_fqn}' is not a descendant of parent module '{parent_fqn}'."
            "This is currently unsupported."
            "Please try to make child module attach to parent module directly."
        )
    return ".".join(child_split[len(parent_split) :])


def _check_graph_equivalence(x: torch.nn.Module, y: torch.nn.Module):
    # [FUNCTION DOC] HELPER: Returns True if the two modules' fx graphs have identical
    # textual dumps (op, target for calls/attributes, and arguments with nodes numbered by
    # position). Raises AssertionError if either graph is not an fx Graph.
    def graph_dump(graph: torch.fx.Graph) -> str:
        # [FUNCTION DOC] HELPER: Renders the graph as one line per node, with node arguments
        # shown as position references so names do not matter.
        ret = []
        nodes_idx: dict[int, int] = {}

        def arg_dump(arg) -> str:
            # [FUNCTION DOC] HELPER: Renders a node as its position reference, other values via text conversion.
            if isinstance(arg, torch.fx.Node):
                return "%" + str(nodes_idx[id(arg)])
            return str(arg)

        for i, node in enumerate(graph.nodes):
            args_dump = [str(arg) for arg in pytree.tree_map(arg_dump, node.args)]
            args_dump += [
                f"{key}={value}"
                for key, value in pytree.tree_map(arg_dump, node.kwargs).items()
            ]
            target = node.target if node.op in ("call_function", "get_attr") else ""
            # pyrefly: ignore [bad-argument-type]
            ret.append(f"{i}: {node.op}[{target}]({', '.join(args_dump)})")
            nodes_idx[id(node)] = i
        return "\n".join(ret)

    if not isinstance(x.graph, torch.fx.Graph):
        raise AssertionError(
            f"expected x.graph to be torch.fx.Graph, got {type(x.graph)}"
        )
    if not isinstance(y.graph, torch.fx.Graph):
        raise AssertionError(
            f"expected y.graph to be torch.fx.Graph, got {type(y.graph)}"
        )
    return graph_dump(x.graph) == graph_dump(y.graph)


def _add_spec(gm: torch.nn.Module, spec) -> str:
    # [FUNCTION DOC] HELPER: Stores the tree spec on the module under the first unused name
    # of the form _spec_N and returns that name. Mutates the module.
    i = 0
    while hasattr(gm, f"_spec_{i}"):
        i += 1
    name = f"_spec_{i}"
    setattr(gm, name, spec)
    return name


def _generate_flatten(gm: torch.fx.GraphModule, node) -> torch.fx.Node:
    # [FUNCTION DOC] HELPER: Adds graph nodes that flatten the node's value with the pytree
    # flatten function and select the leaf list; returns that selecting node.
    flatten = gm.graph.call_function(pytree.tree_flatten, (node,))
    getitem_0 = gm.graph.call_function(operator.getitem, (flatten, 0))
    return getitem_0


def _generate_flatten_spec(
    gm: torch.fx.GraphModule | InterpreterModule | UnflattenedModule, node, spec
) -> torch.fx.Node:
    # [FUNCTION DOC] HELPER: Stores the spec on the module, then adds graph nodes that
    # flatten the given node according to that spec; returns the flatten node.
    name = _add_spec(gm, spec)
    spec_node = gm.graph.get_attr(name)
    return gm.graph.call_function(fx_pytree.tree_flatten_spec, (node, spec_node))


def _generate_unflatten(
    gm: torch.fx.GraphModule | InterpreterModule | UnflattenedModule, nodes, spec
) -> torch.fx.Node:
    # [FUNCTION DOC] HELPER: Stores the spec on the module, then adds a graph node that
    # rebuilds a structure from the leaf nodes according to that spec; returns that node.
    name = _add_spec(gm, spec)
    spec_node = gm.graph.get_attr(name)
    return gm.graph.call_function(pytree.tree_unflatten, (nodes, spec_node))


def _get_submodule(mod: torch.nn.Module, target: str):
    # [FUNCTION DOC] HELPER: Walks the dotted path from the module and returns the final
    # attribute (any type), or None if an intermediate attribute is missing or is not a module.
    *prefix, field = target.split(".")

    for item in prefix:
        submod = getattr(mod, item, None)

        if submod is None:
            return None

        if not isinstance(submod, torch.nn.Module):
            return None

        mod = submod

    return getattr(mod, field, None)


def _add_submodule(
    mod: torch.nn.Module,
    target: str,
    module_to_add: torch.nn.Module,
    create_module: Callable[[str], torch.nn.Module] | None = None,
):
    # [FUNCTION DOC] HELPER: Adds module_to_add at the dotted path under mod, creating missing
    # intermediate modules with the create_module factory (given the partial path) or empty
    # modules. Returns False if an existing intermediate attribute is not a module, else None.
    # Mutates the module tree.
    *prefix, field = target.split(".")

    for i, item in enumerate(prefix):
        submod = getattr(mod, item, None)

        if submod is None:
            if create_module is not None:
                submod = create_module(".".join(prefix[: i + 1]))
            else:
                submod = torch.nn.Module()
            setattr(mod, item, submod)

        if not isinstance(submod, torch.nn.Module):
            return False

        mod = submod

    mod.add_module(field, module_to_add)


def _call_name(base: str, n: int) -> str:
    # [FUNCTION DOC] HELPER: Returns the name of the n-th call (1-based) of the module: the
    # base for the first call, else the base with an @ and the zero-based repeat index.
    # Given n >= 0, generate call names to a submodule `base` of the form
    # `base`, `base@1`, `base@2`, etc.
    return base if n == 1 else f"{base}@{n - 1}"


def _is_call_name(call_name: str, base: str) -> bool:
    # [FUNCTION DOC] HELPER: True if call_name is the base name or the base followed by an
    # @ and digits (any call variant of that module).
    # Recognize when call_name = _call_name(base, n) for some n >= 0.
    return re.match(re.escape(base) + r"(@\d+)?$", call_name) is not None


class _ModuleFrame:
    # [CLASS DOC] HELPER: Working state for outlining one call of one module from the
    # flat graph. Holds the flat graph, the nodes belonging to this frame, shared
    # bookkeeping (seen nodes, seen module entries, seen attrs, created modules), the parent
    # frame, the module stack and call signature info, the target module and its graph,
    # a node map from flat-graph nodes to this graph, and placeholders. Its constructor
    # attaches the module to the parent and wires call arguments; run_outer and
    # its helpers copy nodes, create placeholders and outputs, and recurse into children.
    def __init__(
        self,
        flat_graph: torch.fx.Graph,
        nodes: tuple[torch.fx.Node, ...],
        seen_nodes,
        seen_modules,
        seen_attrs,
        created_modules,
        parent,
        module_stack: list[tuple[str, str | None, int]],
        module_id,
        module_call_graph: dict[str, ModuleCallSignature],
        module: torch.fx.GraphModule | UnflattenedModule | None = None,
    ):
        # [FUNCTION DOC] Sets up the frame for the last entry of the module stack: derives the
        # call name, picks or creates the module (an existing created module or a new
        # InterpreterModule), and, for non-root frames, adds it to the parent module, emits a
        # call_module node in the parent graph, shares submodules with earlier calls of the
        # same module, and records a submodule entry. If the module's call signature was
        # preserved, builds input placeholders and flatten nodes in this graph and unflatten
        # nodes in the parent so arguments match the original call. Raises AssertionError on
        # unexpected signature structure.
        self.flat_graph = flat_graph
        self.nodes = nodes
        self.seen_nodes = seen_nodes
        self.seen_modules = seen_modules
        self.seen_attrs = seen_attrs
        self.created_modules = created_modules
        self.parent = parent
        self.module_stack = module_stack
        self.module_id = module_id

        self.module_call_graph = module_call_graph
        self.verbose = False

        self.fqn, ty, num_calls = self.module_stack[-1]
        # generate call name for self.fqn
        self.child_fqn = _call_name(self.fqn, num_calls + 1)

        self.module: torch.fx.GraphModule | UnflattenedModule | InterpreterModule
        if module is not None:
            self.module = module
            self.ivals = module.ivals if hasattr(module, "ivals") else {}  # type: ignore[var-annotated]
        else:
            self.module = self.created_modules.get(
                self.fqn,
                InterpreterModule(torch.fx.Graph(), ty=ty),
            )
            self.ivals = parent.ivals

        self.graph = self.module.graph

        # Mapping of nodes in the flat graph to nodes in this graph.
        self.node_map: dict[torch.fx.Node, torch.fx.Node] = {}
        self.node_to_placeholder = {}

        self.parent_call_module: torch.fx.Node | None = None
        if parent is not None:
            accessor = _compute_accessor(parent.fqn, self.child_fqn)

            def create_module(fqn):
                # [FUNCTION DOC] HELPER: Returns the intermediate module for the given partial
                # name under the parent, reusing one already created or creating and recording
                # a new empty InterpreterModule of the current type.
                path = f"{parent.fqn}.{fqn}" if parent.fqn else fqn
                if path in self.created_modules:
                    return self.created_modules[path]
                submod = InterpreterModule(torch.fx.Graph(), ty=ty)
                self.created_modules[path] = submod
                return submod

            _add_submodule(parent.module, accessor, self.module, create_module)
            self.parent_call_module = parent.graph.call_module(accessor)
            if self.seen_modules[self.module_id]:
                base_module_frame = self.seen_modules[self.module_id][0]
                self.module._modules = base_module_frame.module._modules
            self.seen_modules[self.module_id].append(
                _SubmoduleEntry(
                    parent_fqn=self.parent.fqn,
                    parent_module=self.parent.module,
                    parent_call_module=self.parent_call_module,
                    fqn=self.fqn,
                    call_idx=num_calls + 1,
                    module=self.module,
                )
            )

        signature = module_call_graph.get(self.child_fqn)
        if signature is not None and self.parent is not None:
            if signature.in_spec.num_children != 2:
                raise AssertionError(
                    f"expected in_spec to have 2 children, got {signature.in_spec.num_children}"
                )
            if signature.in_spec.type is not tuple:
                raise AssertionError(
                    f"expected in_spec.type to be tuple, got {signature.in_spec.type}"
                )
            args_spec, kwargs_spec = signature.in_spec.children()
            if args_spec.type is not tuple:
                raise AssertionError(
                    f"expected args_spec.type to be tuple, got {args_spec.type}"
                )
            if kwargs_spec.type is not dict:
                raise AssertionError(
                    f"expected kwargs_spec.type to be dict, got {kwargs_spec.type}"
                )

            with self.graph.inserting_after(None):
                arg_nodes = [
                    self.graph.placeholder(f"_positional_arg_{idx}")
                    for idx in range(args_spec.num_children)
                ]
                kwarg_nodes = {}
                for name in kwargs_spec.context:
                    kwarg_nodes[name] = self.graph.placeholder(name)
                flat_args = _generate_flatten_spec(
                    self.module,
                    (tuple(arg_nodes), kwarg_nodes),
                    signature.in_spec,
                )
                for idx, arg in enumerate(signature.inputs):
                    flat_arg_node = self.graph.create_node(
                        op="call_function",
                        target=operator.getitem,
                        args=(flat_args, idx),
                        name=(
                            arg.name
                            if not isinstance(arg, ConstantArgument)
                            else f"_constant_{idx}"
                        ),
                    )
                    if isinstance(arg, ConstantArgument):
                        continue

                    if arg.name in self.seen_nodes:
                        flat_arg_node.meta = copy.copy(self.seen_nodes[arg.name].meta)
                        self.node_to_placeholder[self.seen_nodes[arg.name]] = (
                            flat_arg_node
                        )

            with self.parent.graph.inserting_before(self.parent_call_module):
                input_nodes: list[torch.fx.Node | None] = []
                for input in signature.inputs:
                    if isinstance(input, ConstantArgument):
                        input_nodes.append(input.value)  # type: ignore[arg-type]
                    elif input.name not in self.seen_nodes:
                        input_nodes.append(None)
                    else:
                        if not isinstance(
                            input,
                            (
                                TensorArgument,
                                SymIntArgument,
                                SymBoolArgument,
                                SymFloatArgument,
                            ),
                        ):
                            raise AssertionError(
                                f"expected input to be TensorArgument, SymIntArgument, "
                                f"SymBoolArgument, or SymFloatArgument, got {type(input)}"
                            )
                        input_nodes.append(
                            self.parent.remap_input(self.seen_nodes[input.name])
                        )

                inputs_node = _generate_unflatten(
                    self.parent.module,
                    input_nodes,
                    signature.in_spec,
                )

                args_node = self.parent.graph.call_function(
                    operator.getitem, (inputs_node, 0)
                )
                kwargs_node = self.parent.graph.call_function(
                    operator.getitem, (inputs_node, 1)
                )
                arg_nodes = [
                    self.parent.graph.call_function(operator.getitem, (args_node, i))
                    for i in range(args_spec.num_children)
                ]
                kwarg_nodes = {
                    k: self.parent.graph.call_function(
                        operator.getitem, (kwargs_node, k)
                    )
                    for k in kwargs_spec.context
                }
            if self.parent_call_module is None:
                raise AssertionError("parent_call_module must not be None")
            # pyrefly: ignore [bad-assignment]
            self.parent_call_module.args = tuple(arg_nodes)
            self.parent_call_module.kwargs = kwarg_nodes  # type: ignore[assignment]

    def add_placeholder(self, x):
        # [FUNCTION DOC] Adds a placeholder to this frame's graph for flat-graph node x (copying
        # name, type and metadata) and records it in the node-to-placeholder map; returns nothing.
        # Raises AssertionError for the root frame or if x is not from the flat graph.
        if self.fqn == "":
            raise AssertionError(f"Cannot add placeholder {x} to root module")
        if x.graph is not self.flat_graph:
            raise AssertionError(
                "expected x.graph to be flat_graph, got different graph"
            )
        # x is not in subgraph, create a new placeholder for subgraph
        with self.graph.inserting_before(None):
            placeholder_node = self.graph.placeholder(x.name, type_expr=x.type)
        # copy all meta fields, even if some fields might be irrelevant for
        # the placeholder node
        placeholder_node.meta = copy.copy(x.meta)
        self.node_to_placeholder[x] = placeholder_node

    def copy_sym_call_function(self, x):
        # [FUNCTION DOC] Re-creates the call_function node x (symbolic-value computations that
        # export deduplicated) in this frame's graph, remapping its node arguments, copying
        # metadata and recording it in the node map. Returns the new node.
        # This only exists because we deduplicate sym_size nodes in the flat export graph,
        # and if preserve_module_call_signature is set, we may not be able to pass sym_size
        # nodes, or their downstream users, as inputs to submodule calls.
        # To avoid this we copy these call_function nodes with sym_type results.
        # This should however only be done for sym_type nodes - call_function nodes on tensors
        # should not be deduplicated in the first place.
        args = pytree.tree_map_only(torch.fx.Node, self.remap_input, x.args)
        kwargs = pytree.tree_map_only(torch.fx.Node, self.remap_input, x.kwargs)
        node = self.graph.call_function(x.target, args, kwargs)
        node.meta = copy.copy(x.meta)
        self.node_map[x] = node
        return node

    def remap_input(self, x):
        # [FUNCTION DOC] Maps a flat-graph node used as an input to the corresponding node in
        # this frame's graph. Order of resolution: already copied node; existing placeholder;
        # a new placeholder (for graph inputs, or when call signatures are not preserved), also
        # threaded through the parent call node; a copy of a sym-size-like or operator call
        # function; or, for an intermediate value of a mutation, a read recorded in the ivals
        # tracker. Raises AssertionError if x is not from the flat graph, RuntimeError for an
        # unsupported node kind.
        if x.graph is not self.flat_graph:
            raise AssertionError(
                "expected x.graph to be flat_graph, got different graph"
            )
        if x in self.node_map:
            return self.node_map[x]
        self.print(f"remap_input({x})")
        if x in self.node_to_placeholder:
            return self.node_to_placeholder[x]
        elif (
            x.op == "placeholder" or self.module_call_graph.get(self.fqn) is None
            # allow placeholder creation if we are not preserving module call signature
        ):
            self.add_placeholder(x)
            if self.parent_call_module is not None:
                # Important to *prepend* the output to match how we are
                # inserting placeholder nodes.
                with self.parent.graph.inserting_before(self.parent_call_module):
                    self.parent_call_module.insert_arg(0, self.parent.remap_input(x))
            return self.node_to_placeholder[x]
        elif x.op == "call_function" and (
            x.target
            in (
                torch.ops.aten.sym_size.int,
                torch.ops.aten.item.default,
                torch.ops.aten.unbind.int,
                torch.ops.aten.sum.dim_IntList,
                torch.ops.aten.view.default,
                torch.ops.aten.diff.default,
            )
            or (hasattr(x.target, "__module__") and x.target.__module__ == "_operator")
        ):
            # export deduplicates sym_size nodes, and may need to re-copy them
            # if module call signature needs to be preserved
            self.copy_sym_call_function(x)
            return self.node_map[x]
        elif self.module_call_graph.get(self.fqn) is not None:
            # x is reading the intermediate value of a mutation, so record it;
            # later we will find where it was created and perform the update
            return self.ivals.read(self, x)  # type: ignore[operator, union-attr]
        else:
            raise RuntimeError(
                f"Could not run remap_input() on op type: {x.op} for node {x}"
            )

    def uplift_common_custom_metadata(self) -> None:
        # [FUNCTION DOC] If every node copied into this frame carries identical, non-empty
        # custom metadata, moves it onto the parent's call node and deletes it from the
        # children. Raises AssertionError if there is common metadata but no parent call node.
        # KNOWN ISSUE: deleting the custom key from every node assumes each node has it; nodes
        # lacking it would raise KeyError (they would have made the metadata differ, so only the
        # empty-versus-missing case is at risk).
        # Copy custom metadata if all nodes have same custom metadata
        custom_meta = None
        for node in self.node_map.values():
            curr_meta = node.meta.get("custom", {})
            if custom_meta is None:
                # first node
                custom_meta = curr_meta
                continue

            if curr_meta != custom_meta:
                custom_meta = {}
                break

        if custom_meta:
            # Lift common custom metadata to parent node and clear children node's custom metadata
            if self.parent_call_module is None:
                raise AssertionError(
                    "parent_call_module must not be None when uplifting custom metadata"
                )
            self.parent_call_module.meta["custom"] = custom_meta
            for node in self.node_map.values():
                del node.meta["custom"]

    def finalize_outputs(self):
        # [FUNCTION DOC] Closes this frame by emitting its graph output and rewiring the
        # parent. With a preserved call signature the outputs are rebuilt into the original
        # output structure and the parent flattens the call result; otherwise every copied node
        # used by nodes outside this frame becomes an output. Parent-side getitem nodes are
        # created for each output and registered in the parent's node map, output value metadata
        # is set, and common custom metadata is lifted. Also forgets this module from the
        # created-modules table. Raises RuntimeError for unsupported output types or a missing
        # output node.
        self.created_modules.pop(self.fqn, None)

        orig_outputs = []

        signature = self.module_call_graph.get(self.child_fqn)
        if signature is not None and self.parent is not None:
            for output in signature.outputs:
                if isinstance(
                    output,
                    (
                        TensorArgument,
                        SymIntArgument,
                        SymBoolArgument,
                        SymFloatArgument,
                        ConstantArgument,
                    ),
                ):
                    if output.name in self.seen_nodes:
                        orig_outputs.append(self.seen_nodes[output.name])
                    else:
                        orig_outputs.append(None)
                else:
                    raise RuntimeError(
                        f"Unsupported data type for output node: {output}"
                    )

            def get_actual_output_node(output):
                # [FUNCTION DOC] HELPER: Returns this frame's node (copied node or placeholder)
                # for a signature output, or None for a missing one. Raises RuntimeError if
                # the value was not produced or imported in this frame.
                if output is None:
                    return None

                seen_node = self.seen_nodes[output.name]
                if seen_node in self.node_map:
                    return self.node_map[seen_node]
                elif seen_node in self.node_to_placeholder:
                    return self.node_to_placeholder[seen_node]
                else:
                    raise RuntimeError(
                        f"Could not find output node {output}. Graph: {self.graph}"
                    )

            tree_out_node = _generate_unflatten(
                self.module,
                tuple(get_actual_output_node(output) for output in orig_outputs),
                signature.out_spec,
            )
            parent_out: torch.fx.Node | None = _generate_flatten_spec(
                self.parent.module, self.parent_call_module, signature.out_spec
            )
            graph_outputs: torch.fx.Node | list[torch.fx.Node] = tree_out_node
        else:
            graph_outputs = []
            # Iterate through nodes we have copied into self.graph.
            for orig_node in self.node_map:
                for user_node in orig_node.users:
                    if user_node.name not in self.seen_nodes:
                        # external user node, need to expose as an output
                        orig_outputs.append(orig_node)
                        graph_outputs.append(self.node_map[orig_node])
                        break

            parent_out = self.parent_call_module
            if len(graph_outputs) == 1:
                graph_outputs = graph_outputs[0]

        if not isinstance(graph_outputs, (list, torch.fx.Node)):
            raise AssertionError(
                f"expected graph_outputs to be list or torch.fx.Node, got {type(graph_outputs)}"
            )

        self.graph.output(graph_outputs)

        # Rewrite outputs in parent module
        if parent_out is None:
            return

        parent_out.meta["val"] = (
            graph_outputs.meta.get("val")
            if isinstance(graph_outputs, torch.fx.Node)
            else [o.meta.get("val") for o in graph_outputs]
        )
        self.uplift_common_custom_metadata()

        if len(orig_outputs) == 1 and signature is None:
            self.parent.node_map[orig_outputs[0]] = parent_out
        else:
            for i, orig_output in enumerate(orig_outputs):
                if orig_output is None:
                    continue
                # Use Proxy to record getitem access.
                proxy_out = torch.fx.Proxy(parent_out)[i].node  # type: ignore[index]
                proxy_out.meta["val"] = orig_output.meta.get("val")
                self.parent.node_map[orig_output] = proxy_out

    def copy_node(self, node):
        # [FUNCTION DOC] Copies a flat-graph node into this frame's graph (remapping its
        # inputs), records the mapping, and marks the node name as seen.
        self.print("copying", node.format_node())
        self.node_map[node] = self.graph.node_copy(node, self.remap_input)
        self.seen_nodes[node.name] = node

    def run_outer(self):
        # [FUNCTION DOC] Drives outlining from the root frame: copies the leading placeholder
        # nodes, runs the nested outlining over the remaining nodes, then copies the output
        # node(s) of the flat graph. Prints debug info only when verbose.
        for i, node in enumerate(self.flat_graph.nodes):
            self.print(i, node.meta.get("nn_module_stack"), node.format_node())

        # Copy all graph inputs
        node_idx: int = 0
        node = self.nodes[node_idx]
        while node.op == "placeholder":
            self.copy_node(node)
            node_idx += 1
            node = self.nodes[node_idx]

        self.run_from(node_idx)

        # Copy graph outputs
        for node in self.flat_graph.nodes:
            if node.op == "output":
                self.copy_node(node)

    def print(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Prints the arguments to standard output only when the verbose flag is set.
        if self.verbose:
            # pyrefly: ignore [not-iterable]
            print(*args, **kwargs)

    def run_from(self, node_idx):
        # [FUNCTION DOC] Walks nodes from node_idx, using each node's module stack to decide
        # what to do: on the output node, finalize (except in the root frame) and return; on a
        # node outside this frame's module, finalize this frame and return the index; on a node
        # in a deeper module, start a nested frame and continue from where it stopped; on a node
        # in this module, copy it (recording graph-argument attribute names for get_attr nodes).
        # Returns the index at which this frame stopped. Raises AssertionError on unexpected
        # placeholders or stack mismatch, RuntimeError if a node has no module stack.
        # KNOWN ISSUE: module_idx is incremented but never used, and the None check on the
        # computed module stack comes after it was already used.
        module_idx = 0
        # Walk through the graph, building up a new graph with the right submodules
        while node_idx < len(self.nodes):
            node = self.nodes[node_idx]
            if node.op == "placeholder":
                raise AssertionError(f"unexpected placeholder node at index {node_idx}")

            self.print()
            self.print("STEP", node_idx, node.format_node())
            self.print(self.module_stack)
            depth = len(self.module_stack)
            if node.op == "output":
                if depth == 1:
                    # We want the output node of the original graph to be handled
                    # specially by the outermost stack frame (in run_outer). So
                    # skip finalization here.
                    return node_idx

                # We've reached the end of the graph. Wrap up all the existing stack frames.
                self.finalize_outputs()
                return node_idx

            if len(node.meta.get("nn_module_stack", {})) == 0:
                raise RuntimeError(f"Unable to find nn_module_stack for node {node}")

            nn_module_stack = node.meta["nn_module_stack"]
            from torch._export.passes._node_metadata_hook import (
                _EMPTY_NN_MODULE_STACK_KEY,
            )

            if (
                len(nn_module_stack) == 1
                and _EMPTY_NN_MODULE_STACK_KEY in nn_module_stack
            ):
                # Empty case from the node_metadata_hook
                node_module_stack = self.module_stack
            else:
                node_module_stack = [
                    (
                        path,
                        ty if path else None,
                        int(k.split("@")[-1]) if "@" in k else 0,
                    )
                    for k, (path, ty) in node.meta["nn_module_stack"].items()
                ]

            if node_module_stack[:depth] != self.module_stack:
                # This means that the current module is done executing and the
                # current node is the beginning of a new module.
                #
                # In this case, we should finalize this module and return without
                # incrementing the node counter.
                self.finalize_outputs()
                self.print("outlining", self.fqn)
                self.print(self.graph)
                return node_idx

            if node_module_stack is None:
                raise AssertionError("node_module_stack must not be None")

            if _is_prefix(self.module_stack, node_module_stack):
                # This means that the current node represents the execution of a new
                # module.
                next_module = node_module_stack[depth]
                self.print("Creating new stack frame for", next_module)
                # Run a nested version of module outliner from the current node
                # counter. Once it is complete, continue from that point.
                next_module_key = list(node.meta["nn_module_stack"].keys())[depth]
                node_idx = _ModuleFrame(
                    self.flat_graph,
                    self.nodes,
                    self.seen_nodes,
                    self.seen_modules,
                    self.seen_attrs,
                    self.created_modules,
                    self,
                    self.module_stack + [next_module],
                    next_module_key.split("@")[0],
                    self.module_call_graph,
                ).run_from(node_idx)
                module_idx += 1
                continue

            # The only remaining possibility is that we are in the right stack
            # frame. Copy the node into this frame's graph and increment the node counter.
            if node_module_stack != self.module_stack:
                raise AssertionError(
                    f"expected node_module_stack {node_module_stack} to equal module_stack {self.module_stack}"
                )

            if node.op == "get_attr":
                # this must be a graph argument for a HOP
                self.seen_attrs[self.child_fqn].add(node.target)

            self.copy_node(node)
            # pyrefly: ignore [unsupported-operation]
            node_idx += 1


@dataclass
class _SubmoduleEntry:
    # [CLASS DOC] HELPER: Record of one call of a module during outlining: the parent's name,
    # module and call node, this module's name, its 1-based call index, and the module itself.
    parent_fqn: str
    parent_module: torch.nn.Module
    parent_call_module: torch.fx.Node
    fqn: str
    call_idx: int
    module: torch.nn.Module


def _outline_submodules(orig_graph: torch.fx.Graph, root_module: UnflattenedModule):
    # [FUNCTION DOC] HELPER: Splits the flat graph into per-module graphs under the root module by
    # running a root _ModuleFrame over it (using the preserved call signatures from the root's
    # call graph). Returns a pair: mapping of module identity to the list of its call entries,
    # and mapping of call name to graph-argument attribute names. Mutates the root module.
    seen_nodes: dict[str, torch.fx.Node] = {}
    seen_modules: dict[int, list[_SubmoduleEntry]] = defaultdict(list)
    seen_attrs: dict[str, set[str]] = defaultdict(set)
    created_modules: dict[str, torch.nn.Module] = {}
    _ModuleFrame(
        orig_graph,
        tuple(orig_graph.nodes),
        seen_nodes,
        seen_modules,
        seen_attrs,
        created_modules,
        None,
        [("", None, 0)],
        "",
        {
            entry.fqn: entry.signature
            for entry in root_module.module_call_graph
            if entry.signature
        },
        module=root_module,
    ).run_outer()
    return seen_modules, seen_attrs


def _reorder_submodules(
    parent: torch.nn.Module, fqn_order: dict[str, int], prefix: str = ""
):
    # [FUNCTION DOC] HELPER: Recursively re-registers submodules so each module's children are
    # ordered by the given name-to-position mapping (unknown names go last, call variants use
    # their base name's position). At the top level it first adds empty modules for any ordered
    # names missing from the tree. Mutates the module tree.
    # TODO Can be optimized by adding submodules ahead of time.
    if prefix == "":
        for fqn in list(fqn_order.keys())[1:]:
            if _get_submodule(parent, fqn) is None:
                _add_submodule(parent, fqn, torch.nn.Module())

    children = []
    for name, child in list(parent._modules.items()):
        if child is None:
            continue
        fqn = prefix + name
        _reorder_submodules(child, fqn_order, prefix=fqn.split("@")[0] + ".")
        delattr(parent, name)
        base_fqn = fqn.split("@")[0]
        children.append(
            (fqn_order.get(fqn, fqn_order.get(base_fqn, len(fqn_order))), name, child)
        )
    children.sort(key=operator.itemgetter(0))
    for _, name, child in children:
        parent.register_module(name, child)


class _IVals:
    # [CLASS DOC] HELPER: Tracker for intermediate values of functionalized mutations that are
    # read across module boundaries. State: node_names_by_fqn, a mapping of module name to the
    # set of node names whose in-place mutation must be re-applied. read records a value;
    # update later inserts the in-place copy nodes into every call of the shared modules.
    """
    Collect the intermediate values of mutations in a graph.

    Example: in the following graph, suppose that buf_in and buf_out
    are the input and output values of a buffer.

        buf_in = placeholder()
        ...
        ival1 = f0(buf_in, ...)  # inside self.n0(...)
        ...
        ival2 = f1(ival1, ...)  # inside self.n1(...)
        ...
        buf_out = f2(ival2, ...)  # inside self.n2(...)
        return buf_out, ...

    Here ival1 and ival2 are intermediate values created inside
    calls to n0 and n1 respectively, and used inside calls to
    n1 and n2 respectively.
    """

    def __init__(self):
        # [FUNCTION DOC] Creates the empty mapping from module name to node-name sets.
        # for each fqn, set of node names corresponding to intermediate values
        self.node_names_by_fqn = defaultdict(set)

    def _is_mutable(self, target):
        # [FUNCTION DOC] HELPER: True if the target is an operator overload whose schema is
        # mutable; False for anything else.
        if isinstance(target, torch._ops.OpOverload):
            return target._schema.is_mutable
        return False

    def read(self, mf, node):
        # [FUNCTION DOC] Called when a frame reads an intermediate mutation value (a call_function
        # node). If the operator is not itself mutable (the mutation was functionalized), records
        # the node name under the innermost module of its stack so update can re-apply the
        # mutation. Returns the frame's remapping of the node's first argument (the mutated
        # input). Raises AssertionError for non call_function nodes.
        # KNOWN ISSUE: unconditionally prints a debug line ("Checking mutability") to standard
        # output on every call.
        """
        Read state corresponding to a given intermediate value.
        """
        # we can assume that the node must be from a mutation
        if node.op != "call_function":
            raise AssertionError(
                f"expected node.op to be 'call_function', got {node.op!r}"
            )
        b = self._is_mutable(node.target)
        print("Checking mutability", node.target, b)
        if not b:
            # so the mutation was functionalized;
            # we will apply the original mutation later (see below)
            fqn, _ = next(reversed(node.meta["nn_module_stack"].values()))
            self.node_names_by_fqn[fqn].add(node.name)
        return mf.remap_input(node.args[0])

    def update(self, partitions):
        # [FUNCTION DOC] For each group of calls sharing a module, finds in each call's graph the
        # recorded nodes and inserts after each an in-place copy of its value into its first
        # argument, copying metadata. Mutates the graphs.
        """
        Update states corresponding to intermediate values that were read.
        """
        for shared_submodules in partitions:
            for entry in shared_submodules:
                graph = entry.module.graph
                node_names = self.node_names_by_fqn[entry.fqn]
                nodes = [n for n in graph.nodes if n.name in node_names]
                for node in nodes:
                    # so node must be from a functionalized mutation;
                    # we perform the original mutation now
                    with graph.inserting_after(node):
                        new_node = graph.create_node(
                            "call_function",
                            torch.ops.aten.copy_.default,
                            (node.args[0], node),
                        )
                        new_node.meta = copy.copy(node.meta)


def _copy_graph_attrs(
    gm: torch.fx.GraphModule,
    root_module: UnflattenedModule,
    seen_attrs: dict[str, set[str]],
):
    # [FUNCTION DOC] HELPER: For each module name, copies the named graph-argument attributes
    # (for example subgraph modules used by higher-order ops) from the original graph module onto
    # the corresponding unflattened module (the root for an empty name). Mutates modules.
    for child_fqn, names in seen_attrs.items():
        module = _get_attr(root_module, child_fqn) if child_fqn else root_module
        for name in names:
            val = getattr(gm, name)
            setattr(module, name, val)


def _deduplicate_modules(partitions):
    # [FUNCTION DOC] HELPER: Within each group of calls sharing a module, finds calls whose graphs
    # are equivalent to an earlier call. Same module name: removes the duplicate from the
    # hierarchy and points the parent's call node at the earlier call. Different module name:
    # replaces the duplicate with the earlier module object but keeps the call name. Returns a
    # mapping of removed call names to the kept call names. Mutates the module tree and call nodes.
    redirected_call_indices = {}
    for shared_submodules in partitions:
        for i, entry in enumerate(shared_submodules):
            child_fqn = _call_name(entry.fqn, entry.call_idx)
            target = _compute_accessor(entry.parent_fqn, child_fqn)
            deduplicated = False
            # Iterate over all previously seen modules, and deduplicate if possible
            for seen in shared_submodules[:i]:
                if _check_graph_equivalence(seen.module, entry.module):
                    parent = entry.parent_module
                    # Since graphs are equivalent, we can deduplicate.
                    # There are two cases.
                    if seen.fqn == entry.fqn:
                        # Case 1: The current module has the same fqn as the seen module.
                        # In this case we have generated a call name that can be optimized away.
                        # So we remove the current module from the hierarchy and replace
                        # the current call name with the seen call name in the parent graph.
                        *prefix, name = target.split(".")
                        _get_attr_via_attr_list(parent, prefix)._modules.pop(name)
                        seen_child_fqn = _call_name(seen.fqn, seen.call_idx)
                        seen_target = _compute_accessor(
                            entry.parent_fqn, seen_child_fqn
                        )
                        entry.parent_call_module.target = seen_target
                        redirected_call_indices[child_fqn] = seen_child_fqn
                        break
                    elif not deduplicated:
                        # Case 2: The current module has a different fqn than the seen module.
                        # In this case we replace the current module with the seen module.
                        # There should be nothing pointing to the current module any more,
                        # so it can be garbage collected.
                        # NOTE: We *do not* replace the current call name with the seen call name
                        # in the parent graph, because this will lose information on which fqn
                        # was actually called. However, it is possible that the current call name
                        # will be optimized away when we find another seen module with the same fqn,
                        # so we do not break out of the loop yet.
                        parent.set_submodule(target, seen.module)
                        deduplicated = True

    return redirected_call_indices


def _sink_params(
    module: torch.nn.Module,
    inputs_to_state: dict[str, list[str]],
    scope: list[str],
    module_id_to_inputs_removed: dict[int, set[str]] | None = None,
):
    # [FUNCTION DOC] HELPER: Recursively replaces placeholder inputs that stand for parameters,
    # buffers or constants with get_attr nodes reading the module's own attribute, erasing the
    # placeholder and removing the matching argument from call_module nodes in parents. Children
    # are processed first; a module already processed (by identity) returns its recorded
    # removals. Inputs are matched using the scope path and the name-to-state-names mapping;
    # unmatched shared-module inputs are skipped. Finalizes InterpreterModules. Returns a mapping
    # of module identity to removed placeholder names. Raises AssertionError on wrong graph or
    # attribute types.
    # KNOWN ISSUE: the early-exit and no-graph paths return a mapping for different keys (all
    # modules versus only this module), so the return value is not uniform.
    """Sink params, buffers, and constants from graph inputs into get_attr nodes.

    Exported modules are purely functional, so they pass their parameters and
    buffers in as inputs to the graph.

    To replicate eager's semantics, we need to get them from the module state
    via get_attr instead.

    module: GraphModule, potentially containing nested submodules.
    inputs_to_state: mapping graph input names to the corresponding key in the state_dict.
    scope: tracks where we are in the module hierarchy, so that we can emit the
        right `getattr(self, "foo.bar")` calls, etc.
    module_id_to_inputs_removed: records inputs removed by child modules, mapping
        the module object id to the list of placeholder node names in the child module
        that were removed.
    """
    if module_id_to_inputs_removed is None:
        module_id_to_inputs_removed = defaultdict(set)

    if id(module) in module_id_to_inputs_removed:
        return {id(module): module_id_to_inputs_removed[id(module)]}

    # We need to use _modules here instead of named_children(), because we
    # explicitly want duplicate modules to show up in the traversal.
    for name, submodule in module._modules.items():
        submod_id_to_inputs_removed = _sink_params(
            cast("torch.nn.Module", submodule),
            inputs_to_state,
            scope + [name],
            module_id_to_inputs_removed,
        )
        for k, v in submod_id_to_inputs_removed.items():
            module_id_to_inputs_removed[k].update(v)

    graph = getattr(module, "graph", None)
    if graph is None or len(graph.nodes) == 0:
        # Not all modules have graphs defined, if they are empty modules with no operations (like ParameterList)
        return module_id_to_inputs_removed

    if not isinstance(graph, torch.fx.Graph):
        raise AssertionError(f"expected graph to be torch.fx.Graph, got {type(graph)}")

    inputs = list(filter(lambda n: n.op == "placeholder", graph.nodes))
    the_last_input = None if len(inputs) == 0 else inputs[-1]

    # Also remove from call_module nodes
    call_module_nodes = filter(lambda n: n.op == "call_module", graph.nodes)
    for node in call_module_nodes:
        submodule = _get_attr(module, node.target)
        # remove placeholder from call_module node arguments, only if we've
        # erased the placeholder node in the corresponding _sink_params() call
        if submodule is not None and id(submodule) in module_id_to_inputs_removed:
            node.args = tuple(
                filter(
                    lambda n: n.name not in module_id_to_inputs_removed[id(submodule)],
                    node.args,
                )
            )

    # Filter out inputs_to_state corresponding to current scope.
    inputs_to_state_of_scope: dict[torch.fx.Node, list[str]] = {}
    for node in inputs:
        if node.name not in inputs_to_state:
            continue

        state_name = None
        for sn in inputs_to_state[node.name]:
            sn_split = sn.split(".")
            if sn_split[: len(scope)] == [x.split("@")[0] for x in scope]:
                state_name = sn_split
                break

        # If there's a mismatch between scope name and state name, then
        # there must be multiple scopes pointing to the same state name,
        # meaning some modules are shared. In such case, we can simply skip
        # updating the current node because another later iteration will
        # take care of this input node when the unique match between scope
        # and state name occurs.  To make sure this always happen, we should
        # enforce the invariant that no placeholder node in the unflattened
        # graph appears in inputs_to_state dict, which means all the extra
        # input nodes have been handled.
        if state_name is None:
            continue

        inputs_to_state_of_scope[node] = state_name

    # Record name of removed inputs for return purpose.
    inputs_removed: set[str] = set()

    for node, state_name in inputs_to_state_of_scope.items():
        if len(node.users) > 0:
            attr_path = state_name[len(scope) :]
            state_attr = _get_attr_via_attr_list(module, attr_path)
            if not isinstance(state_attr, (torch.Tensor, torch.ScriptObject)):
                raise AssertionError(
                    f"expected state_attr to be torch.Tensor or torch.ScriptObject, got {type(state_attr)}"
                )

            # Make sure the newly created get_attr node is placed after the last placeholder node
            with graph.inserting_after(the_last_input):
                new_node = graph.create_node("get_attr", ".".join(attr_path))

            node.replace_all_uses_with(new_node, propagate_meta=True)

        graph.erase_node(node)
        inputs_removed.add(node.name)

    if isinstance(module, InterpreterModule):
        module.finalize()

    return {id(module): inputs_removed}
