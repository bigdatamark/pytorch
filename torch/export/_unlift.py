# mypy: allow-untyped-defs
# [MODULE DOC] Implements "unlifting" of an ExportedProgram into a callable GraphModule: lifted parameters, buffers,
# constants and custom objects become get_attr nodes, mutated outputs are turned back into in-place copy operations,
# a pytree codegen with the original input/output specs is installed, and optionally a _guards_fn submodule that
# asserts shape/value guards on inputs is inserted. Also holds the input tree-spec matching and input-constraint
# pre-hook logic and the GuardsFn module class. No import-time side effects beyond imports.
import copy
import inspect
import math
import warnings
from collections.abc import Sequence
from itertools import chain
from typing import Any

import sympy

import torch
import torch.utils._pytree as pytree
from torch._export.non_strict_utils import (
    _enter_enable_graph_inputs_of_type_nn_module,
    _exit_enable_graph_inputs_of_type_nn_module,
    _get_graph_inputs_of_type_nn_module,
)
from torch._export.passes.add_runtime_assertions_for_constraints_pass import (
    _convert_range_to_int,
)
from torch._export.utils import _check_input_constraints_for_graph
from torch.export.unflatten import _assign_attr, _AttrKind
from torch.fx.experimental.proxy_tensor import _pytree_subclasses_that_lose_info
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.fx.traceback import NodeSource, NodeSourceAction
from torch.utils._sympy.solve import try_solve
from torch.utils._sympy.value_ranges import ValueRanges

from ._remove_effect_tokens_pass import _remove_effect_tokens
from ._tree_utils import reorder_kwargs
from .exported_program import (
    ExportedProgram,
    ExportGraphSignature,
    InputKind,
    OutputKind,
)


def eq_spec(self: pytree.TreeSpec, other: pytree.TreeSpec) -> bool:
    # [FUNCTION DOC] Returns True if two pytree TreeSpecs have the same structure, treating container types that lose
    # information in tracing (such as torch.Size versus tuple) as equal, and ignoring key order of dict contexts.
    # Compares recursively on type, context, child count and children. Pure.
    """
    Refinement of TreeSpec.__eq__ where, e.g., torch.Size(...) matches tuple(...).
    See _pytree_subclasses_that_lose_info in proxy_tensor.py for more details.
    """

    def _normalize_type(t):
        # [FUNCTION DOC] HELPER: Maps a type to its string form after substituting types that lose info under tracing.
        return str(_pytree_subclasses_that_lose_info.get(t, t))

    def _match_normalized_structure(a, b):
        # [FUNCTION DOC] HELPER: Recursive structural comparison of two TreeSpecs as described for eq_spec; returns bool.
        if a is b:
            return True
        if _normalize_type(a.type) != _normalize_type(b.type):
            return False
        if a.type is dict and b.type is dict:
            # in the case of dict, the context is list of keys and we allow the keys to be in any order
            if set(a.context) != set(b.context):
                return False
        elif a.context != b.context:
            return False
        if a.num_children != b.num_children:
            return False
        return all(
            _match_normalized_structure(a, b)
            for a, b in zip(a.children(), b.children())
        )

    return _match_normalized_structure(self, other)


def _check_inputs_match(args, kwargs, in_spec: pytree.TreeSpec) -> list:
    # [FUNCTION DOC] HELPER: Reorders kwargs to match in_spec, flattens (args, kwargs) with key paths, and returns the
    # list of (path, leaf) pairs. Raises ValueError if the resulting tree spec does not match in_spec per eq_spec.
    reordered_kwargs = reorder_kwargs(kwargs, in_spec)
    flat_args_with_path, received_spec = pytree.tree_flatten_with_path(
        (args, reordered_kwargs)
    )

    if not eq_spec(received_spec, in_spec):
        raise ValueError(
            "Trying to flatten user inputs with exported input tree spec: \n"
            f"{in_spec}\n"
            "but actually got inputs with tree spec of: \n"
            f"{received_spec}.\n"
            "Please check that the inputs have the same number and type of "
            "args and kwargs as the ones you used when tracing."
        )

    return flat_args_with_path


def _force_ep_signature_match(ep_guards_code: list[str], input_paths):
    # [FUNCTION DOC] HELPER: Returns a new list of guard strings where each placeholder of the form flat_args index is
    # replaced by the user input path text of the corresponding entry in input_paths. Inputs are not mutated.
    # TODO (tmanlaibaatar)
    # This is band-aid solution to export new tracer replacing
    # shape env sources to flat_args. The real fix should be replacing
    # shape env sources to original user sources but this is quite
    # involved because you need to carefully construct new sources using
    # dynamo and replace all instances of it inside shape env. But it is
    # lot easier to manipulate after we turn them into strings and only
    # time we use these guards is during retracing or running exported program,
    # so it is probably ok to have "not useful" guards on ep for now.
    name_mapping = {}
    for idx, path in enumerate(input_paths):
        name_mapping[f"L['flat_args'][{idx}]"] = f"L{pytree.keystr(path)}"

    new_guards_code = []
    for guard in ep_guards_code:
        for old_name, new_name in name_mapping.items():
            guard = guard.replace(old_name, new_name)
        new_guards_code.append(guard)

    return new_guards_code


def _force_gm_signature_match(ep_guards_code: list[str], signature):
    # [FUNCTION DOC] HELPER: Reconciles variadic-argument naming between guard strings and the unlifted module signature:
    # where a guard refers to a name indexed by a number and the signature has a parameter named name_number, rewrites
    # the guard to the flat parameter name. Returns the new list of guard strings.
    """
    The signature of the originally exported module may not match
    the signature of the unlifted graph module extracted from the
    exported program. The guards code extracted from the exported
    program is based on the former, but the generated guards fn is
    based on the latter; thus we need to reconcile any such diff.
    """

    import re

    # Handle case where signatures may differ in var args.
    orig_arg_names = set()
    for g in ep_guards_code:
        # match substrings of the form L['<name>'][<number>]
        orig_arg_names.update(re.findall(r"L\[\'([^\']+)\'\]\[([0-9]+)\]", g))

    sig_arg_names = set()
    for n in signature.parameters:
        # match substrings of the form <name>_<number>
        sig_arg_names.update(re.findall(r"(.+)_([0-9]+)", n))

    # replace L['<name>'][<number>] with L['<name>_<number>']
    new_guards_code = ep_guards_code
    for match in orig_arg_names:
        if match in sig_arg_names:
            base, idx = match
            new_guards_code = [
                g.replace(f"L['{base}'][{idx}]", f"L['{base}_{idx}']")
                for g in new_guards_code
            ]

    return new_guards_code


def _convert_guards_code_to_fn(
    guards_code: list[str],
    paths_of_placeholders: list[pytree.KeyPath],
):
    # [FUNCTION DOC] HELPER: Turns guard expression strings into a GuardsFn module. Source references are replaced by
    # positional flat args for the executable code and by user paths for the error message; each guard becomes an
    # assert call that fails with a message naming the guard. The generated source is executed (exec) in a namespace
    # with sympy interpreter globals, math and inf; the function is installed as forward (Dynamo may trace through
    # it) and the module is marked impure so dead code elimination keeps it. If normalizing a message hits
    # RecursionError, warns and uses the raw string. Returns the GuardsFn instance.
    """
    Generates Python code given guards code and paths of placeholders.
    We assume that, based on source information,
    - the tracer generates the guards code
    - the input spec generates the paths of placeholders.

    Example:

    Suppose we are given the guards code "L['z']['k'].size()[1] == 3"
    and we are given that ['z']['k'] is the path of placeholder #2.
    Then we will generate:
    ```
    torch._assert(
        args[2].size()[0] == 3,
        "Guard failed: z['k'].size()[0] == 3",
    )
    ```

    FAQ: Why do we generate code based on (flattened) args instead of
    the original (unflattened) inputs? Because this would require
    inserting an additional pytree.unflatten call in our graph.

    FAQ: Why do we not emit RuntimeError on guard failure as we used to?
    Because it is inconvenient :/, get used to AssertionError instead.
    """

    import ast

    from torch.fx.experimental.symbolic_shapes import SYMPY_INTERP

    actual_guards_code = []
    shadow_guards_code = []
    for c in guards_code:
        a, s = c, c
        for idx, path in enumerate(paths_of_placeholders):
            # e.g., replace L['z']['k'] with args[2] for Python code (actual)
            a = a.replace("L" + pytree.keystr(path), f"args[{idx}]")
            # e.g., replace L['z']['k'] with z['k'] for error message (shadow)
            s = s.replace(
                "L" + pytree.keystr(path),
                path[0].key + pytree.keystr(path[1:]),  # type: ignore[attr-defined]
            )
        actual_guards_code.append(a)
        shadow_guards_code.append(s.replace("\n", ""))

    # generate function code as str
    code_str = "\ndef _(*args):\n"
    for actual, shadow in zip(actual_guards_code, shadow_guards_code):
        # printing guards code may potentially introduce redundant parens;
        # we can normalize them out for readability by parsing/unparsing
        # NOTE: this is not necessary for correctness, just deemed desirable
        try:
            _shadow = ast.unparse(ast.parse(shadow, mode="eval"))
        except RecursionError as e:
            # A deeply nested guard expression (e.g. a sum over many symbolic
            # sizes) can exceed the recursion limit in ast.parse/ast.unparse.
            # This normalization only affects the assert error message, so fall
            # back to the un-normalized guard string instead of crashing.
            warnings.warn(f"ast.unparse failed for guard expression: {e}", stacklevel=2)
            _shadow = shadow
        # actual code and shadow error message
        code_str += f'  torch._assert({actual}, "Guard failed: {_shadow}")\n'
    code_str += "  return\n"

    # populate namespace with sympy globals, materialize function (named `_`)
    namespace = {
        **SYMPY_INTERP,
        "math": math,
        "inf": float("inf"),
    }
    exec(code_str, namespace)

    # create and return a module whose forward is the materialized function
    # NOTE: we want Dynamo to trace through this module, to repopulate guards:
    # otherwise we would lose them when retracing
    # NOTE: calling this module will be a side effect (no users): so it must
    # be marked impure to avoid being not cleaned up by DCE
    guards_fn = GuardsFn()
    guards_fn.forward = torch._dynamo.dont_skip_tracing(namespace["_"])  # type: ignore[call-overload, method-assign]
    guards_fn._is_impure = True  # type: ignore[assignment]
    return guards_fn


@torch._dynamo.disable
def _check_input_constraints_for_module(self, args, kwargs):
    # [FUNCTION DOC] HELPER: Validates inputs against the module in_spec and, via the graph placeholders and the module
    # range_constraints, checks input constraints; raises on mismatch. Excluded from Dynamo compilation.
    flat_args_with_path = _check_inputs_match(args, kwargs, self._in_spec)
    _check_input_constraints_for_graph(
        self.graph.find_nodes(op="placeholder"),
        flat_args_with_path,
        self.range_constraints,
    )


def _check_input_constraints_pre_hook(self, args, kwargs):
    # [FUNCTION DOC] HELPER: Forward pre-hook (with kwargs) for the stateful graph module. Does nothing if validate_inputs
    # is false. If a _guards_fn exists, only checks that the input tree structure matches. Otherwise, unless Dynamo is
    # compiling, runs the full input constraint check. Returns None; raises ValueError or constraint errors on mismatch.
    # preserve current behavior for clients that do not want any validation
    if not self.validate_inputs:
        return

    # when a guards function exists, assume that the graph does calls it!
    # so we do not need to check input constraints...but we still want
    # to check inputs match, otherwise we'd get obscure pytree errors
    if hasattr(self, "_guards_fn"):
        _check_inputs_match(args, kwargs, self._in_spec)
        return

    # NOTE: for some reason, Dynamo is tracing into this, we should see why and
    # put compile at the right place. Until then, we can skip the input
    # constraint checks.
    if not torch.compiler.is_dynamo_compiling():
        _check_input_constraints_for_module(self, args, kwargs)


def _unlift_inputs_as_getattr(
    gm: torch.fx.GraphModule,
    lifted_inputs: Sequence[str | None],
) -> tuple[dict[str, torch.fx.Node], dict[str, torch.fx.Node]]:
    # [FUNCTION DOC] HELPER: For each placeholder, if the matching lifted_inputs entry is a name, replaces the placeholder
    # with a get_attr node of that name (copying meta and recording node-source provenance) and erases it; otherwise
    # keeps it as a user input. Returns two mappings: lifted attribute name to new get_attr node, and user input name
    # to placeholder. Mutates gm. Raises AssertionError if lengths differ.
    """
    Unlift inputs referring to params/buffers/constants as getattr nodes in the
    graph
    """
    unlifted_name_to_node = {}
    input_name_to_node = {}

    placeholder_nodes = [node for node in gm.graph.nodes if node.op == "placeholder"]
    if len(lifted_inputs) != len(placeholder_nodes):
        raise AssertionError(
            f"Number of lifted inputs ({len(lifted_inputs)}) does not match "
            f"placeholder nodes ({len(placeholder_nodes)})"
        )
    for input_node, lifted_node in zip(placeholder_nodes, lifted_inputs):
        if lifted_node is None:
            input_name_to_node[input_node.name] = input_node

        else:
            with gm.graph.inserting_after(input_node):
                # It is fine to ignore this warning because
                # it is guaranteed that we will populate this
                # attr later.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    getattr_node = gm.graph.get_attr(lifted_node)
                input_node.replace_all_uses_with(getattr_node)
                metadata = input_node.meta
                gm.graph.erase_node(input_node)
                getattr_node.meta = metadata
                getattr_node.meta["from_node"] = [
                    NodeSource(
                        input_node,
                        "ExportedProgram.module().unlift()",
                        [NodeSourceAction.CREATE, NodeSourceAction.REPLACE],
                    )
                ]
                unlifted_name_to_node[lifted_node] = getattr_node

    return unlifted_name_to_node, input_name_to_node


def _insert_copy_for_mutations(
    gm: torch.fx.GraphModule,
    mutated_outputs: Sequence[str | None],
    unlifted_name_to_node: dict[str, torch.fx.Node],
    input_name_to_node: dict[str, torch.fx.Node],
) -> None:
    # [FUNCTION DOC] HELPER: Rewrites the output node of gm so only user outputs are returned. For each output whose
    # mutated_outputs entry names a buffer or input, inserts an in-place copy_ of that output into the unlifted
    # node, and the copy takes the place of the output. Mutates gm. Raises AssertionError on length mismatch and
    # RuntimeError if a named target is found in neither mapping.
    """
    Find the all the buffers and inputs that were mutated and insert copy_
    operators to reflect mutations.
    """
    output_node = gm.graph.output_node()
    outputs = pytree.tree_flatten(output_node.args)[0]
    if len(outputs) != len(mutated_outputs):
        raise AssertionError(
            f"Number of outputs ({len(outputs)}) does not match "
            f"mutated outputs ({len(mutated_outputs)})"
        )

    user_output_nodes = []
    return_nodes_to_copy = {}
    for return_node, mutated_node_name in zip(outputs, mutated_outputs):
        if mutated_node_name is None:
            user_output_nodes.append(return_node)
            continue

        if mutated_node_name in unlifted_name_to_node:
            mutated_node = unlifted_name_to_node[mutated_node_name]
        elif mutated_node_name in input_name_to_node:
            mutated_node = input_name_to_node[mutated_node_name]
        else:
            raise RuntimeError(
                f"Could not find {mutated_node_name} in either buffer or input nodes"
            )

        with gm.graph.inserting_before(output_node):
            copy_node = gm.graph.call_function(
                torch.ops.aten.copy_.default, (mutated_node, return_node)
            )
            return_nodes_to_copy[return_node] = copy_node

    output_args = tuple(
        return_nodes_to_copy.get(node, node) for node in user_output_nodes
    )
    with gm.graph.inserting_before(output_node):
        # Only return user outputs
        new_output = gm.graph.output(output_args)
        output_node.replace_all_uses_with(new_output)
        gm.graph.erase_node(output_node)
        new_output.name = output_node.name
        new_output.meta.update(output_node.meta)
        new_output.meta["from_node"] = [
            NodeSource(
                output_node,
                "ExportedProgram.module().unlift()",
                [NodeSourceAction.CREATE, NodeSourceAction.REPLACE],
            )
        ]


def _get_codegen(
    in_spec: pytree.TreeSpec,
    out_spec: pytree.TreeSpec | None,
    forward_arg_names: list[str] | None = None,
) -> _PyTreeCodeGen:
    # [FUNCTION DOC] HELPER: Builds a pytree codegen holding the in/out specs and argument names. Names come from
    # forward_arg_names if given, else positional names (plus kwarg names) derived from in_spec. Returns the codegen.
    """
    Create the codegen for the graph module based on the in/out specs
    """
    if forward_arg_names:
        names = forward_arg_names
    elif (
        in_spec.type is tuple
        and in_spec.num_children == 2
        and in_spec.child(0).type is tuple
        and in_spec.child(1).type is dict
    ):
        # if in_spec contains the args (tuple) and kwargs (dict)
        names = [f"arg_{i}" for i in range(in_spec.child(0).num_children)]
        # add kwarg names
        names.extend(in_spec.child(1).context)
    else:
        names = [f"arg_{i}" for i in range(in_spec.num_children)]

    return _PyTreeCodeGen(
        _PyTreeInfo(
            names,
            in_spec,
            out_spec,
        )
    )


def _unlift(
    gm: torch.fx.GraphModule,
    lifted_inputs: Sequence[str | None],
    mutated_outputs: Sequence[str | None],
    in_spec: pytree.TreeSpec,
    out_spec: pytree.TreeSpec | None,
    forward_arg_names: list[str] | None = None,
):
    # [FUNCTION DOC] HELPER: Unlifts gm in place: converts lifted inputs to get_attr nodes, restores mutation copies,
    # installs the pytree codegen from in_spec, out_spec and forward_arg_names, lints and recompiles. Returns gm.
    """
    Args:
        lifted_inputs: A list matching the graph module's input nodes. For
        an input node that is referring to a lifted parameter/buffer, this
        list will contain the fqn the corresponding attribute. Otherwise, this
        list will contain None. This is used to unlift the lifted parameters as
        get_attr nodes.

        mutated_outputs: A list matching the graph module's output nodes. For
        an output node that is referring to a mutated buffer or user input, this
        list will contain the name of the corresponding buffer or user input
        that needs to be mutated. Otherwise, this list will contain None. This
        is used to re-insert an inplace copy_ operator to copy the mutated
        values back to the original node.
    """
    unlifted_name_to_node, input_name_to_node = _unlift_inputs_as_getattr(
        gm, lifted_inputs
    )
    _insert_copy_for_mutations(
        gm, mutated_outputs, unlifted_name_to_node, input_name_to_node
    )
    gm.graph._codegen = _get_codegen(in_spec, out_spec, forward_arg_names)
    gm.graph.lint()
    gm.recompile()
    return gm


def _register_attrs_to_new_gm(
    new_gm: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
    state_dict: dict[str, Any],
    constants: dict[str, Any],
) -> None:
    # [FUNCTION DOC] HELPER: Assigns buffers (non-persistent ones read from constants, others from state_dict), parameters,
    # and lifted custom objects and tensor constants onto new_gm using the signature's names. Mutates new_gm.
    # KNOWN ISSUE: the comment about aliased constants notes they are not tracked here; aliasing is fixed later.
    non_persistent_buffers = set(graph_signature.non_persistent_buffers)
    for name in graph_signature.buffers:
        if name in non_persistent_buffers:
            persistent = False
            value = constants[name]
        else:
            persistent = True
            value = state_dict[name]
        _assign_attr(
            value, new_gm, name, attr_kind=_AttrKind.BUFFER, persistent=persistent
        )
    for name in graph_signature.parameters:
        value = state_dict[name]
        _assign_attr(
            value,
            new_gm,
            name,
            attr_kind=_AttrKind.PARAMETER,
        )

    # Technically this doesn't account for the aliased multiple constants but
    # it is ok because we have a separate pass later in the stack that populates
    # the final gm.
    for name in chain(
        graph_signature.lifted_custom_objs, graph_signature.lifted_tensor_constants
    ):
        value = constants[name]
        _assign_attr(
            value,
            new_gm,
            name,
            attr_kind=_AttrKind.CONSTANT,
        )


class _StatefulGraphModuleFactory(type):
    # [CLASS DOC] HELPER: Metaclass that blocks direct construction of _StatefulGraphModule; only the _create
    # classmethod-like factory method may instantiate it.
    """
    Metaclass that ensures a private constructor for _StatefulGraphModule
    """

    def __call__(cls, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Always raises TypeError saying the class has no public constructor.
        raise TypeError(
            f"{cls.__module__}.{cls.__qualname__} has no public constructor. "
        )

    def _create(cls, root, graph, range_constraints=None):
        # [FUNCTION DOC] HELPER: Bypasses the blocked constructor and builds an instance via the type call with root, graph
        # and range_constraints.
        return super().__call__(
            root,
            graph,
            range_constraints=range_constraints,
        )


class _StatefulGraphModule(torch.fx.GraphModule, metaclass=_StatefulGraphModuleFactory):
    # [CLASS DOC] HELPER: GraphModule carrying export state: range_constraints (list or mapping) and validate_inputs
    # flag (default true) consulted by the input check pre-hook. Create it only via the private factory.
    def __init__(self, root, graph, range_constraints=None):
        # [FUNCTION DOC] HELPER: Initializes the GraphModule, stores range_constraints (empty list if falsy) and sets
        # validate_inputs to true.
        super().__init__(root, graph)
        # Need to fix up non-persistent buffers.
        self.range_constraints = range_constraints or []
        self.validate_inputs = True


def _create_stateful_graph_module(
    plain_graph_module: torch.fx.GraphModule,
    range_constraints,
    ep: ExportedProgram,
) -> _StatefulGraphModule:
    # [FUNCTION DOC] HELPER: Wraps plain_graph_module in a _StatefulGraphModule, registers forward pre-hooks (enable
    # nn.Module-typed graph inputs, then input constraint check) and a forward hook that always undoes the enabling.
    # Then repairs state that fx loses: detaches (with a warning) lifted tensor constants and constants requiring
    # grad (keeping aliases consistent), re-registers lifted tensor constants and ep.constants as constants, and
    # restores non-persistent buffers. Returns the module.
    # KNOWN ISSUE: the first warning text is missing a comma/space ('gradient. but') versus the second warning.
    stateful_gm = _StatefulGraphModule._create(
        plain_graph_module,
        plain_graph_module.graph,
        range_constraints=range_constraints,
    )

    module_types = _get_graph_inputs_of_type_nn_module(ep.example_inputs)
    stateful_gm.register_forward_pre_hook(
        lambda *args, **kwargs: _enter_enable_graph_inputs_of_type_nn_module(
            module_types
        )
    )
    stateful_gm.register_forward_pre_hook(
        _check_input_constraints_pre_hook, with_kwargs=True
    )

    stateful_gm.register_forward_hook(
        lambda *args, **kwargs: _exit_enable_graph_inputs_of_type_nn_module(
            module_types
        ),
        always_call=True,
    )

    # When we have a constant that has requires_grad=True, we need to detach it
    # when we unlift as the tensors that require gradients should be registered
    # via parameters. But this is problematic when we have aliasing two constants
    # because when we call detach, they will become different tensors. This dict
    # keeps track of this logic.
    original_tensor_to_detached_tensor = {}

    # Fix up lifted tensor constants.
    # fx.GraphModule() constructor silently turns a constant attribute of plain_graph_module
    # into a buffer in stateful_gm and creates an inconsistency with graph_signature.
    # We fix this by de-registering these buffers in lifted_tensor_constants
    # and call _assign_attr(attr_kind=CONSTANT) to register them as constants.
    for constant_fqn in ep.graph_signature.lifted_tensor_constants:
        # Sometimes, the constant can require gradient, this is probably a bug in user code,
        # e.g. `self.const = torch.randn(2, 2, requires_grad=True)`.
        # We call detach on the constant_val since they're tensor constants and we don't need to
        # compute their gradients anyway.
        # Users should properly register it as parameter if they want it to require gradient.
        buffer = stateful_gm.get_buffer(constant_fqn)
        if buffer.requires_grad:
            warnings.warn(
                f"A model attribute `{constant_fqn}` requires gradient. "
                f"but it's not properly registered as a parameter. "
                f"torch.export will detach it and treat it as a constant tensor "
                f"but please register it as parameter instead.",
                stacklevel=2,
            )
            detached_buffer = buffer.detach()
            original_tensor_to_detached_tensor[buffer] = detached_buffer
            buffer = detached_buffer
        *prefix, field = constant_fqn.rsplit(".")
        submod = torch.fx.graph_module._get_attr_via_attr_list(stateful_gm, prefix)
        delattr(submod, field)
        _assign_attr(buffer, stateful_gm, constant_fqn, attr_kind=_AttrKind.CONSTANT)

    # Constants are not preserved well when we create a new GraphModule unlike param/buffers
    for const_name, value in ep.constants.items():
        if not torch.fx.graph_module._has_attr(stateful_gm, const_name):
            if isinstance(value, torch.Tensor):
                if value.requires_grad:
                    warnings.warn(
                        f"A model attribute `{const_name}` requires gradient "
                        f"but it's not properly registered as a parameter. "
                        f"torch.export will detach it and treat it as a constant tensor "
                        f"but please register it as parameter instead.",
                        stacklevel=2,
                    )
                    if value in original_tensor_to_detached_tensor:
                        value = original_tensor_to_detached_tensor[value]
                    else:
                        detached_value = value.detach()
                        original_tensor_to_detached_tensor[value] = detached_value
                        value = detached_value
            _assign_attr(
                value,
                stateful_gm,
                const_name,
                attr_kind=_AttrKind.CONSTANT,
            )

    # Fix up non-persistent buffers. torch.fx does not distinguish between
    # persistent and non-persistent buffers, so we must restore that distinction
    # here.
    for buffer in ep.graph_signature.non_persistent_buffers:
        _assign_attr(
            plain_graph_module.get_buffer(buffer),
            stateful_gm,
            buffer,
            attr_kind=_AttrKind.BUFFER,
            persistent=False,
        )

    return stateful_gm


def _get_input_paths(example_inputs, signature):
    # [FUNCTION DOC] HELPER: Binds example args and kwargs to signature (applying defaults) and returns the pytree key
    # paths of all leaves of the bound arguments, in order.
    """
    Generate paths of placeholders, needed for generating the guards function.

    NOTE: Here we make use of the example inputs used for export as well as
    the signature of the unlifted graph module (not preserved by export).
    """

    args, kwargs = example_inputs
    binded = signature.bind(*args, **kwargs)
    binded.apply_defaults()
    ctx = binded.arguments
    flat_example_inputs_with_paths = pytree.tree_leaves_with_path(ctx)
    return [path for path, _ in flat_example_inputs_with_paths]


def _replace_sources(result_str: str, flat_input_paths: list[Any]):
    # [FUNCTION DOC] HELPER: Returns result_str with flat_args index source references replaced by user input path text.
    """
    Given user specified input paths, maybe fix up the guard string
    to reflect user path instead of tracer path.
    """
    name_mapping = {}
    for idx, path in enumerate(flat_input_paths):
        name_mapping[f"L['flat_args'][{idx}]"] = f"L{pytree.keystr(path)}"

    replace = result_str
    for key, val in name_mapping.items():
        replace = replace.replace(key, val)
    return replace


def _get_input_guards_for_graph(
    placeholders: list[torch.fx.Node],
    range_constraints: dict[sympy.Symbol, ValueRanges],
    paths_for_placeholders: list[pytree.KeyPath],
):
    # [FUNCTION DOC] HELPER: Builds guard expression strings mirroring input constraint checks: equality for specialized
    # int, float (including nan and infinities), str and tensor dims; lower and upper bounds for dynamic dims within
    # range_constraints; equality between inputs sharing a symbol; and equations for derived dims solved from a
    # known source. Returns the list of guard strings. Does not mutate inputs.
    # KNOWN ISSUE: the int check is a separate if rather than part of the elif chain, so an int placeholder is still
    # tested by the float branch chain (harmless); bool values count as int.
    """
    Guards generated by the tracer include conditions observed in code, but
    do not include some additional checks we typically do in export.
    For example, when dynamic shapes get specialized, are specified to be
    within a range, or are specified to be in some equational relation,
    corresponding input invalidation is done within a pre_hook, specifically,
    `_check_input_constraints_for_graph`.

    Here we generate guards corresponding to the checks that happen in
    `_check_input_constraints_for_graph`, and add them to the guards already
    generated by the tracer. In the future, it may be worthwhile to separate
    them so that we can allow clients to turn off one but not the other.
    (Looking at you, AOTI.)

    NOTE: We should eventually reconcile this logic with `build_guards` that
    is used by AOT Precompile.
    """

    deferred_expressions = []
    new_guards_code = []
    sources: dict[sympy.Expr, str] = {}

    def handle_symint(expr, src):
        # [FUNCTION DOC] HELPER: Records guards for a symbolic expression sourced at src: defers single-symbol expressions for
        # later solving, emits equality with the first source if seen before, else records the source and adds lower
        # (above 2) and upper (finite) bound guards. Mutates the enclosing lists and mapping.
        if len(expr.free_symbols) == 1:
            # complex equations (e.g., involving derived dims) need to
            # handled later, since we may not have enough information
            # just as we are passing through the placeholders in order
            deferred_expressions.append((src, expr))
        if expr in sources:
            # expressions that appear in multiple sources should force
            # inputs corresponding to those sources to be equal
            # e.g., x.shape[0] == y.shape[1]
            orig_src = sources[expr]
            new_guards_code.append(f"{src} == {orig_src}")
        else:
            sources[expr] = src
            # process value ranges as elsewhere in export
            min_val, max_val = _convert_range_to_int(range_constraints[expr])
            if min_val > 2:
                new_guards_code.append(f"{src} >= {min_val}")
            if max_val < math.inf:
                new_guards_code.append(f"{src} <= {max_val}")

    for placeholder, path in zip(placeholders, paths_for_placeholders):
        src = "L" + pytree.keystr(path)
        meta = placeholder.meta["val"]
        # specializations
        if isinstance(meta, int):
            new_guards_code.append(f"{src} == {meta}")
        if isinstance(meta, float):
            if math.isnan(meta):
                new_guards_code.append(f"math.isnan({src})")
            elif meta == math.inf:
                new_guards_code.append(f"{src} == math.inf")
            elif meta == -math.inf:
                new_guards_code.append(f"{src} == -math.inf")
            else:
                new_guards_code.append(f"{src} == {meta}")
        elif isinstance(meta, str):
            new_guards_code.append(f"{src} == '{meta}'")
        # range constraints and equalities
        elif isinstance(meta, torch.SymInt) and meta.node.expr in range_constraints:
            handle_symint(meta.node.expr, src)
        elif isinstance(meta, torch.Tensor):
            for i, dim in enumerate(meta.shape):
                src = "L" + pytree.keystr(path) + f".size()[{i}]"
                if isinstance(dim, int):
                    # specializations
                    new_guards_code.append(f"{src} == {dim}")
                elif (
                    isinstance(dim, torch.SymInt) and dim.node.expr in range_constraints
                ):
                    # range constraints and equalities
                    handle_symint(dim.node.expr, src)

    unification_map: dict[sympy.Symbol, sympy.Expr] = {}
    py_printer = torch.utils._sympy.printers.PythonPrinter()

    # process complex equations (e.g., involving derived dims)
    for src, expr in deferred_expressions:
        # we know this is the only symbol in expr (see check above)
        symbol = next(iter(expr.free_symbols))
        if symbol in sources:
            # if s0 is already known to be directly sourced from inputs,
            # e.g., z.shape[2], we do not need to do anything further
            # (assume we have already processed constraints on s0 above)
            continue

        # otherwise s0 has some "hidden" source like 'dim'
        # example: src = y.shape[1], expr = s0 + 1
        if symbol in unification_map:
            # suppose that we already know that s0 = x.shape[0] * 2
            # so we can emit the guard: x.shape[0] * 2 + 1 = y.shape[1]
            substitution = expr.subs(unification_map)
            new_guards_code.append(
                py_printer.doprint(sympy.Eq(substitution, sympy.Symbol(src)))
            )
        else:
            # we do not yet know what s0 is, but given s0 + 1 = y.shape[1],
            # we can solve for s0...now knowing that s0 = y.shape[1] - 1
            solution = try_solve(sympy.Eq(expr, sympy.Symbol(src)), symbol)
            if solution is not None:
                definition = solution[1]
                unification_map[symbol] = definition

    return new_guards_code


def _ok_to_generate_guards_fn():
    # [FUNCTION DOC] HELPER: Walks the call stack and returns False if any frame filename contains a listed client name
    # (executorch, modai, on_device_ai, torchao), which cannot handle call-module guard nodes; otherwise True.
    patterns = [
        "executorch",
        "modai",
        "on_device_ai",
        "torchao",
    ]
    # force check_guards=False for files matching `patterns`
    # because they have too many calls to .module() and
    # do not like any call modules in the graph
    # TODO: fix these files to handle guard fns
    frame = inspect.currentframe()
    while frame is not None:
        if any(path in frame.f_code.co_filename for path in patterns):
            return False
        frame = frame.f_back

    return True


def _unlift_exported_program_lifted_states(
    ep: ExportedProgram, check_guards=True
) -> torch.fx.GraphModule:
    # [FUNCTION DOC] Converts an ExportedProgram into a runnable GraphModule with the original calling convention. Works
    # on a copy of the graph and a shallow copy of the program (removing effect tokens unless the dialect is
    # TRAINING), registers state, unlifts inputs/mutations, tags nodes with source provenance, wraps in a stateful
    # module, and, when check_guards is true, the environment allows it, and example inputs exist, builds guards from
    # range constraints plus the program guards and inserts a call to a _guards_fn submodule after the placeholders.
    # Raises AssertionError if call_spec in_spec is None. Returns the module.
    # KNOWN ISSUE: the local list ep_guards built with _replace_sources is never used (dead code).
    check_guards = check_guards and _ok_to_generate_guards_fn()

    source_node_dict = {
        node.name: node for node in ep.graph.nodes if node.op != "placeholder"
    }
    # placeholder node name might change after deepcopy
    placeholder_source_node_dict = {
        node.target: node for node in ep.graph.nodes if node.op == "placeholder"
    }

    new_gm = torch.fx.GraphModule(ep.graph_module, copy.deepcopy(ep.graph))
    new_gm.meta.update(ep.graph_module.meta)
    ep = copy.copy(ep)
    ep._graph_signature = ExportGraphSignature(
        ep._graph_signature.input_specs, ep._graph_signature.output_specs
    )
    ep._graph_module = new_gm

    # TODO T206340015
    if ep.verifiers[0].dialect != "TRAINING":
        ep = _remove_effect_tokens(ep)

    _register_attrs_to_new_gm(new_gm, ep.graph_signature, ep.state_dict, ep.constants)
    forward_arg_names = (
        sig.forward_arg_names if (sig := ep.module_call_graph[0].signature) else None
    )
    lifted_inputs: list[str | None] = [
        (
            in_spec.target
            if in_spec.kind
            in (
                InputKind.BUFFER,
                InputKind.CONSTANT_TENSOR,
                InputKind.PARAMETER,
                InputKind.CUSTOM_OBJ,
            )
            else None
        )
        for in_spec in ep.graph_signature.input_specs
    ]

    mutated_outputs: list[str | None] = [
        (
            out_spec.target
            if out_spec.kind
            in (
                OutputKind.BUFFER_MUTATION,
                OutputKind.USER_INPUT_MUTATION,
                OutputKind.PARAMETER_MUTATION,
            )
            else None
        )
        for out_spec in ep.graph_signature.output_specs
    ]

    for node in new_gm.graph.nodes:
        source_node = None
        if node.op == "placeholder":
            source_node = placeholder_source_node_dict.get(node.target)
        else:
            if node.name in source_node_dict:
                source_node = source_node_dict.get(node.name)
        node.meta["from_node"] = [
            NodeSource(
                source_node,
                "ExportedProgram.module()",
                NodeSourceAction.CREATE,
            )
        ]

    if ep.call_spec.in_spec is None:
        raise AssertionError("ep.call_spec.in_spec cannot be None")
    new_gm = _unlift(
        new_gm,
        lifted_inputs,
        mutated_outputs,
        ep.call_spec.in_spec,
        ep.call_spec.out_spec,
        forward_arg_names=forward_arg_names,
    )
    unlift_gm = _create_stateful_graph_module(new_gm, ep.range_constraints, ep)
    unlift_gm.meta.update(ep.graph_module.meta)

    # create a _guards_fn submodule and insert a call to it after placeholders
    graph = unlift_gm.graph
    placeholders = graph.find_nodes(op="placeholder")
    if check_guards and placeholders and ep.example_inputs:
        sig = inspect.signature(unlift_gm.forward)
        input_paths = _get_input_paths(
            ep.example_inputs,
            sig,
        )

        # TODO (tmanlaibaatar)
        # This is band-aid solution to export new tracer replacing
        # shape env sources to flat_args. The real fix should be replacing
        # shape env sources to original user sources but this is quite
        # involved because you need to carefully construct new sources using
        # dynamo and replace all instances of it inside shape env. But it is
        # lot easier to manipulate after we turn them into strings and only
        # time we use these guards is during retracing or running exported program,
        # so it is probably ok to have "not useful" guards on ep for now.
        ep_guards = []
        for guard in ep._guards_code:
            ep_guards.append(_replace_sources(guard, input_paths))

        guards_code = _get_input_guards_for_graph(
            placeholders, ep.range_constraints, input_paths
        )

        ep_guards_code = _force_ep_signature_match(ep._guards_code, input_paths)
        ep_guards_code = _force_gm_signature_match(ep_guards_code, sig)
        guards_code.extend(ep_guards_code)
        unlift_gm._guards_fn = _convert_guards_code_to_fn(guards_code, input_paths)

        root_nn_module_stack = torch.fx._utils.first_call_function_nn_module_stack(
            graph
        )
        with graph.inserting_after(placeholders[-1]):
            node = graph.call_module("_guards_fn", tuple(placeholders))
            node.meta["nn_module_stack"] = root_nn_module_stack

        unlift_gm.recompile()

    return unlift_gm


class GuardsFn(torch.nn.Module):
    # [CLASS DOC] Module holding the generated guard-checking function; _convert_guards_code_to_fn replaces its
    # forward instance attribute with the generated function. Marked impure when instantiated there.
    """
    Module class for guard functions.
    """

    def forward(self, *args):
        # [FUNCTION DOC] Placeholder that accepts any arguments and does nothing; replaced per instance by generated code.
        pass
