# [MODULE DOC] Validity checking for ExportedProgram objects. Defines SpecViolationError, the Verifier
# base class (dialect ATEN) and TrainingIRVerifier (dialect TRAINING) built through the
# _VerifierMeta metaclass that keeps a registry of verifiers by dialect, node-level checks
# (val metadata, torch_fn), and checks that the module call graph and graph signature agree
# with the graph. Entry points: Verifier.check and load_verifier.
# Import-time side effects: defining the two verifier classes registers them by dialect.
# mypy: allow-untyped-defs
import inspect
import math
import operator
from collections.abc import Iterable
from typing import Any, final, TYPE_CHECKING

import torch
from torch._library.opaque_object import is_custom_class
from torch._ops import HigherOrderOperator, OpOverload
from torch._subclasses.fake_tensor import FakeTensor
from torch.export.graph_signature import (
    CustomObjArgument,
    InputKind,
    SymBoolArgument,
    SymFloatArgument,
    SymIntArgument,
    TensorArgument,
    TokenArgument,
)
from torch.fx import GraphModule


if TYPE_CHECKING:
    from torch.export.exported_program import ExportedProgram


class SpecViolationError(Exception):
    # [CLASS DOC] Exception raised whenever an exported program violates the export specification.
    pass


def is_functional(op: OpOverload) -> bool:
    # [FUNCTION DOC] Return True if the operator overload's schema is not mutable.
    return not op._schema.is_mutable


def _check_has_fake_tensor(node: torch.fx.Node) -> None:
    # [FUNCTION DOC] HELPER: Legacy alias that delegates to _check_val for the node.
    # TODO(angelayi): remove this in favor of _check_val
    return _check_val(node)


def _check_val(node: torch.fx.Node) -> None:
    # [FUNCTION DOC] HELPER: Validate the val entry of a node's metadata. A missing val is allowed only for
    # a call_function node whose operator overload returns nothing. Otherwise val must be a
    # supported kind (None, scalar, str, memory format, dtype, device, layout, tensor, symbolic
    # number, custom object argument, registered custom class, or an iterable of those).
    # Raises SpecViolationError when val is missing or invalid.
    from torch.fx.experimental.symbolic_shapes import SymBool, SymFloat, SymInt

    def _check_correct_val(val):
        # [FUNCTION DOC] HELPER: Recursively decide whether val is an allowed metadata value; returns bool.
        if val is None:
            return True
        elif isinstance(val, (int, bool, str, float)):
            return True
        elif isinstance(
            val, (torch.memory_format, torch.dtype, torch.device, torch.layout)
        ):
            return True
        elif isinstance(
            val, (FakeTensor, torch.Tensor)
        ):  # TODO(zhxchen17) Remove Tensor.
            return True
        elif isinstance(val, (SymInt, SymFloat, SymBool)):
            return True
        elif isinstance(val, CustomObjArgument):
            return True
        elif isinstance(val, Iterable):
            return all(_check_correct_val(x) for x in val)
        elif is_custom_class(type(val)):
            return True
        return False

    def _no_returns(op):
        # [FUNCTION DOC] HELPER: True if op is an OpOverload whose schema has no return values.
        if not isinstance(op, OpOverload):
            return False
        return len(op._schema.returns) == 0

    if "val" not in node.meta:
        if node.op == "call_function" and _no_returns(node.target):
            return
        raise SpecViolationError(f"Node.meta {node.name} is missing val field.")

    val = node.meta["val"]
    if not _check_correct_val(val):
        raise SpecViolationError(f"Node.meta {node.name} has invalid val field {val}")


def _check_torch_fn(node: torch.fx.Node) -> None:
    # [FUNCTION DOC] HELPER: Require a torch_fn metadata entry on the node; raises SpecViolationError if
    # it is missing or malformed.
    # KNOWN ISSUE: the malformed check negates only the tuple test and then ANDs the string
    # tests, so a tuple of wrong element types is never rejected and a non-tuple is indexed.
    torch_fn = node.meta.get("torch_fn")
    if torch_fn is None:
        raise SpecViolationError(
            f"Unable to find torch_fn metadata for node {node.name}"
        )
    if (
        not isinstance(torch_fn, tuple)
        and isinstance(torch_fn[0], str)
        and isinstance(torch_fn[1], str)
    ):
        raise SpecViolationError(
            f"Node.meta {node.name} has invalid torch_fn field {torch_fn}"
        )


class _VerifierMeta(type):
    # [CLASS DOC] HELPER: Metaclass for Verifier classes. Holds a class-level registry mapping dialect
    # name to verifier class. Enforces that the base class defines check and
    # _check_graph_module with dialect ATEN, and that subclasses define a non-ATEN string
    # dialect and do not override those two methods.
    _registry: dict[str, type["Verifier"]] = {}

    def __new__(metacls, name, bases, attrs):
        # [FUNCTION DOC] HELPER: Validate the class definition as described on the metaclass and register
        # the new class under its dialect. Raises SyntaxError if a subclass overrides check or
        # _check_graph_module; raises AssertionError for a missing or invalid dialect or
        # missing base methods. Side effect: updates the shared registry (a later class with
        # the same dialect replaces the earlier one).
        if bases:
            if "check" in attrs or "_check_graph_module" in attrs:
                raise SyntaxError("Overriding method check is not allowed.")
            if "dialect" not in attrs or attrs["dialect"] == "ATEN":
                raise AssertionError(
                    f"subclass must define dialect != 'ATEN', got {attrs.get('dialect')}"
                )
        else:
            if "check" not in attrs:
                raise AssertionError("base class must define 'check' method")
            if "_check_graph_module" not in attrs:
                raise AssertionError(
                    "base class must define '_check_graph_module' method"
                )
            if attrs["dialect"] != "ATEN":
                raise AssertionError(
                    f"base class dialect must be 'ATEN', got {attrs['dialect']}"
                )

        if not isinstance(attrs["dialect"], str):
            raise AssertionError(f"dialect must be str, got {type(attrs['dialect'])}")
        ret = type.__new__(metacls, name, bases, attrs)
        metacls._registry[attrs["dialect"]] = ret  # type: ignore[assignment]
        return ret


def getattr_recursive(obj: Any, target: str) -> Any:
    # [FUNCTION DOC] Resolve a dotted attribute path on obj and return the final attribute. Raises
    # RuntimeError naming the resolved prefix when an attribute along the path is missing.
    target_atoms = target.split(".")
    attr_itr = obj
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


class Verifier(metaclass=_VerifierMeta):
    # [CLASS DOC] Base ATEN-dialect verifier. Subclasses (other dialects) customize the allowed builtin
    # ops, operator types, get_attr types, per-operator checks and extra graph checks.
    # check validates an ExportedProgram: every graph module (including nested ones), the
    # module call graph, and the graph signature; failures raise SpecViolationError.
    dialect = "ATEN"

    def allowed_builtin_ops(self) -> list:
        # [FUNCTION DOC] Return the list of Python builtin/operator/math callables permitted as call_function
        # targets (arithmetic, comparison, logic, getitem, rounding helpers).
        return [
            operator.getitem,
            operator.add,
            operator.mul,
            operator.sub,
            operator.truediv,
            operator.ge,
            operator.le,
            operator.gt,
            operator.lt,
            operator.eq,
            operator.ne,
            operator.floordiv,
            operator.mod,
            operator.and_,
            operator.or_,
            operator.not_,
            operator.pow,
            operator.neg,
            operator.abs,
            operator.lshift,
            operator.rshift,
            math.ceil,
            math.floor,
            math.trunc,
            round,
        ]

    def allowed_op_types(self) -> tuple[type[Any], ...]:
        # [FUNCTION DOC] Return the operator types permitted as call_function targets: OpOverload and
        # HigherOrderOperator.
        return (OpOverload, HigherOrderOperator)

    def allowed_getattr_types(self) -> tuple[type[Any], ...]:
        # [FUNCTION DOC] Return the types allowed for get_attr results in the top-level graph module
        # (GraphModule and TreeSpec).
        return (torch.fx.GraphModule, torch.utils._pytree.TreeSpec)

    def allowed_getattr_types_for_subgm(self) -> tuple[type[Any], ...]:
        # [FUNCTION DOC] Return the types allowed for get_attr results in nested graph modules; additionally
        # permits parameters and tensors because higher-order-op subgraphs may read weights.
        # subgm in HOP's argument could have getattr(weight) nodes, thus stateful
        return (
            torch.fx.GraphModule,
            torch.nn.parameter.Parameter,
            torch.Tensor,  # for buffer and constant tensor
            torch.utils._pytree.TreeSpec,
        )

    def check_valid_op(self, op):
        # [FUNCTION DOC] Hook for dialect-specific operator checks; the base implementation accepts everything.
        pass

    def check_additional(self, gm: GraphModule) -> None:
        # [FUNCTION DOC] Hook for extra dialect-specific checks on a graph module; does nothing by default.
        """
        Additional checks that are specific to some dialects.
        """

    @final
    def check(self, ep: "ExportedProgram") -> None:
        # [FUNCTION DOC] Verify an ExportedProgram: check its graph module (recursively), then its module
        # call graph and graph signature. Returns nothing; raises SpecViolationError (or
        # AssertionError/RuntimeError from helpers) on the first violation. Final: not overridable.
        self._check_graph_module(ep.graph_module)
        _verify_exported_program_module_call_graph(ep)
        _verify_exported_program_signature(ep)

    @final
    def _check_graph_module(self, gm: torch.fx.GraphModule) -> None:
        # [FUNCTION DOC] HELPER: Lint and check every GraphModule found in gm.modules(). call_module and
        # call_method nodes are rejected; call_function nodes need valid val metadata and an
        # allowed operator (an allowed type, builtin or whitelisted torch function; non-training
        # dialects also require OpOverloads to be non-mutating); get_attr targets must resolve
        # to allowed types (with special acceptance of valid lowered backend modules and AOT
        # Inductor wrappers); placeholders need valid val metadata. Finally calls
        # check_additional. Raises SpecViolationError on violations.
        def _allowed_getattr_types(is_toplevel_gm) -> tuple[type[Any], ...]:
            # [FUNCTION DOC] HELPER: Pick the allowed get_attr types for the top-level or a nested module;
            # raises AssertionError if the choice contains the generic object type.
            if is_toplevel_gm:
                ret = self.allowed_getattr_types()
            else:
                ret = self.allowed_getattr_types_for_subgm()
            if any(t is object for t in ret):
                raise AssertionError("allowed_getattr_types must not contain 'object'")
            return ret

        def _check_valid_op(op) -> None:
            # [FUNCTION DOC] HELPER: Check one call_function target against the allowed types, builtins and
            # torch functions, enforce functional ops for non-TRAINING dialects, then run the
            # dialect hook. Raises SpecViolationError for a disallowed or mutable operator.
            def _allowed_builtin_ops() -> list:
                # [FUNCTION DOC] HELPER: Return the allowed builtin ops, raising AssertionError if any is not a builtin.
                ret = self.allowed_builtin_ops()
                if not all(inspect.isbuiltin(op) for op in ret):
                    raise AssertionError("allowed_builtin_ops must all be builtins")
                return ret

            def _allowed_op_types() -> tuple[type[Any], ...]:
                # [FUNCTION DOC] HELPER: Return the allowed op types, raising AssertionError if it includes object.
                ret = self.allowed_op_types()
                if any(t is object for t in ret):
                    raise AssertionError("allowed_op_types must not contain 'object'")
                return ret

            # TODO Remove this allowlist.
            _allowed_torch_functions = (
                torch.autograd.grad_mode.set_grad_enabled,
                torch.sym_int,
                torch.sym_float,
                torch.sym_ite,
                torch.sym_max,
                torch.sym_min,
                torch.sym_not,
                torch.sym_sqrt,
                torch.sym_sum,
                torch.export.custom_ops._call_custom_autograd_function_in_pre_dispatch,
                # TODO (tmanlaibaatar)
                # Predispatch export is able to contain autograd ops.
                # These will be modeled as HOO later
                torch._C._set_grad_enabled,
                torch.amp.autocast_mode._enter_autocast,
                torch.amp.autocast_mode._exit_autocast,
                torch.fx.experimental.symbolic_shapes.cast_symbool_to_symint_guardless,
                torch._functorch.predispatch._add_batch_dim,
                torch._functorch.predispatch._remove_batch_dim,
                torch._functorch.predispatch._vmap_increment_nesting,
                torch._functorch.predispatch._vmap_decrement_nesting,
                torch._functorch.predispatch.lazy_load_decompositions,
                torch._functorch.predispatch._make_dual,
                torch._functorch.predispatch._unpack_dual,
                torch._functorch.predispatch._jvp_increment_nesting,
                torch._functorch.predispatch._jvp_decrement_nesting,
                torch._functorch.predispatch._unwrap_for_grad,
                torch._functorch.predispatch._enter_dual_level,
                torch._functorch.predispatch._exit_dual_level,
            )

            if not isinstance(op, _allowed_op_types()):
                if (
                    op not in _allowed_builtin_ops()
                    and op not in _allowed_torch_functions
                ):
                    raise SpecViolationError(
                        f"Operator '{op}' is not an allowed operator type: {_allowed_op_types()}\n"
                        f"Valid builtin ops: {_allowed_builtin_ops()}"
                        f"Valid torch functions: {_allowed_torch_functions}"
                    )

            if isinstance(op, OpOverload):
                # All ops functional
                # TODO (tmanlaibaatar) more proper way is needed here
                if self.dialect != "TRAINING" and not is_functional(op):
                    raise SpecViolationError(f"operator '{op}' is not functional")
            self.check_valid_op(op)

        for mod in gm.modules():
            is_toplevel_gm = mod is gm

            if not isinstance(mod, torch.fx.GraphModule):
                continue

            mod.graph.lint()
            for node in mod.graph.nodes:
                # TODO(T140410192): should have fake tensor for all dialects
                if node.op in {"call_module", "call_method"}:
                    raise SpecViolationError(
                        f"call_module is not valid: got a class '{node.target}' ",
                    )

                elif node.op == "call_function":
                    _check_val(node)

                    _check_valid_op(node.target)

                elif node.op == "get_attr":
                    if not isinstance(node.target, str):
                        raise SpecViolationError(
                            f"Expected get_attr target to be string, but got {type(node.target)}"
                        )

                    attr = getattr_recursive(mod, node.target)
                    if isinstance(attr, torch.nn.Module):

                        def _is_type(name, ty):
                            # [FUNCTION DOC] HELPER: True if the attribute named name on the module is an instance of ty.
                            return isinstance(getattr(attr, name, None), ty)

                        if type(attr).__name__ == "LoweredBackendModule":
                            if (
                                _is_type("backend_id", str)
                                and hasattr(attr, "original_module")
                                and hasattr(attr, "module_name")
                                and getattr(attr, "backend_id", None) == "aoti"
                            ):
                                continue
                            if (
                                _is_type("backend_id", str)
                                and _is_type("processed_bytes", bytes)
                                and _is_type("compile_specs", list)
                                and hasattr(attr, "original_module")
                            ):
                                continue
                            else:
                                backend_id = getattr(attr, "backend_id", None)
                                processed_bytes = getattr(attr, "processed_bytes", None)
                                compile_specs = getattr(attr, "compile_specs", None)
                                raise SpecViolationError(
                                    f"Invalid get_attr type {type(attr)}. \n"
                                    f"LoweredBackendModule fields: "
                                    f"backend_id(str) : {type(backend_id)}, "
                                    f"processed_bytes(bytes) : {type(processed_bytes)}, "
                                    f"compile_specs(list) : {type(compile_specs)}"
                                )
                        elif type(attr).__name__ == "AOTInductorEPModule":
                            continue

                        elif type(attr).__name__ == "AOTInductorRunnerWrapper":
                            continue

                    if not isinstance(attr, _allowed_getattr_types(is_toplevel_gm)):
                        raise SpecViolationError(
                            f"Invalid get_attr type {type(attr)} on target {node.target}. \n"
                            f"Valid get_attr types: {_allowed_getattr_types(is_toplevel_gm)}"
                        )

                elif node.op == "placeholder":
                    _check_val(node)
                # TODO(zhxchen17)
                # elif node.op == "output":
                #     _check_flattened_outputs()

        self.check_additional(gm)


class TrainingIRVerifier(Verifier):
    # [CLASS DOC] Verifier for the TRAINING dialect; same checks as the base but mutable operators are allowed.
    dialect = "TRAINING"


def _verify_exported_program_module_call_graph(exported_program) -> None:
    # [FUNCTION DOC] HELPER: For every module call graph entry with a signature, check that each named
    # input and output argument corresponds to a node in the program graph. Raises
    # SpecViolationError for a missing name.
    module_call_graph = exported_program.module_call_graph
    nodes = {node.name for node in exported_program.graph.nodes}
    for entry in module_call_graph:
        if entry.signature is not None:
            for arg in entry.signature.inputs:
                if arg.name and arg.name not in nodes:
                    raise SpecViolationError(
                        f"Input {arg.name} does not exist in the graph."
                    )
            for arg in entry.signature.outputs:
                if arg.name and arg.name not in nodes:
                    raise SpecViolationError(
                        f"Output {arg.name} does not exist in the graph."
                    )


def _verify_exported_program_signature(exported_program) -> None:
    # [FUNCTION DOC] HELPER: Check the graph signature against the graph and program state. Inputs: the
    # placeholder count must equal the input spec count, names must match for tensor and
    # symbolic args, and each kind (parameter, buffer, constant tensor, custom object, token)
    # must have the right argument type, a target, and be present in the state dict or
    # constants as appropriate (persistent buffers may also be found in constants or a
    # lazily built graph module state dict; non-persistent ones must not be in the state dict).
    # Outputs: counts must match, mutated-buffer/parameter/user-input outputs must point to
    # existing entries, and user outputs must be in signature order. Raises SpecViolationError.
    # KNOWN ISSUE: the token-kind error message wrongly says constant tensor / tensor argument.
    # Check ExportedProgram signature matches
    gs = exported_program.graph_signature

    # Lazily computed on first use; avoids calling graph_module.state_dict()
    # unless at least one buffer is missing from the top-level state_dict.
    _gm_state_dict: dict | None = None

    # Check every node in the signature exists in the graph
    input_node_names = [
        node.name for node in exported_program.graph.nodes if node.op == "placeholder"
    ]

    if len(input_node_names) != len(gs.input_specs):
        input_spec_names = [
            spec.arg.name for spec in gs.input_specs if hasattr(spec.arg, "name")
        ]
        missing_in_specs = set(input_node_names) - set(input_spec_names)
        missing_in_graph = set(input_spec_names) - set(input_node_names)
        raise SpecViolationError(
            f"Number of graph inputs ({len(input_node_names)}) "
            f"does not match number of inputs in the graph signature ({len(gs.input_specs)})\n"
            f"Placeholders missing input_specs: {missing_in_specs}\n"
            f"Input_specs missing placeholders: {missing_in_graph}"
        )

    for input_spec, node in zip(gs.input_specs, input_node_names):
        if isinstance(
            input_spec.arg,
            (TensorArgument, SymIntArgument, SymFloatArgument, SymBoolArgument),
        ):
            if input_spec.arg.name != node:
                raise SpecViolationError(
                    f"Input spec name {input_spec.arg.name} does not match node name {node}"
                )

        if input_spec.kind == InputKind.USER_INPUT:
            continue

        elif input_spec.kind == InputKind.PARAMETER:
            if not isinstance(input_spec.arg, TensorArgument):
                raise SpecViolationError(
                    f"Parameter {input_spec.name} is not a tensor argument. Found {input_spec.arg} instead."
                )
            if input_spec.target is None:
                raise SpecViolationError(
                    f"InputSpec for {input_spec.name} has no target."
                )

            param = input_spec.target
            if param not in exported_program.state_dict:
                raise SpecViolationError(f"Parameter {param} is not in the state dict.")

            if not isinstance(exported_program.state_dict[param], torch.nn.Parameter):
                raise SpecViolationError(
                    f"State dict entry for parameter {param} is not an instance of torch.nn.Parameter."
                )

        elif input_spec.kind == InputKind.BUFFER:
            if not isinstance(input_spec.arg, TensorArgument):
                raise SpecViolationError(
                    f"Buffer {input_spec.name} is not a tensor argument. Found {input_spec.arg} instead."
                )
            if input_spec.target is None:
                raise SpecViolationError(
                    f"InputSpec for {input_spec.name} has no target."
                )

            buffer = input_spec.target
            if input_spec.persistent is None:
                raise SpecViolationError(
                    f"Buffer {buffer} is missing a persistence flag"
                )

            if (
                input_spec.persistent is True
                and buffer not in exported_program.state_dict
            ):
                # Allow buffers that live in constants or in a subgraph
                # submodule (e.g. lifted tensor constants from
                # invoke_subgraph tracing stored under repeated_subgraph0).
                # Use a lazy copy of the graph module state to avoid the
                # cost of state_dict() when no fallback is needed.
                if buffer not in exported_program.constants:
                    if _gm_state_dict is None:
                        _gm_state_dict = exported_program.graph_module.state_dict()
                    if buffer not in _gm_state_dict:
                        raise SpecViolationError(
                            f"Buffer {buffer} is not in the state dict."
                        )

            if input_spec.persistent is False and buffer in exported_program.state_dict:
                raise SpecViolationError(
                    f"Non-persistent buffer {buffer} is in the state dict, it should not be."
                )
        elif input_spec.kind == InputKind.CONSTANT_TENSOR:
            if not isinstance(input_spec.arg, TensorArgument):
                raise SpecViolationError(
                    f"Constant tensor {input_spec.name} is not a tensor argument. Found {input_spec.arg} instead."
                )
            if input_spec.target is None:
                raise SpecViolationError(
                    f"InputSpec for {input_spec.name} has no target."
                )

            tensor_const = input_spec.target
            if tensor_const not in exported_program.constants:
                raise SpecViolationError(
                    f"Constant tensor {tensor_const} is not in the constants dictionary."
                )
        elif input_spec.kind == InputKind.CUSTOM_OBJ:
            if not isinstance(input_spec.arg, CustomObjArgument):
                raise SpecViolationError(
                    f"Custom object {input_spec.name} is not a custom object argument. Found {input_spec.arg} instead."
                )
            if input_spec.target is None:
                raise SpecViolationError(
                    f"InputSpec for {input_spec.name} has no target."
                )

            custom_obj = input_spec.target
            if custom_obj not in exported_program.constants:
                raise SpecViolationError(
                    f"Custom object {custom_obj} is not in the constants dictionary."
                )
        elif input_spec.kind == InputKind.TOKEN:
            if not isinstance(input_spec.arg, TokenArgument):
                raise SpecViolationError(
                    f"Constant tensor {input_spec.name} is not a tensor argument. Found {input_spec.arg} instead."
                )
        else:
            raise SpecViolationError(f"Unknown InputKind {input_spec.kind}.")

    # Check outputs
    output_node = list(exported_program.graph.nodes)[-1]
    if output_node.op != "output":
        raise AssertionError(f"last node must be output, got {output_node.op}")
    output_nodes = [
        arg.name if isinstance(arg, torch.fx.Node) else arg
        for arg in output_node.args[0]
    ]

    if len(output_nodes) != len(gs.output_specs):
        output_spec_names = [
            spec.arg.name if hasattr(spec.arg, "name") else str(spec.arg)
            for spec in gs.output_specs
        ]
        missing_out_specs = set(output_nodes) - set(output_spec_names)
        missing_out_graph = set(output_spec_names) - set(output_nodes)
        raise SpecViolationError(
            f"Number of output nodes {len(output_nodes)} is different "
            f"Than the number of outputs specified by the graph signature: {len(gs.output_specs)}\n"
            f"Nodes missing output_specs: {missing_out_specs}\n"
            f"Output_specs missing nodes: {missing_out_graph}"
        )

    num_tokens = len(gs.output_tokens)
    buffers_to_mutate = gs.buffers_to_mutate
    parameters_to_mutate = gs.parameters_to_mutate
    user_inputs_to_mutate = gs.user_inputs_to_mutate
    end = (
        len(buffers_to_mutate)
        + len(parameters_to_mutate)
        + len(user_inputs_to_mutate)
        + num_tokens
    )
    mutate_nodes: list[str] = output_nodes[num_tokens:end]
    user_output_nodes = output_nodes[end : end + len(gs.user_outputs)]

    for mutation_node in mutate_nodes:
        if mutation_node in buffers_to_mutate:
            if buffers_to_mutate[mutation_node] not in gs.buffers:
                raise SpecViolationError(
                    f"Buffer output {mutation_node} does not point to a buffer that exists. \n"
                    f"Dict of buffers that are mutated, in order: {buffers_to_mutate} \n"
                    f"Buffer nodes available: {gs.buffers} \n"
                )
        elif mutation_node in parameters_to_mutate:
            if parameters_to_mutate[mutation_node] not in gs.parameters:
                raise SpecViolationError(
                    f"Parameter output {mutation_node} does not point to a parameter that exists. \n"
                    f"Dict of parameters that are mutated, in order: {parameters_to_mutate} \n"
                    f"Parameter nodes available: {gs.parameters} \n"
                )
        elif mutation_node in user_inputs_to_mutate:
            if user_inputs_to_mutate[mutation_node] not in gs.user_inputs:
                raise SpecViolationError(
                    f"User input output {mutation_node} does not point to a user input that exists. \n"
                    f"Dict of user inputs that are mutated, in order: {user_inputs_to_mutate} \n"
                    f"User input nodes available: {gs.user_inputs} \n"
                )
        else:
            raise SpecViolationError(
                f"Mutation node {mutation_node} is neither a buffer nor a user input. "
                f"Buffers to mutate: {buffers_to_mutate}, User inputs to mutate: {user_inputs_to_mutate}"
            )

    for user_output_node, user_output_name in zip(user_output_nodes, gs.user_outputs):
        if user_output_node != user_output_name:
            raise SpecViolationError(
                f"User output {user_output_node} is not in the correct "
                "order or is not found in the "
                f"exported program's user_output list: {gs.user_outputs}. "
            )


def load_verifier(dialect: str) -> type[Verifier]:
    # [FUNCTION DOC] Return the verifier class registered for dialect. ATEN or the empty string yields the
    # base Verifier (registered as ATEN, or Verifier itself as a fallback); other dialects
    # are looked up directly and raise KeyError if unregistered.
    if dialect == "ATEN" or dialect == "":
        return _VerifierMeta._registry.get(dialect, Verifier)
    return _VerifierMeta._registry[dialect]
