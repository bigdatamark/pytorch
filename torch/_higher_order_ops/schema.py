# [MODULE DOC] Generates torch FunctionSchema objects for HigherOrderOperator invocations from
# example values. Contains HopArgumentInfo (argument description), generators for argument infos,
# JIT types, C arguments and function schemas, the HopSchemaGenerator builder, the HopSchema
# FunctionSchema subclass (which also carries a pytree spec) and find_hop_schema for FX graphs.
# Import-time side effects: builds a mapping from Python scalar types to JIT types.
import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import is_custom_class
from torch.fx.node import Target


# Below is an implementation of generating FunctionSchema from example values.
# This is helpful for generating FunctionSchema for HigherOrderOperator, where
# we don't have a function to inspect and each call of the higher order operator
# would have different schema.
@dataclass(frozen=True, slots=True)
class HopArgumentInfo:
    # [CLASS DOC] Immutable record describing one hop argument or output: name, example value,
    # default value, whether the subgraph mutates it (always False for outputs) and whether it is
    # keyword-only.
    # Could give a name to the operand by default it's empty string.
    name: str
    example_value: Any
    # Provide a default_value
    default_value: Any
    # Whether this argument gets mutated in the hop subgraph.
    # For output, this should always be False
    is_mutated: bool
    kw_only: bool


class HopArgumentInfoGen:
    # [CLASS DOC] Namespace of static factory helpers that build HopArgumentInfo objects.
    @staticmethod
    def from_example(
        example_value: Any,
        *,
        name: str = "",
        default_value: Any | None = None,
        is_mutated: bool = False,
        kw_only: bool = False,
    ) -> HopArgumentInfo:
        # [FUNCTION DOC] Builds a HopArgumentInfo from an example value and optional name,
        # default, mutation flag and keyword-only flag. If a default is given its type must be
        # exactly the type of the example value, else AssertionError is raised.
        if default_value is not None:
            if type(example_value) is not type(default_value):
                raise AssertionError(
                    f"example_value type {type(example_value)} doesn't match default_value type: {type(default_value)}"
                )

        return HopArgumentInfo(
            name=name,
            example_value=example_value,
            default_value=default_value,
            is_mutated=is_mutated,
            kw_only=kw_only,
        )


class CTypeGen:
    # [CLASS DOC] Maps example Python values to torch JIT types. Class attribute convert_to_base_ty
    # maps int, float, str and bool to their JIT types (KNOWN ISSUE: not used by from_example).
    convert_to_base_ty = {
        int: torch._C.IntType.get(),
        float: torch._C.FloatType.get(),
        str: torch._C.StringType.get(),
        bool: torch._C.BoolType.get(),
    }

    # should return torch._C.JitType but that annotation is busted
    @staticmethod
    def from_example(obj: Any) -> Any:
        # [FUNCTION DOC] Returns the JIT type for an example value: GraphModule maps to the Any
        # type; SymInt, SymBool map to their sym types; SymFloat to Float; fake script objects,
        # pytree specs and registered opaque/custom classes to the PyObject type; anything else
        # uses JIT type inference (which may fail for unsupported values).
        import torch

        if isinstance(obj, torch.fx.GraphModule):
            return torch._C.AnyType.get()
        elif isinstance(obj, torch.SymInt):
            return torch._C.SymIntType.get()
        elif isinstance(obj, torch.SymBool):
            return torch._C.SymBoolType.get()
        elif isinstance(obj, torch.SymFloat):
            return torch._C.FloatType.get()
        elif isinstance(obj, (FakeScriptObject, pytree.TreeSpec)) or is_custom_class(
            type(obj)
        ):
            return torch._C.PyObjectType.get()  # pyrefly: ignore[missing-attribute]
        return torch._C._jit_try_infer_type(obj).type()


class CArgumentGen:
    # [CLASS DOC] Namespace of static helpers that build torch._C.Argument objects.
    @staticmethod
    def from_hop_argument_info(
        arg_idx: int, arg_info: HopArgumentInfo, is_output: bool = False
    ) -> Any:
        # [FUNCTION DOC] Converts a HopArgumentInfo to a torch._C.Argument. Outputs get an unnamed
        # argument of the inferred type. Inputs carry name, default, keyword-only flag and alias
        # info; a mutated input gets a write alias set unique to its position (arg_idx).
        typ = CTypeGen.from_example(arg_info.example_value)
        if is_output:
            return torch._C.Argument("", typ, None, None, False, None)

        alias_set = set({f"alias::a{arg_idx}"}) if arg_info.is_mutated else set()
        alias_info = torch._C._AliasInfo(arg_info.is_mutated, alias_set, alias_set)  # type: ignore[attr-defined]
        return torch._C.Argument(
            arg_info.name,
            typ,
            None,
            arg_info.default_value,
            arg_info.kw_only,
            alias_info,
        )


class HopSchemaGenerator:
    # [CLASS DOC] Builder for a hop's FunctionSchema. State: arg_infos (ordered argument records),
    # example_outputs (flat list of output examples), schema_tree_spec (optional pytree spec of the
    # original unflattened inputs) and the hop. Usage: add_arg per flat input, add_output per
    # output, optionally add_schema_tree_spec, then gen_schema.
    def __init__(self, hop: torch._ops.HigherOrderOperator):
        # [FUNCTION DOC] Stores the hop and initializes empty argument, output and tree spec state.
        self.arg_infos: list[HopArgumentInfo] = []
        self.example_outputs: list[Any] = []
        self.schema_tree_spec: pytree.TreeSpec | None = None
        self.hop = hop

    def add_arg(
        self,
        name: str,
        example_value: Any,
        default_value: Any | None = None,
        is_mutated: bool = False,
        kw_only: bool = False,
    ) -> None:
        # [FUNCTION DOC] Appends an argument description. Callable examples must be GraphModules
        # or operators (AssertionError otherwise, except registered custom classes). The example
        # must be a pytree leaf, otherwise RuntimeError is raised. Mutates arg_infos.
        if callable(example_value) and not is_custom_class(type(example_value)):
            if not isinstance(
                example_value, (torch.fx.GraphModule, torch._ops.OperatorBase)
            ):
                raise AssertionError(
                    "Expect callable to be a GraphModule or an OperatorBase. Please call materialize_as_graph first "
                    f"to turn callable arguments {example_value} into a GraphModule."
                )
        _, flat_spec = pytree.tree_flatten(example_value)
        if not flat_spec.is_leaf():
            raise RuntimeError(
                f"example_value {example_value} is not a leaf node. "
                "Please only add flattened inputs to the hop schema. "
                "If you need some structure in the arguments, please"
                "add_arg for flattened args one by one then "
                "call add_schema_tree_spec to register the original pytree "
                " spec of the args."
            )

        arg_info = HopArgumentInfoGen.from_example(
            example_value=example_value,
            name=name,
            default_value=default_value,
            is_mutated=is_mutated,
            kw_only=kw_only,
        )
        self.arg_infos.append(arg_info)

    def add_output(self, output: Any) -> None:
        # [FUNCTION DOC] Appends one example output value to the output list.
        self.example_outputs.append(output)

    def add_schema_tree_spec(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Records the pytree spec obtained by flattening the given positional and
        # keyword arguments together, so the original structured inputs can be rebuilt. Overwrites
        # any earlier spec.
        """schema tree spec is the tree spec from flattening all inputs to the hop with pytree.tree_flatten
        Since torch.FunctionSchema only have proper mutation/alias support for flattened inputs, we need
        to store the tree spec in order to reconstruct the inputs to the hop.
        """
        self.schema_tree_spec = pytree.tree_flatten((args, kwargs))[1]

    def gen_schema(self) -> torch._C.FunctionSchema:
        # [FUNCTION DOC] Builds the HopSchema. Raises RuntimeError if any argument example is not a
        # pytree leaf and no tree spec was registered. Outputs are packed into a single tuple
        # record named out. Returns the schema named after the hop.
        for i, arg_info in enumerate(self.arg_infos):
            arg_spec = pytree.tree_flatten(arg_info.example_value)[1]
            if not arg_spec.is_leaf() and self.schema_tree_spec is None:
                raise RuntimeError(
                    f"example_value of arg_infos[{i}] is {arg_info.example_value}, which is not a leaf node. "
                    "Please call add_schema_tree_spec to add a schema tree spec first. "
                    "Or consider changing the hop's signature to only take flattened arguments."
                )

        return CFunctionSchemaGen.from_hop_argument_info(
            str(self.hop),
            self.arg_infos,
            HopArgumentInfoGen.from_example(tuple(self.example_outputs), name="out"),
            self.schema_tree_spec,
        )


class CFunctionSchemaGen:
    # [CLASS DOC] Namespace holding the static function that assembles a HopSchema from argument
    # records; enforces flat inputs, tuple outputs and no output mutation (see docstring note).
    """
    Note: [HigherOrderOperator schema generation]
    Each invocation of a HigherOrderOperator will have a different schema.
    For example, the schema of torch.cond varies depending on the true_fn and
    false_fn. So we need a way to generate the schema for each invocation of a HOP.

    We want to enforce the following invariants for HOP's schema:
        1. Flattened inputs. There should be no pytree structure in it.
        2. Flattened outputs. Note even if the hop returns a single value, it should be wrapped as a tuple.
        3. No aliasing. This includes inp-inp aliasing, inp-out aliasing and out-out aliasing.

    By enforcing these invariants, we could make HOP's schema meets the requirement of schema parser
    and makes hop easier to handle downstream. For example, suppose we have an invoke_quant_test HOP:

    class GraphModule(torch.nn.Module):
        def forward(self, l_x_, l_y_):
            subgraph_0 = self.subgraph_0
            invoke_quant_test = torch.ops.higher_order.invoke_quant_test(subgraph_0, l_x_, l_y_, scheme = 'nf4');

        class subgraph_0(torch.nn.Module):
            def forward(self, l_x_, l_y_):
                add_ = l_x_.add_(1)
                matmul = l_x_ @ l_y_
                sin = matmul.sin()
                child = sin.cos()
                child_1 = l_x_ + l_y_
                child_2 = l_x_ - l_y_
                child_3 = l_x_ @ l_y_
                return (child, child_1, child_2, child_3)

    By encoding the inputs of hop into a list of HopArgumentInfo and output as a single HopArgumentInfo,
    we would get the following schema:
        invoke_quant_test(Any arg0, Tensor(!) arg1, Tensor arg2, str scheme="\\"nf4\\"") -> (Tensor, Tensor, Tensor, Tensor)
    """

    @staticmethod
    def from_hop_argument_info(
        op_name: str,
        inp_argument_info: list[HopArgumentInfo],
        out_argument_info: HopArgumentInfo,
        schema_tree_spec: pytree.TreeSpec | None,
    ) -> Any:
        # [FUNCTION DOC] Builds a HopSchema from op name, input records, a single output record and
        # the optional tree spec. The output example must be a tuple (AssertionError otherwise) and
        # must not be marked mutated. A one-element tuple yields a single return of the tuple type
        # (the whole output record); otherwise one return per element named out0, out1 and so on.
        args = []
        for i, arg_info in enumerate(inp_argument_info):
            args.append(CArgumentGen.from_hop_argument_info(i, arg_info))

        # NOTE: we want the output to always be a single argument with torch._C.TupleType.
        if not isinstance(out_argument_info.example_value, tuple):
            raise AssertionError(
                f"expect out_argument_info's example_value to be a tuple but got {out_argument_info.example_value}"
            )
        if out_argument_info.is_mutated:
            raise AssertionError(
                "out_argument_info.is_mutated should always be set to False."
            )
        rets = None
        if len(out_argument_info.example_value) == 1:
            rets = [CArgumentGen.from_hop_argument_info(0, out_argument_info, True)]
        else:
            rets = [
                CArgumentGen.from_hop_argument_info(
                    i,
                    HopArgumentInfoGen.from_example(
                        name=f"out{i}",
                        example_value=val,
                        default_value=None,
                        is_mutated=False,
                    ),
                    is_output=True,
                )
                for i, val in enumerate(out_argument_info.example_value)
            ]

        return HopSchema(
            op_name,
            "",
            args,
            rets,
            False,
            False,
            schema_tree_spec,
        )


class HopSchema(torch._C.FunctionSchema):
    # [CLASS DOC] FunctionSchema subclass that additionally stores tree_spec, the pytree spec of the
    # original hop inputs, plus is_vararg and is_varret flags.
    def __init__(
        self,
        name: str,
        overload_name: str,
        arguments: list[torch._C.Argument],
        returns: list[torch._C.Argument],
        is_vararg: bool,
        is_varret: bool,
        schema_tree_spec: pytree.TreeSpec | None,
    ):
        # [FUNCTION DOC] Stores the tree spec and vararg flags, then initializes the base schema
        # with the given name, overload name, arguments and returns.
        self.tree_spec = schema_tree_spec
        self.is_vararg = is_vararg
        self.is_varret = is_varret
        super().__init__(
            name,
            overload_name,
            arguments,
            returns,
            self.is_vararg,
            self.is_varret,
        )

    def __deepcopy__(self, memo: Any) -> "HopSchema":
        # [FUNCTION DOC] Returns a new HopSchema sharing arguments and returns but with a deep copy
        # of the tree spec. memo is unused.
        # Need to additionally copy the tree_spec since
        # it's not a member of torch._C.FunctionSchema
        return HopSchema(
            self.name,
            self.overload_name,
            self.arguments,
            self.returns,
            self.is_vararg,
            self.is_varret,
            copy.deepcopy(self.tree_spec),
        )


def find_hop_schema(
    gm: torch.fx.GraphModule, target: Target
) -> list[torch._C.FunctionSchema]:
    # [FUNCTION DOC] For every call_function node in gm targeting the given hop, reconstructs
    # example args/kwargs (get_attr nodes resolved to module attributes, other nodes to their
    # example_value or val metadata) and calls the target's gen_schema with them. Returns the list
    # of schemas in node order. Raises KeyError if a node has neither example_value nor val meta.
    schemas = []
    for node in gm.graph.find_nodes(op="call_function", target=target):

        def _get_example_value(node: torch.fx.Node) -> Any:
            # [FUNCTION DOC] HELPER: Returns the module attribute for a get_attr node (target must
            # be a string, else AssertionError), otherwise the node's example_value or val meta.
            if node.op == "get_attr":
                if not isinstance(node.target, str):
                    raise AssertionError(
                        f"expected node.target to be str for get_attr, got {type(node.target)}"
                    )
                return getattr(gm, node.target)
            else:
                return (
                    node.meta["example_value"]
                    if "example_value" in node.meta
                    else node.meta["val"]
                )

        fake_args, fake_kwargs = pytree.tree_map_only(
            torch.fx.Node,
            _get_example_value,
            (node.args, node.kwargs),
        )
        schema = node.target.gen_schema(*fake_args, **fake_kwargs)
        schemas.append(schema)
    return schemas
