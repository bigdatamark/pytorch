# [MODULE DOC] Utilities that build torchgen schema objects (types, arguments, returns and
# function schemas) from real example values rather than from native_functions.yaml text.
# Not used by the code generators themselves; used e.g. for HigherOrderOperator schemas.
# No import-time side effects besides imports.
from typing import Any

from torchgen.model import (
    Annotation,
    Argument,
    Arguments,
    BaseOperatorName,
    BaseTy,
    BaseType,
    CustomClassType,
    FunctionSchema,
    ListType,
    OperatorName,
    Return,
)


# Note: These aren't actually used in torchgen, they're some utilities for generating a schema
# from real arguments. For example, this is used to generate HigherOrderOperators' schema since
# their schemas can vary for different instances of the same HOP.


class TypeGen:
    # [CLASS DOC] Stateless namespace that infers schema types from example values.
    # convert_to_base_ty maps the Python scalar types int, float, str, bool to BaseTy members.
    convert_to_base_ty = {
        int: BaseTy.int,
        float: BaseTy.float,
        str: BaseTy.str,
        bool: BaseTy.bool,
    }

    @staticmethod
    def from_example(obj: Any) -> BaseType | ListType | CustomClassType:
        # [FUNCTION DOC] Infers a schema type from an example runtime value.
        # GraphModule, Tensor, SymInt and SymBool map to their base types; a script object
        # maps to a custom class type named after its class; int, float, str and bool map to
        # the matching base type. A list or tuple maps to a list type carrying the length and
        # the element type. Imports torch lazily.
        # Raises AssertionError for an empty sequence, RuntimeError for a sequence whose
        # elements have differing types, and RuntimeError for any other unsupported type.
        import torch

        if isinstance(obj, torch.fx.GraphModule):
            return BaseType(BaseTy.GraphModule)
        elif isinstance(obj, torch.Tensor):
            return BaseType(BaseTy.Tensor)
        elif isinstance(obj, torch.SymInt):
            return BaseType(BaseTy.SymInt)
        elif isinstance(obj, torch.SymBool):
            return BaseType(BaseTy.SymBool)
        elif isinstance(obj, torch.ScriptObject):
            return CustomClassType(obj._type().name())  # type: ignore[attr-defined]
        elif isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                raise AssertionError("list/tuple must be non-empty")
            all_base_tys = [TypeGen.from_example(x) for x in obj]
            if len(set(all_base_tys)) > 1:
                raise RuntimeError(
                    f"Cannot generate schema for a sequence of args of heterogeneous types: {all_base_tys}. "
                    "Consider unpacking the argument and give proper names to them if possible "
                    "instead of using *args."
                )
            return ListType(all_base_tys[0], len(obj))
        tp = type(obj)
        if tp not in TypeGen.convert_to_base_ty:
            raise RuntimeError(f"unsupported type {tp}")
        return BaseType(TypeGen.convert_to_base_ty[tp])


class ReturnGen:
    # [CLASS DOC] Stateless namespace that builds schema Return objects from example values.
    @staticmethod
    def from_example(
        name: str | None, obj: Any, annotation: Annotation | None
    ) -> Return:
        # [FUNCTION DOC] Builds a Return with optional name and annotation, whose type is
        # inferred from the example value obj. Raises on unsupported values.
        return Return(name, TypeGen.from_example(obj), annotation)


class ArgumentGen:
    # [CLASS DOC] Stateless namespace that builds schema Argument objects from example values.
    @staticmethod
    def from_example(
        name: str, obj: Any, default: str | None, annotation: Annotation | None
    ) -> Argument:
        # [FUNCTION DOC] Builds an Argument with the given name, default string and annotation,
        # whose type is inferred from the example value obj. Raises on unsupported values.
        return Argument(
            name, TypeGen.from_example(obj), default=default, annotation=annotation
        )


class FunctionSchemaGen:
    # [CLASS DOC] Stateless namespace that builds a whole FunctionSchema from example values.
    @staticmethod
    def from_example(
        op_name: str,
        example_inputs: tuple[tuple[str, Any], ...],
        example_outputs: tuple[Any, ...],
    ) -> FunctionSchema:
        # [FUNCTION DOC] Builds a FunctionSchema for an operator from example runtime values.
        # op_name: operator name. example_inputs: sequence of (argument name, example value)
        # pairs. example_outputs: sequence of example return values.
        # All inputs become positional arguments without defaults or alias annotations;
        # returns are unnamed. Raises whatever the type inference raises for unsupported values.
        args = []
        for name, inp in example_inputs:
            args.append(ArgumentGen.from_example(name, inp, None, None))
        # ignore the annotations and other attributes for now, we could add more when needed.
        arguments = Arguments(
            tuple(), None, tuple(args), tuple(), None, tuple(), tuple()
        )
        returns = tuple(
            ReturnGen.from_example(None, out, None) for out in example_outputs
        )
        op_name = OperatorName(BaseOperatorName(op_name, False, False, False), "")
        return FunctionSchema(op_name, arguments, returns)
