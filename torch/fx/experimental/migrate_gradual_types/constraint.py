# [MODULE DOC]
# Data classes for the constraint language used by the gradual type migration
# (type inference over FX graphs). Defines the Constraint hierarchy (logical
# Conj/Disj, constants T/F, binary constraints over tensors or dimensions, and
# operator-specific constraints such as greatest upper bound, reshape, index select,
# transpose, get item, conv, max pool, broadcasting and product), the variable
# classes TVar, DVar and BVar, the _Operand type alias, and the predicates
# is_algebraic_expression, is_bool_expr and is_dim. Every class provides structural
# equality and a readable repr. No import-time side effects.
# KNOWN ISSUE: classes define equality without a hash method, so instances are
# unhashable.
from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias


if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "ApplyBroadcasting",
    "BinConstraintD",
    "BinConstraintT",
    "BinaryConstraint",
    "BVar",
    "CalcConv",
    "CalcMaxPool",
    "CalcProduct",
    "CanReshape",
    "Conj",
    "Constraint",
    "DGreatestUpperBound",
    "Disj",
    "DVar",
    "F",
    "GetItem",
    "GetItemTensor",
    "IndexSelect",
    "Prod",
    "T",
    "TGreatestUpperBound",
    "Transpose",
    "TVar",
    "is_algebraic_expression",
    "is_bool_expr",
    "is_dim",
]

from torch.fx.experimental.migrate_gradual_types.operation import (
    op_add,
    op_div,
    op_eq,
    op_gt,
    op_lt,
    op_mod,
    op_mul,
    op_neq,
    op_sub,
)
from torch.fx.tensor_type import _DynType, Dyn, TensorType


class Constraint:
    # [CLASS DOC] Common base class of all constraint objects; carries no state.
    pass


class Conj(Constraint):
    # [CLASS DOC] Logical conjunction of constraints; holds the list of conjuncts.
    # KNOWN ISSUE: the attribute is spelled conjucts (typo) and used consistently.
    def __init__(self, conjuncts: Sequence[Constraint]) -> None:
        # [FUNCTION DOC] Copies the given sequence of constraints into a list.
        """
        :param conjuncts: Conjunction of constraints
        """
        self.conjucts = list(conjuncts)

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is a Conj with an equal list of conjuncts.
        if isinstance(other, Conj):
            return self.conjucts == other.conjucts
        else:
            return False

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text form: And followed by the list of conjuncts.
        return f"And({self.conjucts})"


class Disj(Constraint):
    # [CLASS DOC] Logical disjunction of constraints; holds the list of disjuncts.
    def __init__(self, disjuncts: Sequence[Constraint]) -> None:
        # [FUNCTION DOC] Copies the given sequence of constraints into a list.
        """
        :param disjuncts: Disjunction of constraints
        """
        self.disjuncts = list(disjuncts)

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is a Disj with an equal list of disjuncts.
        if isinstance(other, Disj):
            return self.disjuncts == other.disjuncts
        else:
            return False

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text form: Or followed by the list of disjuncts.
        return f"Or({self.disjuncts})"


class Prod(Constraint):
    # [CLASS DOC] Algebraic expression: the product of a list of dimensions
    # (dimension variables, integers or the dynamic type).
    def __init__(self, products: Sequence[DVar | int | _DynType]) -> None:
        # [FUNCTION DOC] Copies the given dimensions into a list of factors.
        """
        :param products: lists of dimensions to multiply
        """
        self.products = list(products)

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is a Prod with an equal list of factors.
        if isinstance(other, Prod):
            return self.products == other.products
        else:
            return False

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text form: Product followed by the list of factors.
        return f"Product({self.products})"


class T(Constraint):
    # [CLASS DOC] The constant true constraint; stateless, all instances are equal.
    """
    True
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] No-op initializer.
        pass

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal to any instance of T.
        return isinstance(other, T)

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text form: True.
        return "True"


class F(Constraint):
    # [CLASS DOC] The constant false constraint; stateless, all instances are equal.
    """
    False
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] No-op initializer.
        pass

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal to any instance of F.
        return isinstance(other, F)

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text form: False.
        return "False"


class BinaryConstraint(Constraint):
    # [CLASS DOC] Generic binary constraint holding a left operand, a right operand
    # and an operation name (one of the operation module constants). Base class of
    # BinConstraintT and BinConstraintD.
    """
    Represents all binary operations
    """

    def __init__(self, lhs: _Operand, rhs: _Operand, op: str | None) -> None:
        # [FUNCTION DOC] Stores lhs, rhs and op without validation.
        """
        :param lhs: lhs of the constraint
        :param rhs: rhs of the constraint
        :param op: string representing the operation
        """
        self.lhs = lhs
        self.rhs = rhs
        self.op = op

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is any BinaryConstraint (subclass included)
        # with equal lhs, rhs and op; the exact subclass is not compared.
        if isinstance(other, BinaryConstraint):
            return (
                self.lhs == other.lhs and self.rhs == other.rhs and self.op == other.op
            )
        else:
            return False

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text form: parenthesized lhs, op, rhs.
        return f"({self.lhs} {self.op} {self.rhs})"


class BinConstraintT(BinaryConstraint):
    # [CLASS DOC] Binary constraint whose operands are tensor-level: tensor
    # variables, tensor types, integers or the dynamic type.
    """
    Binary constraints about tensors
    """

    def __init__(self, lhs: _Operand, rhs: _Operand, op: str | None) -> None:
        # [FUNCTION DOC] Validates that both operands are TVar, TensorType, int or
        # Dyn, raising AssertionError otherwise, then stores them via the base class.
        if not (
            (isinstance(lhs, (TVar, TensorType, int)) or lhs == Dyn)
            and (isinstance(rhs, (TVar, TensorType, int)) or rhs == Dyn)
        ):
            raise AssertionError(f"Invalid types: lhs={type(lhs)}, rhs={type(rhs)}")
        super().__init__(lhs, rhs, op)


class BinConstraintD(BinaryConstraint):
    # [CLASS DOC] Binary constraint whose operands are dimension-level: algebraic
    # expressions, dimensions or boolean expressions.
    """
    Binary constraints about dimensions
    """

    def __init__(self, lhs: _Operand, rhs: _Operand, op: str | None) -> None:
        # [FUNCTION DOC] Validates each operand is an algebraic expression, a
        # dimension or a boolean expression, raising AssertionError otherwise, then
        # stores them via the base class.
        if not (is_algebraic_expression(lhs) or is_dim(lhs) or is_bool_expr(lhs)):
            raise AssertionError(f"Invalid lhs type: {type(lhs)}")
        if not (is_algebraic_expression(rhs) or is_dim(rhs) or is_bool_expr(rhs)):
            raise AssertionError(f"Invalid rhs type: {type(rhs)}")

        super().__init__(lhs, rhs, op)


class TGreatestUpperBound(Constraint):
    # [CLASS DOC] Constraint that res is the greatest upper bound (with respect to
    # the dynamic type) of two tensor operands; stores res, rhs1 and rhs2.
    """
    Greatest Upper bound for tensors with dynamic type
    """

    def __init__(self, res: TVar, rhs1: TVar, rhs2: TVar) -> None:
        # [FUNCTION DOC] Stores the result variable and both operands unvalidated.
        """
        :param res: tensor variable that stores the result of the output
        :param rhs1: tensor or tensor variable
        :param rhs2: tensor or tensor variable
        """
        self.res = res
        self.rhs1 = rhs1
        self.rhs2 = rhs2

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text form: res equals rhs1 joined (starred join symbol) with rhs2.
        return f"{self.res} = {self.rhs1}\u2294*{self.rhs2}"

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is a TGreatestUpperBound with equal fields.
        if isinstance(other, TGreatestUpperBound):
            return (
                self.res == other.res
                and self.rhs1 == other.rhs1
                and self.rhs2 == other.rhs2
            )
        else:
            return False


class DGreatestUpperBound(Constraint):
    # [CLASS DOC] Constraint that res is the greatest upper bound of two dimensions;
    # stores res, rhs1 and rhs2, all of which must be dimensions.
    """
    Greatest Upper bound for dimensions
    """

    def __init__(
        self,
        res: DVar | int | _DynType,
        rhs1: DVar | int | _DynType,
        rhs2: DVar | int | _DynType,
    ) -> None:
        # [FUNCTION DOC] Raises AssertionError unless res, rhs1 and rhs2 are each a
        # dimension variable, integer or the dynamic type; then stores them.
        """
        :param res: Dimension variable to store the result
        :param rhs1: dimension variable 1
        :param rhs2: dimension variable 2
        """
        if not is_dim(res):
            raise AssertionError(f"Expected dimension for res, got {type(res)}")
        if not is_dim(rhs1):
            raise AssertionError(f"Expected dimension for rhs1, got {type(rhs1)}")
        if not is_dim(rhs2):
            raise AssertionError(f"Expected dimension for rhs2, got {type(rhs2)}")

        self.res = res
        self.rhs1 = rhs1
        self.rhs2 = rhs2

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text form: res equals rhs1 joined (join symbol) with rhs2.
        return f"{self.res} = {self.rhs1}\u2294{self.rhs2}"

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is a DGreatestUpperBound with equal fields.
        if isinstance(other, DGreatestUpperBound):
            return (
                self.res == other.res
                and self.rhs1 == other.rhs1
                and self.rhs2 == other.rhs2
            )
        else:
            return False


class CanReshape(Constraint):
    # [CLASS DOC] Constraint that a tensor variable src can be reshaped to the
    # concrete tensor type target; stores both.
    """
    can_reshape constraint
    """

    def __init__(self, src: TVar, target: TensorType) -> None:
        # [FUNCTION DOC] Stores src and target without validation.
        """
        :param src: tensor variable
        :param target: tensor
        """
        self.src = src
        self.target = target

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text form: can-reshape applied to src and target.
        return f"can-reshape({self.src}, {self.target})"

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is a CanReshape with equal src and target.
        if isinstance(other, CanReshape):
            return self.src == other.src and self.target == other.target
        else:
            return False


class IndexSelect(Constraint):
    # [CLASS DOC] Constraint describing index_select on a tensor of known rank
    # tensor_size: output is input_var with the dimension at index replaced by
    # dim_replace. Stores input_var, tensor_size, dim_replace, index and output.
    def __init__(
        self,
        tensor_size: int,
        input_var: TVar,
        dim_replace: DVar | _DynType,
        index: int,
        output: TVar,
    ) -> None:
        # [FUNCTION DOC] Raises AssertionError unless input_var and output are TVar,
        # dim_replace is a DVar or Dyn, and index is an int; then stores all fields.
        """
        Args:
            input_var: input to index_select
            tensor_size: tensor size we are considering
            dim_replace: the dimension of the output at "index"
            index: location of the dimensions to replace in the input
            output: variable to store the result
        """
        if not isinstance(input_var, TVar):
            raise AssertionError(f"Expected TVar, got {type(input_var)}")
        if not isinstance(output, TVar):
            raise AssertionError(f"Expected TVar, got {type(output)}")
        if not (isinstance(dim_replace, DVar) or dim_replace == Dyn):
            raise AssertionError(f"Expected DVar or Dyn, got {type(dim_replace)}")
        if not isinstance(index, int):
            raise AssertionError(f"Expected int, got {type(index)}")

        self.input_var = input_var
        self.tensor_size = tensor_size
        self.dim_replace = dim_replace
        self.index = index
        self.output = output

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text form: output equals IndexSelect with input, size, dim, index.
        return (
            f" {self.output} = "
            f"IndexSelect({self.input_var}, "
            f"tensor_size: {self.tensor_size}, "
            f"{self.dim_replace}, "
            f"{self.index})"
        )

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is an IndexSelect with all five fields equal.
        if isinstance(other, IndexSelect):
            return (
                self.tensor_size == other.tensor_size
                and self.dim_replace == other.dim_replace
                and self.index == other.index
                and self.output == other.output
                and self.input_var == other.input_var
            )
        else:
            return False


class Transpose(Constraint):
    # [CLASS DOC] Constraint that output is input_var (of rank tensor_size) with the
    # dimensions at index1 and index2 swapped. Stores all five fields.
    def __init__(
        self, tensor_size: int, input_var: TVar, index1: int, index2: int, output: TVar
    ) -> None:
        # [FUNCTION DOC] Raises AssertionError unless input_var and output are TVar
        # and index1 and index2 are ints; then stores all fields.
        """
        Args:
            tensor_size: current tensor size
            input_var: variable to hold input
            index1: dimension 1
            index2: dimension 2
            output: output that stores result
        """
        if not isinstance(input_var, TVar):
            raise AssertionError(f"Expected TVar, got {type(input_var)}")
        if not isinstance(output, TVar):
            raise AssertionError(f"Expected TVar, got {type(output)}")
        if not isinstance(index1, int):
            raise AssertionError(f"Expected int, got {type(index1)}")
        if not isinstance(index2, int):
            raise AssertionError(f"Expected int, got {type(index2)}")

        self.input_var = input_var
        self.tensor_size = tensor_size
        self.index1 = index1
        self.index2 = index2
        self.output = output

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text form: output equals Transpose with input, size, indices.
        return (
            f" {self.output} = "
            f"Transpose({self.input_var}, "
            f"tensor_size: {self.tensor_size}, "
            f"{self.index1}, "
            f"{self.index2})"
        )

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is a Transpose with all five fields equal.
        if isinstance(other, Transpose):
            return (
                self.tensor_size == other.tensor_size
                and self.index1 == other.index1
                and self.index2 == other.index2
                and self.output == other.output
                and self.input_var == other.input_var
            )
        else:
            return False


class GetItem(Constraint):
    # [CLASS DOC] Constraint that dimension variable res is the dimension at integer
    # position index of input_var, a tensor of rank tensor_size.
    def __init__(
        self, tensor_size: int, index: int, res: DVar, input_var: TVar
    ) -> None:
        # [FUNCTION DOC] Raises AssertionError unless res is a DVar; stores all fields
        # (other arguments are not validated).
        """
        Constraint for getting item given a tensor size
        :param tensor_size: actual number
        :param index: actual number representing the index
        :param res: dimension variable to carry the item we get
        :param input_var: a tensor variable from which we will get item
        """
        if not isinstance(res, DVar):
            raise AssertionError(f"Expected DVar, got {type(res)}")

        self.res = res
        self.tensor_size = tensor_size
        self.index = index
        self.input_var = input_var

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text form: res equals GetItem with input, size and index.
        return f" {self.res} = GetItem({self.input_var}, tensor_size: {self.tensor_size}, {self.index})"

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is a GetItem with all four fields equal.
        if isinstance(other, GetItem):
            return (
                self.res == other.res
                and self.tensor_size == other.tensor_size
                and self.index == other.index
                and self.input_var == other.input_var
            )
        else:
            return False


class GetItemTensor(Constraint):
    # [CLASS DOC] Constraint for indexing a tensor (rank tensor_size) with a tuple of
    # None / slice entries, producing tensor variable res. Stores all four fields.
    def __init__(
        self,
        tensor_size: int,
        index_tuple: tuple[None | slice, ...],
        res: TVar,
        input_var: TVar,
    ) -> None:
        # [FUNCTION DOC] Raises AssertionError unless res is a TVar; stores all fields
        # (other arguments are not validated).
        """
        Constraint for getting item given a tensor size
        However, when the argument is a tuple, we will
        expect a tensor
        :param tensor_size: actual number representing the rank
        :param index_tuple: tuple for indexing
        :param res: tensor variable to carry the item we get
        :param input_var: a tensor variable from which we will get item
        """
        if not isinstance(res, TVar):
            raise AssertionError(f"Expected TVar, got {type(res)}")

        self.res = res
        self.tensor_size = tensor_size
        self.index_tuple = index_tuple
        self.input_var = input_var

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text form: res equals GetItemT with input, size and index tuple.
        return f" {self.res} = GetItemT({self.input_var}, tensor_size: {self.tensor_size}, {self.index_tuple})"

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is a GetItemTensor with all four fields equal.
        if isinstance(other, GetItemTensor):
            return (
                self.res == other.res
                and self.tensor_size == other.tensor_size
                and self.index_tuple == other.index_tuple
                and self.input_var == other.input_var
            )
        else:
            return False


class CalcConv(Constraint):
    # [CLASS DOC] Constraint that conv_result is the output type of a 2D convolution
    # over input_var with c_out output channels and the given kernel, padding, stride
    # and dilation. matching_constraint holds the dimension variables used by the
    # constraint transformation. Pure data holder.
    def __init__(
        self,
        conv_result: TVar,
        input_var: TVar,
        c_out: int,
        kernel: int | tuple[int, int],
        padding: int | tuple[int, int],
        stride: int | tuple[int, int],
        dilation: int | tuple[int, int],
        matching_constraint_vars: list[DVar],
    ) -> None:
        # [FUNCTION DOC] Stores all arguments without validation; the list of
        # matching dimension variables is stored as matching_constraint.
        """
        :param conv_result: the convolution result
        :param input_var: input to convolution
        :param c_out: output channel type
        :param kernel: kernel tuple
        """
        self.conv_result = conv_result
        self.input_var = input_var
        self.c_out = c_out
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.matching_constraint = matching_constraint_vars

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text form: result equals calc-conv of input, c_out, kernel,
        # padding, stride and dilation (matching_constraint is omitted).
        return (
            f"{self.conv_result} ="
            f" calc-conv({self.input_var},"
            f" {self.c_out}, {self.kernel}, "
            f"{self.padding}, {self.stride},"
            f" {self.dilation})"
        )

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is a CalcConv with all fields equal.
        if isinstance(other, CalcConv):
            return (
                self.conv_result == other.conv_result
                and self.input_var == other.input_var
                and self.c_out == other.c_out
                and self.kernel == other.kernel
                and self.padding == other.padding
                and self.stride == other.stride
                and self.dilation == other.dilation
                and self.matching_constraint == other.matching_constraint
            )
        else:
            return False


class CalcMaxPool(Constraint):
    # [CLASS DOC] Constraint that maxpool_result is the output type of a 2D max pool
    # over input_var with the given kernel, padding, stride and dilation.
    # matching_constraint holds the dimension variables used by the constraint
    # transformation. Pure data holder.
    def __init__(
        self,
        maxpool_result: TVar,
        input_var: TVar,
        kernel: int | tuple[int, int],
        padding: int | tuple[int, int],
        stride: int | tuple[int, int],
        dilation: int | tuple[int, int],
        matching_constraint_vars: list[DVar],
    ) -> None:
        # [FUNCTION DOC] Stores all arguments without validation; the list of
        # matching dimension variables is stored as matching_constraint.
        # KNOWN ISSUE: the docstring says input_var is the input to convolution
        # (copy of the CalcConv text); it is the max pool input.
        """
        :param maxpool_result: the result of maxpool
        :param input_var: input to convolution
        :param kernel: kernel tuple
        """
        self.maxpool_result = maxpool_result
        self.input_var = input_var
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.matching_constraint = matching_constraint_vars

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text form: result equals calc-maxpool of input, kernel,
        # padding, stride and dilation (matching_constraint is omitted).
        return (
            f"{self.maxpool_result} ="
            f" calc-maxpool({self.input_var},"
            f"  {self.kernel}, "
            f"{self.padding}, {self.stride},"
            f" {self.dilation})"
        )

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is a CalcMaxPool with all fields equal.
        if isinstance(other, CalcMaxPool):
            return (
                self.maxpool_result == other.maxpool_result
                and self.input_var == other.input_var
                and self.kernel == other.kernel
                and self.padding == other.padding
                and self.stride == other.stride
                and self.dilation == other.dilation
                and self.matching_constraint == other.matching_constraint
            )
        else:
            return False


class ApplyBroadcasting(Constraint):
    # [CLASS DOC] Constraint that res1 and res2 are input1 and input2 after
    # broadcasting them against each other. Stores all four tensor variables.
    def __init__(self, res1: TVar, res2: TVar, input1: TVar, input2: TVar) -> None:
        # [FUNCTION DOC] Stores the two result and two input variables unvalidated.
        """
        :param res1: resulting tensor 1
        :param res2: resulting tensor 2
        :param input1: tensor variable 1
        :param input2: tensor variable 2
        """
        self.res1 = res1
        self.res2 = res2
        self.input1 = input1
        self.input2 = input2

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is an ApplyBroadcasting with equal fields.
        if isinstance(other, ApplyBroadcasting):
            return (
                self.res1 == other.res1
                and self.res2 == other.res2
                and self.input1 == other.input1
                and self.input2 == other.input2
            )
        else:
            return False

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text form: both results equal apply-broadcasting of both inputs.
        return (
            f"{self.res1}, {self.res2} ="
            f" apply-broadcasting({self.input1},"
            f" {self.input2})"
        )


class CalcProduct(Constraint):
    # [CLASS DOC] Constraint for flatten: flattened is the tensor obtained by
    # multiplying the dimensions of dims_to_flatten between positions start and end,
    # with the dynamic type propagated. Stores all four fields.
    """
    Given correct dimensions, calculate the product for flatten accounting for Dyn
    """

    def __init__(
        self, start: int, end: int, flattened: TVar, dims_to_flatten: list[DVar]
    ) -> None:
        # [FUNCTION DOC] Raises AssertionError unless dims_to_flatten is a list,
        # flattened is a TVar, and start and end are ints; then stores all fields.
        """
        :param start: start index
        :param end: end index
        :param flattened: variable to store the product
        :param dims_to_flatten: the type which we will flatten
        """
        if not isinstance(dims_to_flatten, list):
            raise AssertionError(f"Expected list, got {type(dims_to_flatten)}")
        if not isinstance(flattened, TVar):
            raise AssertionError(f"Expected TVar, got {type(flattened)}")
        if not isinstance(start, int):
            raise AssertionError(f"Expected int, got {type(start)}")
        if not isinstance(end, int):
            raise AssertionError(f"Expected int, got {type(end)}")

        self.start = start
        self.end = end
        self.dims_to_flatten = dims_to_flatten
        self.flattened = flattened

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is a CalcProduct with all fields equal.
        if isinstance(other, CalcProduct):
            return (
                self.start == other.start
                and self.end == other.end
                and self.dims_to_flatten == other.dims_to_flatten
                and self.flattened == other.flattened
            )

        else:
            return False

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text form: flattened equals CalcProduct with start, end, dims.
        return f"{self.flattened} = CalcProduct({self.start}, {self.end}, {self.dims_to_flatten})"


class TVar:
    # [CLASS DOC] Tensor variable identified by an integer id; not a Constraint
    # subclass. Two TVars are equal when their ids are equal.
    """
    Tensor variable with no tensor constructor
    """

    def __init__(self, tvar: int) -> None:
        # [FUNCTION DOC] Stores the integer id of the variable.
        """
        :param tvar: tensor variable
        """
        self.tvar = tvar

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text form: TV with the id.
        return f"TV({self.tvar})"

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is a TVar with the same id.
        if isinstance(other, TVar):
            return self.tvar == other.tvar
        else:
            return False


class DVar:
    # [CLASS DOC] Dimension variable identified by an integer id c; two DVars are
    # equal when their ids are equal.
    """
    Dimension variable
    """

    def __init__(self, c: int) -> None:
        # [FUNCTION DOC] Stores the id. KNOWN ISSUE: the docstring says character or
        # number, but the annotation and all uses are integers.
        """
        :param c: character or number
        """
        self.c = c

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text form: DV with the id.
        return f"DV({self.c})"

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is a DVar with the same id.
        if isinstance(other, DVar):
            return self.c == other.c
        else:
            return False


class BVar:
    # [CLASS DOC] Boolean variable identified by an integer id c; two BVars are
    # equal when their ids are equal.
    """
    Boolean variable
    """

    def __init__(self, c: int) -> None:
        # [FUNCTION DOC] Stores the id. KNOWN ISSUE: the docstring says character or
        # number, but the annotation and all uses are integers.
        """
        :param c: character or number
        """
        self.c = c

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text form: BV with the id.
        return f"BV({self.c})"

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is a BVar with the same id.
        if isinstance(other, BVar):
            return self.c == other.c
        else:
            return False


_Operand: TypeAlias = (
    TVar
    | TensorType
    | DVar
    | int
    | float
    | bool
    | _DynType
    | BinConstraintD
    | Prod
    | BVar
    | Conj
    | Disj
    | None
)


def is_algebraic_expression(constraint: object) -> bool:
    # [FUNCTION DOC] True for a dimension binary constraint whose operation is add,
    # subtract, divide, multiply or modulo, or for a Prod; false otherwise.
    if isinstance(constraint, BinConstraintD):
        return constraint.op in [op_add, op_sub, op_div, op_mul, op_mod]
    else:
        return isinstance(constraint, Prod)


def is_bool_expr(constraint: object) -> bool:
    # [FUNCTION DOC] True for a dimension binary constraint whose operation is
    # greater-than, less-than, not-equal or equal, or for a BVar, Conj or Disj.
    if isinstance(constraint, BinConstraintD):
        return constraint.op in [op_gt, op_lt, op_neq, op_eq]
    else:
        return isinstance(constraint, (BVar, Conj, Disj))


def is_dim(d: object) -> bool:
    # [FUNCTION DOC] True if d is a DVar, an int (booleans included) or the dynamic type.
    return isinstance(d, (DVar, int)) or d == Dyn
