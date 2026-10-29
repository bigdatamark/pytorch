# [MODULE DOC] Small helpers for the gradual-type constraint generator: functions that
# create fresh tensor, dimension and boolean variables from an integer counter, and
# one that builds natural-number constraints. No import-time side effects.
from torch.fx.experimental.migrate_gradual_types.constraint import (
    BinConstraintD,
    BVar,
    DVar,
    TVar,
)
from torch.fx.experimental.migrate_gradual_types.operation import op_leq


def gen_tvar(curr: int) -> tuple[TVar, int]:
    # [FUNCTION DOC] Increments the counter and returns a fresh tensor variable
    # named by the new counter value, together with the new counter. No side effects.
    """
    Generate a tensor variable
    :param curr: The current counter
    :return: a tensor variable and the updated counter
    """
    curr += 1
    return TVar(curr), curr


def gen_dvar(curr: int) -> tuple[DVar, int]:
    # [FUNCTION DOC] Increments the counter and returns a fresh dimension variable
    # named by the new counter value, together with the new counter.
    """
    Generate a dimension variable
    :param curr: the current counter
    :return: a dimension variable and an updated counter
    """
    curr += 1
    return DVar(curr), curr


def gen_bvar(curr: int) -> tuple[BVar, int]:
    # [FUNCTION DOC] Increments the counter and returns a fresh boolean variable
    # named by the new counter value, together with the new counter.
    """
    Generate a boolean variable
    :param curr: the current counter
    :return: a boolean variable and an updated counter
    """
    curr += 1
    return BVar(curr), curr


def gen_tensor_dims(n: int, curr: int) -> tuple[list[DVar], int]:
    # [FUNCTION DOC] Creates n fresh dimension variables in order, threading the counter
    # through each creation. Returns the list and the final counter. n of zero gives an
    # empty list and an unchanged counter.
    """
    Generate a list of tensor dimensions
    :param n:  the number of dimensions
    :param curr: the current counter
    :return: a list of dimension variables and an updated counter
    """
    dims = []
    for _ in range(n):
        dvar, curr = gen_dvar(curr)
        dims.append(dvar)
    return dims, curr


def gen_nat_constraints(list_of_dims: list[DVar]) -> list[BinConstraintD]:
    # [FUNCTION DOC] Returns one constraint per dimension variable stating that zero is
    # less than or equal to that dimension (non-negativity), in input order.
    """
    Generate natural number constraints for dimensions
    """
    return [BinConstraintD(0, d, op_leq) for d in list_of_dims]
