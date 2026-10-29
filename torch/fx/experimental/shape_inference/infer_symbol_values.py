# [MODULE DOC]
# Helpers that parse shape-mismatch error messages and solve for symbol values.
# infer_symbol_values recognizes several known error message patterns (broadcast
# non-singleton, bmm/matmul dimension mismatch, split sizes, invalid reshape) and updates
# the list of symbolic sizes in place using sympy; calculate_value, solve_equation and
# update_equation are its helpers. Import-time: defines three module-level regex
# pattern strings for bracketed, parenthesized and "s<number>" symbol text.
import re
from collections import defaultdict
from typing import Any

import numpy as np
import sympy as sp

import torch


square_brackets_pattern = r"\[([^]]+)\]"
parentheses_pattern = r"\((.*?)\)"
s_pattern = r"s\d+"


def infer_symbol_values(
    symints: list[torch.SymInt | int],
    init_symints: list[torch.SymInt | int],
    symbol_idx_dict: dict[str, int],
    padding_constraints: defaultdict[torch.SymInt, list[sp.Expr | int]],
    constraint: str,
) -> None:
    # [FUNCTION DOC]
    # Interprets an error message (constraint) and updates symints in place. Recognized
    # messages: a non-singleton broadcast mismatch (two parenthesized expressions are
    # equated), a batch matmul first-two-dimension mismatch and a reduction dimension
    # mismatch (bracketed shapes give the two expressions), a split sizes total mismatch
    # (two numbers or expressions in the text), and an invalid reshape for a given input
    # size (solves for one unknown symbol, recording and combining padding constraints so the
    # symbol becomes a multiple of the least common multiple of required factors, via
    # update_equation). Unrecognized messages do nothing. Mutates symints and
    # padding_constraints; may raise IndexError, KeyError or sympy errors on unexpected text.
    # KNOWN ISSUE: the reshape branch removes items from the list while iterating it, which
    # can skip elements; the argument init_symints is only used for that branch.
    if constraint.find("non-singleton") != -1:
        left_expression, right_expression = re.findall(parentheses_pattern, constraint)
        calculate_value(left_expression, right_expression, symints, symbol_idx_dict)

    elif constraint.find("first two dimensions of batch2 tensor to be") != -1:
        matches = re.findall(square_brackets_pattern, constraint)
        left_expression, right_expression = (
            matches[i].split(",")[1].strip() for i in (0, 1)
        )
        calculate_value(left_expression, right_expression, symints, symbol_idx_dict)

    elif constraint.find("a and b must have same reduction dim") != -1:
        matches = re.findall(square_brackets_pattern, constraint)
        left_expression = matches[0].split(",")[1].strip()
        right_expression = matches[1].split(",")[0].strip()
        calculate_value(left_expression, right_expression, symints, symbol_idx_dict)

    elif constraint.find("Split sizes add up to") != -1:
        match_1 = re.search(r"to\s+(.*?)\s+but", constraint)
        extracted_value_1 = match_1.group(1) if match_1 else None
        match_2 = re.search(r"of\s+(.*?)$", constraint)
        extracted_value_2 = match_2.group(1) if match_2 else None
        calculate_value(extracted_value_1, extracted_value_2, symints, symbol_idx_dict)

    elif constraint.find("is invalid for input of size") != -1:
        matches = re.findall(square_brackets_pattern, constraint)
        left_elements = matches[0].split(",")
        left_equation = sp.sympify(1)
        left_num = 1
        right_equation = sp.sympify(constraint.split("size")[1].strip())

        for left_element in left_elements:
            if sp.sympify(left_element) == sp.sympify("-1"):
                continue
            elif sp.sympify(left_element).is_number:
                left_num *= int(left_element)
            else:
                left_equation *= sp.sympify(left_element)
        right_equation = sp.cancel(right_equation / left_equation)

        right_vars = list(right_equation.free_symbols)
        for right_var in right_vars:
            if sp.sympify(right_var) == sp.sympify("s0"):
                right_equation = sp.cancel(right_equation / right_var)
                right_vars.remove(right_var)  # noqa: B909

        var = right_vars[0]
        idx = symbol_idx_dict[str(var)]
        if var not in padding_constraints:
            padding_constraints[var].append(right_equation)
        update_equation(
            symints,
            init_symints,
            padding_constraints,
            padding_constraints[var][0],  # type: ignore[arg-type]
            left_num,
            var,
            idx,
        )


def calculate_value(
    left_expression: str | Any | None,
    right_expression: str | Any | None,
    symints: list[torch.SymInt | int],
    symbol_idx_dict: dict[str, int],
) -> None:
    # [FUNCTION DOC] Solves left minus right for a symbol (see solve_equation), looks up the
    # symbol's index via symbol_idx_dict, and replaces that entry of symints with its
    # previous expression with the symbol substituted by the solved integer value.
    # Mutates symints; raises KeyError for an unknown symbol.
    var, val = solve_equation(left_expression, right_expression)
    idx = symbol_idx_dict[var]
    pre_equation = sp.sympify(f"{symints[idx]}")
    symints[idx] = pre_equation.subs(sp.sympify(var), val)


def solve_equation(
    left_expression: str | Any | None,
    right_expression: str | Any | None,
) -> tuple[str, int]:
    # [FUNCTION DOC] Builds the expression left minus right, picks the first symbol named
    # s followed by digits as the unknown, and returns (symbol name, integer solution).
    # If the expression contains a parenthesized floor-division term, that term is treated
    # as a separate unknown and a two-equation system is solved for the symbol instead.
    # Raises IndexError if no symbol is found; sympy or int conversion errors propagate.
    expression = f"{left_expression} - {right_expression}"
    var = re.findall(s_pattern, expression)[0]
    if re.findall(parentheses_pattern, expression):
        sub_expression = re.findall(parentheses_pattern, expression)[0]
        var, coeff = sub_expression.split("//")
        x = sp.symbols("x")
        sub_equation = sp.sympify(f"{var} - {coeff} * {x}")
        modified_equation = (
            sp.sympify(x) + sp.sympify(expression) - sp.sympify(sub_expression)
        )

        solution = sp.solve((modified_equation, sub_equation), (x, var))
        return (var, int(solution[sp.sympify(var)]))
    else:
        solution = sp.solve(expression, var)
        val = int(solution[0])
        return (var, val)


def update_equation(
    symints: list[torch.SymInt | int],
    init_symints: list[torch.SymInt | int],
    padding_constraints: defaultdict[torch.SymInt, list[sp.Expr | int]],
    init_eq: sp.Expr,
    new_mod_num: int,
    var: torch.SymInt,
    idx: int,
) -> None:
    # [FUNCTION DOC] Appends new_mod_num to the constraint list for var, computes the least
    # common multiple of all recorded multiples (skipping the first entry, which is the
    # initial equation), and sets symints at idx to that multiple times the initial symbol,
    # minus the remainder of the initial equation's numeric term modulo the multiple, if one
    # exists. Mutates padding_constraints and symints.
    padding_constraints[var].append(new_mod_num)
    mod_num = np.lcm.reduce(padding_constraints[var][1:])  # type: ignore[arg-type]
    eq = mod_num * init_symints[idx]
    eq_const = [arg for arg in init_eq.args if arg.is_number]
    if eq_const:
        rem = int(eq_const[0] % mod_num)
        eq -= rem
    symints[idx] = eq
