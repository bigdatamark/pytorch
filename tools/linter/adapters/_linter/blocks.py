# [MODULE DOC]
# Builds the list of Block objects (every def and class) for a PythonFile: finds the keyword
# tokens, computes each block's token range and docstring, links parents and children by
# line containment, marks local blocks and methods, and assigns dotted full names (with
# numeric suffixes for duplicate names at one level). No import-time side effects.
from __future__ import annotations

import token
from typing import TYPE_CHECKING

from . import is_empty
from .block import Block


if TYPE_CHECKING:
    from collections.abc import Sequence

    from .python_file import PythonFile


def blocks(pf: PythonFile) -> list[Block]:
    # [FUNCTION DOC] Returns all def and class blocks of the file in source order. Creates a
    # block per keyword token, then sets each block's parent and children (a later block is a
    # child while it falls within the earlier block's lines), index, is_local (some enclosing
    # block is not a class) and is_method (a function directly inside a class), and finally
    # the dotted full names. Mutates the new blocks only.
    blocks: list[Block] = []

    it = (i for i, t in enumerate(pf.tokens) if t.string in ("class", "def"))
    blocks = [_make_block(pf, i) for i in it]

    for i, parent in enumerate(blocks):
        for j in range(i + 1, len(blocks)):
            if parent.contains(child := blocks[j]):
                child.parent = i
                parent.children.append(j)
            else:
                break

    for i, b in enumerate(blocks):
        b.index = i
        parents = [b]
        while (p := parents[-1].parent) is not None:
            parents.append(blocks[p])
        parents = parents[1:]

        b.is_local = not all(p.is_class for p in parents)
        b.is_method = not b.is_class and bool(parents) and parents[0].is_class

    _add_full_names(blocks, [b for b in blocks if b.parent is None])
    return blocks


def _add_full_names(
    blocks: Sequence[Block], children: Sequence[Block], prefix: str = ""
) -> None:
    # [FUNCTION DOC] HELPER: Recursively sets full_name on the given sibling blocks as prefix
    # plus name, adding a one-based bracketed number when several siblings share a name, then
    # recurses into each block's children with the block's full name and a dot as prefix.
    # Would be trivial except that there can be duplicate names at any level
    dupes: dict[str, list[Block]] = {}
    for b in children:
        dupes.setdefault(b.name, []).append(b)

    for dl in dupes.values():
        for i, b in enumerate(dl):
            suffix = f"[{i + 1}]" if len(dl) > 1 else ""
            b.full_name = prefix + b.name + suffix

    for b in children:
        if kids := [blocks[i] for i in b.children]:
            _add_full_names(blocks, kids, b.full_name + ".")


def _make_block(pf: PythonFile, begin: int) -> Block:
    # [FUNCTION DOC] HELPER: Creates a Block for the def or class keyword token at index
    # begin. The name is the first name token after it. The end is the last non-empty token
    # before the matching dedent of the body (or the ellipsis token for a one-line stub).
    # The docstring is the string token that directly follows the start of the body, else
    # empty. Family fields are left at defaults.
    name = docstring = ""
    end = 0

    for i in range(begin + 1, len(pf.tokens)):
        t = pf.tokens[i]
        if not name and t.type == token.NAME:
            name = t.string
        elif not end:
            if t.type == token.INDENT:
                end = pf.indent_to_dedent[i]
                while is_empty(pf.tokens[end := end - 1]):
                    pass
            elif t.string == "...":
                end = i
        elif t.type == token.STRING:
            docstring = t.string
            break
        elif not is_empty(t):
            break

    category = Block.Category[pf.tokens[begin].string.upper()]
    return Block(
        begin=begin,
        category=category,
        docstring=docstring,
        end=end,
        name=name,
        tokens=pf.tokens,
    )
