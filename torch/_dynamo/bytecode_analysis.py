# [MODULE DOC] Bytecode analysis helpers for Dynamo: dead code removal, pointless jump removal,
# line number propagation and cleanup, live-variable analysis, and stack-size analysis, working
# on lists of Instruction objects. Import-time side effects: builds version-dependent module
# constants (terminal opcodes, jump opcodes and names, local and free opcode sets).
"""
This module provides utilities for analyzing and optimizing Python bytecode.
Key functionality includes:
- Dead code elimination
- Jump instruction optimization
- Stack size analysis and verification
- Live variable analysis
- Line number propagation and cleanup
- Exception table handling for Python 3.11+

The utilities in this module are used to analyze and transform bytecode
for better performance while maintaining correct semantics.
"""

import bisect
import dataclasses
import dis
import itertools
import sys
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    # TODO(lucaskabela): consider moving Instruction into this file
    # and refactoring in callsite; that way we don't have to guard this import
    from .bytecode_transformation import Instruction

TERMINAL_OPCODES = {
    dis.opmap["RETURN_VALUE"],
    dis.opmap["JUMP_FORWARD"],
    dis.opmap["RAISE_VARARGS"],
    # TODO(jansel): double check exception handling
}
TERMINAL_OPCODES.add(dis.opmap["RERAISE"])
if sys.version_info >= (3, 11):
    TERMINAL_OPCODES.add(dis.opmap["JUMP_BACKWARD"])
    TERMINAL_OPCODES.add(dis.opmap["JUMP_FORWARD"])
else:
    TERMINAL_OPCODES.add(dis.opmap["JUMP_ABSOLUTE"])

if (3, 12) <= sys.version_info < (3, 14):
    TERMINAL_OPCODES.add(dis.opmap["RETURN_CONST"])
if sys.version_info >= (3, 13):
    TERMINAL_OPCODES.add(dis.opmap["JUMP_BACKWARD_NO_INTERRUPT"])
JUMP_OPCODES = set(dis.hasjrel + dis.hasjabs)
JUMP_OPNAMES = {dis.opname[opcode] for opcode in JUMP_OPCODES}
HASLOCAL = set(dis.haslocal)
HASFREE = set(dis.hasfree)

stack_effect = dis.stack_effect


def get_indexof(insts: list["Instruction"]) -> dict["Instruction", int]:
    # [FUNCTION DOC] Maps each instruction object (by identity) to its list index. Raises an
    # error if the same instruction object appears twice. Pure otherwise.
    """
    Get a mapping from instruction memory address to index in instruction list.
    Additionally checks that each instruction only appears once in the list.
    """
    # pyrefly: ignore [implicit-any]
    indexof = {}
    for i, inst in enumerate(insts):
        if inst in indexof:
            raise AssertionError(
                f"duplicate instruction at index {i} in instruction list"
            )
        indexof[inst] = i
    return indexof


def remove_dead_code(instructions: list["Instruction"]) -> list["Instruction"]:
    # [FUNCTION DOC] Returns the instructions reachable from the first one, following fall-through,
    # jump targets and exception handler targets, stopping at terminal opcodes. On Python 3.11+
    # also shrinks each live instruction's exception table entry so its start and end are live
    # instructions (mutating the entries). Raises an error on jumps without targets or
    # inconsistent entries.
    """Dead code elimination"""
    indexof = get_indexof(instructions)
    live_code = set()

    def find_live_code(start: int) -> None:
        # [FUNCTION DOC] HELPER: Marks instructions live from index start onward, recursing into
        # jump and exception targets; stops at an already-live index or a terminal opcode.
        for i in range(start, len(instructions)):
            if i in live_code:
                return
            live_code.add(i)
            inst = instructions[i]
            if inst.exn_tab_entry:
                find_live_code(indexof[inst.exn_tab_entry.target])
            if inst.opcode in JUMP_OPCODES:
                if inst.target is None:
                    raise AssertionError(
                        f"jump instruction {inst.opname} has no target"
                    )
                find_live_code(indexof[inst.target])
            if inst.opcode in TERMINAL_OPCODES:
                return

    find_live_code(0)

    # change exception table entries if start/end instructions are dead
    # assumes that exception table entries have been propagated,
    # e.g. with bytecode_transformation.propagate_inst_exn_table_entries,
    # and that instructions with an exn_tab_entry lies within its start/end.
    if sys.version_info >= (3, 11):
        live_idx = sorted(live_code)
        for i, inst in enumerate(instructions):
            if i in live_code and inst.exn_tab_entry:
                # find leftmost live instruction >= start
                start_idx = bisect.bisect_left(
                    live_idx, indexof[inst.exn_tab_entry.start]
                )
                if start_idx >= len(live_idx):
                    raise AssertionError(
                        "no live instruction found at or after exn_tab_entry start"
                    )
                # find rightmost live instruction <= end
                end_idx = (
                    bisect.bisect_right(live_idx, indexof[inst.exn_tab_entry.end]) - 1
                )
                if end_idx < 0:
                    raise AssertionError(
                        "no live instruction found at or before exn_tab_entry end"
                    )
                if not (live_idx[start_idx] <= i <= live_idx[end_idx]):
                    raise AssertionError(
                        f"instruction {i} not within live range "
                        f"[{live_idx[start_idx]}, {live_idx[end_idx]}]"
                    )
                inst.exn_tab_entry.start = instructions[live_idx[start_idx]]
                inst.exn_tab_entry.end = instructions[live_idx[end_idx]]

    return [inst for i, inst in enumerate(instructions) if i in live_code]


def remove_pointless_jumps(instructions: list["Instruction"]) -> list["Instruction"]:
    # [FUNCTION DOC] Returns a new list without absolute jumps whose target is the immediately
    # following instruction.
    # KNOWN ISSUE: only matches JUMP_ABSOLUTE, which does not exist in Python 3.11+, so it is a
    # no-op on newer versions.
    """Eliminate jumps to the next instruction"""
    pointless_jumps = {
        id(a)
        for a, b in itertools.pairwise(instructions)
        if a.opname == "JUMP_ABSOLUTE" and a.target is b
    }
    return [inst for inst in instructions if id(inst) not in pointless_jumps]


def propagate_line_nums(instructions: list["Instruction"]) -> None:
    # [FUNCTION DOC] Mutates instructions so each has starts_line set to the most recent line
    # number seen (carrying forward; stays None before any line is seen).
    """Ensure every instruction has line number set in case some are removed"""
    cur_line_no = None

    def populate_line_num(inst: "Instruction") -> None:
        # [FUNCTION DOC] HELPER: Updates the running line number from the instruction if it
        # starts a line, then stores the running value on the instruction.
        nonlocal cur_line_no
        if inst.starts_line:
            cur_line_no = inst.starts_line

        inst.starts_line = cur_line_no

    for inst in instructions:
        populate_line_num(inst)


def remove_extra_line_nums(instructions: list["Instruction"]) -> None:
    # [FUNCTION DOC] Mutates instructions, clearing starts_line when it repeats the previous
    # recorded line number, so only line changes remain.
    """Remove extra starts line properties before packing bytecode"""

    cur_line_no = None

    def remove_line_num(inst: "Instruction") -> None:
        # [FUNCTION DOC] HELPER: Clears the instruction's starts_line if equal to the running
        # line number; otherwise records it as the running value; None is left alone.
        nonlocal cur_line_no
        if inst.starts_line is None:
            return
        elif inst.starts_line == cur_line_no:
            inst.starts_line = None
        else:
            cur_line_no = inst.starts_line

    for inst in instructions:
        remove_line_num(inst)


@dataclasses.dataclass
class ReadsWrites:
    # [CLASS DOC] HELPER: Accumulator for liveness analysis holding sets of variable names read,
    # written, and visited instruction start indices.
    reads: set[Any]
    writes: set[Any]
    visited: set[Any]


def livevars_analysis(
    instructions: list["Instruction"], instruction: "Instruction"
) -> set[Any]:
    # [FUNCTION DOC] Returns the set of local or free variable names that may be read starting
    # from the given instruction. Walks forward along the straight-line path (the "must" state,
    # where earlier writes kill later reads) and explores jump and exception targets in a "may"
    # state; stops at terminal opcodes. Raises an error for unhandled local/free opcodes or
    # jumps lacking a target. Pure.
    # KNOWN ISSUE: after a jump, the rest of the current walk continues in the may state, but
    # writes are checked only against the must state.
    indexof = get_indexof(instructions)
    must = ReadsWrites(set(), set(), set())
    may = ReadsWrites(set(), set(), set())

    def walk(state: ReadsWrites, start: int) -> None:
        # [FUNCTION DOC] HELPER: Scans from start (once per start per state) recording reads
        # and writes of local/free variables in the given state, recursing into targets.
        if start in state.visited:
            return
        state.visited.add(start)

        for i in range(start, len(instructions)):
            inst = instructions[i]
            if inst.opcode in HASLOCAL or inst.opcode in HASFREE:
                if "LOAD" in inst.opname or "DELETE" in inst.opname:
                    if inst.argval not in must.writes:
                        state.reads.add(inst.argval)
                elif "STORE" in inst.opname:
                    state.writes.add(inst.argval)
                elif inst.opname == "MAKE_CELL":
                    pass
                else:
                    raise NotImplementedError(f"unhandled {inst.opname}")
            if inst.exn_tab_entry:
                walk(may, indexof[inst.exn_tab_entry.target])
            if inst.opcode in JUMP_OPCODES:
                if inst.target is None:
                    raise AssertionError(
                        f"jump instruction {inst.opname} has no target"
                    )
                walk(may, indexof[inst.target])
                state = may
            if inst.opcode in TERMINAL_OPCODES:
                return

    walk(must, indexof[instruction])
    return must.reads | may.reads


@dataclasses.dataclass
class FixedPointBox:
    # [CLASS DOC] HELPER: Shared mutable flag; true while the stack-size analysis has not changed
    # any bound in the current pass.
    value: bool = True


@dataclasses.dataclass
class StackSize:
    # [CLASS DOC] HELPER: Lower and upper bound of the stack depth at one instruction, plus the
    # shared fixed-point flag that is cleared whenever a bound changes.
    low: int | float
    high: int | float
    fixed_point: FixedPointBox

    def zero(self) -> None:
        # [FUNCTION DOC] HELPER: Sets both bounds to zero (entry point) and clears the flag.
        self.low = 0
        self.high = 0
        self.fixed_point.value = False

    def offset_of(self, other: "StackSize", n: int) -> None:
        # [FUNCTION DOC] HELPER: Widens these bounds to include the other's bounds shifted by n;
        # clears the fixed-point flag if anything changed.
        prior = (self.low, self.high)
        self.low = min(self.low, other.low + n)
        self.high = max(self.high, other.high + n)
        if (self.low, self.high) != prior:
            self.fixed_point.value = False

    def exn_tab_jump(self, depth: int) -> None:
        # [FUNCTION DOC] HELPER: Widens the bounds to include an exception handler entry depth;
        # clears the fixed-point flag if anything changed.
        prior = (self.low, self.high)
        self.low = min(self.low, depth)
        self.high = max(self.high, depth)
        if (self.low, self.high) != prior:
            self.fixed_point.value = False


def stacksize_analysis(instructions: list["Instruction"]) -> int | float:
    # [FUNCTION DOC] Computes the maximum stack depth needed by the instruction list. Iterates
    # (at most 100 passes) propagating depth bounds across fall-through, jump and exception
    # handler edges until a fixed point. Returns the highest upper bound. Raises an error for an
    # empty list, a missing next instruction or jump target, failure to converge, or a negative
    # lower bound. Pure.
    if not instructions:
        raise AssertionError("instructions list must not be empty")
    fixed_point = FixedPointBox()
    stack_sizes = {
        inst: StackSize(float("inf"), float("-inf"), fixed_point)
        for inst in instructions
    }
    stack_sizes[instructions[0]].zero()

    for _ in range(100):
        if fixed_point.value:
            break
        fixed_point.value = True

        for inst, next_inst in zip(instructions, instructions[1:] + [None]):
            stack_size = stack_sizes[inst]
            if inst.opcode not in TERMINAL_OPCODES:
                if next_inst is None:
                    raise AssertionError(f"missing next inst: {inst}")
                eff = stack_effect(inst.opcode, inst.arg, jump=False)
                stack_sizes[next_inst].offset_of(stack_size, eff)
            if inst.opcode in JUMP_OPCODES:
                if inst.target is None:
                    raise AssertionError(f"missing target: {inst}")
                stack_sizes[inst.target].offset_of(
                    stack_size, stack_effect(inst.opcode, inst.arg, jump=True)
                )
            if inst.exn_tab_entry:
                # see https://github.com/python/cpython/blob/3.11/Objects/exception_handling_notes.txt
                # on why depth is computed this way.
                depth = inst.exn_tab_entry.depth + int(inst.exn_tab_entry.lasti) + 1
                stack_sizes[inst.exn_tab_entry.target].exn_tab_jump(depth)

    low = min(x.low for x in stack_sizes.values())
    high = max(x.high for x in stack_sizes.values())

    if not fixed_point.value:
        raise AssertionError("failed to reach fixed point")
    if low < 0:
        raise AssertionError(f"stack size analysis produced negative low value: {low}")
    return high
