# [MODULE DOC]
# Low-level bytecode toolkit for Dynamo. Defines a mutable Instruction type and
# exception-table entry type, factories for version-specific instructions (calls, swaps,
# rotations, null pushes, slices), the pipeline that converts code objects to Instruction
# lists (cleaned_instructions, virtualized jumps and exception tables) and assembles
# them back into code objects (transform_code_object, assemble, line/exception tables),
# plus helpers to clone code, generate code from template functions and fix up jump
# targets. Version-conditional definitions are selected at import time by the running
# Python version; no other import-time side effects.
"""
This module provides utilities for analyzing, transforming and manipulating Python bytecode.
It includes functionality for:
- Converting between different bytecode formats and versions
- Virtualizing jumps and managing jump targets
- Handling exception tables and their entries
- Managing instruction offsets and extended arguments
- Providing a clean API for bytecode modification and transformation
- Supporting Python version-specific bytecode features
- Generating bytecode from template functions

The module is designed to work across different Python versions (3.7+) and handles
version-specific bytecode differences transparently.
"""

import copy
import dataclasses
import dis
import functools
import itertools
import re
import sys
import types
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, cast, Optional, TYPE_CHECKING

from . import config
from .bytecode_analysis import (
    get_indexof,
    propagate_line_nums,
    remove_extra_line_nums,
    stacksize_analysis,
)
from .utils import is_safe_constant


if TYPE_CHECKING:
    from .output_graph import DynamoTracerOutput


@dataclasses.dataclass(slots=True)
class InstructionExnTabEntry:
    # [CLASS DOC]
    # One exception-table entry expressed over Instruction objects (not offsets): the
    # inclusive start and end instructions of the protected range, the handler target
    # instruction, the stack depth to unwind to, and whether the last-instruction offset
    # is pushed. Equality is by identity of the three instructions plus depth and lasti.
    start: "Instruction"
    end: "Instruction"
    target: "Instruction"
    depth: int
    lasti: bool

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns a readable string using short representations of the three instructions.
        return (
            f"InstructionExnTabEntry(start={self.start.short_inst_repr()}, "
            f"end={self.end.short_inst_repr()}, "
            f"target={self.target.short_inst_repr()}, "
            f"depth={self.depth}, lasti={self.lasti})"
        )

    def __eq__(self, o: object) -> bool:
        # [FUNCTION DOC]
        # True only for another entry with the identical start, end and target
        # instruction objects and equal depth and lasti; False for other types.
        if not isinstance(o, InstructionExnTabEntry):
            return False
        return (
            self.start is o.start
            and self.end is o.end
            and self.target is o.target
            and self.depth == o.depth
            and self.lasti == o.lasti
        )


@dataclasses.dataclass(slots=True)
class Instruction:
    # [CLASS DOC]
    # Mutable counterpart of dis.Instruction: opcode, name, raw arg, resolved argval,
    # offset, source line, jump-target flag, positions, plus Dynamo extras: target (the
    # Instruction a jump points to), exn_tab_entry and argrepr. Hashing and equality
    # are by object identity so instructions can key mappings while being mutated.
    """A mutable version of dis.Instruction"""

    opcode: int
    opname: str
    arg: int | None
    argval: Any
    offset: int | None = None
    starts_line: int | None = None
    is_jump_target: bool = False

    positions: Optional["dis.Positions"] = None
    # extra fields to make modification easier:
    target: Optional["Instruction"] = None
    exn_tab_entry: InstructionExnTabEntry | None = None
    argrepr: str | None = None

    def __hash__(self) -> int:
        # [FUNCTION DOC]
        # Identity-based hash.
        return id(self)

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Identity-based equality: true only for the same object.
        return id(self) == id(other)

    def short_inst_repr(self) -> str:
        # [FUNCTION DOC]
        # Returns a compact string with just the opcode name and offset.
        return f"Instruction(opname={self.opname}, offset={self.offset})"

    def copy_positions(self, other: "Instruction") -> None:
        # [FUNCTION DOC]
        # Copies the source line number and position info from another instruction onto
        # this one (mutates this instruction).
        self.starts_line = other.starts_line
        self.positions = other.positions


if sys.version_info >= (3, 13):

    def convert_instruction(i: dis.Instruction) -> Instruction:
        # [FUNCTION DOC]
        # Converts a dis instruction to a mutable Instruction (3.13+ variant: line comes
        # from the line_number field; positions are kept).
        return Instruction(
            i.opcode,
            i.opname,
            i.arg,
            i.argval,
            i.offset,
            i.line_number,
            i.is_jump_target,
            i.positions,
        )

elif sys.version_info >= (3, 11):

    def convert_instruction(i: dis.Instruction) -> Instruction:
        # [FUNCTION DOC]
        # Converts a dis instruction to a mutable Instruction (3.11/3.12 variant: uses
        # starts_line and keeps positions).
        return Instruction(
            i.opcode,
            i.opname,
            i.arg,
            i.argval,
            i.offset,
            i.starts_line,
            i.is_jump_target,
            i.positions,
        )

else:

    def convert_instruction(i: dis.Instruction) -> Instruction:
        # [FUNCTION DOC]
        # Converts a dis instruction to a mutable Instruction (pre-3.11 variant: no
        # position info available, so positions is None).
        return Instruction(
            i.opcode,
            i.opname,
            i.arg,
            i.argval,
            i.offset,
            i.starts_line,
            i.is_jump_target,
            None,
        )


class _NotProvided:
    # [CLASS DOC] HELPER: Sentinel type marking an argument as not supplied, so None can
    # remain a valid explicit value.
    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the sentinel's name.
        return "_NotProvided"


if sys.version_info >= (3, 12):

    def inst_has_op_bits(name: str) -> bool:
        # [FUNCTION DOC]
        # True if the opcode name encodes flag bits in its arg (3.12+: load-attr,
        # load-global, load-super-attr).
        return name in ("LOAD_ATTR", "LOAD_GLOBAL", "LOAD_SUPER_ATTR")

elif sys.version_info >= (3, 11):

    def inst_has_op_bits(name: str) -> bool:
        # [FUNCTION DOC]
        # True if the opcode name encodes flag bits in its arg (3.11: load-global only).
        return name == "LOAD_GLOBAL"

else:

    def inst_has_op_bits(name: str) -> bool:
        # [FUNCTION DOC]
        # Always False before 3.11: no opcodes carry flag bits in the arg.
        return False


def create_instruction(
    name: str,
    *,
    arg: int | None = None,
    argval: Any | None = _NotProvided,
    target: Instruction | None = None,
) -> Instruction:
    # [FUNCTION DOC]
    # Builds an Instruction by opcode name. At most one of arg, argval and target may be
    # given (except opcodes with flag bits, where arg is the flag value, default 0, and
    # argval may also be given; target is then forbidden). Raises a runtime error on
    # ambiguous arguments, a non-integer arg, or target on a flag-bit opcode. Raises
    # KeyError for an unknown opcode name for this Python version.
    """
    At most one of `arg`, `argval`, and `target` can be not None/_NotProvided.
    This is to prevent ambiguity, e.g. does
        create_instruction("LOAD_CONST", 5)
    mean load the constant at co_consts[5], or load the constant 5?

    If `arg` is not provided, it will be computed during assembly from
    `argval` or `target`.

    Bits in the args of instructions LOAD_GLOBAL, LOAD_ATTR (3.12+), and LOAD_SUPER_ATTR
    modify the behavior of the instruction. In this case, we allow both `arg`
    and `argval` to be set. The value of `arg` here is expected to be the value of
    the op bits and the true value of `arg` will be computed during assembly.
    If `arg` is not set, the bits are assumed to be 0.
    """

    # allow for instructions with op bits to have both arg and argval specified
    if inst_has_op_bits(name):
        if target is not None:
            raise RuntimeError("target cannot be specified for instruction")
        if arg is None:
            arg = 0
    else:
        cnt = (arg is not None) + (argval is not _NotProvided) + (target is not None)
        if cnt > 1:
            raise RuntimeError(
                "only one of arg, argval, and target can be not None/_NotProvided"
            )
    if arg is not None and not isinstance(arg, int):
        raise RuntimeError("instruction arg must be int or None")
    return Instruction(
        opcode=dis.opmap[name], opname=name, arg=arg, argval=argval, target=target
    )


# Python 3.11 remaps
def create_jump_absolute(target: Instruction) -> Instruction:
    # [FUNCTION DOC]
    # Builds an unconditional jump to target: a forward jump on 3.11+ (absolute jumps
    # no longer exist, jumps are relative and later fixed up), else absolute jump.
    inst = "JUMP_FORWARD" if sys.version_info >= (3, 11) else "JUMP_ABSOLUTE"
    return create_instruction(inst, target=target)


def is_jump_absolute(target: Instruction) -> bool:
    # [FUNCTION DOC]
    # True if the instruction is an unconditional jump (forward or absolute form).
    # The parameter is named target but is the instruction being tested.
    return target.opname in ("JUMP_FORWARD", "JUMP_ABSOLUTE")


def create_load_const(val: Any, checked: bool = True) -> Instruction:
    # [FUNCTION DOC]
    # Builds a constant-load instruction carrying val. When checked, raises an error if
    # val is not a safe (immutable) constant; unchecked allows arbitrary objects.
    """
    In general we should only create `LOAD_CONST` for immutable objects, but
    sometimes it's convenient _and safe_ for Dynamo create `LOAD_CONST` for
    mutable objects. In such cases, use `checked=False`.
    """
    if checked:
        if not is_safe_constant(val):
            raise AssertionError(f"unsafe constant {val}")
    return create_instruction("LOAD_CONST", argval=val)


def create_breakpoint() -> list[Instruction]:
    # [FUNCTION DOC]
    # Returns a load-constant of the bytecode debugger's marker followed by a pop; a
    # no-op at runtime that the bytecode debugger recognizes as a breakpoint. Imports
    # the marker lazily from the bytecode debugger module.
    """
    Create instructions that trigger the bytecode debugger to stop.

    Usage:
        codegen.extend_output(create_breakpoint())

    When the bytecode debugger is active, execution will pause at this point.
    At runtime, these instructions have no effect on program state.
    """
    from .bytecode_debugger import BREAKPOINT_MARKER

    return [
        create_instruction("LOAD_CONST", argval=BREAKPOINT_MARKER),
        create_instruction("POP_TOP"),
    ]


def create_dup_top() -> Instruction:
    # [FUNCTION DOC]
    # Returns an instruction duplicating the top of stack (copy 1 on 3.11+, dup-top
    # before).
    if sys.version_info >= (3, 11):
        return create_instruction("COPY", arg=1)
    return create_instruction("DUP_TOP")


def create_rot_n(n: int) -> list[Instruction]:
    # [FUNCTION DOC]
    # Returns instructions moving top of stack down to position n. Empty for n <= 1.
    # On 3.11+ a descending series of swaps; before 3.11 a single rotate opcode (two,
    # three, four, or general ROT_N) which may not exist, in which case an AttributeError
    # style failure from opcode lookup propagates and callers fall back.
    """
    Returns a "simple" sequence of instructions that rotates TOS to the n-th
    position in the stack. For Python < 3.11, returns a single ROT_*
    instruction. If no such instruction exists, an error is raised and the
    caller is expected to generate an equivalent sequence of instructions.
    For Python >= 3.11, any rotation can be expressed as a simple sequence of
    swaps.
    """
    if n <= 1:
        # don't rotate
        return []

    if sys.version_info >= (3, 11):
        # rotate can be expressed as a sequence of swap operations
        # e.g. rotate 3 is equivalent to swap 3, swap 2
        return [create_instruction("SWAP", arg=i) for i in range(n, 1, -1)]

    if n <= 4:
        return [create_instruction("ROT_" + ["TWO", "THREE", "FOUR"][n - 2])]
    return [create_instruction("ROT_N", arg=n)]


def add_push_null(
    inst_or_insts: Instruction | list[Instruction],
) -> list[Instruction]:
    # [FUNCTION DOC]
    # Given an instruction or list that pushes exactly one callable, returns a list that
    # also pushes the NULL used by the 3.11+ call convention: sets the low arg bit on a
    # load-global/load-attr/load-super-attr when possible (mutating it), else adds a
    # push-null before (3.11, 3.12) or after (3.13+) the callable. Returns the list
    # unchanged before 3.11. Raises an error if a non-list non-Instruction is passed.
    """
    Appends or prepends a PUSH_NULL instruction to `inst_or_insts`,
    depending on Python version. Used when you know that
    `inst_or_insts` generates a callable that will be called.

    NOTE: Assumes `inst_or_insts` is a single instruction or sequence of
    instructions that pushes exactly 1 object to the stack that is to
    be called. It is important that you include ALL instructions that
    construct the callable - not just the first instruction/a prefix.

    Will attempt to use the NULL push bit for instructions
    with such bits (LOAD_GLOBAL 3.11+, LOAD_ATTR 3.12+, LOAD_SUPER_ATTR).
    In this case, instructions WILL be modified.
    """
    if isinstance(inst_or_insts, Instruction):
        insts: list[Instruction] = [inst_or_insts]
    else:
        if not isinstance(inst_or_insts, list):
            raise AssertionError(
                f"expected inst_or_insts to be a list, got {type(inst_or_insts)}"
            )
        insts = inst_or_insts

    def inst_has_bit_set(idx: int) -> bool:
        # [FUNCTION DOC] HELPER: True if the low (push-null) bit of the arg of the
        # instruction at idx is set; raises an error if that arg is None.
        if insts[idx].arg is None:
            raise AssertionError(f"expected insts[{idx}].arg to be not None")
        return insts[idx].arg & 1 == 1  # type: ignore[operator]

    def set_inst_bit(idx: int) -> None:
        # [FUNCTION DOC] HELPER: Sets the low (push-null) bit of the arg of the
        # instruction at idx in place; raises an error if that arg is None.
        if insts[idx].arg is None:
            raise AssertionError(f"expected insts[{idx}].arg to be not None")
        insts[idx].arg |= 1  # type: ignore[operator]

    if sys.version_info >= (3, 13):
        # In 3.13, NULL follows the callable
        if inst_has_op_bits(insts[-1].opname) and not inst_has_bit_set(-1):
            # All insts with op bits have the push_null bit as the last one.
            # Only set the bit if it hasn't been set - otherwise, we need
            # to add another PUSH_NULL.
            set_inst_bit(-1)
        else:
            insts = insts + [create_instruction("PUSH_NULL")]
    elif sys.version_info >= (3, 12):
        # LOAD_ATTR/LOAD_SUPER_ATTR at the end
        # We assume that `insts` will only load 1 object, so
        # LOAD_GLOBAL at the end doesn't need to be checked
        if inst_has_op_bits(insts[-1].opname) and not inst_has_bit_set(-1):
            set_inst_bit(-1)
        elif insts[0].opname == "LOAD_GLOBAL" and not inst_has_bit_set(0):
            set_inst_bit(0)
        else:
            insts = [create_instruction("PUSH_NULL")] + insts
    elif sys.version_info >= (3, 11):
        # 3.11 introduced NULL preceding callable
        if inst_has_op_bits(insts[0].opname) and not inst_has_bit_set(0):
            set_inst_bit(0)
        else:
            insts = [create_instruction("PUSH_NULL")] + insts
    return insts


def add_push_null_call_function_ex(
    inst_or_insts: Instruction | list[Instruction],
) -> list[Instruction]:
    # [FUNCTION DOC]
    # Variant of add_push_null for callables that will be invoked with the call-ex
    # opcode: only a load-global's low bit may be set (the load-attr bit is not used);
    # otherwise a push-null is added before (3.11/3.12) or after (3.13+). Unchanged
    # before 3.11. Raises an error for invalid input type or a load-global with no arg.
    """Like add_push_null, but the low bit of LOAD_ATTR/LOAD_SUPER_ATTR
    is not set, due to an expected CALL_FUNCTION_EX instruction.
    """
    if isinstance(inst_or_insts, Instruction):
        insts: list[Instruction] = [inst_or_insts]
    else:
        if not isinstance(inst_or_insts, list):
            raise AssertionError(
                f"expected inst_or_insts to be a list, got {type(inst_or_insts)}"
            )
        insts = inst_or_insts

    if sys.version_info < (3, 11):
        return insts

    idx = -1 if sys.version_info >= (3, 13) else 0
    if insts[idx].opname == "LOAD_GLOBAL":
        if insts[idx].arg is None:
            raise AssertionError(
                f"expected insts[{idx}].arg to be not None for LOAD_GLOBAL"
            )
        if insts[idx].arg & 1 == 0:  # type: ignore[operator]
            insts[idx].arg |= 1  # type: ignore[operator]
            return insts

    if sys.version_info >= (3, 13):
        insts = insts + [create_instruction("PUSH_NULL")]
    else:
        insts = [create_instruction("PUSH_NULL")] + insts

    return insts


def create_call_function(nargs: int, push_null: bool) -> list[Instruction]:
    # [FUNCTION DOC]
    # Returns instructions calling the callable on the stack with nargs positional
    # args. On 3.11+: optionally pushes a NULL and rotates it into place when push_null
    # is true, then a precall (3.11 only) and call; before 3.11 a single call-function.
    """
    Creates a sequence of instructions that makes a function call.

    `push_null` is used in Python 3.11+ only. It is used in codegen when
    a function call is intended to be made with the NULL + fn convention,
    and we know that the NULL has not been pushed yet. We will push a
    NULL and rotate it to the correct position immediately before making
    the function call.

    `push_null` should be True if no NULL is pushed for the callable.
    Conversely, `push_null` should be False if a NULL was pushed for the callable.
    Prefer using `push_null=False` when possible since we will not need to rotate
    NULL to the right place, which is less efficient.

    Generally, you should codegen a function by using `add_push_null` then
    `create_call_function` with `push_null=False`.

    Example of when to set push_null False:

    insts = [
        create_instruction("LOAD_GLOBAL", argval="torch"),
        create_instruction("LOAD_ATTR", argval="nn"),
        create_instruction("LOAD_ATTR", argval="functional"),
        create_instruction("LOAD_ATTR", argval="relu"),
    ]
    insts = add_push_null(insts)
    insts.append(create_instruction("LOAD_FAST", argval="x"))
    insts.extend(create_call_function(1, False))

    Example of when to set push_null True:

    insts = [create_instruction("LOAD_FAST", x)]
    for should_wrap, wrapper_name in wrappers:
        if should_wrap:
            insts.extend([
                create_instruction("LOAD_GLOBAL", argval="wrapper1"),
                create_instruction("SWAP", arg=2),
                *create_call_function(1, True),
            )
    """
    if sys.version_info >= (3, 11):
        output = []
        if push_null:
            output.append(create_instruction("PUSH_NULL"))
            # 3.13 swapped NULL and callable
            rots = nargs + 1 if sys.version_info >= (3, 13) else nargs + 2
            output.extend(create_rot_n(rots))
        if sys.version_info < (3, 12):
            output.append(create_instruction("PRECALL", arg=nargs))
        output.append(create_instruction("CALL", arg=nargs))
        return output
    return [create_instruction("CALL_FUNCTION", arg=nargs)]


def create_call_function_ex(
    has_kwargs: bool, push_null: bool, ignore_314_kwargs_push: bool = False
) -> list[Instruction]:
    # [FUNCTION DOC]
    # Returns instructions for a call-ex call (args tuple, optional kwargs mapping on
    # the stack). On 3.14+ without kwargs, pushes a NULL as the kwargs slot unless the
    # caller already did. With push_null on 3.11+, pushes a NULL and rotates it below the
    # callable. The flag argument encodes whether kwargs are present.
    """
    Assumes that in 3.14+, if has_kwargs=False, there is NOT a NULL
    on the TOS for the kwargs. This utility function will add a PUSH_NULL.

    If the caller has already pushed a NULL for the kwargs, then set ignore_314_kwargs_push=True
    so we don't push another NULL for the kwargs.
    """
    if sys.version_info >= (3, 11):
        output = []
        if (
            sys.version_info >= (3, 14)
            and not has_kwargs
            and not ignore_314_kwargs_push
        ):
            output.append(create_instruction("PUSH_NULL"))
            has_kwargs = True
        if push_null:
            output.append(create_instruction("PUSH_NULL"))
            # 3.13 swapped NULL and callable
            # if flags == 1, 2 values popped - otherwise if flags == 0, 1 value
            rots = (
                int(has_kwargs) + 2
                if sys.version_info >= (3, 13)
                else int(has_kwargs) + 3
            )
            output.extend(create_rot_n(rots))
        output.append(create_instruction("CALL_FUNCTION_EX", arg=int(has_kwargs)))
        return output
    return [create_instruction("CALL_FUNCTION_EX", arg=int(has_kwargs))]


def create_call_method(nargs: int) -> list[Instruction]:
    # [FUNCTION DOC]
    # Returns the version-specific instructions that call a method loaded earlier with
    # nargs args: a call (3.12+), precall plus call (3.11), or call-method (earlier).
    if sys.version_info >= (3, 12):
        return [create_instruction("CALL", arg=nargs)]
    if sys.version_info >= (3, 11):
        return [
            create_instruction("PRECALL", arg=nargs),
            create_instruction("CALL", arg=nargs),
        ]
    return [create_instruction("CALL_METHOD", arg=nargs)]


def create_load_method(name: str) -> Instruction:
    # [FUNCTION DOC]
    # Returns an instruction that loads a method by name: a load-attr with the method
    # flag bit set on 3.12+, else a load-method.
    if sys.version_info >= (3, 12):
        # in 3.12, create a LOAD_ATTR instruction with the low bit set
        return create_instruction("LOAD_ATTR", arg=1, argval=name)
    return create_instruction("LOAD_METHOD", argval=name)


def create_setup_with(target: Instruction) -> Instruction:
    # [FUNCTION DOC]
    # Returns the context-manager entry instruction (before-with on 3.11+, setup-with
    # before) whose jump target is the given handler instruction.
    opname = "BEFORE_WITH" if sys.version_info >= (3, 11) else "SETUP_WITH"
    return create_instruction(opname, target=target)


def create_swap(n: int) -> list[Instruction]:
    # [FUNCTION DOC]
    # Returns instructions swapping the top of stack with the n-th item. A single swap
    # on 3.11+. Before 3.11: nothing for n of 1, rotate opcodes for 2 and 3, and for
    # larger n a sequence that packs the top n-1 items in a list, swaps its last
    # element with the top, and unpacks it (reversing as needed).
    if sys.version_info >= (3, 11):
        return [create_instruction("SWAP", arg=n)]
    # in Python < 3.11, SWAP is a macro that expands to multiple instructions
    if n == 1:
        return []
    elif n == 2:
        return [create_instruction("ROT_TWO")]
    elif n == 3:
        return [create_instruction("ROT_THREE"), create_instruction("ROT_TWO")]
    """
    e.g. swap "a" and "b" in this stack:
    0 a 1 2 3 b
    0 a [1 2 3 b]
    0 a [1 2 3 b] [1 2 3 b]
    0 a [1 2 3 b] [1 2 3 b] -1
    0 a [1 2 3 b] b
    0 b a [1 2 3 b]
    0 b a [1 2 3 b] [1 2 3 b]
    0 b [1 2 3 b] a [1 2 3 b]
    0 b [1 2 3 b] a [1 2 3 b] -1
    0 b [1 2 3 a]
    0 b [1 2 3 a] [1 2 3 a]
    0 b [1 2 3 a] [1 2 3 a] reverse
    0 b [a 3 2 1] None
    0 b [a 3 2 1]
    0 b 1 2 3 a
    """
    return [
        create_instruction("BUILD_LIST", arg=n - 1),
        create_instruction("DUP_TOP"),
        create_instruction("LOAD_CONST", argval=-1),
        create_binary_subscr(),
        create_instruction("ROT_THREE"),
        create_instruction("DUP_TOP"),
        create_instruction("ROT_THREE"),
        create_instruction("LOAD_CONST", argval=-1),
        create_instruction("STORE_SUBSCR"),
        create_instruction("DUP_TOP"),
        create_load_method("reverse"),
        *create_call_method(0),
        create_instruction("POP_TOP"),
        create_instruction("UNPACK_SEQUENCE", arg=n - 1),
    ]


def get_call_callable_depth(opname: str, arg: int) -> int:
    # [FUNCTION DOC]
    # Returns the 1-based stack depth (top = 1) of the callable for a call instruction,
    # given its opname and arg, using per-Python-version stack layouts (call, call with
    # keywords, call-function variants, call-ex). Raises ValueError for non-call opnames.
    """Return the depth of the callable from TOS for a call instruction.

    Depth is 1-indexed: TOS = 1, below-TOS = 2, etc.  This encodes the
    per-version stack layouts for every call instruction variant.

    Note: on 3.11-3.12, CPython's CALL can have either NULL or self at the
    bottom depending on function vs method call.  Dynamo's LOAD_METHOD
    normalizes method calls to NULL + bound_method, so the bottom is always
    NULL and the callable is always at depth arg + 1.

    Stack layouts (bottom-to-top within the call's frame):
      CALL (3.13+):            callable  NULL  arg0 ... argN-1
      CALL (3.11-3.12):        NULL  callable  arg0 ... argN-1
      CALL_KW (3.13+):         callable  NULL  arg0 ... argN-1  kw_names
      CALL_FUNCTION (3.10):    callable  arg0 ... argN-1
      CALL_FUNCTION_KW (3.10): callable  arg0 ... argN-1  kw_names
      CALL_FUNCTION_EX:
        3.14+:                 callable  NULL  args_tuple  kwargs_or_null
        3.13:                  callable  NULL  args_tuple [kwargs_dict]
        3.11-3.12:             NULL  callable  args_tuple [kwargs_dict]
        3.10:                  callable  args_tuple [kwargs_dict]
    """
    if opname in ("CALL", "CALL_KW"):
        kw_extra = 1 if opname == "CALL_KW" else 0
        if sys.version_info >= (3, 13):
            return arg + 2 + kw_extra
        else:
            # 3.11-3.12: NULL is always at bottom (Dynamo normalizes
            # LOAD_METHOD to NULL + bound_method)
            return arg + 1 + kw_extra
    elif opname in ("CALL_FUNCTION", "CALL_FUNCTION_KW"):
        kw_extra = 1 if opname == "CALL_FUNCTION_KW" else 0
        return arg + 1 + kw_extra
    elif opname == "CALL_FUNCTION_EX":
        if sys.version_info >= (3, 14):
            return 4
        elif sys.version_info >= (3, 13):
            return 3 + (arg & 1)
        elif sys.version_info >= (3, 11):
            return 2 + (arg & 1)
        else:
            return 2 + (arg & 1)
    else:
        raise ValueError(f"not a call instruction: {opname}")


def create_binary_slice(
    start: int | None, end: int | None, store: bool = False
) -> list[Instruction]:
    # [FUNCTION DOC]
    # Returns instructions performing a slice load (or store when store is true) of
    # start:end on the container on the stack: a slice constant plus subscript (3.14+),
    # two constants plus slice opcode (3.12/3.13), or build-slice plus subscript (older).
    # KNOWN ISSUE: the docstring refers to a `set` flag but the parameter is `store`.
    """
    BINARY_SLICE and STORE_SLICE (if `set` is True) for all Python versions
    """
    if sys.version_info >= (3, 14):
        subscr_inst = (
            create_instruction("STORE_SUBSCR") if store else create_binary_subscr()
        )
        return [
            create_load_const(slice(start, end)),
            subscr_inst,
        ]
    elif sys.version_info >= (3, 12):
        inst_name = "STORE_SLICE" if store else "BINARY_SLICE"
        return [
            create_load_const(start),
            create_load_const(end),
            create_instruction(inst_name),
        ]
    else:
        inst_name = "STORE_SUBSCR" if store else "BINARY_SUBSCR"
        return [
            create_load_const(start),
            create_load_const(end),
            create_instruction("BUILD_SLICE", arg=2),
            create_instruction(inst_name),
        ]


def create_copy(i: int) -> list[Instruction]:
    # [FUNCTION DOC]
    # Returns instructions pushing a copy of the i-th stack item on top. Native copy on
    # 3.11+; before that, dup-top for 1 and otherwise a swap/dup/swap sequence.
    if sys.version_info >= (3, 11):
        return [create_instruction("COPY", arg=i)]
    if i == 1:
        return [create_instruction("DUP_TOP")]
    # COPY 4
    # 0 1 2 3
    # 3 1 2 0
    # 3 1 2 0 0
    # 0 1 2 0 3
    # 0 1 2 3 0
    return [
        *create_swap(i),
        create_dup_top(),
        *create_swap(i + 1),
        *create_swap(2),
    ]


# mainly for debugging generated bytecode
def create_print_on_stack(depth: int) -> list[Instruction]:
    # [FUNCTION DOC] HELPER: Debug aid. Returns instructions that copy the stack item at
    # the given depth and print it, leaving the stack unchanged.
    return [
        *add_push_null(create_instruction("LOAD_CONST", argval=print)),
        *create_copy(depth + (2 if sys.version_info >= (3, 11) else 1)),
        *create_call_function(1, False),
        create_instruction("POP_TOP"),
    ]


# mainly for debugging generated bytecode
def create_print_value(value: Any) -> list[Instruction]:
    # [FUNCTION DOC] HELPER: Debug aid. Returns instructions that print a constant
    # value at runtime, leaving the stack unchanged.
    return [
        *add_push_null(create_instruction("LOAD_CONST", argval=print)),
        create_instruction("LOAD_CONST", argval=value),
        *create_call_function(1, False),
        create_instruction("POP_TOP"),
    ]


def create_binary_subscr() -> Instruction:
    # [FUNCTION DOC]
    # Returns a subscript instruction: dedicated opcode before 3.14, else the generic
    # binary-op with the hard-coded subscript operator code.
    if sys.version_info < (3, 14):
        return create_instruction("BINARY_SUBSCR")
    # https://github.com/python/cpython/blob/0e46c0499413bc5f9f8336fe76e2e67cf93f64d8/Include/opcode.h#L36
    return create_instruction("BINARY_OP", arg=26)


def create_build_tuple(n: int) -> Instruction:
    # [FUNCTION DOC]
    # Returns an instruction building a tuple from n stack items; on 3.14+ an empty
    # tuple is instead a constant load.
    if sys.version_info >= (3, 14) and n == 0:
        return create_load_const(())
    return create_instruction("BUILD_TUPLE", arg=n)


def linetable_writer(
    first_lineno: int,
) -> tuple[list[int], Callable[[int, int], None], Callable[[int], None]]:
    # [FUNCTION DOC]
    # Python 3.10 only (raises an error otherwise). Returns a mutable byte list holding
    # the line table plus two writers: update(new_line, new_byte_offset) records a line
    # change, and end(total_bytes) flushes the remainder. Encoding is pairs of byte
    # delta and signed line delta, split into chunks within the format limits.
    """
    Used to create typing.CodeType.co_linetable
    See https://github.com/python/cpython/blob/3.10/Objects/lnotab_notes.txt
    This is the internal format of the line number table for Python 3.10
    """
    if sys.version_info[:2] != (3, 10):
        raise AssertionError(
            f"linetable_writer requires Python 3.10, got {sys.version_info[:2]}"
        )
    linetable: list[int] = []
    lineno = first_lineno
    lineno_delta = 0
    byteno = 0

    def _update(byteno_delta: int, lineno_delta: int) -> None:
        # [FUNCTION DOC] HELPER: Appends one or more (byte delta, line delta) pairs to the
        # line table, clamping each to the encodable ranges, until both deltas are consumed.
        while byteno_delta != 0 or lineno_delta != 0:
            byte_offset = max(0, min(byteno_delta, 254))
            line_offset = max(-127, min(lineno_delta, 127))
            if byte_offset == 0 and line_offset == 0:
                raise AssertionError("byte_offset and line_offset cannot both be 0")
            byteno_delta -= byte_offset
            lineno_delta -= line_offset
            linetable.extend((byte_offset, line_offset & 0xFF))

    def update(lineno_new: int, byteno_new: int) -> None:
        # [FUNCTION DOC] HELPER: Flushes the pending delta for the bytes since the last
        # call and remembers the new line delta to emit with the next update or end.
        nonlocal lineno, lineno_delta, byteno
        byteno_delta = byteno_new - byteno
        byteno = byteno_new
        _update(byteno_delta, lineno_delta)
        lineno_delta = lineno_new - lineno
        lineno = lineno_new

    def end(total_bytes: int) -> None:
        # [FUNCTION DOC] HELPER: Flushes the final pending byte and line deltas given the
        # total code size in bytes.
        _update(total_bytes - byteno, lineno_delta)

    return linetable, update, end


def encode_varint(n: int) -> list[int]:
    # [FUNCTION DOC]
    # Encodes a non-negative integer as 6-bit chunks, least significant first, with the
    # continuation bit on all but the last (3.11 location table format). Raises an error
    # for negative input.
    """
    6-bit chunk encoding of an unsigned integer
    See https://github.com/python/cpython/blob/3.11/Objects/locations.md
    """
    if n < 0:
        raise AssertionError(f"encode_varint requires non-negative n, got {n}")
    b = [n & 63]
    n >>= 6
    while n > 0:
        b[-1] |= 64
        b.append(n & 63)
        n >>= 6
    return b


if sys.version_info >= (3, 11):

    def linetable_311_writer(
        first_lineno: int,
    ) -> tuple[list[int], Callable[[dis.Positions | None, int], None]]:
        # [FUNCTION DOC]
        # Python 3.11+ only. Returns a mutable byte list holding the location table plus
        # an update(positions, inst_size) writer that appends entries (long form with
        # column info when positions are complete, no-column form otherwise), splitting
        # instructions longer than 8 code units into several entries. Line deltas are
        # relative to the previously written line.
        """
        Used to create typing.CodeType.co_linetable
        See https://github.com/python/cpython/blob/3.11/Objects/locations.md
        This is the internal format of the line number table for Python 3.11
        """
        if sys.version_info < (3, 11):
            raise AssertionError(
                f"linetable_311_writer requires Python 3.11+, got {sys.version_info}"
            )
        linetable = []
        lineno = first_lineno

        def update(positions: dis.Positions | None, inst_size: int) -> None:
            # [FUNCTION DOC] HELPER: Appends location-table entries covering inst_size
            # code units for the given positions (None means no line change).
            nonlocal lineno
            lineno_new = positions.lineno if positions else None

            def _update(delta: int, size: int) -> None:
                # [FUNCTION DOC] HELPER: Emits one entry for up to 8 code units: a header
                # byte, the zigzag-encoded line delta, and (when positions are complete)
                # end-line delta and 1-based columns. Raises an error if size is outside
                # 1..8.
                if not (0 < size <= 8):
                    raise AssertionError(f"size must be in range (0, 8], got {size}")
                # first byte - use 13 (no column info) is positions is
                # malformed, otherwise use 14 (long form)
                other_varints: tuple[int, ...] = ()
                if (
                    positions
                    and positions.lineno is not None
                    and positions.end_lineno is not None
                    and positions.col_offset is not None
                    and positions.end_col_offset is not None
                ):
                    linetable.append(0b1_1110_000 + size - 1)
                    # for whatever reason, column offset needs `+ 1`
                    # https://github.com/python/cpython/blob/1931c2a438c50e6250725c84dff94fc760b9b951/Python/compile.c#L7603
                    other_varints = (
                        positions.end_lineno - positions.lineno,
                        positions.col_offset + 1,
                        positions.end_col_offset + 1,
                    )
                else:
                    linetable.append(0b1_1101_000 + size - 1)
                # encode signed int
                if delta < 0:
                    delta = ((-delta) << 1) | 1
                else:
                    delta <<= 1
                # encode unsigned int
                linetable.extend(encode_varint(delta))
                for n in other_varints:
                    linetable.extend(encode_varint(n))

            if lineno_new is None:
                lineno_delta = 0
            else:
                lineno_delta = lineno_new - lineno
                lineno = lineno_new
            while inst_size > 8:
                _update(lineno_delta, 8)
                inst_size -= 8
            _update(lineno_delta, inst_size)

        return linetable, update


@dataclasses.dataclass(slots=True)
class ExceptionTableEntry:
    # [CLASS DOC]
    # Offset-based exception table entry as stored in a code object: byte offsets of
    # the protected range start and (inclusive) last code unit, handler target offset,
    # stack depth, and lasti flag.
    start: int
    end: int
    target: int
    depth: int
    lasti: bool


def encode_exception_table_varint(n: int) -> list[int]:
    # [FUNCTION DOC]
    # Encodes a non-negative integer as 6-bit chunks, most significant first, with the
    # continuation bit set on all but the last. Raises an error for negative input.
    """
    Similar to `encode_varint`, but the 6-bit chunks are ordered in reverse.
    """
    if n < 0:
        raise AssertionError(
            f"encode_exception_table_varint requires non-negative n, got {n}"
        )
    b = [n & 63]
    n >>= 6
    while n > 0:
        b.append(n & 63)
        n >>= 6
    b.reverse()
    for i in range(len(b) - 1):
        b[i] |= 64
    return b


def decode_exception_table_varint(bytes_iter: Iterator[int]) -> int:
    # [FUNCTION DOC]
    # Consumes bytes from the iterator and decodes one exception-table varint. Raises
    # StopIteration if the iterator is exhausted.
    """
    Inverse of `encode_exception_table_varint`.
    """
    b = next(bytes_iter)
    val = b & 63
    while b & 64:
        val <<= 6
        b = next(bytes_iter)
        val |= b & 63
    return val


def check_exception_table(tab: list[ExceptionTableEntry]) -> None:
    # [FUNCTION DOC]
    # Validates that entries are non-empty, sorted and non-overlapping; raises an
    # error naming the offending adjacent pair otherwise.
    """
    Verifies that a list of ExceptionTableEntries will make a well-formed
    jump table: entries are non-empty, sorted, and do not overlap.
    """
    for i in range(len(tab) - 1):
        if not (
            tab[i].start <= tab[i].end
            and tab[i].end < tab[i + 1].start
            and tab[i + 1].start <= tab[i + 1].end
        ):
            raise AssertionError(
                f"exception table entries are not well-formed at index {i}: "
                f"entry[{i}]=(start={tab[i].start}, end={tab[i].end}), "
                f"entry[{i + 1}]=(start={tab[i + 1].start}, end={tab[i + 1].end})"
            )


def parse_exception_table(exntab: bytes) -> list[ExceptionTableEntry]:
    # [FUNCTION DOC]
    # Decodes the raw exception table bytes into entries with byte offsets (values in
    # the table are in code units and are doubled), then validates them. Raises an
    # error if the entries are malformed.
    """
    Parse the exception table according to
    https://github.com/python/cpython/blob/3.11/Objects/exception_handling_notes.txt
    """
    exntab_iter = iter(exntab)
    tab = []
    try:
        while True:
            start = decode_exception_table_varint(exntab_iter) * 2
            length = decode_exception_table_varint(exntab_iter) * 2
            end = start + length - 2
            target = decode_exception_table_varint(exntab_iter) * 2
            dl = decode_exception_table_varint(exntab_iter)
            depth = dl >> 1
            lasti = bool(dl & 1)
            tab.append(ExceptionTableEntry(start, end, target, depth, lasti))
    except StopIteration:
        check_exception_table(tab)
        return tab


def assemble_exception_table(tab: list[ExceptionTableEntry]) -> bytes:
    # [FUNCTION DOC]
    # Encodes entries back to the raw table bytes: start (with the high bit marking the
    # entry start), length, target, and depth-plus-lasti, all in code units.
    """
    Inverse of parse_exception_table - encodes list of exception
    table entries into bytes.
    """
    b = []
    for entry in tab:
        first_entry = encode_exception_table_varint(entry.start // 2)
        first_entry[0] |= 1 << 7
        b.extend(first_entry)
        length = entry.end - entry.start + 2
        b.extend(encode_exception_table_varint(length // 2))
        b.extend(encode_exception_table_varint(entry.target // 2))
        dl = (entry.depth << 1) + entry.lasti
        b.extend(encode_exception_table_varint(dl))
    return bytes(b)


def assemble(instructions: list[Instruction], firstlineno: int) -> tuple[bytes, bytes]:
    # [FUNCTION DOC]
    # Encodes instructions into raw bytecode bytes and a line/location table bytes
    # object. Each instruction emits its opcode and low arg byte, padded with zero
    # code units to its full size (inline cache entries). On 3.11+ extended-arg
    # instructions adopt the positions of the real instruction following them. Returns
    # (code bytes, line table bytes).
    """Do the opposite of dis.get_instructions()"""
    code: list[int] = []
    if sys.version_info >= (3, 11):
        lnotab, update_lineno = linetable_311_writer(firstlineno)
        num_ext = 0
        for i, inst in enumerate(instructions):
            if inst.opname == "EXTENDED_ARG":
                inst_size = 1
                num_ext += 1
                # copy positions from the actual instruction
                for j in (1, 2, 3):
                    if instructions[i + j].opname != "EXTENDED_ARG":
                        inst.positions = instructions[i + j].positions
                        break
            else:
                inst_size = instruction_size(inst) // 2 + num_ext
                num_ext = 0
            update_lineno(inst.positions, inst_size)
            num_ext = 0
            arg = inst.arg or 0
            code.extend((inst.opcode, arg & 0xFF))
            for _ in range(instruction_size(inst) // 2 - 1):
                code.extend((0, 0))
    else:
        lnotab, update_lineno, end = linetable_writer(firstlineno)

        for inst in instructions:
            if inst.starts_line is not None:
                update_lineno(inst.starts_line, len(code))
            arg = inst.arg or 0
            code.extend((inst.opcode, arg & 0xFF))

        end(len(code))

    return bytes(code), bytes(lnotab)


def _get_instruction_by_offset(
    offset_to_inst: dict[int, Instruction], offset: int
) -> Instruction | None:
    # [FUNCTION DOC] HELPER: Looks up the instruction at the offset, skipping any
    # leading extended-arg prefixes (up to three). Returns None if all four are extended
    # args; raises KeyError if an offset is missing.
    """
    Get the instruction located at a given offset, accounting for EXTENDED_ARGs
    """
    for n in (0, 2, 4, 6):
        if offset_to_inst[offset + n].opcode != dis.EXTENDED_ARG:
            return offset_to_inst[offset + n]
    return None


def virtualize_jumps(instructions: Iterable[Instruction]) -> None:
    # [FUNCTION DOC]
    # For every absolute or relative jump, sets its target attribute to the Instruction
    # object at the jump's destination offset (skipping extended-arg prefixes), so code
    # can be edited without recomputing offsets. Mutates instructions in place.
    """Replace jump targets with pointers to make editing easier"""
    jump_targets = {
        inst.offset: inst for inst in instructions if inst.offset is not None
    }

    for inst in instructions:
        if inst.opcode in dis.hasjabs or inst.opcode in dis.hasjrel:
            inst.target = _get_instruction_by_offset(jump_targets, inst.argval)


_REL_JUMPS = set(dis.hasjrel)


def flip_jump_direction(instruction: Instruction) -> None:
    # [FUNCTION DOC]
    # Switches a relative jump between its forward and backward opcode variant in place
    # (3.11+). Raises RuntimeError before 3.11, AttributeError if the opname is neither
    # forward nor backward, and an assertion error if the result is not a relative jump.
    if sys.version_info < (3, 11):
        raise RuntimeError("Cannot flip jump direction in Python < 3.11")
    if "FORWARD" in instruction.opname:
        instruction.opname = instruction.opname.replace("FORWARD", "BACKWARD")
    elif "BACKWARD" in instruction.opname:
        instruction.opname = instruction.opname.replace("BACKWARD", "FORWARD")
    else:
        raise AttributeError("Instruction is not a forward or backward jump")
    instruction.opcode = dis.opmap[instruction.opname]
    if instruction.opcode not in _REL_JUMPS:
        raise AssertionError(
            f"expected {instruction.opname} to be a relative jump instruction"
        )


def _get_instruction_front(instructions: list[Instruction], idx: int) -> Instruction:
    # [FUNCTION DOC] HELPER: Returns the first of up to three consecutive extended-arg
    # instructions preceding instructions[idx], or the instruction itself if none; this
    # is where a jump to it must land.
    """
    i.e. get the first EXTENDED_ARG instruction (if any) when targeting
    instructions[idx] with a jump.
    """
    target = instructions[idx]
    for offset in (1, 2, 3):
        if idx >= offset and instructions[idx - offset].opcode == dis.EXTENDED_ARG:
            target = instructions[idx - offset]
        else:
            break
    return target


def devirtualize_jumps(instructions: list[Instruction]) -> None:
    # [FUNCTION DOC]
    # Inverse of virtualize_jumps: computes arg, argval and argrepr of every jump from
    # its target object using current offsets. First flips relative jump direction
    # where the target lies on the opposite side (3.11+; raises RuntimeError for a
    # negative jump before 3.11), recomputes offsets (sizes may change), then sets args
    # (absolute offset/2 before 3.11; relative magnitude in code units otherwise).
    # Raises on missing targets/offsets or on absolute jumps in 3.11+.
    """Fill in args for virtualized jump target after instructions may have moved"""
    jumps = set(dis.hasjabs).union(set(dis.hasjrel))

    # check for negative jump args and fix them
    for inst in instructions:
        if inst.opcode in jumps:
            if inst.opcode not in dis.hasjabs:
                if inst.target is None:
                    raise AssertionError(
                        f"jump instruction {inst.opname} has no target"
                    )
                if inst.target.offset is None:
                    raise AssertionError(f"jump target for {inst.opname} has no offset")
                if inst.offset is None:
                    raise AssertionError(
                        f"jump instruction {inst.opname} has no offset"
                    )
                if inst.target.offset < inst.offset:
                    if sys.version_info < (3, 11):
                        raise RuntimeError("Got negative jump offset for Python < 3.11")
                    # forward jumps become backward
                    if "FORWARD" in inst.opname:
                        flip_jump_direction(inst)
                else:
                    # backward jumps become forward
                    if sys.version_info >= (3, 11) and "BACKWARD" in inst.opname:
                        flip_jump_direction(inst)

    # jump instruction size may have changed due to flips
    update_offsets(instructions)
    indexof = get_indexof(instructions)

    # compute jump instruction arg
    for inst in instructions:
        if inst.opcode in jumps:
            if inst.target is None:
                raise AssertionError(f"jump instruction {inst.opname} has no target")
            target = _get_instruction_front(instructions, indexof[inst.target])
            if inst.opcode in dis.hasjabs:
                if sys.version_info < (3, 11):
                    # `arg` is expected to be bytecode offset, whereas `offset` is byte offset.
                    # Divide since bytecode is 2 bytes large.
                    if target.offset is None:
                        raise AssertionError("absolute jump target has no offset")
                    inst.arg = int(target.offset / 2)
                else:
                    raise RuntimeError("Python 3.11+ should not have absolute jumps")
            else:  # relative jump
                # byte offset between target and next instruction
                if target.offset is None:
                    raise AssertionError("relative jump target has no offset")
                if inst.offset is None:
                    raise AssertionError(
                        f"relative jump instruction {inst.opname} has no offset"
                    )
                inst.arg = abs(
                    int(target.offset - inst.offset - instruction_size(inst))
                )
                # pyrefly: ignore [unsupported-operation]
                inst.arg //= 2
            inst.argval = target.offset
            inst.argrepr = f"to {target.offset}"


def virtualize_exception_table(
    exn_tab_bytes: bytes, instructions: list[Instruction]
) -> None:
    # [FUNCTION DOC]
    # Parses the raw exception table and attaches to each covered instruction an
    # InstructionExnTabEntry (a copy) whose start, end and target are Instruction
    # objects. Walks instructions in order alongside the sorted table. The end is
    # resolved to the last instruction starting at or before the entry's last byte.
    # Mutates instructions; raises an error if any instruction lacks an offset.
    """Replace exception table entries with pointers to make editing easier"""
    exn_tab = parse_exception_table(exn_tab_bytes)
    offset_to_inst = {cast(int, inst.offset): inst for inst in instructions}
    offsets = sorted(offset_to_inst.keys())
    end_offset_idx = 0
    exn_tab_iter = iter(exn_tab)
    try:

        def step() -> tuple[ExceptionTableEntry, InstructionExnTabEntry]:
            # [FUNCTION DOC] HELPER: Advances to the next raw table entry (StopIteration
            # when exhausted) and returns it with its instruction-based equivalent.
            nonlocal end_offset_idx
            entry = next(exn_tab_iter)
            # find rightmost offset <= entry.end, since entry.end may not be
            # an actual instruction, e.g. if the end instruction is LOAD_GLOBAL,
            # which takes more than 2 bytes, then entry.end points to the end
            # of the LOAD_GLOBAL instruction, not the beginning.
            while (
                end_offset_idx < len(offsets) and offsets[end_offset_idx] <= entry.end
            ):
                end_offset_idx += 1
            if end_offset_idx <= 0:
                raise AssertionError(
                    "end_offset_idx must be positive in exception table virtualization"
                )
            end_offset = offsets[end_offset_idx - 1]
            inst_entry = InstructionExnTabEntry(
                _get_instruction_by_offset(offset_to_inst, entry.start),  # type: ignore[arg-type]
                _get_instruction_by_offset(offset_to_inst, end_offset),  # type: ignore[arg-type]
                _get_instruction_by_offset(offset_to_inst, entry.target),  # type: ignore[arg-type]
                entry.depth,
                entry.lasti,
            )
            return entry, inst_entry

        entry, inst_entry = step()
        for inst in instructions:
            if inst.offset is None:
                raise AssertionError(
                    f"instruction {inst.opname} has no offset during exception "
                    "table virtualization"
                )
            while inst.offset > entry.end:
                entry, inst_entry = step()
            if inst.offset >= entry.start:
                inst.exn_tab_entry = copy.copy(inst_entry)
    except StopIteration:
        pass


def compute_exception_table(
    instructions: list[Instruction],
) -> list[ExceptionTableEntry]:
    # [FUNCTION DOC]
    # Builds an offset-based exception table from instructions' exn_tab_entry fields
    # (offsets must be current). Collects distinct (start, end) ranges, extending starts
    # to include extended-arg prefixes, raising an error on conflicting handlers for the
    # same range. Because Python forbids overlap, nested ranges are flattened: outer
    # ranges are split around inner ones. Validates the result and returns it.
    """Compute exception table in list format from instructions with exn_tab_entries"""
    exn_dict: dict[tuple[int, int], tuple[int, int, bool]] = {}
    indexof = get_indexof(instructions)

    for inst in instructions:
        if inst.exn_tab_entry:
            # account for prefixed EXTENDED_ARGS
            start = _get_instruction_front(
                instructions, indexof[inst.exn_tab_entry.start]
            ).offset
            if start is None:
                raise AssertionError(
                    "exception table entry start instruction has no offset"
                )
            # point to the last 2 bytes of the end instruction
            end = (
                cast(int, inst.exn_tab_entry.end.offset)
                + instruction_size(inst.exn_tab_entry.end)
                - 2
            )
            if end is None:
                raise AssertionError(
                    "exception table entry end instruction has no offset"
                )
            target = _get_instruction_front(
                instructions, indexof[inst.exn_tab_entry.target]
            ).offset
            if target is None:
                raise AssertionError(
                    "exception table entry target instruction has no offset"
                )
            key = (start, end)
            val = (target, inst.exn_tab_entry.depth, inst.exn_tab_entry.lasti)
            if key in exn_dict:
                if exn_dict[key] != val:
                    raise AssertionError(
                        f"conflicting exception table entries for range "
                        f"{key}: {exn_dict[key]} vs {val}"
                    )
            exn_dict[key] = val

    # Dynamo may construct nested exception table entries for convenience,
    # but Python expects exception table entries to not overlap.
    # NOTE: below, "keys" refer to old instruction entries' starts and ends,
    # and "entries" refer to the generated exception table entries.

    # Sort keys by increasing start, then decreasing end
    keys_sorted = sorted(exn_dict.keys(), key=lambda t: (t[0], -t[1]))
    # smallest byte that the next exception table entry can start at
    nexti = 0
    # stack of current nested keys
    key_stack: list[tuple[int, int]] = []
    exn_tab: list[ExceptionTableEntry] = []

    def pop() -> None:
        # [FUNCTION DOC] HELPER: Pops the innermost open range and, if any of it lies
        # at or after the next free byte, appends a flat entry for it and advances the
        # next free byte.
        """
        Pop the key_stack and append an exception table entry if possible.
        """
        nonlocal nexti
        if key_stack:
            key = key_stack.pop()
            if nexti <= key[1]:
                exn_tab.append(
                    ExceptionTableEntry(max(key[0], nexti), key[1], *exn_dict[key])
                )
                nexti = key[1] + 2

    for key in keys_sorted:
        # pop keys that are no longer nested over the current key
        while key_stack and key_stack[-1][1] < key[0]:
            pop()
        if key_stack:
            # create an entry covering to the current key, if possible
            if not (key_stack[-1][0] <= key[0] <= key[1] <= key_stack[-1][1]):
                raise AssertionError(
                    f"exception table key {key} is not properly nested within "
                    f"parent {key_stack[-1]}"
                )
            left = max(nexti, key_stack[-1][0])
            if left < key[0]:
                exn_tab.append(
                    ExceptionTableEntry(left, key[0] - 2, *exn_dict[key_stack[-1]])
                )
            nexti = key[0]
        key_stack.append(key)
    while key_stack:
        pop()
    check_exception_table(exn_tab)
    return exn_tab


def check_inst_exn_tab_entries_nested(
    tab: list[InstructionExnTabEntry], indexof: dict[Instruction, int]
) -> None:
    # [FUNCTION DOC]
    # Verifies the entries (sorted by start ascending then end descending, by
    # instruction index) form a properly nested family with no partial overlaps. Raises
    # an assertion error describing the first violation.
    """
    Checks `tab` is a properly sorted list of nested InstructionExnTabEntry's,
    i.e. no entries partially overlap.
    "Properly sorted" means entries are sorted by increasing starts, then
    decreasing ends.
    """
    entry_stack: list[tuple[int, int]] = []
    for entry in tab:
        key = (indexof[entry.start], indexof[entry.end])
        while entry_stack and entry_stack[-1][1] < key[0]:
            entry_stack.pop()
        if entry_stack:
            if not (entry_stack[-1][0] <= key[0] <= key[1] <= entry_stack[-1][1]):
                raise AssertionError(
                    f"exception table entry {key} is not properly nested "
                    f"within parent {entry_stack[-1]}"
                )
        entry_stack.append(key)


def propagate_inst_exn_table_entries(instructions: list[Instruction]) -> None:
    # [FUNCTION DOC]
    # Gathers distinct entries from instructions that carry one, checks they nest
    # properly, then assigns a copy of each entry to every instruction in its range
    # (inner entries applied later so they win). Raises on conflicting duplicates.
    """
    Copies exception table entries to all instructions in an entry's range.
    Supports nested exception table entries.
    """
    indexof = get_indexof(instructions)
    entries: dict[tuple[int, int], InstructionExnTabEntry] = {}
    for inst in instructions:
        if inst.exn_tab_entry:
            key = (
                indexof[inst.exn_tab_entry.start],
                indexof[inst.exn_tab_entry.end],
            )
            if key in entries:
                if inst.exn_tab_entry != entries[key]:
                    raise AssertionError(
                        f"conflicting exn_tab_entry for instruction at key {key}"
                    )
            entries[key] = inst.exn_tab_entry
    sorted_entries = [
        entries[key] for key in sorted(entries.keys(), key=lambda t: (t[0], -t[1]))
    ]
    check_inst_exn_tab_entries_nested(sorted_entries, indexof)
    # Propagation of nested entries works since nested entries come later
    # in sorted order.
    for entry in sorted_entries:
        for i in range(indexof[entry.start], indexof[entry.end] + 1):
            instructions[i].exn_tab_entry = copy.copy(entry)


def check_inst_exn_tab_entries_valid(instructions: list[Instruction]) -> None:
    # [FUNCTION DOC]
    # Asserts every instruction's exception entry is well formed: the entry object is
    # not shared, its start, end and target are in the list, and the instruction lies
    # within start..end. Also fails on duplicate instructions or on any entry before
    # Python 3.11.
    """
    Checks that exn_tab_entries of instructions are valid.
    An entry's start, end, and target must be in instructions.
    Instructions with an exn_tab_entry are located within
    the entry's start and end instructions.
    Instructions do not share exn_tab_entries.

    Implicitly checks for no duplicate instructions.
    """
    indexof = get_indexof(instructions)
    exn_tab_entry_set = set()
    for i, inst in enumerate(instructions):
        if inst.exn_tab_entry:
            if sys.version_info < (3, 11):
                raise AssertionError("exn_tab_entry is only supported in Python 3.11+")
            if id(inst.exn_tab_entry) in exn_tab_entry_set:
                raise AssertionError(
                    f"duplicate exn_tab_entry at instruction index {i}"
                )
            exn_tab_entry_set.add(id(inst.exn_tab_entry))
            entry = inst.exn_tab_entry
            if entry.start not in indexof:
                raise AssertionError(
                    "exn_tab_entry start instruction not found in instructions"
                )
            if entry.end not in indexof:
                raise AssertionError(
                    "exn_tab_entry end instruction not found in instructions"
                )
            if entry.target not in indexof:
                raise AssertionError(
                    "exn_tab_entry target instruction not found in instructions"
                )
            if not (indexof[entry.start] <= i <= indexof[entry.end]):
                raise AssertionError(
                    f"instruction at index {i} is outside its exn_tab_entry "
                    f"range [{indexof[entry.start]}, {indexof[entry.end]}]"
                )


def strip_extended_args(instructions: list[Instruction]) -> None:
    # [FUNCTION DOC]
    # Removes all extended-arg instructions from the list in place (they are regenerated
    # during assembly).
    instructions[:] = [i for i in instructions if i.opcode != dis.EXTENDED_ARG]


# Overwrites old_inst with a sequence of new instructions.
# This is necessary in order to preserve jump targets to the old
# instruction, exception table entries, and positions.
# Returns the modified sequence of instructions (including the modified
# old instruction!) that can be manipulated elsewhere.
def overwrite_instruction(
    old_inst: Instruction, new_insts: list[Instruction]
) -> list[Instruction]:
    # [FUNCTION DOC]
    # Replaces old_inst by the sequence new_insts without disturbing jumps into it: the
    # old object is mutated to become the first new instruction (opcode, name, arg,
    # argval, target) and the rest copy its exception entry (extending the entry end if
    # old_inst was the end) and positions. Returns the old object followed by the
    # remaining new instructions, which the caller must splice into the list.
    # update old_inst.exn_tab_entry.end if necessary
    if (
        old_inst.exn_tab_entry
        and old_inst.exn_tab_entry.end is old_inst
        and len(new_insts) > 1
    ):
        old_inst.exn_tab_entry.end = new_insts[-1]
    # preserve exception table entries and positions
    for inst in new_insts[1:]:
        inst.exn_tab_entry = copy.copy(old_inst.exn_tab_entry)
        inst.positions = old_inst.positions
    # modify old_inst in-place to preserve jump target
    old_inst.opcode = new_insts[0].opcode
    old_inst.opname = new_insts[0].opname
    old_inst.arg = new_insts[0].arg
    old_inst.argval = new_insts[0].argval
    old_inst.target = new_insts[0].target
    return [old_inst] + new_insts[1:]


def remove_load_call_method(instructions: list[Instruction]) -> list[Instruction]:
    # [FUNCTION DOC]
    # Pre-3.11 only (raises an error on 3.11+). Rewrites method-load into plain
    # attribute-load and method-call into plain function-call in place, avoiding the
    # extra NULL on the stack. Returns the same list.
    """LOAD_METHOD puts a NULL on the stack which causes issues, so remove it"""
    if sys.version_info >= (3, 11):
        raise AssertionError(
            "remove_load_call_method should not be called on Python 3.11+"
        )
    rewrites = {"LOAD_METHOD": "LOAD_ATTR", "CALL_METHOD": "CALL_FUNCTION"}
    for inst in instructions:
        if inst.opname in rewrites:
            inst.opname = rewrites[inst.opname]
            inst.opcode = dis.opmap[inst.opname]
    return instructions


def remove_jump_if_none(instructions: list[Instruction]) -> None:
    # [FUNCTION DOC]
    # Rewrites every jump-if-none / jump-if-not-none instruction into a load of None,
    # an identity test (negated for the not-none form) and a pop-jump-if-true to the same
    # target, using overwrite_instruction to keep inbound jumps valid. Replaces the list
    # contents in place.
    new_insts = []
    for inst in instructions:
        if "_NONE" in inst.opname:
            is_op = create_instruction("IS_OP", arg=int("NOT" in inst.opname))
            # need both argval and arg set correctly now (not later)
            is_op.argval = is_op.arg

            if sys.version_info < (3, 12):
                jump_op = create_instruction(
                    (
                        "POP_JUMP_FORWARD_IF_TRUE"
                        if "FORWARD" in inst.opname
                        else "POP_JUMP_BACKWARD_IF_TRUE"
                    ),
                    target=inst.target,
                )
            else:
                jump_op = create_instruction("POP_JUMP_IF_TRUE", target=inst.target)

            replace_insts = [
                create_instruction("LOAD_CONST", argval=None),
                is_op,
                jump_op,
            ]
            new_insts.extend(overwrite_instruction(inst, replace_insts))
        else:
            new_insts.append(inst)
    instructions[:] = new_insts


def remove_binary_store_slice(instructions: list[Instruction]) -> None:
    # [FUNCTION DOC]
    # Rewrites each slice-load / slice-store instruction into a build-slice (arg 2) in
    # place followed by a new subscript-load / subscript-store, copying exception entry
    # and positions (and moving the entry end if needed). Replaces list contents in
    # place.
    new_insts = []
    for inst in instructions:
        new_insts.append(inst)
        if inst.opname in ("BINARY_SLICE", "STORE_SLICE"):
            # new instruction
            if sys.version_info >= (3, 14) and inst.opname == "BINARY_SLICE":
                subscr_inst = create_binary_subscr()
            else:
                subscr_inst = create_instruction(inst.opname.replace("SLICE", "SUBSCR"))
            if inst.exn_tab_entry and inst.exn_tab_entry.end is inst:
                inst.exn_tab_entry.end = subscr_inst
            subscr_inst.exn_tab_entry = copy.copy(inst.exn_tab_entry)
            subscr_inst.positions = inst.positions
            # modify inst in-place to preserve jump target
            inst.opcode = dis.opmap["BUILD_SLICE"]
            inst.opname = "BUILD_SLICE"
            inst.arg = 2
            inst.argval = 2
            new_insts.append(subscr_inst)
    instructions[:] = new_insts


def remove_load_attr_method_variant(instructions: list[Instruction]) -> None:
    # [FUNCTION DOC]
    # Replaces each method-flavored load-attr (odd arg) by a plain load-attr, a push of
    # NULL, and on versions before 3.13 a swap so NULL ends up below the callable.
    # Uses overwrite_instruction to preserve jumps; replaces list contents in place.
    """On 3.12+, LOAD_ATTR with arg%2==1 is the method variant that pushes 2
    values (NULL/self + method).  Normalize it to a non-method LOAD_ATTR
    (1 value) followed by PUSH_NULL (+ SWAP on 3.12 where NULL goes below
    the callable).  This lets break_graph_if_unsupported handle LOAD_ATTR
    uniformly as a single-output instruction."""
    new_insts: list[Instruction] = []
    for inst in instructions:
        if inst.opname == "LOAD_ATTR" and inst.arg is not None and inst.arg % 2:
            replace_insts = [
                create_instruction("LOAD_ATTR", arg=inst.arg & ~1, argval=inst.argval),
                create_instruction("PUSH_NULL"),
            ]
            if sys.version_info < (3, 13):
                replace_insts.append(create_instruction("SWAP", arg=2))
            new_insts.extend(overwrite_instruction(inst, replace_insts))
        else:
            new_insts.append(inst)
    instructions[:] = new_insts


FUSED_INSTS = {
    "LOAD_FAST_LOAD_FAST": ("LOAD_FAST", "LOAD_FAST"),
    "LOAD_FAST_BORROW_LOAD_FAST_BORROW": ("LOAD_FAST_BORROW", "LOAD_FAST_BORROW"),
    "STORE_FAST_STORE_FAST": ("STORE_FAST", "STORE_FAST"),
    "STORE_FAST_LOAD_FAST": ("STORE_FAST", "LOAD_FAST"),
}


def remove_fused_load_store(instructions: list[Instruction]) -> None:
    # [FUNCTION DOC]
    # Splits each fused two-variable load/store instruction (3.13+) into its two simple
    # instructions using the pair of names in argval, preserving jumps and exception
    # entries via overwrite_instruction. Replaces the list contents in place.
    new_insts = []
    for inst in instructions:
        if inst.opname in FUSED_INSTS:
            inst0, inst1 = FUSED_INSTS[inst.opname]
            argval0, argval1 = inst.argval

            replace_insts = [
                create_instruction(inst0, argval=argval0),
                create_instruction(inst1, argval=argval1),
            ]
            new_insts.extend(overwrite_instruction(inst, replace_insts))
        else:
            new_insts.append(inst)
    instructions[:] = new_insts


# adds GRAPH_BREAK_IF_LEAF (not a real instruction) before RETURN_* instructions
# for testing purposes
def add_graph_break_if_leaf_instructions(instructions: list[Instruction]) -> None:
    # [FUNCTION DOC] HELPER: Testing aid. Before every return instruction, inserts a
    # marker no-op whose argval is a leaf-graph-break tag (keeping jumps valid). Replaces
    # the list contents in place.
    new_insts = []
    for inst in instructions:
        if "RETURN" in inst.opname:
            replace_insts = [
                create_instruction("NOP", argval="GRAPH_BREAK_IF_LEAF"),
                create_instruction(inst.opname, argval=inst.argval),
            ]
            new_insts.extend(overwrite_instruction(inst, replace_insts))
        else:
            new_insts.append(inst)
    instructions[:] = new_insts


def remove_graph_break_if_leaf_instructions(instructions: list[Instruction]) -> None:
    # [FUNCTION DOC] HELPER: Removes leaf-graph-break marker no-ops that directly precede
    # a return instruction, retargeting any jumps and exception-entry start/end/target
    # that referenced the marker to the following return. Replaces list contents in
    # place. Requires a non-empty list.
    new_insts = []
    for inst, next_inst in itertools.pairwise(instructions):
        if (
            inst.opname == "NOP"
            and inst.argval == "GRAPH_BREAK_IF_LEAF"
            and next_inst.opname.startswith("RETURN")
        ):
            # remove this instruction and update all other instructions' jump targets
            for i in range(len(instructions)):
                if instructions[i].target is inst:
                    instructions[i].target = next_inst
                if instructions[i].exn_tab_entry:
                    # linter is mistakenly complaining that None has no attribute "..."
                    # but this codepath only runs if instructions[i] is not None
                    if instructions[i].exn_tab_entry.start is inst:  # type: ignore[union-attr]
                        instructions[i].exn_tab_entry.start = next_inst  # type: ignore[union-attr]
                    if instructions[i].exn_tab_entry.end is inst:  # type: ignore[union-attr]
                        instructions[i].exn_tab_entry.end = next_inst  # type: ignore[union-attr]
                    if instructions[i].exn_tab_entry.target is inst:  # type: ignore[union-attr]
                        instructions[i].exn_tab_entry.target = next_inst  # type: ignore[union-attr]
        else:
            new_insts.append(inst)
    new_insts.append(instructions[-1])
    instructions[:] = new_insts


def explicit_super(code: types.CodeType, instructions: list[Instruction]) -> None:
    # [FUNCTION DOC]
    # For Python before 3.12: after each load of the global super that is immediately
    # called with zero args, inserts loads of the class cell and the first local
    # variable (as a cell load if it is a cell/free var, else a fast load) and changes the
    # call to take two args (also the following call when a precall precedes it on 3.11).
    # Does nothing for functions without locals. Raises an error if no class cell exists.
    # Replaces list contents in place.
    """convert super() with no args into explicit arg form"""
    cell_and_free = (code.co_cellvars or ()) + (code.co_freevars or ())
    if not len(code.co_varnames):
        # A function with no argument cannot contain a valid "super()" call
        return
    output = []
    for idx, inst in enumerate(instructions):
        output.append(inst)
        if inst.opname == "LOAD_GLOBAL" and inst.argval == "super":
            nexti = instructions[idx + 1]
            if nexti.arg == 0 and (
                (sys.version_info >= (3, 12) and nexti.opname == "CALL")
                or (
                    sys.version_info >= (3, 11)
                    and sys.version_info < (3, 12)
                    and nexti.opname == "PRECALL"
                )
                or (sys.version_info < (3, 11) and nexti.opname == "CALL_FUNCTION")
            ):
                if "__class__" not in cell_and_free:
                    raise AssertionError(
                        "__class__ not found in cell_and_free for super() call"
                    )
                output.append(create_instruction("LOAD_DEREF", argval="__class__"))
                first_var = code.co_varnames[0]
                if first_var in cell_and_free:
                    output.append(create_instruction("LOAD_DEREF", argval=first_var))
                else:
                    output.append(create_instruction("LOAD_FAST", argval=first_var))
                nexti.arg = 2
                nexti.argval = 2
                if nexti.opname == "PRECALL":
                    # also update the following CALL instruction
                    call_inst = instructions[idx + 2]
                    call_inst.arg = 2
                    call_inst.argval = 2

    instructions[:] = output


def fix_extended_args(instructions: list[Instruction]) -> int:
    # [FUNCTION DOC]
    # Inserts the extended-arg prefix instructions needed for any instruction whose arg
    # exceeds one byte (up to three prefixes), reusing and trimming existing prefixes
    # but never shrinking the code. Mutates the list in place and returns the number of
    # instructions added; raises an error if the count would be negative.
    """Fill in correct argvals for EXTENDED_ARG ops"""
    output: list[Instruction] = []

    def maybe_pop_n(n: int) -> None:
        # [FUNCTION DOC] HELPER: Removes up to n trailing extended-arg instructions from
        # the output being built (stopping when the last one is not an extended arg).
        for _ in range(n):
            if output and output[-1].opcode == dis.EXTENDED_ARG:
                output.pop()

    for inst in instructions:
        if inst.opcode == dis.EXTENDED_ARG:
            # Leave this instruction alone for now so we never shrink code
            inst.arg = 0
        elif inst.arg and inst.arg > 0xFFFFFF:
            maybe_pop_n(3)
            output.append(create_instruction("EXTENDED_ARG", arg=inst.arg >> 24))
            output.append(create_instruction("EXTENDED_ARG", arg=inst.arg >> 16))
            output.append(create_instruction("EXTENDED_ARG", arg=inst.arg >> 8))
        elif inst.arg and inst.arg > 0xFFFF:
            maybe_pop_n(2)
            output.append(create_instruction("EXTENDED_ARG", arg=inst.arg >> 16))
            output.append(create_instruction("EXTENDED_ARG", arg=inst.arg >> 8))
        elif inst.arg and inst.arg > 0xFF:
            maybe_pop_n(1)
            output.append(create_instruction("EXTENDED_ARG", arg=inst.arg >> 8))
        output.append(inst)

    added = len(output) - len(instructions)
    if added < 0:
        raise AssertionError(
            f"fix_extended_args removed instructions (added={added}), "
            "expected non-negative"
        )
    instructions[:] = output
    return added


def instruction_size(inst: Instruction) -> int:
    # [FUNCTION DOC]
    # Returns the size in bytes of the instruction including inline cache entries: two
    # bytes on Python before 3.11, else two times one plus the opcode's cache entry
    # count taken from the eval-frame extension. Imports torch lazily.
    import torch

    if sys.version_info >= (3, 11):
        return 2 * (torch._C._dynamo.eval_frame.py_opcode_caches[inst.opcode] + 1)
    return 2


def check_offsets(instructions: Sequence[Instruction]) -> None:
    # [FUNCTION DOC]
    # Verifies that each instruction's offset equals the running sum of preceding
    # instruction sizes; raises an assertion error naming the first mismatch.
    offset = 0
    for inst in instructions:
        if inst.offset != offset:
            raise AssertionError(
                f"instruction {inst.opname} has offset {inst.offset}, expected {offset}"
            )
        offset += instruction_size(inst)


def update_offsets(instructions: Sequence[Instruction]) -> None:
    # [FUNCTION DOC]
    # Assigns each instruction an offset equal to the running sum of preceding sizes
    # (mutates the instructions).
    offset = 0
    for inst in instructions:
        inst.offset = offset

        offset += instruction_size(inst)


def debug_bytes(*args: bytes) -> str:
    # [FUNCTION DOC] HELPER: Formats the given byte strings as aligned decimal rows with
    # an index row and a final row flagging positions where the last two differ. Returns
    # a message starting with "bytes mismatch". Requires at least two arguments.
    index = range(max(map(len, args)))
    result = [
        " ".join(f"{x:03}" for x in arg)
        for arg in [index]
        + list(args)
        + [[int(a != b) for a, b in zip(args[-1], args[-2])]]
    ]

    return "bytes mismatch\n" + "\n".join(result)


def debug_checks(code: types.CodeType) -> None:
    # [FUNCTION DOC] HELPER: Debug check that disassembling and reassembling the code
    # object with no transformation (safe mode) reproduces identical raw bytecode;
    # raises an assertion error with a byte diff otherwise.
    """Make sure our assembler produces same bytes as we start with"""
    dode, _ = transform_code_object(code, lambda x, y: None, safe=True)
    if code.co_code != dode.co_code:
        raise AssertionError(debug_bytes(code.co_code, dode.co_code))
    # TODO: reenable after better understanding how to handle sourceless
    # (None) codes
    # assert list(code.co_lines()) == list(dode.co_lines()), (
    #     "line table mismatch via co_lines()"
    # )


HAS_LOCAL = set(dis.haslocal)
HAS_NAME = set(dis.hasname)
HAS_FREE = set(dis.hasfree)
HAS_CONST = set(dis.hasconst)


def get_const_index(code_options: dict[str, Any], val: Any) -> int:
    # [FUNCTION DOC]
    # Returns the index of val in the constants tuple of code_options, comparing by
    # identity (so equal-but-distinct values such as 0.0 and -0.0 stay separate). If not
    # found, appends it to the tuple (mutating code_options) and returns the new index.
    for i, v in enumerate(code_options["co_consts"]):
        # NOTE: stronger comparison is required, since we have
        # examples where two values compare equal but have
        # different semantic meaning in some cases, e.g.
        # 0.0 == -0.0 but have different effects in torch.copysign.
        if val is v:
            return i
    code_options["co_consts"] += (val,)
    return len(code_options["co_consts"]) - 1


def fix_vars(
    instructions: list[Instruction],
    code_options: dict[str, Any],
    varname_from_oparg: Callable[..., Any] | None = None,
) -> None:
    # [FUNCTION DOC]
    # Computes the integer arg of every instruction from its symbolic argval: names (adding
    # missing ones to co_names), local/free variable slots, constant indices (adding
    # missing constants), and the packed forms for load-global, load-attr, load-super-attr
    # (name index shifted with flag bits kept) and fused instructions (3.13+). On 3.11+
    # varname_from_oparg is required to map slots to names; before 3.11 it must be None.
    # Mutates instructions and code_options; raises assertion errors on missing argvals
    # or args.
    # compute instruction arg from argval if arg is not provided
    names = {name: idx for idx, name in enumerate(code_options["co_names"])}

    def get_name_index(name: str) -> int:
        # [FUNCTION DOC] HELPER: Returns the index of the name in co_names, appending it
        # to the names tuple (and the local lookup) when absent.
        try:
            idx = names[name]
        except KeyError:
            # Add a missing item to co_names
            idx = names[name] = len(names)
            code_options["co_names"] = (*code_options["co_names"], name)
            if len(code_options["co_names"]) != len(names):
                raise AssertionError(
                    f"co_names length {len(code_options['co_names'])} does not "
                    f"match names dict length {len(names)}"
                ) from None
        return idx

    if sys.version_info < (3, 11):
        if varname_from_oparg is not None:
            raise AssertionError("varname_from_oparg should be None for Python < 3.11")
        varnames = {name: idx for idx, name in enumerate(code_options["co_varnames"])}
        freenames = {
            name: idx
            for idx, name in enumerate(
                code_options["co_cellvars"] + code_options["co_freevars"]
            )
        }
    else:
        if not callable(varname_from_oparg):
            raise AssertionError("varname_from_oparg must be callable for Python 3.11+")
        allnames = {}
        for idx in itertools.count():
            try:
                name = varname_from_oparg(idx)
                allnames[name] = idx
            except IndexError:
                break
        varnames = {name: allnames[name] for name in code_options["co_varnames"]}
        freenames = {
            name: allnames[name]
            for name in code_options["co_cellvars"] + code_options["co_freevars"]
        }
    for i in range(len(instructions)):

        def should_compute_arg() -> bool:
            # [FUNCTION DOC] HELPER: True if the current instruction has an argval, which
            # takes priority over an existing arg.
            # argval is prioritized over arg
            return instructions[i].argval is not _NotProvided

        if instructions[i].opname == "LOAD_GLOBAL":
            # 3.11 LOAD_GLOBAL requires both arg and argval - see create_instruction
            if instructions[i].argval is _NotProvided:
                raise AssertionError("LOAD_GLOBAL instruction must have argval set")
            if sys.version_info >= (3, 11):
                if instructions[i].arg is None:
                    raise AssertionError(
                        "LOAD_GLOBAL instruction must have arg set in Python 3.11+"
                    )
                instructions[i].arg = (get_name_index(instructions[i].argval) << 1) + (
                    cast(int, instructions[i].arg) % 2
                )
            else:
                instructions[i].arg = get_name_index(instructions[i].argval)
        elif instructions[i].opname == "LOAD_ATTR":
            # 3.12 LOAD_ATTR requires both arg and argval, like LOAD_GLOBAL
            if instructions[i].argval is _NotProvided:
                raise AssertionError("LOAD_ATTR instruction must have argval set")
            if sys.version_info >= (3, 12):
                if instructions[i].arg is None:
                    raise AssertionError(
                        "LOAD_ATTR instruction must have arg set in Python 3.12+"
                    )
                instructions[i].arg = (get_name_index(instructions[i].argval) << 1) + (
                    cast(int, instructions[i].arg) % 2
                )
            else:
                instructions[i].arg = get_name_index(instructions[i].argval)
        elif instructions[i].opname == "LOAD_SUPER_ATTR":
            if instructions[i].arg is None:
                raise AssertionError("LOAD_SUPER_ATTR instruction must have arg set")
            if instructions[i].argval is _NotProvided:
                raise AssertionError("LOAD_SUPER_ATTR instruction must have argval set")
            # Copy low bit, force second bit on for explicit super (the "+ 2")
            instructions[i].arg = (
                (get_name_index(instructions[i].argval) << 2)
                + (cast(int, instructions[i].arg) % 2)
                + 2
            )
        elif instructions[i].opname in FUSED_INSTS:
            if sys.version_info < (3, 13):
                raise AssertionError(
                    f"fused instruction {instructions[i].opname} requires Python 3.13+"
                )
            if not isinstance(instructions[i].argval, tuple):
                raise AssertionError(
                    f"fused instruction {instructions[i].opname} argval must be a tuple, "
                    f"got {type(instructions[i].argval)}"
                )
            if len(instructions[i].argval) != 2:
                raise AssertionError(
                    f"fused instruction {instructions[i].opname} argval must have "
                    f"length 2, got {len(instructions[i].argval)}"
                )
            arg_tuple = tuple(
                varnames[name] if name in varnames else freenames[name]
                for name in instructions[i].argval
            )
            instructions[i].arg = (arg_tuple[0] << 4) + (arg_tuple[1] & 15)
        elif instructions[i].opcode in HAS_LOCAL:
            if should_compute_arg():
                if (
                    sys.version_info >= (3, 13)
                    and instructions[i].argval not in varnames
                ):
                    # instructions like LOAD_FAST used for both local and free vars
                    instructions[i].arg = freenames[instructions[i].argval]
                else:
                    instructions[i].arg = varnames[instructions[i].argval]
        elif instructions[i].opcode in HAS_NAME:
            if should_compute_arg():
                arg = get_name_index(instructions[i].argval)
                if (
                    sys.version_info >= (3, 15)
                    and instructions[i].opname == "IMPORT_NAME"
                ):
                    arg <<= 2
                    # Set the second bit to ensure eager imports, since lazy imports are only valid at module scope
                    arg |= 0x02
                instructions[i].arg = arg
        elif instructions[i].opcode in HAS_FREE:
            if should_compute_arg():
                instructions[i].arg = freenames[instructions[i].argval]
        elif instructions[i].opcode in HAS_CONST:
            # NOTE: only update argval if arg is not provided. This assumes
            # that any additions to co_consts are appended.
            if instructions[i].arg is None:
                # cannot use a dictionary since consts may not be hashable
                idx = get_const_index(code_options, instructions[i].argval)
                if idx < 0:
                    raise AssertionError(
                        f"failed to find constant index for {instructions[i].argval}"
                    )
                instructions[i].arg = idx


def clear_instruction_args(instructions: list[Instruction]) -> None:
    # [FUNCTION DOC]
    # Resets arg to None for instructions that have a symbolic argval and use a
    # name/local/free/const index (except load-global, load-attr, load-super-attr, whose
    # flag bits are in arg), so args are recomputed during assembly. Mutates in place.
    # Clear the instruction arg for instructions that have argvals.
    # Useful for using dis'd bytecode within generated bytecode.
    for inst in instructions:
        if (
            inst.argval is not _NotProvided
            and (
                inst.opcode in HAS_LOCAL
                or inst.opcode in HAS_NAME
                or inst.opcode in HAS_FREE
                or inst.opcode in HAS_CONST
            )
            and inst.opname not in ("LOAD_GLOBAL", "LOAD_ATTR", "LOAD_SUPER_ATTR")
        ):
            inst.arg = None


@functools.lru_cache
def get_code_keys() -> list[str]:
    # [FUNCTION DOC]
    # Returns (cached) the ordered list of code-object attribute names matching the
    # code-object constructor positional order for the running Python version (adds
    # qualified name and exception table on 3.11+).
    # Python 3.11 changes to code keys are not fully documented.
    # See https://github.com/python/cpython/blob/3.11/Objects/clinic/codeobject.c.h#L24
    # for new format.
    keys = ["co_argcount"]
    keys.append("co_posonlyargcount")
    keys.extend(
        [
            "co_kwonlyargcount",
            "co_nlocals",
            "co_stacksize",
            "co_flags",
            "co_code",
            "co_consts",
            "co_names",
            "co_varnames",
            "co_filename",
            "co_name",
        ]
    )
    if sys.version_info >= (3, 11):
        keys.append("co_qualname")
    keys.append("co_firstlineno")
    keys.append("co_linetable")
    if sys.version_info >= (3, 11):
        # not documented, but introduced in https://github.com/python/cpython/issues/84403
        keys.append("co_exceptiontable")
    keys.extend(
        [
            "co_freevars",
            "co_cellvars",
        ]
    )
    return keys


def transform_code_object(
    code: types.CodeType,
    transformations: Callable[
        [list[Instruction], dict[str, Any]], Optional["DynamoTracerOutput"]
    ],
    safe: bool = False,
) -> tuple[types.CodeType, Optional["DynamoTracerOutput"]]:
    # [FUNCTION DOC]
    # Main entry for rewriting a code object. Extracts its attributes into a mutable
    # mapping, disassembles to Instruction objects (cleaned; safe mode skips rewrites),
    # re-propagates line numbers, calls the transformations callback with (instructions,
    # code options) to mutate them in place, then assembles a new code object. Returns
    # (new code, whatever the callback returned). Raises an error if local variable name
    # count disagrees with the local count.
    keys = get_code_keys()
    code_options = {k: getattr(code, k) for k in keys}
    if len(code_options["co_varnames"]) != code_options["co_nlocals"]:
        raise AssertionError(
            f"co_varnames length {len(code_options['co_varnames'])} does not "
            f"match co_nlocals {code_options['co_nlocals']}"
        )

    instructions = cleaned_instructions(code, safe)
    # propagate line nums again for added instructions
    propagate_line_nums(instructions)

    tracer_output = transformations(instructions, code_options)
    _, bytecode = clean_and_assemble_instructions(instructions, keys, code_options)
    return bytecode, tracer_output


def clean_and_assemble_instructions(
    instructions: list[Instruction], keys: list[str], code_options: dict[str, Any]
) -> tuple[list[Instruction], types.CodeType]:
    # [FUNCTION DOC]
    # Finalizes edited instructions into a code object. Drops leaf graph-break markers,
    # validates exception entries, refreshes the local count, resolves args from argvals,
    # then iterates offsets/jump resolution/extended-arg insertion to a fixed point.
    # Assembles bytes and line table, computes stack size, assembles the exception table
    # (3.11+), and builds the code object from code_options in key order. Returns
    # (instructions, new code). Mutates instructions and code_options; raises on key
    # mismatches or invalid exception entries.
    remove_graph_break_if_leaf_instructions(instructions)
    # also implicitly checks for no duplicate instructions
    check_inst_exn_tab_entries_valid(instructions)

    code_options["co_nlocals"] = len(code_options["co_varnames"])
    varname_from_oparg = None
    if sys.version_info >= (3, 11):
        # temporary code object with updated names
        tmp_code = types.CodeType(*[code_options[k] for k in keys])
        varname_from_oparg = tmp_code._varname_from_oparg  # type: ignore[attr-defined]
    fix_vars(instructions, code_options, varname_from_oparg=varname_from_oparg)

    dirty = True
    while dirty:
        update_offsets(instructions)
        devirtualize_jumps(instructions)
        # this pass might change offsets, if so we need to try again
        dirty = bool(fix_extended_args(instructions))

    remove_extra_line_nums(instructions)
    bytecode, lnotab = assemble(instructions, code_options["co_firstlineno"])

    code_options["co_linetable"] = lnotab
    code_options["co_code"] = bytecode
    code_options["co_stacksize"] = stacksize_analysis(instructions)
    if set(keys) - {"co_posonlyargcount"} != set(code_options.keys()) - {
        "co_posonlyargcount"
    }:
        raise AssertionError(
            f"code_options keys mismatch: expected {set(keys) - {'co_posonlyargcount'}}, "
            f"got {set(code_options.keys()) - {'co_posonlyargcount'}}"
        )
    if sys.version_info >= (3, 11):
        code_options["co_exceptiontable"] = assemble_exception_table(
            compute_exception_table(instructions)
        )

    return instructions, types.CodeType(*[code_options[k] for k in keys])


def populate_kw_names_argval(instructions: Sequence[Instruction], consts: Any) -> None:
    # [FUNCTION DOC]
    # For each keyword-names instruction, sets argval to the constant at index arg in the
    # given constants (so the tuple of names is available symbolically). Mutates in place.
    for inst in instructions:
        if inst.opname == "KW_NAMES":
            inst.argval = consts[inst.arg]


# If safe=True, we do not make any bytecode modifications.
# Mainly used for debugging bytecode_transformation (see debug_checks)
def cleaned_instructions(code: types.CodeType, safe: bool = False) -> list[Instruction]:
    # [FUNCTION DOC]
    # Returns a fresh, editable list of normalized Instructions for the code object.
    # Uses a cache keyed by (code, safe) and returns a deep-enough clone each call
    # since callers mutate the result. Safe mode skips bytecode rewrites.
    instructions = _cached_cleaned_instructions(code, safe)
    # We have a lot of code that implicitly mutates the instruction array. We
    # could do better here by making the copies explicit when necessary.
    return _clone_instructions(instructions)


# Copy an instructions array, making sure to remap the individual instruction targets.
def _clone_instructions(instructions: Sequence[Instruction]) -> list[Instruction]:
    # [FUNCTION DOC] HELPER: Copies instructions field by field and remaps jump targets
    # and exception entries (new entry objects) to point at the copies. Hot path; avoids
    # generic copy utilities.
    # This is super hot and this is the fastest way to do this (tried copy.copy
    # and dataclasses.replace).
    copied = [
        Instruction(
            i.opcode,
            i.opname,
            i.arg,
            i.argval,
            i.offset,
            i.starts_line,
            i.is_jump_target,
            i.positions,
            i.target,
            i.exn_tab_entry,
            i.argrepr,
        )
        for i in instructions
    ]

    remap = dict(zip(instructions, copied))
    # Handle `None` in the remapper so we don't need an extra `if`.
    remap[None] = None  # type: ignore[index, assignment]

    for i in copied:
        i.target = remap[i.target]  # type: ignore[index]
        if entry := i.exn_tab_entry:
            i.exn_tab_entry = InstructionExnTabEntry(
                remap[entry.start],
                remap[entry.end],
                remap[entry.target],
                entry.depth,
                entry.lasti,
            )
    return copied


@functools.lru_cache
def _cached_cleaned_instructions(
    code: types.CodeType, safe: bool = False
) -> Sequence[Instruction]:
    # [FUNCTION DOC] HELPER: Cached worker for cleaned_instructions. Disassembles the code,
    # propagates line numbers, checks offsets, virtualizes keyword names, the exception
    # table and jumps, strips extended args, and (unless safe) applies version-specific
    # rewrites (method calls, explicit super, jump-if-none, slices, load-attr method
    # variant, fused loads/stores, optional leaf-return graph breaks). Finally recomputes
    # offsets and jump args on 3.11+. The returned list is shared by the cache and must
    # not be mutated.
    instructions = list(map(convert_instruction, dis.get_instructions(code)))
    # propagate now in case we remove some instructions
    propagate_line_nums(instructions)
    check_offsets(instructions)
    if sys.version_info >= (3, 11):
        populate_kw_names_argval(instructions, code.co_consts)
        virtualize_exception_table(code.co_exceptiontable, instructions)
    virtualize_jumps(instructions)
    strip_extended_args(instructions)
    if not safe:
        if sys.version_info < (3, 11):
            remove_load_call_method(instructions)
        if sys.version_info < (3, 12):
            explicit_super(code, instructions)
        if sys.version_info >= (3, 11):
            remove_jump_if_none(instructions)
            if sys.version_info >= (3, 12):
                remove_binary_store_slice(instructions)
                remove_load_attr_method_variant(instructions)
            if sys.version_info >= (3, 13):
                remove_fused_load_store(instructions)
        if config.debug_force_graph_break_on_leaf_return:
            add_graph_break_if_leaf_instructions(instructions)
    if sys.version_info >= (3, 11):
        update_offsets(instructions)
        devirtualize_jumps(instructions)
    return instructions


_unique_id_counter = itertools.count()


def unique_id(name: str, with_uuid: bool = False) -> str:
    # [FUNCTION DOC]
    # Returns the name suffixed with an underscore and the next value of a process-global
    # counter; optionally also a random uuid (with dashes turned into underscores).
    # Side effect: advances the counter.
    ret = f"{name}_{next(_unique_id_counter)}"
    if with_uuid:
        ret += f"_{uuid.uuid4()}".replace("-", "_")
    return ret


COMPILED_FN_PREFIX = "__compiled_fn"
_COMPILED_FN_NAME_RE = re.compile(
    rf"^{COMPILED_FN_PREFIX}_\d+_"
    r"[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}$"
)


def make_compiled_fn_name() -> str:
    # [FUNCTION DOC]
    # Returns a new unique name for a compiled graph function: the compiled-function
    # prefix plus counter and uuid (matches the pattern checked by is_compiled_fn_name).
    return unique_id(COMPILED_FN_PREFIX, with_uuid=True)


def is_compiled_fn_name(name: str) -> bool:
    # [FUNCTION DOC]
    # True if the whole name matches the generated compiled-function name pattern
    # (prefix, counter, then a uuid with underscores).
    return _COMPILED_FN_NAME_RE.fullmatch(name) is not None


def is_generator(code: types.CodeType) -> bool:
    # [FUNCTION DOC]
    # True if the code object's flags include the generator flag.
    co_generator = 0x20
    return (code.co_flags & co_generator) > 0


def bytecode_from_template(
    fn: Callable[..., Any],
    varname_map: Mapping[Any, Any] | None = None,
    noreturn: bool = True,
    noprefix: bool = True,
) -> list[Instruction]:
    # [FUNCTION DOC]
    # Produces a reusable Instruction list from a template function's code. Clears
    # symbolic-arg-derived args, optionally drops everything through the first resume
    # (noprefix), clears line info, and renames argvals via varname_map. With noreturn,
    # converts return-constant to constant load (3.12+) and replaces returns by jumps to
    # a final no-op (a single trailing return is just removed); returned values stay on
    # the stack. Returns the list of instructions.
    """Generates bytecode from a template function `fn` for use in
    dynamo bytecode generation.

    For example, we can generate Python-version-independent bytecode
    for looping through a dictionary and copying the values to a new dictionary.

    def template(d1, d2):
        for k, v in d1.items():
            d2[k] = v


    or a try block:

    def template():
        try:
            dummy1
        except:
            dummy2
            raise
        dummy3

    Args:
        fn: a function template to generate bytecode from
        varname_map: a mapping of `fn`'s varnames to new names. This
            map will be applied to the generated bytecode's varnames.
            For example, local variables in `fn` can be replaced with
            new names that are generated by `OutputGraph.new_var`.
        noreturn: remove all RETURN_* bytecodes and replace them with a jump
            to the end of the bytecode. NOTE: any items pushed to the stack
            for return WILL remain on the stack! Append a POP_TOP if you don't want
            that item to be present.
        noprefix: remove prefix bytecodes (all bytecode before the first RESUME, inclusive).
    """
    insts = cleaned_instructions(fn.__code__)
    clear_instruction_args(insts)

    if noprefix:
        for i, inst in enumerate(insts):
            if inst.opname == "RESUME":
                insts = insts[i + 1 :]
                break

    for inst in insts:
        # If we don't reset starts_line, then the generated
        # bytecode's line number will be based on fn's.
        inst.starts_line = None
        inst.positions = None
        if varname_map and inst.argval in varname_map:
            inst.argval = varname_map[inst.argval]

    if noreturn:
        if sys.version_info >= (3, 12):
            # replace RETURN_CONST with LOAD_CONST RETURN_VALUE
            new_insts = []
            for inst in insts:
                if inst.opname == "RETURN_CONST":
                    inst.opcode = dis.opmap["LOAD_CONST"]
                    inst.opname = "LOAD_CONST"
                    new_insts.append(inst)
                    # no need to propagate target/exn table
                    new_insts.append(create_instruction("RETURN_VALUE"))
                else:
                    new_insts.append(inst)
            insts = new_insts

        # pyrefly: ignore [implicit-any]
        returns = []
        for inst in insts:
            if inst.opname == "RETURN_VALUE":
                returns.append(inst)

        if len(returns) == 1 and returns[0] is insts[-1]:
            # only 1 return at the end - just pop it
            insts.pop(-1)
        elif len(returns) > 0:
            # create jump target - if the last inst is a return,
            # we can replace it with a NOP and make that the jump target.
            if insts[-1] is returns[-1]:
                insts[-1].opname = "NOP"
                insts[-1].opcode = dis.opmap["NOP"]
                insts[-1].arg = None
                insts[-1].argval = _NotProvided
                returns.pop(-1)
            else:
                insts.append(create_instruction("NOP"))

            # replace returns with jumps
            for inst in returns:
                # don't replace inst with new instruction
                # due to targeting/exn table/etc.
                jump_inst = create_jump_absolute(insts[-1])
                inst.opname = jump_inst.opname
                inst.opcode = jump_inst.opcode
                inst.arg = jump_inst.arg
                inst.argval = jump_inst.argval
                inst.target = jump_inst.target

    return insts
