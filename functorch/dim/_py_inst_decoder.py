# [MODULE DOC]
# Small helper that disassembles a code object and reads the target variable name of a
# store instruction near a given bytecode offset (used to infer names assigned from dims).
import dis
from typing import Any


class _PyInstDecoder:
    # [CLASS DOC] HELPER: Cursor over the disassembled instructions of a code object.
    # State: code_object, the full instruction list, and offset (current instruction index).
    # Positioned at construction at the instruction at or before a given byte offset.
    """
    Decodes Python bytecode instructions to extract variable names
    """

    def __init__(self, code_object: Any, lasti: int) -> None:
        # [FUNCTION DOC] HELPER: Disassembles code_object and positions the cursor at the
        # instruction located at or before byte offset lasti.
        self.code_object = code_object
        self.instructions = list(dis.get_instructions(code_object))
        self.offset = self._find_instruction_index(lasti)

    def _find_instruction_index(self, lasti: int) -> int:
        # [FUNCTION DOC] HELPER: Returns the index of the last instruction whose byte offset
        # is not greater than lasti (0 if none). Pure lookup.
        """Find instruction index corresponding to lasti (byte offset)."""
        # Find the instruction at or before lasti
        # This should find the CALL instruction, not the next one
        best_idx = 0
        for i, instr in enumerate(self.instructions):
            if instr.offset <= lasti:
                best_idx = i
            else:
                break
        return best_idx

    def next(self) -> None:
        # [FUNCTION DOC] HELPER: Moves the cursor forward by one instruction (no bounds check).
        """Advance to the next instruction."""
        self.offset += 1

    def opcode(self) -> str | None:
        # [FUNCTION DOC] HELPER: Returns the opcode name of the current instruction, or None
        # when the cursor is past the end.
        """Get the opcode name of the current instruction."""
        if self.offset < len(self.instructions):
            return self.instructions[self.offset].opname
        return None

    def oparg(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the integer argument of the current instruction, or 0
        # when it has none or the cursor is past the end.
        """Get the argument of the current instruction."""
        if self.offset < len(self.instructions):
            return self.instructions[self.offset].arg or 0
        return 0

    def name(self) -> str | None:
        # [FUNCTION DOC] HELPER: If the current instruction stores into a name, global, local
        # or cell/free variable, returns that variable name looked up from the matching
        # table of the code object; returns None for other opcodes or out-of-range args.
        """
        Extract variable name from current instruction.
        """
        opname = self.opcode()
        if not opname:
            return None

        names = None
        if opname in ("STORE_NAME", "STORE_GLOBAL"):
            names = self.code_object.co_names
        elif opname == "STORE_FAST":
            names = self.code_object.co_varnames
        elif opname == "STORE_DEREF":
            names = self.code_object.co_cellvars
            if not names:
                names = self.code_object.co_freevars
        else:
            return None

        arg = self.oparg()
        if names and 0 <= arg < len(names):
            return names[arg]

        return None
