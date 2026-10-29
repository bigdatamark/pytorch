# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for torch._dynamo bytecode_transformation and bytecode_analysis: linetable
# and position round-tripping, jump offsets, exception table encode/parse/compute/propagate,
# dead-code removal, bytecode_from_template, and the bytecode hook. Many tests are gated by
# Python version (3.10 vs 3.11+ vs 3.12+). No import-time side effects besides imports;
# run_tests is invoked under the main guard.

import collections
import dis
import sys
import unittest

import torch
import torch._dynamo.test_case
from torch._dynamo import bytecode_analysis, bytecode_transformation
from torch._dynamo.testing import skipIfNotPy311, skipIfNotPy312


def coalesced_co_lines(code):
    # [FUNCTION DOC] HELPER: Return the code object's co_lines() entries as a list of (start,
    # end, lineno) tuples, merging adjacent entries that have the same line number and touch
    # (previous end equals next start). Used to compare linetables without the benign
    # non-coalesced differences the assembler can emit.
    # co_lines() entries are not guaranteed to be maximally coalesced: our
    # linetable assembler can emit adjacent entries with the same line number
    # (e.g. around EXTENDED_ARG) that CPython's compiler would fold into one.
    # This is a benign encoding difference (co_positions is unaffected), so
    # merge contiguous same-line entries before comparing.
    result = []
    for start, end, lineno in code.co_lines():
        if result and result[-1][2] == lineno and result[-1][1] == start:
            result[-1] = (result[-1][0], end, lineno)
        else:
            result.append((start, end, lineno))
    return result


class BytecodeTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case (dynamo TestCase) with checks on bytecode assembly and
    # transformation utilities: linetable writers, jump offsets, exception-table handling,
    # template bytecode extraction and analysis regressions.
    @skipIfNotPy311
    def test_linetable_311_writer1(self):
        # [FUNCTION DOC] Reassemble a simple function's cleaned instructions with
        # clean_and_assemble_instructions and assert the new code's co_positions equal the
        # original's, and that coalesced co_lines match. Python 3.11+ only.
        # Nested fn is the function under test; g is an undefined global that is never called.
        def fn():
            # [FUNCTION DOC] HELPER: Sample function with several locals and an f-string; never
            # called, only its code object is inspected.
            a = 10
            b = 20
            # prevent LOAD_FAST_LOAD_FAST in 3.13 by wrapping b with g()
            c = a + g(b)
            f = "linetable_writer"
            return f"Test if {f} generates correct co_linetable: {c}"

        keys = bytecode_transformation.get_code_keys()
        code_options = {k: getattr(fn.__code__, k) for k in keys}
        result = bytecode_transformation.clean_and_assemble_instructions(
            bytecode_transformation.cleaned_instructions(fn.__code__),
            keys,
            code_options,
        )
        l1, l2 = list(fn.__code__.co_positions()), list(result[1].co_positions())
        self.assertEqual(len(l1), len(l2))
        for p1, p2 in zip(l1, l2):
            self.assertEqual(p1, p2)
        self.assertEqual(coalesced_co_lines(fn.__code__), coalesced_co_lines(result[1]))

    @skipIfNotPy311
    def test_linetable_311_writer2(self):
        # [FUNCTION DOC] Build a function via exec with many local assignments (forcing
        # EXTENDED_ARG) and an attribute-method call, reassemble it, and assert EXTENDED_ARG and
        # the LOAD_METHOD/LOAD_ATTR opcode survive in the instruction list. Then assert the
        # reassembled code's co_positions match the per-instruction positions (repeated by
        # instruction size). Python 3.11+ only.
        """
        test large ops (LOAD_METHOD) and EXTENDED_ARGS
        fn_str is in the form:
        def fn():
            ...
            x0 = 1
            x1 = 1
            ...
            l = [x0, x1, ...]
        """
        fn_str = f"""\
def fn():
    foo.bar(1, 2, 3)
{str(chr(10)).join(" " * 4 + "x" + str(i) + " = 1" for i in range(1 << 9))}
    l = [{" ".join("x" + str(i) + "," for i in range(1 << 9))}]
        """
        locals = {}
        exec(fn_str, {}, locals)
        fn = locals["fn"]
        orig_inst_str = "\n".join(list(map(str, dis.get_instructions(fn))))
        self.assertIn("EXTENDED_ARG", orig_inst_str)
        load_method_str = "LOAD_ATTR" if sys.version_info >= (3, 12) else "LOAD_METHOD"
        self.assertIn(load_method_str, orig_inst_str)
        keys = bytecode_transformation.get_code_keys()
        code_options = {k: getattr(fn.__code__, k) for k in keys}
        result = bytecode_transformation.clean_and_assemble_instructions(
            bytecode_transformation.cleaned_instructions(fn.__code__),
            keys,
            code_options,
        )
        new_inst_str = "\n".join(list(map(str, result[0])))
        self.assertIn("EXTENDED_ARG", new_inst_str)
        self.assertIn(load_method_str, new_inst_str)
        # Verify positions by checking the reassembled code's co_positions()
        # matches the positions we set on each instruction. We compare against
        # the instruction list rather than the original code because
        # cleaned_instructions may normalize the instruction set (e.g.,
        # LOAD_ATTR method variant -> LOAD_ATTR + PUSH_NULL on 3.12+).
        expected_positions = []
        for inst in result[0]:
            if inst.opname == "EXTENDED_ARG":
                expected_positions.append(inst.positions)
            else:
                n = bytecode_transformation.instruction_size(inst) // 2
                expected_positions.extend([inst.positions] * n)
        actual_positions = list(result[1].co_positions())
        self.assertEqual(len(expected_positions), len(actual_positions))
        for ep, ap in zip(expected_positions, actual_positions):
            self.assertEqual(ep, ap)

    @unittest.skipIf(
        sys.version_info >= (3, 11),
        "linetable test for Python 3.10",
    )
    def test_linetable_310_writer(self):
        # [FUNCTION DOC] On Python 3.10 only, assemble the instructions of a simple function
        # with bytecode_transformation.assemble and assert the resulting linetable equals the
        # original co_linetable.
        def fn():
            # [FUNCTION DOC] HELPER: Sample function whose bytecode is assembled; never called.
            a = 10
            b = 20
            c = a + b
            f = "linetable_writer"
            return f"Test if {f} generates correct co_linetable: {c}"

        inst = dis.get_instructions(fn)
        result = bytecode_transformation.assemble(inst, fn.__code__.co_firstlineno)
        self.assertTrue(result[1] == fn.__code__.co_linetable)

    def test_if_tensor_is_none(self):
        # [FUNCTION DOC] Compile a function that tests a value against None (is / is not) with
        # the eager backend and fullgraph, and assert the result is 6 when called with None and
        # a tensor. On 3.11+ also assert cleaned_instructions yields no opcode whose name
        # contains _NONE (the new None-jump opcodes are not supported).
        """
        Python 3.11 adds new jump instructions that check if
        TOS is None. We do not support these instructions.
        """

        def f(x, y):
            # [FUNCTION DOC] HELPER: Function under compile; multiplies z by 2 if x is None and
            # by 3 if y is not None, returns z.
            z = 1
            if x is None:
                z *= 2
            if y is not None:
                z *= 3
            return z

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(opt_f(None, torch.ones(2)), 6)

        if sys.version_info >= (3, 11):
            insts = bytecode_transformation.cleaned_instructions(f.__code__)
            for inst in insts:
                self.assertNotIn("_NONE", inst.opname)

    @skipIfNotPy311
    def test_py311_jump_offset(self):
        # [FUNCTION DOC] For each of forward/backward jump opcodes and two targets, assemble a
        # hand-built instruction list (including large LOAD_GLOBAL instructions) and check the
        # final jump instruction's direction is as expected (FORWARD or BACKWARD in its name).
        # Then install the code on a dummy function, call it, and also call it through
        # torch.compile with the eager backend; both must return the expected constant. Python
        # 3.11+ only.
        new_inst = bytecode_transformation.create_instruction
        consts = (None, 1, 2, 3, 4)

        def create_test_code(jump_opname, target_idx):
            # [FUNCTION DOC] HELPER: Build and assemble an instruction list with a RESUME, a
            # forward jump to the jump instruction under test, and two return-value targets; the
            # jump_opname jumps to the target selected by target_idx. Returns the (instructions,
            # code) pair from clean_and_assemble_instructions using a fixed ordered mapping of
            # code options.
            targets = [
                new_inst("LOAD_CONST", argval=1),
                new_inst("LOAD_CONST", argval=3),
            ]
            jump_to_target_inst = new_inst(jump_opname, target=targets[target_idx])
            """
            pseudocode of generated bytecode:
            def test_py311_fn():
                goto target1
            target0:
                return 1
            target1:
                goto [target0/target2] (via fwd or bwd jump)
                return 2
            target2:
                return 3
                return 4
            """
            # test with LOAD_GLOBAL since it has a different instruction size
            insts = [
                new_inst("RESUME", arg=0),
                new_inst("JUMP_FORWARD", target=jump_to_target_inst),
                targets[0],
                new_inst("LOAD_GLOBAL", arg=0, argval="print"),
                new_inst("POP_TOP"),
                new_inst("RETURN_VALUE"),
                jump_to_target_inst,
                new_inst("LOAD_CONST", argval=2),
                new_inst("LOAD_GLOBAL", arg=0, argval="print"),
                new_inst("POP_TOP"),
                new_inst("RETURN_VALUE"),
                targets[1],
                new_inst("RETURN_VALUE"),
                new_inst("LOAD_CONST", argval=4),
                new_inst("RETURN_VALUE"),
            ]
            code_options = collections.OrderedDict(
                [
                    ("co_argcount", 0),
                    ("co_posonlyargcount", 0),
                    ("co_kwonlyargcount", 0),
                    ("co_nlocals", 0),
                    ("co_stacksize", 2),
                    ("co_flags", 3),
                    ("co_code", b""),
                    ("co_consts", consts),
                    ("co_names", ("print",)),
                    ("co_varnames", ()),
                    ("co_filename", __file__),
                    ("co_name", "test_py311_fn"),
                    ("co_qualname", "test_py311_fn"),
                    ("co_firstlineno", 1),
                    ("co_linetable", b""),
                    ("co_exceptiontable", b""),
                    ("co_freevars", ()),
                    ("co_cellvars", ()),
                ]
            )
            return bytecode_transformation.clean_and_assemble_instructions(
                insts,
                list(code_options.keys()),
                code_options,
            )

        # format: jump_opname, target_idx, expected forward jump, expected return value
        test_args = (
            ("JUMP_FORWARD", 0, False, 1),
            ("JUMP_FORWARD", 1, True, 3),
            ("JUMP_BACKWARD", 0, False, 1),
            ("JUMP_BACKWARD", 1, True, 3),
        )

        for test in test_args:
            insts, code = create_test_code(test[0], test[1])
            # check if offset of latest jump instruction is forward/backward
            for inst in reversed(insts):
                if inst.opname.startswith("JUMP"):
                    if test[2]:
                        self.assertIn("FORWARD", inst.opname)
                    else:
                        self.assertIn("BACKWARD", inst.opname)
                    break
            # run the code and check result

            def dummy_fn():
            # [FUNCTION DOC] HELPER: Empty placeholder function whose __code__ is replaced by
            # the assembled code.
                pass

            dummy_fn.__code__ = code
            self.assertEqual(dummy_fn(), test[3])

            dummy_opt = torch.compile(dummy_fn, backend="eager")
            self.assertEqual(dummy_opt(), test[3])

    def test_exception_table_encode_varint(self):
        # [FUNCTION DOC] Round-trip two integers through encode_exception_table_varint and
        # decode_exception_table_varint (decoding from a byte iterator until StopIteration) and
        # assert the decoded list equals the inputs.
        # these numbers have no real meaning to them
        nums = [
            0b111_101010_000000,
            0b1100_111000_010101_101010,
        ]
        b = bytecode_transformation.encode_exception_table_varint(
            nums[0]
        ) + bytecode_transformation.encode_exception_table_varint(nums[1])
        nums_new = []
        b_iter = iter(bytes(b))
        while True:
            try:
                nums_new.append(
                    bytecode_transformation.decode_exception_table_varint(b_iter)
                )
            except StopIteration:
                break
        self.assertEqual(nums, nums_new)

    @skipIfNotPy311
    def test_exception_table_parsing(self):
        # [FUNCTION DOC] Parse a function's co_exceptiontable with parse_exception_table and
        # reassemble it with assemble_exception_table; assert the bytes are identical to the
        # original. Python 3.11+ only.
        def fn():
            # [FUNCTION DOC] HELPER: Function with nested try/with/except/finally; only its
            # exception table is used.
            try:
                with a():
                    b()
                c()
            except Exception:
                d()
            finally:
                e()
            f()

        tab = bytecode_transformation.parse_exception_table(
            fn.__code__.co_exceptiontable
        )
        b = bytecode_transformation.assemble_exception_table(tab)
        self.assertEqual(b, fn.__code__.co_exceptiontable)

    @skipIfNotPy311
    def test_exception_table_e2e(self):
        # [FUNCTION DOC] Run transform_code_object with a no-op transformation on a function
        # containing try/with/except/finally and assert the resulting exception table equals the
        # original. Python 3.11+ only.
        def fn():
            # [FUNCTION DOC] HELPER: Function with nested try/with/except/finally; only its code
            # is used.
            try:
                with a():
                    b()
                c()
            except Exception:
                d()
            finally:
                e()
            f()

        def nothing(*args):
            # [FUNCTION DOC] HELPER: No-op transformation callback that ignores its arguments.
            pass

        code, _ = bytecode_transformation.transform_code_object(fn.__code__, nothing)
        self.assertEqual(code.co_exceptiontable, fn.__code__.co_exceptiontable)

    @skipIfNotPy311
    def test_exception_table_e2e_2(self):
        # [FUNCTION DOC] Same as the end-to-end exception table round trip, but for a function
        # whose try region ends with a large instruction (LOAD_GLOBAL); asserts the exception
        # table is unchanged after a no-op transform_code_object. Python 3.11+ only.
        # last instructions of an exn_table entry is a large instruction
        # i.e., LOAD_GLOBAL a
        def fn():
            # [FUNCTION DOC] HELPER: Function returning a global inside try/except; only its
            # code is used.
            try:
                return a
            except Exception:
                pass

        def nothing(*args):
            # [FUNCTION DOC] HELPER: No-op transformation callback that ignores its arguments.
            pass

        code, _ = bytecode_transformation.transform_code_object(fn.__code__, nothing)
        self.assertEqual(code.co_exceptiontable, fn.__code__.co_exceptiontable)

    @skipIfNotPy311
    def test_exception_table_entry_propagation(self):
        # [FUNCTION DOC] Create ten NOP instructions, attach overlapping exception table entries
        # to some of them, run propagate_inst_exn_table_entries, and assert every instruction
        # ends up with an entry whose target is the expected instruction (the innermost covering
        # entry's target). Python 3.11+ only.
        insts = []
        for _ in range(10):
            insts.append(bytecode_transformation.create_instruction("NOP"))
        insts[8].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[0], insts[9], insts[0], 0, True
        )
        insts[0].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[0], insts[0], insts[1], 0, True
        )
        insts[1].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[0], insts[2], insts[2], 0, True
        )
        insts[5].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[4], insts[6], insts[3], 0, True
        )
        insts[9].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[9], insts[9], insts[4], 0, True
        )
        insts[7].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[7], insts[9], insts[5], 0, True
        )
        bytecode_transformation.propagate_inst_exn_table_entries(insts)
        expected = [1, 2, 2, 0, 3, 3, 3, 5, 5, 4]
        for inst, exp in zip(insts, expected):
            self.assertIsNotNone(inst.exn_tab_entry)
            self.assertIs(inst.exn_tab_entry.target, insts[exp])

    @skipIfNotPy311
    def test_compute_exception_table_nested(self):
        # [FUNCTION DOC] Create twenty NOP instructions with nested and disjoint exception table
        # entries, run update_offsets and compute_exception_table, and assert the resulting
        # table has the expected nine entries with start, end and target equal to the expected
        # instruction indices times two (instruction size). Python 3.11+ only.
        insts = []
        for _ in range(20):
            insts.append(bytecode_transformation.create_instruction("NOP"))
        insts[10].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[1], insts[10], insts[0], 0, True
        )
        insts[0].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[1], insts[1], insts[1], 0, True
        )
        insts[1].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[1], insts[3], insts[2], 0, True
        )
        insts[5].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[5], insts[7], insts[3], 0, True
        )
        insts[9].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[10], insts[10], insts[4], 0, True
        )
        insts[7].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[8], insts[10], insts[5], 0, True
        )
        insts[14].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[13], insts[17], insts[6], 0, True
        )
        insts[16].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[15], insts[16], insts[7], 0, True
        )
        bytecode_transformation.update_offsets(insts)
        tab = bytecode_transformation.compute_exception_table(insts)
        expected = [
            (1, 1, 1),
            (2, 3, 2),
            (4, 4, 0),
            (5, 7, 3),
            (8, 9, 5),
            (10, 10, 4),
            (13, 14, 6),
            (15, 16, 7),
            (17, 17, 6),
        ]
        self.assertEqual(len(tab), len(expected))
        for entry, exp in zip(tab, expected):
            self.assertEqual(entry.start, exp[0] * 2)
            self.assertEqual(entry.end, exp[1] * 2)
            self.assertEqual(entry.target, exp[2] * 2)

    @skipIfNotPy311
    def test_remove_dead_code_with_exn_table_entries(self):
        # [FUNCTION DOC] Build instructions where an exception-table range's start and end
        # markers are in dead code (after an unconditional jump), propagate entries, run
        # remove_dead_code, and assert the dead instructions are removed, the targets remain
        # (five instructions left), and the recomputed table has a single entry with start 2,
        # end 4, target 6. Python 3.11+ only.
        create_instruction = bytecode_transformation.create_instruction
        target1 = create_instruction("NOP")
        target2 = create_instruction("NOP")
        target3 = create_instruction("NOP")
        exn_start = create_instruction("NOP")
        exn_end = create_instruction("NOP")
        insts = [
            create_instruction("JUMP_FORWARD", target=target1),
            exn_start,  # dead
            target1,
            create_instruction("JUMP_FORWARD", target=target3),
            exn_end,  # dead
            target2,
            target3,
        ]
        exn_start.exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            exn_start, exn_end, target2, 0, True
        )
        bytecode_transformation.propagate_inst_exn_table_entries(insts)
        insts = bytecode_analysis.remove_dead_code(insts)
        self.assertEqual(len(insts), 5)
        self.assertNotIn(exn_start, insts)
        self.assertNotIn(exn_end, insts)
        self.assertIn(target2, insts)
        self.assertIn(target3, insts)
        bytecode_transformation.update_offsets(insts)
        tab = bytecode_transformation.compute_exception_table(insts)
        self.assertEqual(len(tab), 1)
        self.assertEqual(tab[0].start, 2)
        self.assertEqual(tab[0].end, 4)
        self.assertEqual(tab[0].target, 6)

    def test_bytecode_from_template(self):
        # [FUNCTION DOC] Call bytecode_from_template on a function with a varname_map renaming
        # locals/globals and assert every instruction has no starts_line, loaded names are not
        # the original mapped names, non-global loads have no arg, and no RETURN instructions
        # remain.
        def fn(d1):
            # [FUNCTION DOC] HELPER: Template function; iterates d1.items() and stores into
            # global d2, never called.
            for k, v in d1.items():
                d2[k] = v

        varname_map = {"d1": "var1", "d2": "var2", "k": "var3", "v": "var4"}
        insts = bytecode_transformation.bytecode_from_template(fn, varname_map)
        for inst in insts:
            self.assertIsNone(inst.starts_line)
            if inst.opname.startswith("LOAD"):
                self.assertNotIn(inst.argval, varname_map)
                if inst.opname not in (
                    "LOAD_GLOBAL",
                    "LOAD_ATTR",
                    "LOAD_COMMON_CONSTANT",
                ):
                    self.assertIsNone(inst.arg)
            self.assertFalse(inst.opname.startswith("RETURN"))

    @skipIfNotPy311
    def test_bytecode_from_template_noprefix(self):
        # [FUNCTION DOC] Assert that for a closure function the plain disassembly contains
        # RESUME and COPY_FREE_VARS but the instructions from bytecode_from_template do not
        # (prefix instructions removed). Python 3.11+ only.
        # Test that 3.11+ prefix instructions are removed
        def gen_fn():
            # [FUNCTION DOC] HELPER: Factory returning a closure that reads a free variable.
            cl = None

            def fn():
            # [FUNCTION DOC] HELPER: Closure returning its free variable; the code under
            # inspection.
                return cl

            return fn

        fn = gen_fn()

        dis_insts = list(dis.get_instructions(fn))
        names = {inst.opname for inst in dis_insts}
        self.assertIn("RESUME", names)
        self.assertIn("COPY_FREE_VARS", names)

        insts = bytecode_transformation.bytecode_from_template(fn)
        names = {inst.opname for inst in insts}
        self.assertNotIn("RESUME", names)
        self.assertNotIn("COPY_FREE_VARS", names)

    def test_bytecode_from_template_noreturn1(self):
        # [FUNCTION DOC] With noprefix False, assert a function with multiple returns converts
        # to the same number of instructions where each RETURN becomes a jump targeting a final
        # NOP appended instruction (the last return becomes the NOP).
        # Test that functions with multiple returns will have their
        # returns replaced with jumps to the end
        def fn():
            # [FUNCTION DOC] HELPER: Template function with two return statements.
            if x:
                return y
            z = 3
            return z

        dis_insts = list(dis.get_instructions(fn))
        dis_returns = list(filter(lambda x: x.opname.startswith("RETURN"), dis_insts))
        self.assertGreater(len(dis_returns), 1)
        self.assertTrue(dis_insts[-1].opname.startswith("RETURN"))

        insts = bytecode_transformation.bytecode_from_template(fn, noprefix=False)
        self.assertEqual(insts[-1].opname, "NOP")
        self.assertEqual(len(dis_insts), len(insts))
        for i0, i1 in zip(dis_insts, insts):
            if i0.opname.startswith("RETURN"):
                if i1 is insts[-1]:
                    continue
                self.assertIn("JUMP", i1.opname)
                self.assertIs(i1.target, insts[-1])

    # Should work with 3.10, but testing with 3.11+ is sufficient.
    # In 3.8, `fn` ends with a RETURN_VALUE.
    @skipIfNotPy311
    def test_bytecode_from_template_noreturn2(self):
        # [FUNCTION DOC] For a function that ends with a raise (no final return), assert
        # bytecode_from_template with noprefix False appends a NOP, keeps the final raise
        # instruction before it, and turns each RETURN into a jump to the NOP. Python 3.11+
        # only.
        # Test function that doesn't end with RETURN_VALUE
        def fn():
            # [FUNCTION DOC] HELPER: Template function with two returns and a trailing raise.
            if x:
                return x
            if x:
                return x
            raise RuntimeError

        dis_insts = list(dis.get_instructions(fn))
        self.assertFalse(dis_insts[-1].opname.startswith("RETURN"))

        insts = bytecode_transformation.bytecode_from_template(fn, noprefix=False)
        self.assertEqual(insts[-1].opname, "NOP")
        self.assertEqual(insts[-2].opname, dis_insts[-1].opname)
        self.assertEqual(len(dis_insts) + 1, len(insts))
        for i0, i1 in zip(dis_insts, insts):
            if i0.opname.startswith("RETURN"):
                self.assertIn("JUMP", i1.opname)
                self.assertIs(i1.target, insts[-1])

    @unittest.skipIf(sys.version_info >= (3, 14), "3.14+ removed RETURN_CONST")
    @skipIfNotPy312
    def test_bytecode_from_template_noreturn_const(self):
        # [FUNCTION DOC] For Python 3.12 and 3.13, assert RETURN_CONST instructions in the
        # template are replaced by LOAD_CONST followed by a jump to the final NOP (except the
        # final one, which is followed directly by the NOP).
        # Test 3.12+ RETURN_CONST
        def fn():
            # [FUNCTION DOC] HELPER: Template function with two constant returns.
            if x:
                return 1
            return 0

        dis_insts = list(dis.get_instructions(fn))
        dis_return_consts = list(
            filter(lambda x: x.opname == "RETURN_CONST", dis_insts)
        )
        self.assertGreater(len(dis_return_consts), 1)
        self.assertTrue(dis_insts[-1].opname == "RETURN_CONST")

        insts = bytecode_transformation.bytecode_from_template(fn, noprefix=False)
        self.assertEqual(insts[-1].opname, "NOP")
        insts_i = 0
        for inst in dis_insts:
            if inst.opname == "RETURN_CONST":
                self.assertEqual(insts[insts_i].opname, "LOAD_CONST")
                insts_i += 1
                if insts_i != len(insts) - 1:
                    self.assertIn("JUMP", insts[insts_i].opname)
                    self.assertIs(insts[insts_i].target, insts[-1])
            insts_i += 1

    def test_bytecode_analysis_jump_backward_no_interrupt(self):
        # [FUNCTION DOC] Regression test: compile with the eager backend a function that
        # graph-breaks and then runs a no_grad block with try/except handlers, so bytecode_analysis
        # analyzes the remainder (JUMP_BACKWARD_NO_INTERRUPT must be treated as terminal in
        # 3.13+). Asserts the result is correct.
        # bytecode_analysis fails if JUMP_BACKWARD_NO_INTERRUPT is not terminal in 3.13+
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function that graph-breaks, then adds 1 to x
            # inside a no_grad try/except.
            # graph break causes bytecode_analysis to analyze the rest of this function
            torch._dynamo.graph_break()
            with torch.no_grad():
                try:
                    x = x + 1
                except NotImplementedError:
                    x = x + 1
                except Exception:
                    x = x + 1
            return x

        self.assertEqual(fn(torch.ones(3)), torch.ones(3) + 1)

    # https://github.com/pytorch/pytorch/issues/160471
    def test_extended_args_starts_line(self):
        # [FUNCTION DOC] Regression test for issue 160471: build via exec a function with 300
        # additions of large constants (producing an EXTENDED_ARG that has starts_line set),
        # assert such an instruction exists, then run transform_code_object with a callback that
        # checks an instruction with starts_line 301 exists. Fails the test if either is
        # missing.
        # NOTE: need to LOAD_CONST i before LOAD_FAST x
        # in order to get an EXTENDED_ARG with starts_line set
        # NOTE: 3.14+ introduced LOAD_SMALL_INT, so integers need to be >= 256
        # in order for LOAD_CONST to be generated
        lines = "\n".join(f"    x = {i + 1000} + x" for i in range(300))
        fn_str = f"def fn(x):\n{lines}"
        locals = {}
        exec(fn_str, {}, locals)
        fn = locals["fn"]

        for inst in dis.get_instructions(fn):
            if inst.opname == "EXTENDED_ARG" and inst.starts_line:
                break
        else:
            self.assertTrue(
                False, "bad test case: no EXTENDED_ARG with starts_line found"
            )

        def transformations(instructions, _):
            # [FUNCTION DOC] HELPER: Transformation callback that scans for an instruction with
            # starts_line 301 (failing the test if absent) and returns the instruction list
            # unchanged.
            for inst in instructions:
                if inst.starts_line == 301:
                    break
            else:
                self.assertTrue(False, "test failure: 301 starts_line not found")
            return instructions

        bytecode_transformation.transform_code_object(fn.__code__, transformations)


class BytecodeHookTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for the bytecode hook registration API in convert_frame.
    def test_bytecode_hook(self):
        # [FUNCTION DOC] Register a bytecode hook that prints the original and generated code
        # and returns the new code, compile a simple function with the eager backend and call it
        # with tensors of ten sizes, then always remove the hook handle. Calls
        # torch._dynamo.reset first.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Function compiled by the test; computes a - b * 10.
            return a - b * 10

        def hook(code, out_code):
            # [FUNCTION DOC] HELPER: Bytecode hook that prints both code objects and returns the
            # first argument.
            print(code)
            print(out_code)
            return code

        torch._dynamo.reset()
        handle = torch._dynamo.convert_frame.register_bytecode_hook(hook)
        try:
            opt_fn = torch.compile(fn, backend="eager")
            for i in range(2, 12):
                opt_fn(torch.randn(i), torch.randn(i))
        finally:
            handle.remove()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
