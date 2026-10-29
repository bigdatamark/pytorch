# [MODULE DOC] Tests for tools/build_with_debinfo.py: the debugify command rewriting and
# create_build_plan, which selects the compile and dependent link commands to rerun with
# debug info. Pure string-processing tests, no I/O.
from __future__ import annotations

import unittest
from pathlib import Path

from tools.build_with_debinfo import create_build_plan, debugify


class TestDebugify(unittest.TestCase):
    # [CLASS DOC] Tests for the debugify command-line rewriting function. Stateless.
    def test_swaps_optimization_for_debug(self) -> None:
        # [FUNCTION DOC] Asserts optimization flags (level two and three) are replaced by
        # the debug flag in a compile command.
        self.assertEqual(debugify("cc -O2 -c a.cpp"), "cc -g -c a.cpp")
        self.assertEqual(debugify("cc -O3 -c a.cpp"), "cc -g -c a.cpp")

    def test_leaves_other_flags_untouched(self) -> None:
        # [FUNCTION DOC] Asserts a command without optimization flags comes back identical.
        cmd = "cc -DNDEBUG -I/x -fPIC -c a.cpp -o a.o"
        self.assertEqual(debugify(cmd), cmd)

    def test_metal_gets_debug_flags_once(self) -> None:
        # [FUNCTION DOC] Asserts a Metal compiler command gains the record-sources and
        # line-tables-only debug flags, and that applying the rewrite again is idempotent.
        out = debugify("xcrun metal -c a.metal")
        self.assertIn("-frecord-sources", out)
        self.assertIn("-gline-tables-only", out)
        # Idempotent: do not append a second time.
        self.assertEqual(out, debugify(out))


class TestCreateBuildPlan(unittest.TestCase):
    # [CLASS DOC] Tests for create_build_plan, which maps changed sources to an ordered
    # list of (label, command) steps. Stateless.
    def test_follows_dependent_links(self) -> None:
        # [FUNCTION DOC] Given a command list with compile and shared-library link steps,
        # asserts the plan for one source holds its debug-flag recompile followed by only
        # the link commands that transitively depend on its object file, in build order,
        # excluding unrelated links.
        commands = "\n".join(
            [
                "c++ -O3 -o obj/other.o -c /repo/other.cpp",
                "c++ -O3 -o obj/a.o -c /repo/a.cpp",
                "c++ -shared -o lib/libtorch_cpu.so obj/a.o obj/other.o",
                "c++ -shared -o lib/libunrelated.so obj/other.o",
                "c++ -shared -o lib/libtorch_python.so lib/libtorch_cpu.so",
            ]
        )
        self.assertEqual(
            create_build_plan(["/repo/a.cpp"], commands, Path("/repo/build")),
            [
                ("debug rebuild", "c++ -g -o obj/a.o -c /repo/a.cpp"),
                (
                    "rebuild dependent",
                    "c++ -shared -o lib/libtorch_cpu.so obj/a.o obj/other.o",
                ),
                (
                    "rebuild dependent",
                    "c++ -shared -o lib/libtorch_python.so lib/libtorch_cpu.so",
                ),
            ],
        )

    def test_handles_metal_custom_commands(self) -> None:
        # [FUNCTION DOC] Asserts a Metal source yields a three step plan (shader compile,
        # library packaging, dependent link) and the first step carries the Metal debug
        # flags.
        commands = "\n".join(
            [
                "cd /repo/build/metal && xcrun metal -c /repo/a.metal -o a.air",
                "cd /repo/build/metal && xcrun metallib -o a.metallib a.air",
                "c++ -shared -Wl,-sectcreate,/repo/build/metal/a.metallib -o lib/a.so",
            ]
        )
        plan = create_build_plan(["/repo/a.metal"], commands, Path("/repo/build"))
        self.assertEqual(len(plan), 3)
        self.assertIn("-frecord-sources -gline-tables-only", plan[0][1])

    def test_raises_when_source_is_absent(self) -> None:
        # [FUNCTION DOC] Asserts a runtime error is raised when a requested source file
        # appears in no compile command.
        with self.assertRaises(RuntimeError):
            create_build_plan(
                ["/repo/missing.cpp"],
                "c++ -o obj/a.o -c /repo/a.cpp",
                Path("/repo/build"),
            )


if __name__ == "__main__":
    unittest.main()
