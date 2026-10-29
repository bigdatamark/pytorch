# [MODULE DOC] CUDA plugin checks: three Check subclasses emitting generated code that runs the
# fuzzed program eagerly and under full-graph dynamic torch.compile, optionally with backward
# and a numeric comparison of output sums.
"""Concrete Check subclasses provided by the CUDA device plugin."""

from torchfuzz.checks import Check


class EagerVsFullGraphDynamicCompileCheck(Check):
    # [CLASS DOC] Check whose generated code runs the program eagerly, then compiled with full
    # graph and dynamic shapes, only verifying that both runs complete (no output comparison).
    """Standard check that runs eager then fullgraph+dynamic compilation."""

    def codegen(self, args_tuple: str) -> list[str]:
        # [FUNCTION DOC] Returns code lines that bind the arguments, run the program eagerly,
        # compile it with full graph and dynamic shapes, run it again, and print success
        # markers after each run.
        return [
            f"args = {args_tuple}",
            "result_original = fuzzed_program(*args)",
            "print('✅ eager success')",
            "compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)",
            "result_compiled = compiled_program(*args)",
            "print('✅ compile success')",
        ]


class EagerVsFullGraphDynamicCompileWithBackwardCheck(Check):
    # [CLASS DOC] Like the plain eager vs compile check, but the generated code also runs the
    # backward pass (on the summed output) in both the eager and the compiled run.
    """Check that runs eager then fullgraph+dynamic compilation with backward pass."""

    def codegen(self, args_tuple: str) -> list[str]:
        # [FUNCTION DOC] Returns code lines that run the program eagerly and compiled, each
        # followed by a backward call on the summed result, printing success markers.
        return [
            f"args = {args_tuple}",
            "result_original = fuzzed_program(*args)",
            "result_original.sum().backward()",
            "print('✅ eager + backward success')",
            "compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)",
            "result_compiled = compiled_program(*args)",
            "result_compiled.sum().backward()",
            "print('✅ compile + backward success')",
        ]


class EagerVsFullGraphDynamicCompileWithNumericsCheck(Check):
    # [CLASS DOC] Check whose generated code runs eager and compiled (with backward) and then
    # compares the sums of the two outputs, failing the generated program when both the
    # relative and absolute differences are large.
    # KNOWN ISSUE: the class docstring says it compares forward numerics only, but the generated
    # code also runs the backward pass.
    """Check that runs eager and compiled, compares forward numerics."""

    def codegen(self, args_tuple: str) -> list[str]:
        # [FUNCTION DOC] Returns code lines that run eager and compiled programs with backward,
        # then compute the absolute and percent relative difference of the output sums and print
        # them. If the relative difference exceeds 5 percent and the absolute one exceeds 1, the
        # generated code prints details and exits with status 1.
        # KNOWN ISSUE: one printed message line is an f-string without placeholders (harmless).
        return [
            f"args = {args_tuple}",
            "out_eager = fuzzed_program(*args)",
            "out_eager.sum().backward()",
            "print('Eager Success! ✅')",
            "compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)",
            "out_compiled = compiled_program(*args)",
            "out_compiled.sum().backward()",
            "print('Compile Success! ✅')",
            "out_eager_sum = out_eager.sum()",
            "out_compiled_sum = out_compiled.sum()",
            "diff = (out_eager_sum - out_compiled_sum).abs().item()",
            "rel_diff = diff / (out_eager_sum.abs().item() + 1e-12) * 100",
            "print(f'Relative diff (sum): {rel_diff:.6f}%')",
            "if rel_diff > 5 and diff > 1:",
            "    print(f'❌ Forward output sums differ significantly (relative and absolute)!')",
            "    print('out_eager_sum:', out_eager_sum.item())",
            "    print('out_compiled_sum:', out_compiled_sum.item())",
            "    print('Absolute diff:', diff)",
            "    print('Relative diff (%):', rel_diff)",
            "    import sys; sys.exit(1)",
        ]
