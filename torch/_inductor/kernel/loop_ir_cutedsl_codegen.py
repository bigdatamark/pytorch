# mypy: allow-untyped-defs
# [MODULE DOC] Converts the loop-body IR of fused GEMM epilogue buffers into the source text of a
# CuTeDSL function that takes the accumulator fragment and extra input fragments and returns the
# output fragments. Main content: LoopIRCuteDSLCodegen. No import-time side effects.
"""Lower Inductor GEMM epilogue loop IR to CuTeDSL source."""

from collections.abc import Sequence
from typing import Any

import torch
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import CuteDSLCSEVariable
from torch._inductor.ir import ComputedBuffer
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.value_ranges import ValueRanges

from .gemm_epilogue_codegen import (
    GemmEpilogueCuteDSLKernel,
    GemmEpilogueCuteDSLOpOverrides,
)
from .loop_ir_epilogue_lowering import GemmEpilogueIRAnalysis, GemmEpilogueIRExpression


class LoopIRCuteDSLCodegen:
    # [CLASS DOC] One-shot lowering driver. State: accumulator buffer name, removed_buffers set,
    # a fake CuTeDSL kernel collecting body lines, reads (ordered set of external input names),
    # input_values (cache of CSE variables per input) and the accum variable (float32). Use
    # from_buffers for the full flow; render emits the function source.
    """Lower scheduler loop-body expressions to a CuTeDSL fragment function."""

    def __init__(self, accumulator: str, removed_buffers: OrderedSet[str]) -> None:
        # [FUNCTION DOC] Store the accumulator buffer name and removed buffers, and create the fake
        # kernel, empty read set, input cache and the float32 accumulator variable named accum.
        self.accumulator = accumulator
        self.removed_buffers = removed_buffers
        self.kernel = GemmEpilogueCuteDSLKernel()
        self.reads: OrderedSet[str] = OrderedSet()
        self.input_values: dict[str, CuteDSLCSEVariable] = {}
        self.accumulator_value = CuteDSLCSEVariable(
            "accum", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
        )

    def load(self, name: str, stored: Any) -> Any:
        # [FUNCTION DOC] Resolve a buffer read. If the buffer was produced inside the epilogue
        # (stored is not None) lower its stored expression; if it is the accumulator return the
        # accum variable; otherwise record it as an external read and return a cached CSE variable
        # typed from the graph dtype. Side effect: updates reads and input_values.
        if stored is not None:
            return self.lower(stored)
        if name == self.accumulator:
            return self.accumulator_value
        self.reads.add(name)
        if name not in self.input_values:
            dtype = V.graph.get_dtype(name)
            self.input_values[name] = CuteDSLCSEVariable(
                name, ValueRanges.unknown(), dtype=dtype, shape=(1,)
            )
        return self.input_values[name]

    def lower(self, value: Any) -> Any:
        # [FUNCTION DOC] Recursively lower an IR expression into CuTeDSL values. Non-expression
        # values pass through (tuples are lowered element-wise). Load nodes go to load, identity
        # nodes lower their argument, reduction and bitcast ops raise NotImplementedError, as does
        # any op missing from GemmEpilogueCuteDSLOpOverrides. Other ops call the matching override
        # with lowered args and kwargs, which may emit lines into the kernel body.
        if not isinstance(value, GemmEpilogueIRExpression):
            if isinstance(value, tuple):
                return tuple(self.lower(item) for item in value)
            return value
        if value.op == "load":
            name, _, stored = value.args
            return self.load(name, stored)
        if value.op == "identity":
            return self.lower(value.args[0])
        if value.op in ("reduction", "to_dtype_bitcast"):
            raise NotImplementedError(
                f"CuTeDSL GEMM epilogue does not support {value.op}"
            )
        lowered_args = tuple(self.lower(arg) for arg in value.args)
        lowered_kwargs = {key: self.lower(arg) for key, arg in value.kwargs}
        op = getattr(GemmEpilogueCuteDSLOpOverrides, value.op, None)
        if op is None:
            raise NotImplementedError(
                f"CuTeDSL GEMM epilogue op is not implemented: {value.op}"
            )
        return op(*lowered_args, **lowered_kwargs)

    def render(
        self, buffers: Sequence[ComputedBuffer], fn_name: str
    ) -> tuple[list[str], list[str], dict[str, str], str]:
        # [FUNCTION DOC] Analyze buffers, lower every store whose buffer is not removed (plus the
        # accumulator itself if not removed), and write one result assignment per output; the last
        # is named D, others output0, output1, and so on. Returns (read names, output buffer
        # names, rename map from function-local names to buffer names, function source text with
        # parameters accum plus reads). Raises NotImplementedError if there are no outputs.
        analysis = GemmEpilogueIRAnalysis.from_buffers(buffers)
        outputs: list[tuple[str, Any]] = []
        if self.accumulator not in self.removed_buffers:
            outputs.append((self.accumulator, self.accumulator_value))
        for name, store in analysis.stores.items():
            if name not in self.removed_buffers:
                outputs.append((name, self.lower(store.value)))
        if not outputs:
            raise NotImplementedError("CuTeDSL GEMM epilogue has no outputs")

        renames = {name: name for name in self.reads}
        result_names: list[str] = []
        for index, (buffer_name, value) in enumerate(outputs):
            result_name = "D" if index == len(outputs) - 1 else f"output{index}"
            self.kernel.body.writeline(f"{result_name} = {value}")
            renames[result_name] = buffer_name
            result_names.append(result_name)

        params = ", ".join(("accum", *self.reads))
        body = "\n".join(f"    {line}" for line in self.kernel.body.lines)
        source = (
            f"def {fn_name}({params}):\n{body}\n    return {', '.join(result_names)}"
        )
        return list(self.reads), [name for name, _ in outputs], renames, source

    @classmethod
    def from_buffers(
        cls,
        accumulator: str,
        buffers: Sequence[ComputedBuffer],
        removed_buffers: OrderedSet[str],
        fn_name: str,
    ) -> tuple[list[str], list[str], dict[str, str], str]:
        # [FUNCTION DOC] Convenience entry point: build a codegen, install its kernel and the GEMM
        # epilogue op overrides as the active virtualized handlers, and return render's result.
        codegen = cls(accumulator, removed_buffers)
        with (
            V.set_kernel_handler(codegen.kernel),
            V.set_ops_handler(GemmEpilogueCuteDSLOpOverrides()),
        ):
            return codegen.render(buffers, fn_name)
