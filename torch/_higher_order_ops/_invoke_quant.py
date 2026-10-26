# mypy: allow-untyped-defs
# need to fix prim_hop_base type annotations first

import dataclasses

import torch
from torch._higher_order_ops.base_hop import BaseHOP, FunctionWithNoFreeVars


class InvokeQuantTracer(BaseHOP):
    # [CLASS DOC] BaseHOP subclass named "invoke_quant_packed". It wraps the subgraph so
    # that it must have no free variables, then forwards to the base HOP call. Stateless.
    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the HOP under the name invoke_quant_packed.
        super().__init__("invoke_quant_packed")

    def __call__(self, subgraph, *operands, scheme=None, quant_options=None):
        # [FUNCTION DOC] Wraps subgraph in FunctionWithNoFreeVars and calls the base HOP
        # with the operands plus the scheme and quant_options keyword arguments.
        # Returns whatever the base HOP returns (the subgraph outputs).
        subgraph = FunctionWithNoFreeVars(subgraph)
        return super().__call__(
            subgraph, *operands, scheme=scheme, quant_options=quant_options
        )


invoke_quant_packed = InvokeQuantTracer()


class InvokeQuantUnpacked(BaseHOP):
    # [CLASS DOC] BaseHOP subclass named "invoke_quant"; no behavior beyond the base class.
    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the HOP under the name invoke_quant.
        super().__init__("invoke_quant")


invoke_quant = InvokeQuantUnpacked()


@dataclasses.dataclass(frozen=True, repr=True)
class InvokeQuant:
    """
    Invoke a quantization function that will be preserved as a single operator. Preservation
    as a single operator aids in pattern matching and custom lowerings.

    The operation appears as:
        torch.ops.higher_order.invoke_quant(subgraph, *args, scheme=scheme)

    Args:
        codegen_low_precision: Use observed subgraph dtypes for codegen instead of
            upcasting to fp32. Can improve performance for prologue fusion but
            requires careful testing of numerics.
    """
    # [CLASS DOC] Frozen dataclass acting as a callable wrapper around invoke_quant_packed.
    # Sole field codegen_low_precision (default true) is passed along as quant_options
    # (the instance itself) so lowering can keep low-precision dtypes.

    codegen_low_precision: bool = True

    def __call__(
        self,
        *args,
        scheme: str | None = None,
        **kwargs,
    ):
        # [FUNCTION DOC] First positional arg is the quantization function, the rest are
        # its operands. When not compiling, just calls the function eagerly with the rest
        # and returns its result. When compiling, adds scheme to the keyword arguments if
        # given and calls invoke_quant_packed with this instance as quant_options.
        if not torch.compiler.is_compiling():
            return args[0](*args[1:], **kwargs)

        if scheme is not None:
            kwargs["scheme"] = scheme

        return invoke_quant_packed(*args, **kwargs, quant_options=self)  # type: ignore[call-arg]
