# [MODULE DOC] Dynamo VariableTracker for the C++ scaled-dot-product-attention params struct
# (SDPAParams): SDPAParamsVariable is a read-only container holding an FX proxy and the VTs of
# its seven fields (PARAM_NAMES). Supports attribute access via graph getattr nodes and
# reconstruction by calling the torch._C params constructor. No import-time side effects.
from inspect import getattr_static
from typing import Any, TYPE_CHECKING, TypeGuard

from torch._guards import Source
from torch.backends.cuda import SDPAParams
from torch.fx.proxy import Proxy

from ..bytecode_transformation import create_call_function
from ..exc import unimplemented
from ..source import AttrSource
from .base import VariableTracker


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase

PARAM_NAMES = [
    "query",
    "key",
    "value",
    "attn_mask",
    "dropout",
    "is_causal",
    "enable_gqa",
]


class SDPAParamsVariable(VariableTracker):
    # [CLASS DOC] Read-only VT for an SDPAParams struct. State: proxy (FX proxy of the struct)
    # and param_vars (list of VTs for query, key, value, attn_mask, dropout, is_causal,
    # enable_gqa). Attribute reads become graph getattr nodes; python_type is SDPAParams.
    """Represents the c++ params struct for scaled dot product attention.
    This is a read-only container."""

    @staticmethod
    def create(
        tx: "InstructionTranslatorBase", value: Any, source: Source
    ) -> VariableTracker:
        # [FUNCTION DOC] Factory from a real SDPAParams value: build a VT (with attribute source)
        # for each field in PARAM_NAMES and call the SDPAParams constructor as an in-graph
        # function, returning the resulting VT (an SDPAParamsVariable).
        from .torch import TorchInGraphFunctionVariable

        params = [
            VariableTracker.build(tx, getattr(value, p), AttrSource(source, p))
            for p in PARAM_NAMES
        ]
        return TorchInGraphFunctionVariable(SDPAParams).call_function(tx, params, {})

    def __init__(
        self, proxy: Proxy, param_vars: list[VariableTracker], **kwargs: Any
    ) -> None:
        # [FUNCTION DOC] Store the struct's FX proxy and the VTs of its fields, then initialize
        # the base VT with remaining keyword options.
        self.proxy = proxy
        self.param_vars = param_vars
        super().__init__(**kwargs)

    def python_type(self) -> type:
        # [FUNCTION DOC] The python type represented: SDPAParams.
        return SDPAParams

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emit bytecode that rebuilds the struct by calling the torch._C params
        # constructor with the field VTs in order. Raises AssertionError if the VT has a source
        # or param_vars is None.
        if self.source is not None:
            raise AssertionError(
                "SDPAParamsVariable should not have a source during reconstruct"
            )
        if self.param_vars is None:
            raise AssertionError("SDPAParamsVariable.param_vars must not be None")
        codegen.add_push_null(
            lambda: codegen.load_import_from("torch._C", "_SDPAParams")
        )
        codegen.foreach(self.param_vars)
        codegen.extend_output(create_call_function(len(self.param_vars), False))

    def as_proxy(self) -> Proxy:
        # [FUNCTION DOC] Return the stored FX proxy.
        return self.proxy

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Attribute read: the name must exist statically on the C params class
        # (else graph break). Emits a getattr proxy on the struct proxy and wraps it, adding an
        # attribute source when this VT has a source. Returns the wrapped VT.
        import torch._C

        from .builder import wrap_fx_proxy
        from .misc import GetAttrVariable

        try:
            getattr_static(torch._C._SDPAParams, name)
        except AttributeError:
            import torch._dynamo.graph_break_hints as graph_break_hints

            unimplemented(
                gb_type="unsupported torch._C._SDPAParams attribute",
                context=f"name: {name}",
                explanation=f"Unable to fetch attribute {name} from torch._C._SDPAParams.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        proxy = GetAttrVariable.create_getattr_proxy(self.as_proxy(), name)
        if self.source is not None:
            return wrap_fx_proxy(
                tx=tx, proxy=proxy, source=AttrSource(self.source, name)
            )
        else:
            return wrap_fx_proxy(tx=tx, proxy=proxy)

    @staticmethod
    def is_sdpa_params(value: Any) -> TypeGuard["SDPAParams"]:
        # [FUNCTION DOC] True if value is the SDPAParams class itself (identity check, not an
        # instance check).
        return value is SDPAParams
