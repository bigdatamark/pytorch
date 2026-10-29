# [MODULE DOC] Thin entry point lowering a reference quantized model to the fbgemm backend
# by delegating to the shared native-backend lowering. Exports lower_to_fbgemm.
from torch.ao.quantization.qconfig import QConfigAny
from torch.fx import GraphModule

from ._lower_to_native_backend import _lower_to_native_backend


__all__ = ["lower_to_fbgemm"]


def lower_to_fbgemm(
    model: GraphModule,
    qconfig_map: dict[str, QConfigAny],
    node_name_to_scope: dict[str, tuple[str, type]],
    keep_original_weights: bool = False,
) -> GraphModule:
    # [FUNCTION DOC] Lower a reference quantized GraphModule to fbgemm by calling the shared
    # native-backend lowering with the model, qconfig_map, node_name_to_scope and the
    # keep_original_weights flag. Returns the lowered GraphModule.
    """Lower a quantized reference model (with reference quantized operator patterns)
    to fbgemm
    """
    return _lower_to_native_backend(
        model, qconfig_map, node_name_to_scope, keep_original_weights
    )
