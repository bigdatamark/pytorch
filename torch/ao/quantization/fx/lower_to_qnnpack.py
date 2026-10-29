# [MODULE DOC] Thin entry point lowering a reference quantized model to the qnnpack backend
# by delegating to the shared native-backend lowering. Exports lower_to_qnnpack.
from torch.ao.quantization.qconfig import QConfigAny
from torch.fx import GraphModule

from ._lower_to_native_backend import _lower_to_native_backend


__all__ = ["lower_to_qnnpack"]


def lower_to_qnnpack(
    model: GraphModule,
    qconfig_map: dict[str, QConfigAny],
    node_name_to_scope: dict[str, tuple[str, type]],
) -> GraphModule:
    # [FUNCTION DOC] Lower a reference quantized GraphModule to qnnpack by calling the shared
    # native-backend lowering with the model, qconfig_map and node_name_to_scope (default
    # keep_original_weights). Returns the lowered GraphModule.
    """Lower a quantized reference model (with reference quantized operator patterns)
    to qnnpack
    """
    return _lower_to_native_backend(model, qconfig_map, node_name_to_scope)
