# [PACKAGE DOC]
# FX graph passes used by the ONNX export path. Re-exports InsertTypePromotion, a pass that
# inserts explicit type promotion casts into an FX graph.
from .type_promotion import InsertTypePromotion


__all__ = [
    "InsertTypePromotion",
]
