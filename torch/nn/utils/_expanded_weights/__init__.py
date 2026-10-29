# [PACKAGE DOC] Expanded weights: support for computing per-sample gradients through ordinary
# modules. Importing the submodules registers per-sample-gradient autograd functions (conv,
# embedding, group norm, instance norm, layer norm, linear) for their torch functions. Only
# ExpandedWeight is listed in __all__; the other classes are imported for their registration
# side effect.
from .conv_expanded_weights import ConvPerSampleGrad
from .embedding_expanded_weights import EmbeddingPerSampleGrad
from .expanded_weights_impl import ExpandedWeight
from .group_norm_expanded_weights import GroupNormPerSampleGrad
from .instance_norm_expanded_weights import InstanceNormPerSampleGrad
from .layer_norm_expanded_weights import LayerNormPerSampleGrad
from .linear_expanded_weights import LinearPerSampleGrad


__all__ = ["ExpandedWeight"]
