# [PACKAGE DOC] Private package of generic communication hooks; exposes the default hooks module as "default" and LOW_PRECISION_HOOKS (fp16 and bf16 compress hooks).
from . import default_hooks as default


LOW_PRECISION_HOOKS = [
    default.fp16_compress_hook,
    default.bf16_compress_hook,
]
