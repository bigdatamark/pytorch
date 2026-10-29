# [MODULE DOC] Defines the module-level dispatch decorator: multipledispatch's
# dispatch pre-bound to a private shared namespace dict, so every registration
# in the unification package lands in one registry. Exports dispatch and namespace.
from functools import partial

from .multipledispatch import dispatch as _dispatch  # type: ignore[import]


namespace = {}  # type: ignore[var-annotated]

dispatch = partial(_dispatch, namespace=namespace)
