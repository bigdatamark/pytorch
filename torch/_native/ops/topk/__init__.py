# [PACKAGE DOC] Package for the native topk operator override. Importing it registers the CuTeDSL topk
# implementation with the dispatcher (import-time side effect) via the aliased register function.
from .cutedsl_impl import register_to_dispatch as _register_cutedsl


_register_cutedsl()
