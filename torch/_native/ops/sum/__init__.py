# [PACKAGE DOC] Package for the native sum operator override. Importing it registers the CuTeDSL sum
# implementation with the dispatcher (import-time side effect) by calling register_to_dispatch.
from .cutedsl_impl import register_to_dispatch


register_to_dispatch()
