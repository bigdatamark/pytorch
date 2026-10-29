# [PACKAGE DOC] Native CuTeDSL override of scatter_add. Importing calls register_to_dispatch from cutedsl_impl to install the override.
from .cutedsl_impl import register_to_dispatch


register_to_dispatch()
