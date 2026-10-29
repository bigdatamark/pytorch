# [PACKAGE DOC] Native override of bmm for outer-product-shaped inputs (Triton). Importing calls register_to_dispatch from triton_impl to install the override.
from .triton_impl import register_to_dispatch


register_to_dispatch()
