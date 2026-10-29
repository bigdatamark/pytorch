# [PACKAGE DOC] Native override of foreach matmul. Importing calls register_to_dispatch from impl to install the override.
from .impl import register_to_dispatch


register_to_dispatch()
