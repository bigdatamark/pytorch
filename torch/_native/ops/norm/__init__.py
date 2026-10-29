# [PACKAGE DOC] Native norm overrides. Importing calls register_rmsnorm_overrides from rmsnorm_impl to install them.
from .rmsnorm_impl import register_rmsnorm_overrides


register_rmsnorm_overrides()
