# [PACKAGE DOC] Package torch.distributed.nn: differentiable distributed collective
# functions (star-imported from .functional) plus RemoteModule, which is
# imported only when the RPC framework is available.
import torch

from .functional import *  # noqa: F403


if torch.distributed.rpc.is_available():
    from .api.remote_module import RemoteModule
