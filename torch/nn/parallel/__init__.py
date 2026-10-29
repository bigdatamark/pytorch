# [PACKAGE DOC] Public API of torch.nn.parallel: re-exports replicate, scatter, gather,
# parallel_apply, data_parallel, DataParallel and DistributedDataParallel, and defines the
# deprecated DistributedDataParallelCPU alias class.
from typing_extensions import deprecated

from torch.nn.parallel.data_parallel import data_parallel, DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter


__all__ = [
    "replicate",
    "scatter",
    "parallel_apply",
    "gather",
    "data_parallel",
    "DataParallel",
    "DistributedDataParallel",
]


@deprecated(
    "`torch.nn.parallel.DistributedDataParallelCPU` is deprecated, "
    "please use `torch.nn.parallel.DistributedDataParallel` instead.",
    category=FutureWarning,
)
class DistributedDataParallelCPU(DistributedDataParallel):
    # [CLASS DOC] Deprecated subclass of DistributedDataParallel with no added behavior; the
    # deprecated decorator emits a FutureWarning when it is used, pointing users at
    # DistributedDataParallel.
    pass
