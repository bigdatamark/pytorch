# Copyright (c) Meta Platforms, Inc. and affiliates
# [PACKAGE DOC] Public Tensor Parallel API package. Re-exports parallelize_module (applies a
# parallelization plan to an nn.Module), loss_parallel (context manager for sharded-class-dim
# loss), and the ParallelStyle classes: ColwiseParallel, RowwiseParallel, SequenceParallel,
# PrepareModuleInput, PrepareModuleOutput, PrepareModuleInputOutput. Defines __all__.
from torch.distributed.tensor.parallel.api import parallelize_module
from torch.distributed.tensor.parallel.loss import loss_parallel
from torch.distributed.tensor.parallel.style import (
    ColwiseParallel,
    ParallelStyle,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
)


__all__ = [
    "ColwiseParallel",
    "ParallelStyle",
    "PrepareModuleInput",
    "PrepareModuleInputOutput",
    "PrepareModuleOutput",
    "RowwiseParallel",
    "SequenceParallel",
    "parallelize_module",
    "loss_parallel",
]
