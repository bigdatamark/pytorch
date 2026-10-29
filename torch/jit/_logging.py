# [MODULE DOC]
# Python aliases for the TorchScript runtime logging API: stat-value and
# timing ops from torch.ops.prim and logger classes/functions from torch._C.
# Import-time side effect: looks up these ops and C++ bindings by attribute.
import torch


add_stat_value = torch.ops.prim.AddStatValue

set_logger = torch._C._logging_set_logger
LockingLogger = torch._C.LockingLogger
AggregationType = torch._C.AggregationType
NoopLogger = torch._C.NoopLogger

time_point = torch.ops.prim.TimePoint
