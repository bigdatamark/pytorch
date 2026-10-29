# [PACKAGE DOC] Analysis utilities for torch.package. Re-exports
# find_first_use_of_broken_modules (maps broken modules from a PackagingError to the
# dependency path where each was first seen) and trace_dependencies (runs a callable under a
# profiler to discover which modules it uses).
from .find_first_use_of_broken_modules import find_first_use_of_broken_modules
from .trace_dependencies import trace_dependencies
