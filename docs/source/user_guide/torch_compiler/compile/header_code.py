# [MODULE DOC] Setup snippet run at the top of the compile user guide notebooks. At
# import time it replaces torch.compile with a version defaulting to the eager backend
# (shortens notebook execution), clears the torch logs format environment variable,
# and reinitializes the torch logging formatter and log settings so output lacks the
# usual prefix.
import functools
import os

import torch


# to lower notebook execution time while hiding backend="eager"
torch.compile = functools.partial(torch.compile, backend="eager")

# to clear torch logs format
os.environ["TORCH_LOGS_FORMAT"] = ""
torch._logging._internal.DEFAULT_FORMATTER = (
    torch._logging._internal._default_formatter()
)
torch._logging._internal._init_logs()
