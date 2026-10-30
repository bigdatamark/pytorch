# [MODULE DOC] Static typing check (type-checker input, not a runtime test) that a CUDA stream
# object returned by the current-stream query is accepted by the tensor method that records
# stream usage. Importing it only imports torch; no code runs at import time.
import torch


def foo(x: torch.Tensor) -> None:
    # [FUNCTION DOC] Typing sample: takes a tensor, obtains the current CUDA stream through the
    # cuda module, and passes it to the tensor record_stream method. Returns nothing. At runtime
    # this would need a CUDA device; it is only meant to be type checked and is never called
    # here.
    stream = torch.cuda.current_stream()
    x.record_stream(stream)
