# [MODULE DOC] Shared registry for gpt_fast benchmarks. Holds a module level mapping from
# experiment name to experiment function, a dataclass describing one benchmark result, and a
# decorator that registers functions into the mapping.
import dataclasses
from collections.abc import Callable


all_experiments: dict[str, Callable] = {}


@dataclasses.dataclass
class Experiment:
    # [CLASS DOC] Plain dataclass record of one benchmark measurement: experiment name, the
    # metric name, the target (expected) value, the actual measured value, dtype and device
    # strings, an architecture string (GPU name for CUDA or CPU arch for CPU) and a flag marking
    # whether the experiment is a whole model run (default False). No methods or validation.
    name: str
    metric: str
    target: float
    actual: float
    dtype: str
    device: str
    arch: str  # GPU name for CUDA or CPU arch for CPU
    is_model: bool = False


def register_experiment(name: str | None = None):
    # [FUNCTION DOC] Return a decorator that registers a benchmark function in the global
    # mapping of experiments. The key is the given name, or the function's own name if name is
    # None or empty. A later registration under the same key silently replaces the earlier one.
    def decorator(func):
        # [FUNCTION DOC] HELPER: Store the function in the global experiment mapping under the
        # chosen key and return it unchanged.
        key = name or func.__name__
        all_experiments[key] = func
        return func

    return decorator
