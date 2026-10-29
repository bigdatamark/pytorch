# mypy: allow-untyped-defs
# [MODULE DOC] Deprecated backward-compatibility shim providing a no-op worker_init_fn.
from typing_extensions import deprecated as _deprecated


@_deprecated(
    "Usage of `backward_compatibility.worker_init_fn` is deprecated "
    "as `DataLoader` automatically applies sharding in every worker",
    category=FutureWarning,
)
def worker_init_fn(worker_id) -> None:
    # [FUNCTION DOC] Deprecated no-op worker init function; accepts a worker id and does nothing.
    # Calling it emits a FutureWarning because DataLoader now shards automatically in each worker.
    pass
