#!/usr/bin/env python3
# [MODULE DOC] Command-line pre-flight check for ROCm/RCCL on distributed CI runners. Spawns
# up to two ranks (one per visible GPU) that each initialize an NCCL-backend process group on
# localhost and run a small all_reduce, so a broken GPU topology fails fast with a CI error
# annotation rather than hanging later. Exit status is zero on success, one on failure. Main
# entry: main. Import-time side effect: imports torch and defines a 60 second group timeout.
"""Fast ROCm/RCCL pre-flight health check for distributed CI shards.

Some gfx950 runner pods come up with a container that cannot read part of the
KFD/HSA topology -- RCCL fails every collective init with
"ncclUnhandledCudaError: Call to CUDA function failed / Could not read node #N"
(N is a fixed topology-node index for that pod). When a distributed shard lands
on such a pod, the first collective crashes and a later test hangs the whole
shard until the 270-minute job timeout. Host-side ``rocminfo`` still enumerates
the GPUs on these pods, so the existing GPU-count gate does not catch it.

This runs a tiny multi-rank all_reduce inside the test container before the
suite. If it fails (or is killed by the outer ``timeout``), the job fails in
seconds with a clear message instead of hanging, and the bad pod can be drained.
"""

import datetime
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


_PG_TIMEOUT = datetime.timedelta(seconds=60)


def _worker(rank: int, world_size: int) -> None:
    # [FUNCTION DOC] HELPER: Per-rank body run by the process spawner. Defaults the master
    # address and port environment variables, selects the GPU with the rank index, joins the
    # process group (60 second timeout), all-reduces a small ones tensor, synchronizes, and
    # always destroys the process group. Exceptions propagate to the spawner.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, timeout=_PG_TIMEOUT
    )
    try:
        t = torch.ones(16, device=f"cuda:{rank}")
        dist.all_reduce(t)
        torch.cuda.synchronize()
    finally:
        dist.destroy_process_group()


def main() -> int:
    # [FUNCTION DOC] Entry point. Counts visible GPUs (prints an error annotation and returns
    # one if none), spawns the worker on min(2, count) ranks and waits. Any exception is
    # reported as a CI error annotation with the underlying message and returns one; on
    # success prints a pass message and returns zero.
    n = torch.cuda.device_count()
    if n < 1:
        print("::error::ROCm pre-flight: no GPUs visible to the container")
        return 1
    world_size = min(2, n)
    try:
        mp.spawn(_worker, args=(world_size,), nprocs=world_size, join=True)
    except Exception as e:
        print(
            "::error::ROCm/RCCL pre-flight failed: the container cannot "
            f"initialize a {world_size}-rank collective on this runner -- "
            "likely a broken KFD/HSA topology on this pod (look for "
            '"Could not read node #N"). Failing fast instead of hanging; '
            f"re-run to land on a healthy runner. Underlying error: {e}"
        )
        return 1
    print(f"ROCm/RCCL pre-flight passed ({world_size}-rank all_reduce)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
