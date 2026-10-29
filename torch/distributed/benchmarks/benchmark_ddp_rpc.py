# mypy: allow-untyped-defs
# [MODULE DOC]
# Benchmark script combining DistributedDataParallel (dense part) and RPC with remote
# EmbeddingBag tables on parameter servers (sparse part). Rank layout: trainers first,
# then parameter servers, then one master (last rank) that drives training and prints
# per-trainer and overall percentile timings. Run as a script: prints GPU topology via
# nvidia-smi, parses command-line options, sets MASTER_ADDR/MASTER_PORT environment
# variables, overrides the module-level config globals, and spawns one process per rank.
# Module-level config: NUM_TRAINERS, NUM_PS, NUM_EMBEDDINGS, EMBEDDING_DIM, WARMUP_CYCLES.

# pyre-unsafe
import argparse
import io
import os
import random
import shlex
import subprocess
import time

import numpy as np

import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef, TensorPipeRpcBackendOptions
from torch.distributed.rpc.backend_registry import BackendType
from torch.nn.parallel import DistributedDataParallel as DDP


# Config
NUM_TRAINERS = 8
NUM_PS = 8

NUM_EMBEDDINGS = 300
EMBEDDING_DIM = 64

WARMUP_CYCLES = 5


class HybridModel(torch.nn.Module):
    # [CLASS DOC] Model with a sparse part (remote embedding bag tables referenced by the
    # list emb_rref_list, one per parameter server) and a dense part (a stack of linear
    # layers wrapped in DistributedDataParallel on the given device). State: emb_rref_list,
    # ddp, device. Forward queries every embedding table remotely and feeds the result
    # through the DDP-wrapped dense stack.
    r"""
    The model consists of a sparse part and a dense part.

    The dense part is an nn.Linear module that is replicated across all trainers using
    DistributedDataParallel. The sparse part has nn.EmbeddingBags stored on multiple
    parameter servers.

    The model holds a Remote Reference to the embedding tables on the parameter
    servers.
    """

    def __init__(self, emb_rref_list, device):
        # [FUNCTION DOC] Stores the remote embedding references, builds a six-stage dense
        # sequential model (input width 512, output width 8), moves it to the device and wraps
        # it in DDP (requires an initialized process group among trainers). Saves the device.
        super().__init__()
        self.emb_rref_list = emb_rref_list
        fc1 = torch.nn.Linear(512, 256)
        fc2 = torch.nn.Linear(256, 128)
        relu = torch.nn.ReLU()
        fc3 = torch.nn.Linear(128, 64)
        fc4 = torch.nn.Linear(64, 32)
        fc5 = torch.nn.Linear(32, 8)
        sec = nn.Sequential(fc1, fc2, relu, fc3, fc4, fc5)
        self.ddp = DDP(sec.to(device), device_ids=[device])
        self.device = device

    def forward(self, indices, offsets):
        # [FUNCTION DOC] Performs a blocking remote forward on each embedding table with the
        # indices and offsets, concatenates the outputs along the feature dimension, reshapes
        # to rows of width 512 (batch scaled by the ratio of total embedding width to 512) and
        # returns the DDP dense model output. Raises AssertionError if total embedding width
        # (NUM_PS times EMBEDDING_DIM) is smaller than 512.
        # KNOWN ISSUE: the concatenation sits inside the loop over tables, so it is recomputed
        # each iteration and is undefined if the list of references is empty. Also the integer
        # division truncates if the width is not a multiple of 512.
        emb_lookups = []

        for emb_rref in self.emb_rref_list:
            emb_lookups.append(
                emb_rref.rpc_sync().forward(
                    indices, offsets
                )  # embedding_sum(input, offsets)
            )
            emb_lookups_cat = torch.cat(emb_lookups, dim=1)

        # Make sure combined PS dimension is always bigger or equal than the FC input
        if NUM_PS * EMBEDDING_DIM < 512:
            raise AssertionError
        dim_normalizer = int(NUM_PS * EMBEDDING_DIM / 512)
        emb_lookups_reshaped = emb_lookups_cat.reshape(  # type: ignore[possibly-undefined]
            # pyrefly: ignore [unbound-name]
            [emb_lookups_cat.shape[0] * dim_normalizer, 512]
        )

        return self.ddp(emb_lookups_reshaped)


def _retrieve_embedding_parameters(emb_rref):
    # [FUNCTION DOC] HELPER: runs on a parameter server; returns a list of remote references
    # to each parameter of the locally held embedding module behind the given reference.
    return [RRef(p) for p in emb_rref.local_value().parameters()]


def _print_header():
    # [FUNCTION DOC] HELPER: prints the column header line (four repeated pairs of
    # seconds-per-epoch and epochs-per-second labels, one per percentile) to stdout.
    # KNOWN ISSUE: the loop variable (the percentile values) is unused, so labels carry no percentile.
    _print_cont("\n")
    _print_cont(" " * 10)
    for _ in [50, 75, 90, 95]:
        _print_cont(f"{'sec/epoch':14s}{'epoch/sec':10s}")
    _print_cont("\n")


def _print_benchmark(prefix, nelem, measurements):
    # [FUNCTION DOC] HELPER: prints, for the 50th, 75th, 90th and 95th percentiles of the
    # time measurements, the duration and the throughput (nelem divided by duration), labelled
    # with prefix. Writes to stdout.
    # KNOWN ISSUE: throughput is a float but is formatted with an integer format code, which
    # raises ValueError at runtime.
    measurements = sorted(measurements)
    _print_cont(f"{prefix:8s}:")
    for p in [50, 75, 90, 95]:
        v = np.percentile(measurements, p)
        _print_cont(f"  p{p:02d}:  {v:1.3f}s  {nelem / v:6d}/s")
    _print_cont("\n")


def _print_cont(msg):
    # [FUNCTION DOC] HELPER: prints the message to stdout with no trailing newline and flushes.
    print(msg, end="", flush=True)


def _run_printable(cmd):
    # [FUNCTION DOC] HELPER: runs the shell-style command string as a subprocess capturing
    # output, raises AssertionError on a non-zero exit code, and returns a one-element list
    # holding the decoded stdout text (round-tripped through torch save and load in memory).
    proc = subprocess.run(shlex.split(cmd), capture_output=True, check=False)  # type: ignore[call-overload]
    if proc.returncode != 0:
        raise AssertionError

    buffer = io.BytesIO()
    torch.save(proc.stdout.decode("utf-8"), buffer)
    input_tensor = torch.ByteTensor(list(buffer.getvalue()))

    output = []
    buffer = io.BytesIO(np.asarray(input_tensor).tobytes())
    output.append(torch.load(buffer))
    return output


def _run_trainer(emb_rref_list, rank):
    # [FUNCTION DOC] Runs on a trainer via RPC. Builds the HybridModel on the GPU with index
    # rank, gathers remote references to all embedding parameters (from each server) plus local
    # model parameters, and creates a distributed SGD optimizer (learning rate 0.05) and
    # cross-entropy loss. Then runs 100 plus WARMUP_CYCLES epochs of 10 random batches each;
    # per batch: opens a distributed autograd context, forward, loss, distributed backward,
    # optimizer step. Records wall time per epoch, drops the warm-up epochs.
    # Returns a tuple of rank, list of epoch times, and the batch size count of the last epoch.
    r"""
    Each trainer runs a forward pass which involves an embedding lookup on the 8 parameter servers,
    and running nn.Linear locally.

    During the backward pass, DDP is responsible for aggregating the gradients for the dense part
    (nn.Linear) and distributed autograd ensures gradients updates are
    propagated to the parameter servers.
    """
    # Setup the model.
    model = HybridModel(emb_rref_list, rank)

    # Retrieve all model parameters as rrefs for DistributedOptimizer.

    # Retrieve parameters from all embedding tables for the current trainer.
    model_parameter_rrefs = []
    for ind, emb_rref in enumerate(emb_rref_list):
        ps_name = f"ps{ind}"
        model_parameter_rrefs.extend(
            rpc.rpc_sync(ps_name, _retrieve_embedding_parameters, args=(emb_rref,))
        )

    # model.parameters() only includes local parameters.
    model_parameter_rrefs.extend(RRef(param) for param in model.parameters())

    # Setup distributed optimizer
    opt = DistributedOptimizer(optim.SGD, model_parameter_rrefs, lr=0.05)

    criterion = torch.nn.CrossEntropyLoss()

    def get_next_batch(rank):
        # [FUNCTION DOC] HELPER: generator yielding 10 random batches. Each has random indices
        # (20 to 50 entries drawn from the embedding table size), offsets splitting them into
        # bags of random length 1 to 10, and random class targets (0-7) moved to GPU index rank.
        for _ in range(10):
            num_indices = random.randint(20, 50)
            indices = torch.LongTensor(num_indices).random_(0, NUM_EMBEDDINGS)

            # Generate offsets.
            offsets = []
            start = 0
            batch_size = 0

            while start < num_indices:
                offsets.append(start)
                start += random.randint(1, 10)
                batch_size += 1

            offsets_tensor = torch.LongTensor(offsets)
            target = torch.LongTensor(batch_size).random_(8).cuda(rank)

            yield indices, offsets_tensor, target

    measurements = []
    # Include warm-up cycles during training
    for _ in range(100 + WARMUP_CYCLES):
        start = time.time()
        batch_size = 0

        # create distributed autograd context
        for indices, offsets, target in get_next_batch(rank):
            batch_size += len(target)

            with dist_autograd.context() as context_id:
                output = model(indices, offsets)
                loss = criterion(output, target)

                # Run distributed backward pass
                dist_autograd.backward(context_id, [loss])

                # Run distributed optimizer. Gradients propagated all the way to the parameter servers
                opt.step(context_id)

                # Not necessary to zero grads as each iteration creates a different
                # distributed autograd context which hosts different grads

        measurements.append(time.time() - start)
        # print("Training done for epoch {}".format(epoch))

    # Throw away warm-up measurements
    measurements = measurements[WARMUP_CYCLES:]
    return rank, measurements, batch_size  # type: ignore[possibly-undefined]


def run_worker(rank, world_size):
    # [FUNCTION DOC] Per-process entry point started by multiprocessing spawn. Role depends on
    # rank: the last rank is the master (creates one remote EmbeddingBag per parameter server,
    # launches training on every trainer asynchronously, waits for results and prints
    # per-trainer and aggregate benchmarks); low ranks are trainers (initialize a gloo process
    # group for DDP plus RPC, then wait for work); middle ranks are parameter servers
    # (initialize RPC only). All roles end with RPC shutdown, which blocks until all finish.
    # KNOWN ISSUE: the trainer branch compares against NUM_PS rather than NUM_TRAINERS, and the
    # master does not pass the backend options; correct only when the counts are equal.
    r"""
    Initialize RPC, calls the function, and shuts down RPC.
    """
    # Using different port numbers in TCP init_method for init_rpc and
    # init_process_group to avoid port conflicts.
    rpc_backend_options = TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method = "tcp://localhost:29500"

    # Rank 16. Master
    if rank == (NUM_TRAINERS + NUM_PS):
        rpc.init_rpc(
            "master",
            rank=rank,
            backend=BackendType.TENSORPIPE,  # type: ignore[attr-defined]
            world_size=world_size,
        )

        # Build the Embedding tables on the Parameter Servers.
        emb_rref_list = []
        index = 0
        while index < NUM_PS:
            ps_name = f"ps{index}"
            emb_rref = rpc.remote(
                ps_name,
                torch.nn.EmbeddingBag,
                args=(NUM_EMBEDDINGS, EMBEDDING_DIM),
                kwargs={"mode": "sum"},
            )
            emb_rref_list.append(emb_rref)
            index += 1

        # Run training loop on the trainers.
        futs = []
        for trainer_rank in range(NUM_TRAINERS):
            trainer_name = f"trainer{trainer_rank}"
            fut = rpc.rpc_async(
                trainer_name, _run_trainer, args=(emb_rref_list, trainer_rank)
            )
            futs.append(fut)

        _print_header()

        measurements_all_trainers = []
        batch_size_all_trainers = 0
        # Wait for all training to finish.
        for fut in futs:
            rank, measurements, batch_size = fut.wait()
            _print_benchmark(f"Trainer{rank}", batch_size, measurements)
            batch_size_all_trainers += batch_size
            measurements_all_trainers.append(measurements)

        _print_benchmark("All", batch_size_all_trainers, measurements_all_trainers)

    # Rank 0-7. Trainers
    elif rank >= 0 and rank < NUM_PS:
        # Initialize process group for Distributed DataParallel on trainers.
        dist.init_process_group(
            backend=dist.Backend.GLOO,
            rank=rank,
            world_size=NUM_TRAINERS,
            init_method="tcp://localhost:29501",
        )

        # Initialize RPC. Trainer just waits for RPCs from master.
        trainer_name = f"trainer{rank}"
        rpc.init_rpc(
            trainer_name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

    # Rank 8-15. Parameter Servers
    elif rank >= NUM_TRAINERS and rank < NUM_TRAINERS + NUM_PS:
        ps_name = f"ps{rank - NUM_TRAINERS}"
        rpc.init_rpc(
            ps_name,
            rank=rank,
            world_size=world_size,
            backend=BackendType.TENSORPIPE,  # type: ignore[attr-defined]
            rpc_backend_options=rpc_backend_options,
        )
        # parameter server do nothing

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == "__main__":
    """Initializing the distributed environment."""

    output = _run_printable("nvidia-smi topo -m")
    print("-------------------------------------------")
    print("                  Info                     ")
    print("-------------------------------------------")
    print()
    print(f"* PyTorch version: {torch.__version__}")
    print(f"* CUDA version: {torch.version.cuda}")
    print()
    print("------------ nvidia-smi topo -m -----------")
    print()
    print(output[0])
    print("-------------------------------------------")
    print("PyTorch Distributed Benchmark (DDP and RPC)")
    print("-------------------------------------------")

    # Cmd arguments to enable automated runs (e.g. Chronos, SSH, etc).
    parser = argparse.ArgumentParser(description="PyTorch DDP and RPC Benchmark")
    parser.add_argument(
        "--master-addr", type=str, default="localhost", help="Address of master node."
    )
    parser.add_argument("--master-port", type=str, default="29500", help="Master port.")

    parser.add_argument(
        "--number-trainers",
        type=int,
        default=NUM_TRAINERS,
        help="Number of Trainer Nodes.",
    )
    parser.add_argument(
        "--number-ps", type=int, default=NUM_PS, help="Number of Parameter Servers."
    )
    parser.add_argument(
        "--number-embeddings",
        type=int,
        default=NUM_EMBEDDINGS,
        help="Number of test embeddings to be generated.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=EMBEDDING_DIM,
        help="Number of embedding dimensions.",
    )
    parser.add_argument(
        "--warmup-cycles",
        type=int,
        default=WARMUP_CYCLES,
        help="Number of cycles to warm-up each process before running the benchmark.",
    )

    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    NUM_TRAINERS = args.number_trainers
    NUM_PS = args.number_ps

    NUM_EMBEDDINGS = args.number_embeddings
    EMBEDDING_DIM = args.embedding_dim

    WARMUP_CYCLES = args.warmup_cycles

    # Defaults:
    #  8 trainers (rank 0-7),
    #  8 parameter servers (rank 8-15),
    #  1 master (rank 16).
    world_size = NUM_TRAINERS + NUM_PS + 1  # Trainers + PS + Master
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
