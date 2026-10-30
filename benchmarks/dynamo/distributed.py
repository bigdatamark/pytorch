# [MODULE DOC] Command-line driver that times a training step of a torchbench or toy model
# under eager or dynamo, optionally with DDP or FSDP, torchviz graph dump, or profiler trace.
# Meant to be launched once per rank (rank/world size come from the environment). Imports
# helpers with a relative-then-absolute fallback; running as a script parses arguments and
# prints the mean latency.
import argparse
import logging
import os
from functools import partial

import torch
import torch._dynamo as dynamo
import torch.utils._pytree as pytree
from torch._dynamo.testing import reduce_to_scalar_loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity, record_function


try:
    from .common import timed
    from .dist_util import apply_fsdp, cleanup, get_model, model_iter_fn, setup
except ImportError:
    from common import timed
    from dist_util import apply_fsdp, cleanup, get_model, model_iter_fn, setup

log = logging.getLogger(__name__)


def torchviz_model(args, model, inputs, rank):
    # [FUNCTION DOC] Run one forward pass, build an autograd graph with torchviz (imported
    # lazily) showing attrs and saved tensors, and on rank 0 render it to a file named
    # torchviz.dot. The args parameter is unused.
    from torchviz import make_dot

    outputs = model(*inputs)
    loss = reduce_to_scalar_loss(outputs)
    parameter_names = dict(model.named_parameters())
    dot = make_dot(loss, params=parameter_names, show_attrs=True, show_saved=True)
    if rank == 0:
        dot.render("torchviz.dot")


def profile_model(args, model, inputs, rank):
    # [FUNCTION DOC] Profile args.repeat forward+backward iterations on CPU and CUDA, with
    # labeled Forward and Backward regions. On rank 0 export a chrome trace to args.trace_file.
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(args.repeat):
            with record_function("Forward"):
                outputs = model(*inputs)
                loss = reduce_to_scalar_loss(outputs)
            with record_function("Backward"):
                loss.backward()
    if rank == 0:
        prof.export_chrome_trace(args.trace_file)


def run_model(args, model, inputs, key):
    # [FUNCTION DOC] Benchmark one model in this process. Reads rank and world size from the
    # environment, initializes the process group, moves model and tensor inputs to the device
    # of this rank, wraps with FSDP or DDP if requested, and optionally compiles with dynamo
    # (backend name from args.dynamo; "print" prints each captured graph). Resets dynamo,
    # may change dynamo config, and disables inductor cudagraphs when combined with FSDP.
    # Runs 3 warmup iterations then args.repeat timed ones, optionally dumps torchviz/profile
    # output, destroys the process group and returns the total time. The key argument is unused.
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    # result_q = []

    setup(rank, world_size)
    if args.device == "cuda":
        # needed for FSDP
        torch.cuda.set_device(rank)

    dev_rank = f"{args.device}:{rank}"
    model = model.to(dev_rank)

    def move_tensor(maybe_tensor):
        # [FUNCTION DOC] HELPER: Move a value to this rank's device if it is a tensor, else
        # return it unchanged.
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(dev_rank)
        return maybe_tensor

    inputs = pytree.tree_map(move_tensor, inputs)

    if args.fsdp:
        model = apply_fsdp(
            args,
            model,
            use_checkpointing=args.fsdp_checkpoint,
            use_wrap_policy=args.fsdp_wrap,
        )
    elif args.ddp:
        model = DDP(model)

    if args.verbose:
        print(model)

    if args.dynamo:
        dynamo.reset()
        if args.verbose:
            dynamo.config.verbose = True
            dynamo.config.log_level = logging.DEBUG
        if args.dynamo_no_optimize_ddp:
            dynamo.config.optimize_ddp = False
        if args.dynamo == "inductor" and args.fsdp:
            torch._inductor.config.triton.cudagraphs = False
            log.warning("disabling inductor cudagraphs for compatibility with FSDP")

        def print_compile(gm, ex):
            # [FUNCTION DOC] HELPER: Debug dynamo backend: print the captured graph and return the
            # graph module unchanged (example inputs unused).
            print(
                f"print_compile:\n{str(gm.graph)}\n-----------------------------------------"
            )
            return gm

        dynamo_ctx = dynamo.optimize(
            print_compile if args.dynamo == "print" else args.dynamo
        )
        model = dynamo_ctx(model)

    # warmup
    _ = timed(model, model_iter_fn, inputs, times=3, return_result=False)
    t_total = timed(
        model, model_iter_fn, inputs, times=args.repeat, return_result=False
    )
    if args.torchviz:
        torchviz_model(args, model, inputs, rank)
    if args.profile:
        profile_model(args, model, inputs, rank)

    cleanup()
    return t_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dynamo",
        default=None,
        help="if set to a str, uses dynamo[str] backend. else, eager",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--batch-size", "--batch_size", default=None)
    parser.add_argument(
        "--torchviz", action="store_true", help="Dump autograd graph with torchviz"
    )
    parser.add_argument("--profile", action="store_true", help="Run the profiler")
    parser.add_argument(
        "--trace-file", "--trace_file", default="profile.json", help="Run the profiler"
    )
    parser.add_argument("--repeat", default=10, help="Repeats for timing run")
    parser.add_argument(
        "--dynamo-no-optimize-ddp",
        "--dynamo_no_optimize_ddp",
        action="store_true",
        help="Disable dynamo's ddp optimizer (enabled by default)",
    )
    parser.add_argument(
        "--fsdp-checkpoint",
        "--fsdp_checkpoint",
        action="store_true",
        help="Use gradient checkpointing via model-specific policy",
    )
    parser.add_argument(
        "--fsdp-wrap",
        "--fsdp_wrap",
        action="store_true",
        help="Apply fsdp to submodules via model-specific policy",
    )

    dist_arg = parser.add_mutually_exclusive_group()
    dist_arg.add_argument("--ddp", action="store_true")
    dist_arg.add_argument("--fsdp", action="store_true")

    model_arg = parser.add_mutually_exclusive_group(required=True)
    model_arg.add_argument(
        "--torchbench-model",
        "--torchbench_model",
        help="name of torchbench model, e.g. BERT_pytorch",
    )
    model_arg.add_argument(
        "--toy-model", "--toy_model", action="store_true", help="use toy model instead"
    )
    args = parser.parse_args()

    model_name = args.torchbench_model
    if args.toy_model:
        model_name = "ToyModel"
    model, inputs = get_model(args)

    fn = partial(run_model, args, model, inputs)

    world_size = os.getenv("WORLD_SIZE", 1)
    t_total = fn(f"{model_name}_{world_size}")
    print(f"mean latency {t_total / args.repeat} across {args.repeat} runs")
