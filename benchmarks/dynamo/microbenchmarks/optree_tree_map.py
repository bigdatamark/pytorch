#!/usr/bin/env python3
# [MODULE DOC] Script profiling Dynamo compile time of a function that calls optree tree_map
# repeatedly over two identical-shaped pytrees of tensors, using the eager backend and full
# graph. Command line options pick device, loop count, tree depth, branching, tensor size and
# profile output path. Import-time side effect: defines the default profile path next to this
# file.

import argparse
import time
from pathlib import Path

import optree

import torch
import torch._dynamo
from torch._dynamo.debug_utils import profile_to_file


PROFILE_PATH = Path(__file__).with_name("optree_tree_map.prof")


def make_tensor_tree(depth: int, branching_factor: int, tensor_size: int, device: str):
    # [FUNCTION DOC] Build a pytree of tensors to the given depth. At depth zero a leaf is a
    # random square tensor on the device; otherwise a mapping with keys for a tensor, a list of
    # the child trees and a tuple of the same child trees. Returns the tree.
    """Create a moderately deep pytree populated with tensors."""

    def _make_level(level: int):
        # [FUNCTION DOC] HELPER: Recursive builder for one tree level; returns a random tensor
        # at level zero, else a mapping with a tensor, a list of branching-factor children and a
        # tuple of the same children.
        if level == 0:
            return torch.randn(tensor_size, tensor_size, device=device)

        children = [_make_level(level - 1) for _ in range(branching_factor)]
        return {
            "tensor": torch.randn(tensor_size, tensor_size, device=device),
            "list": list(children),
            "tuple": tuple(children),
        }

    return _make_level(depth)


def add_leaf(lhs: torch.Tensor, *rest: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] Leaf function for tree_map: returns the sum of its first tensor argument
    # and all extra tensor arguments.
    out = lhs
    for other in rest:
        out = out + other
    return out


def optree_tree_map_loop(lhs, rhs, loop_iters):
    # [FUNCTION DOC] Repeatedly map the leaf-adding function over the running tree and the
    # second tree in the torch optree namespace, for the given number of iterations, starting
    # from the first tree. Returns the final tree.
    tree = lhs
    for _ in range(loop_iters):
        tree = optree.tree_map(
            add_leaf,
            tree,
            rhs,
            namespace="torch",
        )
    return tree


def _capture_compile_profile(args, lhs, rhs) -> None:
    # [FUNCTION DOC] HELPER: Create the profile output directory if missing, then compile the
    # loop function with the eager backend and full graph after resetting Dynamo, run it once
    # under the file profiler decorator, and print where the profile is collected. The profile
    # file is written when the process exits.
    profile_path = Path(args.profile_out)
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    @profile_to_file(str(profile_path))
    def _run_compile() -> None:
        # [FUNCTION DOC] HELPER: Profiled compile-and-run of the loop function on the enclosing
        # trees using the configured iteration count.
        torch._dynamo.reset()
        compiled = torch.compile(
            optree_tree_map_loop,
            backend="eager",
            fullgraph=True,
        )
        compiled(lhs, rhs, args.loop_iters)

    print(f"Collecting compile-only cProfile at {profile_path}")
    _run_compile()


def _parse_args():
    # [FUNCTION DOC] HELPER: Define and parse command line options (device default cuda when
    # available, loop iterations default 50, tree depth default 2, branching factor default 2,
    # tensor size default 1, profile output path) and return the namespace.
    parser = argparse.ArgumentParser()
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device, help="Device to run on")
    parser.add_argument(
        "--loop-iters",
        type=int,
        default=50,
        help="Number of tree_map calls per compiled invocation",
    )
    parser.add_argument(
        "--tree-depth", type=int, default=2, help="Depth of the constructed pytree"
    )
    parser.add_argument(
        "--branching-factor",
        type=int,
        default=2,
        help="Branching factor for list/tuple nodes",
    )
    parser.add_argument(
        "--tensor-size",
        type=int,
        default=1,
        help="Edge length for square tensor leaves",
    )
    parser.add_argument(
        "--profile-out",
        default=str(PROFILE_PATH),
        help="Destination .prof file for the compile-time cProfile",
    )
    return parser.parse_args()


def main() -> None:
    # [FUNCTION DOC] Parse arguments, build the two trees, time the profiled compile run, and
    # print the elapsed seconds.
    args = _parse_args()

    lhs = make_tensor_tree(
        args.tree_depth, args.branching_factor, args.tensor_size, args.device
    )
    rhs = make_tensor_tree(
        args.tree_depth, args.branching_factor, args.tensor_size, args.device
    )

    t0 = time.perf_counter()
    _capture_compile_profile(args, lhs, rhs)
    t1 = time.perf_counter()
    print(f"Took {t1 - t0:.1f}s")


if __name__ == "__main__":
    main()
