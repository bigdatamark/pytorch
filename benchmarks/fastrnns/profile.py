# [MODULE DOC] Profiling driver for the fastrnns benchmarks. Provides run_rnn (cuda only, runs
# forward/backward iterations), profile (loops over named runners), and helpers to shell out to
# nvprof and write an nvvp output file. The main block parses sizes and either runs the profile
# directly (internal-run) or re-invokes this module under nvprof. Output directory is a constant.
import argparse
import datetime
import subprocess
import sys
import time

import torch

from .runner import get_nn_runners


def run_rnn(
    name,
    rnn_creator,
    nloops=5,
    seqLength=100,
    numLayers=1,
    inputSize=512,
    hiddenSize=512,
    miniBatch=64,
    device="cuda",
    seed=None,
):
    # [FUNCTION DOC] Run nloops forward/backward iterations of one RNN creator for profiling.
    # Raises AssertionError unless device is cuda. Builds the model definition from the sizes and
    # seed and runs run_iter nloops times. Returns None.
    def run_iter(modeldef):
        # [FUNCTION DOC] HELPER: One iteration: forward, optional backward setup and backward,
        # zero the gradients of all params if backward exists (no None check), then synchronize
        # cuda.
        # Forward
        forward_output = modeldef.forward(*modeldef.inputs)

        # "loss computation" and backward
        if modeldef.backward_setup is not None:
            backward_input = modeldef.backward_setup(forward_output)
        else:
            backward_input = forward_output
        if modeldef.backward is not None:
            modeldef.backward(*backward_input)

        # "Update" parameters
        if modeldef.backward is not None:
            with torch.no_grad():
                for param in modeldef.params:
                    param.grad.zero_()
        torch.cuda.synchronize()

    if device != "cuda":
        raise AssertionError(f"device must be 'cuda', but got '{device}'")
    creator_args = dict(
        seqLength=seqLength,
        numLayers=numLayers,
        inputSize=inputSize,
        hiddenSize=hiddenSize,
        miniBatch=miniBatch,
        device=device,
        seed=seed,
    )
    modeldef = rnn_creator(**creator_args)

    [run_iter(modeldef) for _ in range(nloops)]


def profile(
    rnns,
    sleep_between_seconds=1,
    nloops=5,
    internal_run=True,  # Unused, get rid of this TODO
    seqLength=100,
    numLayers=1,
    inputSize=512,
    hiddenSize=512,
    miniBatch=64,
    device="cuda",
    seed=None,
):
    # [FUNCTION DOC] For every runner name in rnns, resolve it via get_nn_runners, run run_rnn
    # inside its context manager with the given sizes and then sleep sleep_between_seconds. The
    # internal_run argument is accepted only so the parsed command line can be passed through; it
    # is unused.
    params = dict(
        seqLength=seqLength,
        numLayers=numLayers,
        inputSize=inputSize,
        hiddenSize=hiddenSize,
        miniBatch=miniBatch,
        device=device,
        seed=seed,
    )
    for name, creator, context in get_nn_runners(*rnns):
        with context():
            run_rnn(name, creator, nloops, **params)
            time.sleep(sleep_between_seconds)


def system(command):
    # [FUNCTION DOC] HELPER: Print the command, run it through the shell, wait for it and return
    # (return code, stdout text, stderr text), both decoded as ascii (non ascii output would
    # raise).
    """Returns (return-code, stdout, stderr)"""
    print(f"[system] {command}")
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output, err = p.communicate()
    rc = p.returncode
    output = output.decode("ascii")
    err = err.decode("ascii")
    return rc, output, err


def describe_sizes(**sizes):
    # [FUNCTION DOC] HELPER: Build a short tag string from the five size keywords (seqLength,
    # numLayers, inputSize, hiddenSize, miniBatch); a missing key raises KeyError.
    # seqLength, numLayers, inputSize, hiddenSize, miniBatch
    return "s{}-l{}-i{}-h{}-b{}".format(
        sizes["seqLength"],
        sizes["numLayers"],
        sizes["inputSize"],
        sizes["hiddenSize"],
        sizes["miniBatch"],
    )


OUTPUT_DIR = "~/profout/"


def nvprof_output_filename(rnns, **params):
    # [FUNCTION DOC] HELPER: Build the nvvp output path: output directory constant, runner names
    # joined with dashes, the size tag and a current date and time stamp. Returns the string.
    rnn_tag = "-".join(rnns)
    size_tag = describe_sizes(**params)
    date_tag = datetime.datetime.now().strftime("%m%d%y-%H%M")
    return f"{OUTPUT_DIR}prof_{rnn_tag}_{size_tag}_{date_tag}.nvvp"


def nvprof(cmd, outpath):
    # [FUNCTION DOC] HELPER: Run the command under nvprof writing to outpath, via the system
    # helper. Returns the (return code, stdout, stderr) triple.
    return system(f"nvprof -o {outpath} {cmd}")


def full_profile(rnns, **args):
    # [FUNCTION DOC] Re-launch this module as a subprocess under nvprof with internal-run. Builds
    # command line flags from the keyword arguments plus the runner list, derives the output path
    # and raises RuntimeError with captured stderr and stdout if nvprof exits non zero.
    # KNOWN ISSUE: every keyword (including the boolean internal_run) is turned into a key=value
    # flag, so a store_true flag gets an explicit value such as False, which argparse likely
    # rejects.
    profile_args = []
    for k, v in args.items():
        profile_args.append(f"--{k}={v}")
    profile_args.append(f"--rnns {' '.join(rnns)}")
    profile_args.append("--internal-run")

    outpath = nvprof_output_filename(rnns, **args)

    cmd = f"{sys.executable} -m fastrnns.profile {' '.join(profile_args)}"
    rc, stdout, stderr = nvprof(cmd, outpath)
    if rc != 0:
        raise RuntimeError(f"stderr: {stderr}\nstdout: {stdout}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile RNNs")

    parser.add_argument("--seqLength", default="100", type=int)
    parser.add_argument("--numLayers", default="1", type=int)
    parser.add_argument("--inputSize", default="512", type=int)
    parser.add_argument("--hiddenSize", default="512", type=int)
    parser.add_argument("--miniBatch", default="64", type=int)
    parser.add_argument(
        "--sleep-between-seconds", "--sleep_between_seconds", default="1", type=int
    )
    parser.add_argument("--nloops", default="5", type=int)

    parser.add_argument("--rnns", nargs="*", help="What to run. cudnn, aten, jit, etc")

    # if internal_run, we actually run the rnns.
    # if not internal_run, we shell out to nvprof with internal_run=T
    parser.add_argument(
        "--internal-run",
        "--internal_run",
        default=False,
        action="store_true",
        help="Don't use this",
    )
    args = parser.parse_args()
    if args.rnns is None:
        args.rnns = ["cudnn", "aten", "jit"]
    print(args)

    if args.internal_run:
        profile(**vars(args))
    else:
        full_profile(**vars(args))
