# [MODULE DOC] Correctness checker for the fastrnns benchmark models. Compares each requested
# runner against the cuDNN LSTM creator (forward outputs, backward gradients) and optionally the
# variable length python LSTM against the variable length cuDNN LSTM. Main block parses sizes,
# seed and runner names, requires cuda when a cuda device is chosen, and runs the checks.
# Mismatches drop into the debugger.
import argparse

import torch
import torch.nn as nn

from .factory import pytorch_lstm_creator, varlen_pytorch_lstm_creator
from .runner import get_nn_runners


def barf():
    # [FUNCTION DOC] HELPER: Start the interactive pdb debugger at the point of the mismatch.
    import pdb

    pdb.set_trace()


def assertEqual(tensor, expected, threshold=0.001):
    # [FUNCTION DOC] HELPER: Recursively compare tensors; lists and tuples are compared element
    # by element (extra elements of the longer one ignored). A tensor pair whose maximum absolute
    # difference exceeds the threshold triggers the debugger instead of raising. The threshold is
    # not passed down in the recursive calls (default used there). Returns None.
    if isinstance(tensor, (list, tuple)):
        for t, e in zip(tensor, expected):
            assertEqual(t, e)
    else:
        if (tensor - expected).abs().max() > threshold:
            barf()


def filter_requires_grad(tensors):
    # [FUNCTION DOC] HELPER: Return the sublist of tensors that require gradients.
    return [t for t in tensors if t.requires_grad]


def test_rnns(
    experim_creator,
    control_creator,
    check_grad=True,
    verbose=False,
    seqLength=100,
    numLayers=1,
    inputSize=512,
    hiddenSize=512,
    miniBatch=64,
    device="cuda",
    seed=17,
):
    # [FUNCTION DOC] Compare an experiment model against a control model built with identical
    # sizes and seed. Checks inputs and params match, forward outputs match, then runs backward
    # with seeded gradients on both and compares parameter gradients. Raises AssertionError if
    # either model lacks backward_setup or backward. With verbose prints the experiment's
    # optimized graph. The check_grad argument is accepted but not used. Prints progress.
    creator_args = dict(
        seqLength=seqLength,
        numLayers=numLayers,
        inputSize=inputSize,
        hiddenSize=hiddenSize,
        miniBatch=miniBatch,
        device=device,
        seed=seed,
    )

    print("Setting up...")
    control = control_creator(**creator_args)
    experiment = experim_creator(**creator_args)

    # Precondition
    assertEqual(experiment.inputs, control.inputs)
    assertEqual(experiment.params, control.params)

    print("Checking outputs...")
    control_outputs = control.forward(*control.inputs)
    experim_outputs = experiment.forward(*experiment.inputs)
    assertEqual(experim_outputs, control_outputs)

    print("Checking grads...")
    if control.backward_setup is None:
        raise AssertionError("control.backward_setup must not be None")
    if experiment.backward_setup is None:
        raise AssertionError("experiment.backward_setup must not be None")
    if control.backward is None:
        raise AssertionError("control.backward must not be None")
    if experiment.backward is None:
        raise AssertionError("experiment.backward must not be None")
    control_backward_inputs = control.backward_setup(control_outputs, seed)
    experim_backward_inputs = experiment.backward_setup(experim_outputs, seed)

    control.backward(*control_backward_inputs)
    experiment.backward(*experim_backward_inputs)

    control_grads = [p.grad for p in control.params]
    experim_grads = [p.grad for p in experiment.params]
    assertEqual(experim_grads, control_grads)

    if verbose:
        print(experiment.forward.graph_for(*experiment.inputs))
    print()


def test_vl_py(**test_args):
    # [FUNCTION DOC] Compare the variable length python LSTM runner against the variable length
    # cuDNN LSTM. Takes the creator sizes, seed and verbose from test_args (a missing key raises
    # KeyError). Pads the experiment output, compares outputs and the concatenated final states,
    # then compares gradients after a seeded backward. Raises AssertionError if backward pieces
    # are missing. Prints progress and optionally the optimized graph.
    # XXX: This compares vl_py with vl_lstm.
    # It's done this way because those two don't give the same outputs so
    # the result isn't an apples-to-apples comparison right now.
    control_creator = varlen_pytorch_lstm_creator
    name, experim_creator, context = get_nn_runners("vl_py")[0]
    with context():
        print(f"testing {name}...")
        creator_keys = [
            "seqLength",
            "numLayers",
            "inputSize",
            "hiddenSize",
            "miniBatch",
            "device",
            "seed",
        ]
        creator_args = {key: test_args[key] for key in creator_keys}

        print("Setting up...")
        control = control_creator(**creator_args)
        experiment = experim_creator(**creator_args)

        # Precondition
        assertEqual(experiment.inputs, control.inputs[:2])
        assertEqual(experiment.params, control.params)

        print("Checking outputs...")
        control_out, control_hiddens = control.forward(*control.inputs)
        control_hx, control_cx = control_hiddens
        experim_out, experim_hiddens = experiment.forward(*experiment.inputs)
        experim_hx, experim_cx = experim_hiddens

        experim_padded = nn.utils.rnn.pad_sequence(experim_out).squeeze(-2)
        assertEqual(experim_padded, control_out)
        assertEqual(torch.cat(experim_hx, dim=1), control_hx)
        assertEqual(torch.cat(experim_cx, dim=1), control_cx)

        print("Checking grads...")
        if control.backward_setup is None:
            raise AssertionError("control.backward_setup must not be None")
        if experiment.backward_setup is None:
            raise AssertionError("experiment.backward_setup must not be None")
        if control.backward is None:
            raise AssertionError("control.backward must not be None")
        if experiment.backward is None:
            raise AssertionError("experiment.backward must not be None")
        control_backward_inputs = control.backward_setup(
            (control_out, control_hiddens), test_args["seed"]
        )
        experim_backward_inputs = experiment.backward_setup(
            (experim_out, experim_hiddens), test_args["seed"]
        )

        control.backward(*control_backward_inputs)
        experiment.backward(*experim_backward_inputs)

        control_grads = [p.grad for p in control.params]
        experim_grads = [p.grad for p in experiment.params]
        assertEqual(experim_grads, control_grads)

        if test_args["verbose"]:
            print(experiment.forward.graph_for(*experiment.inputs))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test lstm correctness")

    parser.add_argument("--seqLength", default="100", type=int)
    parser.add_argument("--numLayers", default="1", type=int)
    parser.add_argument("--inputSize", default="512", type=int)
    parser.add_argument("--hiddenSize", default="512", type=int)
    parser.add_argument("--miniBatch", default="64", type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--check-grad", "--check_grad", default="True", type=bool)
    parser.add_argument("--variable-lstms", "--variable_lstms", action="store_true")
    parser.add_argument("--seed", default="17", type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--rnns", nargs="*", help="What to run. jit_premul, jit, etc")
    args = parser.parse_args()
    if args.rnns is None:
        args.rnns = ["jit_premul", "jit"]
    print(args)

    if "cuda" in args.device:
        if not torch.cuda.is_available():
            raise AssertionError(
                f"CUDA device requested ({args.device}) but CUDA is not available"
            )

    rnn_runners = get_nn_runners(*args.rnns)

    should_test_varlen_lstms = args.variable_lstms
    test_args = vars(args)
    del test_args["rnns"]
    del test_args["variable_lstms"]

    if should_test_varlen_lstms:
        test_vl_py(**test_args)

    for name, creator, context in rnn_runners:
        with context():
            print(f"testing {name}...")
            test_rnns(creator, pytorch_lstm_creator, **test_args)
