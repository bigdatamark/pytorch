# [MODULE DOC] Command-line benchmark measuring the overhead of the autograd record-function
# machinery on jit-compiled ResNet-50 and LSTM models, for several thread counts, with the
# callbacks enabled (using an empty test observer) and disabled. Results are gathered with the
# benchmark timer and printed as a comparison table. Defines a registry of model preparers and
# the thread-count list; the main block parses options and runs the selected models. Depends on
# the fastrnns benchmark factory and torchvision.
import argparse
import sys

from benchmarks.fastrnns.factory import lstm_creator

from torchvision.models import resnet50

import torch
import torch.utils.benchmark as benchmark_utils


def prepare_lstm_jit(bench_args):
    # [FUNCTION DOC] HELPER: Builds a scripted LSTM through the fastrnns creator on cpu using
    # the sequence length, layer count, input size, hidden size and mini-batch size options from
    # the argument namespace. Returns a pair of the model inputs and its forward function.
    model_def = lstm_creator(
        script=True,
        seqLength=bench_args.lstmSeqLength,
        numLayers=bench_args.lstmNumLayers,
        inputSize=bench_args.lstmInputSize,
        hiddenSize=bench_args.lstmHiddenSize,
        miniBatch=bench_args.lstmMiniBatch,
        device="cpu",
    )
    return model_def.inputs, model_def.forward


def prepare_resnet50_jit(bench_args):
    # [FUNCTION DOC] HELPER: Builds a ResNet-50, traces it with one random batch of 32
    # images and returns a pair of the inputs tuple and the traced model. The argument
    # namespace is unused.
    model = resnet50()
    inputs = (torch.randn(32, 3, 224, 224),)
    model = torch.jit.trace(model, inputs)
    return inputs, model


MODELS = {
    "resnet50_jit": prepare_resnet50_jit,
    "lstm_jit": prepare_lstm_jit,
}

NUM_THREADS = [1, 2, 4, 8, 16, 32]


def run_bench(model_names, bench_args):
    # [FUNCTION DOC] Runs the overhead benchmark for each named model. For each model it builds
    # the model and inputs, runs the warmup iterations, then for every thread count measures
    # with record function enabled (clearing callbacks and installing an empty test observer
    # with a small sampling rate) and with it disabled, using a timer with automatic run counts
    # and the minimum run time option, printing progress and each result. Finally prints a
    # comparison table of all results. Side effects: toggles global autograd record-function
    # state and callbacks, writes to stdout. Raises a key error for a model name not in the
    # registry. The loops option is accepted by the parser but not used here.
    results = []
    for model_name in model_names:
        model_creator = MODELS[model_name]
        inputs, model = model_creator(bench_args)

        print("Benchmarking RecordFunction overhead for", model_name)
        print("Running warmup...", end=" ")
        sys.stdout.flush()
        for _ in range(bench_args.warmup):
            model(*inputs)
        print("finished")

        for num_threads in NUM_THREADS:
            for with_rec_fn in [True, False]:
                torch.autograd._enable_record_function(with_rec_fn)
                torch.autograd._clear_callbacks()
                if with_rec_fn:
                    torch.autograd._set_empty_test_observer(True, 0.0001)

                print(
                    "Running {} RecordFunction, num threads {} ...".format(
                        "with" if with_rec_fn else "without", num_threads
                    ),
                    end=" ",
                )
                sys.stdout.flush()
                timer = benchmark_utils.Timer(
                    stmt="model(*inputs)",
                    globals={"model": model, "inputs": inputs},
                    description=model_name,
                    label="Record function overhead",
                    sub_label=f"with{'' if with_rec_fn else 'out'}_rec_fn, num_threads {num_threads}",
                    num_threads=num_threads,
                )
                result = timer.blocked_autorange(
                    min_run_time=bench_args.timer_min_run_time
                )
                print("finished")
                print(result)
                sys.stdout.flush()
                results.append(result)

    comparison = benchmark_utils.Compare(results)
    comparison.trim_significant_figures()
    comparison.highlight_warnings()
    comparison.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark RecordFunction overhead for ResNet and LSTM models"
    )

    parser.add_argument(
        "--models",
        nargs="*",
        default=["lstm_jit"],
        help="What model to run: " + str(MODELS.keys()),
    )

    parser.add_argument("--lstmSeqLength", default="100", type=int)
    parser.add_argument("--lstmNumLayers", default="1", type=int)
    parser.add_argument("--lstmInputSize", default="512", type=int)
    parser.add_argument("--lstmHiddenSize", default="512", type=int)
    parser.add_argument("--lstmMiniBatch", default="64", type=int)
    parser.add_argument("--warmup", default="2", type=int)
    parser.add_argument("--nloops", default="50", type=int)
    parser.add_argument(
        "--timer-min-run-time", "--timer_min_run_time", default=120, type=int
    )

    args = parser.parse_args()

    models = args.models or MODELS.keys()

    for model in models:
        if model not in MODELS:
            raise AssertionError(
                f"Unknown model: {model}. Available models: {list(MODELS.keys())}"
            )
    run_bench(models, args)
