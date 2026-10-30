# [MODULE DOC] CI tool comparing an actual accuracy CSV from a dynamo benchmark run against
# the expected baseline CSV. Holds a global set of known flaky model names, prints one status
# line per model, and exits with status 1 from the entry point if any model regressed or
# improved relative to the baseline (so the baseline must be updated).
import argparse
import os
import sys
import textwrap

import pandas as pd


# Hack to have something similar to DISABLED_TEST. These models are flaky.

flaky_models = {
    "yolov3",
    "detectron2_maskrcnn_r_101_c4",
    "XGLMForCausalLM",  # discovered in https://github.com/pytorch/pytorch/pull/128148
    "moondream",  # discovered in https://github.com/pytorch/pytorch/pull/159291
    # discovered in https://github.com/pytorch/pytorch/issues/161419. Its not flaky but really hard to repro, so skipping it
    "mobilenetv3_large_100",
    # https://github.com/pytorch/pytorch/issues/163670
    "vision_maskrcnn",
    # Old TorchBench SAM is flaky even with mask IoU checking.
    # Borderline mask thresholding can flip a single pixel and give IoU=0
    # when both masks are nearly empty.
    # https://github.com/pytorch/pytorch/issues/176776
    "sam",
}


def get_eager_nondeterministic_models(expected_filename: str) -> set[str]:
    # [FUNCTION DOC] Returns the set of model names whose accuracy may legitimately show the
    # eager-run-differs status. Depends only on the expected file path (path separators
    # normalized): non-empty for the timm and torchbench inductor training baselines, and an
    # empty set for every other file.
    expected_filename = expected_filename.replace(os.sep, "/")
    if expected_filename.endswith("ci_expected_accuracy/inductor_timm_training.csv"):
        return {
            "mobilenetv2_100",
            "tf_efficientnet_b0",
        }
    if expected_filename.endswith(
        "ci_expected_accuracy/inductor_torchbench_training.csv"
    ):
        return {
            "mnasnet1_0",
            "mobilenet_v2",
            "shufflenet_v2_x1_0",
        }
    return set()


def get_field(csv, model_name: str, field: str):
    # [FUNCTION DOC] HELPER: Returns the single value in the given column for the row whose
    # name column matches the model, or None if the lookup fails for any reason (no row,
    # several rows, or missing column).
    try:
        return csv.loc[csv["name"] == model_name][field].item()
    except Exception:
        return None


def check_accuracy(actual_csv, expected_csv, expected_filename):
    # [FUNCTION DOC] Compares per-model accuracy of the actual and expected data frames and
    # prints a status line for each. Statuses: matching values give PASS or XFAIL; a missing
    # actual or expected value or a non-pass result is a failure; a pass-to-pass-expected
    # difference for allowed eager-nondeterministic models, or any mismatch for flaky models,
    # is only reported; a pass where failure was expected is an improvement. If the expected
    # file path mentions rocm, a larger set of flaky models is added to the module-level
    # global set (side effect). Returns the failed list, or the improved list if none failed
    # (possibly empty), and a message explaining how to update the baseline.
    failed = []
    improved = []
    eager_nondeterministic_models = get_eager_nondeterministic_models(expected_filename)

    if "rocm" in expected_filename:
        flaky_models.update(
            {
                "Background_Matting",
                "mnasnet1_0",
                "llava",
                "repvgg_a2",
                "resnet152",
                "resnet18",
                "resnet50",
                "stable_diffusion_unet",
                "torchrec_dlrm",
                "shufflenet_v2_x1_0",
                "vgg16",
                "BERT_pytorch",
                # LLM
                "google/gemma-2-2b",
                "tts_angular",  # RuntimeError: Cannot access data pointer of Tensor
                # Discovered on gfx950 CI after ROCm 7.2 upgrade, eager mode non determinism
                "alexnet",
                "demucs",
            }
        )

    for model in actual_csv["name"]:
        accuracy = get_field(actual_csv, model, "accuracy")
        expected_accuracy = get_field(expected_csv, model, "accuracy")

        if accuracy is None:
            status = "MISSING_ACCURACY:"
            failed.append(model)
        elif expected_accuracy is None:
            status = "MISSING_EXPECTED:"
            failed.append(model)
        elif accuracy == expected_accuracy:
            status = "PASS" if expected_accuracy == "pass" else "XFAIL"
            print(f"{model:34}  {status}")
            continue
        elif (
            expected_accuracy == "pass"
            and accuracy == "eager_two_runs_differ"
            and model in eager_nondeterministic_models
        ):
            status = "EAGER_NONDETERMINISTIC:"
        elif model in flaky_models:
            if accuracy == "pass":
                # model passed but marked xfailed
                status = "PASS_BUT_FLAKY:"
            else:
                # model failed but marked passe
                status = "FAIL_BUT_FLAKY:"
        elif accuracy != "pass":
            status = "FAIL:"
            failed.append(model)
        else:
            status = "IMPROVED:"
            improved.append(model)
        print(
            f"{model:34}  {status:9} accuracy={accuracy}, expected={expected_accuracy}"
        )

    msg = ""
    if failed or improved:
        if failed:
            msg += textwrap.dedent(
                f"""
            Error: {len(failed)} models have accuracy status regressed:
                {" ".join(failed)}

            """
            )
        if improved:
            msg += textwrap.dedent(
                f"""
            Improvement: {len(improved)} models have accuracy status improved:
                {" ".join(improved)}

            """
            )
        sha = os.getenv("SHA1", "{your CI commit sha}")
        msg += textwrap.dedent(
            f"""
        If this change is expected, you can update `{expected_filename}` to reflect the new baseline.
        from pytorch/pytorch root, run
        `python benchmarks/dynamo/ci_expected_accuracy/update_expected.py {sha}`
        and then `git add` the resulting local changes to expected CSVs to your commit.
        """
        )
    return failed or improved, msg


def main():
    # [FUNCTION DOC] Entry point: reads the required actual and expected CSV paths, runs the
    # accuracy check, and prints the message and exits with status 1 if anything regressed
    # or improved.
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", type=str, required=True)
    parser.add_argument("--expected", type=str, required=True)
    args = parser.parse_args()

    actual = pd.read_csv(args.actual)
    expected = pd.read_csv(args.expected)

    failed, msg = check_accuracy(actual, expected, args.expected)
    if failed:
        print(msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
