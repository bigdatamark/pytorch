# mypy: allow-untyped-defs
# [MODULE DOC] Debug utilities that numerically compare a model before and after pre/post grad fx passes.
# Runs both models for several iterations with deterministic seeds and compares parameters, forward outputs,
# gradients and (optionally) parameters after an SGD step; mismatches are only logged, never raised.
# Import-time side effect: sets the CUBLAS_WORKSPACE_CONFIG environment variable to a fixed value.
import gc
import logging
import os
import random
import traceback

import numpy

import torch
import torch.optim as optim
from torch.utils._ordered_set import OrderedSet

from .. import config


logger: logging.Logger = logging.getLogger(__name__)

MAIN_RANDOM_SEED = 1337

# Set the CUBLAS_WORKSPACE_CONFIG environment variable
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# If the two forward functions involve any non-deterministic operations,
# such as certain types of parallelism or asynchronous execution,
# this can also lead to different outputs.
def set_deterministic() -> None:
    # [FUNCTION DOC] Seeds torch, random and numpy with the fixed main seed and enables deterministic algorithms globally.
    # Side effects: mutates global RNG state and the torch deterministic-algorithms flag. Returns nothing.
    """Make torch manual seed deterministic."""

    torch.manual_seed(MAIN_RANDOM_SEED)
    random.seed(MAIN_RANDOM_SEED)
    numpy.random.seed(MAIN_RANDOM_SEED)
    torch.use_deterministic_algorithms(True)


def clean_memory() -> None:
    # [FUNCTION DOC] Runs garbage collection and empties the CUDA caching allocator cache to reduce OOM risk.
    # Requires a CUDA-capable torch build. Returns nothing.
    """Clean memory to avoid OOM."""
    gc.collect()
    torch.cuda.empty_cache()


# We compare the numerical results before and after pre/post grad fx passes
# transformation to make sure the numerical results are the same.
def compare_dict_tensors(dict_base, dict_control, precision):
    # [FUNCTION DOC] Compares two mappings of name to tensor within a tolerance (rtol and atol both set to precision, NaNs equal).
    # Returns False immediately if the key counts differ; otherwise returns True only if every pair of non-None tensors is close.
    # Entries where either side is None are skipped; mismatches are logged as warnings/debug, not raised.
    # KNOWN ISSUE: a key missing from the control mapping only logs a warning, then the lookup of that key in the control mapping raises a KeyError.
    if len(OrderedSet(dict_base.keys())) != len(OrderedSet(dict_control.keys())):
        logger.warning("Mismatch keys found before and after pre/post grad fx passes.")
        logger.debug("keys before pre/post grad fx passes %s", dict_base.keys())
        logger.debug("keys after pre/post grad fx passes %s", dict_control.keys())
        return False
    is_allclose = True
    for key in dict_base:
        if key not in dict_control:
            logger.warning(
                "Mismatch parameter name %s does not exist after pre/post grad fx passes",
                key,
            )
        # Some parameters have `None`, and not every param has a valid .grad field, we skip them
        if dict_base[key] is None or dict_control[key] is None:
            continue
        if not torch.allclose(
            dict_base[key],
            dict_control[key],
            rtol=precision,
            atol=precision,
            equal_nan=True,
        ):
            logger.warning(
                "Mismatch parameter values found before and after pre/post grad fx passes."
            )
            logger.debug("value before pre/post grad fx passes %s", dict_base[key])
            logger.debug("value after pre/post grad fx passes %s", dict_control[key])
            is_allclose = False
    return is_allclose


def compare_tuple_tensors(tuple_base, tuple_control, precision):
    # [FUNCTION DOC] Compares two sequences of tensors elementwise within a tolerance (rtol and atol both set to precision, NaNs equal).
    # Returns False if lengths differ; otherwise True only if all pairs of non-None entries are close. None entries are skipped.
    # Mismatches are logged only.
    if len(tuple_base) != len(tuple_control):
        logger.warning(
            "Mismatch fw output length. before transformation: %s, after transformation: %s",
            len(tuple_base),
            len(tuple_control),
        )
        return False
    is_allclose = True
    for i in range(len(tuple_base)):
        # Some parameters have `None`, we skip them
        if tuple_base[i] is None or tuple_control[i] is None:
            continue
        if not torch.allclose(
            tuple_base[i],
            tuple_control[i],
            rtol=precision,
            atol=precision,
            equal_nan=True,
        ):
            logger.debug(
                "forward output before pre/post grad fx passes %s", tuple_base[i]
            )
            logger.debug(
                "forward output after pre/post grad fx passes %s", tuple_control[i]
            )
            is_allclose = False
    return is_allclose


def compare_parameters(model_base, model_control, precision):
    # [FUNCTION DOC] Compares the named parameters of two models via the dict comparison helper. Returns True if all are close.
    return compare_dict_tensors(
        dict(model_base.named_parameters()),
        dict(model_control.named_parameters()),
        precision,
    )


def compare_forward_output(pred_base, pred_control, precision):
    # [FUNCTION DOC] Compares two forward outputs (sequences of tensors) via the tuple comparison helper. Returns True if all are close.
    return compare_tuple_tensors(
        pred_base,
        pred_control,
        precision,
    )


def compare_gradients(model_base, model_control, precision):
    # [FUNCTION DOC] Collects the gradient of each named parameter of both models and compares them via the dict comparison helper.
    # Parameters without a gradient are None and are skipped. Returns True if all are close.
    grad_base = {key: param.grad for key, param in model_base.named_parameters()}
    grad_pt2 = {key: param.grad for key, param in model_control.named_parameters()}
    return compare_dict_tensors(
        grad_base,
        grad_pt2,
        precision,
    )


def run_model(
    model_base, model_control, model_input, num_iterations=10, precision=1e-4
):
    # [FUNCTION DOC] Runs both models on the same inputs for num_iterations iterations and logs numeric comparison results.
    # Each iteration: reseed deterministically before each forward, compare parameters and forward outputs, then run backward on the
    # first output of each model and compare gradients (exceptions are logged and printed, not raised). If the config option
    # requires_optimizer is set, also steps an SGD optimizer (learning rate 0.01) on each model and compares parameters again.
    # Side effects: frees CUDA memory first, mutates model parameters and gradients, logs. Returns nothing.
    clean_memory()
    for i in range(num_iterations):
        logger.info("start %s iteration", i)
        set_deterministic()
        pred_base = model_base(*model_input)
        set_deterministic()
        pred_control = model_control(*model_input)

        res = compare_parameters(model_base, model_control, precision)
        logger.info("compare parameters. Numerical result : %s", res)

        res = compare_forward_output(pred_base, pred_control, precision)
        logger.info("compare loss/predict. Numerical result : %s", res)
        # tensor may not have a grad_fn
        try:
            _ = pred_base[0].sum().backward(retain_graph=True)
            _ = pred_control[0].sum().backward(retain_graph=True)
            res = compare_gradients(model_base, model_control, precision)
            logger.info("compare param grad. Numerical result : %s", res)
        except Exception:
            logger.exception("Exception when comparing gradients")
            traceback.print_exc()

        if config.fx_passes_numeric_check["requires_optimizer"]:
            try:
                optimizer_base = optim.SGD(
                    [param for name, param in model_base.named_parameters()], lr=0.01
                )
                optimizer_base.step()

                optimizer_control = optim.SGD(
                    [param for name, param in model_control.named_parameters()], lr=0.01
                )
                optimizer_control.step()

                res = compare_parameters(model_base, model_control, precision)
                logger.info(
                    "compare parameters with optimizer added. Numerical result : %s",
                    res,
                )
            except Exception:
                logger.exception(
                    "Exception when optimizer is added to check parameter names"
                )
                traceback.print_exc()
        else:
            logger.warning(
                "no parameter with optimizer to compare with length %s before transformation"
                " and the length %s after transformation",
                len(dict(model_base.named_parameters())),
                len(dict(model_control.named_parameters())),
            )


def numeric_check_if_enabled(
    gm_before_fx_passes,
    gm_after_fx_passes,
    example_inputs,
    num_iterations,
    precision,
):
    # [FUNCTION DOC] Runs the numeric comparison of the graph module before and after fx passes with anomaly detection enabled.
    # Args: the two graph modules, example inputs, iteration count and precision. All exceptions are caught, logged as a warning
    # and printed, so a failing check never blocks the model run. Returns nothing.
    # need to topo-sort graphmodule before we run the model,
    # otherwise it may fail as refer before def
    # fail silently in order not to block the model run
    try:
        with torch.autograd.set_detect_anomaly(True):
            run_model(
                gm_before_fx_passes,
                gm_after_fx_passes,
                example_inputs,
                num_iterations=num_iterations,
                precision=precision,
            )
    except Exception as e:
        logger.warning(
            "Runtime numeric check failed in pre grad fx passes with error: %s", e
        )
        traceback.print_exc()
