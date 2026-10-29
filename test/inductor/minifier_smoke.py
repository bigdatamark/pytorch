# Owner(s): ["module: inductor"]
#
# This smoketest is referenced in the internal-only minifier runbook
# https://docs.google.com/document/d/18L9e7bZSBpJ7gGbwlUV13LasmjiEX2lree2pl-SdbCU/edit
# [MODULE DOC] Smoke test script for the internal minifier workflow. At import time it sets the
# environment variable that makes dynamo generate a repro after the dynamo stage, imports torch,
# and sets an inductor config flag that injects a compile error for relu on CPU (testing only).
# Then it immediately runs the minifier driver, so importing this module has side effects and
# is expected to fail with a compile error and emit minifier output.
import os


os.environ["TORCHDYNAMO_REPRO_AFTER"] = "dynamo"
import torch
import torch._dynamo as torchdynamo
import torch._inductor.config
import torch._ops


torch._inductor.config.cpp.inject_relu_bug_TESTING_ONLY = "compile_error"


def func(x):
    # [FUNCTION DOC] HELPER: Model function for the smoke test. Applies sigmoid, multiplies by
    # a ones tensor of two elements, applies relu (where the bug is injected), adds a zeros
    # tensor of two elements and rounds. Returns the resulting tensor.
    x = torch.sigmoid(x)
    x = torch.mul(x, torch.ones(2))
    x = torch.relu(x)
    x = torch.add(x, torch.zeros(2))
    x = torch.ops.aten.round(x)
    return x


def run_internal_minifier():
    # [FUNCTION DOC] HELPER: Sets the dynamo debug directory root to the current directory (global
    # config side effect), compiles func with torch.compile and calls it on a ones tensor of two
    # elements. The injected relu bug makes this trigger the minifier. Returns nothing.
    torchdynamo.config.debug_dir_root = "."
    f_opt = torch.compile(func)
    f_opt(torch.ones(2))


run_internal_minifier()
