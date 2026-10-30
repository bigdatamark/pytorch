# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the reorder_for_locality_in_training inductor option. Checks that the
# environment variable controls the default (off unless set to one) by reading the flag in a
# fresh subprocess, that inference graphs still run the reorder pass, and that on a compiled
# two-branch training model the flag runs the pass, moves nodes, and leaves gradients and
# updated parameters unchanged. Device-generic tests are instantiated at import time for the
# available devices.
"""Tests for the reorder_for_locality_in_training flag: env parsing, and that on
a real fwd+bwd training graph the flag reorders a node without changing
grads/params."""

import os
import subprocess
import sys
from unittest.mock import patch as mock_patch

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from torch._inductor.fx_passes import post_grad
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
    xfailIfNoAcceleratorTriton,
)


def _flag_in_subprocess(env_value):
    # [FUNCTION DOC] HELPER: Starts a fresh Python interpreter with the locality-training
    # environment variable removed (when env_value is None) or set to the given string, imports
    # the inductor config, and returns the printed integer value of the flag as a stripped
    # string. Raises CalledProcessError if the child fails.
    # Parse-only: a fresh interpreter imports just the config module and prints
    # the resolved flag. env_value=None means the var is unset (ship default).
    env = os.environ.copy()
    if env_value is None:
        env.pop("TORCHINDUCTOR_REORDER_LOCALITY_TRAINING", None)
    else:
        env["TORCHINDUCTOR_REORDER_LOCALITY_TRAINING"] = env_value
    out = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import torch._inductor.config as c;"
            "print(int(c.reorder_for_locality_in_training))",
        ],
        env=env,
    )
    return out.decode().strip()


class _Recorder:
    # [CLASS DOC] HELPER: Callable stand-in for the reorder_for_locality pass. It keeps the
    # original function, forwards each graph to it, and records the number of calls (calls) and
    # whether any call changed the order of node names (moved).
    """Wraps ``reorder_for_locality`` and records, per call, whether the pass
    changed the graph node order. ``orig`` is captured before patching so the
    real pass still runs."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Saves the real post_grad reorder function and zeroes the call
        # counter and the moved flag.
        self.orig = post_grad.reorder_for_locality
        self.calls = 0
        self.moved = False

    def __call__(self, graph):
        # [FUNCTION DOC] HELPER: Runs the real pass on the graph (mutating it), compares node-
        # name order before and after, increments the call count, and sets moved when the order
        # differs.
        before = [n.name for n in graph.nodes]
        self.orig(graph)
        after = [n.name for n in graph.nodes]
        self.calls += 1
        if before != after:
            self.moved = True


class _TwoBranch(torch.nn.Module):
    # [CLASS DOC] HELPER: Small module with two independent matmul branches (tanh and sigmoid)
    # of 16 by 16 parameters that are combined and summed per row; designed so that the reorder
    # pass has real moves to make after functionalization.
    # Two independent matmul branches that only meet at the end. After
    # functionalization the branch producers land far from their sole
    # consumers, so reorder_for_locality has real moves to make.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Creates the two random 16 by 16 weight parameters.
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.randn(16, 16))
        self.w2 = torch.nn.Parameter(torch.randn(16, 16))

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Computes both branch activations from the input and returns the
        # row sum of a combination of their product and trig terms.
        a = torch.tanh(x @ self.w1)
        b = torch.sigmoid(x @ self.w2)
        return (a * b + torch.sin(a) * torch.cos(b)).sum(dim=1)


def _train_once(device, flag_on, reorder_on=True):
    # [FUNCTION DOC] HELPER: Compiles and trains the two-branch model for one step with fixed
    # seed. Resets dynamo, builds the model and input on the device, patches inductor config
    # (reorder option as given, training flag as given, caches disabled) and patches the pass
    # with a recorder, then runs compiled forward and backward. Returns parameters after one SGD
    # step, the gradients, and the recorder.
    rec = _Recorder()
    torch._dynamo.reset()
    torch.manual_seed(1234)
    model = _TwoBranch().to(device)
    x = torch.randn(8, 16, device=device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with inductor_config.patch(
        {
            "reorder_for_locality": reorder_on,
            "reorder_for_locality_in_training": flag_on,
            "fx_graph_cache": False,
            "force_disable_caches": True,
        }
    ):
        with mock_patch.object(post_grad, "reorder_for_locality", rec):
            compiled = torch.compile(model, backend="inductor", fullgraph=True)
            compiled(x).sum().backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    opt.step()
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    return params, grads, rec


class TestReorderForLocalityInTrainingEnv(TestCase):
    # [CLASS DOC] Tests how the training flag's default is parsed from the environment and that
    # the inference path is unaffected by it.
    def test_default_off(self):
        # [FUNCTION DOC] Checks the flag defaults to zero when the environment variable is
        # unset.
        self.assertEqual(_flag_in_subprocess(None), "0")

    def test_env_one_turns_on(self):
        # [FUNCTION DOC] Checks the flag resolves to one when the environment variable is set to
        # one.
        self.assertEqual(_flag_in_subprocess("1"), "1")

    def test_env_zero_keeps_off(self):
        # [FUNCTION DOC] Checks the flag stays zero when the environment variable is set to
        # zero.
        self.assertEqual(_flag_in_subprocess("0"), "0")

    def test_inference_gate_unchanged(self):
        # [FUNCTION DOC] Builds a tiny FX graph, runs post grad passes for inference with the
        # training flag off and the pass patched by a recorder, and asserts the pass was called
        # at least once.
        # Gate-only check (invocation, not a feature exercise): inference graphs
        # must still run the pass regardless of the training flag. Pure FX, no
        # device execution, so it stays out of the device-generic class below.
        rec = _Recorder()
        with inductor_config.patch({"reorder_for_locality_in_training": False}):
            with mock_patch.object(post_grad, "reorder_for_locality", rec):
                g = torch.fx.Graph()
                xn = g.placeholder("x")
                g.output(g.call_function(torch.relu, args=(xn,)))
                gm = torch.fx.GraphModule(torch.nn.Module(), g)
                post_grad.post_grad_passes(gm, is_inference=True)
        self.assertGreater(rec.calls, 0)


class TestReorderForLocalityInTraining(TestCase):
    # [CLASS DOC] Device-generic tests (instantiated per device) of the training flag's effect
    # on a compiled training graph. Test methods take a device argument and are expected to fail
    # when no accelerator with Triton is available.
    @xfailIfNoAcceleratorTriton
    def test_training_flag_reorders_and_preserves_semantics(self, device):
        # [FUNCTION DOC] Trains once with the flag off and once on, then asserts equal gradient
        # keys and equal gradients and updated parameters, that the pass was never called with
        # the flag off, and that with it on the pass was called and moved at least one node.
        p_off, g_off, rec_off = _train_once(device, flag_on=False)
        p_on, g_on, rec_on = _train_once(device, flag_on=True)

        # (a) semantics preserved: same seed/init/input, so grads and post-step
        # params match between flag-off and flag-on. reorder_for_locality is a
        # mathematical (not bitwise) equivalence -- it can change fusion grouping
        # and thus low-bit FP accumulation -- so use default tolerances.
        self.assertEqual(set(g_off), set(g_on))
        for k in g_off:
            self.assertEqual(g_on[k], g_off[k])
            self.assertEqual(p_on[k], p_off[k])

        # (b) the flag actually exercised the pass: with it on the reorder ran
        # and moved at least one node on a training graph; with it off the pass
        # never ran on training. The moved check is what gives this teeth, a
        # no-op graph would leave moved False and fail here.
        self.assertEqual(rec_off.calls, 0)
        self.assertGreater(rec_on.calls, 0)
        self.assertTrue(
            rec_on.moved,
            "flag on must reorder at least one node on the training graph",
        )

    @xfailIfNoAcceleratorTriton
    def test_training_flag_noop_when_reorder_off(self, device):
        # [FUNCTION DOC] With the training flag on but the base reorder option off, trains once
        # and asserts the pass was never called.
        # The training flag is gated by reorder_for_locality: enabling it while
        # reorder_for_locality is off must not run the pass on a training graph.
        _, _, rec = _train_once(device, flag_on=True, reorder_on=False)
        self.assertEqual(rec.calls, 0)


instantiate_device_type_tests(TestReorderForLocalityInTraining, globals())


if __name__ == "__main__":
    run_tests()
