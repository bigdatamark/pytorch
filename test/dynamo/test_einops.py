# Owner(s): ["module: dynamo"]
# [MODULE DOC]
# Tests that torch.compile (Dynamo, mostly with fullgraph) can trace the einops library
# functions (rearrange, reduce, repeat, pack, unpack, einsum) and layers (Rearrange, Reduce,
# EinMix). The whole TestEinops class is skipped when einops is not installed.
# At import time it detects the einops version and sets HAS_EINOPS, HAS_EINOPS_PACK and
# EINOPS_SUPPORTS_DYNAMO_TRACING (true when the version is at least 0.8.2; an unparsable
# version string counts as unsupported). Parametrized tests are instantiated at module level.
import importlib
import os
import subprocess
import sys
import unittest
import warnings

import torch
from torch import nn
from torch._dynamo.test_case import TestCase
from torch._vendor.packaging.version import InvalidVersion, Version
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


HAS_EINOPS = importlib.util.find_spec("einops")

if HAS_EINOPS:
    import einops

    einops_version = einops.__version__
else:
    einops_version = "none"
einops_version_sanitized = einops_version.replace(".", "_")
HAS_EINOPS_PACK = HAS_EINOPS and hasattr(einops, "pack")
try:
    EINOPS_SUPPORTS_DYNAMO_TRACING = HAS_EINOPS and Version(einops_version) >= Version(
        "0.8.2"
    )
except InvalidVersion:
    EINOPS_SUPPORTS_DYNAMO_TRACING = False


@unittest.skipIf(not HAS_EINOPS, "these tests require einops")
class TestEinops(TestCase):
    # [CLASS DOC]
    # Dynamo test case exercising compiled einops functions and layers against eager results,
    # plus subprocess-based checks of how einops calls appear in captured graphs. Skipped
    # entirely if einops is missing. Holds no instance state; tests are parametrized on the
    # sanitized einops version string so each version gets a distinct test name.
    """
    These tests adapted from similar tests in the einops repo.
    https://github.com/arogozhnikov/einops/blob/main/einops/tests/test_other.py#L254

    The goal of this test suite is to test torch.compile x einops for multiple
    versions of einops. Our goal is to prevent regressions in einops from changes
    in PyTorch.
    """

    @parametrize("version", [einops_version_sanitized])
    def test_functions(self, version):
        # [FUNCTION DOC]
        # Builds a module that chains repeat, reduce, optional pack/unpack (only when einops
        # has them, else a plain rearrange), another rearrange and an einsum, with axis names
        # altered by a suffix argument. Compiles it with inductor and fullgraph, then for
        # several tensor sizes and suffixes compares against the eager module run in double
        # precision and cast back to float. Fails if any result differs.
        from einops import einsum, rearrange, reduce, repeat

        has_pack = HAS_EINOPS_PACK
        if has_pack:
            from einops import pack, unpack

        class TorchModuleWithOperations(nn.Module):
            # [CLASS DOC] HELPER: Parameterless module whose forward runs a fixed pipeline
            # of einops operations on a 3-D input; used as the compile target.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only initializes the base module; no state.
                super().__init__()

            def forward(self, x_abc, suffix=""):
                # [FUNCTION DOC] HELPER: Takes a 3-D tensor and a string suffix used to
                # rename pattern axes. Repeats along a new axis of size 4, reduces it back
                # with min, packs/unpacks copies (or rearranges when pack is missing),
                # rearranges back to three dims and adds an einsum-derived vector.
                # Returns a tensor shaped like the input.
                a, b, c = x_abc.shape

                def suf(pattern):
                    # [FUNCTION DOC] HELPER: Splits the pattern on whitespace and appends
                    # the enclosing suffix to every token whose last character is a, c or
                    # d; returns the tokens rejoined with single spaces.
                    parts = pattern.split()
                    return " ".join(
                        [p if p[-1] not in "acd" else p + suffix for p in parts]
                    )

                # patterns look a bit strange because names a, c, d will be modified on every run
                # by suf function
                x_abcd = repeat(x_abc, suf("a b c -> a b c 4"))
                x_abc = reduce(x_abcd, suf("a b c d -> a b c"), "min")
                if has_pack:
                    x_abdc, ps = pack([x_abc] * (2 + len(suffix)), suf("a b * c"))
                    x_array = unpack(
                        rearrange(x_abdc, suf("a b d c -> (a b ) 1 c d")),
                        ps,
                        "ab one1 c *",
                    )
                    x1 = x_array[0] + len(x_array)
                else:
                    x1 = rearrange(x_abc, suf("a b c -> (a b ) 1 c"))
                x1 = rearrange(x1, suf("(a b ) 1 c -> a b c"), b=b)
                addition = einsum(x_abc, x_abcd, suf("a b c , a b c d -> d"))[0]
                return x1 + addition

        original = TorchModuleWithOperations()
        # Einops only interacts with Dynamo but we test backend="inductor" just in case
        compiled = torch.compile(original, backend="inductor", fullgraph=True)
        for size in [10, 20, 40]:
            x = torch.rand([size, size + 1, size + 2])
            for suffix in ["", "suf1", "other_suffix"]:
                result1 = compiled(x, suffix)
                result2 = original(x.double(), suffix).float()
                self.assertEqual(result1, result2)

    @parametrize("version", [einops_version_sanitized])
    def test_layers(self, version):
        # [FUNCTION DOC]
        # Builds a sequential model of the einops layers Rearrange, EinMix and Reduce,
        # compiles it with inductor and fullgraph, and checks that for several square input
        # sizes the compiled output equals the eager output.
        from einops.layers.torch import EinMix, Rearrange, Reduce

        original = nn.Sequential(
            Rearrange("b (t c) -> b t c", c=16),
            EinMix(
                "b t c -> qkv b t cout",
                weight_shape="qkv c cout",
                bias_shape="qkv cout",
                qkv=3,
                c=16,
                cout=8,
            ),
            Reduce("qkv b t cout -> b t qkv", "min", cout=8),
        )

        # Einops only interacts with Dynamo but we test backend="inductor" just in case
        compiled = torch.compile(original, backend="inductor", fullgraph=True)

        for size in [16, 32, 64]:
            x = torch.rand([size, size])
            result1 = original(x)
            result2 = compiled(x)
            self.assertEqual(result1, result2)

    @parametrize("version", [einops_version_sanitized])
    def test_no_recompile_on_lazy_state(self, version):
        # [FUNCTION DOC]
        # Runs a script in a fresh Python subprocess that compiles a module using all the
        # einops functions with the eager backend, calls it once, then calls it again under
        # the fail_on_recompile stance. A recompile caused by einops lazy initialization
        # makes the subprocess exit nonzero, which raises CalledProcessError here.
        """einops has some lazy state that gets initialized the first time an API
        is called. This should not trigger a recompile."""
        script = """\
import torch
import torch.nn as nn
import einops
from einops import einsum, reduce, repeat, rearrange

has_pack = hasattr(einops, "pack")
if has_pack:
    from einops import pack, unpack

class TorchModuleWithOperations(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_abc, suffix=""):
        a, b, c = x_abc.shape

        def suf(pattern):
            parts = pattern.split()
            return " ".join([p if p[-1] not in "acd" else p + suffix for p in parts])

        # patterns look a bit strange because names a, c, d will be modified on every run
        # by suf function
        x_abcd = repeat(x_abc, suf("a b c -> a b c 4"))
        x_abc = reduce(x_abcd, suf("a b c d -> a b c"), "min")
        if has_pack:
            x_abdc, ps = pack([x_abc] * (2 + len(suffix)), suf("a b * c"))
            x_array = unpack(rearrange(x_abdc, suf("a b d c -> (a b ) 1 c d")), ps, "ab one1 c *")
            x1 = x_array[0] + len(x_array)
        else:
            x1 = rearrange(x_abc, suf("a b c -> (a b ) 1 c"))
        x1 = rearrange(x1, suf("(a b ) 1 c -> a b c"), b=b)
        addition = einsum(x_abc, x_abcd, suf("a b c , a b c d -> d"))[0]
        return x1 + addition

compiled_fn = torch.compile(TorchModuleWithOperations(), fullgraph=True, backend="eager")
x = torch.arange(2 * 3 * 5).view(2, 3, 5)
y = compiled_fn(x)

# Should not recompile!
with torch.compiler.set_stance("fail_on_recompile"):
    z = compiled_fn(x)
"""
        subprocess.check_output([sys.executable, "-c", script])

    def _run_in_subprocess(self, flag, method, einops_method, snippet):
        # [FUNCTION DOC] HELPER: Runs a generated script in a subprocess (cwd is this test's
        # directory) that defines a function from the given code snippet ending in a sin,
        # compiles it with fullgraph and a graph-recording backend, asserts the result
        # matches eager and that exactly one graph was captured, then prints the normalized
        # graph. The parent fails the test with the subprocess output if it errors. When flag
        # is true the einops op name must be absent from the printed graph (traced through);
        # otherwise it must be present. The method argument is used only in the error text.
        # run in a different process
        script = f"""
import torch
from torch._dynamo.testing import EagerAndRecordGraphs, normalize_gm

import einops

backend = EagerAndRecordGraphs()

def f(x):
    {snippet}
    return y.sin()

x = torch.randn(3, 4, 5)
expected = f(x)
got = torch.compile(f, backend=backend, fullgraph=True)(x)

assert torch.allclose(expected, got)
assert len(backend.graphs) == 1, len(backend.graphs)
graph = backend.graphs[0]
print(normalize_gm(graph.print_readable(print_output=False)))
"""
        script = script.strip()
        try:
            output = (
                subprocess.check_output(
                    [sys.executable, "-c", script],
                    stderr=subprocess.STDOUT,
                    cwd=os.path.dirname(os.path.realpath(__file__)),
                )
                .decode("utf-8")
                .strip()
            )
        except subprocess.CalledProcessError as e:
            self.fail(
                msg=(f"Subprocess exception {method}:\n" + e.output.decode("utf-8"))
            )
        else:
            if flag:
                self.assertNotIn(einops_method, output)
            else:
                self.assertIn(einops_method, output)

    @parametrize(
        "method",
        ["reduce", "repeat", "pack", "unpack", "einsum", "rearrange"],
        name_fn=lambda f: f,
    )
    def test_einops_method(self, method):
        # [FUNCTION DOC]
        # For one einops function name, skips if einops lacks it, picks a small snippet
        # exercising it and the expected op-name marker (einops_einops_ prefix, or
        # einops_packing_ for pack and unpack), then delegates to the subprocess helper with
        # the flag telling whether this einops version is directly traceable by Dynamo.
        # Any unexpected method name fails the test.
        flag = EINOPS_SUPPORTS_DYNAMO_TRACING
        if not hasattr(einops, method):
            self.skipTest(f"Needs einops.{method}")

        if method == "reduce":
            einops_method = f"einops_einops_{method}"
            snippet = "y = einops.reduce(x, 'a b c -> a b', 'min')"
        elif method == "repeat":
            einops_method = f"einops_einops_{method}"
            snippet = "y = einops.repeat(x, 'a b c -> a b c d', d=2)"
        elif method == "rearrange":
            einops_method = f"einops_einops_{method}"
            snippet = "y = einops.rearrange(x, 'a b c -> a c b')"
        elif method == "einsum":
            einops_method = f"einops_einops_{method}"
            snippet = "y = einops.einsum(x, 'a b c -> a c b')"
        elif method == "pack":
            einops_method = f"einops_packing_{method}"
            snippet = "y, meta = einops.pack([x], '* b')"
        elif method == "unpack":
            einops_method = f"einops_packing_{method}"
            snippet = "x_packed, meta = einops.pack([x], '* b'); y = einops.unpack(x_packed, meta, '* b')[0]"
        else:
            self.fail(method)
        self._run_in_subprocess(flag, method, einops_method, snippet)

    @parametrize(
        "method",
        ["rearrange", "pack", "unpack", "einsum"],
        name_fn=lambda f: f,
    )
    def test_no_warning(self, method):
        # [FUNCTION DOC]
        # Compiles a function calling the chosen einops function (rearrange, pack, unpack or
        # einsum) with the eager backend and fullgraph, runs it once while recording all
        # warnings, and asserts that none mention functools.lru_cache. Skips pack and unpack
        # when einops lacks pack support.
        if method in ("pack", "unpack") and not HAS_EINOPS_PACK:
            self.skipTest(f"Needs einops.{method}")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; applies the einops function selected by
            # the enclosing method name to x (flatten, pack, pack then unpack, or transpose
            # via einsum) and returns the result. Fails the test for an unknown name.
            if method == "rearrange":
                return einops.rearrange(x, "... -> (...)")
            if method == "pack":
                y, _ = einops.pack([x], "*")
                return y
            if method == "unpack":
                x_packed, meta = einops.pack([x], "*")
                return einops.unpack(x_packed, meta, "*")[0]
            if method == "einsum":
                return einops.einsum(x, "a b -> b a")
            self.fail(method)

        x = torch.randn(2, 3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fn(x)

        lru_cache_warnings = [
            warning
            for warning in caught
            if "functools.lru_cache" in str(warning.message)
        ]
        self.assertEqual(lru_cache_warnings, [])

    @unittest.skipUnless(
        EINOPS_SUPPORTS_DYNAMO_TRACING, "Needs directly traceable einops"
    )
    def test_lazy_init_preserves_rng_state(self):
        # [FUNCTION DOC]
        # Seeds the global RNG, snapshots its state, calls the private Dynamo hook that
        # registers einops functions as allowed in graph, and asserts the RNG state is
        # unchanged. Only runs when the installed einops is directly traceable.
        torch.manual_seed(1234)
        before = torch.random.get_rng_state().clone()
        torch._dynamo.decorators._allow_in_graph_einops()
        self.assertEqual(torch.random.get_rng_state(), before)


instantiate_parametrized_tests(
    TestEinops,
)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
