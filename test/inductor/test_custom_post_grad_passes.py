# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for user supplied custom graph passes in inductor config: joint-graph pre
# and post passes, post-grad pre and post passes, pre-grad passes (including lists and tuples run
# in order), pattern-matcher based passes registering an mkldnn conv-relu fusion, backend
# specific graph module passes, and inference-aware passes. Defines a shared base test case with
# freezing enabled, simple recording and cos-to-sin passes. The tests run only on Linux with a
# CPU and mkldnn available.
import contextlib
import operator
from collections import defaultdict

import torch
import torch._inductor.pattern_matcher as pattern_matcher
import torch.fx as fx
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.codegen.common import get_custom_backend_pass_for_device
from torch._inductor.custom_graph_pass import (
    CustomGraphModulePass,
    CustomGraphPass,
    CustomInferenceAwareGraphPass,
    get_hash_for_files,
)
from torch._inductor.lowering import lowerings as L
from torch._inductor.pattern_matcher import Arg, CallFunction, PatternMatcherPass
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CPU, patch_inductor_backend


@config.patch({"freezing": True})
class TestCustomPassBase(TestCase):
    # [CLASS DOC] HELPER: Base test case (config freezing enabled for all tests) offering
    # helpers to clone inputs and to compare compiled against eager output while checking
    # pattern matcher counters.
    def _clone_inputs(self, inputs):
        # [FUNCTION DOC] HELPER: Returns a tuple with a copy of each tensor input; non-tensor
        # inputs are passed through unchanged.
        def clone(x):
            # [FUNCTION DOC] HELPER: Returns a clone of a tensor, or the value itself if
            # not a tensor.
            if not isinstance(x, torch.Tensor):
                return x
            return x.clone()

        return tuple(clone(x) for x in inputs)

    def _test_common(
        self,
        mod,
        inputs,
        matcher_count,
        matcher_nodes,
        atol=1e-5,
        rtol=1.3e-6,
    ):
        # [FUNCTION DOC] HELPER: Clears inductor counters, then under no_grad runs the module
        # eagerly and compiled on cloned inputs, asserts the outputs are close within the
        # given absolute and relative tolerances, and asserts the pattern matcher match count
        # and matched node count counters equal the expected numbers (raising an assertion
        # failure on mismatch).
        counters.clear()
        maybe_autocast = contextlib.nullcontext()
        with torch.no_grad(), maybe_autocast:
            clone_inputs = self._clone_inputs(inputs)
            expected = mod(*inputs)
            actual = torch.compile(mod)(*clone_inputs)
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            self.assertEqual(
                counters["inductor"]["pattern_matcher_count"], matcher_count
            )
            self.assertEqual(
                counters["inductor"]["pattern_matcher_nodes"],
                matcher_nodes,
            )


aten = torch.ops.aten
mkldnn = torch.ops.mkldnn


def change_cos_pass(graph):
    # [FUNCTION DOC] HELPER: Graph pass that walks all nodes and, for every function call node
    # whose target is the aten cosine, retargets it in place to the aten sine.
    for node in graph.nodes:
        if node.op == "call_function" and node.target == aten.cos.default:
            node.target = aten.sin.default


class ChangeCosCustomPass(CustomGraphPass):
    # [CLASS DOC] HELPER: Custom graph pass that rewrites cosine calls into sine calls by
    # applying the cos-to-sin graph function; its uuid hashes this test file.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Calls the base initializer only.
        super().__init__()

    def __call__(self, g: torch.fx.graph.Graph):
        # [FUNCTION DOC] HELPER: Applies the cos-to-sin rewrite to the given graph in place.
        change_cos_pass(g)

    def uuid(self) -> bytes:
        # [FUNCTION DOC] HELPER: Returns a hash of this file's contents as the pass identifier.
        return get_hash_for_files((__file__,))


class RecordCustomPass(CustomGraphPass):
    # [CLASS DOC] HELPER: Custom graph pass that does not change the graph; each call appends its
    # name to a shared list so tests can check the order passes run in. Stores the name and the
    # shared calls list; the uuid is the name.
    def __init__(self, name, calls) -> None:
        # [FUNCTION DOC] HELPER: Stores the pass name and the list that call records
        # are appended to.
        super().__init__()
        self.name = name
        self.calls = calls

    def __call__(self, g: torch.fx.graph.Graph):
        # [FUNCTION DOC] HELPER: Appends the pass name to the shared calls list; the
        # graph is untouched.
        self.calls.append(self.name)

    def uuid(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the pass name as its identifier.
        return self.name


class TestPostGradCustomPrePostPass(TestCustomPassBase):
    # [CLASS DOC] Test case for custom pre and post passes at the joint-graph, post-grad and pre-
    # grad stages. Contains helpers (a pattern-matcher backed custom pass class, a small conv
    # plus relu model and a function that registers the mkldnn conv-relu fusion pattern).
    #  mkldnn fusion's pattern_matcher
    # (torch/_inductor/fx_passes/mkldnn_fusion.py),
    # and apply it to custom post_grad_passes.
    def _register_mkldnn_conv_relu_fusion(self, custom_pass_dict):
        # [FUNCTION DOC] HELPER: Registers into the given custom pass mapping a lowering pattern
        # that fuses an mkldnn pointwise convolution followed by a relu into a single convolution
        # call. It builds the pattern, defines the registration utilities and registers them;
        # mutates the passed mapping.
        # pattern
        def _mkldnn_conv_relu_pattern():
            # [FUNCTION DOC] HELPER: Returns the pattern for relu applied to a single-user mkldnn
            # pointwise convolution call that has ten arbitrary arguments.
            return CallFunction(
                aten.relu,
                CallFunction(
                    mkldnn._convolution_pointwise.default,
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    _users=1,
                ),
            )

        # utils of pattern matcher registration
        def _register_fusion_lowering(pattern, custom_pass_dict):
            # [FUNCTION DOC] HELPER: Registers the replacement for the given pattern
            # into the custom pass mapping using an always-true check, and returns
            # the replacement function.
            def dummy_check(m):
                # [FUNCTION DOC] HELPER: Extra pattern check that accepts every match.
                return True

            def register_custom_lowering_pattern(
                pattern, extra_check, custom_pass_dict
            ):
                # [FUNCTION DOC] HELPER: Forwards to the pattern matcher lowering pattern
                # registration using the given custom pass mapping as the pass dictionary;
                # returns the registration decorator.
                return pattern_matcher.register_lowering_pattern(
                    pattern, extra_check, pass_dict=custom_pass_dict
                )

            @register_custom_lowering_pattern(pattern, dummy_check, custom_pass_dict)
            def fn(match, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Replacement: drops the last three arguments of the
                # matched convolution and appends the relu attribute with empty scalars and
                # algorithm, then calls the mkldnn pointwise convolution lowering, producing a
                # fused convolution with relu.
                computation_args = list(args)[:-3] + ["relu", [], ""]
                return L[mkldnn._convolution_pointwise.default](*computation_args)

            return fn

        _register_fusion_lowering(_mkldnn_conv_relu_pattern(), custom_pass_dict)

    # custom post grad pass
    class _CustomPass(PatternMatcherPass, CustomGraphPass):
        # [CLASS DOC] HELPER: Custom post-grad pass that is both a pattern matcher pass and a
        # custom graph pass; calling it applies the registered patterns to the graph. Its uuid
        # hashes this file.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Calls the base initializer only.
            super().__init__()

        def __call__(self, g: torch.fx.graph.Graph):
            # [FUNCTION DOC] HELPER: Applies the registered patterns to the graph.
            self.apply(g)

        def uuid(self) -> bytes:
            # [FUNCTION DOC] HELPER: Returns a hash of this file's contents as
            # the pass identifier.
            return get_hash_for_files((__file__,))

    # case model
    class _ConvReLU(torch.nn.Module):
        # [CLASS DOC] HELPER: Test model that is a three by three convolution with padding one
        # followed by relu.
        def __init__(self, ic, oc):
            # [FUNCTION DOC] HELPER: Creates the convolution layer with the given input
            # and output channels.
            super().__init__()
            self.conv = torch.nn.Conv2d(ic, oc, kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Applies the convolution and returns its relu.
            x1 = self.conv(x)
            return x1.relu()

    def test_custom_joint_pass_pre(self):
        # [FUNCTION DOC] Sets the cos-to-sin pass as the joint-graph custom pre pass and checks
        # that the compiled triple-cosine function equals the triple-sine function.
        with config.patch(joint_custom_pre_pass=ChangeCosCustomPass()):

            def g(x):
                # [FUNCTION DOC] HELPER: Reference function applying sine three times.
                return x.sin().sin().sin()

            def f(x):
                # [FUNCTION DOC] HELPER: Function under test applying cosine three times.
                return x.cos().cos().cos()

            x = torch.randn(8, dtype=torch.float32)
            torch.testing.assert_close(torch.compile(f)(x), g(x))

    def test_custom_joint_pass_post(self):
        # [FUNCTION DOC] Same as the joint pre pass test but configures the cos-to-sin pass as
        # the joint-graph custom post pass.
        with config.patch(joint_custom_post_pass=ChangeCosCustomPass()):

            def g(x):
                # [FUNCTION DOC] HELPER: Reference function applying sine three times.
                return x.sin().sin().sin()

            def f(x):
                # [FUNCTION DOC] HELPER: Function under test applying cosine three times.
                return x.cos().cos().cos()

            x = torch.randn(8, dtype=torch.float32)
            torch.testing.assert_close(torch.compile(f)(x), g(x))

    def test_custom_pass_list_runs_in_order(self):
        # [FUNCTION DOC] For each of the five custom pass config fields (post-grad pre and post,
        # joint pre and post, pre-grad), configures a list of two recording passes and checks
        # they ran in list order; then configures a tuple of two passes for the post-grad post
        # field and checks the same order. Resets dynamo between compiles.
        def f(x):
            # [FUNCTION DOC] HELPER: Function under test: sine, add one, relu.
            return (x.sin() + 1).relu()

        x = torch.randn(8, dtype=torch.float32)
        for field in (
            "post_grad_custom_pre_pass",
            "post_grad_custom_post_pass",
            "joint_custom_pre_pass",
            "joint_custom_post_pass",
            "pre_grad_custom_pass",
        ):
            with self.subTest(field=field):
                calls = []
                torch._dynamo.reset()
                with config.patch(
                    {
                        field: [
                            RecordCustomPass(f"{field}_first", calls),
                            RecordCustomPass(f"{field}_second", calls),
                        ]
                    }
                ):
                    torch.compile(f)(x)
                self.assertEqual(calls, [f"{field}_first", f"{field}_second"])

        calls = []
        torch._dynamo.reset()
        with config.patch(
            {
                "post_grad_custom_post_pass": (
                    RecordCustomPass("tuple_first", calls),
                    RecordCustomPass("tuple_second", calls),
                )
            }
        ):
            torch.compile(f)(x)
        self.assertEqual(calls, ["tuple_first", "tuple_second"])

    def test_custom_pre_pass(self):
        # [FUNCTION DOC] Disables the default pattern matcher, installs the pattern-matcher
        # custom pass as the post-grad pre pass, registers the conv-relu fusion into it, then
        # compiles a conv-relu module on a one by sixteen by 56 by 56 input and checks the
        # output and that one fusion match (two nodes) plus one weight prepack match (one
        # node) were counted.
        with config.patch(
            # leave custom pass only in post_grad_passes()
            pattern_matcher=False,
            post_grad_custom_pre_pass=self._CustomPass(),
            # define pattern match as custom post grad opt pass
            post_grad_custom_post_pass=None,
        ):
            # init mkldnn fusion on custom_matcher
            self._register_mkldnn_conv_relu_fusion(config.post_grad_custom_pre_pass)

            mod = self._ConvReLU(16, 16).eval()
            x = torch.randn((1, 16, 56, 56), dtype=torch.float32)

            match_count = 1
            match_nodes = 2
            other_match_count = 1  # conv prepack weight
            other_match_nodes = 1  # conv prepack weight
            self._test_common(
                mod,
                (x,),
                match_count + other_match_count,
                match_nodes + other_match_nodes,
            )

    def test_custom_post_pass(self):
        # [FUNCTION DOC] Same as the pre pass test but installs the pattern-matcher custom
        # pass as the post-grad post pass (with the pre pass cleared) and registers the
        # fusion into it.
        with config.patch(
            # leave custom pass only in post_grad_passes()
            pattern_matcher=False,
            # define pattern match as custom post grad opt pass
            post_grad_custom_pre_pass=None,
            post_grad_custom_post_pass=self._CustomPass(),
        ):
            # init mkldnn fusion on custom_matcher
            self._register_mkldnn_conv_relu_fusion(config.post_grad_custom_post_pass)

            mod = self._ConvReLU(16, 16).eval()
            x = torch.randn((1, 16, 56, 56), dtype=torch.float32)

            match_count = 1
            match_nodes = 2
            other_match_count = 1  # conv prepack weight
            other_match_nodes = 1  # conv prepack weight
            self._test_common(
                mod,
                (x,),
                match_count + other_match_count,
                match_nodes + other_match_nodes,
            )

    def test_custom_pre_grad_pass(self):
        # [FUNCTION DOC] Tests a custom pre-grad pass, installed with early timing, that merges
        # matrix multiplies sharing the same right operand into one multiply over concatenated
        # inputs followed by a split. Compiles a function multiplying four sequences by a shared
        # weight and checks the pass saw the graph and exactly one matrix multiply remains.
        saved_graph = [None]

        def merge_mm_shared_rhs(graph: fx.Graph):
            # [FUNCTION DOC] HELPER: Proof-of-concept pre-grad pass (documented as unsafe, it
            # ignores dependence between the left operands). Saves the graph, groups matrix
            # multiplies by right operand, and for each group with more than one member inserts a
            # concatenation of the left operands, a single multiply and a split sized from
            # example value metadata before the first one, then retargets each original multiply
            # to a getitem of the split result.
            """
            Bad POC of merging mm with a shared RHS.
            i.e. [mm(x, W), mm(x2, W)] => mm(cat(x, x2), W).split()

            Isn't actually safe for a couple reasons. For example, it doesn't handle the
            case where the LHS inputs depend on each other
            """
            saved_graph[0] = graph
            matmuls = [n for n in graph.nodes if n.target == torch.mm]
            rhs_vals = defaultdict(set)
            for m in matmuls:
                rhs_vals[m.args[1]].add(m)

            order = {n: idx for idx, n in enumerate(graph.nodes)}

            for rhs, matmuls in rhs_vals.items():
                if len(matmuls) == 1:
                    continue
                matmuls = sorted(matmuls, key=lambda x: order[x])
                with graph.inserting_before(matmuls[0]):
                    lhs_vals = [m.args[0] for m in matmuls]
                    new_cat = graph.create_node(
                        "call_function", torch.cat, args=(lhs_vals, 0)
                    )
                    new_mm = graph.create_node(
                        "call_function", torch.mm, args=(new_cat, rhs)
                    )
                    split_vals = graph.create_node(
                        "call_function",
                        torch.split,
                        args=(
                            new_mm,
                            [l.meta["example_value"].shape[0] for l in lhs_vals],
                        ),
                    )
                for idx, m in enumerate(matmuls):
                    m.target = operator.getitem
                    m.args = (split_vals, idx)

        @config.patch(
            pre_grad_custom_pass=merge_mm_shared_rhs, pre_grad_pass_timing="early"
        )
        def inner_test():
            # [FUNCTION DOC] HELPER: Compiles a function multiplying each of four bfloat16
            # sequences by one weight under the patched config, runs it, and raises
            # AssertionError if no graph was saved or the saved graph does not contain
            # exactly one multiply.
            @torch.compile
            def f(W, nested_seqs):
                # [FUNCTION DOC] HELPER: Returns the list of products of each sequence
                # with the weight.
                outs = [torch.mm(s, W) for s in nested_seqs]
                return outs

            W = torch.randn(16, 16, dtype=torch.bfloat16)
            nested_seqs = [
                torch.randn(l, 16, dtype=torch.bfloat16) for l in [4, 8, 5, 3]
            ]

            f(W, nested_seqs)
            if saved_graph[0] is None:
                raise AssertionError("saved_graph[0] is None")
            matmuls = [n for n in saved_graph[0].nodes if n.target == torch.mm]
            if len(matmuls) != 1:
                raise AssertionError(f"Expected 1 matmul, got {len(matmuls)}")

        inner_test()

    def test_custom_backend_pass(self):
        # [FUNCTION DOC] Defines a custom graph-module pass that optionally runs an existing
        # backend pass, then rewrites cosine to sine, patches the CPU inductor backend to
        # use it (wrapping the existing CPU backend pass) and checks the compiled triple-
        # cosine equals triple-sine.
        class CustomBackendPass(CustomGraphModulePass):
            # [CLASS DOC] HELPER: Graph module pass that runs an optional existing pass and then
            # applies the cos-to-sin rewrite to the module's graph.
            def __init__(self, existing_pass: CustomGraphModulePass = None):
                # [FUNCTION DOC] HELPER: Stores the optional existing pass.
                super().__init__()
                self.existing_pass = existing_pass

            def __call__(self, gm: fx.GraphModule) -> None:
                # [FUNCTION DOC] HELPER: Runs the existing pass on the graph module if one was
                # given, then rewrites cosine to sine in the module's graph. Returns nothing.
                if self.existing_pass:
                    self.existing_pass(gm)

                change_cos_pass(gm.graph)

            def uuid(self) -> bytes:
                # [FUNCTION DOC] HELPER: Returns a hash of this file's contents as
                # the pass identifier.
                return get_hash_for_files((__file__,))

        custom_backend_pass = CustomBackendPass(
            get_custom_backend_pass_for_device("cpu")
        )
        with patch_inductor_backend("cpu", custom_pass=custom_backend_pass):

            def g(x):
                # [FUNCTION DOC] HELPER: Reference function applying sine three times.
                return x.sin().sin().sin()

            def f(x):
                # [FUNCTION DOC] HELPER: Function under test applying cosine three times.
                return x.cos().cos().cos()

            x = torch.randn(8, dtype=torch.float32)
            torch.testing.assert_close(torch.compile(f)(x), g(x))

    def test_custom_pass_inference_flag(self):
        # [FUNCTION DOC] Installs an inference-aware custom pass as the post-grad post pass and
        # checks it receives the inference flag: true when compiling with an input that does not
        # require grad, false after the input requires grad and the function is compiled again.
        class CustomPass(CustomInferenceAwareGraphPass):
            # [CLASS DOC] HELPER: Inference-aware pass that records the inference flag passed to
            # its most recent call (initially false) and leaves the graph unchanged.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the recorded inference flag to false.
                super().__init__()
                self.is_infer = False

            def __call__(self, g: torch.fx.graph.Graph, is_inference: bool):
                # [FUNCTION DOC] HELPER: Records whether the graph is being
                # compiled for inference.
                self.is_infer = is_inference

            def uuid(self) -> bytes:
                # [FUNCTION DOC] HELPER: Returns a hash of this file's contents as
                # the pass identifier.
                return get_hash_for_files((__file__,))

        custom_pass = CustomPass()
        with config.patch(post_grad_custom_post_pass=custom_pass):

            def f(x):
                # [FUNCTION DOC] HELPER: Function under test returning the absolute value
                # of the input.
                return x.abs()

            x = torch.randn(8, dtype=torch.float32)
            torch.testing.assert_close(torch.compile(f)(x), f(x))
            self.assertTrue(custom_pass.is_infer)
            x.requires_grad_()
            torch.testing.assert_close(torch.compile(f)(x), f(x))
            self.assertFalse(custom_pass.is_infer)


if __name__ == "__main__":
    if IS_LINUX and HAS_CPU and torch.backends.mkldnn.is_available():
        run_tests()
