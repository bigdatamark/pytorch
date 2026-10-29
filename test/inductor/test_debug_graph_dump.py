# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for Inductor debug graph dump options: how the graph output format is
# chosen from environment variables (legacy svg flags, global format variable, explicit per-
# artifact format), that dot output is written through the raw writer without invoking
# Graphviz layout, and that the debug formatter uses the configured diagram formats for its
# output file names.

import os
import tempfile
from unittest import mock

import torch
from functorch.compile import draw_graph
from torch._functorch import partitioners
from torch._inductor import config as inductor_config, debug as inductor_debug
from torch._inductor.debug import DebugFormatter
from torch._inductor.test_case import run_tests, TestCase


def _make_graph_module() -> torch.fx.GraphModule:
    # [FUNCTION DOC] HELPER: Builds a minimal graph module with one placeholder, a negation call
    # on it, and an output of that result; used as a stand-in graph for drawing tests.
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    y = graph.call_function(torch.neg, (x,))
    graph.output(y)
    return torch.fx.GraphModule({}, graph)


class _FakeDebugHandler:
    # [CLASS DOC] HELPER: Fake debug handler rooted at a directory. It resolves file names under
    # the root but treats any attempt to open a file or a file context as a test failure.
    def __init__(self, root: str) -> None:
        # [FUNCTION DOC] HELPER: Stores the root directory path.
        self.root = root

    def filename(self, name: str) -> str:
        # [FUNCTION DOC] HELPER: Returns the given file name joined onto the root directory.
        return os.path.join(self.root, name)

    def fopen(self, name: str) -> None:
        # [FUNCTION DOC] HELPER: Always raises an assertion error stating that opening a file
        # was unexpected, so tests fail if the formatter writes through the handler.
        raise AssertionError(f"unexpected fopen({name})")

    def fopen_context(self) -> None:
        # [FUNCTION DOC] HELPER: Always raises an assertion error stating that opening a file
        # context was unexpected.
        raise AssertionError("unexpected fopen_context()")


class TestDebugGraphDump(TestCase):
    # [CLASS DOC] Tests of the graph-format environment resolution in the inductor config and of
    # the dot output paths in graph drawing and the debug formatter. No Graphviz is needed;
    # drawing collaborators are replaced with mocks.
    def test_legacy_svg_flags_default_to_svg_with_global_dot_format(self) -> None:
        # [FUNCTION DOC] With the environment cleared and only the legacy svg flags set
        # (plus a global dot format), asserts the format resolver returns svg for both the
        # original-graph and post-fusion artifacts, i.e. legacy flags take priority over the
        # global format.
        with mock.patch.dict(
            os.environ,
            {
                "INDUCTOR_ORIG_FX_SVG": "1",
                "INDUCTOR_POST_FUSION_SVG": "1",
                "TORCH_COMPILE_GRAPH_FORMAT": "dot",
            },
            clear=True,
        ):
            self.assertEqual(
                inductor_config._get_debug_graph_format(
                    "INDUCTOR_ORIG_FX_GRAPH_FORMAT", "INDUCTOR_ORIG_FX_SVG"
                ),
                "svg",
            )
            self.assertEqual(
                inductor_config._get_debug_graph_format(
                    "INDUCTOR_POST_FUSION_GRAPH_FORMAT",
                    "INDUCTOR_POST_FUSION_SVG",
                ),
                "svg",
            )

    def test_new_graph_flags_use_global_graph_format(self) -> None:
        # [FUNCTION DOC] With the environment cleared and only the new graph-enable flags set
        # (no legacy svg flags) plus a global dot format, asserts the format resolver returns
        # dot for both the original-graph and post-fusion artifacts.
        with mock.patch.dict(
            os.environ,
            {
                "INDUCTOR_ORIG_FX_GRAPH": "1",
                "INDUCTOR_POST_FUSION_GRAPH": "1",
                "TORCH_COMPILE_GRAPH_FORMAT": "dot",
            },
            clear=True,
        ):
            self.assertEqual(
                inductor_config._get_debug_graph_format(
                    "INDUCTOR_ORIG_FX_GRAPH_FORMAT", "INDUCTOR_ORIG_FX_SVG"
                ),
                "dot",
            )
            self.assertEqual(
                inductor_config._get_debug_graph_format(
                    "INDUCTOR_POST_FUSION_GRAPH_FORMAT",
                    "INDUCTOR_POST_FUSION_SVG",
                ),
                "dot",
            )

    def test_explicit_per_artifact_graph_format_wins(self) -> None:
        # [FUNCTION DOC] Sets an explicit per-artifact format for the original graph (raw)
        # alongside a legacy svg flag, and an explicit svg format for post-fusion alongside a
        # global dot format. Asserts the explicit per-artifact value wins in both cases.
        with mock.patch.dict(
            os.environ,
            {
                "INDUCTOR_ORIG_FX_SVG": "1",
                "INDUCTOR_ORIG_FX_GRAPH_FORMAT": "raw",
                "INDUCTOR_POST_FUSION_GRAPH": "1",
                "INDUCTOR_POST_FUSION_GRAPH_FORMAT": "svg",
                "TORCH_COMPILE_GRAPH_FORMAT": "dot",
            },
            clear=True,
        ):
            self.assertEqual(
                inductor_config._get_debug_graph_format(
                    "INDUCTOR_ORIG_FX_GRAPH_FORMAT", "INDUCTOR_ORIG_FX_SVG"
                ),
                "raw",
            )
            self.assertEqual(
                inductor_config._get_debug_graph_format(
                    "INDUCTOR_POST_FUSION_GRAPH_FORMAT",
                    "INDUCTOR_POST_FUSION_SVG",
                ),
                "svg",
            )

    def test_draw_graph_dot_uses_raw_writer(self) -> None:
        # [FUNCTION DOC] Draws a small graph module to a dot file path with a mocked graph
        # drawer whose dot graph raises if the layout-running dot or svg writers are used.
        # Asserts the plain write method is called once with the path and the dot and svg
        # writers are never called.
        gm = _make_graph_module()
        fake_dot_graph = mock.Mock()
        fake_dot_graph.write_dot.side_effect = AssertionError(
            "write_dot invokes Graphviz layout"
        )
        fake_dot_graph.write_svg.side_effect = AssertionError(
            "write_svg invokes Graphviz layout"
        )
        fake_drawer = mock.Mock()
        fake_drawer.return_value.get_main_dot_graph.return_value = fake_dot_graph

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "graph.dot")
            with mock.patch.object(
                partitioners.graph_drawer, "FxGraphDrawer", fake_drawer
            ):
                draw_graph(
                    gm,
                    path,
                    clear_meta=False,
                    prog=["dot", "-Gmaxiter=1"],
                )

        fake_dot_graph.write.assert_called_once_with(path)
        fake_dot_graph.write_dot.assert_not_called()
        fake_dot_graph.write_svg.assert_not_called()

    def test_draw_buffers_dot_does_not_require_graphviz(self) -> None:
        # [FUNCTION DOC] Calls the buffer drawing function with an empty node list and a dot
        # file name while the Graphviz availability check is mocked to raise, graph creation is
        # mocked and the inner drawing function is mocked. Asserts the drawing function was
        # called once with the path as its second positional argument, showing dot output does
        # not need Graphviz.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "graph.dot")
            with (
                mock.patch.object(
                    inductor_debug,
                    "has_dot",
                    side_effect=AssertionError("raw DOT should not require Graphviz"),
                ),
                mock.patch.object(
                    inductor_debug,
                    "create_fx_from_snodes",
                    return_value=_make_graph_module().graph,
                ),
                mock.patch.object(inductor_debug, "draw_graph") as draw_graph_mock,
            ):
                inductor_debug.draw_buffers([], fname=path)

        draw_graph_mock.assert_called_once()
        self.assertEqual(draw_graph_mock.call_args.args[1], path)

    def test_debug_formatter_uses_configured_graph_formats(self) -> None:
        # [FUNCTION DOC] Creates a debug formatter over the fake handler in a temporary
        # directory, sets both diagram format configs to dot, mocks the drawing functions and
        # annotation, then calls the graph diagram and original graph drawing methods. Asserts
        # the buffer drawing and graph drawing mocks are each called once with file names ending
        # in the dot suffix for the graph diagram and the original graph diagram respectively.
        gm = _make_graph_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            formatter = DebugFormatter(_FakeDebugHandler(tmpdir))
            with (
                inductor_config.patch(
                    {
                        "trace.graph_diagram_format": "dot",
                        "trace.orig_fx_graph_diagram_format": "dot",
                    }
                ),
                mock.patch("torch._inductor.debug.draw_buffers") as draw_buffers,
                mock.patch("torch._inductor.debug.annotate_orig_fx_with_snodes"),
                mock.patch("torch._inductor.debug.draw_graph") as inductor_draw_graph,
            ):
                formatter.graph_diagram([])
                formatter.draw_orig_fx_graph(gm, [])

            draw_buffers.assert_called_once()
            self.assertEqual(
                draw_buffers.call_args.kwargs["fname"],
                os.path.join(tmpdir, "graph_diagram.dot"),
            )
            inductor_draw_graph.assert_called_once()
            self.assertEqual(
                inductor_draw_graph.call_args.kwargs["fname"],
                os.path.join(tmpdir, "orig_fx_graph_diagram.dot"),
            )


if __name__ == "__main__":
    run_tests()
