# [MODULE DOC] Inductor debugging and tracing support: DebugContext/DebugFormatter that write debug artifacts (fx graphs, IR,
# diagrams, autotuning results, profiles, logs) into a per-compilation directory, graph drawing helpers, kernel
# provenance tracking globals (kernel to post-grad to pre-grad node mappings), structured-trace logging of collective
# schedules / runtimes / graph execution order, saving and replaying compile_fx_inner arguments, and the AOT Inductor minifier wrapper.
# Import-time side effects: creates the module logger and two artifact loggers, and defines module-level global state
# (graph execution tracking flags, provenance dictionaries, a save-args counter).
import collections
import contextlib
import copy
import dataclasses
import functools
import io
import itertools
import json
import logging
import os
import os.path
import pickle
import pstats
import shutil
import tempfile
import traceback
from collections.abc import Callable, Iterator, Sequence
from typing import Any, IO
from unittest.mock import patch

import torch
from functorch.compile import draw_graph, get_aot_graph_name, get_graph_being_compiled
from torch import fx
from torch._dynamo.repro.after_aot import save_graph_repro
from torch._dynamo.utils import get_debug_dir
from torch._functorch import config as functorch_config
from torch._inductor import utils
from torch._logging import getArtifactLogger
from torch._logging._internal import trace_structured
from torch._utils_internal import signpost_event
from torch.fx.graph_module import GraphModule
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.fx.passes.tools_common import legalize_graph
from torch.types import FileLike
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_leaves, tree_map

from . import config, ir
from .ir import ExternKernel
from .scheduler import (
    BaseSchedulerNode,
    FusedSchedulerNode,
    NopKernelSchedulerNode,
    OutputNode,
    SchedulerNode,
)
from .virtualized import V


log = logging.getLogger(__name__)

# Graph execution tracking for debugging
GRAPH_EXECUTION_ORDER: list[dict[str, object]] | None = None
RECORD_GRAPH_EXECUTION: bool = False
GRAPH_COMPILE_IDS: dict[int, str | None] | None = None

ir_pre_fusion_log = getArtifactLogger(__name__, "ir_pre_fusion")
ir_post_fusion_log = getArtifactLogger(__name__, "ir_post_fusion")
SchedulerNodeList = list[Any]
BufMeta = collections.namedtuple("BufMeta", ["name", "n_origin"])
GRAPHVIZ_COMMAND_SCALABLE = ["dot", "-Gnslimit=2", "-Gnslimit1=2", "-Gmaxiter=5000"]
_RAW_DOT_GRAPH_FORMATS = OrderedSet(["dot", "raw"])


@functools.cache
def has_dot() -> bool:
    # [FUNCTION DOC] Returns True if the graphviz dot executable is found on PATH. Result is cached.
    return shutil.which("dot") is not None


def _get_graph_output_format(fname: str | None) -> str:
    # [FUNCTION DOC] HELPER: Choose the graph output format: the file extension of fname if it has one, otherwise the functorch
    # torch_compile_graph_format config.
    if fname is not None:
        _, ext = os.path.splitext(fname)
        if ext:
            return ext.lstrip(".")
    return functorch_config.torch_compile_graph_format


def draw_buffers(
    nodes: list[BaseSchedulerNode],
    print_graph: bool = False,
    fname: str | None = None,
) -> None:
    # [FUNCTION DOC] Draw the scheduler node graph to a file via graphviz. Args: scheduler nodes, print_graph to also print the fx graph,
    # fname output path (defaults to the graph being compiled). Logs a warning and returns if a non-raw format needs dot
    # and it is missing. Builds an fx graph from the nodes, attaches tensor metadata (shape group and dtype) to each node,
    # legalizes it and renders it. Side effects: writes the diagram file, optional print. Returns None.
    """
    Draw a graph in fname.
    """
    graph_format = _get_graph_output_format(fname)
    if graph_format not in _RAW_DOT_GRAPH_FORMATS and not has_dot():
        log.warning("draw_buffers() requires `graphviz` package")
        return

    if fname is None:
        fname = get_graph_being_compiled()

    graph = create_fx_from_snodes(nodes)

    for node in graph.nodes:
        if "fusion_meta" not in node.meta:
            continue
        group = node.meta["fusion_meta"].group
        if isinstance(group, tuple):
            if isinstance(group[1], int):
                group = (group[1],)
            else:
                group = group[1]
        elif isinstance(group, str):
            # extern / template / nop nodes store a string like "extern"
            # as the group instead of a shape tuple.  Extract the real
            # output shape from the scheduler node's first output buffer.
            try:
                snode = node.meta["fusion_meta"].snode
                size = snode.get_outputs()[0].node.maybe_get_size()
                group = tuple(size) if size else ()
            except Exception:
                group = ()

        # gather meta data
        dtype = None
        if isinstance(node, ir.ComputedBuffer):
            dtype = node.data.dtype

        metadata = TensorMetadata(group, dtype, None, None, None, None, None)  # type: ignore[arg-type]

        node.meta["tensor_meta"] = metadata

    if print_graph:
        print(graph)

    gm = GraphModule({}, graph)
    legalize_graph(gm)
    gm.graph.lint()
    draw_graph(
        gm, fname, clear_meta=False, dot_graph_shape=config.trace.dot_graph_shape
    )


def create_fx_from_snodes(snodes: list[BaseSchedulerNode]) -> fx.Graph:
    # [FUNCTION DOC] Build a fx graph from scheduler nodes. Each scheduler node becomes a fake call_function node named after the node,
    # with fusion metadata (group, scheduler node, type among extern, template, nop, compute, fused). Nodes that feed
    # graph outputs become the graph outputs; edges are made from read dependencies (missing buffers become placeholders,
    # self edges are skipped). Raises RuntimeError for an unknown scheduler node type. Returns the fx Graph.
    """
    Creates a FX Graph from a list of SchedulerNode objects.
    """

    def get_fake_func(name: str) -> Callable[..., int]:
        # [FUNCTION DOC] HELPER: Create a placeholder function with the given name that always returns 0, used as a fake call target.
        def func1(*args: Any) -> int:
            # [FUNCTION DOC] HELPER: Fake call target; ignores arguments and returns 0.
            return 0

        func1.__name__ = name
        return func1

    FusionMeta = collections.namedtuple("FusionMeta", ["group", "snode", "type"])

    buf_to_fx_node = {}
    node_to_fx_node = {}
    graph = torch.fx.Graph()
    first_node = None

    outputs = []
    group: Any = None
    # create call_function node for each Buffer and Kernel
    # pyrefly: ignore [bad-assignment]
    for snode in snodes:
        if snode.is_extern():
            node_type = "extern"
            group = node_type
        elif snode.is_template():
            node_type = "template"
            group = node_type
        elif isinstance(snode, NopKernelSchedulerNode):
            node_type = "nop"
            group = node_type
        elif isinstance(snode, SchedulerNode):
            node_type = "compute"
            group = snode.group
        elif isinstance(snode, FusedSchedulerNode):
            node_type = "fused"
            group = snode.group
        else:
            raise RuntimeError("Unknown node type")

        fused_name = torch._inductor.utils.get_fused_kernel_name(
            snode.get_nodes(), "original_aten"
        )
        func_name = f"{node_type}: {fused_name}"
        node_func = get_fake_func(func_name)
        kwargs = {}
        if hasattr(snode, "get_device"):
            kwargs = {"device": snode.get_device()}
        fx_node = graph.call_function(node_func, args=(), kwargs=kwargs)  # type: ignore[arg-type]

        def in_output(snode: BaseSchedulerNode | FusedSchedulerNode) -> bool:
            # [FUNCTION DOC] HELPER: Whether a scheduler node (or any node inside a fused node) has an output buffer used by an output node.
            if isinstance(snode, FusedSchedulerNode):
                return any(in_output(x) for x in snode.snodes)
            return any(
                isinstance(user.node, OutputNode)
                for buf in snode.get_outputs()
                for user in buf.users
            )

        if in_output(snode):
            outputs.append(fx_node)
        name = snode.get_name()
        fx_node.name = name

        fx_node.meta["fusion_meta"] = FusionMeta(group, snode, node_type)

        node_to_fx_node[name] = fx_node
        for buf in snode.get_outputs():
            buf_to_fx_node[buf.get_name()] = fx_node

        if first_node is None:
            first_node = fx_node

    # create edges between nodes
    for snode in snodes:
        name = snode.get_name()
        deps = snode.read_writes.reads

        fx_node = node_to_fx_node[name]
        new_args = []
        for dep in deps:
            if dep.name in buf_to_fx_node:
                dep_node = buf_to_fx_node[dep.name]
            else:
                with graph.inserting_before(first_node):
                    dep_node = graph.placeholder(dep.name)
                    buf_to_fx_node[dep.name] = dep_node
            if dep_node == fx_node:  # to avoid cycles
                continue
            new_args.append(dep_node)

        fx_node.args = tuple(new_args)

    graph.output(outputs[0] if len(outputs) == 1 else tuple(outputs))
    return graph


def update_orig_fx_node_name_to_buf_name(
    nodes: SchedulerNodeList | None,
    node_name_to_buf_name: dict[str, str],
    parent_buf_name: str | None = None,
    n_origins: int = 0,
) -> None:
    # [FUNCTION DOC] HELPER: Recursively map original fx node names to the scheduler buffer name that first claims them via IR origins.
    # Fused nodes (more than one child) are recursed into using the outermost buffer name as the owner. Mutates the given mapping;
    # does nothing for None. Raises AssertionError if a non-fused node does not have exactly itself as child.
    # KNOWN ISSUE: the n_origins argument is unused.
    if nodes is None:
        return
    for node in nodes:
        # for FusedSchedulerNode, traverse recursively into get_nodes()
        buf_name = node.get_name()
        children_nodes = node.get_nodes()
        if children_nodes is not None and len(children_nodes) > 1:
            update_orig_fx_node_name_to_buf_name(
                children_nodes,
                node_name_to_buf_name,
                buf_name if parent_buf_name is None else parent_buf_name,
            )
            continue
        else:
            # pyrefly: ignore [bad-argument-type, unsupported-operation]
            if not (len(children_nodes) == 1 and children_nodes[0] == node):
                raise AssertionError(
                    "expected a single child node equal to the current node"
                )

        ir_node = node.node
        if ir_node is None or ir_node.origins is None:
            continue
        for origin in ir_node.origins:
            node_name = origin.name
            # when buf1 and buf2 both have origin=node1
            # we draw node1 according to buf1
            if node_name not in node_name_to_buf_name:
                node_name_to_buf_name[node_name] = (
                    buf_name if parent_buf_name is None else parent_buf_name
                )


def get_node_name_to_buf_meta(
    node_name_to_buf_name: dict[str, str],
) -> dict[str, BufMeta]:
    # [FUNCTION DOC] HELPER: Convert a mapping from node name to buffer name into a mapping from node name to BufMeta (buffer name and
    # the number of node names that share that buffer).
    buf_name_to_n_node = {}
    for node_name, buf_name in node_name_to_buf_name.items():
        if buf_name not in buf_name_to_n_node:
            buf_name_to_n_node[buf_name] = OrderedSet([node_name])
        else:
            # pyrefly: ignore [missing-attribute]
            buf_name_to_n_node[buf_name].add(node_name)

    node_name_to_buf_meta = {}
    for node_name, buf_name in node_name_to_buf_name.items():
        n_node = len(buf_name_to_n_node[buf_name])
        node_name_to_buf_meta[node_name] = BufMeta(buf_name, n_node)
    return node_name_to_buf_meta


def annotate_orig_fx_with_snodes(
    gm: torch.fx.GraphModule,
    snodes: SchedulerNodeList,
) -> None:
    # [FUNCTION DOC] Annotate the original fx graph module nodes with buffer metadata (stored in node meta under buf_meta) derived from
    # the scheduler nodes. Mutates gm in place; returns None.
    # KNOWN ISSUE: the docstring wrongly says it creates an FX graph; the check that the mapping is None can never be true.
    """
    Creates a FX Graph from a list of SchedulerNode objects.
    """
    node_name_to_buf_name: dict[str, str] = {}
    update_orig_fx_node_name_to_buf_name(snodes, node_name_to_buf_name)
    if node_name_to_buf_name is None:
        return
    node_name_to_buf_meta = get_node_name_to_buf_meta(node_name_to_buf_name)
    for node in gm.graph.nodes:
        if node.name in node_name_to_buf_meta:
            node.meta["buf_meta"] = node_name_to_buf_meta.get(node.name)


@contextlib.contextmanager
def enable_aot_logging() -> Iterator[None]:
    # [FUNCTION DOC] Context manager that, when the TORCH_COMPILE_DEBUG environment variable equals 1, patches the functorch
    # debug_partitioner config and attaches a file handler (debug log under the debug dir torchinductor folder) to the
    # aot_autograd logger for the duration; otherwise it just yields. The handler and patches are removed on exit. Yields None.
    compile_debug = os.environ.get("TORCH_COMPILE_DEBUG", "0") == "1"

    import torch._functorch.aot_autograd

    log = logging.getLogger(torch._functorch.aot_autograd.__name__)

    stack = contextlib.ExitStack()
    if not compile_debug:
        try:
            yield
        finally:
            stack.close()
        return

    # Enable all graphs to be logged to a file by setting the flags to True
    # and the log level of the file logger to DEBUG
    stack.enter_context(patch("functorch.compile.config.debug_partitioner", True))

    path = os.path.join(get_debug_dir(), "torchinductor")
    os.makedirs(path, exist_ok=True)

    fh = logging.FileHandler(
        os.path.join(
            path,
            f"aot_{get_aot_graph_name()}_debug.log",
        )
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("[%(filename)s:%(lineno)d %(levelname)s] %(message)s")
    )
    log.addHandler(fh)
    try:
        yield
    finally:
        log.removeHandler(fh)
        stack.close()


# Used for provenance tracking
# They are not stored in DebugContext because they are not set in
# _inductor_triton_kernel_to_post_grad_node_info's Debug Context
_inductor_post_to_pre_grad_nodes: dict[str, dict[str, list[str]]] = {}
_inductor_triton_kernel_to_post_grad_node_info: dict[str, list[str]] = {}
_pre_grad_graph_id: int | None = None
_inductor_pre_grad_node_stack_trace: dict[str, str] = {}
_inductor_kernel_stack_trace: dict[str, list[str]] = {}
_kernel_information_jsons: dict[str, dict[str, Any]] = {}
_inductor_kernel_provenance_debug_handle: int = 0


@dataclasses.dataclass
class _KernelExternInfo:
    # [CLASS DOC] HELPER: Per-kernel record for extern kernels: whether it is extern, an optional semantic key and optional shape/dtype metadata.
    is_extern: bool = False
    semantic_key: str | None = None
    shapes: dict[str, Any] | None = None


_inductor_kernel_extern_info: dict[str, _KernelExternInfo] = {}


def get_kernel_information_jsons() -> dict[str, dict[str, Any]]:
    # [FUNCTION DOC] Returns the module-level mapping of kernel information JSON dictionaries.
    return _kernel_information_jsons


def alias_kernel_provenance(original_kernel_name: str, alias_kernel_name: str) -> None:
    # [FUNCTION DOC] Make existing kernel provenance (stack traces and post-grad node info) for one kernel name available under an
    # alias name. Mutates the module-level dictionaries; missing entries are skipped.
    """Expose existing kernel provenance under an additional generated name."""
    if original_kernel_name in _inductor_kernel_stack_trace:
        _inductor_kernel_stack_trace[alias_kernel_name] = _inductor_kernel_stack_trace[
            original_kernel_name
        ]
    if original_kernel_name in _inductor_triton_kernel_to_post_grad_node_info:
        _inductor_triton_kernel_to_post_grad_node_info[alias_kernel_name] = (
            _inductor_triton_kernel_to_post_grad_node_info[original_kernel_name]
        )


def reset_inductor_kernel_provenance_debug_handle() -> None:
    # [FUNCTION DOC] Reset the module-level kernel provenance debug handle counter to zero.
    global _inductor_kernel_provenance_debug_handle
    _inductor_kernel_provenance_debug_handle = 0


@contextlib.contextmanager
def reset_provenance_globals() -> Iterator[None]:
    # [FUNCTION DOC] Context manager that saves copies of the provenance globals, resets them to defaults (graph id -1, empty mappings,
    # handle 0) on entry, and restores the saved values on exit (even on error). Yields None.
    """Context manager that resets provenance tracking globals upon entering
    and restores their original values when exiting."""
    global _pre_grad_graph_id
    global _inductor_post_to_pre_grad_nodes
    global _inductor_triton_kernel_to_post_grad_node_info
    global _inductor_pre_grad_node_stack_trace
    global _inductor_kernel_stack_trace
    global _inductor_kernel_provenance_debug_handle
    global _inductor_kernel_extern_info

    # Store original values
    original_pre_grad_graph_id = _pre_grad_graph_id
    original_post_to_pre_grad_nodes = _inductor_post_to_pre_grad_nodes.copy()
    original_triton_kernel_to_post_grad_node_info = (
        _inductor_triton_kernel_to_post_grad_node_info.copy()
    )
    original_inductor_pre_grad_node_stack_trace = (
        _inductor_pre_grad_node_stack_trace.copy()
    )
    original_inductor_kernel_stack_trace = _inductor_kernel_stack_trace.copy()
    original_inductor_kernel_provenance_debug_handle = (
        _inductor_kernel_provenance_debug_handle
    )
    original_inductor_kernel_extern_info = dict(_inductor_kernel_extern_info)

    # Reset to default values
    _pre_grad_graph_id = -1
    _inductor_post_to_pre_grad_nodes = {}
    _inductor_triton_kernel_to_post_grad_node_info = {}
    _inductor_pre_grad_node_stack_trace = {}
    _inductor_kernel_stack_trace = {}
    # Kernel timeline information is populated from compile_fx_inner while this
    # context is active.  It must outlive the reset scope so the profiler can
    # consume it during export_chrome_trace, which then clears it.
    _inductor_kernel_provenance_debug_handle = 0
    _inductor_kernel_extern_info = {}

    try:
        yield
    finally:
        # Restore original values
        _pre_grad_graph_id = original_pre_grad_graph_id
        _inductor_post_to_pre_grad_nodes = original_post_to_pre_grad_nodes
        _inductor_triton_kernel_to_post_grad_node_info = (
            original_triton_kernel_to_post_grad_node_info
        )
        _inductor_kernel_stack_trace = original_inductor_kernel_stack_trace
        _inductor_pre_grad_node_stack_trace = (
            original_inductor_pre_grad_node_stack_trace
        )
        _inductor_kernel_provenance_debug_handle = (
            original_inductor_kernel_provenance_debug_handle
        )
        _inductor_kernel_extern_info = original_inductor_kernel_extern_info


class DebugContext:
    # [CLASS DOC] Context manager created per compilation that manages the debug directory (state: _path, _prof, _stack of exit
    # callbacks), log capture and installs itself as the debug handler on V. Attribute access for unknown names returns the
    # matching DebugFormatter method when trace config for that name is enabled, else a no-op callable.
    _counter = itertools.count()

    @staticmethod
    def create_debug_dir(folder_name: str) -> str | None:
        # [FUNCTION DOC] Static: create a unique numbered directory under the debug dir (config trace debug_dir or the default) in a
        # torchinductor folder, named by folder_name and a counter; skips counters whose directory already exists.
        # Returns the path (None only if the counter is exhausted). Creates directories on disk.
        debug_dir = config.trace.debug_dir or get_debug_dir()
        for n in DebugContext._counter:
            dirname = os.path.join(
                debug_dir,
                "torchinductor",
                f"{folder_name}.{n}",
            )
            try:
                os.makedirs(dirname, exist_ok=False)
                return dirname
            except FileExistsError:
                # Another process (e.g. a peer rank with the same debug_dir)
                # created this counter slot first; advance to the next n.
                continue
        return None

    def __init__(self) -> None:
        # [FUNCTION DOC] Initialize with no profiler, no path and an empty exit stack.
        self._prof = None
        self._path = None
        self._stack = contextlib.ExitStack()

    def copy(self, new_path: str) -> None:
        # [FUNCTION DOC] Copy the debug directory to new_path (must end in .debug, else AssertionError) under a file lock, replacing any
        # existing directory. Does nothing if no debug path. OS errors are logged as a warning.
        if not self._path:
            return
        if not new_path.endswith(".debug"):
            raise AssertionError(new_path)
        from filelock import FileLock

        try:
            with FileLock(f"{new_path}.lock"):
                if os.path.exists(new_path):
                    shutil.rmtree(new_path)
                shutil.copytree(self._path, new_path)
        except OSError:
            log.warning(
                "Failed to copy debug files from %s to %s", self._path, new_path
            )

    def fopen(
        self,
        filename: str,
        write_mode: str = "w",
        *args: Any,
        **kwargs: Any,
    ) -> IO[Any]:
        # [FUNCTION DOC] Open a file inside the debug directory with the given mode and extra open arguments. Raises AssertionError if no
        # debug path is set. Returns the file object.
        if not self._path:
            raise AssertionError("expected self._path to be set")
        return open(os.path.join(self._path, filename), write_mode, *args, **kwargs)

    @contextlib.contextmanager
    def fopen_context(
        self,
        filename: str,
        write_mode: str = "w",
        *args: Any,
        **kwargs: Any,
    ) -> Iterator[IO[Any]]:
        # [FUNCTION DOC] Context manager version of fopen: opens a file in the debug directory and yields it, closing on exit. Raises
        # AssertionError if no debug path is set.
        if not self._path:
            raise AssertionError("expected self._path to be set")
        with open(os.path.join(self._path, filename), write_mode, *args, **kwargs) as f:
            yield f

    def filename(self, suffix: str) -> str:
        # [FUNCTION DOC] Returns the path of a file with the given suffix inside the debug directory. Raises AssertionError if no path is set.
        if not self._path:
            raise AssertionError("expected self._path to be set")
        return os.path.join(self._path, suffix)

    def upload_tar(self) -> None:
        # [FUNCTION DOC] If the trace upload_tar config callable is set, tar-gzip the debug directory and call the callable with the
        # tarball path. Raises AssertionError if no debug path. Side effects: writes a tar file, external upload.
        if config.trace.upload_tar is not None:
            import tarfile

            if not self._path:
                raise AssertionError("expected self._path to be set")
            tar_file = os.path.join(
                self._path, f"{os.path.basename(self._path)}.tar.gz"
            )
            with tarfile.open(tar_file, "w:gz") as tar:
                tar.add(self._path, arcname=os.path.basename(self._path))
            config.trace.upload_tar(tar_file)

    def __enter__(self) -> None:
        # [FUNCTION DOC] Enter the context: with config debug, set the torch._dynamo logger to DEBUG (restored on exit); install this
        # context as the debug handler on V; if tracing is enabled create the debug directory and set up debug.log and/or
        # info.log capture per config. Returns None.
        if config.debug:
            log = logging.getLogger("torch._dynamo")
            prev_level = log.level
            log.setLevel(logging.DEBUG)

            def reset_log_level(level: Any) -> None:
            # [FUNCTION DOC] HELPER: Restore a logger level (callback registered on the exit stack).
                log.setLevel(level)

            self._stack.callback(reset_log_level, prev_level)

        self._stack.enter_context(V.set_debug_handler(self))

        if not config.trace.enabled:
            return

        self._path = self.create_debug_dir(get_aot_graph_name())  # type: ignore[assignment]

        if config.trace.debug_log:
            self._setup_log_capture("debug.log", logging.DEBUG)
        if config.trace.info_log:
            self._setup_log_capture("info.log", logging.INFO)

    def _setup_log_capture(
        self,
        filename: str,
        level: int,
    ) -> None:
        # [FUNCTION DOC] HELPER: Open a log file in the debug directory and attach a stream handler at the level to the torch._inductor
        # logger (lowering its level if needed); handler removal is registered on the exit stack.
        log = logging.getLogger("torch._inductor")
        fd = self._stack.enter_context(self.fopen(filename))
        ch = logging.StreamHandler(fd)
        ch.setLevel(level)
        ch.setFormatter(
            logging.Formatter("[%(filename)s:%(lineno)d %(levelname)s] %(message)s")
        )
        log.addHandler(ch)
        log.setLevel(min(log.level, level))
        self._stack.callback(log.removeHandler, ch)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        # [FUNCTION DOC] Exit the context: if a profiler exists disable it and save its stats; if a debug path exists upload the tarball and
        # log a warning with the path; close the exit stack (undoing handlers and log levels). Does not suppress exceptions.
        if self._prof:
            self._prof.disable()
            self._save_profile_data()

        if self._path:
            self.upload_tar()
            log.warning("%s debug trace: %s", get_graph_being_compiled(), self._path)
        self._stack.close()

    def _save_profile_data(self) -> None:
        # [FUNCTION DOC] HELPER: Dump profiler data to compile.prof and write cumulative and total time statistics (top 100 each) to
        # compile.stats in the debug directory. Raises AssertionError if no profiler is set.
        if not self._prof:
            raise AssertionError("expected self._prof to be set")
        self._prof.dump_stats(self.filename("compile.prof"))
        with self.fopen("compile.stats") as fd:
            stats = pstats.Stats(self._prof, stream=fd)
            stats.strip_dirs()
            stats.sort_stats("cumtime")
            stats.print_stats(100)
            stats.sort_stats("tottime")
            stats.print_stats(100)

    def __getattr__(self, name: str) -> Callable[..., None] | None:
        # [FUNCTION DOC] Dynamic dispatch of debug hooks. If tracing is enabled and the config flag with this name is truthy, return the
        # same-named DebugFormatter method (None with a warning if that fails); otherwise return a no-op callable accepting any arguments.
        if config.trace.enabled and getattr(config.trace, name):
            try:
                return getattr(DebugFormatter(self), name)
            except Exception:
                log.warning("Ignoring exception in debug code", exc_info=True)
                return None
        else:

            def ignored(*args: Any, **kwargs: Any) -> None:
            # [FUNCTION DOC] HELPER: No-op replacement hook.
                pass

            return ignored


class DebugFormatter:
    # [CLASS DOC] Writes the individual debug artifacts for a DebugContext; holds the handler and its file helper methods (fopen,
    # fopen_context, filename). Method names match trace config flags and are looked up via DebugContext attribute access.
    def __init__(self, handler: DebugContext) -> None:
        # [FUNCTION DOC] Store the handler and its file helpers.
        self.fopen = handler.fopen
        self.fopen_context = handler.fopen_context
        self.filename = handler.filename
        self.handler = handler

    def fx_graph(
        self,
        gm: torch.fx.GraphModule,
        inputs: list[torch.Tensor],
    ) -> None:
        # [FUNCTION DOC] Write fx_graph_runnable.py (a repro script via save_graph_repro, optionally saving real tensors when configured, with
        # tracing disabled during the save) and fx_graph_readable.py (printed readable graph) into the debug directory.
        with self.fopen("fx_graph_runnable.py") as fd:
            save_dir = None
            if torch._inductor.config.trace.save_real_tensors:
                inputs = torch._subclasses.fake_utils.try_convert_fake_to_real(inputs)
                save_dir = os.path.dirname(fd.name)

            # don't try to use stable hash torchinductor compilation if saving real tensors
            # and avoid recursively trying to save real tensors inside of the inductor compilation
            # regardless
            stable_hash = torch._inductor.config.trace.save_real_tensors
            with torch._inductor.config.patch(
                {"trace.enabled": False, "trace.save_real_tensors": False}
            ):
                save_graph_repro(
                    fd,
                    gm,
                    inputs,
                    "inductor",
                    save_dir=save_dir,
                    stable_hash=stable_hash,
                )

        with self.fopen("fx_graph_readable.py") as fd:
            fd.write(gm.print_readable(print_output=False))

    def fx_graph_transformed(
        self,
        gm: torch.fx.GraphModule,
        inputs: list[torch.Tensor],
    ) -> None:
        # [FUNCTION DOC] Write the readable form of the transformed graph module to fx_graph_transformed.py. Inputs are unused.
        with self.fopen("fx_graph_transformed.py") as fd:
            fd.write(gm.print_readable(print_output=False))

    def ir_pre_fusion(self, nodes: SchedulerNodeList) -> None:
        # [FUNCTION DOC] Write the pre-fusion IR text of the scheduler nodes to ir_pre_fusion.txt.
        with self.fopen("ir_pre_fusion.txt") as fd:
            fd.write(self._write_ir(nodes))

    def ir_post_fusion(self, nodes: SchedulerNodeList) -> None:
        # [FUNCTION DOC] Write the post-fusion IR text of the scheduler nodes to ir_post_fusion.txt.
        with self.fopen("ir_post_fusion.txt") as fd:
            fd.write(self._write_ir(nodes))

    @staticmethod
    def _write_ir(nodes: SchedulerNodeList) -> str:
        # [FUNCTION DOC] HELPER: Static: concatenate debug_str of each node separated by blank lines and return the string.
        buf = io.StringIO()
        for node in nodes:
            buf.write(node.debug_str())
            buf.write("\n\n\n")
        return buf.getvalue()

    def graph_diagram(self, nodes: SchedulerNodeList) -> None:
        # [FUNCTION DOC] Draw the scheduler node graph to graph_diagram with the configured format extension in the debug directory.
        draw_buffers(
            nodes,
            fname=self.filename(f"graph_diagram.{config.trace.graph_diagram_format}"),
        )

    def draw_orig_fx_graph(
        self,
        gm: torch.fx.GraphModule,
        nodes: SchedulerNodeList,
    ) -> None:
        # [FUNCTION DOC] Annotate the original fx graph with scheduler buffer info and draw it to the orig fx graph diagram file in the
        # debug directory, using a scalable graphviz command and stack trace parsing.
        annotate_orig_fx_with_snodes(gm, nodes)
        draw_graph(
            gm,
            fname=self.filename(
                f"orig_fx_graph_diagram.{config.trace.orig_fx_graph_diagram_format}"
            ),
            clear_meta=False,
            prog=GRAPHVIZ_COMMAND_SCALABLE,
            parse_stack_trace=True,
            dot_graph_shape=config.trace.dot_graph_shape,
        )

    def output_code(self, filename: str, extension: str = "py") -> None:
        # [FUNCTION DOC] Copy the given generated code file into the debug directory as output_code with the given extension.
        shutil.copy(filename, self.filename(f"output_code.{extension}"))

    def log_autotuning_results(
        self,
        name: str,
        input_nodes: list[ir.IRNode],
        timings: dict["ChoiceCaller", float],  # type: ignore[name-defined] # noqa: F821
        elapse: float,
        precompile_elapse: float,
        prescreening_elapse: float | None,
    ) -> None:
        # [FUNCTION DOC] Append one JSON line per autotuning candidate to autotuning_result_json_list.txt in the debug directory. Each line merges
        # the candidate's info dict with general properties (op name, CUDA device name and count, input node descriptions,
        # autotune, precompile and prescreen times) and its benchmark time. Requires CUDA for the device queries.
        from .ir import FixedLayout

        def build_node_info(node: ir.IRNode) -> dict[str, str]:
            # [FUNCTION DOC] HELPER: Describe an IR node as a dictionary of strings (name, type, layout with static hints, dtype, device, stride,
            # size, numel, and nested data node); fields that raise errors are omitted.
            if hasattr(node, "name"):
                node_name = node.name
            else:
                node_name = ""
            node_info = {
                "name": node_name,
                "type": type(node).__name__,
            }
            try:
                layout = node.get_output_spec()
                if isinstance(layout, FixedLayout):
                    static_layout = FixedLayout(
                        layout.device,
                        dtype=layout.dtype,
                        size=V.graph.sizevars.optimization_hints(layout.size),
                        stride=V.graph.sizevars.optimization_hints(layout.stride),
                        offset=V.graph.sizevars.optimization_hint(
                            layout.offset, fallback=0
                        ),
                    )
                    node_info["layout"] = str(static_layout)
                else:
                    node_info["layout"] = str(layout)
            except Exception:
                pass
            try:
                node_info["dtype"] = str(node.get_dtype())
            except Exception:
                pass
            try:
                node_info["device"] = str(node.get_device())
            except Exception:
                pass
            try:
                node_info["stride"] = str(
                    V.graph.sizevars.optimization_hints(node.get_stride())
                )
            except Exception:
                pass
            try:
                node_info["size"] = str(
                    V.graph.sizevars.optimization_hints(node.get_size())
                )  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                node_info["numel"] = str(
                    V.graph.sizevars.optimization_hint(node.get_numel())
                )
            except Exception:
                pass
            if hasattr(node, "data") and isinstance(node.data, ir.IRNode):
                node_info["data"] = build_node_info(node.data)
            return node_info

        general_properties = {
            "op_name": name,
            "cuda_device_name": torch.cuda.get_device_name(),
            "cuda_device_count": torch.cuda.device_count(),
            "input_nodes": [build_node_info(node) for node in input_nodes],
            "autotuning_time": elapse,
            "precompile_time": precompile_elapse,
            "prescreening_time": prescreening_elapse,
        }
        with self.fopen_context(
            "autotuning_result_json_list.txt", "at", encoding="utf-8"
        ) as fd:
            for caller, time in timings.items():
                info_dict = dict(caller.info_dict())
                info_dict.update(general_properties)
                info_dict["benchmark_result"] = time
                json.dump(info_dict, fd)
                fd.write("\n")


def log_ir_pre_fusion(nodes: SchedulerNodeList) -> None:
    # [FUNCTION DOC] Log the pre-fusion IR at info level when enabled and forward the nodes to the debug handler hook.
    if ir_pre_fusion_log.isEnabledFor(logging.INFO):
        ir_pre_fusion_log.info("BEFORE FUSION\n%s", DebugFormatter._write_ir(nodes))

    V.debug.ir_pre_fusion(nodes)


def log_ir_post_fusion(nodes: SchedulerNodeList) -> None:
    # [FUNCTION DOC] Log the post-fusion IR at info level when enabled and forward the nodes to the debug handler hook.
    if ir_post_fusion_log.isEnabledFor(logging.INFO):
        ir_post_fusion_log.info("AFTER FUSION\n%s", DebugFormatter._write_ir(nodes))

    V.debug.ir_post_fusion(nodes)


def _dump_collective_schedule(schedule: list[str | None]) -> None:
    # [FUNCTION DOC] HELPER: Emit the collective schedule list as a structured trace artifact (json). Errors are logged at debug level, never raised.
    try:
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "inductor_collective_schedule",
                "encoding": "json",
            },
            payload_fn=lambda: schedule,
        )
    except Exception:
        log.debug(
            "Failed to log inductor_collective_schedule via structured logging",
            exc_info=True,
        )


def log_collective_schedule(nodes: Sequence[BaseSchedulerNode]) -> None:
    # [FUNCTION DOC] Collect the kernel names of collective kernels among the scheduler nodes, in order, and dump them as a structured
    # trace artifact only if at least one exists.
    schedule = [
        getattr(op, "python_kernel_name", None)
        for node in nodes
        if isinstance(op := getattr(node, "node", None), ir._CollectiveKernel)
    ]

    # Only log when there is at least one collective op
    if schedule:
        _dump_collective_schedule(schedule)


def log_runtime_and_tensor_meta(node_runtimes: Sequence[tuple[Any, float]]) -> None:
    # [FUNCTION DOC] Emit a structured trace artifact (json) with, per scheduler node, name, type (collective or compute), estimated
    # runtime in ns and output shape/stride/dtype. Sizes are converted via optimization hints. All errors are swallowed
    # with a debug log.
    """Log per-op runtime estimates and output tensor metadata for TLParse."""

    try:
        to_optimization_hints = V.graph.sizevars.optimization_hints

        def to_list(x: Sequence[Any] | None) -> list[Any]:
            # [FUNCTION DOC] HELPER: Convert a size or stride sequence to a list of hinted values; None gives an empty list.
            return list(to_optimization_hints(x)) if x is not None else []

        def dtype_to_str(dtype: Any) -> str | None:
            # [FUNCTION DOC] HELPER: Convert a dtype to a string without the torch prefix; None stays None.
            if dtype is None:
                return None
            s = str(dtype)
            s = s.removeprefix("torch.")
            return s

        ops: list[dict[str, Any]] = []
        for s, runtime_ns in node_runtimes:
            name = getattr(s.node, "python_kernel_name", s.get_name())
            op_type = "collective" if utils.is_collective(s.node) else "compute"

            # Build outputs metadata if available
            outputs: list[dict[str, Any]] = []
            try:
                for buf in s.get_outputs():
                    irnode = buf.node
                    shape = irnode.maybe_get_size()
                    stride = (
                        irnode.get_stride()
                        if isinstance(irnode.layout, ir.Layout)
                        else None
                    )
                    dtype = irnode.maybe_get_dtype()
                    outputs.append(
                        {
                            "shape": to_list(shape),
                            "stride": to_list(stride),
                            "dtype": dtype_to_str(dtype),
                        }
                    )
            except Exception:
                pass

            ops.append(
                {
                    "name": name,
                    "type": op_type,
                    "estimated_runtime_ns": runtime_ns,
                    "outputs": outputs,
                }
            )

        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "inductor_runtime_and_tensor_meta",
                "encoding": "json",
            },
            payload_fn=lambda: {"ops": ops},
        )
    except Exception:
        log.debug("Failed to log inductor_runtime_and_tensor_meta", exc_info=True)


def log_graph_execution() -> None:
    # [FUNCTION DOC] Emit the recorded graph execution order as a structured trace artifact if any exists; errors are logged at debug level.
    """Emit a structured artifact with the graph execution order."""
    if not GRAPH_EXECUTION_ORDER:
        return
    try:
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "graph_execution",
                "encoding": "json",
            },
            payload_fn=lambda: {"graph_execution_order": GRAPH_EXECUTION_ORDER},
        )
    except Exception:
        log.debug("Failed to log graph_execution", exc_info=True)


@contextlib.contextmanager
def record_and_log_graph_execution_order() -> Iterator[None]:
    # [FUNCTION DOC] Context manager that enables graph execution recording (resets execution order and compile id globals), and on
    # exit logs the order and clears the globals and flag. Yields None.
    """Record graph execution order and log it once on exit."""
    global RECORD_GRAPH_EXECUTION, GRAPH_EXECUTION_ORDER, GRAPH_COMPILE_IDS
    GRAPH_EXECUTION_ORDER = []
    GRAPH_COMPILE_IDS = {}
    RECORD_GRAPH_EXECUTION = True
    try:
        yield
    finally:
        log_graph_execution()
        RECORD_GRAPH_EXECUTION = False
        GRAPH_EXECUTION_ORDER = None
        GRAPH_COMPILE_IDS = None


@dataclasses.dataclass
class TensorMetadataHolder:
    # [CLASS DOC] HELPER: Pickle-friendly holder of tensor metadata and its device, replacing tensors when saving arguments.
    tensor_metadata: TensorMetadata
    device: torch.device


@dataclasses.dataclass
class SymExprMetadataHolder:
    # [CLASS DOC] HELPER: Pickle-friendly holder of a symbolic scalar: type name, expression, hint and per-symbol hints.
    pytype: str
    expr: Any
    hint: int | float | bool | None
    symbol_hints: dict[str, int | float | bool]


save_args_cnt = itertools.count()


def create_mapping_pre_post_grad_nodes(
    pre_grad_graph_id: int | None,
    post_to_pre_grad_nodes_json: dict[str, Any],
) -> dict[str, dict[str, list[str]]]:
    # [FUNCTION DOC] Build bidirectional mappings (preToPost and postToPre) between pre-grad graph node names and post-grad node keys, from
    # a provenance mapping, considering only nodes whose graph_id equals the pre-grad graph id and following nested from_node
    # entries. Returns empty mappings on malformed input, non-int graph id or any exception (an error is signposted and logged).
    """
    Create bidirectional mappings between pre_grad graph nodes
    and post_grad graph code nodes, and vice versa.
    """
    # return a dummy dict if there's any error
    empty_return: dict[str, dict[str, list[str]]] = {
        "preToPost": {},
        "postToPre": {},
    }

    if not isinstance(post_to_pre_grad_nodes_json, dict):
        log.error(
            "Provenance tracking error: post_to_pre_grad_nodes_json is not a dict"
        )
        return empty_return

    if not isinstance(pre_grad_graph_id, int):
        # pre_grad_graph_id may be empty if there's no pre_grad graph
        # and there's only a backward graph from backward pass engine
        return empty_return

    pre_to_post: dict[str, Any] = collections.defaultdict(OrderedSet)
    post_to_pre: dict[str, Any] = collections.defaultdict(OrderedSet)

    try:

        def check_format(node: dict[str, Any]) -> bool:
            # [FUNCTION DOC] HELPER: Validate that a provenance node is a mapping with graph_id, name and from_node keys; logs an error and returns False if not.
            if not isinstance(node, dict):
                log.error(
                    "Provenance tracking error: node provenance in post_to_pre_grad_nodes_json is not a dict"
                )
                return False
            if "graph_id" not in node or "name" not in node or "from_node" not in node:
                log.error(
                    "Provenance tracking error: node provenance in post_to_pre_grad_nodes_json has wrong format"
                )
                return False
            return True

        for outer_key, node_array in post_to_pre_grad_nodes_json.items():
            if not isinstance(node_array, list):
                log.error(
                    "Provenance tracking error: post_to_pre_grad_nodes_json value is not a list"
                )
                return empty_return
            for node in node_array:
                if not check_format(node):
                    return empty_return
                # Check the current node first
                if node.get("graph_id") == pre_grad_graph_id:
                    pre_to_post[node["name"]].add(outer_key)
                    post_to_pre[outer_key].add(node["name"])

                # Check nested from_node array recursively, add node with the right graph_id to the map
                stack = [(n, outer_key) for n in node.get("from_node", [])]
                while stack:
                    current_node, parent_key = stack.pop()
                    if not check_format(current_node):
                        return empty_return
                    if current_node.get("graph_id") == pre_grad_graph_id:
                        pre_to_post[current_node["name"]].add(parent_key)
                        post_to_pre[parent_key].add(current_node["name"])
                    stack.extend(
                        (n, parent_key) for n in current_node.get("from_node", [])
                    )

        def convert_sets_to_lists(d: dict[str, Any]) -> None:
            # [FUNCTION DOC] HELPER: Convert each value of the mapping to a list in place so it is JSON serializable.
            # KNOWN ISSUE: the final rebinding of d to a copy has no effect.
            for key in d:
                d[key] = list(d[key])
            d = dict(d)

        # convert to list because set is not JSON serializable
        convert_sets_to_lists(pre_to_post)
        convert_sets_to_lists(post_to_pre)
        return {
            "preToPost": pre_to_post,
            "postToPre": post_to_pre,
        }
    except Exception as e:
        # Since this is just logging code, it should never interfere with regular
        # program execution, so we use this try-except to guard against any error
        signpost_event(
            "inductor",
            "provenance_tracking_error",
            {
                "function": "create_mapping_pre_post_grad_nodes",
                "error_msg": str(e),
                "stack_trace": traceback.format_exc(),
            },
        )
        log.error("post_to_pre_grad_nodes_json:  %s", post_to_pre_grad_nodes_json)
        log.error("pre_grad_graph_id:  %s", pre_grad_graph_id)
        return empty_return


def create_node_mapping_kernel_to_post_grad(
    triton_kernel_to_post_grad_json: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    # [FUNCTION DOC] Build the kernel-to-post-grad and post-grad-to-kernel mappings (cppCodeToPost and postToCppCode) from a mapping of kernel
    # name to list of post-grad node names. Returns empty mappings on malformed input or any exception (signposted and logged).
    """Create bidirectional mappings between triton kernel name and post_grad
    graph code nodes, and vice versa.
    """

    # return a dummy dict if there's any error
    empty_return: dict[str, dict[str, Any]] = {
        "cppCodeToPost": {},
        "postToCppCode": {},
    }

    if not isinstance(triton_kernel_to_post_grad_json, dict):
        log.error(
            "Provenance tracking error: triton_kernel_to_post_grad_json is not a dict"
        )
        return empty_return

    post_to_cpp_code: dict[str, Any] = collections.defaultdict(OrderedSet)

    try:
        for outer_key, node_array in triton_kernel_to_post_grad_json.items():
            if not isinstance(node_array, list):
                log.error(
                    "Provenance tracking error: triton_kernel_to_post_grad_json value is not a list"
                )
                return empty_return
            for curr_node in node_array:
                post_to_cpp_code[curr_node].add(outer_key)

        def convert_sets_to_lists(d: dict[str, Any]) -> None:
            # [FUNCTION DOC] HELPER: Convert each value of the mapping to a list in place so it is JSON serializable.
            # KNOWN ISSUE: the final rebinding of d to a copy has no effect.
            for key in d:
                d[key] = list(d[key])
            d = dict(d)

        # convert to list because set is not JSON serializable
        convert_sets_to_lists(post_to_cpp_code)
        return {
            "cppCodeToPost": triton_kernel_to_post_grad_json,
            "postToCppCode": post_to_cpp_code,
        }
    except Exception as e:
        # Since this is just logging code, it should never interfere with regular
        # program execution, so we use this try-except to guard against any error
        signpost_event(
            "inductor",
            "provenance_tracking_error",
            {
                "function": "create_mapping_kernel_to_post_grad",
                "error_msg": str(e),
                "stack_trace": traceback.format_exc(),
            },
        )
        log.error(
            "triton_kernel_to_post_grad_json:  %s", triton_kernel_to_post_grad_json
        )
        return empty_return


def dump_inductor_provenance_info() -> dict[str, Any]:
    # [FUNCTION DOC] Build the provenance node mapping dictionary: when a pre-grad graph id is set, merge the post-to-pre mapping with the
    # kernel-to-post-grad mapping and, if tracing is enabled, write it to a json file in the debug directory; always add version 2.0.
    # Returns an empty dictionary on any exception (signposted).
    try:
        global _pre_grad_graph_id
        global _inductor_post_to_pre_grad_nodes
        global _inductor_triton_kernel_to_post_grad_node_info
        node_mapping: dict[str, Any] = {}
        if _pre_grad_graph_id:
            node_mapping_kernel = create_node_mapping_kernel_to_post_grad(
                _inductor_triton_kernel_to_post_grad_node_info
            )
            node_mapping = {
                **_inductor_post_to_pre_grad_nodes,
                **node_mapping_kernel,
            }
            if config.trace.enabled:
                with V.debug.fopen(
                    "inductor_provenance_tracking_node_mappings.json", "w"
                ) as fd:
                    json.dump(node_mapping, fd)
        # we need to update the node mapping version when node mapping format changes
        # so the tlparse tool knows which node mapping version it is looking at
        node_mapping["version"] = 2.0
        return node_mapping
    except Exception as e:
        # Since this is just debugging, it should never interfere with regular
        # program execution, so we use this try-except to guard against any error
        signpost_event(
            "inductor",
            "provenance_tracking_error",
            {
                "function": "dump_inductor_provenance_info",
                "error_msg": str(e),
                "stack_trace": traceback.format_exc(),
            },
        )
        return {}


def _serialize_provenance_shape(size: Sequence[Any]) -> list[int | str]:
    # [FUNCTION DOC] HELPER: Convert dimensions to a list where static dims become ints and symbolic dims become strings.
    shape: list[int | str] = []
    for dim in size:
        if ir._is_static(dim):
            shape.append(int(dim))
        else:
            shape.append(str(dim))
    return shape


def _get_tensor_metadata(node: ir.IRNode) -> tuple[list[int | str], str] | None:
    # [FUNCTION DOC] HELPER: Returns the serialized shape and dtype string of an IR node, or None if it has no tensor output or unknown size or dtype.
    if not node.has_tensor_output():
        return None

    size = node.maybe_get_size()
    dtype = node.maybe_get_dtype()
    if size is None or dtype is None:
        return None

    return _serialize_provenance_shape(size), str(dtype)


def create_kernel_information_json() -> dict[str, dict[str, Any]]:
    # [FUNCTION DOC] Build per-kernel information (stack traces, post-grad nodes, pre-grad nodes, extern semantic key plus optional extern shape
    # metadata) from the provenance globals, over all known kernels. The semantic key is the explicit annotation, else the single
    # pre-grad node for extern kernels, else None. Returns an empty dictionary on any exception (signposted and logged).
    """Create kernel information JSON with extended metadata for cross-hardware comparison."""
    try:
        global _inductor_post_to_pre_grad_nodes
        global _inductor_kernel_stack_trace
        global _inductor_triton_kernel_to_post_grad_node_info
        global _inductor_kernel_extern_info

        post_to_pre = _inductor_post_to_pre_grad_nodes.get("postToPre", {})
        all_kernels = OrderedSet(_inductor_kernel_stack_trace.keys()) | OrderedSet(
            _inductor_triton_kernel_to_post_grad_node_info.keys()
        )

        result = {}
        for kernel_key in all_kernels:
            post_grad_nodes = _inductor_triton_kernel_to_post_grad_node_info.get(
                kernel_key, []
            )

            pre_grad_nodes: OrderedSet[str] = OrderedSet()
            for post_node in post_grad_nodes:
                pre_grad_nodes.update(post_to_pre.get(post_node, []))

            stack_traces = _inductor_kernel_stack_trace.get(kernel_key, [])
            pre_grad_list = list(pre_grad_nodes)

            # extern_semantic_key precedence:
            # 1. Explicit annotation (e.g., FP8 bridge) — highest priority
            # 2. Single-element pre_grad_nodes for externs — derived identity
            # 3. None
            info = _inductor_kernel_extern_info.get(kernel_key)
            if info is not None and info.semantic_key is not None:
                extern_semantic_key = info.semantic_key
            elif info is not None and info.is_extern and len(pre_grad_list) == 1:
                extern_semantic_key = pre_grad_list[0]
            else:
                extern_semantic_key = None

            result[kernel_key] = {
                "stack_traces": stack_traces,
                "post_grad_nodes": post_grad_nodes,
                "pre_grad_nodes": pre_grad_list,
                "extern_semantic_key": extern_semantic_key,
            }

            # Merge shape metadata if available (extern ops only)
            if info is not None and info.shapes:
                result[kernel_key].update(info.shapes)

        return result
    except Exception as e:
        signpost_event(
            "inductor",
            "provenance_tracking_error",
            {
                "function": "create_kernel_information_json",
                "error_msg": str(e),
                "stack_trace": traceback.format_exc(),
            },
        )
        log.warning("create_kernel_information_json: exception: %s", e)
        return {}


def set_kernel_post_grad_provenance_tracing(
    node_schedule: Sequence[BaseSchedulerNode] | ExternKernel,
    kernel_name: str,
    is_extern: bool = False,
) -> int | None:
    # [FUNCTION DOC] Record provenance for a generated kernel: returns None if provenance tracking level is 0 or on any exception. Otherwise
    # increments the debug handle, names the kernel with the handle appended, and for an extern kernel (must be an ExternKernel,
    # else AssertionError caught and returns None) records origin nodes, semantic key and input/output shapes and dtypes; for a
    # normal kernel (must be a list) records origins and stack traces from the scheduler nodes. Mutates the module globals.
    # Returns the new debug handle.
    """
    Set the mapping between `kernel_name` and the post_grad nodes in `node_schedule`.

    Returns a unique int debug handler for each call to this function.
    """

    if config.effective_provenance_tracking_level() == 0:
        return None

    try:
        from .codegen.simd_kernel_features import DisableReduction, EnableReduction

        global _inductor_triton_kernel_to_post_grad_node_info
        global _inductor_kernel_stack_trace
        global _inductor_kernel_provenance_debug_handle
        global _inductor_kernel_extern_info

        _inductor_kernel_provenance_debug_handle += 1
        stack_traces: list[str] = []
        kernel_name = f"{kernel_name}:{_inductor_kernel_provenance_debug_handle}"
        if is_extern:
            if not isinstance(node_schedule, ExternKernel):
                raise AssertionError(
                    f"expected ExternKernel, got {type(node_schedule)}"
                )
            extern_info = _inductor_kernel_extern_info.setdefault(
                kernel_name, _KernelExternInfo()
            )
            extern_info.is_extern = True
            curr_node_info = _inductor_triton_kernel_to_post_grad_node_info.setdefault(
                kernel_name, []
            )
            # 'origins' on IR nodes gives what FX IR nodes contributed to any given fused kernel.
            # "origin_node" is more precise and says that the contents of this node corresponds
            # EXACTLY to the output of a particular FX node, but it's not always available
            if node_schedule.origin_node:
                origin_node_name = node_schedule.origin_node.name
                if origin_node_name not in curr_node_info:
                    curr_node_info.append(origin_node_name)
            else:
                curr_node_info.extend(
                    origin.name
                    for origin in node_schedule.origins
                    if origin.name not in curr_node_info
                )
            # Extract extern_semantic_key if stamped by a transform pass (e.g., FP8)
            if node_schedule.origin_node and "extern_semantic_key" in getattr(
                node_schedule.origin_node, "meta", {}
            ):
                extern_info.semantic_key = node_schedule.origin_node.meta[
                    "extern_semantic_key"
                ]
            else:
                for origin in node_schedule.origins:
                    if "extern_semantic_key" in getattr(origin, "meta", {}):
                        extern_info.semantic_key = origin.meta["extern_semantic_key"]
                        break
            # Extract input/output shapes and dtypes for extern ops
            try:
                input_shapes: list[list[int | str]] = []
                input_dtypes: list[str] = []
                for inp in tree_leaves(node_schedule.inputs):
                    if isinstance(inp, ir.IRNode):
                        metadata = _get_tensor_metadata(inp)
                        if metadata is None:
                            continue
                        shape, dtype = metadata
                        input_shapes.append(shape)
                        input_dtypes.append(dtype)

                shape_metadata: dict[str, Any] = {
                    "input_shapes": input_shapes,
                    "input_dtypes": input_dtypes,
                }
                output_metadata = _get_tensor_metadata(node_schedule)
                if output_metadata is not None:
                    output_shape, output_dtype = output_metadata
                    shape_metadata.update(
                        {
                            "output_shape": output_shape,
                            "output_dtype": output_dtype,
                        }
                    )
                extern_info.shapes = shape_metadata
            except Exception as e:
                log.debug("Shape extraction failed for %s: %s", kernel_name, e)
            stack_traces = list(node_schedule.get_stack_traces())
        else:
            if not isinstance(node_schedule, list):
                raise AssertionError(f"expected list, got {type(node_schedule)}")
            stack_traces_set: OrderedSet[str] = OrderedSet()
            for snode in node_schedule:
                if snode not in (EnableReduction, DisableReduction):
                    if snode.node is not None:
                        curr_node_info = (
                            _inductor_triton_kernel_to_post_grad_node_info.setdefault(
                                kernel_name, []
                            )
                        )
                        # pyrefly: ignore [missing-attribute]
                        stack_traces_set.update(snode.node.get_stack_traces())
                        curr_node_info.extend(
                            origin.name
                            # pyrefly: ignore [missing-attribute]
                            for origin in snode.node.origins
                            if origin.name not in curr_node_info
                        )
            stack_traces = list(stack_traces_set)
        _inductor_kernel_stack_trace.setdefault(kernel_name, []).extend(stack_traces)
        return _inductor_kernel_provenance_debug_handle
    except Exception as e:
        # Since this is just debugging, it should never interfere with regular
        # program execution, so we use this try-except to guard against any error
        signpost_event(
            "inductor",
            "provenance_tracking_error",
            {
                "function": "set_kernel_post_grad_provenance_tracing",
                "error_msg": str(e),
                "stack_trace": traceback.format_exc(),
            },
        )
        return None


def save_args_for_compile_fx_inner(*args: Any, **kwargs: Any) -> None:
    # [FUNCTION DOC] Pickle the arguments of a compile_fx_inner call into a numbered file under inductor_saved_args in the temp directory
    # (creating the folder), after replacing tensors and symbolic values and graph module metadata with picklable holders.
    # At debug log level prints the replay instructions. Side effects: file I/O, print.
    """
    This function is used to save arguments for a compile_fx_inner function call
    to the file system.  Later on one can replay the compile_fx_inner call
    with the saved arguments using load_args_and_run_compile_fx_inner.
    """

    folder = os.path.join(tempfile.gettempdir(), "inductor_saved_args")
    if not os.path.exists(folder):
        os.mkdir(folder)

    def python_value(value: Any) -> int | float | bool | None:
        # [FUNCTION DOC] HELPER: Convert a numeric or symbolic value to a plain int, float or bool; None stays None.
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        # SymPy numbers expose is_integer as a tri-state property.
        if getattr(value, "is_integer", None) is False:
            return float(value)
        try:
            return int(value)
        except (TypeError, ValueError):
            return float(value)

    def handle_sym_expr(x: Any, pytype: str) -> SymExprMetadataHolder:
        # [FUNCTION DOC] HELPER: Build a SymExprMetadataHolder for a symbolic scalar, capturing its expression, hint and backed hints of its free symbols.
        node = x.node
        shape_env = node.shape_env
        symbol_hints: dict[str, int | float | bool] = {}
        if shape_env is not None:
            for symbol in node.expr.free_symbols:
                hint = shape_env.backed_var_to_val.get(symbol)
                if hint is not None:
                    python_hint = python_value(hint)
                    if python_hint is not None:
                        symbol_hints[str(symbol)] = python_hint
        return SymExprMetadataHolder(
            pytype,
            node.expr,
            python_value(node.hint),
            symbol_hints,
        )

    def handle_tensor(x: Any) -> Any:
        # [FUNCTION DOC] HELPER: Replace a value with a picklable form: graph modules are deep-copied with node metadata converted; tensors become
        # TensorMetadataHolder; symbolic int, float, bool become SymExprMetadataHolder; others are returned unchanged.
        """
        Pickle FakeTensor/SymInt will result in errors like:
        AttributeError: Can't pickle local object 'WeakValueDictionary.__init__.<locals>.remove'

        Convert tensors and symbolic values to metadata. This may also make
        pickle faster.
        """
        if isinstance(x, GraphModule):
            gm = copy.deepcopy(x)
            for node in gm.graph.nodes:
                node.meta = tree_map(handle_tensor, node.meta)
            return gm
        if isinstance(x, torch.Tensor):
            return TensorMetadataHolder(
                tree_map(handle_tensor, _extract_tensor_metadata(x)), x.device
            )
        elif isinstance(x, torch.SymInt):
            return handle_sym_expr(x, "int")
        elif isinstance(x, torch.SymFloat):
            return handle_sym_expr(x, "float")
        elif isinstance(x, torch.SymBool):
            return handle_sym_expr(x, "bool")
        else:
            return x

    args_to_save, kwargs_to_save = tree_map(handle_tensor, (args, kwargs))

    fn_name = "compile_fx_inner"
    path = f"{folder}/{fn_name}_{next(save_args_cnt)}.pkl"
    with open(path, "wb") as f:
        pickle.dump((args_to_save, kwargs_to_save), f)

    if log.isEnabledFor(logging.DEBUG):
        message = f"""
Arguments for a compile_fx_inner call is saved to {path}. To replay the call,
run the following:

from torch._inductor.debug import load_args_and_run_compile_fx_inner
load_args_and_run_compile_fx_inner({path!r})
        """
        # call print rather than log.debug. log.debug will print message
        # prefix for each line which makes the code snippet harder to be
        # copied.
        # Not a big deal since the code is already been guarded by checking
        # the log level.
        print(message)


def load_args_and_run_compile_fx_inner(path: str) -> Any:
    # [FUNCTION DOC] Load arguments saved by save_args_for_compile_fx_inner from a pickle file, rebuild fake tensors (random strided) and symbolic values
    # under a fake mode with a fresh shape env, and run compile_fx_inner with save_args disabled. Returns its result.
    from torch._dynamo.source import ConstantSource
    from torch._inductor.compile_fx import compile_fx_inner
    from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv

    with open(path, "rb") as f:
        args, kwargs = pickle.load(f)

    shape_env = ShapeEnv()
    symbol_cache: dict[str, Any] = {}

    def restore_sym_expr(x: SymExprMetadataHolder) -> Any:
        # [FUNCTION DOC] HELPER: Recreate a symbolic int, float or bool from a SymExprMetadataHolder using the shared shape env and symbol cache
        # (backed symbols from hints, otherwise unbacked). Raises AssertionError for an unknown type.
        expr = x.expr
        if isinstance(expr, str):
            import sympy

            import torch.utils._sympy.functions as sympy_functions

            expr = sympy.sympify(expr, locals=sympy_functions.__dict__)
        replacements = {}
        for symbol in expr.free_symbols:
            symbol_name = str(symbol)
            if symbol_name not in symbol_cache:
                hint = x.symbol_hints.get(symbol_name)
                if hint is None:
                    if x.hint is not None and expr == symbol:
                        hint = x.hint
                    else:
                        symbol_cache[symbol_name] = shape_env.create_unbacked_symint(
                            ConstantSource(symbol_name)
                        ).node.expr
                        replacements[symbol] = symbol_cache[symbol_name]
                        continue
                symbol_cache[symbol_name] = shape_env.create_symbol(
                    hint,
                    ConstantSource(symbol_name),
                    dynamic_dim=DimDynamic.DYNAMIC,
                    do_not_specialize_zero_one=True,
                )
            replacements[symbol] = symbol_cache[symbol_name]
        expr = expr.xreplace(replacements)
        if x.pytype == "int":
            return shape_env.create_symintnode(expr, hint=x.hint)
        elif x.pytype == "float":
            return shape_env.create_symfloatnode(expr, hint=x.hint)
        elif x.pytype == "bool":
            return shape_env.create_symboolnode(expr)
        else:
            raise AssertionError(f"Unexpected symbolic expression type: {x.pytype}")

    def handle_tensor(x: Any) -> Any:
        # [FUNCTION DOC] HELPER: Inverse of the save-time conversion: graph module metadata is restored in place, tensor holders become random
        # strided tensors, symbolic holders become symbolic values; others are unchanged.
        if isinstance(x, GraphModule):
            for node in x.graph.nodes:
                node.meta = tree_map(handle_tensor, node.meta)
            return x
        elif isinstance(x, TensorMetadataHolder):
            tensor_metadata = tree_map(handle_tensor, x.tensor_metadata)
            return torch._dynamo.testing.rand_strided(
                tensor_metadata.shape,
                tensor_metadata.stride,
                tensor_metadata.dtype,
                x.device,
            )
        elif isinstance(x, SymExprMetadataHolder):
            return restore_sym_expr(x)
        else:
            return x

    fake_mode = torch._subclasses.FakeTensorMode(
        allow_non_fake_inputs=True, shape_env=shape_env
    )
    with fake_mode, config.patch("save_args", False):
        args, kwargs = tree_map(handle_tensor, (args, kwargs))
        return compile_fx_inner(*args, **kwargs)


def aot_inductor_minifier_wrapper(
    func: Callable[..., str],
    exported_program: torch.export.ExportedProgram,
    *,
    inductor_configs: dict[str, Any],
    package_path: FileLike | None = None,
) -> str:
    # [FUNCTION DOC] Wrapper around an AOT Inductor compile function that supports the minifier. With dump_aoti_minifier: repro level 3 dumps the
    # module first; level 4 first checks accuracy on flattened inputs. Runs func with load_and_run set to the minifier flag and returns its
    # string result. On AccuracyError dumps a minify accuracy repro and re-raises; on other errors with the minifier on, dumps a repro
    # (run for level 1, else minify) and re-raises. Raises AssertionError if the module is not a GraphModule.
    from torch._dynamo.debug_utils import AccuracyError
    from torch._dynamo.repro.aoti import dump_to_minify
    from torch._inductor import config
    from torch._inductor.compile_fx import _aoti_flatten_inputs

    use_minifier = config.aot_inductor.dump_aoti_minifier

    gm = exported_program.module(check_guards=False)
    if not isinstance(gm, torch.fx.GraphModule):
        raise AssertionError(f"expected torch.fx.GraphModule, got {type(gm)}")

    args, kwargs = exported_program.example_inputs

    try:
        if use_minifier and config.aot_inductor.repro_level == 3:
            # Always dump the original module in case we have segfaults
            dump_to_minify(
                exported_program,
                "aot_inductor",
                options=inductor_configs,
            )
        if use_minifier and config.aot_inductor.repro_level == 4:
            # Check for accuracy
            # We will first flatten the inputs before compiling and checking for accuracy.
            # This is ok because we will flatten the inputs in the minifier anyway.
            gm_copy = copy.deepcopy(gm)
            example_inputs_copy = copy.deepcopy(exported_program.example_inputs)
            config_copy = copy.deepcopy(inductor_configs)
            flat_example_inputs, config_copy = _aoti_flatten_inputs(
                gm_copy,
                example_inputs_copy[0],
                example_inputs_copy[1],
                options=config_copy,
            )
            tuple_inputs = tuple(flat_example_inputs)
            flattened_ep = torch.export.export(gm_copy, tuple_inputs, strict=False)
            func(
                flattened_ep.module(check_guards=False),
                tuple_inputs,
                inductor_configs=config_copy,
                package_path=package_path,
                load_and_run=True,
                check_accuracy="accuracy",
            )

        return func(
            gm,
            args,
            kwargs,
            inductor_configs=inductor_configs,
            package_path=package_path,
            load_and_run=use_minifier,
        )
    except AccuracyError as e:
        dump_to_minify(
            exported_program,
            "aot_inductor_accuracy",
            command="minify",
            options=inductor_configs,
        )
        log.warning("Accuracy failed")
        raise e
    except Exception as e:
        if use_minifier:
            command = "minify"

            if config.aot_inductor.repro_level == 1:
                command = "run"

            dump_to_minify(
                exported_program,
                "aot_inductor",
                command=command,
                options=inductor_configs,
            )
        raise e
