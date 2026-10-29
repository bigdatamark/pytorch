# mypy: allow-untyped-defs
# [MODULE DOC] HELPER: Debugging tools that detect Python reference cycles keeping CUDA tensors
# alive and render them. Hooks the garbage collector to capture collectable garbage, builds a
# graph of objects with labelled reference edges (adapted from refcycle), and emits Graphviz
# text or a self-contained HTML page showing allocation stacks on hover. Main entry points:
# warn_tensor_cycles, observe_tensor_cycles, observe_garbage, create_graph, to_dot, to_html.
# Import-time side effects: creates a module logger and computes the cell type.
import gc
import sys
from typing import Any, NamedTuple
import types
import weakref
import json
from tempfile import NamedTemporaryFile
import torch
from torch.cuda._memory_viz import _frames_fmt, _block_extra
import atexit
import logging
logger = logging.getLogger(__name__)

def observe_garbage(observer):
    # [FUNCTION DOC] Install a garbage collector callback that, after each collection, passes
    # the list of saved garbage objects to observer, then frees them. While armed it turns on
    # the save-all debug flag at collection start, and uses a profile hook to run the report
    # outside the collector. Warns if CUDA memory changed while freeing. Registers an exit
    # handler that disables the callback. Returns a function that removes the callback.
    enabled = True

    def disable() -> None:
        # [FUNCTION DOC] HELPER: Exit handler that disables the callback so it does nothing
        # during interpreter shutdown.
        # when GC runs during exit, things like `sys` will already be unloaded
        # so we have to disable the callback to avoid hitting errors.
        nonlocal enabled
        enabled = False
    atexit.register(disable)

    def gc_callback(phase, info) -> None:
        # [FUNCTION DOC] HELPER: Collector hook. On the start phase, enables save-all
        # debugging. On the stop phase, installs a temporary profile function that runs the
        # observer on the next call event, restoring the previous profile function.
        nonlocal enabled
        if not enabled:
            return
        if phase == "start":
            gc.set_debug(gc.DEBUG_SAVEALL)
        elif phase == "stop":
            orig_trace = sys.getprofile()
            self_return = [False]

            def do_collect(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Profile hook; ignores its first call, then restores
                # the original hook, forces a full collection if needed, gives the saved
                # garbage to the observer, clears it, resets debug flags, collects again and
                # logs a warning if CUDA memory changed. Chains to the original hook.
                nonlocal enabled
                if not self_return[0]:
                    self_return[0] = True
                else:
                    sys.setprofile(orig_trace)
                    enabled = False
                    try:
                        # things in gc.garbage have survived a collection
                        # so to free them we have to collect a generation greater than them
                        # but that might _also_ free other stuff and we don't want to miss
                        # that stuff. So we have to now force gc at the highest level here,
                        # report all of what we found, _then_ we can free it up.
                        if info['generation'] != 2:
                            gc.collect()
                        observer(gc.garbage)
                        gc.garbage.clear()
                        # we have to re-run GC to clean up the cycles
                        # we saved from before.
                        gc.set_debug(0)
                        before = torch.cuda.memory_allocated()
                        gc.collect()
                        after = torch.cuda.memory_allocated()
                        if before != after:
                            logger.warning("CUDA Memory changed during GC, %d bytes freed.", before - after)
                    finally:
                        enabled = True
                if orig_trace is not None:
                    return orig_trace(*args, **kwargs)
            sys.setprofile(do_collect)

    gc.callbacks.append(gc_callback)

    # provide a way to disarm the callback
    def remove() -> None:
        # [FUNCTION DOC] HELPER: Unregister the collector callback; raises ValueError if it
        # was already removed.
        gc.callbacks.remove(gc_callback)
    return remove

# Function to visualize cycles adapted from refcycle:
# Copyright 2013 Mark Dickinson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def _get_cell_type():
    # [FUNCTION DOC] HELPER: Return the closure cell type by building a throwaway closure.
    def f(x=None):
        # [FUNCTION DOC] HELPER: Return a closure capturing x so a cell object exists.
        return lambda: x
    return type(f().__closure__[0])

CellType = _get_cell_type()

def annotated_references(obj):
    # [FUNCTION DOC] Describe what obj refers to. Returns a mapping from the identity number
    # of each referent to a list of text labels (attribute names, indexes, keys, locals),
    # since several edges may reach one referent. Uses the object's type hierarchy to pick
    # extraction rules for sequences, mappings, sets, functions, frames, cells, bound
    # methods, weak references and descriptors, plus dict and class attributes.
    """
    Return known information about references held by the given object.

    Returns a mapping from referents to lists of descriptions.  Note that there
    may be more than one edge leading to any particular referent; hence the
    need for a list.  Descriptions are currently strings.

    """
    references: dict[int, list[str]] = {}

    def add_reference(name, obj) -> None:
        # [FUNCTION DOC] HELPER: Record a labelled edge to the given object in the result.
        references.setdefault(id(obj), []).append(name)

    def add_attrs(*attrs) -> None:
        # [FUNCTION DOC] HELPER: For each named attribute that the outer object has, record
        # an edge labelled with that name.
        for attr in attrs:
            if hasattr(obj, attr):
                add_reference(attr, getattr(obj, attr))

    def add_cell_references() -> None:
        # [FUNCTION DOC] HELPER: Record the cell contents edge; empty cells are skipped.
        try:
            add_attrs("cell_contents")
        except ValueError:
            # if cell_contents is empty,
            # accessing it raises ValueError
            # in this case there is no object to
            # annotate
            pass

    def add_function_references() -> None:
        # [FUNCTION DOC] HELPER: Record edges for function attributes such as defaults,
        # closure, globals, code, name, module, annotations and keyword defaults.
        # KNOWN ISSUE: a missing comma after the doc entry concatenates it with the qualified
        # name entry into one nonexistent attribute, so neither is ever recorded.
        add_attrs("__defaults__",
                  "__closure__",
                  "__globals__",
                  "__code__",
                  "__name__",
                  "__module__",
                  "__doc__"
                  "__qualname__",
                  "__annotations__",
                  "__kwdefaults__")


    def add_sequence_references() -> None:
        # [FUNCTION DOC] HELPER: Record an edge labelled with the position for each item.
        for position, item in enumerate(obj):
            add_reference(f"[{position}]", item)

    def add_dict_references() -> None:
        # [FUNCTION DOC] HELPER: For each entry, record an edge to the key labelled key and
        # an edge to the value labelled with the key's representation.
        for key, value in obj.items():
            add_reference("key", key)
            add_reference(f"[{repr(key)}]", value)

    def add_set_references() -> None:
        # [FUNCTION DOC] HELPER: Record an edge labelled element for each member.
        for elt in obj:
            add_reference("element", elt)

    def add_bound_method_references() -> None:
        # [FUNCTION DOC] HELPER: Record edges to the bound instance and underlying function.
        add_attrs("__self__", "__func__", "im_class")

    def add_weakref_references() -> None:
        # [FUNCTION DOC] HELPER: For plain weak references only, if the garbage collector
        # reports exactly one referent, label it as the callback.
        # A reference without a callback has no referents here, so nothing is recorded.
        # For subclasses of weakref, we can't reliably distinguish the
        # callback (if any) from other attributes.
        if type(obj) is weakref.ref:
            referents = gc.get_referents(obj)
            if len(referents) == 1:
                target = referents[0]
                add_reference("__callback__", target)


    def add_frame_references() -> None:
        # [FUNCTION DOC] HELPER: Record edges for frame attributes (caller, code, builtins,
        # globals, trace, locals) and, if the locals are a real dict, one edge per local.
        f_locals = obj.f_locals
        add_attrs("f_back", "f_code", "f_builtins", "f_globals", "f_trace", "f_locals")
        # Some badly-behaved code replaces the f_locals dict with
        # something that doesn't support the full dict interface.  So we
        # only continue with the annotation if f_locals is a Python dict.
        if type(f_locals) is dict:
            for name, local in obj.f_locals.items():
                add_reference(f"local {name}", local)

    def add_getset_descriptor_references() -> None:
        # [FUNCTION DOC] HELPER: Record edges to the owning class, name and doc of a
        # descriptor.
        add_attrs("__objclass__", "__name__", "__doc__")

    type_based_references = {
        tuple: add_sequence_references,
        list: add_sequence_references,
        dict: add_dict_references,
        set: add_set_references,
        frozenset: add_set_references,
        types.FunctionType: add_function_references,
        types.FrameType: add_frame_references,
        CellType: add_cell_references,
        types.MethodType: add_bound_method_references,
        weakref.ref: add_weakref_references,
        types.GetSetDescriptorType: add_getset_descriptor_references,
    }

    for type_ in type(obj).__mro__:
        if type_ in type_based_references:
            type_based_references[type_]()

    add_attrs("__dict__", "__class__")
    if isinstance(obj, type):
        add_attrs("__mro__")

    return references

###############################################################################
# Object annotations.


BASE_TYPES = (int, float, complex, type(None), str, bytes)
FRAME_FILENAME_LIMIT = 32

def object_annotation(obj):
    # [FUNCTION DOC] Build a short, possibly multi-line label for a graph node describing obj:
    # the representation for basic types, kind plus name for functions, methods, modules,
    # types and frames (long file names truncated), element summaries for lists and tuples,
    # size for dicts, liveness for weak references, and module-qualified type name (with
    # shape for CUDA tensors) otherwise.
    """
    Return a string to be used for Graphviz nodes.

    The string should be short but as informative as possible.
    """

    def format_sequence(obj):
        # [FUNCTION DOC] HELPER: Summarize up to the first eight items (representation for
        # basic types, type name otherwise), with a count of the remainder if longer.
        body = ','.join(repr(x) if isinstance(x, BASE_TYPES) else type(x).__name__ for x in obj[:8])
        if len(obj) > 8:
            body = f'{body}, ...{len(obj) - 8}'
        return body

    # For basic types, use the repr.
    if isinstance(obj, BASE_TYPES):
        return repr(obj)
    if type(obj).__name__ == 'function':
        return f"function\n{obj.__name__}"
    elif isinstance(obj, types.MethodType):
        try:
            func_name = obj.__func__.__qualname__
        except AttributeError:
            func_name = "<anonymous>"
        return f"instancemethod\n{func_name}"
    elif isinstance(obj, list):
        return f"[{format_sequence(obj)}]"
    elif isinstance(obj, tuple):
        return f"({format_sequence(obj)})"
    elif isinstance(obj, dict):
        return f"dict[{len(obj)}]"
    elif isinstance(obj, types.ModuleType):
        return f"module\n{obj.__name__}"
    elif isinstance(obj, type):
        return f"type\n{obj.__name__}"
    elif isinstance(obj, weakref.ref):
        referent = obj()
        if referent is None:
            return "weakref (dead referent)"
        else:
            return f"weakref to id 0x{id(referent):x}"
    elif isinstance(obj, types.FrameType):
        filename = obj.f_code.co_filename
        if len(filename) > FRAME_FILENAME_LIMIT:
            filename = "..." + filename[-(FRAME_FILENAME_LIMIT - 3):]
        return f"frame\n{filename}:{obj.f_lineno}"
    elif is_cuda_tensor(obj):
        return f"object\n{type(obj).__module__}.{type(obj).__name__} ({obj.shape})"
    else:
        return f"object\n{type(obj).__module__}.{type(obj).__name__}"



class Node(NamedTuple):
    # [CLASS DOC] One graph node. Fields: label text, optional allocation context text, root
    # flag (true when the object matched the filter, such as a CUDA tensor), and a mutable
    # list of (edge label, target node index) pairs for outgoing references.
    # (The field name referrents is misspelled but used consistently.)
    label: str
    context: str | None
    root: bool
    referrents: list[tuple[str, int]]

def create_graph(objects, *, context=None, filter=None):
    # [FUNCTION DOC] Build a list of Node from a collection of objects (weak proxies are
    # dropped). Edges come from the collector's referents among the given objects, labelled
    # via annotated_references ("?" when unknown). Roots are objects passing filter (default:
    # CUDA tensors); context maps an object to allocation text (default: derived from a CUDA
    # memory snapshot, which is taken on the call). Only nodes that can reach a root are kept,
    # with indexes renumbered. Returns the filtered node list.
    if context is None:
        context = cuda_allocation_context()
    if filter is None:
        filter = is_cuda_tensor

    objects = [obj for obj in objects if not isinstance(obj, weakref.ProxyTypes)]
    nodes = [Node(object_annotation(obj), context(obj), filter(obj), []) for obj in objects]
    node_referrers: list[list[int]] = [[] for obj in objects]

    id_to_node = {id(obj): i for i, obj in enumerate(objects)}
    for obj in objects:
        fidx = id_to_node[id(obj)]
        f = nodes[fidx]
        references = annotated_references(obj)
        for referrent in gc.get_referents(obj):
            rid = id(referrent)
            tidx = id_to_node.get(rid)
            if tidx is None:
                continue
            labels = references.get(rid, ["?"])
            node_referrers[tidx].append(fidx)
            for label in labels:
                f.referrents.append((label, tidx))

    to_search = [i for i, n in enumerate(nodes) if n.root]
    to_keep = set()
    while to_search:
        idx = to_search.pop()
        if idx in to_keep:
            continue
        to_keep.add(idx)
        referrers = node_referrers[idx]
        to_search.extend(referrers)
    id_to_filtered_id: dict[int, int] = {}
    filtered: list[Any] = []
    for i, n in enumerate(nodes):
        if i in to_keep:
            id_to_filtered_id[i] = len(id_to_filtered_id)
            filtered.append(n)
    for n in filtered:
        n.referrents[:] = [(label, id_to_filtered_id[idx])
                           for (label, idx) in n.referrents
                           if idx in id_to_filtered_id]
    return filtered

def escape(n):
    # [FUNCTION DOC] HELPER: Return n as a quoted, escaped JSON string literal.
    return json.dumps(n)


def is_cuda_tensor(obj):
    # [FUNCTION DOC] True if obj is a real (non-fake) tensor on a CUDA device.
    return (
        isinstance(obj, torch.Tensor) and
        obj.device.type == "cuda" and
        not torch._subclasses.fake_tensor.is_fake_tensor(obj)
    )

def cuda_allocation_context():
    # [FUNCTION DOC] Take a snapshot of the CUDA caching allocator and return a function that
    # maps a CUDA tensor to the formatted allocation stack of its storage (or None). Only
    # active allocated blocks are indexed by start address; stacks are available only if
    # memory history recording was enabled.
    snapshot = torch.cuda.memory._snapshot()
    addr_to_frame = {}
    for seg in snapshot['segments']:
        addr = seg['address']
        for blk in seg['blocks']:
            if blk['state'] == 'active_allocated':
                frames, _real_size = _block_extra(blk)
                addr_to_frame[addr] = frames
            addr += blk['size']

    def object_context(obj):
        # [FUNCTION DOC] HELPER: Return the allocation stack text for a CUDA tensor's storage
        # address, or None for other objects or unknown addresses.
        if is_cuda_tensor(obj):
            addr = obj.untyped_storage().data_ptr()
            frames = addr_to_frame.get(addr)
            if frames is not None:
                return '\n'.join(_frames_fmt(frames, full_filename=True))
        return None
    return object_context

def to_dot(nodes):
    # [FUNCTION DOC] Render nodes as Graphviz directed graph text, left to right. Root nodes
    # are drawn red; each edge carries its label. Returns the text.
    lines = ["digraph GraphName {", "node [shape=rect];", 'rankdir=LR;']
    for i, n in enumerate(nodes):
        lines.append(f'{i} [label={escape(n.label)}, color={"red" if n.root else "black"}];')

    for i, f in enumerate(nodes):
        for label, j in f.referrents:
            lines.append(f'{i} -> {j} [label = {escape(label)}]')
    lines.append("}\n")
    return '\n'.join(lines)

_template = """
<!DOCTYPE html>
<html>
<head>
  <style>
    body {
      margin: 0;
      padding: 0;
      overflow: hidden;
    }

    #container {
      display: flex;
      flex-direction: column;
      height: 100vh;
    }

    #main {
      flex: 2;
      height: 60vh;
      overflow: clip;
    }

    #preContainer {
      flex: 1;
      height: 40vh;
      overflow: auto;
    }

    pre {
      margin: 0;
      padding: 10px;
    }
  </style>
</head>
<body>
  <div id="container">
    <div id="main">
    </div>
    <div id="preContainer">
      <pre id="stacktrace">Mouse over tensor objects to see where they were allocated.</pre>
    </div>
  </div>
<script src='https://cdnjs.cloudflare.com/ajax/libs/viz.js/1.8.0/viz-lite.js'></script>
<script>
let dot = $DOT
let image = Viz(dot, {format: 'svg', 'totalMemory': 1024*1024*1024});
let main = document.getElementById('main')
main.innerHTML = image
let svg = main.firstElementChild
// Panning and zooming logic
let isPanning = false;
let startX, startY;
let viewBox = { x: 0, y: 0, width: parseFloat(svg.getAttribute('width')), height: parseFloat(svg.getAttribute('height')) };
svg.removeAttribute('width');
svg.removeAttribute('height');
function updateViewBox() {
    svg.setAttribute('viewBox', `${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`);
}
updateViewBox()
svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
svg.addEventListener('mousedown', function(e) {
    isPanning = true;
    startX = e.clientX;
    startY = e.clientY;
});
svg.addEventListener('mousemove', function(e) {
    if (!isPanning) return;
    const dx = (e.clientX - startX) * (viewBox.width / svg.clientWidth);
    const dy = (e.clientY - startY) * (viewBox.height / svg.clientHeight);
    viewBox.x -= dx;
    viewBox.y -= dy;
    startX = e.clientX;
    startY = e.clientY;
    updateViewBox();
});
svg.addEventListener('mouseup', function() {
    isPanning = false;
});
svg.addEventListener('mouseleave', function() {
    isPanning = false;
});
svg.addEventListener('wheel', function(e) {
    e.preventDefault();
    const zoomFactor = 0.1;
    const zoomAmount = e.deltaY > 0 ? 1 + zoomFactor : 1 - zoomFactor;
    // Calculate mouse position relative to the SVG
    const rect = svg.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const mouseXRel = mouseX / svg.clientWidth;
    const mouseYRel = mouseY / svg.clientHeight;
    // Adjust viewBox to zoom around the mouse position
    const newWidth = viewBox.width * zoomAmount;
    const newHeight = viewBox.height * zoomAmount;
    viewBox.x += (viewBox.width - newWidth) * mouseXRel;
    viewBox.y += (viewBox.height - newHeight) * mouseYRel;
    viewBox.width = newWidth;
    viewBox.height = newHeight;
    updateViewBox();
});
$LISTENERS
</script>
</body>
</html>
"""
_listener_template = """
document.getElementById('node{id}').addEventListener('mouseover', function(event) {{
  document.getElementById("stacktrace").textContent = {stack}
}})
"""
def to_html(nodes):
    # [FUNCTION DOC] Produce a standalone HTML page from nodes: the Graphviz text is embedded
    # in the template, with a hover listener per node that has allocation context. The page
    # loads a rendering script from a CDN at view time. Returns the HTML string.
    # KNOWN ISSUE: listeners target element ids numbered from one, relying on the renderer
    # naming nodes that way; the graph text numbers nodes from zero.
    listeners = []
    for i, n in enumerate(nodes):
        if n.context is None:
            continue
        s = _listener_template.format(id=str(i + 1), stack=escape(f'{n.label}:\n{n.context}'))
        listeners.append(s)
    dot = to_dot(nodes)
    return _template.replace('$DOT', repr(dot)).replace('$LISTENERS', '\n'.join(listeners))

def observe_tensor_cycles(callback):
    # [FUNCTION DOC] Start CUDA memory history recording (global side effect, up to 100000
    # entries) and watch the collector; whenever garbage containing CUDA tensors is found,
    # call callback with the HTML visualization. Returns the function that stops watching.
    torch.cuda.memory._record_memory_history(max_entries=100000)

    def observer(garbage) -> None:
        # [FUNCTION DOC] HELPER: If garbage is non-empty and holds a CUDA tensor, build the
        # graph and pass its HTML to callback; otherwise log that none were found.
        if garbage:
            if not any(is_cuda_tensor(obj) for obj in garbage):
                logger.info("No CUDA Tensors found in garbage")
                return
            callback(to_html(create_graph(garbage)))
    return observe_garbage(observer)


def warn_tensor_cycles():
    # [FUNCTION DOC] Enable cycle reporting for CUDA tensors: each time such a cycle is
    # collected, write an HTML visualization to a temporary file and log a warning with its
    # path. Starts memory history recording. Returns the function that disables it.
    """
    Install a warning that reports whenever a cycle that is holding CUDA memory is observed.

    The warning produces an .html file that visualizes the cycle,
    and links it to the stack frame that allocated the CUDA tensor.

    Reference cycles are freed by the cycle collector rather than being cleaned up
    when the objects in the cycle first become unreachable. If a cycle points to a tensor,
    the CUDA memory for that tensor will not be freed until garbage collection runs.
    Accumulation of CUDA allocations can lead to out of memory errors (OOMs), as well as
    non-deterministic allocation behavior which is harder to debug.
    """
    logger.info("Watching Python reference cycles for CUDA Tensors.")

    def write_and_log(html) -> None:
        # [FUNCTION DOC] HELPER: Write html to a named temporary file and log its path.
        # KNOWN ISSUE: the temporary file is deleted when the block exits, so the logged path
        # no longer exists by the time anyone reads the warning.
        with NamedTemporaryFile('w', suffix='.html') as f:
            f.write(html)
            logger.warning('Reference cycle includes a CUDA Tensor see visualization of cycle %s', f.name)
    return observe_tensor_cycles(write_and_log)
