#!/usr/bin/env python3
# [MODULE DOC] Command line tool (run as a module) that post-processes a profiler trace JSON (plain or gzip)
# using a pickled mapping of graph node id to kernel annotations. Contents: annotate_trace (merge annotations
# into kernel event args, reassign streams, drop noise events, fix metadata), two private timestamp/stream
# cleanup passes, load_trace/save_trace, annotation pickle discovery, and main (argparse entry point).
# Import-time side effects: none; main runs only under the script guard and prints progress, may exit 1.
"""Post-process a profiler trace to add CUDA graph kernel annotations.

Reads a profiler trace (gzipped or plain JSON) and a kernel annotations
pickle, matches kernel events by their graph node id, and writes an
annotated trace with the annotation fields added to each kernel event's
args (displayed alongside grid/block size in trace viewers).

The annotations pickle is auto-discovered from the trace file's parent
directory (one level up, matching the rank from the trace filename).

Usage:
    python -m torch.cuda._annotate_cuda_graph_trace <trace_file> [-a <annotations_pkl>] [-o <output_file>]

Examples:
    # Auto-discover annotations pickle from trace location
    python -m torch.cuda._annotate_cuda_graph_trace \\
        traces/step_000000000014/000000.*.pt.trace.json.gz

    # Explicit annotations pickle
    python -m torch.cuda._annotate_cuda_graph_trace trace.json.gz -a annotations.pkl
"""

import argparse
import gzip
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


_WORK_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


def _move_overlapping_to_stream(
    trace: dict, default_stream: int = 7, overlap_stream: int = 8
) -> int:
    # [FUNCTION DOC] HELPER: Move graphed kernel events (nonzero graph node id) on default_stream that start before
    # the previous such kernel ends onto overlap_stream, updating both tid and the stream arg, mutating the trace.
    # Events are processed in timestamp order; returns the count moved. Missing args are treated as not graphed.
    """Move graphed kernels that overlap with their predecessor to a separate stream.

    Perfetto cannot display overlapping (non-nested) events on the same
    stream -- they get hidden.  This pass detects graphed kernel events on
    *default_stream* whose start timestamp falls before the previous
    kernel's end, and moves them to *overlap_stream* so they're visible.

    Returns the number of events moved.
    """
    graphed_on_default = [
        e
        for e in trace["traceEvents"]
        if e.get("cat") == "kernel"
        and e.get("tid") == default_stream
        and e.get("args", {}).get("graph node id", 0) != 0
    ]
    graphed_on_default.sort(key=lambda e: e["ts"])

    moved = 0
    prev_end = 0.0
    for event in graphed_on_default:
        ts = event["ts"]
        dur = event.get("dur", 0)
        if ts < prev_end:
            event["tid"] = overlap_stream
            event.get("args", {})["stream"] = overlap_stream
            moved += 1
        else:
            prev_end = ts + dur

    return moved


def _fix_overlapping_timestamps(trace: dict, max_adjust_us: float = 1.0) -> int:
    # [FUNCTION DOC] HELPER: For graphed kernel/memcpy/memset events grouped per stream, sort by timestamp and push a
    # start time forward to the previous event end when overlap is at most max_adjust_us; larger overlaps print
    # a warning and are left unchanged. Mutates events in place; returns the number adjusted.
    # Note: after a skipped large overlap the running end time still uses the unchanged event.
    """Clamp graphed kernel/memcpy timestamps so they don't overlap on the same stream.

    CUPTI can produce slightly overlapping timestamps for consecutive graphed
    events, causing Perfetto to hide events that sit entirely "under" their
    neighbours.  This pass sorts graphed work events per stream and ensures
    each event starts at or after the previous event's end.

    Overlaps larger than *max_adjust_us* are flagged as warnings and left
    unchanged, since they likely indicate a real issue rather than CUPTI
    timestamp jitter.

    Returns the number of events adjusted.
    """
    per_stream: dict[int, list[dict]] = defaultdict(list)
    for event in trace["traceEvents"]:
        if (
            event.get("cat") in _WORK_CATEGORIES
            and event.get("args", {}).get("graph node id", 0) != 0
        ):
            per_stream[event.get("tid")].append(event)

    adjusted = 0
    for tid, events in per_stream.items():
        events.sort(key=lambda e: e["ts"])
        prev_end = 0.0
        for event in events:
            ts = event["ts"]
            dur = event.get("dur", 0)
            if ts < prev_end:
                overlap = prev_end - ts
                if overlap > max_adjust_us:
                    print(
                        f"WARNING: large overlap {overlap:.3f}us on stream {tid} "
                        f"for {event.get('name', '?')[:60]}, skipping adjustment"
                    )
                else:
                    event["ts"] = prev_end
                    adjusted += 1
            prev_end = event["ts"] + dur

    return adjusted


def annotate_trace(
    trace: dict,
    annotations: dict[int, list[Any]],
    default_stream: int = 7,
) -> int:
    # [FUNCTION DOC] Annotate graphed events (nonzero graph node id) in the trace in place. For each, merge fields from
    # the annotation list for its id into args (legacy bare strings and the key str become name; values stringified),
    # reassign tid and stream arg to the annotated stream or default_stream, and move matching ac2g flow-finish
    # events with the same old tid and timestamp. Then remove gpu_user_annotation and ac2g finish events on streams
    # without kernel/memcpy/memset work (printing the count), drop metadata for streams with no events, and add
    # thread_name metadata for new work streams. Returns the number of events that had annotations.
    # KNOWN ISSUE: annotated count excludes graphed events lacking annotations even though they are restreamed.
    """Add annotation fields to kernel events matching the annotations dict.

    Each annotation entry is a list (from nested ``mark_kernels`` scopes).
    Fields from all annotations are merged into the event args; if multiple
    annotations define the same key, later entries in the list win.

    For graphed events (graph_node_id != 0), reassigns ``tid`` and
    ``args["stream"]`` to the stream recorded in annotations, or to
    *default_stream* if there is no annotation.  Also moves the
    corresponding ``ac2g`` flow-finish events to the new tid so that
    CPU-to-GPU correlation arrows are preserved.

    Removes ``gpu_user_annotation`` events and orphaned ``ac2g`` events
    from streams that have no kernel or memcpy events after reassignment,
    since CUPTI replicates these onto every stream during graph replay.

    Returns the number of events annotated.
    """
    # Build an index of ac2g 'f' events keyed by (tid, ts) so we can
    # move them together with the kernel events they correspond to.
    ac2g_f_index: dict[tuple, list] = {}
    for event in trace["traceEvents"]:
        if event.get("cat") == "ac2g" and event.get("ph") == "f":
            key = (event.get("tid"), event.get("ts"))
            ac2g_f_index.setdefault(key, []).append(event)

    annotated = 0
    for event in trace.get("traceEvents", []):
        args = event.get("args", {})
        graph_node_id = args.get("graph node id")
        if graph_node_id is None or graph_node_id == 0:
            continue
        stream_id = None
        if graph_node_id in annotations:
            for ann in annotations[graph_node_id]:
                if isinstance(ann, dict):
                    for key, value in ann.items():
                        # Legacy pickles wrapped bare strings as {"str": ...};
                        # the canonical key is "name".
                        args["name" if key == "str" else key] = str(value)
                    if "stream" in ann:
                        stream_id = int(ann["stream"])
                else:
                    # Legacy pickles stored bare strings unwrapped.
                    args["name"] = str(ann)
            annotated += 1

        # Reassign stream: use annotated stream if available, else default
        if stream_id is None:
            stream_id = default_stream
        old_key = (event.get("tid"), event.get("ts"))
        event["tid"] = stream_id
        args["stream"] = stream_id

        # Move the matching ac2g 'f' event(s) to the same new tid
        for ac2g_event in ac2g_f_index.get(old_key, ()):
            ac2g_event["tid"] = stream_id

    # Remove gpu_user_annotation events and ac2g flow-finish events from
    # streams that have no real kernel/memcpy/memset work -- these are
    # noise replicated by CUPTI onto every stream during graph replay.
    tids_with_work = set()
    for event in trace["traceEvents"]:
        if event.get("cat") in _WORK_CATEGORIES:
            tids_with_work.add(event.get("tid"))

    def _is_noise(event: dict) -> bool:
        # [FUNCTION DOC] HELPER: Predicate for the filter: true for annotation or ac2g flow-finish events whose tid has no work.
        cat = event.get("cat")
        if cat == "gpu_user_annotation":
            return event.get("tid") not in tids_with_work
        if cat == "ac2g" and event.get("ph") == "f":
            return event.get("tid") not in tids_with_work
        return False

    original_count = len(trace["traceEvents"])
    trace["traceEvents"] = [
        event for event in trace["traceEvents"] if not _is_noise(event)
    ]
    removed = original_count - len(trace["traceEvents"])
    if removed:
        print(f"Removed {removed} noise events from empty streams")

    # Clean up metadata: remove thread_name / thread_sort_index entries
    # for noise streams that have no real (non-metadata) events, and add
    # thread_name entries for our new annotation streams.
    all_tids_in_trace = {
        e.get("tid") for e in trace["traceEvents"] if e.get("ph") != "M"
    }
    # Find the GPU process pid from existing thread_name metadata
    gpu_pid = 0
    for event in trace["traceEvents"]:
        if (
            event.get("ph") == "M"
            and event.get("name") == "thread_name"
            and str(event.get("args", {}).get("name", "")).startswith("stream ")
        ):
            gpu_pid = event.get("pid", 0)
            break

    # Remove metadata entries for tids with no non-metadata events
    trace["traceEvents"] = [
        event
        for event in trace["traceEvents"]
        if event.get("ph") != "M" or event.get("tid") in all_tids_in_trace
    ]

    # Add thread_name metadata for new annotation tids that lack one
    existing_thread_names = {
        e.get("tid")
        for e in trace["traceEvents"]
        if e.get("ph") == "M" and e.get("name") == "thread_name"
    }
    for tid in sorted(tids_with_work - existing_thread_names):
        trace["traceEvents"].append(
            {
                "ph": "M",
                "pid": gpu_pid,
                "tid": tid,
                "name": "thread_name",
                "args": {"name": f"stream {tid}"},
            }
        )

    return annotated


def load_trace(path: Path) -> dict:
    # [FUNCTION DOC] Load and return the trace JSON dictionary from path, using gzip text mode for .gz files.
    # Raises file or JSON decoding errors from the underlying calls.
    if path.suffix == ".gz" or path.name.endswith(".json.gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    else:
        with open(path) as f:
            return json.load(f)


def save_trace(trace: dict, path: Path) -> None:
    # [FUNCTION DOC] Write the trace dictionary to path as JSON, gzip compressed if the name ends in .gz.
    # Overwrites any existing file; returns nothing.
    if path.suffix == ".gz" or path.name.endswith(".json.gz"):
        with gzip.open(path, "wt") as f:
            json.dump(trace, f)
    else:
        with open(path, "w") as f:
            json.dump(trace, f)


def _find_annotations_pkl(trace_file: Path) -> Path | None:
    # [FUNCTION DOC] HELPER: Find the annotations pickle for a trace file. Parses the leading digits of the trace file name
    # as the rank and returns the first sorted match of the rank-specific annotations pickle in the grandparent
    # directory, or None if the name has no leading digits or nothing matches.
    """Auto-discover the annotations pickle from the trace file location.

    Trace files live in e.g. ``traces/step_000000000014/000000.<id>.pt.trace.json.gz``
    where the leading digits are the rank. The pickle lives one level up:
    ``traces/kernel_annotations_rank0_*.pkl``.
    """
    match = re.match(r"^(\d+)", trace_file.name)
    if not match:
        return None
    rank = int(match.group(1))

    traces_dir = trace_file.parent.parent
    candidates = sorted(traces_dir.glob(f"kernel_annotations_rank{rank}_*.pkl"))
    if candidates:
        return candidates[0]
    return None


def main() -> None:
    # [FUNCTION DOC] Entry point. Parses arguments (trace file, annotations pickle, output path, default stream),
    # auto-discovers the pickle if absent (prints to stderr and exits with status 1 on failure), loads pickle and trace,
    # annotates, fixes overlaps, picks a default .annotated output name, saves, and prints progress.
    # KNOWN ISSUE: the overlap message always says stream 8 regardless of the stream used by the helper default.
    parser = argparse.ArgumentParser(
        description="Annotate a profiler trace with CUDA graph kernel annotations."
    )
    parser.add_argument(
        "trace_file", type=Path, help="Input trace file (.json or .json.gz)"
    )
    parser.add_argument(
        "-a",
        "--annotations",
        type=Path,
        default=None,
        help="Kernel annotations pickle file. Auto-discovered if omitted.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file path. Defaults to <trace_file>.annotated.<ext>",
    )
    parser.add_argument(
        "--default-stream",
        type=int,
        default=7,
        help="Stream ID to assign to unannotated graphed events (default: 7).",
    )
    args = parser.parse_args()

    annotations_pkl = args.annotations
    if annotations_pkl is None:
        annotations_pkl = _find_annotations_pkl(args.trace_file)
        if annotations_pkl is None:
            print(
                f"Could not auto-discover annotations pickle for {args.trace_file}. "
                f"Use -a to specify it explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Auto-discovered annotations: {annotations_pkl}")

    with open(annotations_pkl, "rb") as f:
        annotations = pickle.load(f)
    print(f"Loaded {len(annotations)} kernel annotations")

    trace = load_trace(args.trace_file)
    total_events = len(trace.get("traceEvents", []))
    print(f"Loaded trace with {total_events} events")

    count = annotate_trace(trace, annotations, default_stream=args.default_stream)
    print(f"Annotated {count} kernel events")

    overlap_moved = _move_overlapping_to_stream(
        trace, default_stream=args.default_stream
    )
    if overlap_moved:
        print(f"Moved {overlap_moved} overlapping events to stream 8")

    ts_fixed = _fix_overlapping_timestamps(trace)
    if ts_fixed:
        print(f"Fixed {ts_fixed} overlapping graphed event timestamps")

    output = args.output
    if output is None:
        name = args.trace_file.name
        if name.endswith(".json.gz"):
            output = args.trace_file.with_name(
                name.replace(".json.gz", ".annotated.json.gz")
            )
        elif name.endswith(".json"):
            output = args.trace_file.with_suffix(".annotated.json")
        else:
            output = args.trace_file.with_suffix(args.trace_file.suffix + ".annotated")

    save_trace(trace, output)
    print(f"Saved annotated trace to {output}")


if __name__ == "__main__":
    main()
