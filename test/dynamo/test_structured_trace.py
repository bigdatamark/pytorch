# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for the structured trace (torch.__trace logger) output of torch.compile.
# Defines sample functions, a logging filter and formatter that stabilize trace records
# (stubbing stacks, hashes, metrics, remapping tensor and storage ids), and
# StructuredTraceTest, which compiles functions and compares the captured JSON metadata lines
# against expected inline strings, then validates the raw log with the external tlparse tool
# when installed. Import time: reads the current accelerator type, optionally imports
# FakeStore when distributed is available, and checks whether tlparse is on the path. Runs
# under the dynamo run_tests entry point.
import copy
import functools
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import unittest.mock
from contextlib import contextmanager
from unittest import skipIf

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._logging.structured
import torch.distributed as dist
import torch.fx as fx
from torch._inductor.test_case import TestCase
from torch._logging._internal import TorchLogsFormatter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_utils import find_free_port
from torch.testing._internal.inductor_utils import HAS_XPU_AND_TRITON
from torch.testing._internal.triton_utils import requires_gpu_and_triton


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


if torch.distributed.is_available():
    from torch.testing._internal.distributed.fake_pg import FakeStore

HAS_TLPARSE = shutil.which("tlparse") is not None
requires_tlparse = unittest.skipUnless(HAS_TLPARSE, "requires tlparse")
requires_distributed = functools.partial(
    unittest.skipIf, not dist.is_available(), "requires distributed"
)


def example_fn(a):
    # [FUNCTION DOC] HELPER: Sample compile target. Multiplies the input by a 1000x1000 tensor
    # of ones, adds another such tensor, returns the result. No graph breaks.
    output = a.mul(torch.ones(1000, 1000))
    output = output.add(torch.ones(1000, 1000))
    return output


def example_training_fn(a):
    # [FUNCTION DOC] HELPER: Sample training target. Multiplies the input by a ones tensor
    # that requires grad, adds a ones tensor, calls backward on the sum (causing a graph
    # break), and returns the output.
    # KNOWN ISSUE: expected output embeds source line numbers in resume function names;
    # inserted doc comments shift them, so the test will fail until the numbers are updated.
    output = a.mul(torch.ones(1000, 1000, requires_grad=True))
    output = output.add(torch.ones(1000, 1000))
    output.sum().backward()
    return output


def dynamo_error_fn(a):
    # [FUNCTION DOC] HELPER: Sample target that fails during dynamo tracing: adds a 1000x1000
    # tensor and a 10x10 tensor, whose shapes cannot broadcast.
    output = a.mul(torch.ones(1000, 1000))
    output = output.add(torch.ones(10, 10))
    return output


def inductor_error_fn(a):
    # [FUNCTION DOC] HELPER: Sample target applying round to the input; the inductor error
    # test replaces the round lowering with a failing one.
    output = torch.round(a)
    return output


def inductor_schedule_fn(a):
    # [FUNCTION DOC] HELPER: Sample target for GPU tests. Adds a 1000x1000 ones tensor created
    # on the current accelerator device type to the input and returns it.
    output = a.add(torch.ones(1000, 1000, device=device_type))
    return output


ARGS = (torch.ones(1000, 1000, requires_grad=True),)


def replace_dynamic(buffer, key):
    # [FUNCTION DOC] HELPER: Replaces the floating point value of the given JSON key in the
    # buffer text with the placeholder dynamic marker, using a regular expression. Returns the
    # new string; used for timing fields that vary per run.
    return re.sub(r'("' + key + r'":\s*)(\d+\.\d+)', r"\1<dynamic>", buffer)


class StructuredTraceTestingFilter(logging.Filter):
    # [CLASS DOC] HELPER: Logging filter for the trace handler that drops records whose
    # metadata carries a str entry and the global torch_version artifact (which depends on
    # test ordering). If match_name was given, only passes artifact records with exactly that
    # artifact name, or non-artifact records whose metadata has a key equal to match_name.
    # Attribute: match_name.
    def __init__(self, match_name=None):
        # [FUNCTION DOC] HELPER: Stores the optional match_name used to restrict which records
        # pass the filter.
        self.match_name = match_name

    def filter(self, record):
        # [FUNCTION DOC] HELPER: Returns False to drop a record if its metadata has a str
        # entry or is the torch_version artifact, or if match_name is set and the record does
        # not match it; otherwise True. Reads record.metadata only.
        if "str" in record.metadata:
            return False
        # torch_version is a global artifact emitted once per process the first
        # time the trace handler initializes. Its presence (and commit-hash
        # payload) depends on run context and test ordering, so drop it here to
        # keep each test's expected inline output deterministic.
        if (
            "artifact" in record.metadata
            and record.metadata["artifact"].get("name") == "torch_version"
        ):
            return False
        if self.match_name is not None:
            if "artifact" in record.metadata:
                if self.match_name != record.metadata["artifact"]["name"]:
                    return False
            elif self.match_name not in record.metadata:
                return False
        return True


class ChromiumEventFilter(logging.Filter):
    # [CLASS DOC] HELPER: Logging filter that rejects records carrying chromium_event metadata
    # so tests normally do not see chrome trace events.
    def filter(self, record):
        # [FUNCTION DOC] HELPER: Returns True when the record metadata has no chromium_event
        # key, False otherwise.
        return "chromium_event" not in record.metadata


class StructuredTracePayloadFormatter(logging.Formatter):
    # [CLASS DOC] HELPER: Formatter that emits only the payload text of a structured trace
    # record, stripped of surrounding whitespace; used to capture artifact payloads.
    def format(self, record):
        # [FUNCTION DOC] HELPER: Returns the record payload with leading and trailing
        # whitespace removed.
        return record.payload.strip()


class _DescribeIdNormalizer:
    # [CLASS DOC] HELPER: Renumbers describe_storage, describe_tensor and describe_source ids
    # in trace metadata to small deterministic integers in order of first appearance, and
    # replaces describer_id with a fixed marker. State: two mappings from original to new id
    # (tensor and storage) and two next-id counters, all starting at zero.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Creates empty tensor and storage id mappings and sets both
        # next-id counters to zero.
        self._tensor_id_remap = {}
        self._storage_id_remap = {}
        self._next_tensor_id = 0
        self._next_storage_id = 0

    def normalize(self, metadata):
        # [FUNCTION DOC] HELPER: Mutates the given metadata mapping in place and returns it.
        # Remaps storage ids, tensor ids and the storage reference of tensors, source ids
        # (which share the tensor id mapping), sets describer_id to a fixed marker, and
        # replaces a view_func value with a fixed marker. Missing sections are left alone.
        if "describe_storage" in metadata:
            storage_meta = metadata["describe_storage"]
            if (storage_id := storage_meta.get("id")) is not None:
                storage_meta["id"] = self._normalize_storage_id(storage_id)
            storage_meta["describer_id"] = "ID"
        if "describe_tensor" in metadata:
            tensor_meta = metadata["describe_tensor"]
            if (tensor_id := tensor_meta.get("id")) is not None:
                tensor_meta["id"] = self._normalize_tensor_id(tensor_id)
            if (storage_id := tensor_meta.get("storage")) is not None:
                tensor_meta["storage"] = self._normalize_storage_id(storage_id)
            tensor_meta["describer_id"] = "ID"
            if "view_func" in tensor_meta:
                tensor_meta["view_func"] = "VIEW_FUNC"
        if "describe_source" in metadata:
            source_meta = metadata["describe_source"]
            if (source_id := source_meta.get("id")) is not None:
                source_meta["id"] = self._normalize_tensor_id(source_id)
            source_meta["describer_id"] = "ID"
        return metadata

    def _normalize_tensor_id(self, original_id):
        # [FUNCTION DOC] HELPER: Returns the stable id for an original tensor id, assigning
        # the next counter value on first sight and recording it.
        if original_id not in self._tensor_id_remap:
            self._tensor_id_remap[original_id] = self._next_tensor_id
            self._next_tensor_id += 1
        return self._tensor_id_remap[original_id]

    def _normalize_storage_id(self, original_id):
        # [FUNCTION DOC] HELPER: Returns the stable id for an original storage id, assigning
        # the next counter value on first sight and recording it.
        if original_id not in self._storage_id_remap:
            self._storage_id_remap[original_id] = self._next_storage_id
            self._next_storage_id += 1
        return self._storage_id_remap[original_id]


class StructuredTraceTestingFormatter(logging.Formatter):
    # [CLASS DOC] HELPER: Formatter used by the test handler. Deep-copies record metadata and
    # replaces unstable values (payload hash, stacks, filenames, compilation metrics) with
    # fixed markers, normalizes describe ids, and serializes to JSON. State: a private id
    # normalizer instance.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Initializes the base formatter and creates a fresh id
        # normalizer so remapping restarts per formatter.
        super().__init__()
        self._id_normalizer = _DescribeIdNormalizer()

    def format(self, record):
        # [FUNCTION DOC] HELPER: Formats one record as a JSON string of its stabilized
        # metadata. Stubs has_payload, dynamo_start stack, inductor_output_code filename and
        # file_path, stack, and the four compilation metrics keys; normalizes describe ids;
        # stubs user_stack and stack of create_symbol, guard_added_fast or
        # create_unbacked_symbol entries. For dump_file records the name is replaced by a
        # fixed marker and the right-stripped payload lines are appended after the JSON on
        # following lines. Does not mutate the record.
        metadata = copy.deepcopy(record.metadata)

        # Stub out values that are not stable across runs
        # TODO: Check that these match schema
        if "has_payload" in metadata:
            metadata["has_payload"] = "HASH"
        if "dynamo_start" in metadata:
            metadata["dynamo_start"]["stack"] = "STACK"
        if "inductor_output_code" in metadata:
            metadata["inductor_output_code"]["filename"] = "FILENAME"
            if "file_path" in metadata["inductor_output_code"]:
                metadata["inductor_output_code"]["file_path"] = "FILENAME"
        if "stack" in metadata:
            metadata["stack"] = "STACK"
        if "compilation_metrics" in metadata:
            metadata["compilation_metrics"] = "METRICS"
        if "bwd_compilation_metrics" in metadata:
            metadata["bwd_compilation_metrics"] = "METRICS"
        if "compilation_metrics_runtime" in metadata:
            metadata["compilation_metrics_runtime"] = "METRICS"
        if "bwd_compilation_metrics_runtime" in metadata:
            metadata["bwd_compilation_metrics_runtime"] = "METRICS"
        metadata = self._id_normalizer.normalize(metadata)
        if (
            (k := "create_symbol") in metadata
            or (k := "guard_added_fast") in metadata
            or (k := "create_unbacked_symbol") in metadata
        ):
            metadata[k]["user_stack"] = "STACK"
            metadata[k]["stack"] = "STACK"

        if "dump_file" in metadata:
            # Don't include the actually key number, that's sensitive to other
            # test runs
            metadata["dump_file"]["name"] = "<eval_with_key>"
            return (
                json.dumps(metadata)
                + "\n"
                + "\n".join(l.rstrip() for l in record.payload.splitlines())
            )

        return json.dumps(metadata)


trace_log = logging.getLogger("torch.__trace")

chrome_event_filter = ChromiumEventFilter()


def show_chrome_events(fn):
    # [FUNCTION DOC] HELPER: Decorator for tests that need chromium events. The wrapped test
    # first removes the module level chromium event filter from the test's handler, then runs
    # the test and returns its result.
    """
    Don't hide chrome events for this test
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper produced by the decorator. Removes the chromium event
        # filter from self.handler, then calls the original test with the same arguments and
        # returns its result.
        self.handler.removeFilter(chrome_event_filter)
        return fn(self, *args, **kwargs)

    return wrapper


class StructuredTraceTest(TestCase):
    # [CLASS DOC] Test case for structured tracing. Each test attaches a handler to the
    # torch.__trace logger that captures stabilized JSON lines into a string buffer, plus a
    # raw handler writing the real trace format to a temporary file that tlparse can parse.
    # Attributes set in setUp: buffer, old_level, handler, raw_file, raw_handler. Tests
    # compile functions and compare the buffer against expected inline text.
    def setUp(self):
        # [FUNCTION DOC] Resets dynamo and the structured logging intern table, creates the
        # capture buffer, sets the trace logger to debug level (remembering the old level),
        # and installs the formatted handler (with test filter and chromium filter) and the
        # raw file handler on the trace logger.
        super().setUp()
        torch._dynamo.reset()
        torch._logging.structured.INTERN_TABLE.clear()
        self.buffer = io.StringIO()
        self.old_level = trace_log.level
        trace_log.setLevel(logging.DEBUG)

        self.handler = logging.StreamHandler(self.buffer)
        self.handler.setFormatter(StructuredTraceTestingFormatter())
        self.handler.addFilter(StructuredTraceTestingFilter())
        self.handler.addFilter(chrome_event_filter)
        trace_log.addHandler(self.handler)

        self.raw_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w", delete=True
        )  # set this to False to keep temporary files
        self.raw_handler = logging.StreamHandler(self.raw_file)
        self.raw_handler.setFormatter(TorchLogsFormatter(trace=True))
        trace_log.addHandler(self.raw_handler)

    def tearDown(self):
        # [FUNCTION DOC] Removes both handlers from the trace logger, closes the raw temporary
        # file, restores the logger level, and calls the base teardown.
        trace_log.removeHandler(self.handler)
        trace_log.removeHandler(self.raw_handler)
        self.raw_file.close()
        trace_log.setLevel(self.old_level)
        super().tearDown()

    def assertExpectedInline(self, actual, expected, skip=0):
        # [FUNCTION DOC] Overrides the expected-inline assertion: normalizes rank fields and
        # describe ids in both actual and expected text before delegating to the base
        # assertion, with the skip count raised by one so failures point at the test caller.
        super().assertExpectedInline(
            self._normalize_rank_field(self._normalize_describe_ids(actual)),
            self._normalize_rank_field(self._normalize_describe_ids(expected)),
            skip=skip + 1,
        )

    @staticmethod
    def _normalize_rank_field(text):
        # [FUNCTION DOC] HELPER: Static. Removes the rank zero field from JSON trace text in
        # the three positions it may appear (middle, start, only), so expected text is the
        # same with or without a rank. Non-string input is returned unchanged.
        if not isinstance(text, str):
            return text
        text = text.replace(', "rank": 0', "")
        text = text.replace('"rank": 0, ', "")
        text = text.replace('"rank": 0', "")
        return text

    @staticmethod
    def _normalize_describe_ids(text):
        # [FUNCTION DOC] HELPER: Static. For string input, parses each non-empty line as JSON,
        # normalizes describe ids with a fresh normalizer, and re-serializes; lines that are
        # empty or not valid JSON are kept as is. Preserves a trailing newline. Non-string
        # input is returned unchanged.
        if not isinstance(text, str):
            return text
        normalizer = _DescribeIdNormalizer()
        trailing_newline = text.endswith("\n")
        normalized_lines = []
        for line in text.splitlines():
            if not line:
                normalized_lines.append(line)
                continue
            try:
                metadata = json.loads(line)
            except json.JSONDecodeError:
                normalized_lines.append(line)
                continue
            normalized_lines.append(json.dumps(normalizer.normalize(metadata)))
        result = "\n".join(normalized_lines)
        if trailing_newline:
            result += "\n"
        return result

    def assertParses(self):
        # [FUNCTION DOC] Skips the test if tlparse is not installed. Otherwise runs tlparse on
        # the raw trace file in strict mode with overwrite and no browser into a temporary
        # directory, raising on a nonzero exit, and always removes the directory afterwards.
        if not HAS_TLPARSE:
            self.skipTest("requires tlparse")
        out = tempfile.mkdtemp()
        try:
            subprocess.check_call(
                [
                    "tlparse",
                    "-o",
                    out,
                    "--overwrite",
                    "--no-browser",
                    "--strict",
                    self.raw_file.name,
                ]
            )
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_compile_id_serialization_deserialization(self):
        # [FUNCTION DOC] Checks that CompileId survives a string round trip for ids with and
        # without a compiled autograd id, and with missing frame fields; raises AssertionError
        # on mismatch. Also checks that several malformed id strings make the parser raise
        # ValueError.
        cid = torch._guards.CompileId(
            frame_id=1,
            frame_compile_id=2,
        )
        if cid != torch._guards.CompileId.from_string(str(cid)):
            raise AssertionError("CompileId round-trip failed")

        cid = torch._guards.CompileId(
            compiled_autograd_id=1,
            frame_id=2,
            frame_compile_id=3,
        )
        if cid != torch._guards.CompileId.from_string(str(cid)):
            raise AssertionError("CompileId round-trip failed")

        cid = torch._guards.CompileId(
            compiled_autograd_id=1,
            frame_id=None,
            frame_compile_id=None,
        )
        if cid != torch._guards.CompileId.from_string(str(cid)):
            raise AssertionError("CompileId round-trip failed")

        for bad_cid in ["-/-", "-/1", "1/-", "!1/2", "!1/-/-"]:
            with self.assertRaises(ValueError):
                torch._guards.CompileId.from_string(bad_cid)

    @requires_gpu_and_triton
    def test_schedule(self):
        # [FUNCTION DOC] GPU and triton test. Compiles the schedule sample function with
        # inductor, runs it on a device tensor, and compares the captured trace lines
        # (including triton kernel info) against the expected text, then checks tlparse can
        # parse the raw log.
        fn_opt = torch.compile(inductor_schedule_fn, backend="inductor")
        fn_opt(torch.ones(1000, 1000, device=device_type))
        self.assertExpectedInline(
            self.buffer.getvalue(),
            f"""\
{{"dynamo_start": {{"stack": "STACK"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}}
{{"describe_storage": {{"id": 0, "describer_id": "ID", "size": 4000000}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}}
{{"describe_tensor": {{"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='{device_type}', index=0)", "size": [1000, 1000], "dynamo_hint_overrides": {{}}, "is_leaf": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}}
{{"describe_source": {{"describer_id": "ID", "id": 0, "source": "L['a']"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}}
{{"dynamo_output_graph": {{"sizes": {{"l_a_": [1000, 1000], "ones": [1000, 1000], "add": [1000, 1000]}}}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "aotautograd_cache_miss", "encoding": "json"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "before_pre_grad_graph", "encoding": "string"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "after_pre_grad_graph", "encoding": "string"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "aot_forward_graph_fw_metadata", "encoding": "string"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"aot_inference_graph": {{}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "torch._functorch.config", "encoding": "string"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "before_joint_graph", "encoding": "string"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "after_joint_graph", "encoding": "string"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "fx_graph_runnable", "encoding": "string"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "before_post_grad_graph", "encoding": "string"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "inductor_post_grad_graph", "encoding": "string"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"inductor_output_code": {{"filename": "FILENAME", "file_path": "FILENAME"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "triton_kernel_info", "encoding": "json"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "fx_graph_cache_miss", "encoding": "json"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "runtime_wrapper_orchestration", "encoding": "string"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"dynamo_cpp_guards_str": {{}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}}
{{"compilation_metrics_runtime": "METRICS", "frame_id": 0, "frame_compile_id": 0}}
""",
        )

        self.assertParses()

    @requires_gpu_and_triton
    def test_gpugraphs(self):
        # [FUNCTION DOC] GPU and triton test. Compiles the schedule sample function in reduce-
        # overhead mode (GPU graphs), runs it, compares the trace lines against the expected
        # text, then checks tlparse parsing.
        fn_opt = torch.compile(mode="reduce-overhead")(inductor_schedule_fn)  # noqa: UNSPECIFIED_BACKEND
        fn_opt(torch.ones(1000, 1000, device=device_type))
        self.assertExpectedInline(
            self.buffer.getvalue(),
            f"""\
{{"dynamo_start": {{"stack": "STACK"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}}
{{"describe_storage": {{"id": 0, "describer_id": "ID", "size": 4000000}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}}
{{"describe_tensor": {{"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='{device_type}', index=0)", "size": [1000, 1000], "dynamo_hint_overrides": {{}}, "is_leaf": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}}
{{"describe_source": {{"describer_id": "ID", "id": 0, "source": "L['a']"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}}
{{"dynamo_output_graph": {{"sizes": {{"l_a_": [1000, 1000], "ones": [1000, 1000], "add": [1000, 1000]}}}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "aotautograd_cache_miss", "encoding": "json"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "before_pre_grad_graph", "encoding": "string"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "after_pre_grad_graph", "encoding": "string"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "aot_forward_graph_fw_metadata", "encoding": "string"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"aot_inference_graph": {{}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "torch._functorch.config", "encoding": "string"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "before_joint_graph", "encoding": "string"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "after_joint_graph", "encoding": "string"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "fx_graph_runnable", "encoding": "string"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "before_post_grad_graph", "encoding": "string"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "inductor_post_grad_graph", "encoding": "string"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"inductor_output_code": {{"filename": "FILENAME", "file_path": "FILENAME"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "triton_kernel_info", "encoding": "json"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "fx_graph_cache_miss", "encoding": "json"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "runtime_wrapper_orchestration", "encoding": "string"}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"dynamo_cpp_guards_str": {{}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}}
{{"compilation_metrics_runtime": "METRICS", "frame_id": 0, "frame_compile_id": 0}}
""",
        )

        self.assertParses()

    @requires_tlparse
    def test_recompiles(self):
        # [FUNCTION DOC] Compiles a tensor add with inductor, calls it with two tensors and
        # then with a tensor and an int to force a recompile, and compares the trace (second
        # frame compile id, recompile_reasons artifact, create_symbol) against the expected
        # text, then checks tlparse parsing.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Compile target returning the sum of its two arguments.
            return torch.add(x, y)

        fn_opt = torch.compile(fn, backend="inductor")
        fn_opt(torch.ones(1000, 1000), torch.ones(1000, 1000))
        fn_opt(torch.ones(1000, 1000), 1)

        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "dynamo_hint_overrides": {}, "is_leaf": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['x']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 1, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 1, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "dynamo_hint_overrides": {}, "is_leaf": true, "stride": [1000, 1], "storage": 1, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 1, "source": "L['y']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_x_": [1000, 1000], "l_y_": [1000, 1000], "add": [1000, 1000]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_pre_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "after_pre_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_inference_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "torch._functorch.config", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_joint_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "after_joint_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_post_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "inductor_post_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME", "file_path": "FILENAME"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "runtime_wrapper_orchestration", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "recompile_reasons", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "dynamo_hint_overrides": {}, "is_leaf": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['x']"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"create_symbol": {"symbol": "s48", "val": "1", "vr": "[-int_oo, int_oo]", "source": "L['y']", "user_stack": "STACK", "stack": "STACK"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_x_": [1000, 1000], "add": [1000, 1000]}}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_pre_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "after_pre_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"aot_inference_graph": {}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "torch._functorch.config", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_joint_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "after_joint_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_post_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "inductor_post_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME", "file_path": "FILENAME"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "runtime_wrapper_orchestration", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
""",
        )

        self.assertParses()

    @requires_tlparse
    def test_recompile_backend_match(self):
        # [FUNCTION DOC] Compiles a function with eager backend dynamic then static, capturing
        # the recompile_reasons payload on a separate handler during the second compile, and
        # asserts the payload mentions the backend match reason, cached and new backend text,
        # the wrapper class name and both dynamic settings. Handler is removed in a finally
        # block. Then checks tlparse parsing.
        def fn(x):
            # [FUNCTION DOC] HELPER: Compile target returning the sine plus cosine of the
            # input.
            return x.sin() + x.cos()

        x = torch.ones(10)

        torch.compile(fn, backend="eager", dynamic=True)(x)

        payload_buffer = io.StringIO()
        payload_handler = logging.StreamHandler(payload_buffer)
        payload_handler.setLevel(logging.DEBUG)
        payload_handler.setFormatter(StructuredTracePayloadFormatter())
        payload_handler.addFilter(StructuredTraceTestingFilter("recompile_reasons"))
        trace_log.addHandler(payload_handler)
        try:
            torch.compile(fn, backend="eager", dynamic=False)(x)
        finally:
            trace_log.removeHandler(payload_handler)

        payload = payload_buffer.getvalue()
        self.assertIn("BACKEND_MATCH", payload)
        self.assertIn("Cached backend:", payload)
        self.assertIn("New backend:", payload)
        self.assertIn("_TorchCompileWrapper", payload)
        self.assertIn("dynamic=True", payload)
        self.assertIn("dynamic=False", payload)

        self.assertParses()

    @requires_tlparse
    def test_example_fn(self):
        # [FUNCTION DOC] Compiles the example function with inductor, runs it on a CPU tensor,
        # compares the full trace against the expected text, and checks tlparse parsing.
        fn_opt = torch.compile(example_fn, backend="inductor")
        fn_opt(torch.ones(1000, 1000))
        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "dynamo_hint_overrides": {}, "is_leaf": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_a_": [1000, 1000], "ones": [1000, 1000], "mul": [1000, 1000], "ones_1": [1000, 1000], "add": [1000, 1000]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_pre_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "after_pre_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_inference_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "torch._functorch.config", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_joint_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "after_joint_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_post_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "inductor_post_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME", "file_path": "FILENAME"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "runtime_wrapper_orchestration", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
""",
        )

        self.assertParses()

    @requires_tlparse
    def test_example_training_fn(self):
        # [FUNCTION DOC] Compiles the training sample function with inductor and runs it,
        # replaces timing fields (inductor compile time, code generation time, logging
        # overhead) with a placeholder, compares the trace (graph breaks, resume frames, joint
        # graphs, backward metrics) against the expected text, then checks tlparse parsing.
        # KNOWN ISSUE: expected output embeds source line numbers in resume function names;
        # inserted doc comments shift them, so the test will fail until the numbers are
        # updated.
        fn_opt = torch.compile(example_training_fn, backend="inductor")
        fn_opt(torch.ones(1000, 1000, requires_grad=True))
        buffer = self.buffer.getvalue()
        buffer = replace_dynamic(buffer, "inductor_compile_time_s")
        buffer = replace_dynamic(buffer, "code_gen_time_s")
        buffer = replace_dynamic(buffer, "structured_logging_overhead_s")
        self.assertExpectedInline(
            buffer,
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "dynamo_hint_overrides": {}, "is_leaf": true, "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "dynamo_graph_break_reason", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 1}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "dynamo_hint_overrides": {}, "is_leaf": true, "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 1}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 1}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 1}
{"dynamo_start": {"stack": "STACK"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "torch_dynamo_resume_in_example_training_fn_at_50_ORIGINAL_BYTECODE", "encoding": "string"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "dynamo_graph_break_reason", "encoding": "string"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 1, "frame_compile_id": 0, "attempt": 1}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "dynamo_hint_overrides": {}, "is_leaf": true, "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 1}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['___stack1']"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 1}
{"artifact": {"name": "torch_dynamo_resume_in_example_training_fn_at_50_MODIFIED_BYTECODE", "encoding": "string"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 1, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 1, "frame_compile_id": 0, "attempt": 1}
{"dynamo_start": {"stack": "STACK"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "torch_dynamo_resume_in_example_training_fn_at_50_ORIGINAL_BYTECODE", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 2, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "dynamo_hint_overrides": {}, "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['___stack0']"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "dynamo_graph_break_reason", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "dynamo_hint_overrides": {}, "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['___stack0']"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1}
{"dynamo_output_graph": {"sizes": {"l_stack0_": [1000, 1000], "ones": [1000, 1000], "add": [1000, 1000], "sum_1": []}}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_miss", "encoding": "json"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "before_pre_grad_graph", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "after_pre_grad_graph", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"aot_joint_graph": {}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "torch._functorch.config", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"aot_forward_graph": {}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"aot_backward_graph": {}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "before_post_grad_graph", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "inductor_post_grad_graph", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME", "file_path": "FILENAME"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "backward_prologue", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "backward_epilogue", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "compiled_function_forward", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "compiled_function_backward", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "compiled_fn_wrapper", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "runtime_wrapper_orchestration", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "torch_dynamo_resume_in_example_training_fn_at_50_MODIFIED_BYTECODE", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 2, "frame_compile_id": 0, "attempt": 1}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "before_post_grad_graph", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "inductor_post_grad_graph", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME", "file_path": "FILENAME"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"bwd_compilation_metrics": "METRICS", "frame_id": 2, "frame_compile_id": 0, "attempt": 1}
{"dynamo_start": {"stack": "STACK"}, "frame_id": 3, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "torch_dynamo_resume_in_example_training_fn_at_52_ORIGINAL_BYTECODE", "encoding": "string"}, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 3, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "dynamo_hint_overrides": {}, "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 3, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['output']"}, "frame_id": 3, "frame_compile_id": 0, "attempt": 0}
{"compilation_metrics": "METRICS", "frame_id": 3, "frame_compile_id": 0, "attempt": 0}
""",
        )

        self.assertParses()

    @requires_tlparse
    def test_dynamo_error(self):
        # [FUNCTION DOC] Compiles the dynamo error sample with inductor and runs it,
        # swallowing the exception, then compares the trace (ending with a dynamo_error
        # artifact and metrics) against the expected text and checks tlparse parsing.
        try:
            fn_opt = torch.compile(dynamo_error_fn, backend="inductor")
            fn_opt(*ARGS)
        except Exception:
            pass
        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "dynamo_hint_overrides": {}, "is_leaf": true, "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "dynamo_error", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
""",
        )

        self.assertParses()

    @requires_tlparse
    def test_inductor_error(self):
        # [FUNCTION DOC] Injects a failure into inductor lowering: every registered lowering
        # whose name contains round is replaced (via a patched dictionary) by a function
        # raising AssertionError. Compiles and runs the inductor error sample, swallowing the
        # exception, then compares the trace against the expected text and checks tlparse
        # parsing.
        import torch._inductor.lowering

        def throw(x):
            # [FUNCTION DOC] HELPER: Replacement lowering that ignores its argument and always
            # raises AssertionError.
            raise AssertionError

        # inject an error in the lowerings
        dict_entries = {}
        for x in list(torch._inductor.lowering.lowerings.keys()):
            if "round" in x.__name__:
                dict_entries[x] = throw

        with unittest.mock.patch.dict(torch._inductor.lowering.lowerings, dict_entries):
            try:
                fn_opt = torch.compile(inductor_error_fn, backend="inductor")
                fn_opt(*ARGS)
            except Exception:
                pass

        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "dynamo_hint_overrides": {}, "is_leaf": true, "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_a_": [1000, 1000], "round_1": [1000, 1000]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_pre_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "after_pre_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_joint_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "torch._functorch.config", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_forward_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_backward_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_post_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "inductor_post_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "dynamo_error", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
""",
        )

        self.assertParses()

    @skipIf(HAS_XPU_AND_TRITON, "No backend type associated with device type xpu")
    @requires_distributed()
    @requires_gpu_and_triton
    @unittest.skip("https://github.com/pytorch/pytorch/issues/176188")
    def test_ddp_graphs(self):
        # [FUNCTION DOC] Currently skipped (linked issue) and conditionally skipped for XPU,
        # without distributed, or without GPU and triton. Resets frame counters, initializes a
        # single process gloo group, wraps a toy model in DDP, compiles and runs it, destroys
        # the group, and compares the trace against the expected text including rank fields.
        # Side effects: sets MASTER_ADDR and MASTER_PORT environment variables and does not
        # restore them.
        # KNOWN ISSUE: expected output embeds source line numbers in resume function names;
        # inserted doc comments shift them, so the test will fail until the numbers are
        # updated.
        import torch._dynamo.convert_frame as convert_frame

        convert_frame.FRAME_COUNTER = 0
        convert_frame.FRAME_COMPILE_COUNTER.clear()

        class ToyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Toy module with a sequential stack of two 1024x1024 linear
            # layers, used by the DDP trace test.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the module and builds the sequential
                # container of two linear layers.
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(1024, 1024),
                    torch.nn.Linear(1024, 1024),
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the sequential layers to the input and
                # returns the result.
                return self.layers(x)

        # TODO: this isn't safely bracketed, will leak
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        dist.init_process_group("gloo", rank=0, world_size=1)

        model = DDP(ToyModel().to(f"{device_type}:0"), device_ids=[0], bucket_cap_mb=4)
        ddp_model = torch.compile(model, backend="inductor")

        ddp_model(torch.randn(1024, 1024, device=f"{device_type}:0"))

        dist.destroy_process_group()

        self.assertExpectedInline(
            self.buffer.getvalue(),
            f"""\
{{"dynamo_start": {{"stack": "STACK"}}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}}
{{"artifact": {{"name": "dynamo_graph_break_reason", "encoding": "string"}}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"describe_storage": {{"id": 0, "describer_id": "ID", "size": 4194304}}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 1}}
{{"describe_tensor": {{"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='{device_type}', index=0)", "size": [1024, 1024], "dynamo_hint_overrides": {{}}, "is_leaf": true, "stride": [1024, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 1}}
{{"describe_source": {{"describer_id": "ID", "id": 0, "source": "L['args'][0]"}}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 1}}
{{"dynamo_cpp_guards_str": {{}}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}}
{{"compilation_metrics": "METRICS", "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 1}}
{{"dynamo_start": {{"stack": "STACK"}}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}}
{{"artifact": {{"name": "dynamo_graph_break_reason", "encoding": "string"}}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"dynamo_cpp_guards_str": {{}}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}}
{{"compilation_metrics": "METRICS", "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 1}}
{{"dynamo_start": {{"stack": "STACK"}}, "rank": 0, "frame_id": 2, "frame_compile_id": 0, "attempt": 0}}
{{"artifact": {{"name": "dynamo_graph_break_reason", "encoding": "string"}}, "rank": 0, "frame_id": 2, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"dynamo_cpp_guards_str": {{}}, "rank": 0, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}}
{{"compilation_metrics": "METRICS", "rank": 0, "frame_id": 2, "frame_compile_id": 0, "attempt": 1}}
{{"dynamo_start": {{"stack": "STACK"}}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0}}
{{"artifact": {{"name": "torch_dynamo_resume_in___init___at_103_ORIGINAL_BYTECODE", "encoding": "string"}}, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"compilation_metrics": "METRICS", "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0}}
{{"dynamo_start": {{"stack": "STACK"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_storage": {{"id": 0, "describer_id": "ID", "size": 4194304}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_tensor": {{"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='{device_type}', index=0)", "size": [1024, 1024], "dynamo_hint_overrides": {{}}, "is_leaf": true, "requires_grad": true, "is_parameter": true, "stride": [1024, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_source": {{"describer_id": "ID", "id": 0, "source": "L['self']._modules['layers']._modules['0']._parameters['weight']"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_storage": {{"id": 1, "describer_id": "ID", "size": 4096}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_tensor": {{"id": 1, "ndim": 1, "dtype": "torch.float32", "device": "device(type='{device_type}', index=0)", "size": [1024], "dynamo_hint_overrides": {{}}, "is_leaf": true, "requires_grad": true, "is_parameter": true, "stride": [1], "storage": 1, "view_func": "VIEW_FUNC", "describer_id": "ID"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_source": {{"describer_id": "ID", "id": 1, "source": "L['self']._modules['layers']._modules['0']._parameters['bias']"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_storage": {{"id": 2, "describer_id": "ID", "size": 4194304}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_tensor": {{"id": 2, "ndim": 2, "dtype": "torch.float32", "device": "device(type='{device_type}', index=0)", "size": [1024, 1024], "dynamo_hint_overrides": {{}}, "is_leaf": true, "stride": [1024, 1], "storage": 2, "view_func": "VIEW_FUNC", "describer_id": "ID"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_source": {{"describer_id": "ID", "id": 2, "source": "L['x']"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_storage": {{"id": 3, "describer_id": "ID", "size": 4194304}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_tensor": {{"id": 8, "ndim": 2, "dtype": "torch.float32", "device": "device(type='{device_type}', index=0)", "size": [1024, 1024], "dynamo_hint_overrides": {{}}, "is_leaf": true, "requires_grad": true, "is_parameter": true, "stride": [1024, 1], "storage": 3, "view_func": "VIEW_FUNC", "describer_id": "ID"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_source": {{"describer_id": "ID", "id": 8, "source": "L['self']._modules['layers']._modules['1']._parameters['weight']"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_storage": {{"id": 4, "describer_id": "ID", "size": 4096}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_tensor": {{"id": 9, "ndim": 1, "dtype": "torch.float32", "device": "device(type='{device_type}', index=0)", "size": [1024], "dynamo_hint_overrides": {{}}, "is_leaf": true, "requires_grad": true, "is_parameter": true, "stride": [1], "storage": 4, "view_func": "VIEW_FUNC", "describer_id": "ID"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_source": {{"describer_id": "ID", "id": 9, "source": "L['self']._modules['layers']._modules['1']._parameters['bias']"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"dynamo_output_graph": {{"sizes": {{"l_self_modules_layers_modules_0_parameters_weight_": [1024, 1024], "l_self_modules_layers_modules_0_parameters_bias_": [1024], "l_x_": [1024, 1024], "l_self_modules_layers_modules_1_parameters_weight_": [1024, 1024], "l_self_modules_layers_modules_1_parameters_bias_": [1024], "input_1": [1024, 1024], "input_2": [1024, 1024]}}}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"optimize_ddp_split_graph": {{}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"optimize_ddp_split_child": {{"name": "submod_0"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"optimize_ddp_split_child": {{"name": "submod_1"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"describe_storage": {{"id": 0, "describer_id": "ID", "size": 4194304}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_tensor": {{"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='{device_type}', index=0)", "size": [1024, 1024], "dynamo_hint_overrides": {{}}, "is_leaf": true, "stride": [1024, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_source": {{"describer_id": "ID", "id": 0, "source": "L['x']"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_storage": {{"id": 1, "describer_id": "ID", "size": 4194304}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_tensor": {{"id": 1, "ndim": 2, "dtype": "torch.float32", "device": "device(type='{device_type}', index=0)", "size": [1024, 1024], "dynamo_hint_overrides": {{}}, "is_leaf": true, "requires_grad": true, "is_parameter": true, "stride": [1024, 1], "storage": 1, "view_func": "VIEW_FUNC", "describer_id": "ID"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_source": {{"describer_id": "ID", "id": 1, "source": "L['self']._modules['layers']._modules['0']._parameters['weight']"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_storage": {{"id": 2, "describer_id": "ID", "size": 4096}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_tensor": {{"id": 2, "ndim": 1, "dtype": "torch.float32", "device": "device(type='{device_type}', index=0)", "size": [1024], "dynamo_hint_overrides": {{}}, "is_leaf": true, "requires_grad": true, "is_parameter": true, "stride": [1], "storage": 2, "view_func": "VIEW_FUNC", "describer_id": "ID"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_source": {{"describer_id": "ID", "id": 2, "source": "L['self']._modules['layers']._modules['0']._parameters['bias']"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"artifact": {{"name": "before_pre_grad_graph", "encoding": "string"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "after_pre_grad_graph", "encoding": "string"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "aotautograd_cache_bypass", "encoding": "json"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"aot_joint_graph": {{}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "torch._functorch.config", "encoding": "string"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "aot_forward_graph_fw_metadata", "encoding": "string"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"aot_forward_graph": {{}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"aot_backward_graph": {{}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "fx_graph_runnable", "encoding": "string"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "before_post_grad_graph", "encoding": "string"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "inductor_post_grad_graph", "encoding": "string"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"inductor_output_code": {{"filename": "FILENAME", "file_path": "FILENAME"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "fx_graph_cache_miss", "encoding": "json"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"describe_storage": {{"id": 16, "describer_id": "ID", "size": 4194304}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_tensor": {{"id": 28, "ndim": 2, "dtype": "torch.float32", "device": "device(type='{device_type}', index=0)", "size": [1024, 1024], "dynamo_hint_overrides": {{}}, "is_leaf": true, "requires_grad": true, "is_parameter": true, "stride": [1024, 1], "storage": 16, "view_func": "VIEW_FUNC", "describer_id": "ID"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_source": {{"describer_id": "ID", "id": 28, "source": "L['self']._modules['layers']._modules['1']._parameters['weight']"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_storage": {{"id": 17, "describer_id": "ID", "size": 4096}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_tensor": {{"id": 29, "ndim": 1, "dtype": "torch.float32", "device": "device(type='{device_type}', index=0)", "size": [1024], "dynamo_hint_overrides": {{}}, "is_leaf": true, "requires_grad": true, "is_parameter": true, "stride": [1], "storage": 17, "view_func": "VIEW_FUNC", "describer_id": "ID"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"describe_source": {{"describer_id": "ID", "id": 29, "source": "L['self']._modules['layers']._modules['1']._parameters['bias']"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
{{"artifact": {{"name": "before_pre_grad_graph", "encoding": "string"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "after_pre_grad_graph", "encoding": "string"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "aotautograd_cache_bypass", "encoding": "json"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"aot_joint_graph": {{}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "torch._functorch.config", "encoding": "string"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "aot_forward_graph_fw_metadata", "encoding": "string"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"aot_forward_graph": {{}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"aot_backward_graph": {{}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "fx_graph_runnable", "encoding": "string"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "before_post_grad_graph", "encoding": "string"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "inductor_post_grad_graph", "encoding": "string"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"inductor_output_code": {{"filename": "FILENAME", "file_path": "FILENAME"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"artifact": {{"name": "fx_graph_cache_miss", "encoding": "json"}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"dynamo_cpp_guards_str": {{}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}}
{{"compilation_metrics": "METRICS", "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}}
""",
        )

        self.assertParses()

    @requires_tlparse
    @unittest.skip("https://github.com/pytorch/pytorch/issues/176188")
    def test_graph_breaks(self):
        # [FUNCTION DOC] Currently skipped (linked issue). Compiles a function containing an
        # explicit graph break with inductor, runs it, compares the trace against the expected
        # text, and checks tlparse parsing.
        # KNOWN ISSUE: expected output embeds source line numbers in resume function names;
        # inserted doc comments shift them, so the test will fail until the numbers are
        # updated.
        @torch.compile(backend="inductor")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compile target that triggers a dynamo graph break and
            # then returns the input plus one.
            torch._dynamo.graph_break()
            return x + 1

        fn(torch.ones(1))

        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "dynamo_graph_break_reason", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 1}
{"dynamo_start": {"stack": "STACK"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "torch_dynamo_resume_in_fn_at_808_ORIGINAL_BYTECODE", "encoding": "string"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 1, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1], "dynamo_hint_overrides": {}, "is_leaf": true, "stride": [1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['x']"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_x_": [1], "add": [1]}}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_miss", "encoding": "json"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_pre_grad_graph", "encoding": "string"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "after_pre_grad_graph", "encoding": "string"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_inference_graph": {}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "torch._functorch.config", "encoding": "string"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_joint_graph", "encoding": "string"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "after_joint_graph", "encoding": "string"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_post_grad_graph", "encoding": "string"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "inductor_post_grad_graph", "encoding": "string"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME", "file_path": "FILENAME"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "torch_dynamo_resume_in_fn_at_808_MODIFIED_BYTECODE", "encoding": "string"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
""",
        )

        self.assertParses()

    # TODO: bring in the trace_source tests once we start emitting bytecode

    @requires_tlparse
    def test_graph_sizes_dynamic(self):
        # [FUNCTION DOC] Compiles a matrix multiply function with eager backend, first static
        # and then dynamic, runs each with different shapes, compares the trace
        # (dynamo_output_graph sizes) against the expected text, and checks tlparse parsing.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Compile target returning the matrix product of its two
            # arguments.
            return a @ b

        fn_opt = torch.compile(fn, backend="eager", dynamic=False)
        fn_opt(torch.randn(10, 20), torch.randn(20, 30))

        fn_opt2 = torch.compile(fn, backend="eager", dynamic=True)
        fn_opt2(torch.randn(5, 10), torch.randn(10, 15))

        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 800}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [10, 20], "dynamo_hint_overrides": {}, "is_leaf": true, "stride": [20, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 1, "describer_id": "ID", "size": 2400}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 1, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [20, 30], "dynamo_hint_overrides": {}, "is_leaf": true, "stride": [30, 1], "storage": 1, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 1, "source": "L['b']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_a_": [10, 20], "l_b_": [20, 30], "matmul": [10, 30]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "recompile_reasons", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 200}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [5, 10], "dynamo_hint_overrides": {}, "is_leaf": true, "stride": [10, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"create_symbol": {"symbol": "s97", "val": "5", "vr": "[2, int_oo]", "source": "L['a'].size()[0]", "user_stack": "STACK", "stack": "STACK"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"create_symbol": {"symbol": "s98", "val": "10", "vr": "[2, int_oo]", "source": "L['a'].size()[1]", "user_stack": "STACK", "stack": "STACK"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_storage": {"id": 1, "describer_id": "ID", "size": 600}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_tensor": {"id": 1, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [10, 15], "dynamo_hint_overrides": {}, "is_leaf": true, "stride": [15, 1], "storage": 1, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 1, "source": "L['b']"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"create_symbol": {"symbol": "s52", "val": "10", "vr": "[2, int_oo]", "source": "L['b'].size()[0]", "user_stack": "STACK", "stack": "STACK"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"create_symbol": {"symbol": "s20", "val": "15", "vr": "[2, int_oo]", "source": "L['b'].size()[1]", "user_stack": "STACK", "stack": "STACK"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"guard_added_fast": {"expr": "Eq(s98, s52)", "user_stack": "STACK", "stack": "STACK"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_a_": ["s97", "s52"], "l_b_": ["s52", "s20"], "matmul": ["s97", "s20"]}}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
""",
        )

        self.assertParses()

    @requires_tlparse
    def test_guards_recompiles(self):
        # [FUNCTION DOC] Compiles a function that loops over two sequences with eager backend,
        # calls it with sequences of different lengths to force a recompile, compares the
        # trace (guard and recompile records) against the expected text, and checks tlparse
        # parsing.
        def fn(x, ys, zs):
            # [FUNCTION DOC] HELPER: Compile target that delegates to the inner function.
            return inner(x, ys, zs)

        def inner(x, ys, zs):
            # [FUNCTION DOC] HELPER: Loops over paired items of two sequences, adding each
            # product into the first argument in place, and returns it.
            for y, z in zip(ys, zs):
                x += y * z
            return x

        ys = [1.0, 2.0]
        zs = [3.0]
        x = torch.tensor([1.0])

        fn_opt = torch.compile(fn, backend="eager")
        fn_opt(x, ys, zs)
        fn_opt(x, ys[:1], zs)

        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 1, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1], "dynamo_hint_overrides": {}, "is_leaf": true, "stride": [1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['x']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_x_": [1], "iadd": [1]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "recompile_reasons", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 1, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1], "dynamo_hint_overrides": {}, "is_leaf": true, "stride": [1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['x']"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_x_": [1], "iadd": [1]}}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
""",
        )

        self.assertParses()

    def test_dump_file(self):
        # [FUNCTION DOC] Symbolically traces an add function with fx, compiles the resulting
        # graph module with eager backend, and compares the trace including the dump_file
        # record, whose generated forward source is appended after the JSON line, against the
        # expected text.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Function traced by fx; returns the first argument added
            # to the second.
            return x.add(y)

        gm = fx.symbolic_trace(f)
        torch.compile(gm, backend="eager")(torch.randn(3), torch.randn(3))

        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dump_file": {"name": "<eval_with_key>"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}



def forward(self, x, y):
    add = x.add(y);  x = y = None
    return add

{"describe_storage": {"id": 0, "describer_id": "ID", "size": 12}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 1, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [3], "dynamo_hint_overrides": {}, "is_leaf": true, "stride": [1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['x']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 1, "describer_id": "ID", "size": 12}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 1, "ndim": 1, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [3], "dynamo_hint_overrides": {}, "is_leaf": true, "stride": [1], "storage": 1, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 1, "source": "L['y']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_x_": [3], "l_y_": [3], "add": [3]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
""",
        )

    @requires_tlparse
    @torch._inductor.config.patch("fx_graph_cache", True)
    def test_codecache(self):
        # [FUNCTION DOC] With the fx graph cache enabled, compiles a sine function with
        # inductor, runs it, resets dynamo and runs it again to hit the cache, then compares
        # the trace (which should show output code twice) against the expected text and checks
        # tlparse parsing.
        def fn(a):
            # [FUNCTION DOC] HELPER: Compile target returning the sine of the input.
            return a.sin()

        x = torch.tensor([1.0])
        fn_opt = torch.compile(fn, backend="inductor")
        fn_opt(x)
        torch._dynamo.reset()
        # Trigger a cache hit
        fn_opt(x)

        # Should print twice, including inductor_output_code
        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 1, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1], "dynamo_hint_overrides": {}, "is_leaf": true, "stride": [1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_a_": [1], "sin": [1]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_pre_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "after_pre_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_inference_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "torch._functorch.config", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_joint_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "after_joint_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_post_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "inductor_post_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME", "file_path": "FILENAME"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "runtime_wrapper_orchestration", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 1, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1], "dynamo_hint_overrides": {}, "is_leaf": true, "stride": [1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_a_": [1], "sin": [1]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_inference_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_post_grad_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME", "file_path": "FILENAME"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "inductor_provenance_tracking_node_mappings", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "inductor_provenance_tracking_kernel_stack_traces", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "fx_graph_cache_hit", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "runtime_wrapper_orchestration", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_hit", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
""",
        )
        self.assertParses()

    @requires_tlparse
    def test_make_fx_fail_partial(self):
        # [FUNCTION DOC] Runs make_fx on a function that raises RuntimeError after one add,
        # swallowing the error, while a second handler captures the make_fx_fail_partial
        # payload. Compares the main trace and the partial graph payload text (the add node)
        # against expected text. The extra handler is never removed from the logger (leaks
        # into later tests in the process).
        from torch.fx.experimental.proxy_tensor import make_fx

        payload_buffer = io.StringIO()
        payload_handler = logging.StreamHandler(payload_buffer)
        payload_handler.setFormatter(StructuredTracePayloadFormatter())
        payload_handler.addFilter(StructuredTraceTestingFilter("make_fx_fail_partial"))
        trace_log.addHandler(payload_handler)

        def f(x):
            # [FUNCTION DOC] HELPER: Function traced by make_fx; computes an unused add and
            # then raises RuntimeError.
            y = x + 1  # noqa: F841
            raise RuntimeError("boo")

        try:
            make_fx(f)(torch.randn(2))
        except RuntimeError:
            pass

        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"artifact": {"name": "make_fx_fail_partial", "encoding": "string"}, "stack": "STACK", "has_payload": "HASH"}
""",
        )

        self.assertExpectedInline(
            payload_buffer.getvalue(),
            """\
def forward(self, x_1: "f32[2][1]cpu"):
    # No stacktrace found for following nodes
    add: "f32[2][1]cpu" = torch.ops.aten.add.Tensor(x_1, 1);  x_1 = add = None
""",
        )

    @requires_tlparse
    @torch._inductor.config.patch("fx_graph_cache", True)
    @show_chrome_events
    def test_chromium_event(self):
        # [FUNCTION DOC] With chrome events shown and the fx graph cache enabled, compiles and
        # runs a sine function twice (resetting dynamo between to hit the cache), checks
        # tlparse parsing, and asserts the captured buffer contains a specific chromium_event
        # record line.
        def fn(a):
            # [FUNCTION DOC] HELPER: Compile target returning the sine of the input.
            return a.sin()

        x = torch.tensor([1.0])
        fn_opt = torch.compile(fn, backend="inductor")
        fn_opt(x)
        torch._dynamo.reset()
        # Trigger a cache hit
        fn_opt(x)
        # Should print twice, including inductor_output_code
        self.assertParses()
        chromium_event = (
            '{"chromium_event": {}, "frame_id": 0, "frame_compile_id": 0, '
            '"attempt": 0, "has_payload": "HASH"}'
        )
        self.assertTrue(chromium_event in self.buffer.getvalue())

    @requires_tlparse
    @torch._dynamo.config.patch("compiled_autograd", True)
    @torch._inductor.config.patch("fx_graph_cache", True)
    @show_chrome_events
    def test_compiled_autograd_id(self):
        # [FUNCTION DOC] With compiled autograd, chrome events and fx graph cache enabled,
        # runs a function that calls backward through the dynamo optimize inductor wrapper,
        # resets dynamo, swaps the raw log handler for a fresh temporary file (so tlparse sees
        # only the cache hit run), runs again and checks tlparse parsing. Asserts the buffer
        # has chromium_event records with frame ids and a compiled_autograd_graph record, then
        # that a chromium event carrying that graph's compiled autograd id exists.
        def fn(a):
            # [FUNCTION DOC] HELPER: Compile target that computes the sine sum of the input
            # and runs backward on it.
            return a.sin().sum().backward()

        x = torch.tensor([1.0], requires_grad=True)
        fn_opt = torch._dynamo.optimize("inductor")(fn)
        fn_opt(x)
        torch._dynamo.reset()
        # Reset raw log so assertParses only validates the cache-hit compilation.
        # Without this, the raw log contains two compilations with identical
        # compiled_autograd_id=0 (reset() restarts COMPILE_COUNTER from zero),
        # which tlparse --strict rejects as duplicate compiled_autograd_id.
        trace_log.removeHandler(self.raw_handler)
        self.raw_file.close()
        self.raw_file = tempfile.NamedTemporaryFile(mode="w", delete=True)  # noqa: SIM115
        self.raw_handler = logging.StreamHandler(self.raw_file)
        self.raw_handler.setFormatter(TorchLogsFormatter(trace=True))
        trace_log.addHandler(self.raw_handler)
        # Trigger a cache hit
        fn_opt(x)
        # Verify the cache-hit trace (including inductor_output_code) parses cleanly
        self.assertParses()
        logs = self.buffer.getvalue()
        self.assertRegex(
            logs,
            r'\{"chromium_event": \{\}, "frame_id": \d+, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"\}',
        )
        graph_event = re.search(
            r'\{"compiled_autograd_graph": \{\}, "compiled_autograd_id": (\d+), "attempt": 0, "has_payload": "HASH"\}',
            logs,
        )
        self.assertIsNotNone(graph_event)
        self.assertRegex(
            logs,
            r'\{"chromium_event": \{\}, "compiled_autograd_id": '
            + graph_event.group(1)
            + r', "frame_id": \d+, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"\}',
        )

    @requires_tlparse
    @torch._dynamo.config.patch("compiled_autograd", True)
    def test_compiled_autograd_attribution(self):
        # [FUNCTION DOC] With compiled autograd enabled, compiles a function running a custom
        # autograd function (with print graph breaks in backward) over six input sizes. Parses
        # the dynamo_start records, groups frame ids by compiled autograd id, and asserts
        # exactly two ids exist, the first with at least three frames and the second with at
        # least one. Also checks tlparse parsing.
        # multiple dynamo recompiles should still be attributed to the parent compiled autograd id
        def fn():
            # [FUNCTION DOC] HELPER: Defines the custom sine autograd function and loops over
            # six sizes, applying it, running backward on the sum, and collecting the
            # gradients; returns the list.
            class MySin(torch.autograd.Function):
                # [CLASS DOC] HELPER: Custom autograd function computing sine; its backward
                # prints twice (forcing graph breaks) and multiplies the incoming gradient by
                # cosine of the saved input.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Saves the input for backward and returns its
                    # sine.
                    ctx.save_for_backward(x)
                    return torch.sin(x)

                @staticmethod
                def backward(ctx, gO):
                    # [FUNCTION DOC] HELPER: Prints a message, loads the saved input, prints
                    # again, and returns the incoming gradient times the cosine of the input.
                    print("graph break")
                    (x,) = ctx.saved_tensors
                    print("graph break")
                    return gO * torch.cos(x)

            grads = []
            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = MySin.apply(x)
                loss = out.sum()
                loss.backward()
                grads.append(x.grad)

            return grads

        fn_opt = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
        fn_opt()
        self.assertParses()
        logs = self.buffer.getvalue()
        pattern = (
            r'\{"dynamo_start": \{"stack": "STACK"\}(?:, "compiled_autograd_id": (\d+))?, '
            r'"frame_id": (\d+), "frame_compile_id": 0, "attempt": 0\}'
        )
        starts = re.findall(pattern, logs)
        ca_frames = {}
        for ca_id, frame in starts:
            if ca_id:
                ca_frames.setdefault(ca_id, set()).add(frame)
        self.assertEqual(len(ca_frames), 2)
        first_id, second_id = sorted(ca_frames, key=int)
        self.assertGreaterEqual(len(ca_frames[first_id]), 3)
        self.assertGreaterEqual(len(ca_frames[second_id]), 1)

    @requires_tlparse
    @show_chrome_events
    def test_compiled_autograd_chromium(self):
        # [FUNCTION DOC] With chrome events shown, enables compiled autograd with
        # torch.compile and runs backward for six input sizes, checks tlparse parsing, and
        # asserts the buffer contains three expected chromium_event lines (for the autograd
        # graph and its two frame compiles).
        with torch._dynamo.compiled_autograd._enable(torch.compile):
            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                loss = x.sum()
                loss.backward()

        self.assertParses()
        expected = [
            '{"chromium_event": {}, "compiled_autograd_id": 0, "attempt": 0, "has_payload": "HASH"}',
            '{"chromium_event": {}, "compiled_autograd_id": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, '
            '"has_payload": "HASH"}',
            '{"chromium_event": {}, "compiled_autograd_id": 0, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, '
            '"has_payload": "HASH"}',
        ]
        logs = self.buffer.getvalue()
        self.assertTrue(all(event in logs for event in expected))

    def test_recompile_user_contexts(self):
        # [FUNCTION DOC] Registers a recompile user context hook that counts its calls,
        # compiles an add function, runs it with repeated shapes then varied sizes, and
        # asserts the hook ran once after the first compile and twice after the one recompile.
        # test that user_context is called only once per recompile
        num_calls = 0

        def f(x):
            # [FUNCTION DOC] HELPER: Function that is compiled by the test; returns the input
            # plus one.
            return x + 1

        f = torch.compile(f)  # noqa: UNSPECIFIED_BACKEND

        def user_context() -> str:
            # [FUNCTION DOC] HELPER: Hook that increments the enclosing call counter and
            # returns a string containing the new count.
            nonlocal num_calls
            num_calls += 1
            return "user_context: " + str(num_calls)

        torch._dynamo.register_hook_for_recompile_user_context(user_context)

        for _ in range(10):
            f(torch.randn(1, 5))

        # first compile
        self.assertEqual(num_calls, 1)

        for i in range(2, 10):
            f(torch.randn(i, 5))

        # first compile + recompile once
        self.assertEqual(num_calls, 2)

    def test_recompile_user_contexts_iteration(self):
        # [FUNCTION DOC] Registers a recompile user context hook that reports the current step
        # of a Step object, compiles an add function, and runs it ten times with changing
        # sizes while advancing the step each time. Only exercises the path; no assertions.
        class Step:
            # [CLASS DOC] HELPER: Simple counter holder with a step attribute starting at zero
            # and a method that increments it.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets the step counter to zero.
                self.step = 0

            def next_step(self):
                # [FUNCTION DOC] HELPER: Increments the step counter by one.
                self.step += 1

        step = Step()

        def f(x):
            # [FUNCTION DOC] HELPER: Function that is compiled by the test; returns the input
            # plus one.
            return x + 1

        f = torch.compile(f)  # noqa: UNSPECIFIED_BACKEND

        def user_context() -> str:
            # [FUNCTION DOC] HELPER: Hook returning a string containing the current step
            # value.
            return "user_context: " + str(step.step)

        torch._dynamo.register_hook_for_recompile_user_context(user_context)

        for i in range(10):
            f(torch.randn(i + 2 // 3, 5))
            step.next_step()

    @contextmanager
    def _setup_collective_schedule_capture(self):
        # [FUNCTION DOC] HELPER: Context manager that adds a handler capturing only
        # inductor_collective_schedule payloads into a string buffer, yields that buffer, and
        # removes the handler in a finally block.
        """Helper to turn on and capture the 'inductor_collective_schedule' structured trace."""
        payload_buffer = io.StringIO()
        payload_handler = logging.StreamHandler(payload_buffer)
        payload_handler.setLevel(logging.DEBUG)
        payload_handler.setFormatter(StructuredTracePayloadFormatter())
        payload_handler.addFilter(
            StructuredTraceTestingFilter("inductor_collective_schedule")
        )
        trace_log.addHandler(payload_handler)
        try:
            yield payload_buffer
        finally:
            trace_log.removeHandler(payload_handler)

    @requires_tlparse
    def test_collective_schedule_empty(self):
        # [FUNCTION DOC] Calls the inductor debug collective schedule logger with an empty
        # list and asserts that no collective schedule artifact appears in the main buffer and
        # the captured payload is empty.
        """Verify logging when no collective kernels are present (empty schedule)."""
        with self._setup_collective_schedule_capture() as payload_buffer:
            from torch._inductor.debug import log_collective_schedule

            log_collective_schedule([])

            # With no collectives, artifact should not be logged and payload should be empty
            self.assertNotIn('"inductor_collective_schedule"', self.buffer.getvalue())
            self.assertEqual(payload_buffer.getvalue().strip(), "")

    @requires_tlparse
    @requires_distributed()
    @torch._inductor.config.patch("fx_graph_cache", False)
    def test_collective_schedule_real(self):
        # [FUNCTION DOC] Initializes a fake process group of two ranks with a fake store,
        # compiles a module using functional all_reduce and wait_tensor with inductor with the
        # capture context, and asserts the logged schedule is a list matching the expected
        # collective op names. Checks tlparse parsing and always destroys the process group.
        """Test collective schedule with _c10d_functional ops that work with FakeStore."""
        import torch.distributed as dist

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        class CollectiveModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward applies functional all_reduce,
            # wait_tensor and multiplies by two.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs functional all_reduce with sum, waits on the
                # tensor, and returns the result doubled.
                # Use _c10d_functional ops that actually trigger collective kernels
                y = torch.ops._c10d_functional.all_reduce.default(x, "sum", "0")
                y = torch.ops._c10d_functional.wait_tensor.default(y)
                return y * 2

        try:
            with self._setup_collective_schedule_capture() as payload_buffer:
                torch._dynamo.reset()

                mod = CollectiveModule()
                compiled = torch.compile(mod, backend="inductor")

                compiled(torch.randn(4, 4))

                # Verify collective schedule artifact was logged
                self.assertIn('"inductor_collective_schedule"', self.buffer.getvalue())

                payload_content = payload_buffer.getvalue().strip()
                schedule = json.loads(payload_content)
                self.assertIsInstance(schedule, list)

                # Verify expected collective operations are present
                self.assertExpectedInline(
                    str(schedule),
                    """\
['torch.ops._c10d_functional.all_reduce_.default', 'torch.ops._c10d_functional.wait_tensor.default']\
""",
                )
                self.assertParses()
        finally:
            dist.destroy_process_group()

    @contextmanager
    def _setup_runtime_estimates_capture(self):
        # [FUNCTION DOC] HELPER: Context manager that adds a handler capturing only
        # inductor_runtime_and_tensor_meta payloads into a string buffer, yields that buffer,
        # and removes the handler in a finally block.
        """Helper to turn on and capture the combined 'inductor_runtime_and_tensor_meta' structured trace."""
        payload_buffer = io.StringIO()
        payload_handler = logging.StreamHandler(payload_buffer)
        payload_handler.setLevel(logging.DEBUG)
        payload_handler.setFormatter(StructuredTracePayloadFormatter())
        payload_handler.addFilter(
            StructuredTraceTestingFilter("inductor_runtime_and_tensor_meta")
        )
        trace_log.addHandler(payload_handler)
        try:
            yield payload_buffer
        finally:
            trace_log.removeHandler(payload_handler)

    @requires_tlparse
    @requires_distributed()
    @requires_gpu_and_triton
    @torch._inductor.config.patch("fx_graph_cache", False)
    @torch._inductor.config.patch("log_tlparse", True)
    def test_runtime_estimates_simple(self):
        # [FUNCTION DOC] Initializes a fake two rank process group, compiles a linear plus
        # relu plus all_reduce module on the accelerator with inductor, and asserts the
        # runtime and tensor meta artifact was logged. If a payload was captured, checks there
        # is at least one compute or collective op and every op has a non-null estimated
        # runtime. Checks tlparse parsing; destroys the process group in a finally block.
        """Test runtime estimates logging with simple compute and collective ops."""
        import torch.distributed as dist

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        class SimpleModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one 4x4 linear layer; forward applies linear,
            # relu, all_reduce and wait_tensor.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the module with a 4x4 linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies linear and relu, then functional all_reduce
                # with sum and wait_tensor, and returns the result.
                h = self.linear(x)
                h = torch.relu(h)

                h = torch.ops._c10d_functional.all_reduce.default(h, "sum", "0")
                h = torch.ops._c10d_functional.wait_tensor.default(h)
                return h

        try:
            with self._setup_runtime_estimates_capture() as payload_buffer:
                torch._dynamo.reset()

                mod = SimpleModule().to(device_type)
                compiled = torch.compile(mod, backend="inductor")
                compiled(torch.randn(4, 4, device=device_type))

                # Verify runtime + tensor meta artifact was logged
                self.assertIn(
                    '"inductor_runtime_and_tensor_meta"', self.buffer.getvalue()
                )

                payload_content = payload_buffer.getvalue().strip()
                if payload_content:
                    data = json.loads(payload_content)
                    self.assertIn("ops", data)
                    ops = data["ops"]

                    # Verify runtime estimates
                    compute_ops = [op for op in ops if op["type"] == "compute"]
                    collective_ops = [op for op in ops if op["type"] == "collective"]

                    self.assertTrue(len(compute_ops) > 0 or len(collective_ops) > 0)

                    # Just check each op has an estimated runtime value (any value, including 0)
                    for op in ops:
                        self.assertIn("estimated_runtime_ns", op)
                        self.assertIsNotNone(op["estimated_runtime_ns"])

                self.assertParses()
        finally:
            dist.destroy_process_group()

    @requires_tlparse
    @requires_distributed()
    @requires_gpu_and_triton
    @torch._inductor.config.patch("fx_graph_cache", False)
    @torch._inductor.config.patch("log_tlparse", True)
    def test_runtime_estimates_mixed(self):
        # [FUNCTION DOC] Initializes a fake two rank process group, compiles a module mixing
        # layer norm, gelu, all_reduce, scaling, all_gather and a sum with inductor, and
        # asserts the runtime artifact was logged. If a payload was captured, asserts both
        # compute and collective op types appear and every op has a non-null estimated
        # runtime. Checks tlparse parsing; destroys the process group in a finally block.
        """Test runtime estimates logging with mixed compute and collective sequence."""
        import torch.distributed as dist

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        class MixedModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module mixing compute and collective ops: layer norm, gelu,
            # all_reduce, scale by half, all_gather and a sum over the first dimension.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the module with a layer norm over four
                # features.
                super().__init__()
                self.norm = torch.nn.LayerNorm(4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies layer norm and gelu, all_reduce and wait,
                # scales by 0.5, all_gather into tensor and wait, then returns the sum over
                # dimension zero.
                h = self.norm(x)
                h = torch.nn.functional.gelu(h)

                h = torch.ops._c10d_functional.all_reduce.default(h, "sum", "0")
                h = torch.ops._c10d_functional.wait_tensor.default(h)

                h = h * 0.5

                gathered = torch.ops._c10d_functional.all_gather_into_tensor.default(
                    h, 2, "0"
                )
                gathered = torch.ops._c10d_functional.wait_tensor.default(gathered)

                return gathered.sum(dim=0)

        try:
            with self._setup_runtime_estimates_capture() as payload_buffer:
                torch._dynamo.reset()

                mod = MixedModule().to(device_type)
                compiled = torch.compile(mod, backend="inductor")
                compiled(torch.randn(4, 4, device=device_type))

                # Verify artifact was logged
                self.assertIn(
                    '"inductor_runtime_and_tensor_meta"', self.buffer.getvalue()
                )

                payload_content = payload_buffer.getvalue().strip()
                if payload_content:
                    data = json.loads(payload_content)
                    self.assertIn("ops", data)
                    ops = data["ops"]

                    # Should have both compute and collective ops
                    op_types = {op["type"] for op in ops}
                    self.assertIn("compute", op_types)
                    self.assertIn("collective", op_types)

                    # Just check each op has an estimated runtime value (any value, including 0)
                    for op in ops:
                        self.assertIn("estimated_runtime_ns", op)
                        self.assertIsNotNone(op["estimated_runtime_ns"])

                self.assertParses()
        finally:
            dist.destroy_process_group()

    @requires_tlparse
    @requires_distributed()
    @requires_gpu_and_triton
    @torch._inductor.config.patch("fx_graph_cache", False)
    @torch._inductor.config.patch("log_tlparse", True)
    def test_tensor_metadata_logging_multiple_ops(self):
        # [FUNCTION DOC] Initializes a fake two rank process group, compiles a linear relu
        # all_reduce plus one module, and if a runtime payload was captured asserts the sorted
        # set of op types equals collective and compute. Checks tlparse parsing; destroys the
        # process group in a finally block.
        import torch.distributed as dist

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        class Mixed(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer whose forward applies linear,
            # relu, all_reduce, wait_tensor and adds one.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the module with a 4x4 linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies linear and relu, functional all_reduce with
                # sum and wait_tensor, and returns the result plus one.
                y = torch.relu(self.linear(x))
                y = torch.ops._c10d_functional.all_reduce.default(y, "sum", "0")
                y = torch.ops._c10d_functional.wait_tensor.default(y)
                return y + 1

        try:
            with self._setup_runtime_estimates_capture() as payload_buffer:
                torch._dynamo.reset()
                mod = Mixed().to(device_type)
                compiled = torch.compile(mod, backend="inductor")
                compiled(torch.randn(4, 4, device=device_type))
                payload = payload_buffer.getvalue().strip()
                if payload:
                    data = json.loads(payload)
                    types = sorted({op.get("type") for op in data.get("ops", [])})
                    self.assertExpectedInline(
                        str(types), """['collective', 'compute']"""
                    )
                self.assertParses()
        finally:
            dist.destroy_process_group()

    @requires_tlparse
    @torch._inductor.config.patch("log_tlparse", True)
    def test_tensor_metadata_logging(self):
        # [FUNCTION DOC] With tlparse logging on, compiles a transpose, mean and half
        # precision cast function with fullgraph inductor, asserts the runtime artifact was
        # logged, simplifies the captured ops to type and output shape, stride and dtype, and
        # compares the last such op against the expected compute op with float16 output.
        # Checks tlparse parsing.
        """Emit unified runtime+tensor-metadata artifact and assert a stable simplified JSON inline."""
        with self._setup_runtime_estimates_capture() as payload_buffer:

            def f(x):
                # [FUNCTION DOC] HELPER: Transposes the input, takes the mean over dimension
                # zero, casts to float16 and returns it.
                y = x.transpose(0, 1)
                z = y.mean(dim=0)
                w = z.to(torch.float16)
                return w

            compiled = torch.compile(f, backend="inductor", fullgraph=True)
            compiled(torch.ones(2, 3))

            # Verify artifact was logged
            self.assertIn('"inductor_runtime_and_tensor_meta"', self.buffer.getvalue())

            payload = payload_buffer.getvalue().strip()
            if payload:
                data = json.loads(payload)
                ops = data.get("ops", [])

                simplified_ops = []
                for op in ops:
                    outs = [
                        {
                            "shape": out.get("shape", []),
                            "stride": out.get("stride", []),
                            "dtype": out.get("dtype", None),
                        }
                        for out in op.get("outputs", [])
                    ]
                    if outs:
                        simplified_ops.append(
                            {
                                "type": op.get("type", ""),
                                "outputs": outs,
                            }
                        )

                self.assertExpectedInline(
                    {"ops": simplified_ops[-1:]} if simplified_ops else {"ops": []},
                    """{'ops': [{'type': 'compute', 'outputs': [{'shape': [2], 'stride': [1], 'dtype': 'float16'}]}]}""",
                )

            self.assertParses()

    @requires_tlparse
    @torch._inductor.config.patch("log_tlparse", True)
    def test_tensor_metadata_logging_dynamic_shapes(self):
        # [FUNCTION DOC] Same as the metadata logging test but compiles with dynamic shapes;
        # compares the last simplified op against an expected compute op with float32 and
        # float16 outputs. Checks tlparse parsing.
        """Same as test_tensor_metadata_logging, but with dynamic shapes enabled to cover to_size_hints."""
        with self._setup_runtime_estimates_capture() as payload_buffer:

            def f(x):
                # [FUNCTION DOC] HELPER: Transposes the input, takes the mean over dimension
                # zero, casts to float16 and returns it.
                y = x.transpose(0, 1)
                z = y.mean(dim=0)
                w = z.to(torch.float16)
                return w

            compiled = torch.compile(f, backend="inductor", dynamic=True)
            compiled(torch.ones(2, 3))

            # Verify artifact was logged
            self.assertIn('"inductor_runtime_and_tensor_meta"', self.buffer.getvalue())

            payload = payload_buffer.getvalue().strip()
            if payload:
                data = json.loads(payload)
                ops = data.get("ops", [])

                simplified_ops = []
                for op in ops:
                    outs = [
                        {
                            "shape": out.get("shape", []),
                            "stride": out.get("stride", []),
                            "dtype": out.get("dtype", None),
                        }
                        for out in op.get("outputs", [])
                    ]
                    if outs:
                        simplified_ops.append(
                            {
                                "type": op.get("type", ""),
                                "outputs": outs,
                            }
                        )

                self.assertExpectedInline(
                    {"ops": simplified_ops[-1:]} if simplified_ops else {"ops": []},
                    (
                        "{'ops': [{'type': 'compute', 'outputs': ["
                        "{'shape': [2], 'stride': [1], 'dtype': 'float32'}, "
                        "{'shape': [2], 'stride': [1], 'dtype': 'float16'}]}]}"
                    ),
                )

            self.assertParses()

    @contextmanager
    def _setup_graph_execution_capture(self):
        # [FUNCTION DOC] HELPER: Context manager that adds a handler capturing only
        # graph_execution payloads into a string buffer, yields that buffer, and removes the
        # handler in a finally block.
        """Helper to capture the 'graph_execution' structured trace."""
        payload_buffer = io.StringIO()
        payload_handler = logging.StreamHandler(payload_buffer)
        payload_handler.setLevel(logging.DEBUG)
        payload_handler.setFormatter(StructuredTracePayloadFormatter())
        payload_handler.addFilter(StructuredTraceTestingFilter("graph_execution"))
        trace_log.addHandler(payload_handler)
        try:
            yield payload_buffer
        finally:
            trace_log.removeHandler(payload_handler)

    @requires_tlparse
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_graph_execution_order(self):
        # [FUNCTION DOC] With caches disabled, compiles a function with a graph break using
        # inductor and runs it inside the inductor debug context that records and logs graph
        # execution order. Parses the captured payload, asserts each compile_id entry is a
        # string, compares the JSON to the expected order of compile ids zero slash zero and
        # one slash zero, and checks tlparse parsing.
        """Verify graph execution order is aggregated into a single artifact."""
        torch._dynamo.reset()
        with self._setup_graph_execution_capture() as payload_buffer:

            def fn(x):
                # [FUNCTION DOC] HELPER: Compile target that adds one, triggers a graph break,
                # then adds two.
                y = x + 1
                torch._dynamo.graph_break()
                return y + 2

            compiled = torch.compile(fn, backend="inductor")
            from torch._inductor.debug import record_and_log_graph_execution_order

            with record_and_log_graph_execution_order():
                compiled(torch.randn(1))

            payload_content = payload_buffer.getvalue().strip()
            payload = json.loads(payload_content)
            executions = payload["graph_execution_order"]
            self.assertTrue(all(isinstance(e["compile_id"], str) for e in executions))
            self.assertExpectedInline(
                json.dumps(payload),
                """{"graph_execution_order": [{"compile_id": "0/0"}, {"compile_id": "1/0"}]}""",
            )
            self.assertParses()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
