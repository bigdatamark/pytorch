# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for the torch logging system as exercised through Dynamo and Inductor.
# Defines helpers that normalize log text (shape guards, global-state guard, log line prefixes,
# rank prefixes), small functions used as compile targets, and factories that build logging tests
# for a given record-count expectation. LoggingTests checks per-artifact record counts, recompile
# hints, side-effect logs, trace-source and trace-call logs, guard logs, custom formatting, log
# registration and subprocess-based TORCH_LOGS behavior. At import time it creates module-level
# flags for GPU/distributed availability and the current accelerator device type, and attaches
# one generated single-record test to LoggingTests for each registered logging artifact that is
# not in the exclusion set.
import contextlib
import functools
import logging
import os
import re
import unittest
import unittest.mock

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch.distributed as dist
from torch._dynamo.testing import (
    empty_line_normalizer,
    extract_graph_and_tracker,
    skipIfNotPy311,
)
from torch._dynamo.trace_rules import _as_posix_path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_cuda import SM90OrLater
from torch.testing._internal.common_utils import (
    find_free_port,
    IS_WINDOWS,
    munge_exc,
    skipIfTorchDynamo,
    skipIfWindows,
)
from torch.testing._internal.inductor_utils import (
    HAS_CUDA_AND_TRITON,
    HAS_XPU_AND_TRITON,
)
from torch.testing._internal.logging_utils import (
    LoggingTestCase,
    make_logging_test,
    make_settings_test,
)
from torch.testing._internal.triton_utils import requires_cuda_and_triton


requires_gpu = unittest.skipUnless(
    HAS_CUDA_AND_TRITON or HAS_XPU_AND_TRITON, "requires cuda or xpu with triton"
)

requires_distributed = functools.partial(
    unittest.skipIf, not dist.is_available(), "requires distributed"
)

device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)


def munge_shape_guards(s: str) -> str:
    # [FUNCTION DOC] Normalize a guard log message so shape guards can be compared with golden
    # text. Picks a regular expression for the shape-guard line prefix depending on whether C++
    # symbolic shape guards are enabled, rewrites source location fragments to a placeholder and
    # the prefix to a fixed marker, and returns only the lines that contained a shape guard,
    # joined by newlines.
    SHAPE_GUARD_REGEX = (
        r"[| ]* \+- SYMBOLIC_SHAPE_GUARD:"
        if torch._dynamo.config.enable_cpp_symbolic_shape_guards
        else r"^\+- LAMBDA_GUARD:"
    )

    def munge(s):
        # [FUNCTION DOC] HELPER: Apply two substitutions to one line: replace location fragments
        # (file, line number and function) with a fixed placeholder, then replace the shape-guard
        # prefix. Returns the rewritten text and the number of prefix replacements.
        s = re.sub(r"[^ ]+:\d+ in [^ ]+", "#:# in #", s)
        return re.subn(SHAPE_GUARD_REGEX, "+- __SHAPE_GUARD__:", s)

    lines = [munge(l) for l in s.splitlines()]
    return "\n".join([line for line, nsubs in lines if nsubs > 0])


def munge_global_state_json(text):
    # [FUNCTION DOC] Reduce the global-state guard line of a guard log to a stable form. Finds
    # the line starting at the global-state marker; returns an empty string if absent. Otherwise
    # repeatedly replaces innermost bracketed lists with a quoted placeholder, then replaces
    # numeric, boolean and quoted-string values that follow a colon with the same placeholder,
    # and returns that line.
    import re

    match = re.search(r"\+- GLOBAL_STATE:.*", text)
    if not match:
        return ""

    line = match.group(0)
    while "[" in line:
        line = re.sub(r"\[[^\[\]]*\]", '"#"', line)

    line = re.sub(r':\s*(\d+|true|false|"[^"]*")', r': "#"', line)
    return line


LOG_PREFIX_PATTERNS = [
    re.compile(r"^\[rank\d+\]:\s*"),
    re.compile(r"^[A-Z]+:[^:]+:\s*"),
    re.compile(r"^[A-Z]\d{2,4}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?\s+\d+\s+[^\]]+\]\s*"),
    re.compile(r"^[A-Z](?:\d{4})?\s+[^:]+:\s*"),
]


def normalize_log_line(line: str) -> str:
    # [FUNCTION DOC] Strip trailing whitespace from a formatted log line, then remove the first
    # matching known log prefix (rank tag, level/name tag, glog-style timestamp prefix, or short
    # level prefix) and leading whitespace. At most one pattern is applied. Returns the line.
    line = line.rstrip()
    for pattern in LOG_PREFIX_PATTERNS:
        stripped, count = pattern.subn("", line, count=1)
        if count:
            line = stripped.lstrip()
            break
    return line


def normalize_rank_prefix(output: str) -> str:
    # [FUNCTION DOC] Return the output unchanged if it already contains a rank tag. Otherwise
    # rewrite each line-leading prefix made of upper-case level, a name and a colon into a rank-
    # zero tag, dropping the original prefix, and return the result.
    if "[rank" in output:
        return output

    def repl(match):
        # [FUNCTION DOC] HELPER: Regex replacement callback: returns the preserved line start
        # group followed by the rank-zero tag.
        prefix = match.group(1)
        return f"{prefix}[rank0]: "

    return re.sub(r"(^|\n)(?:[A-Z]+:[^:]+:)", repl, output)


def example_fn(a):
    # [FUNCTION DOC] HELPER: Compile target: multiplies the input by a 1000 by 1000 tensor of
    # ones, adds another such tensor, and returns the result.
    output = a.mul(torch.ones(1000, 1000))
    output = output.add(torch.ones(1000, 1000))
    return output


def dynamo_error_fn(a):
    # [FUNCTION DOC] HELPER: Compile target that fails at trace time: multiplies by a 1000 by
    # 1000 ones tensor, then adds a 10 by 10 ones tensor, which cannot broadcast against the
    # first.
    output = a.mul(torch.ones(1000, 1000))
    output = output.add(torch.ones(10, 10))
    return output


def inductor_error_fn(a):
    # [FUNCTION DOC] HELPER: Compile target that rounds its input elementwise and returns it.
    output = torch.round(a)
    return output


def inductor_schedule_fn(a):
    # [FUNCTION DOC] HELPER: Compile target: adds a 1000 by 1000 ones tensor created on the
    # current accelerator device type (or cpu) to the input and returns the result.
    output = a.add(torch.ones(1000, 1000, device=device_type))
    return output


ARGS = (torch.ones(1000, 1000, requires_grad=True),)


def multi_record_test(num_records, **kwargs):
    # [FUNCTION DOC] HELPER: Factory returning a logging test method. The keyword arguments are
    # the log settings; the generated test compiles the example function with the inductor
    # backend, runs it on the module-level arguments and asserts that exactly num_records log
    # records were gathered.
    @make_logging_test(**kwargs)
    def fn(self, records):
        # [FUNCTION DOC] HELPER: Generated test body: receives the test case and gathered records
        # list, compiles and runs the example function, and asserts the record count equals the
        # expected value.
        fn_opt = torch.compile(example_fn, backend="inductor")
        fn_opt(*ARGS)
        self.assertEqual(len(records), num_records)

    return fn


def within_range_record_test(num_records_lower, num_records_higher, **kwargs):
    # [FUNCTION DOC] HELPER: Factory like the exact-count one, but the generated test asserts
    # that the number of gathered records lies in the inclusive range between the given lower and
    # upper bounds.
    @make_logging_test(**kwargs)
    def fn(self, records):
        # [FUNCTION DOC] HELPER: Generated test body: compiles and runs the example function with
        # the inductor backend and asserts the record count is within the inclusive bounds.
        fn_opt = torch.compile(example_fn, backend="inductor")
        fn_opt(*ARGS)
        self.assertGreaterEqual(len(records), num_records_lower)
        self.assertLessEqual(len(records), num_records_higher)

    return fn


def single_record_test(**kwargs):
    # [FUNCTION DOC] HELPER: Shortcut for the exact-count factory with one expected record.
    return multi_record_test(1, **kwargs)


class LoggingTests(LoggingTestCase):
    # [CLASS DOC] Main logging test case, built on LoggingTestCase (record gathering helpers).
    # Many tests are produced by make_logging_test, which runs the body once with settings from
    # the environment variable and once through the set_logs API. Some tests are class attributes
    # built by the record-count factories, and extra single-record tests are attached at import
    # time for every remaining logging artifact.
    test_bytecode = multi_record_test(2, bytecode=True)
    test_output_code = multi_record_test(3, output_code=True)
    test_aot_graphs = multi_record_test(3, aot_graphs=True)

    @requires_gpu
    @make_logging_test(schedule=True)
    def test_schedule(self, records):
        # [FUNCTION DOC] Requires GPU. With schedule logging, compiles the schedule target with
        # inductor, runs it, and asserts more than zero and fewer than five records.
        fn_opt = torch.compile(inductor_schedule_fn, backend="inductor")
        fn_opt(torch.ones(1000, 1000, device=device_type))
        self.assertGreater(len(records), 0)
        self.assertLess(len(records), 5)

    @requires_gpu
    @make_logging_test(fusion=True)
    def test_fusion(self, records):
        # [FUNCTION DOC] Requires GPU. With fusion logging, compiles and runs the schedule target
        # and asserts the record count is positive and below eight times one plus the loop-
        # ordering-after-fusion config value.
        fn_opt = torch.compile(inductor_schedule_fn, backend="inductor")
        fn_opt(torch.ones(1000, 1000, device=device_type))
        self.assertGreater(len(records), 0)

        # LOAF will add an extra round of fusion and result in more logs
        self.assertLess(
            len(records), 8 * (1 + torch._inductor.config.loop_ordering_after_fusion)
        )

    @requires_cuda_and_triton
    @make_logging_test(cudagraphs=True)
    def test_cudagraphs(self, records):
        # [FUNCTION DOC] Requires CUDA and Triton. With cudagraphs logging, compiles the schedule
        # target in reduce-overhead mode, runs it and asserts the record count is positive and
        # below eight.
        fn_opt = torch.compile(mode="reduce-overhead")(inductor_schedule_fn)  # noqa: UNSPECIFIED_BACKEND
        fn_opt(torch.ones(1000, 1000, device=device_type))
        self.assertGreater(len(records), 0)
        self.assertLess(len(records), 8)

    @make_logging_test(recompiles=True)
    def test_recompiles(self, records):
        # [FUNCTION DOC] With recompile logging, compiles a chain of four nested functions (the
        # innermost loops over zipped lists accumulating products into the input) with the eager
        # backend, calls it with lists of different length to force a recompile, then replaces
        # quoted file paths in the joined messages and asserts the expected user stack trace
        # excerpt appears.
        # KNOWN ISSUE: the expected excerpt hard-codes source line numbers of this file, so any
        # insertion of lines above (including comments) shifts them and the assertion would fail
        # unless the numbers are updated.
        def outmost_fn(x, ys, zs):
            # [FUNCTION DOC] HELPER: Calls the next function in the chain with the same
            # arguments.
            return outer_fn(x, ys, zs)

        def outer_fn(x, ys, zs):
            # [FUNCTION DOC] HELPER: Calls the next function in the chain with the same
            # arguments.
            return fn(x, ys, zs)

        def fn(x, ys, zs):
            # [FUNCTION DOC] HELPER: Calls the innermost function with the same arguments.
            return inner(x, ys, zs)

        def inner(x, ys, zs):
            # [FUNCTION DOC] HELPER: Adds the elementwise product of each pair from the two lists
            # to the input tensor in place and returns it.
            for y, z in zip(ys, zs):
                x += y * z
            return x

        ys = [1.0, 2.0, 3.0]
        zs = [3.0]
        x = torch.tensor([1.0])

        fn_opt = torch.compile(outmost_fn, backend="eager")
        fn_opt(x, ys, zs)
        fn_opt(x, ys[:1], zs)

        record_str = re.sub(
            r'"[^"]*"',
            "[file_path]",
            "\n".join(r.getMessage() for r in records),
        )
        self.assertIn(
            """\
    - User stack trace:
    -   File [file_path], line 199, in outmost_fn
    -     return outer_fn(x, ys, zs)
    -   File [file_path], line 202, in outer_fn
    -     return fn(x, ys, zs)
    -   File [file_path], line 205, in fn
    -     return inner(x, ys, zs)
    -   File [file_path], line 208, in inner
    -     for y, z in zip(ys, zs):""",
            record_str,
        )

    @make_logging_test(recompiles=True)
    def test_recompiles_closure_variable_hint(self, records):
        # [FUNCTION DOC] With recompile logging, compiles a function calling a closure; calls it
        # with two closures differing in the second captured value, and asserts the recompile log
        # contains a hint naming the guarded closure variable n2.
        def make_cl(n1, n2):
            # [FUNCTION DOC] HELPER: Builds and returns a closure that adds both captured numbers
            # to its input.
            def inner(x):
                # [FUNCTION DOC] HELPER: Returns the input plus the two captured numbers.
                return x + n1 + n2

            return inner

        @torch.compile(backend="eager")
        def fn(cl, x):
            # [FUNCTION DOC] HELPER: Compiled function (eager backend) that calls the given
            # closure on x.
            return cl(x)

        fn(make_cl(0, 1), torch.ones(3))
        fn(make_cl(0, 2), torch.ones(3))

        record_str = "\n".join(r.getMessage() for r in records)
        # The recompilation log should include a hint explaining which closure variable
        # the cell_contents refers to
        self.assertIn('(HINT: guard on "n2")', record_str)

    @make_logging_test(recompiles=True)
    def test_recompiles_nested_closure_variable_hint(self, records):
        # [FUNCTION DOC] With recompile logging, builds a mask holder whose function captures
        # another function that captures a value, compiles a function calling it and calls it
        # with two holders whose innermost captured values differ. Asserts the recompile log has
        # a hint guarding on inner_val and includes the nested closure cell path text.
        # block_mask.mask_mod.__closure__[0].cell_contents.__closure__[0].cell_contents
        def make_inner_fn(inner_val):
            # [FUNCTION DOC] HELPER: Builds a closure that adds the captured value to its input.
            def inner(x):
                # [FUNCTION DOC] HELPER: Returns the input plus the captured value.
                return x + inner_val

            return inner

        def make_outer_fn(outer_val, inner_fn):
            # [FUNCTION DOC] HELPER: Builds a closure that applies the given inner function and
            # scales the result by the captured outer value.
            def outer(x):
                # [FUNCTION DOC] HELPER: Returns the inner function applied to the input,
                # multiplied by the captured outer value.
                return inner_fn(x) * outer_val

            return outer

        class BlockMask:
            # [CLASS DOC] HELPER: Simple holder storing a mask function in the attribute
            # mask_mod.
            def __init__(self, mask_mod):
                # [FUNCTION DOC] HELPER: Stores the given function as mask_mod.
                self.mask_mod = mask_mod

        @torch.compile(backend="eager")
        def fn(block_mask, x):
            # [FUNCTION DOC] HELPER: Compiled function (eager backend) that applies the holder's
            # mask function to x.
            return block_mask.mask_mod(x)

        # inner_fn captures inner_val, outer captures (outer_val, inner_fn)
        # This creates: block_mask.mask_mod.__closure__[0].cell_contents.__closure__[0].cell_contents
        bm1 = BlockMask(make_outer_fn(2, make_inner_fn(10)))
        bm2 = BlockMask(make_outer_fn(2, make_inner_fn(20)))

        fn(bm1, torch.ones(3))
        fn(bm2, torch.ones(3))

        record_str = "\n".join(r.getMessage() for r in records)
        # The recompilation log should show the hint for the first closure variable
        self.assertIn('(HINT: guard on "inner_val")', record_str)
        # Verify it shows the full nested path
        self.assertIn("cell_contents.__closure__", record_str)

    @make_logging_test(recompiles=True)
    def test_recompiles_closure_variable_attribute_hint(self, records):
        # [FUNCTION DOC] With recompile logging, compiles a function calling a closure whose
        # captured object is a callable with a scale attribute; calls it with two scales to force
        # a recompile, and asserts the log hint names the closure variable transform followed by
        # the scale attribute.
        # Test when guard is on an attribute of the closure cell contents
        # e.g., block_mask.mask_mod.__closure__[0].cell_contents.__code__
        # The hint should still show which closure variable is involved
        class Transform:
            # [CLASS DOC] HELPER: Callable object multiplying its input by the stored scale.
            def __init__(self, scale):
                # [FUNCTION DOC] HELPER: Stores scale on the instance.
                self.scale = scale

            def __call__(self, x):
                # [FUNCTION DOC] HELPER: Returns the input multiplied by the scale.
                return x * self.scale

        def make_fn(transform):
            # [FUNCTION DOC] HELPER: Builds a closure that applies the captured transform to its
            # input.
            def fn(x):
                # [FUNCTION DOC] HELPER: Returns the captured transform applied to the input.
                return transform(x)

            return fn

        @torch.compile(backend="eager")
        def outer(inner_fn, x):
            # [FUNCTION DOC] HELPER: Compiled function (eager backend) that calls the given
            # function on x.
            return inner_fn(x)

        outer(make_fn(Transform(2)), torch.ones(3))
        outer(make_fn(Transform(3)), torch.ones(3))  # transform.scale changes

        record_str = "\n".join(r.getMessage() for r in records)
        # The hint should show the full path from closure var: "transform".scale
        self.assertIn('(HINT: guard on "transform".scale)', record_str)

    test_dynamo_debug = within_range_record_test(30, 90, dynamo=logging.DEBUG)
    test_dynamo_info = within_range_record_test(2, 10, dynamo=logging.INFO)

    @skipIfTorchDynamo("too slow")
    @make_logging_test(dynamo=logging.DEBUG)
    def test_dynamo_debug_default_off_artifacts(self, records):
        # [FUNCTION DOC] Skipped under Dynamo. With dynamo logging at debug level, compiles and
        # runs the example function with inductor and asserts no records come from the bytecode
        # or output-code artifact loggers, i.e. those artifacts are off by default.
        fn_opt = torch.compile(example_fn, backend="inductor")
        fn_opt(torch.ones(1000, 1000))
        self.assertEqual(len([r for r in records if ".__bytecode" in r.name]), 0)
        self.assertEqual(len([r for r in records if ".__output_code" in r.name]), 0)

    @make_logging_test(hierarchical_compile=True)
    def test_hierarchical_compile(self, records):
        # [FUNCTION DOC] With hierarchical compile logging, compiles a function calling a
        # function marked as a compile region, runs it twice with inductor and asserts at least
        # one record was logged.
        from torch._higher_order_ops.invoke_subgraph import mark_compile_region

        @mark_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Compile region function: doubles its input.
            return x * 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the compile region function on x.
            return gn(x)

        fn_opt = torch.compile(fn, backend="inductor")
        fn_opt(torch.ones(1000, 1000))
        fn_opt(torch.ones(1000, 1000))
        self.assertGreater(len(records), 0)

    @make_logging_test()
    def test_dynamo_error(self, records):
        # [FUNCTION DOC] With default logging settings, compiles the failing dynamo target with
        # inductor, ignores the resulting exception, finds the single record containing the
        # won't-convert text and compares its normalized message to a golden string describing
        # the fake-tensor broadcast failure.
        try:
            fn_opt = torch.compile(dynamo_error_fn, backend="inductor")
            fn_opt(*ARGS)
        except Exception:
            pass
        record = self.getRecord(records, "WON'T CONVERT")
        self.assertExpectedInline(
            munge_exc(record.getMessage()),
            """\
WON'T CONVERT dynamo_error_fn test_logging.py line N
due to:
Traceback (most recent call last):
torch._dynamo.exc.TorchRuntimeError: RuntimeError when making fake tensor call
  Explanation: Dynamo failed to run FX node with fake tensors: call_method add(*(FakeTensor(..., size=(1000, 1000), grad_fn=<MulBackward0>), FakeTensor(..., size=(10, 10))), **{}): got RuntimeError('Attempting to broadcast a dimension of length 10 at -1! Mismatching argument at index 1 had torch.Size([10, 10]); but expected shape should be broadcastable to [1000, 1000]')
  Hint: Your code may result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`.

  Developer debug context:

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4315.html

from user code:
   File "test_logging.py", line N, in dynamo_error_fn
    output = output.add(torch.ones(10, 10))""",
        )

    test_aot = within_range_record_test(2, 6, aot=logging.INFO)
    test_inductor_debug = within_range_record_test(3, 33, inductor=logging.DEBUG)
    test_inductor_info = within_range_record_test(2, 10, inductor=logging.INFO)

    @make_logging_test()
    def test_inductor_error(self, records):
        # [FUNCTION DOC] Patches the inductor lowerings so every lowering whose name contains
        # round raises AssertionError, compiles and runs the rounding target (exceptions
        # swallowed), then compares the normalized won't-convert record to a golden string
        # describing the lowering exception, and finally closes the patch stack.
        # KNOWN ISSUE: the patch stack is closed only at the end of the test, not in a finally
        # block, so a failing assertion leaves the lowerings patched.
        exitstack = contextlib.ExitStack()
        import torch._inductor.lowering

        def throw(x):
            # [FUNCTION DOC] HELPER: Replacement lowering that always raises AssertionError.
            raise AssertionError

        # inject an error in the lowerings
        dict_entries = {}
        for x in list(torch._inductor.lowering.lowerings.keys()):
            if "round" in x.__name__:
                dict_entries[x] = throw

        exitstack.enter_context(
            unittest.mock.patch.dict(torch._inductor.lowering.lowerings, dict_entries)
        )

        try:
            fn_opt = torch.compile(inductor_error_fn, backend="inductor")
            fn_opt(*ARGS)
        except Exception:
            pass
        record = self.getRecord(records, "WON'T CONVERT")
        self.assertExpectedInline(
            munge_exc(record.getMessage()),
            """\
WON'T CONVERT inductor_error_fn test_logging.py line N
due to:
Traceback (most recent call last):
  File "test_logging.py", line N, in throw
    raise AssertionError
torch._inductor.exc.InductorError: LoweringException: AssertionError:
  target: aten.round.default
  args[0]: TensorBox(StorageBox(
    InputBuffer(name='primals_1', layout=FixedLayout('cpu', torch.float32, size=[1000, 1000], stride=[1000, 1]))
  ))AssertionError:
  target: aten.round.default
  args[0]: TensorBox(StorageBox(
    InputBuffer(name='primals_1', layout=FixedLayout('cpu', torch.float32, size=[1000, 1000], stride=[1000, 1]))
  ))
Found from :
   File "test_logging.py", line N, in inductor_error_fn
    output = torch.round(a)""",
        )

        exitstack.close()

    @requires_distributed()
    @requires_cuda_and_triton
    @make_logging_test(ddp_graphs=True)
    def test_ddp_graphs(self, records):
        # [FUNCTION DOC] Requires distributed and CUDA with Triton. With ddp graphs logging, sets
        # the master address and port environment variables (not restored) and starts a one-
        # process gloo group, wraps a toy model in DistributedDataParallel on the first CUDA
        # device with a small bucket size, compiles it with inductor and runs one forward pass,
        # destroys the group, then asserts exactly four records came from the ddp graphs logger.
        class ToyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Toy model: a sequential stack of two 1024 by 1024 linear layers
            # kept in the layers attribute.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds the two-layer linear stack.
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(1024, 1024),
                    torch.nn.Linear(1024, 1024),
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the layer stack to the input.
                return self.layers(x)

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        dist.init_process_group("gloo", rank=0, world_size=1)

        model = DDP(ToyModel().to("cuda:0"), device_ids=[0], bucket_cap_mb=4)
        ddp_model = torch.compile(model, backend="inductor")

        ddp_model(torch.randn(1024, 1024, device="cuda:0"))

        dist.destroy_process_group()
        self.assertEqual(len([r for r in records if "__ddp_graphs" in r.name]), 4)

    # check that logging to a child log of a registered logger
    # does not register it and result in duplicated records
    @make_settings_test("torch._dynamo.output_graph")
    def test_open_registration_with_registered_parent(self, records):
        # [FUNCTION DOC] With only the settings string naming the output-graph logger, logs one
        # info message on that same logger and asserts exactly one record is gathered, i.e.
        # logging to an already registered logger is not duplicated.
        logger = logging.getLogger("torch._dynamo.output_graph")
        logger.info("hi")
        self.assertEqual(len(records), 1)

    # check logging to a random log that is not a child log of a registered
    # logger registers it and sets handlers properly
    @make_settings_test("torch.utils")
    def test_open_registration(self, records):
        # [FUNCTION DOC] With the settings string naming the torch.utils logger (not registered
        # as an artifact), logs one info message to it and asserts exactly one record, showing
        # unregistered loggers are opened and given handlers.
        logger = logging.getLogger("torch.utils")
        logger.info("hi")
        self.assertEqual(len(records), 1)

    # check logging to a random log that is not a child log of a registered
    # logger registers it and sets handlers properly
    @make_logging_test(modules={"torch.utils": logging.INFO})
    def test_open_registration_python_api(self, records):
        # [FUNCTION DOC] Same as the previous open-registration test but configured through the
        # module-level mapping of the set_logs API with the info level, asserting one gathered
        # record.
        logger = logging.getLogger("torch.utils")
        logger.info("hi")
        self.assertEqual(len(records), 1)

    @make_settings_test("torch._logging")
    def test_directory_based_logging(self, records):
        # [FUNCTION DOC] With the settings string naming the torch._logging package logger, logs
        # one message on the package and one on a submodule, asserts at least two records, and
        # that both loggers have handlers or propagate to a parent.
        # Test that the package itself can log
        logger = logging.getLogger("torch._logging")
        logger.info("package log")

        # Test that submodules can also log
        sublogger = logging.getLogger("torch._logging._internal")
        sublogger.info("submodule log")

        # We should have at least 2 records (one from package, one from submodule)
        self.assertGreaterEqual(len(records), 2)

        # Verify both loggers are registered and have handlers
        self.assertTrue(len(logger.handlers) > 0 or logger.propagate)
        self.assertTrue(len(sublogger.handlers) > 0 or sublogger.propagate)

    @make_logging_test(all=logging.DEBUG, dynamo=logging.INFO)
    def test_all(self, _):
        # [FUNCTION DOC] With all logging at debug and dynamo at info, checks every registered
        # logger name: names that start with any dynamo logger name must have effective level
        # info, all others debug. The gathered records argument is unused.
        registry = torch._logging._internal.log_registry

        dynamo_qnames = registry.log_alias_to_log_qnames["dynamo"]
        for logger_qname in torch._logging._internal.log_registry.get_log_qnames():
            logger = logging.getLogger(logger_qname)

            # if logger_qname is a.b.c and dynamo_qnames contains a.b, it still matches dynamo's INFO setting
            if any(logger_qname.find(d) == 0 for d in dynamo_qnames):
                self.assertEqual(
                    logger.getEffectiveLevel(),
                    logging.INFO,
                    msg=lambda msg: f"{msg}\nexpected {logger_qname} is INFO, got {logging.getLevelName(logger.getEffectiveLevel())}",
                )
            else:
                self.assertEqual(
                    logger.getEffectiveLevel(),
                    logging.DEBUG,
                    msg=lambda msg: f"{msg}\nexpected {logger_qname} is DEBUG, got {logging.getLevelName(logger.getEffectiveLevel())}",
                )

    @make_logging_test(graph_breaks=True)
    def test_graph_breaks(self, records):
        # [FUNCTION DOC] With graph break logging, compiles a function that forces a graph break
        # with inductor, runs it and asserts exactly one record.
        @torch.compile(backend="inductor")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function that calls the graph break marker and then
            # adds one.
            torch._dynamo.graph_break()
            return x + 1

        fn(torch.ones(1))

        self.assertEqual(len(records), 1)

    @make_logging_test(side_effects=True)
    def test_side_effects(self, records):
        # [FUNCTION DOC] With side-effect logging, compiles a function that appends to an input
        # list and asserts one record whose normalized message matches a golden text naming the
        # mutated list and the call stack of the append.
        my_list = [1, 2, 3]

        @torch.compile(backend="eager")
        def fn(x, lst):
            # [FUNCTION DOC] HELPER: Compiled function appending four to the list and returning x
            # plus the list length.
            lst.append(4)
            return x + len(lst)

        fn(torch.ones(1), my_list)

        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage()),
            """\
Mutating object of type list (source name: L['lst'])

      File "test_logging.py", line N, in test_side_effects
        fn(torch.ones(1), my_list)
      File "test_logging.py", line N, in fn
        lst.append(4)
""",
        )

    @make_logging_test(side_effects=True)
    def test_side_effects_nested_calls(self, records):
        # [FUNCTION DOC] With side-effect logging, mutates an input list once in a nested inner
        # call and once in the outer compiled function; asserts one record whose golden text
        # shows both mutation stacks separated by a divider.
        outer_list = [1]

        def inner(lst):
            # [FUNCTION DOC] HELPER: Appends two to the list and returns its length.
            lst.append(2)
            return len(lst)

        @torch.compile(backend="eager")
        def outer(x, my_list):
            # [FUNCTION DOC] HELPER: Compiled function that calls the inner helper, appends three
            # to the list and returns a sum.
            result = inner(my_list)
            my_list.append(3)
            return x + result + len(my_list)

        outer(torch.ones(1), outer_list)

        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage()),
            """\
Mutating object of type list (source name: L['my_list'])

      File "test_logging.py", line N, in test_side_effects_nested_calls
        outer(torch.ones(1), outer_list)
      File "test_logging.py", line N, in outer
        result = inner(my_list)
      File "test_logging.py", line N, in inner
        lst.append(2)

    ********

      File "test_logging.py", line N, in test_side_effects_nested_calls
        outer(torch.ones(1), outer_list)
      File "test_logging.py", line N, in outer
        my_list.append(3)
""",
        )

    @make_logging_test(side_effects=True)
    def test_side_effects_multiple_mutations_same_object(self, records):
        # [FUNCTION DOC] With side-effect logging, applies four mutations (two appends, an
        # extend, a pop) to one list and asserts a single record whose golden text lists four
        # stacks separated by dividers.
        my_list = [1, 2, 3]

        @torch.compile(backend="eager")
        def fn(x, lst):
            # [FUNCTION DOC] HELPER: Compiled function performing the four list mutations and
            # returning x plus the list length.
            lst.append(4)
            lst.append(5)
            lst.extend([6, 7])
            lst.pop()
            return x + len(lst)

        fn(torch.ones(1), my_list)

        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage()),
            """\
Mutating object of type list (source name: L['lst'])

      File "test_logging.py", line N, in test_side_effects_multiple_mutations_same_object
        fn(torch.ones(1), my_list)
      File "test_logging.py", line N, in fn
        lst.append(4)

    ********

      File "test_logging.py", line N, in test_side_effects_multiple_mutations_same_object
        fn(torch.ones(1), my_list)
      File "test_logging.py", line N, in fn
        lst.append(5)

    ********

      File "test_logging.py", line N, in test_side_effects_multiple_mutations_same_object
        fn(torch.ones(1), my_list)
      File "test_logging.py", line N, in fn
        lst.extend([6, 7])

    ********

      File "test_logging.py", line N, in test_side_effects_multiple_mutations_same_object
        fn(torch.ones(1), my_list)
      File "test_logging.py", line N, in fn
        lst.pop()
""",
        )

    @make_logging_test(side_effects=True)
    def test_side_effects_dict_mutations(self, records):
        # [FUNCTION DOC] With side-effect logging, assigns two new keys in an input dict inside a
        # compiled function and asserts one record with a golden text listing both assignment
        # stacks.
        my_dict = {"a": 1}

        @torch.compile(backend="eager")
        def fn(x, d):
            # [FUNCTION DOC] HELPER: Compiled function setting two keys in the dict and returning
            # x plus its length.
            d["b"] = 2
            d["c"] = 3
            return x + len(d)

        fn(torch.ones(1), my_dict)

        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage()),
            """\
Mutating object of type dict (source name: L['d'])

      File "test_logging.py", line N, in test_side_effects_dict_mutations
        fn(torch.ones(1), my_dict)
      File "test_logging.py", line N, in fn
        d["b"] = 2

    ********

      File "test_logging.py", line N, in test_side_effects_dict_mutations
        fn(torch.ones(1), my_dict)
      File "test_logging.py", line N, in fn
        d["c"] = 3
""",
        )

    @make_logging_test(side_effects=True)
    def test_side_effects_attribute_mutations(self, records):
        # [FUNCTION DOC] With side-effect logging, sets attributes on an input object three times
        # (one attribute twice) inside a compiled function and asserts one record with a golden
        # text containing three mutation stacks.
        class MyClass:
            # [CLASS DOC] HELPER: Plain object with integer attributes value and count.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets value to ten and count to zero.
                self.value = 10
                self.count = 0

        obj = MyClass()

        @torch.compile(backend="eager")
        def fn(x, o):
            # [FUNCTION DOC] HELPER: Compiled function assigning the object's attributes and
            # returning a sum.
            o.value = 20
            o.count = 1
            o.count = 2
            return x + o.value + o.count

        fn(torch.ones(1), obj)

        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage()),
            """\
Mutating object of type MyClass (source name: L['o'])

      File "test_logging.py", line N, in test_side_effects_attribute_mutations
        fn(torch.ones(1), obj)
      File "test_logging.py", line N, in fn
        o.value = 20

    ********

      File "test_logging.py", line N, in test_side_effects_attribute_mutations
        fn(torch.ones(1), obj)
      File "test_logging.py", line N, in fn
        o.count = 1

    ********

      File "test_logging.py", line N, in test_side_effects_attribute_mutations
        fn(torch.ones(1), obj)
      File "test_logging.py", line N, in fn
        o.count = 2
""",
        )

    @make_logging_test(side_effects=True)
    def test_side_effects_local_list_no_log(self, records):
        # [FUNCTION DOC] With side-effect logging, a list created and mutated inside the compiled
        # region must not be logged: asserts zero records.
        """Test that lists created inside compiled region don't log side effects."""

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function building a local list, appending to it and
            # returning x plus its length.
            my_list = [1, 2, 3]  # Created inside compiled region
            my_list.append(4)
            return x + len(my_list)

        fn(torch.ones(1))

        # Should NOT have logged the list mutation since it's a local variable
        self.assertEqual(len(records), 0)

    @make_logging_test(side_effects=True)
    def test_side_effects_local_object_with_log(self, records):
        # [FUNCTION DOC] With side-effect logging, an object created inside the compiled region
        # and returned is mutated; asserts one record whose golden text says the object was
        # created in the compile region and shows the constructor assignment stack and the later
        # assignment stack.
        """Test that returned objects created inside compiled region still log attribute mutations."""

        class MyClass:
            # [CLASS DOC] HELPER: Plain object whose initializer sets value to ten.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets value to ten.
                self.value = 10

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function creating the object, changing its value,
            # and returning a sum and the object.
            obj = MyClass()  # Created inside compiled region
            obj.value = 20
            return x + obj.value, obj

        fn(torch.ones(1))

        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage()),
            """\
Mutating object of type MyClass (source: created in torch.compile region)

      File "test_logging.py", line N, in test_side_effects_local_object_with_log
        fn(torch.ones(1))
      File "test_logging.py", line N, in fn
        obj = MyClass()  # Created inside compiled region
      File "test_logging.py", line N, in __init__
        self.value = 10

    ********

      File "test_logging.py", line N, in test_side_effects_local_object_with_log
        fn(torch.ones(1))
      File "test_logging.py", line N, in fn
        obj.value = 20
""",
        )

    @make_logging_test(side_effects=True)
    def test_side_effects_nn_module_buffer(self, records):
        # [FUNCTION DOC] With side-effect logging, compiles a module whose forward adds to a
        # registered buffer in place and asserts one record with a golden text naming the
        # module's buffer dictionary as the mutated object.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a random 2 by 2 buffer whose forward increments the
            # buffer in place and returns the input plus it.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the base module and registers the random
                # buffer named buf.
                super().__init__()
                self.register_buffer("buf", torch.rand(2, 2))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds one to the buffer in place and returns the input
                # plus the buffer.
                self.buf += 1
                return x + self.buf

        @torch.compile(backend="eager")
        def fn(mod, x):
            # [FUNCTION DOC] HELPER: Compiled function calling the module on x.
            return mod(x)

        fn(Mod(), torch.ones(1))

        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage()),
            """\
Mutating object of type dict (source name: L['mod']._buffers)

      File "test_logging.py", line N, in test_side_effects_nn_module_buffer
        fn(Mod(), torch.ones(1))
      File "test_logging.py", line N, in fn
        return mod(x)
      File "test_logging.py", line N, in forward
        self.buf += 1
""",
        )

    @make_logging_test(side_effects=True)
    @torch._dynamo.config.patch(side_effect_replay_policy="silent")
    def test_side_effects_silent_config(self, records):
        # [FUNCTION DOC] With side-effect logging and the replay policy config patched to silent,
        # compiles a function appending to an input list and asserts nothing is logged.
        my_list = [1, 2, 3]

        @torch.compile(backend="eager")
        def fn(x, lst):
            # [FUNCTION DOC] HELPER: Compiled function appending to the list and returning x plus
            # its length.
            lst.append(4)
            return x + len(lst)

        fn(torch.ones(1), my_list)

        self.assertEqual(len(records), 0)

    @make_logging_test(side_effects=True)
    def test_side_effects_logs_fullgraph_graph_break(self, records):
        # [FUNCTION DOC] With side-effect logging, a full-graph compile whose function mutates a
        # list and then forces a graph break must raise the Unsupported error; asserts the side
        # effect was still logged (at least one record, first mentioning mutation of a list).
        """Test that side effects are logged even when fullgraph=True causes an error."""
        my_list = [1, 2, 3]

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, lst):
            # [FUNCTION DOC] HELPER: Compiled full-graph function appending to the list, hitting
            # a graph break marker and returning a sum.
            lst.append(4)
            # Force a graph break after the side effect
            torch._dynamo.graph_break()
            return x + len(lst)

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            fn(torch.ones(1), my_list)

        # Side effects should still be logged even though codegen never ran
        self.assertGreater(len(records), 0)
        self.assertIn("Mutating object of type list", records[0].getMessage())

    @make_logging_test(side_effects=True)
    def test_side_effects_logged_on_fullgraph_side_effect_error(self, records):
        # [FUNCTION DOC] With side-effect logging, a full-graph compile appends to a list then
        # calls a checkpointed function that mutates another list, which makes tracing raise
        # Unsupported with an unsafe-side-effect higher-order-op message; asserts that message
        # and that the earlier list mutation was logged.
        tracked_list = []
        hop_list = []

        def fn(x):
            # [FUNCTION DOC] HELPER: Appends one to the captured list and returns the sine of the
            # input.
            hop_list.append(1)
            return x.sin()

        @torch.compile(backend="eager", fullgraph=True)
        def model(x, lst):
            # [FUNCTION DOC] HELPER: Compiled full-graph function that appends to the list and
            # calls a non-reentrant checkpoint of the helper.
            lst.append(1)
            return torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "HOP: Unsafe side effect",
        ):
            model(torch.ones(1), tracked_list)

        self.assertGreaterEqual(len(records), 1)
        self.assertIn("Mutating object of type list", records[0].getMessage())

    @make_settings_test("torch._dynamo.utils")
    def test_dump_compile_times(self, records):
        # [FUNCTION DOC] With logging for the dynamo utils logger, compiles and runs the example
        # function and calls the compile-times dump directly (not the exit hook); asserts exactly
        # one record mentioning TorchDynamo compilation metrics.
        fn_opt = torch.compile(example_fn, backend="inductor")
        fn_opt(torch.ones(1000, 1000))
        # This function runs during exit via atexit.register.
        # We're not actually going to run atexit._run_exit_funcs() here,
        # because it'll destroy state necessary for other tests.
        torch._dynamo.utils.dump_compile_times()
        self.assertEqual(
            len(
                [r for r in records if "TorchDynamo compilation metrics" in str(r.msg)]
            ),
            1,
        )

    @make_logging_test(dynamo=logging.INFO)
    def test_custom_format_exc(self, records):
        # [FUNCTION DOC] With dynamo logging at info, logs an exception, an info record with
        # exception info, and an info record with stack info; asserts three records, then formats
        # them with the logger's torch handler and checks the first two contain a traceback and
        # the third a stack section.
        # KNOWN ISSUE: the handler loop variable remains set after the loop even if no torch
        # handler was found, so the not-None assertion cannot detect that.
        dynamo_log = logging.getLogger(torch._dynamo.__name__)
        try:
            raise RuntimeError("foo")
        except RuntimeError:
            dynamo_log.exception("test dynamo")
            dynamo_log.info("with exc", exc_info=True)
        dynamo_log.info("with stack", stack_info=True)
        self.assertEqual(len(records), 3)
        # unfortunately there's no easy way to test the final formatted log other than
        # to ask the dynamo logger's handler to format it.
        for handler in dynamo_log.handlers:
            if torch._logging._internal._is_torch_handler(handler):
                break
        self.assertIsNotNone(handler)
        self.assertIn("Traceback", handler.format(records[0]))
        self.assertIn("Traceback", handler.format(records[1]))
        self.assertIn("Stack", handler.format(records[2]))

    @make_logging_test(dynamo=logging.INFO)
    def test_custom_format(self, records):
        # [FUNCTION DOC] With dynamo logging at info, logs one message on the dynamo logger and
        # one on a custom artifact logger, asserts two records, formats them with the torch
        # handler and checks that the line normalizer reduces both (and a sample CI-style
        # prefixed line) to the plain message.
        dynamo_log = logging.getLogger(torch._dynamo.__name__)
        test_log = torch._logging.getArtifactLogger(
            torch._dynamo.__name__, "custom_format_test_artifact"
        )
        dynamo_log.info("test dynamo")
        test_log.info("custom format")
        self.assertEqual(len(records), 2)
        # unfortunately there's no easy way to test the final formatted log other than
        # to ask the dynamo logger's handler to format it.
        for handler in dynamo_log.handlers:
            if torch._logging._internal._is_torch_handler(handler):
                break
        self.assertIsNotNone(handler)
        formatted_dynamo = handler.format(records[0])
        self.assertIn("test dynamo", formatted_dynamo)
        self.assertEqual(normalize_log_line(formatted_dynamo), "test dynamo")
        ci_style_line = (
            "I1124 19:43:23.879000 4928 dynamo/test_logging.py:410] test dynamo"
        )
        self.assertEqual(normalize_log_line(ci_style_line), "test dynamo")

        formatted_artifact = handler.format(records[1])
        self.assertIn("custom format", formatted_artifact)
        self.assertEqual(normalize_log_line(formatted_artifact), "custom format")

    @make_logging_test(dynamo=logging.INFO)
    def test_multiline_format(self, records):
        # [FUNCTION DOC] With dynamo logging at info, logs three messages containing newlines
        # (plain, formatted argument, and mixed), asserts three records, formats each and checks
        # that after normalizing each line and dropping empty lines the lines equal the expected
        # word lists.
        dynamo_log = logging.getLogger(torch._dynamo.__name__)
        dynamo_log.info("test\ndynamo")
        dynamo_log.info("%s", "test\ndynamo")
        dynamo_log.info("test\n%s", "test\ndynamo")
        self.assertEqual(len(records), 3)
        # unfortunately there's no easy way to test the final formatted log other than
        # to ask the dynamo logger's handler to format it.
        for handler in dynamo_log.handlers:
            if torch._logging._internal._is_torch_handler(handler):
                break
        self.assertIsNotNone(handler)
        expected_lines = [
            ["test", "dynamo"],
            ["test", "dynamo"],
            ["test", "test", "dynamo"],
        ]

        for record, expected in zip(records, expected_lines):
            formatted = handler.format(record)
            normalized_lines = [
                line
                for line in (normalize_log_line(l) for l in formatted.splitlines())
                if line
            ]
            self.assertEqual(normalized_lines, expected)

    test_trace_source_simple = within_range_record_test(1, 100, trace_source=True)

    @make_logging_test(trace_source=True)
    def test_trace_source_if_stmt(self, records):
        # [FUNCTION DOC] With trace-source logging, compiles a function with a data-dependent
        # branch; checks that a source line of the taken branch appears in the records and that
        # of the other branch does not.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the input doubled if its sum is positive, otherwise
            # tripled.
            if x.sum() > 0:
                return x * 2
            return x * 3

        fn_opt = torch.compile(fn, backend="eager")
        fn_opt(torch.ones(3, 3))

        found_x2 = False
        found_x3 = False
        for record in records:
            msg = record.getMessage()
            if "return x * 2" in msg:
                found_x2 = True
            if "return x * 3" in msg:
                found_x3 = True

        self.assertTrue(found_x2)
        self.assertFalse(found_x3)

    @make_logging_test(trace_source=True)
    def test_trace_source_nested(self, records):
        # [FUNCTION DOC] With trace-source logging, compiles a function chain of three calls;
        # checks the source lines appear and that inline depth is absent for the outer function
        # and one and two for the callees.
        def fn1(x):
            # [FUNCTION DOC] HELPER: Calls the second function and doubles the result.
            x = fn2(x)
            return x * 2

        def fn2(x):
            # [FUNCTION DOC] HELPER: Calls the third function and triples the result.
            x = fn3(x)
            return x * 3

        def fn3(x):
            # [FUNCTION DOC] HELPER: Returns the input times four.
            return x * 4

        fn_opt = torch.compile(fn1, backend="eager")
        fn_opt(torch.ones(3, 3))

        found_x2 = False
        found_x3 = False
        found_x4 = False
        for record in records:
            msg = record.getMessage()
            if "return x * 2" in msg:
                found_x2 = True
                self.assertNotIn("inline depth", msg)
            elif "return x * 3" in msg:
                found_x3 = True
                self.assertIn("inline depth: 1", msg)
            elif "return x * 4" in msg:
                found_x4 = True
                self.assertIn("inline depth: 2", msg)
        self.assertTrue(found_x2)
        self.assertTrue(found_x3)
        self.assertTrue(found_x4)

    @make_logging_test(trace_source=True)
    def test_trace_source_cond(self, records):
        # [FUNCTION DOC] With trace-source logging, compiles a function calling a conditional
        # operator inside nested calls; checks that source lines of both branches appear and that
        # each is logged at inline depth three.
        from functorch.experimental.control_flow import cond

        def true_fn(x):
            # [FUNCTION DOC] HELPER: True branch: input doubled.
            return x * 2

        def false_fn(x):
            # [FUNCTION DOC] HELPER: False branch: input tripled.
            return x * 3

        def inner(pred, x):
            # [FUNCTION DOC] HELPER: Applies the conditional operator to the predicate and input
            # with the two branches.
            return cond(pred, true_fn, false_fn, [x])

        def outer(pred, x):
            # [FUNCTION DOC] HELPER: Calls the inner function.
            return inner(pred, x)

        fn_opt = torch.compile(outer, backend="eager")
        fn_opt(torch.tensor(True), torch.ones(3, 3))

        found_x2 = False
        found_x3 = False
        for record in records:
            msg = record.getMessage()
            if "return x * 2" in msg:
                found_x2 = True
                self.assertIn("inline depth: 3", msg)
            if "return x * 3" in msg:
                found_x3 = True
                self.assertIn("inline depth: 3", msg)

        self.assertTrue(found_x2)
        self.assertTrue(found_x3)

    @make_logging_test(trace_source=True)
    def test_trace_source_funcname(self, records):
        # [FUNCTION DOC] With trace-source logging, compiles a function containing a nested
        # function that builds a tuple from a generator expression; checks that some record
        # mentions the generator name together with the qualified name of the nested function.
        # NOTE: list comprehensions are inlined in 3.12, so test with tuples
        def fn1():
            # [FUNCTION DOC] HELPER: Outer function: returns the result of calling the nested
            # function.
            def fn2():
                # [FUNCTION DOC] HELPER: Returns a tuple of five ones tensors built from a
                # generator expression.
                if True:
                    return tuple(torch.ones(3, 3) for _ in range(5))
                return None

            return fn2()

        fn_opt = torch.compile(fn1, backend="eager")
        fn_opt()

        found_funcname = False
        for record in records:
            msg = record.getMessage()
            if "<genexpr>" in msg and "fn1.fn2" in msg:
                found_funcname = True

        self.assertTrue(found_funcname)

    def test_flex_gemm_log_levels(self):
        # [FUNCTION DOC] Parses the settings text for the flex gemm artifact with and without a
        # plus prefix; asserts its logger gets level info for the plain form and debug for the
        # plus form and that neither adds artifact names.
        from torch._logging._internal import _parse_log_settings

        log_name = "torch._inductor.kernel.flex_gemm.debug"
        concise = _parse_log_settings("flex_gemm")
        verbose = _parse_log_settings("+flex_gemm")

        self.assertEqual(concise.log_qname_to_level[log_name], logging.INFO)
        self.assertEqual(verbose.log_qname_to_level[log_name], logging.DEBUG)
        self.assertEqual(concise.artifact_names, set())
        self.assertEqual(verbose.artifact_names, set())

    def test_invalid_artifact_flag(self):
        # [FUNCTION DOC] Asserts that calling the set_logs API with an invalid integer value for
        # an artifact flag raises ValueError.
        with self.assertRaises(ValueError):
            torch._logging.set_logs(aot_graphs=5)

    def test_invalid_artifact_flag_error_msg(self):
        # [FUNCTION DOC] Starts a Python subprocess importing torch with an unknown name in the
        # TORCH_LOGS environment variable; asserts stderr is short (under fifty lines), the fifth
        # line from the end is the help hint, and the fourth from the end is the valid-settings
        # header.
        env = dict(os.environ)
        env["TORCH_LOGS"] = "not_an_existing_log_artifact_should_error"
        _, stderr = self.run_process_no_exception(
            "import torch",
            env=env,
        )
        lines = stderr.decode().split("\r\n" if IS_WINDOWS else "\n")
        # This is a sanity assert that our error is not spammy.
        # As of this test creation this was 18.
        # See this issue for the purpose o this test:
        # https://github.com/pytorch/pytorch/issues/151055
        self.assertTrue(len(lines) < 50)
        # The other sanity assert - check that the last few lines
        # map to the actual error message we want to raise
        # (I could use an expecttest here, although it would break
        #  whenever someone adds a new logging artifact)
        self.assertEqual(
            lines[-5], 'For more info on various settings, try TORCH_LOGS="help"'
        )
        self.assertEqual(lines[-4], "Valid settings:")

    @requires_distributed()
    @skipIfWindows(msg="TODO: (xuhancn), Can't reproduce locally")
    def test_distributed_rank_logging(self):
        # [FUNCTION DOC] Requires distributed; skipped on Windows. Runs a subprocess with dynamo
        # logging, a fake process group (rank zero of two) and one dynamo info message; asserts
        # the normalized stderr contains the rank tag and the message.
        env = dict(os.environ)
        env["TORCH_LOGS"] = "dynamo"
        _, stderr = self.run_process_no_exception(
            """\
import torch.distributed as dist
import logging
from torch.testing._internal.distributed.fake_pg import FakeStore
store = FakeStore()
dist.init_process_group("fake", rank=0, world_size=2, store=store)
dynamo_log = logging.getLogger("torch._dynamo")
dynamo_log.info("woof")
print("arf")
""",
            env=env,
        )
        stderr_text = stderr.decode("utf-8")
        normalized = normalize_rank_prefix(stderr_text)
        self.assertIn("[rank0]:", normalized)
        self.assertIn("woof", normalized)

    @skipIfNotPy311
    @make_logging_test(trace_call=True)
    def test_trace_call(self, records):
        # [FUNCTION DOC] Skipped before Python 3.11. With trace-call logging, compiles a function
        # with two multiplications and a matrix multiplication, asserts three records, and
        # compares the last two lines of each message (source line and caret markers) to golden
        # strings.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the matrix product of the input x doubled and y
            # tripled.
            return (x * 2) @ (y * 3)

        fn_opt = torch.compile(fn, backend="eager")
        fn_opt(torch.randn(10, 20), torch.randn(20, 30))

        self.assertEqual(len(records), 3)
        # only get last 2 lines
        messages = [
            "\n".join(record.getMessage().split("\n")[-2:]) for record in records
        ]
        self.assertExpectedInline(
            messages[0],
            """\
            return (x * 2) @ (y * 3)
                    ~~^~~""",
        )
        self.assertExpectedInline(
            messages[1],
            """\
            return (x * 2) @ (y * 3)
                              ~~^~~""",
        )
        self.assertExpectedInline(
            messages[2],
            """\
            return (x * 2) @ (y * 3)
                   ~~~~~~~~^~~~~~~~~""",
        )

    @skipIfNotPy311
    @make_logging_test(trace_call=True)
    def test_trace_call_prefix(self, records):
        # [FUNCTION DOC] Skipped before Python 3.11. With trace-call logging, normalizes the
        # first record keeping carets and compares to a golden text with the trace prefix naming
        # the multiplication, file, function and qualified test name.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the matrix product of the input x doubled and y
            # tripled.
            return (x * 2) @ (y * 3)

        fn_opt = torch.compile(fn, backend="eager")
        fn_opt(torch.randn(10, 20), torch.randn(20, 30))

        msg0 = munge_exc(records[0].getMessage(), strip_carets=False)
        self.assertExpectedInline(
            msg0,
            """\
TRACE FX call mul from test_logging.py:N in fn (LoggingTests.test_trace_call_prefix.fn)
            return (x * 2) @ (y * 3)
                    ~~^~~""",
        )

    @skipIfNotPy311
    @make_logging_test(trace_call=True)
    def test_trace_call_inline_call(self, records):
        # [FUNCTION DOC] Skipped before Python 3.11. With trace-call logging, compiles a function
        # calling a helper twice; asserts three records and checks the first two last-two-line
        # excerpts; the third is not checked because newer Python versions drop the carets.
        def g(x):
            # [FUNCTION DOC] HELPER: Returns the input doubled.
            return x * 2

        def f(x):
            # [FUNCTION DOC] HELPER: Applies the doubling helper twice.
            return g(g(x))

        fn_opt = torch.compile(f, backend="eager")
        fn_opt(torch.randn(3, 3))

        self.assertEqual(len(records), 3)
        messages = [
            "\n".join(record.getMessage().split("\n")[-2:]) for record in records
        ]
        self.assertExpectedInline(
            messages[0],
            """\
            return g(g(x))
                     ~^^^""",
        )
        self.assertExpectedInline(
            messages[1],
            """\
            return x * 2
                   ~~^~~""",
        )
        # skip this check since 3.13 removed carets for this case
        # see https://github.com/python/cpython/issues/99180
        # self.assertExpectedInline(
        #     messages[2],
        #     """\
        #     return g(g(x))
        #            ~^^^^^^""",
        # )

    @skipIfNotPy311
    @make_logging_test(trace_call=True)
    def test_trace_call_graph_break(self, records):
        # [FUNCTION DOC] Skipped before Python 3.11. With trace-call logging, compiles a function
        # with a graph break between two multiplications; asserts three records and checks the
        # first and last excerpts.
        def fn(x):
            # [FUNCTION DOC] HELPER: Doubles x, forces a graph break, then triples the result.
            x = x * 2
            torch._dynamo.graph_break()
            return x * 3

        fn_opt = torch.compile(fn, backend="eager")
        fn_opt(torch.randn(3, 3))

        self.assertEqual(len(records), 3)
        messages = [
            "\n".join(record.getMessage().split("\n")[-2:]) for record in records
        ]
        self.assertExpectedInline(
            messages[0],
            """\
            x = x * 2
                ~~^~~""",
        )
        self.assertExpectedInline(
            messages[-1],
            """\
            return x * 3
                   ~~^~~""",
        )

    @make_logging_test(guards=True, recompiles=True)
    def test_guards_recompiles(self, records):
        # [FUNCTION DOC] With guards and recompile logging, compiles a function over lists with
        # the eager backend, calls it with lists of different length, and asserts the joined
        # messages contain a guard on the first element of the second list and a length guard on
        # the first list.
        def fn(x, ys, zs):
            # [FUNCTION DOC] HELPER: Calls the inner helper.
            return inner(x, ys, zs)

        def inner(x, ys, zs):
            # [FUNCTION DOC] HELPER: Adds the elementwise product of paired list items to the
            # input in place.
            for y, z in zip(ys, zs):
                x += y * z
            return x

        ys = [1.0, 2.0]
        zs = [3.0]
        x = torch.tensor([1.0])

        fn_opt = torch.compile(fn, backend="eager")
        fn_opt(x, ys, zs)
        fn_opt(x, ys[:1], zs)

        record_str = "\n".join(r.getMessage() for r in records)

        self.assertIn(
            """L['zs'][0] == 3.0""",
            record_str,
        )
        self.assertIn(
            "len(L['ys']) == 2",
            record_str,
        )

    @make_logging_test(guards=True)
    def test_guards_sloc(self, records):
        # [FUNCTION DOC] With guards logging, compiles a dynamic-shape function with a size-
        # dependent branch and a concatenation, runs it, finds the guard manager record, and
        # compares the normalized shape guard lines to a golden string.
        @torch.compile(dynamic=True, backend="eager")
        def f(x, y, z):
            # [FUNCTION DOC] HELPER: Compiled function: triples x and, if its length is divisible
            # by three, adds the concatenation of y and z, else doubles.
            x = x * 3
            if x.size(0) % 3 == 0:
                return x + torch.cat([y, z])
            else:
                return x * 2

        f(torch.randn(6), torch.randn(3), torch.randn(3))

        record = self.getRecord(records, "TREE_GUARD_MANAGER")
        self.assertExpectedInline(
            munge_shape_guards(record.getMessage()),
            """\
+- __SHAPE_GUARD__: L['x'].size()[0] == 2*L['y'].size()[0]  # return x + torch.cat([y, z])  # #:# in # #:# in #
+- __SHAPE_GUARD__: L['z'].size()[0] == L['y'].size()[0]  # duck sizing added this equality because these variables had the same size 3 (to avoid this specialization, set torch.fx.experimental._config.use_duck_shape = False)
+- __SHAPE_GUARD__: ((2*L['y'].size()[0]) % 3) == 0  # if x.size(0) % 3 == 0:  # #:# in # #:# in #
+- __SHAPE_GUARD__: 2 <= L['y'].size()[0]  # return x + torch.cat([y, z])  # #:# in # (user code shown is first use of this value--the guard itself is not due user code but due to 0/1 specialization in the framework; to avoid specialization try torch._dynamo.decorators.mark_unbacked(tensor, dim))""",
        )

    @make_logging_test(guards=True)
    def test_guards_polyfill_sloc(self, records):
        # [FUNCTION DOC] With guards logging, compiles a dynamic-shape function using an any call
        # over a size comparison and compares the normalized shape guards to a golden string.
        @torch.compile(dynamic=True, backend="eager")
        def f(x, y):
            # [FUNCTION DOC] HELPER: Compiled function returning whether the size of x equals
            # twice the size of y, via any.
            return any([x.size(0) == y.size(0) * 2])

        f(torch.randn(6), torch.randn(3))

        record = self.getRecord(records, "TREE_GUARD_MANAGER")
        self.assertExpectedInline(
            munge_shape_guards(record.getMessage()),
            """\
+- __SHAPE_GUARD__: L['x'].size()[0] == 2*L['y'].size()[0]  # return any([x.size(0) == y.size(0) * 2])  # #:# in # #:# in #
+- __SHAPE_GUARD__: 2 <= L['y'].size()[0]  # return any([x.size(0) == y.size(0) * 2])  # #:# in # (user code shown is first use of this value--the guard itself is not due user code but due to 0/1 specialization in the framework; to avoid specialization try torch._dynamo.decorators.mark_unbacked(tensor, dim))""",
        )

    @make_logging_test(guards=True)
    def test_guards_sloc_vr(self, records):
        # [FUNCTION DOC] With guards logging, compiles a dynamic-shape function with range and
        # equality checks on sizes, and compares the normalized shape guards (a combined range
        # guard and an equality guard) to a golden string.
        @torch.compile(dynamic=True, backend="eager")
        def f(x, y):
            # [FUNCTION DOC] HELPER: Compiled function asserting size bounds and an equality
            # relation through the check API, then returning a true tensor.
            torch._check(x.size(0) > 5)
            torch._check(x.size(0) < 30)
            torch._check(x.size(0) == y.size(0) * 2)
            return torch.tensor(True)

        f(torch.randn(6), torch.randn(3))

        record = self.getRecord(records, "TREE_GUARD_MANAGER")
        self.assertExpectedInline(
            munge_shape_guards(record.getMessage()),
            """\
+- __SHAPE_GUARD__: L['x'].size()[0] == 2*L['y'].size()[0]  # torch._check(x.size(0) == y.size(0) * 2)  # #:# in # #:# in #
+- __SHAPE_GUARD__: 3 <= L['y'].size()[0] <= 14  # torch._check(x.size(0) > 5)  # #:# in # #:# in # and torch._check(x.size(0) < 30)  # #:# in # #:# in #""",
        )

    @make_logging_test(guards=True)
    def test_global_state_guard_logging(self, records):
        # [FUNCTION DOC] With guards logging, compiles a trivial function, finds the guard
        # manager record and compares the normalized global-state guard line to a golden string.
        @torch.compile(backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function adding one.
            return x + 1

        f(torch.randn(3))

        record = self.getRecord(records, "TREE_GUARD_MANAGER")
        self.assertExpectedInline(
            munge_global_state_json(record.getMessage()),
            """+- GLOBAL_STATE: ___check_global_state() against {"allow_bf16_reduce": "#","allow_fp16_reduce": "#","allow_tf32": "#","autocast_state":{"cached_enabled": "#","dtype": "#","enabled": "#"},"default_dtype": "#","deterministic_algorithms": "#","deterministic_algorithms_warn_only": "#","grad_mode": "#","num_threads": "#","torch_function": "#","torch_function_all_disabled": "#"}""",
        )

    @make_logging_test(cudagraph_static_inputs=True)
    def test_cudagraph_static_inputs(self, records):
        # [FUNCTION DOC] With cudagraph static input logging, compiles in reduce-overhead mode a
        # function on an input marked as a static address, and asserts the record count is
        # positive and below four.
        @torch.compile(mode="reduce-overhead")  # noqa: UNSPECIFIED_BACKEND
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function adding one.
            return x + 1

        x = torch.ones(2, 2)
        torch._dynamo.mark_static_address(x)
        fn(x)
        self.assertGreater(len(records), 0)
        self.assertLess(len(records), 4)

    @make_logging_test(perf_hints=True)
    @requires_gpu
    def test_optimizer_non_static_param(self, records):
        # [FUNCTION DOC] With perf hint logging and GPU required, builds two parameters with
        # gradients and an Adam optimizer, compiles its step in reduce-overhead mode, runs it,
        # and asserts one or two records.
        params = [torch.randn(10, 10, device=device_type) for _ in range(2)]
        for param in params:
            param.grad = torch.zeros_like(param)
        opt = torch.optim.Adam(params)
        compiled_opt_step = torch.compile(opt.step, mode="reduce-overhead")  # noqa: UNSPECIFIED_BACKEND
        compiled_opt_step()
        self.assertGreater(len(records), 0)
        self.assertLess(len(records), 3)

    @make_logging_test(autotuning=True)
    @requires_gpu
    @unittest.skipIf(not SM90OrLater, "requires H100+ GPU")
    def test_autotuning(self, records):
        # [FUNCTION DOC] With autotuning logging, requires GPU and a Hopper-class or newer
        # device. Compiles a matrix multiply with max-autotune without cudagraphs inside a fresh
        # cache and asserts the record count is positive and below forty.
        with torch._inductor.utils.fresh_cache():

            def f(a, b):
                # [FUNCTION DOC] HELPER: Returns the matrix product of a and b.
                return torch.mm(a, b)

            f = torch.compile(f, mode="max-autotune-no-cudagraphs")  # noqa: UNSPECIFIED_BACKEND
            f(
                torch.randn(10, 10, device=device_type),
                torch.randn(10, 10, device=device_type),
            )
            self.assertGreater(len(records), 0)
            self.assertLess(len(records), 40)

    @make_logging_test(graph_region_expansion=True)
    def test_graph_region_expansion(self, records):
        # [FUNCTION DOC] With graph region expansion logging, patches the node-tracking config
        # on, extracts the graph and region tracker from a function with repeated sub-
        # computations, asks the tracker for identical regions and asserts at least one record.
        with torch._dynamo.config.patch("track_nodes_for_deduplication", True):

            def inner_fn(x, y):
                # [FUNCTION DOC] HELPER: Returns the sum of the sums of x plus one and y plus
                # two.
                x0 = x + 1
                y0 = y + 2
                z = x0.sum() + y0.sum()
                return z

            def fn(x, y):
                # [FUNCTION DOC] HELPER: Calls the shared computation three times, with a sine in
                # between, and returns a product of results.
                o0 = inner_fn(x, y)
                o1 = torch.sin(o0)
                o2 = inner_fn(x, o1)
                o3 = inner_fn(x, y)
                return o2 * o3 * o3

            graph, tracker = extract_graph_and_tracker(
                fn, torch.randn(10, 10), torch.randn(10, 10)
            )
            tracker.get_identical_regions(graph)
            self.assertGreater(len(records), 0)

    @skipIfTorchDynamo("too slow")
    @make_logging_test(**torch._logging.DEFAULT_LOGGING)
    def test_default_logging(self, records):
        # [FUNCTION DOC] Skipped under Dynamo. With the library default logging settings,
        # compiles a branching function with prints, runs it for two input shapes, and asserts
        # records from the graph break, recompile, symbolic shape and guard loggers and a record
        # containing a source line.
        def fn(a):
            # [FUNCTION DOC] HELPER: Applies sine or cosine depending on the sign of the sum,
            # prints a string, and returns the result plus one.
            if a.sum() < 0:
                a = torch.sin(a)
            else:
                a = torch.cos(a)
            print("hello")
            return a + 1

        fn_opt = torch.compile(fn, backend="eager")
        fn_opt(torch.ones(10, 10))
        fn_opt(-torch.ones(10, 5))

        self.assertGreater(len([r for r in records if ".__graph_breaks" in r.name]), 0)
        self.assertGreater(len([r for r in records if ".__recompiles" in r.name]), 0)
        self.assertGreater(len([r for r in records if ".symbolic_shapes" in r.name]), 0)
        self.assertGreater(len([r for r in records if ".__guards" in r.name]), 0)
        self.assertGreater(
            len([r for r in records if "return a + 1" in r.getMessage()]), 0
        )

    def test_logs_out(self):
        # [FUNCTION DOC] Runs a subprocess that compiles a small function with TORCH_LOGS for
        # dynamo and TORCH_LOGS_OUT pointing to a closed temporary file; asserts that the file
        # content equals the subprocess stderr after normalizing empty lines.
        # KNOWN ISSUE: the class-wide maxDiff is set to unlimited and is restored only if the
        # comparison raises, not after success.
        import tempfile

        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            file_path = _as_posix_path(tmp.name)
            """
            NamedTemporaryFile will include a file open operation.
            On Windowsm the file is opened by NamedTemporaryFile, the
            following run_process_no_exception can't access a opened file.
            And then, raise a PermissionError: [Errno 13] Permission denied: [file_path]
            """
            tmp.close()
            env = dict(os.environ)
            env["TORCH_LOGS"] = "dynamo"
            env["TORCH_LOGS_OUT"] = file_path
            _, stderr = self.run_process_no_exception(
                """\
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils._config_module")
import torch
@torch.compile(backend="eager")
def fn(a):
    return a.sum()

fn(torch.randn(5))
                """,
                env=env,
            )
            with open(
                file_path, encoding="utf-8"
            ) as fd:  # encoding file to UTF-8 for Windows.
                lines = fd.read()
                orig_maxDiff = unittest.TestCase.maxDiff
                unittest.TestCase.maxDiff = None
                try:
                    self.assertEqual(  # process wrap difference: /r/n on Windows, /n on posix.
                        empty_line_normalizer(lines),
                        empty_line_normalizer(stderr.decode("utf-8")),
                    )
                except Exception:
                    unittest.TestCase.maxDiff = orig_maxDiff
                    raise

    @make_settings_test("torch._dynamo.eval_frame")
    def test_log_traced_frames(self, records):
        # [FUNCTION DOC] Clears the Dynamo thread-local state, compiles and runs a function with
        # a nested function containing a graph break and another compiled function, explicitly
        # invokes the traced-frames logger, and compares the normalized record to a golden list
        # of frame names.
        torch._dynamo.eval_frame.clear_dynamo_tls()

        # Test program
        @torch.compile(backend="eager")
        def foo():
            # [FUNCTION DOC] HELPER: Compiled function that creates a ones tensor and returns the
            # result of calling a nested function and that function.
            x = torch.ones([10])

            def bar():
                # [FUNCTION DOC] HELPER: Adds the tensor to itself, forces a graph break,
                # multiplies and returns the result.
                y = x + x
                torch._dynamo.graph_break()
                z = y * x
                return z

            # force top-level trace of bar
            try:
                return bar(), bar
            finally:
                pass

        foo()

        @torch.compile  # noqa: UNSPECIFIED_BACKEND
        def baz(x):
            # [FUNCTION DOC] HELPER: Compiled function adding one.
            return x + 1

        baz(torch.ones(3))

        # `_log_traced_frames` is registered as an atexit callback, so we invoke
        # it explicitly for testing.
        torch._dynamo.eval_frame._log_traced_frames()

        # Get the relevant log.
        record = self.getRecord(records, "TorchDynamo attempted to trace")

        # Check
        self.assertExpectedInline(
            munge_exc(record.getMessage()),
            """\
TorchDynamo attempted to trace the following frames: [
  * foo test_logging.py:N
  * bar test_logging.py:N
  * baz test_logging.py:N
]""",
        )

    @make_logging_test(dynamo=logging.DEBUG)
    def test_skip_reason_logging(self, records):
        # [FUNCTION DOC] With dynamo debug logging, compiles a function with Dynamo disabled by
        # config; finds the first skipping record and asserts it mentions that tracing is
        # disabled and the function name.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the input plus one.
            return x + 1

        x = torch.randn(5)
        with torch._dynamo.config.patch(disable=True):
            torch.compile(fn, backend="eager")(x)

        skip_records = [r for r in records if "skipping:" in r.getMessage()]
        self.assertGreater(len(skip_records), 0)
        msg = skip_records[0].getMessage()
        self.assertIn("Dynamo tracing is disabled", msg)
        self.assertIn("fn", msg)

    @make_logging_test(dynamo=logging.DEBUG)
    def test_skip_reason_logging_dispatch_mode(self, records):
        # [FUNCTION DOC] With dynamo debug logging, compiles a function under a pass-through
        # torch dispatch mode; asserts the first skipping record mentions a non-infra dispatch
        # mode and the function name.
        from torch.utils._python_dispatch import TorchDispatchMode

        class SkipMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Torch dispatch mode that forwards each call unchanged.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Calls the intercepted function with the given arguments
                # and returns its result.
                return func(*args, **kwargs)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the input plus one.
            return x + 1

        x = torch.randn(5)
        with SkipMode():
            torch.compile(fn, backend="eager")(x)

        skip_records = [r for r in records if "skipping:" in r.getMessage()]
        self.assertGreater(len(skip_records), 0)
        msg = skip_records[0].getMessage()
        self.assertIn("non-infra torch dispatch mode present", msg)
        self.assertIn("fn", msg)

    @requires_gpu
    @torch._inductor.config.patch("force_disable_caches", True)
    @make_logging_test(autotuning_inputs=True)
    def test_autotuning_inputs(self, records):
        # [FUNCTION DOC] Requires GPU; caches disabled. With autotuning input logging, compiles a
        # function in max-autotune mode and asserts the autotuning-inputs logger produced records
        # mentioning kernel inputs with shape, dtype and stride text.
        @torch.compile(mode="max-autotune")  # noqa: UNSPECIFIED_BACKEND
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function returning the row sums of the input
            # doubled plus one.
            return (x * 2.0 + 1.0).sum(dim=1)

        f(torch.randn(2048, 4096, device=device_type))

        autotune_records = [r for r in records if ".__autotuning_inputs" in r.name]
        self.assertGreater(len(autotune_records), 0)
        msg = "\n".join(r.getMessage() for r in autotune_records)
        self.assertIn("Autotuning inputs for kernel", msg)
        self.assertIn("shape=(", msg)
        self.assertIn("dtype=torch.float32", msg)
        self.assertIn("stride=(", msg)

    @requires_gpu
    @torch._inductor.config.patch("force_disable_caches", True)
    @make_logging_test(inductor=logging.DEBUG)
    def test_autotuning_inputs_off_by_default(self, records):
        # [FUNCTION DOC] Requires GPU; caches disabled. With inductor debug logging only,
        # compiles the same max-autotune function and asserts the autotuning-inputs logger
        # produces no records (artifact off by default).
        # off_by_default: must stay silent even with the parent inductor log at DEBUG
        @torch.compile(mode="max-autotune")  # noqa: UNSPECIFIED_BACKEND
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function returning the row sums of the input
            # doubled plus one.
            return (x * 2.0 + 1.0).sum(dim=1)

        f(torch.randn(2048, 4096, device=device_type))
        self.assertEqual(
            len([r for r in records if ".__autotuning_inputs" in r.name]), 0
        )


class PartitionedScatterLoggingTests(LoggingTestCase):
    # [CLASS DOC] Test case dedicated to the partitioned scatter logging artifact.
    """
    Dedicated tests for the partitioned_scatter TORCH_LOGS artifact.
    """

    @make_logging_test(partitioned_scatter=True)
    def test_partitioned_scatter(self, records):
        # [FUNCTION DOC] With partitioned scatter logging, compiles an accumulating index_put on
        # a zero vector with inductor (full graph) while the partitioned-scatter config flags are
        # forced on, and asserts at least one record was emitted.
        from torch._inductor import config as inductor_config

        N, n = 8192, 8

        def f(out, idx, vals):
            # [FUNCTION DOC] HELPER: Accumulates values into the output at the given indices and
            # returns the result.
            return out.index_put([idx], vals, accumulate=True)

        with inductor_config.patch(
            partitioned_scatter_enabled=True,
            partitioned_scatter_force=True,
        ):
            fn_opt = torch.compile(f, backend="inductor", fullgraph=True)
            out = torch.zeros(n)
            idx = torch.randint(0, 4, (N,), dtype=torch.int64)
            vals = torch.randn(N)
            fn_opt(out, idx, vals)

        # At least one debug record should be emitted (APPLY or SKIP decision).
        self.assertGreater(len(records), 0)


# non single record tests
exclusions = {
    "bytecode",
    "cudagraphs",
    "output_code",
    "schedule",
    "fusion",
    "overlap",
    "aot_graphs",
    "aot_graphs_effects",
    "pre_grad_graphs",
    "joint_graph_passes",
    "post_grad_graphs",
    "inductor_metrics",
    "ir_pre_fusion",
    "ir_post_fusion",
    "compiled_autograd",
    "compiled_autograd_verbose",
    "recompiles",
    "recompiles_verbose",
    "graph_breaks",
    "side_effects",
    "graph",
    "graph_code",
    "graph_code_verbose",
    "graph_sizes",
    "ddp_graphs",
    "perf_hints",
    "not_implemented",
    "trace_source",
    "trace_call",
    "trace_bytecode",
    "custom_format_test_artifact",
    "onnx",
    "onnx_diagnostics",
    "guards",
    "verbose_guards",
    "sym_node",
    "export",
    "trace_shape_events",
    "cudagraph_static_inputs",
    "benchmarking",
    "loop_ordering",
    "loop_tiling",
    "auto_chunker",
    "autotuning",
    "autotuning_inputs",
    "incremental",
    "graph_region_expansion",
    "hierarchical_compile",
    "compute_dependencies",
    "annotation",
    "node_runtime_estimation",
    "caching",
    "overlap_scheduling",
    "partitioned_scatter",
    # Only emits for torch._native DSL ops, not for a plain torch.compile.
    "native_dsl_compile",
}
for name in torch._logging._internal.log_registry.artifact_names:
    if name not in exclusions:
        setattr(LoggingTests, f"test_{name}", single_record_test(**{name: True}))

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
